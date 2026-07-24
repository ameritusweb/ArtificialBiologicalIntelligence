"""Deep Time Overnight Run: 50 generations in T7+T8 with thinking.

The richest environment stack:
  - Base Environment (pain/endorphin fields, responsive objects, NPC)
  - TieredEnvironment T4+ (multi-NPC, strategic deception, non-stationary
    rules, stochastic hidden confounders, cross-modal sources)
  - PhysicsWorld (rigid bodies, grip, compound objects)
  - T7 abstract problems (8 causal graph templates, hidden variables)
  - T8 self-modification (8 skill zones, curriculum design)
  - Thinking substrate (MCTS with 6 receptor channels)

All competing under evolutionary selection across 50 generations.
Novel receptor detection runs every 5 generations.

Launch: python deep_time_overnight.py
Expected runtime: 5-8 hours.
"""

import os
import json
import time
import numpy as np
from collections import defaultdict
from environment import Environment, Organism, NPC
from environment_tiers import TieredEnvironment
from physics_world import PhysicsWorld
from abstract_env import AbstractProblemEnvironment, SelfModificationEnvironment, CombinedT7T8Environment
from mental_model import build_mental_model, action_to_hash
from receptor_discovery import discover, calibrate_null_thresholds
from model import compute_obs_indices
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from thinking_influence import measure_thinking_influence
from novel_receptor_detector import detect_novel_receptors
from deep_time import EvolvingOrganism, select_and_reproduce

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

def _checkpoint_dir(seed=42):
    if seed == 42:
        return CHECKPOINT_DIR
    return os.path.join(RESULTS_DIR, f'checkpoints_seed{seed}')


def save_resumable_checkpoint(gen, history, organisms, model, cumulative_log,
                               cumulative_windows, cumulative_targets,
                               cumulative_next_pain, null_thresh, rng_state,
                               world_state, t_start, seed=42):
    """Save everything needed to resume from this generation."""
    import torch
    import pickle

    cp_dir = _checkpoint_dir(seed)
    os.makedirs(cp_dir, exist_ok=True)
    cp_path = os.path.join(cp_dir, f'checkpoint_gen{gen}.pt')

    organism_states = []
    for org in organisms:
        organism_states.append({
            'organism_id': org.organism_id,
            'topology_bias': dict(org.topology_bias),
            'body_params': dict(org.body_params),
            'fitness': org.fitness,
            'discovered_receptors': list(org.discovered_receptors),
        })

    torch.save({
        'generation': gen,
        'history': history,
        'organisms': organism_states,
        'model_state_dict': model.state_dict() if model is not None else None,
        'cumulative_log_len': len(cumulative_log),
        'null_thresh': null_thresh,
        'rng_state': rng_state,
        'world_state': world_state,
        'elapsed_min': round((time.time() - t_start) / 60, 1),
    }, cp_path)

    log_path = os.path.join(cp_dir, f'log_gen{gen}.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump({
            'cumulative_log': cumulative_log[-60000:],
            'cumulative_windows': cumulative_windows[-60000:],
            'cumulative_targets': cumulative_targets[-60000:],
            'cumulative_next_pain': cumulative_next_pain[-60000:],
        }, f)

    print(f"  Checkpoint saved: gen {gen} -> {cp_path}")


def load_checkpoint(gen, seed=42):
    """Load checkpoint for resuming."""
    import torch
    import pickle

    cp_dir = _checkpoint_dir(seed)
    cp_path = os.path.join(cp_dir, f'checkpoint_gen{gen}.pt')
    if not os.path.exists(cp_path):
        return None

    cp = torch.load(cp_path, weights_only=False)

    log_path = os.path.join(cp_dir, f'log_gen{gen}.pkl')
    if os.path.exists(log_path):
        with open(log_path, 'rb') as f:
            log_data = pickle.load(f)
    else:
        log_data = {
            'cumulative_log': [],
            'cumulative_windows': [],
            'cumulative_targets': [],
            'cumulative_next_pain': [],
        }

    cp['log_data'] = log_data
    return cp


def find_latest_checkpoint(seed=42):
    """Find the most recent checkpoint generation."""
    cp_dir = _checkpoint_dir(seed)
    if not os.path.exists(cp_dir):
        return -1
    gens = []
    for f in os.listdir(cp_dir):
        if f.startswith('checkpoint_gen') and f.endswith('.pt'):
            try:
                g = int(f[len('checkpoint_gen'):-3])
                gens.append(g)
            except ValueError:
                pass
    return max(gens) if gens else -1


def run_generation_rich(organisms, gen, model, engine, tree, rng,
                        world_state=None, steps_per_episode=200,
                        num_episodes=5, tier=4, use_oracle=False):
    """One generation in the richest environment stack."""
    idx = compute_obs_indices()
    obs_dim = idx['obs_dim']
    num_actions = idx['num_actions']

    env_seed = rng.randint(0, 100000)
    env = TieredEnvironment(seed=env_seed, tier=tier)

    ref_org = organisms[0].create_organism(rng)
    pw = PhysicsWorld(env, ref_org, num_objects=3, seed=env_seed)
    pw.add_compound_objects(['lever', 'spring_gate'])
    pw.enable_persistence()

    if world_state is not None:
        pw.restore_world_state(world_state)

    combined = CombinedT7T8Environment(env, seed=env_seed)

    all_windows = []
    all_targets = []
    all_next_pain = []
    all_logs = {org.organism_id: [] for org in organisms}

    for ep in range(num_episodes):
        combined.new_episode()

        for evo_org in organisms:
            org = evo_org.create_organism(rng)
            pw.org = org
            org.physics_mode = True

            active_npc = env.get_closest_npc(org.x, org.y)
            if active_npc is None:
                active_npc = NPC()
                active_npc.reset(rng)

            episode_pain = []
            episode_reward = 0.0

            prev_mm_certainty = 0.0
            prev_learning_progress = 0.0
            prev_action_hash = 0
            prev_controllability = 0.0
            prev_external_change = 0.0
            prev_planning_value = 0.0
            prev_predicted_pain = None

            for step in range(steps_per_episode):
                env.step_tier(org.x, org.y, step)
                active_npc = env.get_closest_npc(org.x, org.y) or active_npc
                active_npc.step(env, step)

                obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                if engine is not None:
                    mm_fam, mm_qual = engine.get_context_features(obs_before)
                    mm_features = (mm_fam, mm_qual, prev_mm_certainty, prev_learning_progress)
                    pattern_features = None
                    if engine.pattern_store is not None:
                        pa, pc = engine.query_pattern(prev_action_hash, obs_before)
                        pattern_features = (pa, pc)
                    cm, cq = engine.query_concept(prev_action_hash, obs_before)
                    org.concept_match = cm
                    org.concept_quality = cq
                else:
                    mm_features = None
                    pattern_features = None

                if tree is not None and engine is not None:
                    org.thinking_channels = tree.think(obs_before, engine)

                optimal = org.compute_optimal_actions(env, step, npc=active_npc)

                if use_oracle or model is None:
                    policy_action = optimal
                else:
                    window = org.get_observation_window()
                    policy_action, _ = model.predict(window)
                    all_windows.append(window.copy())
                    all_targets.append(optimal.copy())

                r = rng.random()
                if r < PROBE_RATE_FLOOR:
                    executed = np.zeros(num_actions, dtype=np.int32)
                elif r < EXPLORE_RATE:
                    executed = rng.randint(0, 2, size=num_actions).astype(np.int32)
                else:
                    executed = policy_action

                pw.apply_organism_forces(executed)
                pw.check_grips(executed)
                pw.apply_developmental_changes(step)
                pw.step()

                obs, reward = org.step(
                    executed, env, step,
                    predicted_pain=prev_predicted_pain,
                    mm_features=mm_features,
                    pattern_features=pattern_features,
                    agency_features=(prev_controllability, prev_external_change, prev_planning_value),
                    npc=active_npc,
                )
                active_npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
                episode_pain.append(obs[0:6].copy())

                t7_reward = combined.step(org, zone_choice=step % 8, difficulty=1 + gen // 10)
                episode_reward += reward + t7_reward * 0.3

                if engine is not None:
                    ctrl, ext_ch, plan_v = engine.compute_agency_features(obs_before, executed, obs)
                    prev_controllability = ctrl
                    prev_external_change = ext_ch
                    prev_planning_value = plan_v
                    pred, cert, _ = engine.predict_delta(obs_before, executed)
                    lp, cert_after = engine.compute_learning_progress(obs_before, executed, obs, reward)
                    prev_mm_certainty = cert_after
                    prev_learning_progress = lp
                    prev_predicted_pain = obs_before[:6] + pred[:6] if len(pred) >= 6 else obs_before[:6].copy()
                    if hasattr(engine, 'observe_npc'):
                        engine.observe_npc(obs)

                prev_action_hash = action_to_hash(executed)

            evo_org.fitness += episode_reward
            all_logs[evo_org.organism_id].extend(org.experience_log)

            for i in range(steps_per_episode):
                next_p = episode_pain[i + 1] if i + 1 < steps_per_episode else episode_pain[i]
                all_next_pain.append(next_p)

    for evo_org in organisms:
        evo_org.experience_log = all_logs[evo_org.organism_id]

    final_world_state = pw.save_world_state()
    return final_world_state, all_windows, all_targets, all_next_pain


def run_overnight(num_generations=50, population_size=4,
                  num_episodes=5, steps_per_episode=200,
                  bootstrap_episodes=20, epochs_per_gen=8,
                  tier=4, seed=42, resume=True):
    import torch
    t_start = time.time()

    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    start_gen = 0

    # Check for existing checkpoint
    if resume:
        latest = find_latest_checkpoint(seed)
        if latest >= 0:
            print(f"Found checkpoint at generation {latest} (seed {seed}), resuming...")
            cp = load_checkpoint(latest, seed)
            if cp is not None:
                history = cp['history']
                null_thresh = cp['null_thresh']
                world_state = cp.get('world_state')
                rng = np.random.RandomState(seed)
                rng.set_state(cp['rng_state'])

                organisms = []
                for os_data in cp['organisms']:
                    evo = EvolvingOrganism(
                        os_data['organism_id'],
                        parent_bias=os_data['topology_bias'],
                        body_params=os_data['body_params'])
                    evo.fitness = os_data['fitness']
                    evo.discovered_receptors = os_data['discovered_receptors']
                    organisms.append(evo)

                ld = cp['log_data']
                cumulative_log = ld['cumulative_log']
                cumulative_windows = ld['cumulative_windows']
                cumulative_targets = ld['cumulative_targets']
                cumulative_next_pain = ld['cumulative_next_pain']

                engine = build_mental_model(cumulative_log[-30000:])
                tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=3)

                model_sd = cp.get('model_state_dict')
                if model_sd is not None:
                    from model import HierarchicalPolicy
                    model = HierarchicalPolicy(
                        obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
                        energy_obs_index=idx['energy'],
                        conflict_obs_index=idx['conflict'],
                        staged=True).to('cpu')
                    model.load_state_dict(model_sd)
                else:
                    model = None

                start_gen = latest + 1
                print(f"  Resumed: gen {start_gen}, history={len(history)} records, "
                      f"log={len(cumulative_log)}, prior elapsed={cp.get('elapsed_min', 0):.0f}min")
                print()

    if start_gen == 0:
        print("=" * 60)
        print("DEEP TIME OVERNIGHT: T7+T8 + THINKING + NOVEL DETECTION")
        print("=" * 60)
        print(f"  Generations: {num_generations}")
        print(f"  Population: {population_size}")
        print(f"  Episodes/gen: {num_episodes}")
        print(f"  Steps/episode: {steps_per_episode}")
        print(f"  Environment tier: {tier}")
        print(f"  Bootstrap: {bootstrap_episodes} oracle episodes")
        print(f"  Seed: {seed}")
        print()

        rng = np.random.RandomState(seed)
        organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
        world_state = None
        history = []
        cumulative_windows = []
        cumulative_targets = []
        cumulative_next_pain = []
        cumulative_log = []

        # Generation 0: bootstrap
        print("--- Generation 0 (bootstrap) ---")
        world_state, _, _, _ = run_generation_rich(
            organisms, 0, model=None, engine=None, tree=None, rng=rng,
            world_state=world_state, steps_per_episode=steps_per_episode,
            num_episodes=bootstrap_episodes, tier=tier, use_oracle=True)

        for evo_org in organisms:
            cumulative_log.extend(evo_org.experience_log)

        print(f"  Log: {len(cumulative_log)} entries")
        print("  Training initial policy from oracle data...")
        X_boot, Y_boot, Z_boot, _ = generate_training_data(
            num_episodes=bootstrap_episodes, steps_per_episode=steps_per_episode, seed=seed)
        model = train_model(X_boot, Y_boot, Z_boot, epochs=epochs_per_gen,
                            staged=True, steps_per_episode=steps_per_episode)
        cumulative_windows.extend([w for w in X_boot])
        cumulative_targets.extend([t for t in Y_boot])
        cumulative_next_pain.extend([p for p in Z_boot])

        print("  Building mental model...")
        engine = build_mental_model(cumulative_log)
        tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=3)

        # Null calibration
        print("  Calibrating null thresholds...")
        null_thresh = calibrate_null_thresholds(
            cumulative_log[:30000], engine, num_shuffles=5)

        # Gen 0 discovery
        gen_discovered = set()
        for evo_org in organisms:
            if len(evo_org.experience_log) >= 100:
                org_engine = build_mental_model(evo_org.experience_log)
                results = discover(evo_org.experience_log, org_engine,
                                   threshold_overrides=null_thresh, log_provenance='oracle')
                evo_org.discovered_receptors = results['discovered']
                gen_discovered.update(results['discovered'])

        gen_record = {
            'generation': 0,
            'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
            'best_fitness': round(float(max(o.fitness for o in organisms)), 1),
            'discovered': sorted(gen_discovered),
            'num_discovered': len(gen_discovered),
            'avg_bias_size': 0,
            'elapsed_min': round((time.time() - t_start) / 60, 1),
        }

        ti = measure_thinking_influence(model, engine, num_episodes=5,
                                         steps_per_episode=100, seed=99)
        gen_record['thinking'] = {
            'divergence': ti['ablation_divergence'],
            'reward_diff': ti['reward_difference'],
            'partial_corr': ti['thinking_action_partial_corr'],
            'channels': ti['channel_influence'],
        }

        novel = detect_novel_receptors(engine, gen_discovered)
        gen_record['novel_receptors'] = len(novel)

        history.append(gen_record)
        _print_gen_summary(gen_record)

        # Checkpoint gen 0
        save_resumable_checkpoint(0, history, organisms, model, cumulative_log,
                                   cumulative_windows, cumulative_targets,
                                   cumulative_next_pain, null_thresh,
                                   rng.get_state(), world_state, t_start, seed)
        start_gen = 1

    # Generations start_gen+
    for gen in range(start_gen, num_generations):
        print(f"\n--- Generation {gen}/{num_generations-1} "
              f"({(time.time()-t_start)/60:.0f}min elapsed) ---")

        organisms = select_and_reproduce(organisms, population_size, rng)
        for i, org in enumerate(organisms):
            org.organism_id = f"gen{gen}_{i}"

        world_state, windows, targets, next_pain = run_generation_rich(
            organisms, gen, model=model, engine=engine, tree=tree, rng=rng,
            world_state=world_state, steps_per_episode=steps_per_episode,
            num_episodes=num_episodes, tier=tier, use_oracle=False)

        for evo_org in organisms:
            cumulative_log.extend(evo_org.experience_log)
        if windows:
            cumulative_windows.extend(windows)
            cumulative_targets.extend(targets)
            cumulative_next_pain.extend(next_pain)

        # Retrain (use most recent 60K samples)
        X = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Y = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Z = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        if len(X) >= 100:
            model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                                steps_per_episode=steps_per_episode)

        # Rebuild mental model
        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)
        if engine.pattern_store is not None:
            engine.pattern_store.build_from_log(
                log_slice, engine.encoder, engine.store,
                steps_per_episode=steps_per_episode)

        # Discovery
        gen_discovered = set()
        for evo_org in organisms:
            if len(evo_org.experience_log) >= 100:
                org_engine = build_mental_model(evo_org.experience_log)
                results = discover(evo_org.experience_log, org_engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='policy')
                evo_org.discovered_receptors = results['discovered']
                gen_discovered.update(results['discovered'])

        gen_record = {
            'generation': gen,
            'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
            'best_fitness': round(float(max(o.fitness for o in organisms)), 1),
            'discovered': sorted(gen_discovered),
            'num_discovered': len(gen_discovered),
            'avg_bias_size': round(float(np.mean([len(o.topology_bias) for o in organisms])), 1),
            'elapsed_min': round((time.time() - t_start) / 60, 1),
        }

        # Thinking influence (every 5 generations)
        if gen % 5 == 0 or gen == num_generations - 1:
            ti = measure_thinking_influence(model, engine, num_episodes=5,
                                             steps_per_episode=100, seed=99)
            gen_record['thinking'] = {
                'divergence': ti['ablation_divergence'],
                'reward_diff': ti['reward_difference'],
                'partial_corr': ti['thinking_action_partial_corr'],
                'channels': ti['channel_influence'],
            }

        # Novel detection (every 10 generations)
        if gen % 10 == 0 or gen == num_generations - 1:
            novel = detect_novel_receptors(engine, gen_discovered)
            gen_record['novel_receptors'] = len(novel)
            if novel:
                gen_record['novel_descriptions'] = [
                    {'novelty': n['novelty_score'], 'size': n['size'],
                     'desc': n['description']} for n in novel[:5]
                ]

        # Recalibrate null every 10 generations
        if gen % 10 == 0:
            null_thresh = calibrate_null_thresholds(
                log_slice[:30000], engine, num_shuffles=5)

        history.append(gen_record)
        _print_gen_summary(gen_record)

        # Resumable checkpoint every 10 generations
        if gen % 10 == 0 or gen == num_generations - 1:
            save_resumable_checkpoint(gen, history, organisms, model,
                                       cumulative_log, cumulative_windows,
                                       cumulative_targets, cumulative_next_pain,
                                       null_thresh, rng.get_state(),
                                       world_state, t_start, seed)
            _save_checkpoint(history, gen, t_start)

    # Final summary
    _print_final_summary(history, t_start)
    _save_checkpoint(history, num_generations - 1, t_start, final=True)

    return history


def _print_gen_summary(rec):
    print(f"  Fitness: avg={rec['avg_fitness']:.0f}  best={rec['best_fitness']:.0f}")
    print(f"  Receptors: {rec['num_discovered']}  Bias: {rec.get('avg_bias_size', 0):.0f}")
    th = rec.get('thinking', {})
    if th:
        print(f"  Thinking: div={th['divergence']:.4f}  "
              f"reward+={th['reward_diff']:+.1f}  "
              f"pcorr={th['partial_corr']:.4f}")
    novel = rec.get('novel_receptors')
    if novel is not None:
        print(f"  Novel candidates: {novel}")
    print(f"  Elapsed: {rec.get('elapsed_min', 0):.0f} min")


def _print_final_summary(history, t_start):
    total_time = (time.time() - t_start) / 60

    print("\n" + "=" * 60)
    print("OVERNIGHT RUN COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")

    print(f"\n{'Gen':>4} {'Fitness':>8} {'Recept':>7} {'Bias':>5} "
          f"{'Novel':>6} {'Elapsed':>8}")
    print("-" * 50)
    for rec in history:
        novel = rec.get('novel_receptors', '-')
        print(f"{rec['generation']:>4} {rec['avg_fitness']:>8.0f} "
              f"{rec['num_discovered']:>7} {rec.get('avg_bias_size', 0):>5.0f} "
              f"{str(novel):>6} {rec.get('elapsed_min', 0):>7.0f}m")

    all_discovered = set()
    for rec in history:
        all_discovered.update(rec['discovered'])
    gen0 = set(history[0]['discovered'])
    final = set(history[-1]['discovered'])

    print(f"\nTotal unique receptors: {len(all_discovered)}")
    print(f"Gen 0: {len(gen0)}, Final: {len(final)}")
    gained = final - gen0
    lost = gen0 - final
    if gained:
        print(f"Gained ({len(gained)}): {sorted(gained)}")
    if lost:
        print(f"Lost ({len(lost)}): {sorted(lost)}")

    # Thinking emergence
    thinking_gens = [rec for rec in history if 'thinking' in rec]
    if thinking_gens:
        print(f"\nThinking influence curve:")
        for rec in thinking_gens:
            th = rec['thinking']
            ch = th.get('channels', {})
            depth = ch.get('depth_reached', 0)
            depth_marker = " *** DEPTH ACTIVATED ***" if depth > 0.01 else ""
            print(f"  Gen {rec['generation']:>3}: pcorr={th['partial_corr']:.4f}  "
                  f"depth={depth:.4f}{depth_marker}")

    # Novel receptors
    novel_gens = [rec for rec in history if 'novel_descriptions' in rec]
    if novel_gens:
        print(f"\nNovel receptor candidates:")
        for rec in novel_gens:
            print(f"  Gen {rec['generation']}:")
            for nd in rec['novel_descriptions']:
                print(f"    novelty={nd['novelty']:.3f} size={nd['size']}: {nd['desc']}")


def _save_checkpoint(history, gen, t_start, final=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = 'deep_time_overnight_final.json' if final else f'deep_time_overnight_gen{gen}.json'
    all_discovered = set()
    for rec in history:
        all_discovered.update(rec['discovered'])
    with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
        json.dump({
            'history': history,
            'all_discovered': sorted(all_discovered),
            'total_unique': len(all_discovered),
            'elapsed_min': round((time.time() - t_start) / 60, 1),
        }, f, indent=2)
    print(f"  Checkpoint saved: {fname}")


if __name__ == '__main__':
    run_overnight(
        num_generations=50,
        population_size=4,
        num_episodes=5,
        steps_per_episode=200,
        bootstrap_episodes=20,
        epochs_per_gen=8,
        tier=4,
        seed=42,
    )

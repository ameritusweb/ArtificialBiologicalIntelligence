"""Deep Time with Thinking Substrate.

The full evolutionary loop with MCTS-driven cognition:
  1. Bootstrap a policy from oracle data (generation 0 only)
  2. Each generation: organisms compete using their POLICY + THINKING
  3. Mental model builds online during episodes
  4. Receptor discovery runs on accumulated logs
  5. Fitness-proportional selection
  6. Offspring inherit topology bias + body params
  7. Policy retrains on all accumulated data
  8. Thinking influence measured each generation

The oracle is used ONLY in generation 0 bootstrap. After that,
the organisms are on their own — policy drives behavior, MCTS
guides decisions, mental model learns from actual experience.
"""

import os
import json
import numpy as np
from collections import defaultdict
from environment import Environment, Organism, NPC
from physics_world import PhysicsWorld
from mental_model import build_mental_model, action_to_hash
from receptor_discovery import discover, build_tests, calibrate_null_thresholds
from model import compute_obs_indices
from train import train_model, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from thinking_influence import measure_thinking_influence
from deep_time import EvolvingOrganism, select_and_reproduce
from novel_receptor_detector import detect_novel_receptors, report_novel_receptors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_generation_thinking(organisms, env_seed, model, engine, tree,
                            world_state=None, steps_per_episode=200,
                            num_episodes=5, rng=None, use_oracle=False):
    """Run one generation with policy + thinking substrate."""
    if rng is None:
        rng = np.random.RandomState()

    idx = compute_obs_indices()
    obs_dim = idx['obs_dim']
    num_actions = idx['num_actions']

    env = Environment(seed=env_seed)
    ref_org = organisms[0].create_organism(rng)
    pw = PhysicsWorld(env, ref_org, num_objects=4, seed=env_seed)
    pw.add_compound_objects(['lever', 'spring_gate', 'hinged_barrier'])
    pw.enable_persistence()

    if world_state is not None:
        pw.restore_world_state(world_state)

    all_windows = []
    all_targets = []
    all_next_pain = []
    all_logs = {org.organism_id: [] for org in organisms}

    for ep in range(num_episodes):
        for evo_org in organisms:
            org = evo_org.create_organism(rng)
            pw.org = org
            org.physics_mode = True

            npc = NPC()
            npc.reset(rng)
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
                npc.step(env, step)
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
                    thinking_analysis = tree.think(obs_before, engine)
                    org.thinking_channels = thinking_analysis

                optimal = org.compute_optimal_actions(env, step, npc=npc)

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
                pw.step()
                obs, reward = org.step(
                    executed, env, step,
                    predicted_pain=prev_predicted_pain,
                    mm_features=mm_features,
                    pattern_features=pattern_features,
                    agency_features=(prev_controllability, prev_external_change, prev_planning_value),
                    npc=npc,
                )
                npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
                episode_pain.append(obs[0:6].copy())
                episode_reward += reward

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


def run_deep_time_thinking(num_generations=10, population_size=4,
                           num_episodes=5, steps_per_episode=200,
                           bootstrap_episodes=20, epochs_per_gen=8,
                           seed=0):
    """Deep time loop with thinking substrate."""
    print("=" * 60)
    print("DEEP TIME WITH THINKING SUBSTRATE")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {population_size}")
    print(f"  Episodes/gen: {num_episodes}")
    print(f"  Steps/episode: {steps_per_episode}")
    print(f"  Bootstrap: {bootstrap_episodes} oracle episodes")
    print()

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    world_state = None
    history = []
    cumulative_windows = []
    cumulative_targets = []
    cumulative_next_pain = []
    cumulative_log = []

    # Generation 0: bootstrap with oracle
    print("--- Generation 0 (bootstrap with oracle) ---")
    env_seed = rng.randint(0, 100000)
    world_state, windows, targets, next_pain = run_generation_thinking(
        organisms, env_seed, model=None, engine=None, tree=None,
        world_state=world_state, steps_per_episode=steps_per_episode,
        num_episodes=bootstrap_episodes, rng=rng, use_oracle=True)

    for evo_org in organisms:
        cumulative_log.extend(evo_org.experience_log)

    print(f"  Log: {len(cumulative_log)} entries")
    print("  Building mental model...")
    engine = build_mental_model(cumulative_log)
    print(f"  Store: {engine.store.total_count} mappings")

    # Train initial policy from oracle data
    from train import generate_training_data
    print(f"  Generating training data from oracle ({bootstrap_episodes} episodes)...")
    X_boot, Y_boot, Z_boot, boot_log = generate_training_data(
        num_episodes=bootstrap_episodes, steps_per_episode=steps_per_episode, seed=seed)
    print(f"  Training initial policy...")
    model = train_model(X_boot, Y_boot, Z_boot, epochs=epochs_per_gen,
                        staged=True, steps_per_episode=steps_per_episode)
    cumulative_windows.extend([w for w in X_boot])
    cumulative_targets.extend([t for t in Y_boot])
    cumulative_next_pain.extend([p for p in Z_boot])

    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=3)

    # Generation 0 discovery
    print("  Calibrating null thresholds...")
    ref_engine = build_mental_model(cumulative_log[:min(len(cumulative_log), 30000)])
    null_thresh = calibrate_null_thresholds(
        cumulative_log[:min(len(cumulative_log), 30000)], ref_engine, num_shuffles=5)

    gen_discovered = set()
    for evo_org in organisms:
        if len(evo_org.experience_log) >= 100:
            org_engine = build_mental_model(evo_org.experience_log)
            results = discover(evo_org.experience_log, org_engine,
                               threshold_overrides=null_thresh, log_provenance='oracle')
            evo_org.discovered_receptors = results['discovered']
            gen_discovered.update(results['discovered'])

    fitnesses = [org.fitness for org in organisms]
    gen_record = _make_record(0, organisms, gen_discovered, cumulative_log, use_oracle=True)
    history.append(gen_record)
    _print_gen(gen_record)

    # Measure thinking influence at generation 0 (baseline — no thinking exposure yet)
    print("  Measuring thinking influence (baseline)...")
    ti = measure_thinking_influence(model, engine, num_episodes=5,
                                     steps_per_episode=100, seed=99)
    gen_record['thinking'] = _extract_thinking(ti)
    _print_thinking(ti)

    # Generations 1+: self-play with thinking
    for gen in range(1, num_generations):
        print(f"\n--- Generation {gen} ---")

        organisms = select_and_reproduce(organisms, population_size, rng)
        for i, org in enumerate(organisms):
            org.organism_id = f"gen{gen}_{i}"

        env_seed = rng.randint(0, 100000)
        world_state, windows, targets, next_pain = run_generation_thinking(
            organisms, env_seed, model=model, engine=engine, tree=tree,
            world_state=world_state, steps_per_episode=steps_per_episode,
            num_episodes=num_episodes, rng=rng, use_oracle=False)

        for evo_org in organisms:
            cumulative_log.extend(evo_org.experience_log)

        if windows:
            cumulative_windows.extend(windows)
            cumulative_targets.extend(targets)
            cumulative_next_pain.extend(next_pain)

        # Retrain policy
        X = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Y = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Z = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        print(f"  Retraining on {len(X)} samples...")
        model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                            steps_per_episode=steps_per_episode)

        # Rebuild mental model
        print("  Rebuilding mental model...")
        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)
        if engine.pattern_store is not None:
            engine.pattern_store.build_from_log(
                log_slice, engine.encoder, engine.store,
                steps_per_episode=steps_per_episode)

        # Discovery
        gen_discovered = set()
        provenance = 'policy'
        for evo_org in organisms:
            if len(evo_org.experience_log) >= 100:
                org_engine = build_mental_model(evo_org.experience_log)
                results = discover(evo_org.experience_log, org_engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance=provenance)
                evo_org.discovered_receptors = results['discovered']
                gen_discovered.update(results['discovered'])

        gen_record = _make_record(gen, organisms, gen_discovered, cumulative_log)
        history.append(gen_record)
        _print_gen(gen_record)

        # Measure thinking influence
        print("  Measuring thinking influence...")
        ti = measure_thinking_influence(model, engine, num_episodes=5,
                                         steps_per_episode=100, seed=99)
        gen_record['thinking'] = _extract_thinking(ti)
        _print_thinking(ti)

        # Novel receptor detection every 5 generations
        if gen % 5 == 0 or gen == num_generations - 1:
            print("  Scanning for novel receptors...")
            novel = detect_novel_receptors(engine, set(gen_discovered))
            gen_record['novel_receptors'] = len(novel)
            if novel:
                print(f"    Found {len(novel)} novel candidates:")
                for nc in novel[:3]:
                    print(f"      novelty={nc['novelty_score']:.3f}, "
                          f"size={nc['size']}: {nc['description']}")
            else:
                print("    None detected")

        # Re-calibrate null thresholds every 3 generations
        if gen % 3 == 0:
            print("  Recalibrating null thresholds...")
            null_thresh = calibrate_null_thresholds(
                log_slice[:30000], engine, num_shuffles=5)

    # Summary
    print("\n" + "=" * 60)
    print("DEEP TIME SUMMARY")
    print("=" * 60)
    print(f"{'Gen':>4} {'Fitness':>8} {'Receptors':>10} {'Bias':>5} "
          f"{'Diverg':>8} {'Reward+':>8} {'PartCorr':>9}")
    print("-" * 65)
    for rec in history:
        th = rec.get('thinking', {})
        print(f"{rec['generation']:>4} {rec['avg_fitness']:>8.0f} "
              f"{rec['num_discovered']:>10} {rec['avg_bias_size']:>5.0f} "
              f"{th.get('divergence', 0):>8.4f} {th.get('reward_diff', 0):>+8.1f} "
              f"{th.get('partial_corr', 0):>9.4f}")

    all_discovered = set()
    for rec in history:
        all_discovered.update(rec['discovered'])
    print(f"\nTotal unique receptors across all generations: {len(all_discovered)}")

    gen0_set = set(history[0]['discovered'])
    final_set = set(history[-1]['discovered'])
    gained = final_set - gen0_set
    lost = gen0_set - final_set
    if gained:
        print(f"Gained: {sorted(gained)}")
    if lost:
        print(f"Lost: {sorted(lost)}")

    # Thinking emergence across generations
    print(f"\nThinking channel emergence:")
    print(f"{'Gen':>4} {'BestVal':>8} {'Entropy':>8} {'Converg':>8} "
          f"{'PathDiv':>8} {'Unexpl':>8} {'Depth':>8}")
    print("-" * 58)
    for rec in history:
        ch = rec.get('thinking', {}).get('channels', {})
        print(f"{rec['generation']:>4} "
              f"{ch.get('best_value', 0):>8.4f} {ch.get('visit_entropy', 0):>8.4f} "
              f"{ch.get('value_convergence', 0):>8.4f} {ch.get('path_divergence', 0):>8.4f} "
              f"{ch.get('underexplored', 0):>8.4f} {ch.get('depth_reached', 0):>8.4f}")

    # Novel receptor summary
    total_novel = sum(rec.get('novel_receptors', 0) for rec in history)
    print(f"\nNovel receptor candidates detected: {total_novel}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'deep_time_thinking.json'), 'w') as f:
        json.dump({
            'history': history,
            'all_discovered': sorted(all_discovered),
            'gained': sorted(gained),
            'lost': sorted(lost),
            'total_novel_candidates': total_novel,
        }, f, indent=2)
    print(f"\nSaved to data/deep_time_thinking.json")

    return history


def _make_record(gen, organisms, discovered, cumulative_log, use_oracle=False):
    fitnesses = [org.fitness for org in organisms]
    bias_sizes = [len(org.topology_bias) for org in organisms]
    return {
        'generation': gen,
        'avg_fitness': round(float(np.mean(fitnesses)), 1),
        'best_fitness': round(float(max(fitnesses)), 1),
        'discovered': sorted(discovered),
        'num_discovered': len(discovered),
        'population_size': len(organisms),
        'total_log_entries': len(cumulative_log),
        'avg_bias_size': round(float(np.mean(bias_sizes)), 1),
        'used_oracle': use_oracle,
    }


def _extract_thinking(ti):
    return {
        'divergence': ti['ablation_divergence'],
        'reward_diff': ti['reward_difference'],
        'partial_corr': ti['thinking_action_partial_corr'],
        'channels': ti['channel_influence'],
    }


def _print_gen(rec):
    print(f"  Fitness: avg={rec['avg_fitness']:.0f}  best={rec['best_fitness']:.0f}")
    print(f"  Discovered: {rec['num_discovered']} receptors")
    print(f"  Bias: avg={rec['avg_bias_size']:.0f} inherited")


def _print_thinking(ti):
    print(f"  Thinking: diverg={ti['ablation_divergence']:.4f}, "
          f"reward+={ti['reward_difference']:+.1f}, "
          f"pcorr={ti['thinking_action_partial_corr']:.4f}")
    ch = ti['channel_influence']
    top = sorted(ch.items(), key=lambda x: -x[1])[:3]
    if any(v > 0.01 for _, v in top):
        print(f"    Top: {', '.join(f'{k}={v:.3f}' for k, v in top)}")


if __name__ == '__main__':
    run_deep_time_thinking(
        num_generations=10,
        population_size=4,
        num_episodes=5,
        steps_per_episode=200,
        bootstrap_episodes=20,
        epochs_per_gen=8,
        seed=42,
    )

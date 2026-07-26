"""80-generation deep time with full 340-dim Umwelt.

Single condition — we already have the 177-dim baseline from
deep_time_overnight. This tests whether the cognitive break
persists across evolutionary time and whether new receptors
emerge that couldn't emerge at 177 dims.
"""

import os
import json
import numpy as np
import torch
from collections import deque
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash
from model import compute_obs_indices
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from cognitive_state import CognitiveStateDetector
from deep_time import EvolvingOrganism, select_and_reproduce
from receptor_coactivation import CoactivationLogger
from receptor_discovery import discover, calibrate_null_thresholds
from procedural_memory import (PeakExperienceIndex, ReplayEngine,
                               MotorSequenceStore, ShortcutExecutor)
from live_receptors import LiveReceptorBank
from episode_receptors import EpisodeLevelReceptorBank
from receptor_activation import ReceptorActivationManager

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(num_generations=80, population_size=4, num_episodes=5,
        steps_per_episode=1000, seed=42, checkpoint_every=10):
    print("80-GEN FULL UMWELT (340-dim)")
    print(f"{num_generations} gens, {steps_per_episode} steps/ep, seed={seed}")

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    core_obs_dim = idx['core_obs_dim']
    obs_dim = idx['obs_dim']
    num_thinking_channels = idx.get('num_thinking_channels', 6)
    shortcut_threshold = 0.5

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    cumulative_log = []
    history = []

    max_buffer = 60000
    cumulative_windows = deque(maxlen=max_buffer)
    cumulative_targets = deque(maxlen=max_buffer)
    cumulative_next_pain = deque(maxlen=max_buffer)

    print("  Bootstrapping...")
    X, Y, Z, boot_log = generate_training_data(
        num_episodes=15, steps_per_episode=steps_per_episode, seed=seed)
    model = train_model(X, Y, Z, epochs=8, staged=True,
                        steps_per_episode=steps_per_episode)
    cumulative_log.extend(boot_log)
    for i in range(len(X)):
        cumulative_windows.append(X[i].astype(np.float32))
        cumulative_targets.append(Y[i].astype(np.float32))
        cumulative_next_pain.append(Z[i].astype(np.float32))

    engine = build_mental_model(cumulative_log[-max_buffer:])
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=4)
    cog_detector = CognitiveStateDetector()
    peak_index = PeakExperienceIndex(max_size=200)
    replay_engine = ReplayEngine(peak_index)
    motor_store = MotorSequenceStore(num_continuous=0)
    shortcut_executor = ShortcutExecutor()
    receptor_bank = LiveReceptorBank()
    episode_bank = EpisodeLevelReceptorBank()
    activation_mgr = ReceptorActivationManager()

    null_thresh = None

    for gen in range(num_generations):
        print(f"\n--- Generation {gen} ---")
        gen_logger = CoactivationLogger(idx)
        total_shortcuts = 0

        model.to('cpu')
        model.eval()

        for evo_org in organisms:
            org = evo_org.create_organism(rng)
            budget = int(evo_org.body_params.get('thinking_budget', 24))
            tree.max_simulations = budget
            v_keys = {k: v for k, v in evo_org.body_params.items()
                      if k.startswith('v_')}
            tree.v_weights = v_keys if v_keys else None
            thinking_cost = float(evo_org.body_params.get('thinking_cost', 0.001))

            npc = NPC()
            npc.reset(rng)
            env = Environment(seed=rng.randint(0, 100000))
            prev_action_hash = 0
            receptor_bank.reset()

            for ep in range(num_episodes):
                for step in range(steps_per_episode):
                    npc.step(env, step)
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hash)
                    org.thought_type_id = tt
                    org.concept_id = ca

                    shortcut_fired = False
                    if shortcut_executor.is_active():
                        if shortcut_executor.should_abort(obs_before):
                            shortcut_executor.finish(motor_store, gen)
                        elif shortcut_executor.is_complete():
                            shortcut_executor.finish(motor_store, gen)
                        else:
                            executed = shortcut_executor.get_action()
                            if executed is not None:
                                org.thinking_channels = np.zeros(num_thinking_channels)
                                obs, reward = org.step(executed, env, step, npc=npc)
                                rc = receptor_bank.compute(obs, executed, engine, reward)
                                activation_mgr.update_live(rc)
                                rc, _ = activation_mgr.apply_mask(rc, org.episode_receptor_channels)
                                org.receptor_channels = rc
                                shortcut_executor.add_reward(reward)
                                evo_org.fitness += reward
                                gen_logger.record(obs)
                                prev_action_hash = action_to_hash(executed)
                                total_shortcuts += 1
                                shortcut_fired = True

                    if not shortcut_fired and not shortcut_executor.is_active():
                        embedding = engine.encoder.embed(obs_before[:core_obs_dim])
                        match, score = motor_store.query(tt, embedding, min_support=3)
                        if match is not None and score > shortcut_threshold:
                            shortcut_executor.start(match, obs_before)
                            executed = shortcut_executor.get_action()
                            if executed is not None:
                                org.thinking_channels = np.zeros(num_thinking_channels)
                                obs, reward = org.step(executed, env, step, npc=npc)
                                rc = receptor_bank.compute(obs, executed, engine, reward)
                                activation_mgr.update_live(rc)
                                rc, _ = activation_mgr.apply_mask(rc, org.episode_receptor_channels)
                                org.receptor_channels = rc
                                shortcut_executor.add_reward(reward)
                                evo_org.fitness += reward
                                gen_logger.record(obs)
                                prev_action_hash = action_to_hash(executed)
                                total_shortcuts += 1
                                shortcut_fired = True

                    if shortcut_fired:
                        continue

                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)
                    org.energy = max(0.0, org.energy - activation_mgr.get_metabolic_cost())

                    if gen == 0:
                        actions = org.compute_optimal_actions(env, step, npc=npc)
                        executed = actions
                    else:
                        window = org.get_observation_window()
                        policy_action, _ = model.predict(window)
                        optimal = org.compute_optimal_actions(env, step, npc=npc)
                        cumulative_windows.append(window.copy().astype(np.float32))
                        cumulative_targets.append(optimal.copy().astype(np.float32))
                        executed = policy_action

                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(num_actions, dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=num_actions).astype(np.int32)

                    obs, reward = org.step(executed, env, step, npc=npc)
                    evo_org.fitness += reward
                    gen_logger.record(obs)
                    prev_action_hash = action_to_hash(executed)

                    rc = receptor_bank.compute(obs, executed, engine, reward)
                    activation_mgr.update_live(rc)
                    rc, _ = activation_mgr.apply_mask(rc, org.episode_receptor_channels)
                    org.receptor_channels = rc

                    reward_delta = reward - (float(np.mean(obs_before[:6]))
                                             if len(obs_before) >= 6 else 0)
                    if reward_delta > 0.1:
                        peak_index.add(reward_delta, len(org.experience_log) - 1,
                                       executed, obs_before)
                    if replay_engine.should_replay(obs, step):
                        replay_engine.replay(engine, step)

                if len(org.experience_log) >= 20:
                    motor_store.extract_sequences(
                        org.experience_log, peak_index,
                        engine.encoder, cog_detector, core_obs_dim)

                erc = episode_bank.compute(org.experience_log, engine)
                activation_mgr.update_episode(erc)
                _, erc = activation_mgr.apply_mask(org.receptor_channels, erc)
                org.episode_receptor_channels = erc

                ep_pain = [e['obs_after'][0:6].copy()
                           for e in org.experience_log[-steps_per_episode:]]
                for i in range(len(ep_pain)):
                    next_p = ep_pain[i + 1] if i + 1 < len(ep_pain) else ep_pain[-1]
                    cumulative_next_pain.append(next_p.astype(np.float32))

            cumulative_log.extend(org.experience_log)

        model.to(DEVICE)
        if gen > 0 and len(cumulative_windows) >= 100:
            X = np.array(list(cumulative_windows), dtype=np.float32)
            Y = np.array(list(cumulative_targets), dtype=np.float32)
            Z = np.array(list(cumulative_next_pain), dtype=np.float32)
            model = train_model(X, Y, Z, epochs=3, staged=True,
                                steps_per_episode=steps_per_episode)

        log_slice = cumulative_log[-max_buffer:]
        engine = build_mental_model(log_slice)
        motor_store.re_embed_all(engine.encoder, core_obs_dim)
        motor_store.evict_stale(gen, max_staleness=10)

        # Co-activation
        stats = gen_logger.get_stats()
        pain_conflict_lift = 0
        pc_forward = 0
        cp_forward = 0
        anxiety_loop = False
        if stats:
            for p in stats.get('top_coactivations', []):
                if (p['a'] == 'pain' and p['b'] == 'conflict') or \
                   (p['a'] == 'conflict' and p['b'] == 'pain'):
                    pain_conflict_lift = p['lift']
            for s in stats.get('top_sequences', []):
                if s['from'] == 'pain' and s['to'] == 'conflict':
                    pc_forward = s['lift']
                if s['from'] == 'conflict' and s['to'] == 'pain':
                    cp_forward = s['lift']
            anxiety_loop = pc_forward > 1.5 and cp_forward > 1.5

        # Receptor discovery every 10 gens
        discovered = []
        if gen % 10 == 9 or gen == num_generations - 1:
            if null_thresh is None and len(log_slice) >= 200:
                null_thresh = calibrate_null_thresholds(log_slice, engine, num_shuffles=5)
            if null_thresh is not None:
                results = discover(log_slice, engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='oracle')
                discovered = results['discovered']

        ms = motor_store.get_stats()
        act = activation_mgr.get_stats()
        avg_fitness = np.mean([o.fitness for o in organisms])
        coverage = total_shortcuts / max(stats['total_steps'], 1) * 100 if stats else 0

        rec = {
            'generation': gen,
            'avg_fitness': round(float(avg_fitness), 2),
            'num_discovered': len(discovered),
            'discovered': discovered,
            'anxiety_loop': anxiety_loop,
            'pain_to_conflict_lift': round(pc_forward, 3),
            'conflict_to_pain_lift': round(cp_forward, 3),
            'shortcuts_fired': total_shortcuts,
            'shortcut_coverage': round(coverage, 1),
            'motor_entries': ms.get('total_entries', 0),
            'active_receptors': act.get('active_total', 0),
            'active_live': act.get('active_live', 0),
            'active_episode': act.get('active_episode', 0),
        }
        history.append(rec)

        print(f"  Fitness: {avg_fitness:.1f}  Loop: {'YES' if anxiety_loop else 'no'}  "
              f"P->C: {pc_forward:.2f}  Shortcuts: {total_shortcuts} ({coverage:.0f}%)  "
              f"Active: {act.get('active_total', 0)}")
        if discovered:
            print(f"  Discovered: {len(discovered)} receptors")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"gen{gen+1}_{i}"

        # Checkpoint
        if (gen + 1) % checkpoint_every == 0:
            os.makedirs(DATA_DIR, exist_ok=True)
            ckpt = {'history': history, 'generation': gen}
            path = os.path.join(DATA_DIR, f'full_umwelt_80gen_ckpt{gen+1}.json')
            with open(path, 'w') as f:
                json.dump(ckpt, f, indent=2)
            print(f"  Checkpoint saved: {path}")

    # Final save
    os.makedirs(DATA_DIR, exist_ok=True)
    final = {
        'history': history,
        'total_generations': num_generations,
        'obs_dim': obs_dim,
        'seed': seed,
    }
    with open(os.path.join(DATA_DIR, 'full_umwelt_80gen.json'), 'w') as f:
        json.dump(final, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("80-GEN FULL UMWELT SUMMARY")
    print("=" * 60)
    loop_count = sum(1 for r in history if r['anxiety_loop'])
    no_loop = sum(1 for r in history if not r['anxiety_loop'])
    print(f"Anxiety loop: {loop_count}/80 YES, {no_loop}/80 no")

    disc_gens = [r for r in history if r['num_discovered'] > 0]
    if disc_gens:
        print(f"\nReceptor discovery checkpoints:")
        for r in disc_gens:
            print(f"  Gen {r['generation']}: {r['num_discovered']} receptors, "
                  f"active={r['active_receptors']}")

    print(f"\nSaved to data/full_umwelt_80gen.json")
    return history


if __name__ == '__main__':
    run(seed=42)

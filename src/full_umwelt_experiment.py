"""Full Umwelt Experiment: 340-dim obs vector with all 200 receptors live.

Tests T102-T103: does wiring the full receptor topology into the observation
vector break the anxiety loop cognitively (vs mechanistically via shortcuts)?

Two conditions, paired seed:
  - control: 177-dim obs (no live receptors, no episode receptors)
  - full_umwelt: 340-dim obs (73 live + 90 episode receptors feeding back)

Both conditions run with motor store + shortcuts active, so the mechanistic
path is available in both. The question is whether the full Umwelt enables
the organism to break the loop through sensing the pattern, not just
bypassing MCTS.

Key metrics:
  - Anxiety loop frequency (P->C and C->P lift > 1.5)
  - Shortcut coverage (% of steps using shortcuts)
  - Active receptor count (how many of the 163 channels activate)
  - Separation gap (how many fitness-positive receptors are active)
"""

import os
import sys
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
from procedural_memory import (PeakExperienceIndex, ReplayEngine,
                               MotorSequenceStore, ShortcutExecutor)
from live_receptors import LiveReceptorBank
from episode_receptors import EpisodeLevelReceptorBank
from receptor_activation import ReceptorActivationManager

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_condition(condition_name, use_full_umwelt, num_generations=20,
                  population_size=4, num_episodes=5, steps_per_episode=1000,
                  seed=42):
    print(f"\n{'=' * 60}")
    print(f"CONDITION: {condition_name} (full_umwelt={use_full_umwelt})")
    print(f"  {num_generations} gens, {steps_per_episode} steps/ep, seed={seed}")
    print(f"{'=' * 60}")

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

    receptor_bank = LiveReceptorBank() if use_full_umwelt else None
    episode_bank = EpisodeLevelReceptorBank() if use_full_umwelt else None
    activation_mgr = ReceptorActivationManager() if use_full_umwelt else None

    for gen in range(num_generations):
        print(f"\n--- [{condition_name}] Generation {gen} ---")
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

            if receptor_bank is not None:
                receptor_bank.reset()

            for ep in range(num_episodes):
                for step in range(steps_per_episode):
                    npc.step(env, step)
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hash)
                    org.thought_type_id = tt
                    org.concept_id = ca

                    shortcut_fired = False

                    # Motor store shortcut check
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
                                if receptor_bank is not None:
                                    rc = receptor_bank.compute(obs, executed, engine, reward)
                                    if activation_mgr is not None:
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
                                if receptor_bank is not None:
                                    rc = receptor_bank.compute(obs, executed, engine, reward)
                                    if activation_mgr is not None:
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

                    # MCTS thinking
                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)

                    # Receptor metabolic cost
                    if activation_mgr is not None:
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

                    # Live receptors
                    if receptor_bank is not None:
                        rc = receptor_bank.compute(obs, executed, engine, reward)
                        if activation_mgr is not None:
                            activation_mgr.update_live(rc)
                            rc, _ = activation_mgr.apply_mask(rc, org.episode_receptor_channels)
                        org.receptor_channels = rc

                    # Peak indexing
                    reward_delta = reward - (float(np.mean(obs_before[:6]))
                                             if len(obs_before) >= 6 else 0)
                    if reward_delta > 0.1:
                        peak_index.add(reward_delta, len(org.experience_log) - 1,
                                       executed, obs_before)

                    if replay_engine.should_replay(obs, step):
                        replay_engine.replay(engine, step)

                # End of episode
                if len(org.experience_log) >= 20:
                    motor_store.extract_sequences(
                        org.experience_log, peak_index,
                        engine.encoder, cog_detector, core_obs_dim)

                # Episode-level receptors
                if episode_bank is not None:
                    erc = episode_bank.compute(org.experience_log, engine)
                    if activation_mgr is not None:
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
        motor_store.evict_stale(gen, max_staleness=5)

        # Co-activation stats
        stats = gen_logger.get_stats()
        if stats is None:
            continue

        pain_conflict_lift = 0
        pc_forward = 0
        cp_forward = 0
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

        ms_stats = motor_store.get_stats()
        avg_fitness = np.mean([o.fitness for o in organisms])
        act_stats = activation_mgr.get_stats() if activation_mgr else {}

        rec = {
            'generation': gen,
            'condition': condition_name,
            'total_steps': stats['total_steps'],
            'unique_patterns': stats['n_unique_patterns'],
            'pain_conflict_lift': round(pain_conflict_lift, 3),
            'anxiety_loop': anxiety_loop,
            'pain_to_conflict_lift': round(pc_forward, 3),
            'conflict_to_pain_lift': round(cp_forward, 3),
            'shortcuts_fired': total_shortcuts,
            'motor_entries': ms_stats.get('total_entries', 0),
            'motor_success_rate': round(ms_stats.get('success_rate', 0), 3),
            'avg_fitness': round(float(avg_fitness), 2),
            'active_receptors': act_stats.get('active_total', 0),
            'active_live': act_stats.get('active_live', 0),
            'active_episode': act_stats.get('active_episode', 0),
            'receptor_metabolic_cost': act_stats.get('metabolic_cost', 0),
        }
        history.append(rec)

        coverage = total_shortcuts / max(stats['total_steps'], 1) * 100
        act_str = f"  Receptors: {act_stats.get('active_total', 0)} active" if act_stats else ""
        ms_str = f"  Motor: {ms_stats['total_entries']} seqs, {total_shortcuts} shortcuts ({coverage:.0f}%)"

        print(f"  Steps: {stats['total_steps']}  Patterns: {stats['n_unique_patterns']}  "
              f"Fitness: {avg_fitness:.1f}")
        print(f"  Pain<->Conflict: coact={pain_conflict_lift:.2f}  "
              f"P->C={pc_forward:.2f}  C->P={cp_forward:.2f}  "
              f"loop={'YES' if anxiety_loop else 'no'}")
        print(ms_str)
        if act_str:
            print(act_str)

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"gen{gen+1}_{i}"

    return history


def run_experiment(seed=42):
    print("FULL UMWELT EXPERIMENT")
    print("Does the 340-dim obs vector break the anxiety loop cognitively?")
    print(f"Paired seed={seed}, 1000 steps/episode, 20 generations")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    control = run_condition("control_177", use_full_umwelt=False, seed=seed)
    treatment = run_condition("full_umwelt_340", use_full_umwelt=True, seed=seed)

    print("\n" + "=" * 80)
    print("COMPARISON: Control (177-dim) vs Full Umwelt (340-dim)")
    print("=" * 80)
    print(f"\n{'Gen':>4} {'Ctrl Loop':>10} {'Ctrl P->C':>10} {'Ctrl Cuts':>10}"
          f" | {'FU Loop':>8} {'FU P->C':>8} {'FU Cuts':>8} {'Active':>7}")
    print("-" * 85)

    ctrl_loops = 0
    fu_loops = 0
    for c, t in zip(control, treatment):
        g = c['generation']
        c_loop = 'YES' if c['anxiety_loop'] else 'no'
        t_loop = 'YES' if t['anxiety_loop'] else 'no'
        if c['anxiety_loop']:
            ctrl_loops += 1
        if t['anxiety_loop']:
            fu_loops += 1
        print(f"{g:>4} {c_loop:>10} {c['pain_to_conflict_lift']:>10.2f} "
              f"{c['shortcuts_fired']:>10}"
              f" | {t_loop:>8} {t['pain_to_conflict_lift']:>8.2f} "
              f"{t['shortcuts_fired']:>8} {t.get('active_receptors', 0):>7}")

    n = max(len(control), len(treatment))
    print(f"\nAnxiety loop: control={ctrl_loops}/{n}, full_umwelt={fu_loops}/{n}")
    if fu_loops < ctrl_loops:
        diff = ctrl_loops - fu_loops
        print(f"  Full Umwelt REDUCED anxiety loop by {diff} generations")
        if any(not t['anxiety_loop'] and t['shortcuts_fired'] < t['total_steps'] * 0.5
               for t in treatment):
            print("  COGNITIVE BREAK DETECTED: loop broke without majority shortcut coverage")
    elif fu_loops == ctrl_loops:
        print("  No difference — topology refinement alone insufficient")
    else:
        print("  Full Umwelt INCREASED anxiety loop (unexpected)")

    results = {'seed': seed, 'control': control, 'full_umwelt': treatment}
    with open(os.path.join(DATA_DIR, 'full_umwelt_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/full_umwelt_experiment.json")

    return results


if __name__ == '__main__':
    run_experiment(seed=42)

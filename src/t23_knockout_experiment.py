"""T23 Knockout: Self-damping requires a four-family chain.

Tests whether the anxiety loop returns when each of the four
hypothesized self-damping receptors is knocked out individually
from the 340-dim organism:
  1. metacognition (receptor index 8)
  2. processing_speed (receptor index 24)
  3. self_soothing (receptor index 50)
  4. stress_detection (receptor index 9)

If the loop returns when ANY single one is removed, T23 is supported:
the four-family chain is necessary, not just sufficient.
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
from procedural_memory import (PeakExperienceIndex, ReplayEngine,
                               MotorSequenceStore, ShortcutExecutor)
from live_receptors import LiveReceptorBank, NUM_LIVE_RECEPTORS
from episode_receptors import EpisodeLevelReceptorBank
from receptor_activation import ReceptorActivationManager

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KNOCKOUT_TARGETS = {
    'metacognition': 8,
    'processing_speed': 24,
    'self_soothing': 50,
    'stress_detection': 9,
    'control': None,  # no knockout — full Umwelt baseline
}


def run_knockout(knockout_name, knockout_idx, num_generations=20,
                 population_size=4, num_episodes=5, steps_per_episode=1000,
                 seed=42):
    print(f"\n{'=' * 60}")
    print(f"KNOCKOUT: {knockout_name} (idx={knockout_idx})")
    print(f"{'=' * 60}")

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    core_obs_dim = idx['core_obs_dim']
    obs_dim = idx['obs_dim']
    num_thinking_channels = idx.get('num_thinking_channels', 6)

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

    # Build knockout mask
    knockout_mask = np.ones(NUM_LIVE_RECEPTORS, dtype=bool)
    if knockout_idx is not None:
        knockout_mask[knockout_idx] = False

    for gen in range(num_generations):
        print(f"\n--- [{knockout_name}] Gen {gen} ---")
        gen_logger = CoactivationLogger(idx)

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
            receptor_bank = LiveReceptorBank()
            episode_bank = EpisodeLevelReceptorBank()

            for ep in range(num_episodes):
                receptor_bank.reset()
                for step in range(steps_per_episode):
                    npc.step(env, step)
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hash)
                    org.thought_type_id = tt
                    org.concept_id = ca

                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)

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

                    # Live receptors with knockout mask
                    rc = receptor_bank.compute(obs, executed, engine, reward,
                                               active_mask=knockout_mask)
                    org.receptor_channels = rc

                # Episode receptors
                erc = episode_bank.compute(org.experience_log, engine)
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

        stats = gen_logger.get_stats()
        pc_forward = 0
        cp_forward = 0
        if stats:
            for s in stats.get('top_sequences', []):
                if s['from'] == 'pain' and s['to'] == 'conflict':
                    pc_forward = s['lift']
                if s['from'] == 'conflict' and s['to'] == 'pain':
                    cp_forward = s['lift']
        anxiety_loop = pc_forward > 1.5 and cp_forward > 1.5

        avg_fitness = np.mean([o.fitness for o in organisms])
        rec = {
            'generation': gen,
            'knockout': knockout_name,
            'anxiety_loop': anxiety_loop,
            'pain_to_conflict_lift': round(pc_forward, 3),
            'conflict_to_pain_lift': round(cp_forward, 3),
            'avg_fitness': round(float(avg_fitness), 2),
        }
        history.append(rec)

        print(f"  Fitness: {avg_fitness:.1f}  "
              f"Loop: {'YES' if anxiety_loop else 'no'}  "
              f"P->C: {pc_forward:.2f}")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"gen{gen+1}_{i}"

    return history


def run_experiment(seed=42):
    print("T23 KNOCKOUT EXPERIMENT")
    print("Does removing any single self-damping receptor restore the anxiety loop?")
    print()

    all_results = {}
    for name, ko_idx in KNOCKOUT_TARGETS.items():
        history = run_knockout(name, ko_idx, seed=seed)
        all_results[name] = history

    print("\n" + "=" * 70)
    print("T23 KNOCKOUT RESULTS")
    print("=" * 70)
    print(f"\n{'Knockout':<20} {'Loop gens':>10} {'Total':>6} {'Loop %':>7}")
    print("-" * 45)

    for name in KNOCKOUT_TARGETS:
        hist = all_results[name]
        loop_count = sum(1 for r in hist if r['anxiety_loop'])
        total = len(hist)
        pct = loop_count / total * 100
        print(f"{name:<20} {loop_count:>10} {total:>6} {pct:>6.0f}%")

    control_loops = sum(1 for r in all_results['control'] if r['anxiety_loop'])
    knockouts_restore = []
    for name in ['metacognition', 'processing_speed', 'self_soothing', 'stress_detection']:
        ko_loops = sum(1 for r in all_results[name] if r['anxiety_loop'])
        if ko_loops > control_loops + 2:
            knockouts_restore.append(name)

    print()
    if knockouts_restore:
        print(f"T23 SUPPORTED: removing {knockouts_restore} restored the anxiety loop")
    else:
        print("T23 NOT SUPPORTED: no single knockout restored the loop")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 't23_knockout.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to data/t23_knockout.json")

    return all_results


if __name__ == '__main__':
    run_experiment(seed=42)

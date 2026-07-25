"""Deep time with cognitive state channels active.

Runs the full pipeline with thought_type_id and concept_id feeding
back into the observation vector. Tracks co-activation patterns
to see if the organism learns to use its cognitive mode recognition
to exit unproductive states (like the gen 8 suffering pattern).
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash
from model import compute_obs_indices
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from thinking_influence import measure_thinking_influence
from deep_time import EvolvingOrganism, select_and_reproduce
from cognitive_state import CognitiveStateDetector
from receptor_coactivation import CoactivationLogger
from receptor_discovery import discover, calibrate_null_thresholds

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_cognitive_deep_time(num_generations=10, population_size=4,
                             num_episodes=5, steps_per_episode=200,
                             seed=42):
    print("=" * 60)
    print("DEEP TIME WITH COGNITIVE STATE CHANNELS")
    print("=" * 60)
    print(f"  OBS_DIM includes thought_type_id + concept_id")

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    obs_dim = idx['obs_dim']
    print(f"  OBS_DIM: {obs_dim}")

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    cumulative_log = []
    history = []

    # Bootstrap
    print("\n  Bootstrapping...")
    X, Y, Z, boot_log = generate_training_data(
        num_episodes=20, steps_per_episode=steps_per_episode, seed=seed)
    model = train_model(X, Y, Z, epochs=8, staged=True,
                        steps_per_episode=steps_per_episode)
    cumulative_log.extend(boot_log)
    cumulative_windows = list(X)
    cumulative_targets = list(Y)
    cumulative_next_pain = list(Z)

    engine = build_mental_model(cumulative_log)
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=4)
    cog_detector = CognitiveStateDetector()

    null_thresh = calibrate_null_thresholds(cumulative_log[:30000], engine, num_shuffles=3)

    for gen in range(num_generations):
        print(f"\n--- Generation {gen} ---")
        gen_logger = CoactivationLogger(idx)

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

            for ep in range(num_episodes):
                for step in range(steps_per_episode):
                    npc.step(env, step)
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)

                    # Cognitive state detection — feeds back into obs
                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hash)
                    org.thought_type_id = tt
                    org.concept_id = ca

                    if gen == 0:
                        actions = org.compute_optimal_actions(env, step, npc=npc)
                        executed = actions
                    else:
                        window = org.get_observation_window()
                        policy_action, _ = model.predict(window)
                        optimal = org.compute_optimal_actions(env, step, npc=npc)
                        cumulative_windows.append(window.copy())
                        cumulative_targets.append(optimal.copy())
                        executed = policy_action

                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(num_actions, dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=num_actions).astype(np.int32)

                    obs, reward = org.step(executed, env, step, npc=npc)
                    evo_org.fitness += reward
                    prev_action_hash = action_to_hash(executed)

                    gen_logger.record(obs)

                ep_pain = [e['obs_after'][0:6].copy()
                           for e in org.experience_log[-steps_per_episode:]]
                for i in range(len(ep_pain)):
                    next_p = ep_pain[i + 1] if i + 1 < len(ep_pain) else ep_pain[-1]
                    cumulative_next_pain.append(next_p)

            cumulative_log.extend(org.experience_log)

        # Retrain
        X = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Y = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Z = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        if len(X) >= 100 and gen > 0:
            model = train_model(X, Y, Z, epochs=8, staged=True,
                                steps_per_episode=steps_per_episode)

        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)

        # Co-activation stats
        coact_stats = gen_logger.get_stats()

        # Receptor discovery
        gen_discovered = set()
        if len(log_slice) >= 500:
            results = discover(log_slice, engine,
                               threshold_overrides=null_thresh,
                               log_provenance='oracle')
            gen_discovered = set(results['discovered'])

        # Extract metrics
        pain_conflict_lift = 0
        pc_forward = 0
        cp_forward = 0
        anxiety_loop = False
        if coact_stats:
            for p in coact_stats.get('top_coactivations', []):
                if set([p['a'], p['b']]) == set(['pain', 'conflict']):
                    pain_conflict_lift = p['lift']
            for s in coact_stats.get('top_sequences', []):
                if s['from'] == 'pain' and s['to'] == 'conflict':
                    pc_forward = s['lift']
                if s['from'] == 'conflict' and s['to'] == 'pain':
                    cp_forward = s['lift']
            anxiety_loop = pc_forward > 1.5 and cp_forward > 1.5

        pain_rate = coact_stats['activation_rates'].get('pain', 0) if coact_stats else 0
        conflict_rate = coact_stats['activation_rates'].get('conflict', 0) if coact_stats else 0

        rec = {
            'generation': gen,
            'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
            'num_discovered': len(gen_discovered),
            'unique_patterns': coact_stats['n_unique_patterns'] if coact_stats else 0,
            'pain_rate': round(pain_rate, 3),
            'conflict_rate': round(conflict_rate, 3),
            'pain_conflict_lift': round(pain_conflict_lift, 3),
            'anxiety_loop': anxiety_loop,
            'codebook_size': cog_detector.get_stats()['codebook_size'],
        }
        history.append(rec)

        print(f"  Fitness: {rec['avg_fitness']:.0f}  Receptors: {rec['num_discovered']}  "
              f"Patterns: {rec['unique_patterns']}")
        print(f"  Pain: {rec['pain_rate']:.3f}  Conflict: {rec['conflict_rate']:.3f}  "
              f"P<->C: {rec['pain_conflict_lift']:.2f}  Loop: {'YES' if anxiety_loop else 'no'}")
        print(f"  Thought codebook: {rec['codebook_size']} types")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"gen{gen+1}_{i}"

    # Summary
    print("\n" + "=" * 60)
    print("COGNITIVE STATE DEEP TIME SUMMARY")
    print("=" * 60)
    print(f"\n{'Gen':>4} {'Fitness':>8} {'Recept':>7} {'Patterns':>9} "
          f"{'Pain':>6} {'Confl':>6} {'P<->C':>6} {'Loop':>5} {'Codebook':>9}")
    print("-" * 72)
    for rec in history:
        print(f"{rec['generation']:>4} {rec['avg_fitness']:>8.0f} "
              f"{rec['num_discovered']:>7} {rec['unique_patterns']:>9} "
              f"{rec['pain_rate']:>6.3f} {rec['conflict_rate']:>6.3f} "
              f"{rec['pain_conflict_lift']:>6.2f} "
              f"{'YES' if rec['anxiety_loop'] else 'no':>5} "
              f"{rec['codebook_size']:>9}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'cognitive_state_deep_time.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to data/cognitive_state_deep_time.json")

    return history


if __name__ == '__main__':
    run_cognitive_deep_time(num_generations=10, seed=42)

"""Co-activation analysis across deep time generations.

Tracks how thought patterns evolve: do co-activations change as the
topology develops? Do new sequences emerge? Does the anxiety loop
(pain<->conflict) persist or resolve?
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash
from model import compute_obs_indices
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from deep_time import EvolvingOrganism, select_and_reproduce
from receptor_coactivation import CoactivationLogger

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_coactivation_deep_time(num_generations=10, population_size=4,
                                num_episodes=5, steps_per_episode=200,
                                seed=42):
    print("=" * 60)
    print("CO-ACTIVATION ACROSS DEEP TIME")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    cumulative_log = []
    history = []

    # Bootstrap
    print("  Bootstrapping...")
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

            for ep in range(num_episodes):
                for step in range(steps_per_episode):
                    npc.step(env, step)
                    obs_before = org.history[-1].copy() if org.history else np.zeros(idx['obs_dim'])

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

                    # Log co-activation
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

        # Get co-activation stats
        stats = gen_logger.get_stats()
        if stats is None:
            continue

        # Extract key metrics
        pain_conflict_lift = 0
        anxiety_loop = False
        for p in stats.get('top_coactivations', []):
            if (p['a'] == 'pain' and p['b'] == 'conflict') or \
               (p['a'] == 'conflict' and p['b'] == 'pain'):
                pain_conflict_lift = p['lift']

        pc_forward = 0
        cp_forward = 0
        for s in stats.get('top_sequences', []):
            if s['from'] == 'pain' and s['to'] == 'conflict':
                pc_forward = s['lift']
            if s['from'] == 'conflict' and s['to'] == 'pain':
                cp_forward = s['lift']
        if pc_forward > 1.5 and cp_forward > 1.5:
            anxiety_loop = True

        rec = {
            'generation': gen,
            'total_steps': stats['total_steps'],
            'unique_patterns': stats['n_unique_patterns'],
            'pain_conflict_lift': round(pain_conflict_lift, 3),
            'anxiety_loop': anxiety_loop,
            'pain_to_conflict_lift': round(pc_forward, 3),
            'conflict_to_pain_lift': round(cp_forward, 3),
            'top_coactivations': stats['top_coactivations'][:5],
            'top_sequences': stats['top_sequences'][:5],
            'recurring_patterns': stats['recurring_patterns'][:3],
            'activation_rates': stats['activation_rates'],
        }
        history.append(rec)

        print(f"  Steps: {stats['total_steps']}  Patterns: {stats['n_unique_patterns']}")
        print(f"  Pain<->Conflict: coact={pain_conflict_lift:.2f}  "
              f"P->C={pc_forward:.2f}  C->P={cp_forward:.2f}  "
              f"loop={'YES' if anxiety_loop else 'no'}")
        if stats['top_coactivations']:
            top = stats['top_coactivations'][0]
            print(f"  Top coactivation: {top['a']}+{top['b']} lift={top['lift']:.2f}")
        if stats['top_sequences']:
            top = stats['top_sequences'][0]
            print(f"  Top sequence: {top['from']}->{top['to']} lift={top['lift']:.2f}")

        # Reproduce
        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"gen{gen+1}_{i}"

    # Summary
    print("\n" + "=" * 60)
    print("CO-ACTIVATION EVOLUTION SUMMARY")
    print("=" * 60)
    print(f"\n{'Gen':>4} {'Patterns':>9} {'P<->C Coact':>10} {'P->C':>6} {'C->P':>6} {'Loop':>5}")
    print("-" * 45)
    for rec in history:
        print(f"{rec['generation']:>4} {rec['unique_patterns']:>9} "
              f"{rec['pain_conflict_lift']:>10.2f} "
              f"{rec['pain_to_conflict_lift']:>6.2f} "
              f"{rec['conflict_to_pain_lift']:>6.2f} "
              f"{'YES' if rec['anxiety_loop'] else 'no':>5}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'coactivation_deep_time.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to data/coactivation_deep_time.json")

    return history


if __name__ == '__main__':
    run_coactivation_deep_time(num_generations=10, seed=42)

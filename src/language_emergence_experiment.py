"""Language Emergence: Population Density Sweep.

Tests T106: is language emergence a sharp transition (percolation) in
population density? Sweeps population size from 2 to 16, runs 20
generations at each size, measures token receptor activation and
union-individual divergence.

Three predictions:
  1. Token receptors not fitness-positive before theory_of_mind
  2. Sharp transition in population density
  3. After transition, union decouples from individual count
"""

import os
import json
import numpy as np
from collections import deque
from environment import Environment, Organism
from mental_model import build_mental_model, action_to_hash
from model import compute_obs_indices
from train import train_model, generate_training_data
from thinking_substrate import ThinkingTree
from cognitive_state import CognitiveStateDetector
from receptor_discovery import discover, calibrate_null_thresholds
from multi_agent import PopulationManager
from live_receptors import LiveReceptorBank
from episode_receptors import EpisodeLevelReceptorBank

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_population_sweep(population_sizes=None, num_generations=20,
                         num_episodes=3, steps_per_episode=1000,
                         seed=42):
    if population_sizes is None:
        population_sizes = [2, 4, 8, 12, 16]

    print("LANGUAGE EMERGENCE: POPULATION DENSITY SWEEP")
    print(f"Populations: {population_sizes}")
    print(f"{num_generations} gens, {steps_per_episode} steps/ep, seed={seed}")
    print()

    idx = compute_obs_indices()
    all_results = {}

    for pop_size in population_sizes:
        print(f"\n{'=' * 60}")
        print(f"POPULATION SIZE: {pop_size}")
        print(f"{'=' * 60}")

        rng = np.random.RandomState(seed)

        # Bootstrap with standard training
        print("  Bootstrapping...")
        X, Y, Z, boot_log = generate_training_data(
            num_episodes=15, steps_per_episode=steps_per_episode, seed=seed)
        model = train_model(X, Y, Z, epochs=8, staged=True,
                            steps_per_episode=steps_per_episode)
        engine = build_mental_model(boot_log[-60000:])

        tree = ThinkingTree(num_actions=idx['num_actions'],
                            max_simulations=24, max_depth=4)
        cog_detector = CognitiveStateDetector()
        receptor_bank = LiveReceptorBank()
        episode_bank = EpisodeLevelReceptorBank()

        pm = PopulationManager(population_size=pop_size, seed=seed)
        history = []
        cumulative_log = list(boot_log)
        max_buffer = 60000

        cumulative_windows = deque(maxlen=max_buffer)
        cumulative_targets = deque(maxlen=max_buffer)
        cumulative_next_pain = deque(maxlen=max_buffer)
        for i in range(len(X)):
            cumulative_windows.append(X[i].astype(np.float32))
            cumulative_targets.append(Y[i].astype(np.float32))
            cumulative_next_pain.append(Z[i].astype(np.float32))

        null_thresh = None

        for gen in range(num_generations):
            print(f"\n--- Pop {pop_size}, Gen {gen} ---")

            model.to('cpu')
            model.eval()

            organisms, gen_stats = pm.run_generation(
                model, engine, tree, cog_detector,
                receptor_bank=receptor_bank,
                episode_bank=episode_bank,
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode)

            # Collect training data from all organisms
            for evo_org in pm.evo_organisms:
                cumulative_log.extend(evo_org.experience_log)
                for entry in evo_org.experience_log:
                    if 'obs_before' in entry:
                        obs_w = np.zeros((32, idx['obs_dim']), dtype=np.float32)
                        obs_w[-1] = entry['obs_before'][:idx['obs_dim']]
                        cumulative_windows.append(obs_w.astype(np.float32))
                        if 'action' in entry:
                            cumulative_targets.append(entry['action'].astype(np.float32))
                        pain = entry['obs_before'][:6]
                        cumulative_next_pain.append(pain.astype(np.float32))

            # Retrain
            import torch
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(DEVICE)

            if gen > 0 and len(cumulative_windows) >= 100:
                try:
                    X_t = np.array(list(cumulative_windows)[-max_buffer:], dtype=np.float32)
                    Y_t = np.array(list(cumulative_targets)[-max_buffer:], dtype=np.float32)
                    Z_t = np.array(list(cumulative_next_pain)[-max_buffer:], dtype=np.float32)
                    if X_t.shape[0] == Y_t.shape[0] == Z_t.shape[0]:
                        model = train_model(X_t, Y_t, Z_t, epochs=3, staged=True,
                                            steps_per_episode=steps_per_episode)
                except Exception as e:
                    print(f"  Training skipped: {e}")

            log_slice = cumulative_log[-max_buffer:]
            engine = build_mental_model(log_slice)

            # Receptor discovery every 5 generations
            discovered_sets = []
            if gen % 5 == 4 or gen == num_generations - 1:
                if null_thresh is None and len(log_slice) >= 200:
                    null_thresh = calibrate_null_thresholds(
                        log_slice, engine, num_shuffles=5)

                if null_thresh is not None:
                    for evo_org in pm.evo_organisms:
                        if len(evo_org.experience_log) >= 100:
                            org_engine = build_mental_model(evo_org.experience_log[-10000:])
                            results = discover(evo_org.experience_log[-10000:],
                                               org_engine,
                                               threshold_overrides=null_thresh,
                                               log_provenance='oracle')
                            evo_org.discovered_receptors = results['discovered']
                            discovered_sets.append(set(results['discovered']))

            # Population topology
            topo = pm.get_population_topology()

            # Emission diversity
            emission_patterns = set()
            for evo_org in pm.evo_organisms:
                for entry in evo_org.experience_log[-steps_per_episode:]:
                    action = entry.get('action', np.zeros(idx['num_actions']))
                    L = 6
                    emission = action[L*3:]
                    if len(emission) >= 4:
                        pattern = tuple(int(e) for e in emission[:4])
                        if sum(pattern) > 0:
                            emission_patterns.add(pattern)

            # Check for language-related receptors
            language_receptors = {'naming', 'self_talk', 'referential_grounding',
                                  'theory_of_mind', 'belief_attribution',
                                  'social_learning', 'cultural_transmission'}
            found_language = language_receptors & set(topo.get('union', []))

            rec = {
                'generation': gen,
                'population_size': pop_size,
                'token_emissions': gen_stats['token_emissions'],
                'token_receptions': gen_stats['token_receptions'],
                'emission_diversity': len(emission_patterns),
                'avg_fitness': gen_stats['avg_fitness'],
                'union_size': topo['union_size'],
                'intersection_size': topo['intersection_size'],
                'individual_sizes': topo['individual_sizes'],
                'max_individual': topo['max_individual'],
                'union_individual_gap': topo['union_size'] - topo['max_individual'],
                'language_receptors_found': sorted(found_language),
                'num_language_receptors': len(found_language),
            }
            history.append(rec)

            print(f"  Fitness: {gen_stats['avg_fitness']:.1f}  "
                  f"Tokens: {gen_stats['token_emissions']} emit, "
                  f"{gen_stats['token_receptions']} recv")
            print(f"  Emission diversity: {len(emission_patterns)} patterns")
            print(f"  Topology: union={topo['union_size']}, "
                  f"intersection={topo['intersection_size']}, "
                  f"gap={topo['union_size'] - topo['max_individual']}")
            if found_language:
                print(f"  Language receptors: {sorted(found_language)}")

            # Reproduce
            if gen < num_generations - 1:
                pm.select_and_reproduce()

        all_results[pop_size] = history

    # Summary
    print("\n" + "=" * 70)
    print("POPULATION DENSITY SWEEP SUMMARY")
    print("=" * 70)
    print(f"\n{'Pop':>4} {'Emissions':>10} {'Diversity':>10} {'Union':>6} "
          f"{'MaxInd':>7} {'Gap':>5} {'Lang':>5}")
    print("-" * 55)

    for pop_size in population_sizes:
        final = all_results[pop_size][-1]
        print(f"{pop_size:>4} {final['token_emissions']:>10} "
              f"{final['emission_diversity']:>10} "
              f"{final['union_size']:>6} {final['max_individual']:>7} "
              f"{final['union_individual_gap']:>5} "
              f"{final['num_language_receptors']:>5}")

    # Check for percolation
    emission_by_pop = {ps: all_results[ps][-1]['emission_diversity']
                       for ps in population_sizes}
    max_div = max(emission_by_pop.values()) if emission_by_pop else 0
    if max_div > 0:
        threshold_candidates = [ps for ps in population_sizes
                                if emission_by_pop[ps] > max_div * 0.5]
        if threshold_candidates:
            threshold = min(threshold_candidates)
            print(f"\nEmission diversity threshold at pop_size={threshold}")
            below = [ps for ps in population_sizes if ps < threshold]
            above = [ps for ps in population_sizes if ps >= threshold]
            if below and above:
                below_div = np.mean([emission_by_pop[ps] for ps in below])
                above_div = np.mean([emission_by_pop[ps] for ps in above])
                ratio = above_div / max(below_div, 0.1)
                if ratio > 3:
                    print(f"  SHARP transition: above/below ratio = {ratio:.1f}")
                else:
                    print(f"  Gradual transition: above/below ratio = {ratio:.1f}")

    # Check union-individual divergence
    print("\nUnion-Individual Divergence:")
    for pop_size in population_sizes:
        gaps = [r['union_individual_gap'] for r in all_results[pop_size]
                if r['union_size'] > 0]
        if gaps:
            print(f"  Pop {pop_size}: mean gap = {np.mean(gaps):.1f}, "
                  f"max gap = {max(gaps)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    output = {'population_sizes': population_sizes, 'results': all_results}
    with open(os.path.join(DATA_DIR, 'language_emergence.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to data/language_emergence.json")

    return all_results


if __name__ == '__main__':
    run_population_sweep(seed=42)

"""Cross-Environment Transfer Experiment.

Evolve organisms in environment A (T4 physics with hidden confounders),
then drop them into environment B (T7+T8 abstract problems) they've
never seen. Compare against naive organisms.

Measures:
  - Time to first receptor discovery in the new environment
  - Final receptor count vs naive baseline
  - Thinking influence (does depth_reached transfer?)
  - Novel receptor emergence
  - Which receptors transfer and which don't
"""

import os
import json
import time
import numpy as np
from environment import Environment, Organism, NPC
from environment_tiers import TieredEnvironment
from abstract_env import CombinedT7T8Environment
from physics_world import PhysicsWorld
from mental_model import build_mental_model, action_to_hash
from receptor_discovery import discover, calibrate_null_thresholds
from model import compute_obs_indices, HierarchicalPolicy
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from thinking_influence import measure_thinking_influence
from novel_receptor_detector import detect_novel_receptors
from deep_time import EvolvingOrganism, select_and_reproduce
from deep_time_overnight import run_generation_rich, load_checkpoint

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_naive_baseline(num_generations=10, population_size=4,
                       num_episodes=5, steps_per_episode=200,
                       epochs_per_gen=8, tier=4, seed=99):
    """Run naive organisms in T7+T8 — no prior evolution."""
    import torch
    print("=" * 60)
    print("NAIVE BASELINE: No prior evolution")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']

    organisms = [EvolvingOrganism(f"naive_{i}") for i in range(population_size)]
    history = []
    cumulative_log = []

    # Bootstrap from oracle
    print("  Bootstrapping naive organisms...")
    X, Y, Z, boot_log = generate_training_data(
        num_episodes=20, steps_per_episode=steps_per_episode, seed=seed)
    model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                        steps_per_episode=steps_per_episode)
    engine = build_mental_model(boot_log)
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=3)

    null_thresh = calibrate_null_thresholds(boot_log[:30000], engine, num_shuffles=5)
    cumulative_log.extend(boot_log)
    cumulative_windows = list(X)
    cumulative_targets = list(Y)
    cumulative_next_pain = list(Z)

    for gen in range(num_generations):
        print(f"\n--- Naive Gen {gen} ---")
        world_state, windows, targets, next_pain = run_generation_rich(
            organisms, gen, model=model, engine=engine, tree=tree, rng=rng,
            steps_per_episode=steps_per_episode, num_episodes=num_episodes,
            tier=tier, use_oracle=(gen == 0))

        for evo_org in organisms:
            cumulative_log.extend(evo_org.experience_log)
        if windows:
            cumulative_windows.extend(windows)
            cumulative_targets.extend(targets)
            cumulative_next_pain.extend(next_pain)

        X = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Y = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Z = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        if len(X) >= 100:
            model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                                steps_per_episode=steps_per_episode)

        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)

        gen_discovered = set()
        for evo_org in organisms:
            if len(evo_org.experience_log) >= 100:
                org_engine = build_mental_model(evo_org.experience_log)
                results = discover(evo_org.experience_log, org_engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='policy')
                evo_org.discovered_receptors = results['discovered']
                gen_discovered.update(results['discovered'])

        rec = {
            'generation': gen,
            'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
            'num_discovered': len(gen_discovered),
            'discovered': sorted(gen_discovered),
        }
        history.append(rec)
        print(f"  Fitness: {rec['avg_fitness']:.0f}  Receptors: {rec['num_discovered']}")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org in enumerate(organisms):
                org.organism_id = f"naive_gen{gen+1}_{i}"

    return history


def run_transfer(checkpoint_gen=29, num_generations=10, population_size=4,
                 num_episodes=5, steps_per_episode=200,
                 epochs_per_gen=8, tier=4, seed=99):
    """Load evolved organisms from checkpoint, drop into new environment."""
    import torch
    print("=" * 60)
    print(f"TRANSFER: From gen {checkpoint_gen} checkpoint")
    print("=" * 60)

    cp = load_checkpoint(checkpoint_gen)
    if cp is None:
        print(f"  ERROR: No checkpoint at gen {checkpoint_gen}")
        return []

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']

    # Restore organisms with their evolved topology bias
    organisms = []
    for os_data in cp['organisms']:
        evo = EvolvingOrganism(
            os_data['organism_id'],
            parent_bias=os_data['topology_bias'],
            body_params=os_data['body_params'])
        evo.fitness = 0.0  # reset fitness for new environment
        evo.discovered_receptors = os_data['discovered_receptors']
        organisms.append(evo)

    print(f"  Loaded {len(organisms)} organisms with bias size: "
          f"{np.mean([len(o.topology_bias) for o in organisms]):.0f}")

    # Restore model
    model_sd = cp.get('model_state_dict')
    if model_sd is not None:
        model = HierarchicalPolicy(
            obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
            energy_obs_index=idx['energy'],
            conflict_obs_index=idx['conflict'],
            staged=True).to('cpu')
        model.load_state_dict(model_sd)
    else:
        print("  WARNING: No model in checkpoint, using oracle")
        model = None

    # Restore log and engine
    ld = cp['log_data']
    cumulative_log = ld['cumulative_log']
    cumulative_windows = ld['cumulative_windows']
    cumulative_targets = ld['cumulative_targets']
    cumulative_next_pain = ld['cumulative_next_pain']

    engine = build_mental_model(cumulative_log[-30000:])
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=3)
    null_thresh = cp.get('null_thresh', {})

    history = []

    for gen in range(num_generations):
        print(f"\n--- Transfer Gen {gen} ---")
        world_state, windows, targets, next_pain = run_generation_rich(
            organisms, gen, model=model, engine=engine, tree=tree, rng=rng,
            steps_per_episode=steps_per_episode, num_episodes=num_episodes,
            tier=tier, use_oracle=False)

        for evo_org in organisms:
            cumulative_log.extend(evo_org.experience_log)
        if windows:
            cumulative_windows.extend(windows)
            cumulative_targets.extend(targets)
            cumulative_next_pain.extend(next_pain)

        X = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Y = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Z = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        if len(X) >= 100:
            model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                                steps_per_episode=steps_per_episode)

        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)

        gen_discovered = set()
        for evo_org in organisms:
            if len(evo_org.experience_log) >= 100:
                org_engine = build_mental_model(evo_org.experience_log)
                results = discover(evo_org.experience_log, org_engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='policy')
                evo_org.discovered_receptors = results['discovered']
                gen_discovered.update(results['discovered'])

        ti = None
        if gen % 5 == 0 or gen == num_generations - 1:
            ti = measure_thinking_influence(model, engine, num_episodes=5,
                                             steps_per_episode=100, seed=99)

        rec = {
            'generation': gen,
            'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
            'num_discovered': len(gen_discovered),
            'discovered': sorted(gen_discovered),
            'bias_size': round(float(np.mean([len(o.topology_bias) for o in organisms])), 1),
        }
        if ti:
            rec['thinking'] = {
                'divergence': ti['ablation_divergence'],
                'reward_diff': ti['reward_difference'],
                'partial_corr': ti['thinking_action_partial_corr'],
                'depth': ti['channel_influence'].get('depth_reached', 0),
            }
        history.append(rec)
        print(f"  Fitness: {rec['avg_fitness']:.0f}  Receptors: {rec['num_discovered']}  "
              f"Bias: {rec.get('bias_size', 0):.0f}")
        if ti:
            print(f"  Thinking: pcorr={ti['thinking_action_partial_corr']:.4f}  "
                  f"depth={ti['channel_influence'].get('depth_reached', 0):.4f}")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org in enumerate(organisms):
                org.organism_id = f"transfer_gen{gen+1}_{i}"

    return history


if __name__ == '__main__':
    NUM_GENS = 10

    # Run naive baseline
    naive = run_naive_baseline(num_generations=NUM_GENS, seed=99)

    # Run transfer from gen 29 checkpoint
    transfer = run_transfer(checkpoint_gen=29, num_generations=NUM_GENS, seed=99)

    # Comparison
    print("\n" + "=" * 60)
    print("CROSS-ENVIRONMENT TRANSFER COMPARISON")
    print("=" * 60)

    print(f"\n{'Gen':>4} {'Naive Recept':>13} {'Transfer Recept':>15} {'Advantage':>10}")
    print("-" * 48)
    for n, t in zip(naive, transfer):
        adv = t['num_discovered'] - n['num_discovered']
        print(f"{n['generation']:>4} {n['num_discovered']:>13} "
              f"{t['num_discovered']:>15} {adv:>+10}")

    naive_final = set(naive[-1]['discovered'])
    transfer_final = set(transfer[-1]['discovered'])
    transfer_only = transfer_final - naive_final
    naive_only = naive_final - transfer_final
    both = naive_final & transfer_final

    print(f"\nFinal generation:")
    print(f"  Both: {len(both)}")
    print(f"  Transfer-only: {len(transfer_only)}")
    if transfer_only:
        print(f"    {sorted(transfer_only)}")
    print(f"  Naive-only: {len(naive_only)}")
    if naive_only:
        print(f"    {sorted(naive_only)}")

    naive_unique = set()
    transfer_unique = set()
    for r in naive:
        naive_unique.update(r['discovered'])
    for r in transfer:
        transfer_unique.update(r['discovered'])

    print(f"\nCumulative unique:")
    print(f"  Naive: {len(naive_unique)}")
    print(f"  Transfer: {len(transfer_unique)}")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'cross_env_transfer.json'), 'w') as f:
        json.dump({
            'naive': naive,
            'transfer': transfer,
            'naive_unique': sorted(naive_unique),
            'transfer_unique': sorted(transfer_unique),
            'transfer_only': sorted(transfer_only),
            'naive_only': sorted(naive_only),
        }, f, indent=2)
    print(f"\nSaved to data/cross_env_transfer.json")

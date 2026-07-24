"""Self-play vs Oracle comparison experiment.

Runs two pipelines side-by-side:
1. Oracle-only: standard closed-loop training (oracle decides actions)
2. Self-play: bootstrap from oracle, then policy drives behavior

Compares: training accuracy, reward, and receptor discovery counts.
"""

import os
import json
import time
import numpy as np
from environment import Environment, Organism, NPC
from train import (generate_training_data_closed_loop,
                   generate_training_data_self_play,
                   train_model)
from receptor_discovery import discover, calibrate_null_thresholds, build_tests
from mental_model import build_mental_model

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SEED = 42


def run_oracle_pipeline(num_episodes=200, steps=300, epochs=15):
    print("\n" + "=" * 60)
    print("PIPELINE A: Oracle-only (closed-loop)")
    print("=" * 60)

    t0 = time.time()
    X, Y, Z, log, engine = generate_training_data_closed_loop(
        num_bootstrap=50, num_online=num_episodes - 50,
        steps_per_episode=steps, seed=SEED)
    gen_time = time.time() - t0

    print(f"\nTraining model ({epochs} epochs)...")
    model = train_model(X, Y, Z, epochs=epochs, staged=True,
                        steps_per_episode=steps)

    print(f"\nData generation: {gen_time:.1f}s")
    print(f"X shape: {X.shape}")
    print(f"Log entries: {len(log)}")
    print(f"Store mappings: {engine.store.total_count}")

    return log, engine, model, X.shape[0]


def run_self_play_pipeline(num_bootstrap=50, num_self_play=75,
                           num_iterations=2, steps=300, epochs=15):
    print("\n" + "=" * 60)
    print("PIPELINE B: Self-play")
    print("=" * 60)

    t0 = time.time()
    X, Y, Z, log, engine, model = generate_training_data_self_play(
        num_bootstrap=num_bootstrap,
        num_self_play=num_self_play,
        num_iterations=num_iterations,
        steps_per_episode=steps,
        epochs_per_iter=epochs,
        seed=SEED)
    gen_time = time.time() - t0

    print(f"\nData generation: {gen_time:.1f}s")
    print(f"X shape: {X.shape}")
    print(f"Log entries: {len(log)}")
    print(f"Store mappings: {engine.store.total_count}")

    return log, engine, model, X.shape[0]


def run_discovery(log, engine, label):
    print(f"\n--- Receptor Discovery: {label} ---")

    print("  Calibrating null thresholds...")
    thresholds = calibrate_null_thresholds(log, engine, num_shuffles=5, percentile=95)

    print("  Running discovery...")
    results = discover(log, engine, threshold_overrides=thresholds,
                       log_provenance='policy')

    discovered = results['discovered']
    not_found = results['not_found']
    skipped = results['skipped']
    details = results['details']

    print(f"  Discovered: {len(discovered)}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Skipped: {len(skipped)}")

    by_family = {}
    for d in details:
        if d.get('discovered'):
            fam = d['family']
            by_family[fam] = by_family.get(fam, 0) + 1
    print("  Per-family:")
    for fam in sorted(by_family.keys()):
        print(f"    {fam}: {by_family[fam]}")

    return results, details


def compute_reward_stats(log, steps_per_episode=300):
    num_episodes = len(log) // steps_per_episode
    rewards = []
    for ep in range(num_episodes):
        start = ep * steps_per_episode
        end = start + steps_per_episode
        ep_log = log[start:end]
        ep_reward = sum(
            np.mean(e['obs_after'][6:12]) - np.mean(e['obs_after'][0:6])
            for e in ep_log
        )
        rewards.append(ep_reward)
    return {
        'mean': float(np.mean(rewards)) if rewards else 0,
        'std': float(np.std(rewards)) if rewards else 0,
        'min': float(np.min(rewards)) if rewards else 0,
        'max': float(np.max(rewards)) if rewards else 0,
        'num_episodes': num_episodes,
    }


if __name__ == '__main__':
    STEPS = 300
    EPOCHS = 10

    # Pipeline A: Oracle (200 episodes total)
    oracle_log, oracle_engine, oracle_model, oracle_samples = \
        run_oracle_pipeline(num_episodes=200, steps=STEPS, epochs=EPOCHS)

    # Pipeline B: Self-play (50 bootstrap + 2x75 self-play = 200 episodes total)
    sp_log, sp_engine, sp_model, sp_samples = \
        run_self_play_pipeline(num_bootstrap=50, num_self_play=75,
                               num_iterations=2, steps=STEPS, epochs=EPOCHS)

    # Receptor discovery
    oracle_results, oracle_disc = run_discovery(oracle_log, oracle_engine, "Oracle")
    sp_results, sp_disc = run_discovery(sp_log, sp_engine, "Self-play")

    # Reward stats
    oracle_rewards = compute_reward_stats(oracle_log, STEPS)
    sp_rewards = compute_reward_stats(sp_log, STEPS)

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Oracle vs Self-play")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Oracle':>10} {'Self-play':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Training samples':<30} {oracle_samples:>10} {sp_samples:>10}")
    print(f"  {'Log entries':<30} {len(oracle_log):>10} {len(sp_log):>10}")
    print(f"  {'Store mappings':<30} {oracle_engine.store.total_count:>10} {sp_engine.store.total_count:>10}")
    print(f"  {'Mean reward':<30} {oracle_rewards['mean']:>10.1f} {sp_rewards['mean']:>10.1f}")

    oracle_ids = set(oracle_results['discovered'])
    sp_ids = set(sp_results['discovered'])
    print(f"  {'Receptors discovered':<30} {len(oracle_ids):>10} {len(sp_ids):>10}")
    both = oracle_ids & sp_ids
    oracle_only = oracle_ids - sp_ids
    sp_only = sp_ids - oracle_ids

    print(f"\n  Both: {len(both)}")
    print(f"  Oracle-only: {len(oracle_only)}")
    if oracle_only:
        for r in sorted(oracle_only):
            print(f"    - {r}")
    print(f"  Self-play-only: {len(sp_only)}")
    if sp_only:
        for r in sorted(sp_only):
            print(f"    + {r}")

    # Save results
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {
        'oracle': {
            'samples': oracle_samples,
            'log_entries': len(oracle_log),
            'store_mappings': oracle_engine.store.total_count,
            'rewards': oracle_rewards,
            'discovered': len(oracle_ids),
            'discovered_ids': sorted(oracle_ids),
        },
        'self_play': {
            'samples': sp_samples,
            'log_entries': len(sp_log),
            'store_mappings': sp_engine.store.total_count,
            'rewards': sp_rewards,
            'discovered': len(sp_ids),
            'discovered_ids': sorted(sp_ids),
        },
        'comparison': {
            'both': sorted(both),
            'oracle_only': sorted(oracle_only),
            'self_play_only': sorted(sp_only),
        },
        'seed': SEED,
    }
    with open(os.path.join(DATA_DIR, 'self_play_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.join(DATA_DIR, 'self_play_comparison.json')}")

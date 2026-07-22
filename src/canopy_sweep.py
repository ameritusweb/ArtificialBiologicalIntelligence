"""Step 55: Canopy activation sweep.

Run receptor discovery across the physics world at all 8 tiers to test
whether canopy receptors emerge that couldn't in the field world.

Predictions:
  - grip_affordance, lever_affordance, composite_affordance emerge at Tier 3+
  - capability_change_detection emerges when developmental changes present
  - niche_construction emerges when persistence crosses generations
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from environment_tiers import TieredEnvironment
from physics_world import PhysicsWorld
from receptor_discovery import discover as discover_receptors
from mental_model import build_mental_model
from model import compute_obs_indices

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_physics_tier(tier, num_episodes=20, steps_per_episode=200,
                     compound_objects=True, developmental=False,
                     persistent=False, seed=0):
    """Run receptor discovery in a physics world at a specific tier."""
    rng = np.random.RandomState(seed)
    all_logs = []

    for ep in range(num_episodes):
        env_seed = rng.randint(0, 100000)
        if tier > 0:
            env = TieredEnvironment(tier=tier, seed=env_seed)
        else:
            env = Environment(seed=env_seed)
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        pw = PhysicsWorld(env, org, num_objects=3, seed=rng.randint(0, 100000))
        if compound_objects:
            pw.add_compound_objects()
        if persistent:
            pw.enable_persistence()

        for step in range(steps_per_episode):
            npc.step(env, step)
            actions = org.compute_optimal_actions(env, step, npc=npc)
            pw.apply_organism_forces(actions)
            pw.check_grips(actions)
            if developmental:
                pw.apply_developmental_changes(step)
            pw.step()
            obs, reward = org.step(actions, env, step, npc=npc)
            npc.receive_signal(actions[org.NUM_LIMBS * 3:], org.x, org.y)

        if persistent and ep < num_episodes - 1:
            pw.save_world_state()

        all_logs.extend(org.experience_log)

    return all_logs


def run_sweep(num_episodes_per_tier=20, steps=200, seed=0):
    print("=== Step 55: Canopy Activation Sweep ===\n")
    print("Running receptor discovery across physics world at all 8 tiers.\n")

    tiers = list(range(8))
    results = {}

    for tier in tiers:
        developmental = tier >= 4
        persistent = tier >= 5
        compounds = tier >= 3

        print(f"--- Tier {tier} (compounds={compounds}, dev={developmental}, persist={persistent}) ---")
        logs = run_physics_tier(
            tier, num_episodes=num_episodes_per_tier,
            steps_per_episode=steps,
            compound_objects=compounds,
            developmental=developmental,
            persistent=persistent,
            seed=seed + tier,
        )
        print(f"  Log entries: {len(logs)}")

        engine = build_mental_model(logs)
        disco_results = discover_receptors(logs, engine)
        n_discovered = len(disco_results['discovered'])
        n_total = len(disco_results['scores'])
        details = {k: (k in disco_results['discovered']) for k in disco_results['scores']}
        results[tier] = {
            'discovered': n_discovered,
            'total_tests': n_total,
            'details': details,
            'compounds': compounds,
            'developmental': developmental,
            'persistent': persistent,
        }
        print(f"  Discovered: {n_discovered}/{n_total}")
        print(f"  Receptors: {disco_results['discovered'][:10]}{'...' if n_discovered > 10 else ''}")

    print("\n=== EMERGENCE MATRIX ===")
    all_receptors = set()
    for r in results.values():
        all_receptors.update(r['details'].keys())
    all_receptors = sorted(all_receptors)

    invariant = [r for r in all_receptors if all(results[t]['details'].get(r, False) for t in tiers)]
    tier_specific = {}
    for t in tiers:
        specific = [r for r in all_receptors
                    if results[t]['details'].get(r, False)
                    and not all(results[t2]['details'].get(r, False) for t2 in tiers)]
        if specific:
            tier_specific[t] = specific

    print(f"\n  Invariant across all tiers: {len(invariant)}")
    for r in invariant[:15]:
        print(f"    {r}")
    if len(invariant) > 15:
        print(f"    ... and {len(invariant) - 15} more")

    print(f"\n  Tier-specific receptors:")
    for t, receptors in tier_specific.items():
        print(f"    Tier {t}: {receptors}")

    print(f"\n  Discovery counts per tier:")
    for t in tiers:
        print(f"    Tier {t}: {results[t]['discovered']}/{results[t]['total_tests']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        'per_tier': {str(t): results[t] for t in tiers},
        'invariant': invariant,
        'tier_specific': {str(t): v for t, v in tier_specific.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'canopy_sweep_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to data/canopy_sweep_results.json")
    return output


if __name__ == '__main__':
    run_sweep(num_episodes_per_tier=10, steps=150, seed=42)

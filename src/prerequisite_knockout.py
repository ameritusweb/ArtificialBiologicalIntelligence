"""Prerequisite Knockout Experiment.

Take evolved organisms, ablate a prerequisite receptor from their
topology bias, continue evolving. Does the canopy receptor that
depended on it survive?

If yes: the DAG encodes developmental order, not functional dependency.
If no: the DAG encodes genuine functional dependency.

Test case: ablate conflation, check if epistemic_strategy survives.
"""

import os
import json
import numpy as np
from deep_time_overnight import (run_overnight, load_checkpoint,
                                  find_latest_checkpoint, _checkpoint_dir)
from deep_time import EvolvingOrganism, select_and_reproduce
from mental_model import build_mental_model
from receptor_discovery import discover, calibrate_null_thresholds

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_knockout(checkpoint_gen, knockout_receptor, target_receptor,
                 num_generations=15, seed=42):
    """Load checkpoint, remove a prerequisite, continue evolving."""
    print("=" * 60)
    print(f"PREREQUISITE KNOCKOUT: remove '{knockout_receptor}', "
          f"watch '{target_receptor}'")
    print("=" * 60)

    cp = load_checkpoint(checkpoint_gen, seed=seed)
    if cp is None:
        print(f"  No checkpoint at gen {checkpoint_gen} for seed {seed}")
        return []

    # Restore organisms
    organisms = []
    for os_data in cp['organisms']:
        evo = EvolvingOrganism(
            os_data['organism_id'],
            parent_bias=dict(os_data['topology_bias']),
            body_params=dict(os_data['body_params']))
        evo.fitness = 0.0
        evo.discovered_receptors = list(os_data['discovered_receptors'])
        organisms.append(evo)

    # Ablate the knockout receptor from ALL organisms' topology bias
    for org in organisms:
        if knockout_receptor in org.topology_bias:
            del org.topology_bias[knockout_receptor]
            print(f"  Removed '{knockout_receptor}' from {org.organism_id}")

    # Check if target is currently in discovered
    has_target = any(target_receptor in org.discovered_receptors for org in organisms)
    print(f"  Target '{target_receptor}' currently discovered: {has_target}")
    print(f"  Knockout '{knockout_receptor}' removed from bias")
    print(f"  Running {num_generations} more generations...\n")

    # Restore cumulative data
    ld = cp['log_data']
    cumulative_log = ld['cumulative_log']
    null_thresh = cp.get('null_thresh', {})

    history = []
    rng = np.random.RandomState(seed + 1000)

    for gen in range(num_generations):
        # Simple evolution loop (no physics/MCTS for speed)
        from environment import Environment, NPC
        from train import EXPLORE_RATE, PROBE_RATE_FLOOR
        from model import compute_obs_indices
        idx = compute_obs_indices()

        for evo_org in organisms:
            env = Environment(seed=rng.randint(0, 100000))
            org = evo_org.create_organism(rng)
            npc = NPC()
            npc.reset(rng)

            for ep in range(5):
                for step in range(200):
                    npc.step(env, step)
                    actions = org.compute_optimal_actions(env, step, npc=npc)
                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(idx['num_actions'], dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=idx['num_actions']).astype(np.int32)
                    else:
                        executed = actions
                    obs, reward = org.step(executed, env, step, npc=npc)
                    evo_org.fitness += reward

            cumulative_log.extend(org.experience_log)

        # Discovery — use cumulative log, not per-organism
        log_slice = cumulative_log[-60000:]
        gen_discovered = set()
        if len(log_slice) >= 500:
            slice_engine = build_mental_model(log_slice)
            results = discover(log_slice, slice_engine,
                               threshold_overrides=null_thresh,
                               log_provenance='oracle')
            gen_discovered = set(results['discovered'])
            for evo_org in organisms:
                evo_org.discovered_receptors = results['discovered']

        knockout_back = knockout_receptor in gen_discovered
        target_alive = target_receptor in gen_discovered

        rec = {
            'generation': gen,
            'num_discovered': len(gen_discovered),
            'knockout_rediscovered': knockout_back,
            'target_alive': target_alive,
            'discovered': sorted(gen_discovered),
        }
        history.append(rec)
        print(f"  Gen {gen}: receptors={len(gen_discovered)}, "
              f"'{knockout_receptor}'={'BACK' if knockout_back else 'gone'}, "
              f"'{target_receptor}'={'ALIVE' if target_alive else 'GONE'}")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, 4, rng)
            for i, org in enumerate(organisms):
                org.organism_id = f"knockout_gen{gen+1}_{i}"
                if knockout_receptor in org.topology_bias:
                    del org.topology_bias[knockout_receptor]

    # Summary
    target_survived_count = sum(1 for r in history if r['target_alive'])
    knockout_rediscovered = sum(1 for r in history if r['knockout_rediscovered'])

    print(f"\n{'='*60}")
    print(f"KNOCKOUT RESULT")
    print(f"{'='*60}")
    print(f"  '{target_receptor}' survived in {target_survived_count}/{num_generations} generations")
    print(f"  '{knockout_receptor}' rediscovered in {knockout_rediscovered}/{num_generations} generations")

    if target_survived_count > num_generations * 0.5:
        print(f"  CONCLUSION: DAG encodes DEVELOPMENTAL ORDER, not functional dependency")
        print(f"    {target_receptor} survives without {knockout_receptor}")
    else:
        print(f"  CONCLUSION: DAG encodes FUNCTIONAL DEPENDENCY")
        print(f"    {target_receptor} requires {knockout_receptor}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'prerequisite_knockout.json'), 'w') as f:
        json.dump({
            'knockout_receptor': knockout_receptor,
            'target_receptor': target_receptor,
            'checkpoint_gen': checkpoint_gen,
            'history': history,
            'target_survived': target_survived_count,
            'knockout_rediscovered': knockout_rediscovered,
            'conclusion': 'developmental_order' if target_survived_count > num_generations * 0.5
                          else 'functional_dependency',
        }, f, indent=2)
    print(f"  Saved to data/prerequisite_knockout.json")

    return history


if __name__ == '__main__':
    run_knockout(
        checkpoint_gen=29,
        knockout_receptor='conflation',
        target_receptor='epistemic_strategy',
        num_generations=15,
        seed=42,
    )

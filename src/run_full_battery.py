"""Run the full 165-test receptor battery with null calibration.

Generates both oracle and closed-loop data, runs all tests on
the appropriate data type, calibrates thresholds from shuffled nulls,
and reports discoveries per evidence channel.

This is the single command that produces the honest receptor count.
"""

import os
import json
import time
import numpy as np
from train import (generate_training_data_physics,
                    generate_training_data_closed_loop)
from mental_model import build_mental_model
from receptor_discovery import (build_tests, discover,
                                 calibrate_null_thresholds)
from model import compute_obs_indices

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_pipeline(oracle_episodes=30, closed_loop_bootstrap=50,
                 closed_loop_online=100, steps=200,
                 null_shuffles=10, seed=42):
    t0 = time.time()
    idx = compute_obs_indices()
    tests = build_tests()
    cl_tests = [t for t in tests if getattr(t, 'needs_closed_loop', False)]
    any_tests = [t for t in tests if not getattr(t, 'needs_closed_loop', False)]

    print("=== Full Receptor Battery Pipeline ===\n")
    print(f"  Tests: {len(tests)} total ({len(any_tests)} any-log, {len(cl_tests)} closed-loop)")
    print(f"  Oracle episodes: {oracle_episodes}")
    print(f"  Closed-loop: {closed_loop_bootstrap} bootstrap + {closed_loop_online} online")
    print(f"  Null shuffles: {null_shuffles}")

    # Phase 1: Oracle data (physics world)
    print("\n--- Phase 1: Oracle data (physics world) ---")
    X_o, Y_o, Z_o, oracle_log = generate_training_data_physics(
        num_episodes=oracle_episodes, steps_per_episode=steps,
        seed=seed, compound_objects=True, developmental=True)
    print(f"  Oracle log: {len(oracle_log)} entries ({time.time()-t0:.0f}s)")

    print("  Building oracle engine...")
    oracle_engine = build_mental_model(oracle_log)
    print(f"  Store: {oracle_engine.store.total_count} mappings")

    # Phase 2: Closed-loop data
    print("\n--- Phase 2: Closed-loop data ---")
    X_cl, Y_cl, Z_cl, cl_log, cl_engine = generate_training_data_closed_loop(
        num_bootstrap=closed_loop_bootstrap, num_online=closed_loop_online,
        steps_per_episode=steps, seed=seed)
    print(f"  Closed-loop log: {len(cl_log)} entries ({time.time()-t0:.0f}s)")
    print(f"  Store: {cl_engine.store.total_count} mappings")

    # Phase 3: Null calibration (on oracle data — any-log tests)
    print(f"\n--- Phase 3: Null calibration ({null_shuffles} shuffles) ---")
    oracle_null = calibrate_null_thresholds(
        oracle_log, oracle_engine, num_shuffles=null_shuffles)
    print(f"  Oracle null thresholds calibrated ({time.time()-t0:.0f}s)")

    # Phase 3b: Null calibration for closed-loop tests
    cl_null = calibrate_null_thresholds(
        cl_log, cl_engine, num_shuffles=null_shuffles)
    print(f"  Closed-loop null thresholds calibrated ({time.time()-t0:.0f}s)")

    # Merge null thresholds
    null_thresh = {**oracle_null, **cl_null}

    # Phase 4: Run discovery
    print("\n--- Phase 4: Discovery ---")

    print("  Running any-log tests on oracle data...")
    oracle_results = discover(oracle_log, oracle_engine,
                               threshold_overrides=null_thresh,
                               log_provenance='oracle')

    print("  Running closed-loop tests on policy data...")
    cl_results = discover(cl_log, cl_engine,
                           threshold_overrides=null_thresh,
                           log_provenance='policy')

    # Merge results: oracle covers any-log tests, cl covers CL tests
    merged = {
        'discovered': [],
        'skipped': [],
        'not_found': [],
        'scores': {},
        'null_thresholds': {},
        'details': [],
    }

    cl_ids = set(t.receptor_id for t in cl_tests)

    for d in oracle_results['details']:
        if d['receptor_id'] not in cl_ids:
            merged['details'].append(d)
            if d.get('skipped'):
                merged['skipped'].append(d['receptor_id'])
            elif d['discovered']:
                merged['discovered'].append(d['receptor_id'])
            else:
                merged['not_found'].append(d['receptor_id'])
            if d['score'] is not None:
                merged['scores'][d['receptor_id']] = d['score']
                merged['null_thresholds'][d['receptor_id']] = null_thresh.get(d['receptor_id'], 0)

    for d in cl_results['details']:
        if d['receptor_id'] in cl_ids:
            merged['details'].append(d)
            if d.get('skipped'):
                merged['skipped'].append(d['receptor_id'])
            elif d['discovered']:
                merged['discovered'].append(d['receptor_id'])
            else:
                merged['not_found'].append(d['receptor_id'])
            if d['score'] is not None:
                merged['scores'][d['receptor_id']] = d['score']
                merged['null_thresholds'][d['receptor_id']] = null_thresh.get(d['receptor_id'], 0)

    n_disc = len(merged['discovered'])
    n_skip = len(merged['skipped'])
    n_miss = len(merged['not_found'])
    n_total = n_disc + n_skip + n_miss

    print(f"\n=== RESULTS ===")
    print(f"  Discovered: {n_disc}")
    print(f"  Skipped: {n_skip}")
    print(f"  Not found: {n_miss}")
    print(f"  Total: {n_total}")

    # Evidence channel report
    channel_map = {
        'response_recognition': 'certainty', 'missing_piece_located': 'certainty',
        'transitivity': 'prediction', 'conjunction': 'prediction',
        'quantifier': 'prediction', 'it_follows': 'prediction',
        'causal_graph_reasoning': 'prediction', 'causal_chains': 'prediction',
        'prediction': 'prediction', 'self_model': 'prediction',
        'semantic_relation': 'component', 'naming': 'store',
        'referential_grounding': 'store', 'contradiction': 'store',
        'pattern_recognition': 'store', 'concept_formation': 'store',
    }
    channels = {}
    for receptor_id in merged['discovered']:
        ch = channel_map.get(receptor_id, 'independent')
        channels.setdefault(ch, []).append(receptor_id)

    print(f"\n  Per evidence channel:")
    for ch in sorted(channels.keys()):
        recs = channels[ch]
        print(f"    {ch:15s}: {len(recs)} ({', '.join(recs[:5])}{'...' if len(recs)>5 else ''})")

    # Per family
    families = {}
    for d in merged['details']:
        fam = d['family']
        families.setdefault(fam, {'found': 0, 'skipped': 0, 'missing': 0})
        if d.get('skipped'):
            families[fam]['skipped'] += 1
        elif d['discovered']:
            families[fam]['found'] += 1
        else:
            families[fam]['missing'] += 1

    print(f"\n  Per family (found/skipped/missing):")
    for f in sorted(families.keys()):
        i = families[f]
        total_f = i['found'] + i['skipped'] + i['missing']
        print(f"    {f:25s}: {i['found']:2d} / {i['skipped']:2d} / {i['missing']:2d}  ({total_f} total)")

    print(f"\n  Discovered: {sorted(merged['discovered'])}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        'discovered': sorted(merged['discovered']),
        'skipped': sorted(merged['skipped']),
        'not_found': sorted(merged['not_found']),
        'scores': merged['scores'],
        'null_thresholds': merged['null_thresholds'],
        'channels': {ch: sorted(recs) for ch, recs in channels.items()},
        'per_family': families,
        'config': {
            'oracle_episodes': oracle_episodes,
            'closed_loop_bootstrap': closed_loop_bootstrap,
            'closed_loop_online': closed_loop_online,
            'steps': steps,
            'null_shuffles': null_shuffles,
            'seed': seed,
        },
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'full_pipeline_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to data/full_pipeline_results.json")
    return output


if __name__ == '__main__':
    run_pipeline(
        oracle_episodes=30,
        closed_loop_bootstrap=50,
        closed_loop_online=100,
        steps=200,
        null_shuffles=10,
        seed=42,
    )

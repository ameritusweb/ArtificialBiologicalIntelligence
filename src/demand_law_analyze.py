"""Demand@law analyzer — same card as demand_law.py (fixed before
launch). Pools results/demand_law_r*.json.

1. DEMAND ALIGNMENT (the original card's floors and endpoint):
   pooled churn >= 10 AND pooled proposals >= 5; ALIGNED iff the
   top-churn slot's argmax-lift context == the court's most-proposed
   context; secondary Spearman over the shared context vocabulary.
2. HINGE-VS-STRUCTURE: contacted standing-K events (dose_sweep_analyze
   contact/reopen logic); reopens >= 1 -> hinge below law-structure;
   0 across >= 4 contacted -> above every built stratum.
3. Court rate at structure dose (vs parameter-layer 0.056/gen).
4. Splits, if any (CHURN_MIN=2 in-arm).
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

from dose_sweep_analyze import analyze_arm

HERE = os.path.dirname(os.path.abspath(__file__))
CHURN_FLOOR = 10
PROPOSAL_FLOOR = 5


def main():
    reps = [json.load(open(p)) for p in sorted(glob.glob(
        os.path.join(HERE, 'results', 'demand_law_r*.json')))]
    R = len(reps)
    print(f'=== DEMAND@LAW ANALYZER: {R} replicates ===')

    # ---- 1. demand alignment ----
    churn_total = sum(r['total_churn_events'] for r in reps)
    prop_counts = defaultdict(int)
    for r in reps:
        for c, n in r['proposal_counts'].items():
            prop_counts[c] += n
    prop_total = sum(prop_counts.values())
    if churn_total < CHURN_FLOOR or prop_total < PROPOSAL_FLOOR:
        alignment = {'verdict': f'UNTESTED (pooled churn={churn_total}, '
                                f'proposals={prop_total})'}
    else:
        by_slot = defaultdict(int)
        for r in reps:
            for e in r['churn_events']:
                by_slot[e['slot']] += e['n']
        top_slot = max(by_slot.items(), key=lambda kv: (kv[1], kv[0]))[0]
        ctx_windows = defaultdict(int)
        total_windows = 0
        for r in reps:
            for c, n in r['context_windows'].items():
                ctx_windows[c] += n
            total_windows += r['total_windows']
        top_events = [e for r in reps for e in r['churn_events']
                      if e['slot'] == top_slot]
        top_mass = sum(e['n'] for e in top_events)
        lifts = {}
        for c in ctx_windows:
            base = ctx_windows[c] / max(total_windows, 1)
            share = (sum(e['n'] for e in top_events
                         if c in e['contexts']) / max(top_mass, 1))
            if base > 0 and share > 0:
                lifts[c] = share / base
        churn_ctx = (max(lifts.items(), key=lambda kv: (kv[1], kv[0]))[0]
                     if lifts else None)
        lang_ctx = max(prop_counts.items(),
                       key=lambda kv: (kv[1], kv[0]))[0]
        all_mass = defaultdict(float)
        for r in reps:
            for e in r['churn_events']:
                for c in e['contexts']:
                    all_mass[c] += e['n']
        common = [c for c in ctx_windows
                  if c in prop_counts or c in all_mass]
        rho = None
        if len(common) >= 3:
            xs = [all_mass.get(c, 0.0) for c in common]
            ys = [prop_counts.get(c, 0) for c in common]
            rx = np.argsort(np.argsort(xs)).astype(float)
            ry = np.argsort(np.argsort(ys)).astype(float)
            if np.std(rx) > 0 and np.std(ry) > 0:
                rho = float(np.corrcoef(rx, ry)[0, 1])
        alignment = {
            'verdict': ('ALIGNED: one shared demand ledger at the law '
                        'layer' if churn_ctx == lang_ctx else
                        'MISALIGNED: the membranes name different '
                        'structures at the law layer'),
            'top_churn_slot': top_slot,
            'top_churn_context': churn_ctx,
            'top_proposed_context': lang_ctx,
            'lifts': {k: round(v, 3) for k, v in lifts.items()},
            'proposals': dict(prop_counts),
            'spearman_rho': rho}

    # ---- 2. hinge vs structure ----
    events = []
    for r in reps:
        events.extend(analyze_arm(r))
    contacted = [x for x in events if x['contact']]
    reopens = sum(1 for x in contacted if x['reopened'])
    if reopens >= 1:
        hinge = ('HINGE FALLS AT STRUCTURE: law-structure change '
                 'reopens the K in contact — the hierarchy has an '
                 'in-lineage falsification rung')
    elif len(contacted) >= 4:
        hinge = ('HINGE ABOVE ALL BUILT STRATA: the K survives '
                 'law-structure change in contact — in-lineage '
                 'absolute so far')
    else:
        hinge = f'UNTESTED ({len(contacted)} contacted events < 4)'

    p2_gens = sum(sum(1 for g in r['gen_records'] if g['phase'] == 'p2')
                  for r in reps)
    p2_props = sum(r['proposals_by_phase']['p2'] for r in reps)

    out = {'demand_alignment': alignment,
           'hinge_vs_structure': {
               'verdict': hinge,
               'standing_K_events': len(events),
               'contacted': len(contacted), 'reopens': reopens},
           'court_rate_structure': (round(p2_props / p2_gens, 3)
                                    if p2_gens else None),
           'court_rate_parameter_reference': 0.056,
           'pooled_churn': churn_total,
           'pooled_proposals': prop_total,
           'splits': [s for r in reps for s in r['splits']],
           'replicates': R}
    path = os.path.join(HERE, 'results', 'demand_law_pooled.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1, default=str)[:2500])
    print(f'saved {path}')


if __name__ == '__main__':
    main()

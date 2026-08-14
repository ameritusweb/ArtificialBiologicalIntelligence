"""Replay v5 analyzer — pooled verdicts (the card's billing half).

Fixed BEFORE launch, same card as replay_overnight_v5.py. Reads
results/replay_phase_v5_r*.json plus the closability probe receipts and
bills the pooled endpoints (C20 check 7a: genesis endpoints as rates):

1. CLOSURE PRESENCE RATE: replicates with >= 1 closure attempt / R.
   (Reported with the probe receipts — the screen said these lineages
   afford closure; the rate says how reliably.)
2. SHIFT-TEST 2x2, POOLED (contact-gated F23 form): every standing K
   at every world event (metronome tick or closure-paced shift) across
   all replicates; family invariance computed per replicate from its
   own fam_tracker (median variance split); cells pooled.
   SUPPORTED: all assessed cells majority-correct. PARTIAL: one.
   NOT SUPPORTED: none. UNTESTED: nothing assessable.
3. SPLIT-REDUCES-CHURN: per-replicate (the operator acts within a
   web); first billed split per replicate reported; any CONFLATION or
   NON-STATIONARITY verdict bills (they are per-web facts).
4. DEMAND ALIGNMENT, POOLED: churn events and proposals summed across
   replicates against the original floors (10 / 5); top-context
   comparison on pooled lifts vs pooled proposal counts.
5. COURT ENGAGEMENT: pooled proposals per generation vs v4's 1/36.
6. ORPHANING + DORMANCY: pooled counts (the metronome predicts near
   zero orphaning; dormancy events reported).
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

from receptor_eigen_coder import FAMILY_GROUPS

FAMILY_NAMES = [name for name, _ in FAMILY_GROUPS]
REOPEN_FAST = 2
SURVIVE_GENS = 4
MIN_ASSESS_FITS = 12
CHURN_FLOOR = 10
PROPOSAL_FLOOR = 5

HERE = os.path.dirname(os.path.abspath(__file__))


def invariant_families(fam_tracker):
    worlds = [(np.asarray(s) / c) for s, c in fam_tracker.values() if c > 0]
    if len(worlds) < 2:
        return set()
    means = np.stack(worlds)
    var = means.var(axis=0)
    order = np.argsort(var, kind='stable')
    return set(int(i) for i in order[:len(FAMILY_NAMES) // 2])


def shift_tests_for(rep):
    gens = len(rep['per_gen_closed'])
    closed_by_gen = [set(map(int, g)) for g in rep['per_gen_closed']]
    fits = [{int(k): v for k, v in snap.items()}
            for snap in rep['fit_snapshots']]
    inv = invariant_families(rep['fam_tracker'])
    fam_index = {n: i for i, n in enumerate(FAMILY_NAMES)}
    tests = []
    orphaned = 0
    for ev in rep['world_events']:
        sg = ev['gen']
        horizon = min(sg + SURVIVE_GENS, gens - 1)
        for K in ev['standing']:
            sid = K['sid']
            contact = None
            for gg in range(sg + 1, horizon + 1):
                if fits[gg].get(sid, 0) > fits[gg - 1].get(sid, 0):
                    contact = gg
                    break
            reopen = None
            for gg in range(sg + 1, gens):
                if sid not in closed_by_gen[gg]:
                    reopen = gg
                    break
            fits_post = fits[horizon].get(sid, 0) - fits[sg].get(sid, 0)
            fi = fam_index.get(K['family'], -1)
            cls = ('invariant' if fi in inv else 'variant') \
                if fi >= 0 else 'unclassified'
            if contact is None:
                orphaned += 1
                continue
            if cls == 'variant':
                assessed = fits_post >= MIN_ASSESS_FITS
                passed = (assessed and reopen is not None
                          and reopen - contact <= REOPEN_FAST) \
                    if assessed else None
            elif cls == 'invariant':
                assessed = (fits_post >= MIN_ASSESS_FITS
                            and gens - 1 - sg >= SURVIVE_GENS)
                passed = (reopen is None or reopen - sg > SURVIVE_GENS) \
                    if assessed else None
            else:
                assessed, passed = False, None
            if assessed:
                tests.append({'class': cls, 'slot': K['name'],
                              'event_kind': ev['kind'],
                              'shift_gen': sg, 'passed': bool(passed)})
    return tests, orphaned


def main():
    reps = []
    for path in sorted(glob.glob(os.path.join(
            HERE, 'results', 'replay_phase_v5_r*.json'))):
        reps.append((path, json.load(open(path))))
    probes = {os.path.basename(p): json.load(open(p))
              for p in glob.glob(os.path.join(HERE, 'results',
                                              'closability_*.json'))}
    R = len(reps)
    print(f'=== v5 ANALYZER: {R} replicates ===')

    out = {'replicates': [p for p, _ in reps],
           'probe_receipts': {k: v['classification']
                              for k, v in probes.items()}}

    # 1. closure presence rate
    presence = sum(
        1 for _, r in reps
        if r['record']['generations'][-1]['closure_attempts'] >= 1)
    out['closure_presence_rate'] = f'{presence}/{R}'

    # 2. pooled shift-test
    all_tests, orphaned = [], 0
    for _, r in reps:
        t, o = shift_tests_for(r)
        all_tests.extend(t)
        orphaned += o
    cells = {c: [t for t in all_tests if t['class'] == c]
             for c in ('variant', 'invariant')}

    def cell_ok(c):
        return (sum(1 for t in c if t['passed']) > len(c) / 2) if c else None
    ok_v, ok_i = cell_ok(cells['variant']), cell_ok(cells['invariant'])
    if ok_v is None and ok_i is None:
        st = 'UNTESTED (nothing assessable across replicates)'
    else:
        n_cells = sum(1 for o in (ok_v, ok_i) if o is not None)
        wins = sum(1 for o in (ok_v, ok_i) if o)
        if wins == n_cells:
            st = ('SUPPORTED (assessed cells): closure survival tracks '
                  'content invariance')
        elif wins >= 1:
            st = 'PARTIAL: one cell behaves'
        else:
            st = 'NOT SUPPORTED: assessed cells majority-wrong'
    out['shift_test'] = {
        'verdict': st, 'orphaned': orphaned,
        'assessed': {c: [(t['slot'], t['event_kind'], t['passed'])
                         for t in cells[c]] for c in cells}}

    # 3. split-reduces-churn per replicate
    out['split_reduces_churn'] = [
        r['per_replicate_verdicts']['split_reduces_churn']
        for _, r in reps]

    # 4. pooled demand alignment
    churn_total = sum(r['total_churn_events'] for _, r in reps)
    prop_counts = defaultdict(int)
    for _, r in reps:
        for c, n in r['record']['proposal_counts'].items():
            prop_counts[c] += n
    prop_total = sum(prop_counts.values())
    if churn_total < CHURN_FLOOR or prop_total < PROPOSAL_FLOOR:
        out['demand_alignment'] = {
            'verdict': f'UNTESTED (pooled churn={churn_total}, '
                       f'proposals={prop_total})'}
    else:
        ctx_mass = defaultdict(float)
        ctx_windows = defaultdict(int)
        total_windows = 0
        for _, r in reps:
            for e in r['churn_events']:
                for c in e['contexts']:
                    ctx_mass[c] += e['n']
            for c, n in r.get('context_windows', {}).items():
                ctx_windows[c] += n
            total_windows += r.get('total_windows', 0)
        lifts = {}
        total_mass = sum(ctx_mass.values())
        for c, m in ctx_mass.items():
            base = ctx_windows.get(c, 0) / max(total_windows, 1)
            share = m / max(total_mass, 1)
            if base > 0:
                lifts[c] = share / base
        churn_ctx = (max(lifts.items(), key=lambda kv: (kv[1], kv[0]))[0]
                     if lifts else None)
        lang_ctx = max(prop_counts.items(),
                       key=lambda kv: (kv[1], kv[0]))[0]
        out['demand_alignment'] = {
            'verdict': ('ALIGNED: one shared demand ledger'
                        if churn_ctx == lang_ctx else
                        'MISALIGNED: the membranes name different '
                        'structures'),
            'top_churn_context': churn_ctx,
            'top_proposed_context': lang_ctx,
            'pooled_lifts': {k: round(v, 3) for k, v in lifts.items()},
            'pooled_proposals': dict(prop_counts)}

    # 5. court engagement
    gens_total = sum(len(r['record']['generations']) for _, r in reps)
    props_total = sum(r['court_proposals'] for _, r in reps)
    out['court_engagement'] = {
        'proposals_per_gen': round(props_total / max(gens_total, 1), 3),
        'pooled': props_total, 'gens': gens_total,
        'v4_reference': '1/36'}

    # 6. dormancy / orphaning
    out['dormancy_events'] = sum(r['dormancy_events'] for _, r in reps)

    path = os.path.join(HERE, 'results', 'replay_phase_v5_pooled.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: (v if not isinstance(v, dict) or 'verdict' in v
                          else '...')
                      for k, v in out.items()
                      if k in ('closure_presence_rate', 'shift_test',
                               'demand_alignment', 'court_engagement',
                               'dormancy_events')},
                     indent=1, default=str)[:2000])
    print(f'saved {path}')


if __name__ == '__main__':
    main()

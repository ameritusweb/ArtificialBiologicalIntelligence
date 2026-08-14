"""P105 — readout as edge-detector, THE OPERATING-BAND VENUE (card
locked at launch 2026-08-13; the successor F33 pre-registered; runs
overnight-class, R=2 replicates via CLI rep arg; pooled verdict by
p105_band_analyze.py).

VENUE (F37's consolidation): the law-structure lineage — the only
regime where standing content can be contradicted (F28: hinge reopens
~0.06/contacted event, churn 17/replicate; the epistemic-pathology
band). P1 persists to closure+2; the web is RENDERED at P2 start
(before the first dose); P2 = 16 gens with structure events at gens
0/4/8/12 (S=2 per lineage per event) supply the strain the oracle
needs.

RENDER (identical to the v1/v3 cards; pure read, hash-asserted):
  A pose-collisions (band-profile degeneracy), B low-margin closed
  assertions, C round-trip cosine < 0.98. Telemetry T = near_miss>0
  or fit_count==0. Novel N = (A|B|C) - T. Matched controls by
  fit_count (family preferred) from the unflagged pool, v2's
  exhaustion amendment standing (bill the matched subset).

ORACLE (per slot over P2): delta near_miss (primary), reopens,
negative-fit fraction. FLOORS (pooled across reps, analyzer):
oracle mass >= 10 strain events on paired slots (else VOID-BY-ORACLE
— and the budget probe's spectrum, run same day on this venue class,
is the cross-check); pairs >= 8 pooled; else UNTESTED.

VERDICT (pooled, fixed): SUPPORTED iff mean dNM(N) > mean dNM(ctl)
AND wins/(wins+losses) >= 0.7. NOT SUPPORTED: N <= controls with a
live oracle. PARTIAL between. Rides along, exported per gen: flow
counters (P86-v4-comparable), per-phase strain accounting (budget
metering), splits/churn (CHURN_MIN=2).

C20 (eight): 1 domain — standing lineage machinery. 2 endpoint
independence — render is a pure read (hash-asserted); doses write the
world, the oracle reads the web. 3 exogeneity — dose schedule
pre-registered. 4 pairing — within-web matched controls (v1's rule);
render precedes all doses. 5 phenomenon strength — F28's receipts
put strain at this venue at 4-17x the floors; the oracle floor is on
the card (F33's lesson). 6 sensitivity — dNM resolution 1 event;
F28-class runs put churn events O(10) per replicate. 7 genesis/rates
— strain endpoints pooled over 16 gens x 2 reps (rate design);
receipts distribution-bound to pair B. 8 population closure —
membership snapshotted at render; controls matched from the
render-time pool; floors from that census; oracle floored.

Usage: python p105_band.py <rep>
"""

import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from environment_descriptive import classify_behavior_from_features
from environment_lexical import (Lexicon, evolve_one_generation,
                                 ratify_pending, append_ledger)
from law_structure import mutate_law_structure
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, BOOT_SEED)
from replay_overnight_v3 import run_world_v3
from p105_experiment import (band_profile, roundtrip_cos,
                             MARGIN_FLOOR, COS_GATE)
from staged_fit_experiment import family_of

LINEAGE_SEEDS = ((97000, 4), (97001, 3))
P1_MAX = 14
P1_HOLD = 2
P2_GENS = 16
P2_EVENTS = (0, 4, 8, 12)
S = 2
RESULTS_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'p105_band_r%d.json')


def state_hash(web):
    h = hashlib.sha256()
    h.update(str(web._global_step).encode())
    for sid in sorted(web.slots):
        s = web.slots[sid]
        h.update(('%d:%d:%d:%d' % (sid, s.ledger.receipt_count,
                                   s.ledger.fit_count,
                                   s.ledger.near_miss_seen)).encode())
    return h.hexdigest()


def main(rep):
    t0 = time.time()
    ro.CHURN_MIN = 2
    print('=== P105@BAND rep=%d (render-then-strain, law-structure '
          'lineage) ===' % rep)

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id='P105B_%d' % rep)
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'ratified': []}
    epoch = 0
    law_log = []
    flow_per_gen = []

    def one_generation(g, phase):
        nonlocal pending_prev, lexicon
        gen0 = (len(record['composed']) + len(record['splits'])
                + len(record['scan_events']))
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             65000 + rep * 5000 + li * 1000 + g * 17,
                             lived_log, {}, (li, epoch))
            gen_windows.append(w)
        web.anneal_all(web._global_step)
        replay_scans(web, scan, g, record)

        def as_court(ws):
            return [(p, classify_behavior_from_features(
                f, tau_rest, tau_d)) for p, f in ws]
        if pending_prev:
            r_train = [w for ws in gen_windows[:-1]
                       for w in as_court(ws)][:2000]
            r_val = as_court(gen_windows[-1])
            lexicon, _ = ratify_pending(lexicon, pending_prev,
                                        r_train, r_val, ledger)
        train_c = [w for ws in gen_windows[:-1] for w in as_court(ws)]
        val_c = as_court(gen_windows[-1])
        lexicon, _, pending = evolve_one_generation(
            lexicon, train_c, val_c, ledger)
        pending_prev = pending
        st = web.get_stats()
        gen1 = (len(record['composed']) + len(record['splits'])
                + len(record['scan_events']))
        flow_per_gen.append({'gen': g, 'phase': phase,
                             'pending': len(pending),
                             'genesis': gen1 - gen0})
        el = (time.time() - t0) / 60
        print('PB r%d gen %d [%s] (%.1f min): closed=%d pending=%d'
              % (rep, g + 1, phase, el, st['closed'], len(pending)))

    def rebuild():
        nonlocal engine
        engine = build_engine(lived_log)
        web.rebase(engine.encoder)

    # ---------------- P1
    g = 0
    closure_gen = None
    while g < P1_MAX:
        one_generation(g, 'p1')
        closed = {sid for sid, s in web.slots.items()
                  if s.state == 'closed'}
        if closed and closure_gen is None:
            closure_gen = g
        if closure_gen is not None and g - closure_gen >= P1_HOLD:
            g += 1
            break
        rebuild()
        g += 1

    # ---------------- RENDER at P2 start (pure read)
    h0 = state_hash(web)
    active = {sid: s for sid, s in web.slots.items()
              if s.state in ('open', 'closed')}
    profiles = {sid: band_profile(s) for sid, s in active.items()}
    by_profile = {}
    for sid, p in profiles.items():
        by_profile.setdefault(p, []).append(sid)
    set_A = {sid for p, sids in by_profile.items()
             if p != 'nothing' and len(sids) >= 2 for sid in sids}
    set_B = set()
    for sid, s in active.items():
        if s.state == 'closed':
            th = np.sort(s.geometry.family_thresholds)[::-1]
            if len(th) >= 2 and float(th[0] - th[1]) < MARGIN_FLOOR:
                set_B.add(sid)
    set_C = {sid for sid, s in active.items()
             if roundtrip_cos(s) < COS_GATE}
    set_T = {sid for sid, s in active.items()
             if s.ledger.near_miss_seen > 0 or s.ledger.fit_count == 0}
    set_N = (set_A | set_B | set_C) - set_T
    assert state_hash(web) == h0, 'C20 check 2: render not pure'

    baseline = {sid: {'nm': active[sid].ledger.near_miss_seen,
                      'fits': active[sid].ledger.fit_count,
                      'reopens': active[sid].ledger.reopen_count,
                      'contacts': len([r for r in
                                       active[sid].ledger.receipts
                                       if r.kind == 'fit'])}
                for sid in active}
    pool = [sid for sid in active
            if sid not in (set_A | set_B | set_C) and sid not in set_T]
    pairs = []
    for sid in sorted(set_N):
        if not pool:
            break
        fam_n = family_of(active[sid])
        fc_n = baseline[sid]['fits']
        pool.sort(key=lambda x: (abs(baseline[x]['fits'] - fc_n),
                                 0 if family_of(active[x]) == fam_n
                                 else 1, x))
        pairs.append((sid, pool.pop(0)))
    print('  RENDER: A=%d B=%d C=%d T=%d N=%d pairs=%d'
          % (len(set_A), len(set_B), len(set_C), len(set_T),
             len(set_N), len(pairs)))

    # ---------------- P2 with structure events
    p2_start = g
    for k in range(P2_GENS):
        if k in P2_EVENTS:
            rng = np.random.RandomState(6300 + rep * 7919 + epoch * 97)
            for li in range(len(lineages)):
                for _ in range(S):
                    lineages[li], d = mutate_law_structure(
                        lineages[li], rng)
                    law_log.append({'gen': p2_start + k, 'law': d})
            epoch += 1
            print('  >>> STRUCTURE EVENT at gen %d' % (p2_start + k))
        rebuild()
        one_generation(p2_start + k, 'p2')

    # ---------------- oracle
    def strain(sid):
        s = web.slots.get(sid)
        if s is None or s.state not in ('open', 'closed'):
            return None
        b = baseline.get(sid)
        if b is None:
            return None
        fits = [r for r in s.ledger.receipts if r.kind == 'fit']
        post = fits[b['contacts']:]
        neg = (sum(1 for r in post if r.sign < 0) / len(post)
               if post else 0.0)
        return {'d_nm': s.ledger.near_miss_seen - b['nm'],
                'd_reopen': s.ledger.reopen_count - b['reopens'],
                'neg_frac': round(neg, 4)}

    rows = []
    for nsid, csid in pairs:
        sn, sc = strain(nsid), strain(csid)
        if sn is None or sc is None:
            continue
        rows.append({'novel_slot': nsid, 'control_slot': csid,
                     'novel': sn, 'control': sc,
                     'novel_classes': [k for k, ss in
                                       (('A', set_A), ('B', set_B),
                                        ('C', set_C)) if nsid in ss]})
    whole_web_strain = {str(sid): strain(sid) for sid in active
                       if strain(sid) is not None}

    out = {'replicate': rep, 'closure_gen_p1': closure_gen,
           'p2_start': p2_start,
           'sets': {'A': sorted(set_A), 'B': sorted(set_B),
                    'C': sorted(set_C), 'T': sorted(set_T),
                    'N': sorted(set_N)},
           'pairs': rows,
           'whole_web_strain': whole_web_strain,
           'flow_per_gen': flow_per_gen,
           'law_log': law_log,
           'splits': record['splits'],
           'total_churn_events': sum(e['n'] for e in
                                     scan.churn_events),
           'conservation': web.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = RESULTS_TPL % rep
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print('PB r%d done (%.1f min) -> %s'
          % (rep, out['elapsed_min'], path))


if __name__ == '__main__':
    main(int(sys.argv[1]))

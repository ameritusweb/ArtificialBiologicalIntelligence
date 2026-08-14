"""P86 v4 — SPENT-WORLD atrophy: the environmental etiology, no
lesions anywhere. (Card locked at launch 2026-08-13; F37 impl. 6
enacted; supersedes the freeze-based v1-v3 designs, whose receipts
stand as the discovery path.)

THE CLAIM (T158 third arrival, re-derived by F37): atrophy is a
diagnosis of the WORLD-ORGANISM PAIR, not an organismal lesion. An
INTACT organism in a SPENT world (contradiction budget exhausted)
shows FLOW decay — genesis, closure traffic, court proposals — with
STOCK intact, because no invoices arrive; and flow RESTARTS on
migration to a budget-positive world (the rehabilitation prediction:
the founding lived case changed its diet of worlds, not its
machinery).

DESIGN — two full-organism arms (CLI arg 'spent'|'dosed'), identical
machinery, same lineage class (pair B), same phase clock:
  P1  persist to closure + 2 (max 14) — identical in both arms.
  P2  12 gens. SPENT: persistent lineages, NO mutations — the world
      the organism has already learned, re-lived (F37: nothing in it
      can be wrong). DOSED: law-structure events at P2 gens 0/4/8
      (S=2 per lineage per event) — the operating band, budget
      positive by F28's receipts.
  P3  4 gens MIGRATION: BOTH arms receive structure events at P3
      gens 0/2. For the spent arm this is the rehab; for the dosed
      arm it is continuation (the control against 'P3 rates rose
      merely by time or ledger mass').
FLOW metrics per gen (event counts, gen-windowed): court pending;
scan genesis (composed + splits + individuations from the replay
record); closure-lifecycle traffic (closure attempts delta + reopens
delta + dormancy transitions). STOCK metrics: assertable count,
closed count, conservation, whole-web fresh service.

VERDICTS (fixed; pooled per phase, both arms compared at matched
phase): SUPPORTED iff (i) FLOW STARVATION: dosed-arm P2 flow >= 3x
spent-arm P2 flow (pooled events/gen) with the dosed floor met;
(ii) STOCK PARITY: spent-arm assertable at P2 end >= dosed-arm
assertable - 1 (stock holds without contradiction — the spent world
cannot evict what it cannot contradict); (iii) RESTART: spent-arm P3
flow >= 3x its own P2 rate. NOT SUPPORTED (endogeny): spent P2 flow
within 1.5x of dosed (invoices are organism-generated; the world's
budget is not the driver — T158's flow story takes a wound).
NOT SUPPORTED (irreversibility): (i) and (ii) hold but spent P3 flow
< 1.5x its P2 rate (migration does not restart flow at this horizon).
VOID-BY-BAND: dosed-arm P2 flow < 4 pooled events (the operating
band failed to manifest — cross-check against the budget probe's
spectrum, which runs alongside). UNTESTED: either arm fails P1
closure by gen 14 (no standing stock to hold).

C20 (eight): 1 domain — standing lineage machinery, in-dist; C15
satisfied (full organism, all systems, no lesions). 2 endpoint
independence — the treatment writes the WORLD (mutation schedule);
endpoints read organism-side traffic; no channel from the schedule
into the counters except through the organism's life. 3 exogeneity —
mutation schedules pre-registered by arm and phase. 4 pairing — arms
share seeds/model/boot; they diverge only through their worlds'
schedules (the treatment); P1 is schedule-identical. 5 phenomenon
strength — dosed-arm floor above; F28 receipts (churn 17, court
0.222/gen at structure dose). 6 sensitivity — flow counters move by
integers; F28 magnitudes sit 4-17x above the floors. 7 genesis/
rates — flow IS a rate endpoint, pooled per phase over 12/4 gens;
receipts distribution-bound to pair B. 8 population closure — flow
counters are event counts (no membership); stock parity read on the
standing set grown in the schedule-identical P1.
"""

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

LINEAGE_SEEDS = ((97000, 4), (97001, 3))
P1_MAX = 14
P1_HOLD = 2
P2_GENS = 12
P3_GENS = 4
P2_EVENTS = (0, 4, 8)
P3_EVENTS = (0, 2)
S = 2
RESULTS_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'p86_v4_%s.json')


def main(arm, rep=0):
    assert arm in ('spent', 'dosed')
    t0 = time.time()
    ro.CHURN_MIN = 2
    print('=== P86 v4 arm=%s rep=%d (spent-world atrophy) ==='
          % (arm, rep))

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id='P86V4_%s' % arm)
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
    stock_per_gen = []

    def genesis_count():
        return (len(record['composed']) + len(record['splits'])
                + len(record['scan_events']))

    def lifecycle_state():
        return {'reopens': sum(s.ledger.reopen_count
                               for s in web.slots.values()),
                'closed_ever': sum(1 for s in web.slots.values()
                                   if s.state == 'closed')
                + sum(s.ledger.reopen_count
                      for s in web.slots.values()),
                'dormant': sum(1 for s in web.slots.values()
                               if s.dormant)}

    def one_generation(g, phase):
        nonlocal pending_prev, lexicon
        gen0 = genesis_count()
        lc0 = lifecycle_state()
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             55000 + rep * 20000 + li * 1000 + g * 17,
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

        lc1 = lifecycle_state()
        st = web.get_stats()
        flow = {'gen': g, 'phase': phase,
                'court_pending': len(pending),
                'genesis': genesis_count() - gen0,
                'lifecycle': (abs(lc1['reopens'] - lc0['reopens'])
                              + abs(lc1['closed_ever']
                                    - lc0['closed_ever'])
                              + abs(lc1['dormant'] - lc0['dormant']))}
        flow['total'] = (flow['court_pending'] + flow['genesis']
                         + flow['lifecycle'])
        flow_per_gen.append(flow)
        stock_per_gen.append({'gen': g, 'phase': phase,
                              'closed': st['closed'],
                              'assertable': st['assertable'],
                              'dormant': st['dormant']})
        el = (time.time() - t0) / 60
        print('P86v4[%s] gen %d [%s] (%.1f min): flow=%d '
              '(court=%d genesis=%d lifecycle=%d) closed=%d'
              % (arm, g + 1, phase, el, flow['total'],
                 flow['court_pending'], flow['genesis'],
                 flow['lifecycle'], st['closed']))

    def rebuild():
        nonlocal engine
        engine = build_engine(lived_log)
        web.rebase(engine.encoder)

    def apply_dose(g):
        nonlocal epoch
        rng = np.random.RandomState(6200 + rep * 31 + epoch * 97)
        for li in range(len(lineages)):
            for _ in range(S):
                lineages[li], d = mutate_law_structure(lineages[li],
                                                       rng)
                law_log.append({'gen': g, 'lineage': li, 'law': d})
        epoch += 1
        print('  >>> STRUCTURE EVENT at gen %d' % g)

    # P1 — identical schedule both arms
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
    if closure_gen is None:
        print('P86 v4 UNTESTED: no closure by gen %d' % P1_MAX)

    # P2 — the treatment
    p2_start = g
    for k in range(P2_GENS):
        if arm == 'dosed' and k in P2_EVENTS:
            apply_dose(p2_start + k)
        rebuild()
        one_generation(p2_start + k, 'p2')

    # P3 — migration (both arms dosed)
    p3_start = p2_start + P2_GENS
    for k in range(P3_GENS):
        if k in P3_EVENTS:
            apply_dose(p3_start + k)
        rebuild()
        one_generation(p3_start + k, 'p3')

    def pooled(phase):
        rows = [f for f in flow_per_gen if f['phase'] == phase]
        return (sum(f['total'] for f in rows) / len(rows)
                if rows else None)

    out = {'arm': arm, 'closure_gen_p1': closure_gen,
           'p2_start': p2_start, 'p3_start': p3_start,
           'flow_per_gen': flow_per_gen,
           'stock_per_gen': stock_per_gen,
           'flow_pooled': {ph: pooled(ph) for ph in
                           ('p1', 'p2', 'p3')},
           'law_log': law_log,
           'conservation': web.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = (RESULTS_TPL % arm if rep == 0
            else RESULTS_TPL % ('%s_r%d' % (arm, rep)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print('P86v4[%s] done: pooled flow %s (%.1f min) -> %s'
          % (arm, out['flow_pooled'], out['elapsed_min'], path))


if __name__ == '__main__':
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)

"""BUDGET PROBE — the contradiction-budget spectrum as a standing
instrument (F37 impl. 4: F26's dose sweep was budget spectroscopy
without the name). Commissioning card, locked at launch 2026-08-13.

WHAT IT MEASURES: a venue's remaining capacity to contradict standing
content, per mutation stratum. Budget = strain yielded per stratum:
reopens on standing Ks, near-miss and negative-fit mass on the whole
web, court proposals. A venue with zero budget at every stratum
cannot host a pathology card, a learning card, or a strain oracle
(F33/F37's wall, now measurable before booking).

DESIGN (escalating-dose, one run, demand_law lineage class, pair B):
  P1: persist to closure + 2 (max 14 gens) — grow the standing set.
  Then four stratum phases x 2 gens each, doses applied at phase
  start, mutations ACCUMULATE across phases (escalating; honest
  label: the spectrum is escalating-dose, ordering-valid because
  the shallow strata measured first are the ones prior receipts put
  at zero):
    FURNITURE  environment_living.mutate x4 per lineage
    PARAMS     law_mutations.mutate_law x2 per lineage
    STRUCTURE  law_structure.mutate_law_structure x2 per lineage
    RESEED     fresh corpora (same tiers, new seeds)
  Per phase, measured on populations FIXED at phase start (check 8):
    standing-K rows: contacted fits, reopens, dormancy transitions
    (n stated — the standing set is small by nature, 1-3 Ks);
    whole-web rows (n=33+): near-miss delta, negative-fit fraction,
    fresh-fit tightness; court: proposals per gen.

VALIDATION CLAUSE (the instrument's calibration case, P92-style —
it must first detect the known instance): pair B's spectrum from
F26/F27/F28 receipts is furniture ~ 0, params ~ 0 (0 reopens, court
speaks), structure > 0 (hinge reopens ~0.06/contact, churn), reseed
-> orphaning (contact loss, not contradiction). INSTRUMENT PASSES iff
structure-phase strain (reopens + web strain mass) strictly exceeds
furniture-phase strain AND params-phase standing reopens = 0 AND
reseed-phase standing contact collapses (contacted fits < 25% of
structure phase). FAILS otherwise -> instrument suspect, do not use
on new venues until diagnosed.

C20 (eight, commissioning form): 1 domain — the standing lineage
machinery, in-dist. 2 endpoint independence — doses write the world;
endpoints read web strain and court traffic. 3 exogeneity — doses at
pre-registered phase starts. 4 pairing — n/a (single-arm instrument;
its output is a measurement, not a comparison verdict). 5 phenomenon
strength — this instrument IS check 5's meter; its own validation is
the known pair-B spectrum. 6 sensitivity — strata previously moved
these endpoints at these doses (F26-F28 receipts). 7 genesis/rates —
standing-K rows are rate-reported with n; single-run commissioning
accepted for an instrument (verdict-grade venue budgets should pool
2+ probe runs). 8 population closure — all measured populations
snapshotted at phase starts; floors: whole-web rows n >= 30; standing
rows reported at any n with n printed.
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
from environment_living import mutate
from law_mutations import mutate_law
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
PHASE_GENS = 2
STRATA = ('furniture', 'params', 'structure', 'reseed')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'budget_probe_pairB.json')


def main(rep=0):
    t0 = time.time()
    ro.CHURN_MIN = 2
    print('=== BUDGET PROBE (pair B commissioning, rep %d) ===' % rep)

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id='BUDGET_%d' % rep)
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
    pending_per_gen = []

    def one_generation(g):
        nonlocal pending_prev, lexicon
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             45000 + rep * 5000 + li * 1000 + g * 17,
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
        pending_per_gen.append(len(pending))
        st = web.get_stats()
        el = (time.time() - t0) / 60
        print('BP gen %d (%.1f min): closed=%d assertable=%d '
              'pending=%d' % (g + 1, el, st['closed'],
                              st['assertable'], len(pending)))

    def rebuild():
        nonlocal engine
        engine = build_engine(lived_log)
        web.rebase(engine.encoder)

    def web_snapshot():
        snap = {}
        for sid, s in web.slots.items():
            if s.state not in ('open', 'closed'):
                continue
            fits = [r for r in s.ledger.receipts if r.kind == 'fit']
            snap[sid] = {'state': s.state,
                         'nm': s.ledger.near_miss_seen,
                         'fits': s.ledger.fit_count,
                         'contacts': len(fits),
                         'reopens': s.ledger.reopen_count,
                         'dormant': s.dormant}
        return snap

    def phase_delta(pre, post, standing):
        web_rows = {'d_nm': 0, 'd_neg_contacts': 0, 'n': 0}
        st_rows = {'contacted_fits': 0, 'reopens': 0,
                   'went_dormant': 0, 'n': len(standing)}
        for sid, b in pre.items():
            p = post.get(sid)
            if p is None:
                continue
            web_rows['n'] += 1
            web_rows['d_nm'] += p['nm'] - b['nm']
            web_rows['d_neg_contacts'] += \
                (p['contacts'] - b['contacts']) - (p['fits'] - b['fits'])
            if sid in standing:
                st_rows['contacted_fits'] += p['fits'] - b['fits']
                st_rows['reopens'] += p['reopens'] - b['reopens']
                if p['dormant'] and not b['dormant']:
                    st_rows['went_dormant'] += 1
        return {'web': web_rows, 'standing': st_rows}

    # ---------------- P1: grow the standing set
    g = 0
    closure_gen = None
    while g < P1_MAX:
        one_generation(g)
        closed = {sid for sid, s in web.slots.items()
                  if s.state == 'closed'}
        if closed and closure_gen is None:
            closure_gen = g
        if closure_gen is not None and g - closure_gen >= P1_HOLD:
            g += 1
            break
        rebuild()
        g += 1
    standing = sorted(sid for sid, s in web.slots.items()
                      if s.state == 'closed')
    print('  P1 done: closure_gen=%s standing=%s' % (closure_gen,
                                                     standing))

    # ---------------- stratum phases
    spectrum = {}
    rng = np.random.RandomState(6100 + rep * 7919)
    for si, stratum in enumerate(STRATA):
        if stratum == 'furniture':
            for li in range(len(lineages)):
                for _ in range(4):
                    lineages[li], d = mutate(lineages[li], rng)
        elif stratum == 'params':
            for li in range(len(lineages)):
                for _ in range(2):
                    lineages[li], d = mutate_law(lineages[li], rng)
        elif stratum == 'structure':
            for li in range(len(lineages)):
                for _ in range(2):
                    lineages[li], d = mutate_law_structure(
                        lineages[li], rng)
        else:
            lineages = [describe(TieredEnvironment(
                seed=97100 + rep * 10 + i, tier=t))
                for i, (s, t) in enumerate(LINEAGE_SEEDS)]
        epoch += 1
        pre = web_snapshot()
        pend0 = len(pending_per_gen)
        for k in range(PHASE_GENS):
            rebuild()
            one_generation(g)
            g += 1
        post = web_snapshot()
        d = phase_delta(pre, post, set(standing))
        d['court_pending_per_gen'] = (
            sum(pending_per_gen[pend0:]) / PHASE_GENS)
        spectrum[stratum] = d
        print('  %s: %s' % (stratum, d))

    # ---------------- validation clause
    def strain_mass(s):
        return (s['web']['d_nm'] + s['web']['d_neg_contacts']
                + 10 * s['standing']['reopens'])
    fu, pa, st_, re_ = (spectrum['furniture'], spectrum['params'],
                        spectrum['structure'], spectrum['reseed'])
    checks = {
        'structure_exceeds_furniture':
            strain_mass(st_) > strain_mass(fu),
        'params_standing_reopens_zero':
            pa['standing']['reopens'] == 0,
        'reseed_contact_collapse':
            (st_['standing']['contacted_fits'] == 0
             or re_['standing']['contacted_fits']
             < 0.25 * st_['standing']['contacted_fits']),
        'web_population_floor': all(
            s['web']['n'] >= 30 for s in spectrum.values()),
    }
    passed = all(checks.values())
    verdict = ('INSTRUMENT VALIDATED: pair-B spectrum reproduced (%s)'
               % checks if passed else
               'INSTRUMENT SUSPECT: %s' % checks)
    print('\nBUDGET PROBE: %s' % verdict)

    out = {'closure_gen_p1': closure_gen, 'standing': standing,
           'spectrum': spectrum, 'validation': checks,
           'passed': passed, 'verdict': verdict,
           'conservation': web.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

"""XO-0 — the experiment organ's NOMINATION instrument (read-only;
experiment_organ_requirements.md phase 1; card locked at launch
2026-08-13).

WHAT IT IS: no effector change, no new receipts — a lens over what
the organism already records. One standard lineage phase (pair B,
full protocol, 6 gens) populates the web; nomination then reads the
LEDGER: for every active slot pair, co-fire mass (same episode,
|dt| <= W from fit receipts), directional lag (median signed dt —
asymmetry suggests direction, symmetry suggests common drive), and
the CONFOUND SET (third slots co-firing with both above floor).
Candidates ranked by co-fit mass x |lag asymmetry|. Deliverable: the
top candidate list printed beside the corpus's own TriggerEffect
registry (ground truth) for eye validation — no automated match
claimed at XO-0 (slot-to-manifestation correspondence is exactly
what later phases earn).

KILL CONDITION (pre-registered): if the top decile of candidates
holds < 40% of total co-fit mass (nomination not sparse — everything
co-fits everything), the suspicion receptor needs an info-priced
threshold BEFORE any build proceeds. Reported either way.

EXTRA (F48 impl. 8's handoff): closure-covariate table — for slots
that closed vs matched never-closed slots, print candidate
covariates (fit_count, family, threshold margin, near-miss, balance)
as nomination input for the closure-rate mystery. Descriptive only.

C20 (light — instrument run, no verdict): 1 standing venue; 2 pure
read post-run (hash-asserted); 6 co-fire window W=5 steps stated;
8 population = active slots at read time, all classes reported with
class labels (inherited/composed marked). Expected information
(F48 impl. 6, the new line): if sparse -> the organ's first work
list exists; if not sparse -> the threshold requirement is learned
BEFORE XO-1 spends effectors. Both outcomes advance the build."""

import json
import os
import time
from collections import defaultdict

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from environment_descriptive import classify_behavior_from_features
from environment_lexical import (Lexicon, evolve_one_generation,
                                 ratify_pending)
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from p105_experiment import state_hash
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, BOOT_SEED)
from replay_overnight_v3 import run_world_v3

B_SEEDS = ((97000, 4), (97001, 3))
GENS = 6
W = 5
CONFOUND_FLOOR_FRAC = 0.5
TOP_K = 12
SPARSITY_GATE = 0.40
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'xo0_nomination.json')


def main():
    t0 = time.time()
    ro.CHURN_MIN = 2
    print('=== XO-0: nomination instrument (read-only) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    tau_rest, tau_d = calibrate_taus(model)
    engine = build_engine()
    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(),
                        debug_level=0, ledger_id='XO0')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in B_SEEDS]
    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'ratified': []}
    for g in range(GENS):
        if g > 0:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             82000 + li * 1000 + g * 17,
                             lived_log, {}, (li, 0))
            gen_windows.append(w)
        web.anneal_all(web._global_step)
        replay_scans(web, scan, g, record)

        def as_court(ws):
            return [(p, classify_behavior_from_features(
                f, tau_rest, tau_d)) for p, f in ws]
        if pending_prev:
            r_train = [x for ws in gen_windows[:-1]
                       for x in as_court(ws)][:2000]
            lexicon, _ = ratify_pending(lexicon, pending_prev,
                                        r_train,
                                        as_court(gen_windows[-1]),
                                        ledger)
        lexicon, _, pending_prev = evolve_one_generation(
            lexicon,
            [x for ws in gen_windows[:-1] for x in as_court(ws)],
            as_court(gen_windows[-1]), ledger)
        print('  gen %d done (%.1f min)' % (g + 1,
                                            (time.time() - t0) / 60))

    # ---------------- NOMINATION (pure read)
    h0 = state_hash(web)
    active = {sid: s for sid, s in web.slots.items()
              if s.state in ('open', 'closed')}
    fires = {}
    for sid, s in active.items():
        fires[sid] = [(r.episode, r.time_step)
                      for r in s.ledger.receipts
                      if r.kind == 'fit' and r.sign > 0]
    by_ep = defaultdict(lambda: defaultdict(list))
    for sid, evs in fires.items():
        for ep, t in evs:
            by_ep[ep][sid].append(t)
    pair_stats = defaultdict(lambda: [0, []])
    for ep, slot_times in by_ep.items():
        sids = sorted(slot_times)
        for i, a in enumerate(sids):
            ta = np.asarray(slot_times[a])
            for b in sids[i + 1:]:
                tb = np.asarray(slot_times[b])
                d = ta[:, None] - tb[None, :]
                close = np.abs(d) <= W
                n = int(close.sum())
                if n:
                    pair_stats[(a, b)][0] += n
                    pair_stats[(a, b)][1].extend(
                        d[close].tolist()[:50])
    # info-priced weighting (XO-0b, learned from XO-0's first run:
    # the vacuous hub slot 9 polluted raw co-fit — the rent economy's
    # info law applied to nomination: weight a pair by both members'
    # selectivity, (1 - fire_a)(1 - fire_b), terminal fire rates as
    # the approximation, stated).
    fire = {sid: float(s.ledger.fire_rate)
            for sid, s in active.items()}
    cands = []
    for (a, b), (n, lags) in pair_stats.items():
        if n < 20:
            continue
        med = float(np.median(lags))
        asym = abs(med)
        w = (1.0 - fire[a]) * (1.0 - fire[b])
        cands.append({'a': a, 'b': b, 'co_fit': n,
                      'weighted_co_fit': round(n * w, 1),
                      'median_lag': med,
                      'score': n * w * (asym + 0.1)})
    cands.sort(key=lambda c: -c['score'])
    total_mass = sum(c['weighted_co_fit'] for c in cands) or 1
    top_dec = cands[:max(1, len(cands) // 10)]
    dec_frac = sum(c['weighted_co_fit'] for c in top_dec) / total_mass
    raw_total = sum(c['co_fit'] for c in cands) or 1
    raw_sorted = sorted(cands, key=lambda c: -c['co_fit'])
    raw_frac = sum(c['co_fit'] for c in
                   raw_sorted[:max(1, len(cands) // 10)]) / raw_total
    for c in cands[:TOP_K]:
        both = c['co_fit']
        conf = []
        for x in active:
            if x in (c['a'], c['b']):
                continue
            na = pair_stats.get(tuple(sorted((x, c['a']))), [0])[0]
            nb = pair_stats.get(tuple(sorted((x, c['b']))), [0])[0]
            if min(na, nb) >= CONFOUND_FLOOR_FRAC * both:
                conf.append(x)
        c['confounds'] = conf
        c['classes'] = ['inh' if s <= 32 else 'comp'
                        for s in (c['a'], c['b'])]
    assert state_hash(web) == h0, 'XO-0 not a pure read'

    print('\n--- top nominations (info-priced, of %d candidates) ---'
          % len(cands))
    for c in cands[:TOP_K]:
        print('  %s-%s [%s] co_fit=%d weighted=%.0f lag=%+.1f '
              'confounds=%s'
              % (c['a'], c['b'], '/'.join(c['classes']), c['co_fit'],
                 c['weighted_co_fit'], c['median_lag'],
                 c.get('confounds', [])))
    print('top-decile mass fraction: weighted %.2f vs raw %.2f '
          '(gate %.2f) -> %s'
          % (dec_frac, raw_frac, SPARSITY_GATE,
             'SPARSE: work list exists' if dec_frac >= SPARSITY_GATE
             else 'NOT SPARSE even info-priced: deeper threshold '
                  'work before XO-1'))

    print('\n--- ground truth (eye validation): registry entries ---')
    truths = []
    for li, corpus in enumerate(lineages):
        for m in corpus:
            r = repr(m)
            if 'trig' in r.lower() or 'effect' in r.lower():
                truths.append('L%d: %s' % (li, r[:110]))
    for t_ in truths[:12]:
        print('  %s' % t_)
    if not truths:
        print('  (no trigger-like entries in corpus repr — registry '
              'API mapping deferred to XO-1)')

    closed = [s for s in active.values() if s.state == 'closed']
    never = [s for s in active.values()
             if s.state == 'open' and s.ledger.fit_count > 0]
    print('\n--- closure-covariate table (F48 impl. 8 handoff) ---')
    for label, group in (('closed', closed), ('open', never[:8])):
        for s in group:
            th = np.sort(s.geometry.family_thresholds)[::-1]
            margin = float(th[0] - th[1]) if len(th) > 1 else 0.0
            print('  %s slot=%s fam_fit=%d nm=%d margin=%.3f '
                  'balance=%.1f fire=%.2f'
                  % (label, s.slot_id, s.ledger.fit_count,
                     s.ledger.near_miss_seen, margin,
                     float(s.ledger.rent_balance),
                     float(s.ledger.fire_rate)))

    out = {'candidates': cands[:50],
           'n_candidates': len(cands),
           'top_decile_mass_frac': round(dec_frac, 3),
           'sparse': dec_frac >= SPARSITY_GATE,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('\nsaved %s' % RESULTS)


if __name__ == '__main__':
    main()

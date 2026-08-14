"""ACCELERATION v5 — full protocol + pair B + STABLE LAWS (card
locked at launch 2026-08-13; the decomposition's final candidate cell).

DIAGNOSIS (C21, fourth cell): v4 showed pair B + full protocol still
nearly silent (1 closure). The remaining diff vs p86_v4: DOSE
TIMING. p86_v4's closures all occurred in P1 — UNDOSED, stable laws —
and F28's 0.222/gen was the P2 COURT rate, not a closure rate (a
conflation in v2's card, owned here). F20/F26 predict exactly this:
closure demands stability; structure events from gen 0 keep the
corridor churning. v5 removes all doses: 12 stable gens, closure
race. This is also the cleanest form of the yield question — does
capital speed the settlement of a new stable world — and solvent
survival is unconfounded by strain. If closures appear at p86_v4's
receipted P1 rates (1-4 gens), the silence decomposes as
DOSE-TIMING and the comparison finally bills; if not, the harness
diff hunt continues in code with the venue fully exonerated.

Original v3 header follows.

DIAGNOSIS (C21 in action): v1 (plain, 8 gens) and v2 (dosed, 12 gens)
produced 1 and 0 inherited closures on a STRIPPED loop (run_world_v3 +
anneal only). F28's closure receipt (0.222/gen) was earned under the
FULL protocol: CHURN_MIN=2, replay_scans after anneal, and the court
(classify -> ratify -> evolve). v3 runs the full protocol — p86_v4's
one_generation, verbatim in structure — for both arms. If closures
appear, the missing factor was protocol richness (billed as such);
if not, the venue claim itself needs revision.

ARMS: WARM = nursery pickle (naked crossing per F47). COLD = fresh
boot (p86_v4's exact web construction). Same model, worlds
(97100 t4 / 97101 t3), dose schedule (gens 0/4/8, S=2), 12 gens.

ENDPOINTS: PRIMARY (class-closed): inherited closures + first-closure
gen. SECONDARY: world-sensitive flow rates per arm (court_pending +
lifecycle — F38's decomposition discipline), solvent survival under
the full protocol (scans can now COMPOSE new slots in both arms —
new composed are NOT in any endpoint; inspector inventory below).

VERDICTS: UNTESTED if combined inherited closures < 3. SUPPORTED iff
WARM > COLD on closures AND ties-or-beats latency. NOT SUPPORTED iff
COLD ties-or-beats both. PARTIAL between.

C20: 1 — p86_v4's harness verbatim (receipted venue). 2 — closures
via corridor gate; arms differ only in starting web. 3 — shared dose
schedule/seeds. 4 — same process, same boot model. 5 — p86_v4 closed
in P1 within ~4 gens on this exact protocol. 6 — 1 closure / 1 gen.
7 — single pair; replicate on support. 8 (inspector) — inherited
exempt both arms; solvent composed evictable warm-only, excluded
from primary; scan-composed slots (both arms) excluded from all
endpoints; warm's aged EMAs = capital (stated); court state
per-arm-fresh (lexicon, ledger, pending)."""

import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from environment_descriptive import classify_behavior_from_features
from environment_lexical import (Lexicon, evolve_one_generation,
                                 ratify_pending)
from law_structure import mutate_law_structure
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, BOOT_SEED)
from replay_overnight_v3 import run_world_v3

B_SEEDS = ((97000, 4), (97001, 3))
GENS = 12
EVENTS = ()
S = 2
INHERITED_MAX = 32
CLOSURE_FLOOR = 3
WEB_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'nursery_web_s98300.pkl')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'acceleration_v5.json')


def run_arm(name, web, model, tau_rest, tau_d, t0):
    print('--- arm %s ---' % name)
    engine = build_engine()
    lived_log = []
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in B_SEEDS]
    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'ratified': []}
    epoch = 0
    solvent0 = sorted(sid for sid, s in web.slots.items()
                      if sid > INHERITED_MAX
                      and s.state in ('open', 'closed'))
    closed_before = {sid for sid, s in web.slots.items()
                     if sid <= INHERITED_MAX and s.state == 'closed'}
    closures, surv_curve, flow_rows = [], [], []

    def lifecycle_state():
        return {'reopens': sum(s.ledger.reopen_count
                               for s in web.slots.values()),
                'closed_ever': sum(1 for s in web.slots.values()
                                   if s.state == 'closed')
                + sum(s.ledger.reopen_count
                      for s in web.slots.values()),
                'dormant': sum(1 for s in web.slots.values()
                               if s.dormant)}

    for g in range(GENS):
        if g in EVENTS:
            rng = np.random.RandomState(6600 + epoch * 97)
            for li in range(len(lineages)):
                for _ in range(S):
                    lineages[li], d = mutate_law_structure(
                        lineages[li], rng)
            epoch += 1
        if g > 0:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)
        lc0 = lifecycle_state()
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             81000 + li * 1000 + g * 17,
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
        train_c = [w for ws in gen_windows[:-1]
                   for w in as_court(ws)]
        val_c = as_court(gen_windows[-1])
        lexicon, _, pending = evolve_one_generation(
            lexicon, train_c, val_c, ledger)
        pending_prev = pending

        lc1 = lifecycle_state()
        flow_rows.append({'gen': g,
                          'court_pending': len(pending),
                          'lifecycle':
                          abs(lc1['reopens'] - lc0['reopens'])
                          + abs(lc1['closed_ever']
                                - lc0['closed_ever'])
                          + abs(lc1['dormant'] - lc0['dormant'])})
        for sid, s in web.slots.items():
            if sid <= INHERITED_MAX and s.state == 'closed' \
                    and sid not in closed_before:
                closures.append({'slot': sid, 'gen': g + 1})
                closed_before.add(sid)
        alive = sum(1 for sid in solvent0
                    if web.slots.get(sid) is not None
                    and web.slots[sid].state in ('open', 'closed'))
        surv_curve.append(alive)
        print('  gen %d (%.1f min): closures=%d solvent=%d/%d '
              'court=%d lifecycle=%d'
              % (g + 1, (time.time() - t0) / 60, len(closures),
                 alive, len(solvent0), flow_rows[-1]['court_pending'],
                 flow_rows[-1]['lifecycle']))
    ws_events = sum(f['court_pending'] + f['lifecycle']
                    for f in flow_rows)
    return {'closures': closures,
            'first_closure_gen': (closures[0]['gen'] if closures
                                  else None),
            'world_sensitive_events': ws_events,
            'solvent_n0': len(solvent0),
            'solvent_curve': surv_curve,
            'flow_per_gen': flow_rows}


def main():
    t0 = time.time()
    ro.CHURN_MIN = 2
    print('=== ACCELERATION v5: full replay protocol ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    tau_rest, tau_d = calibrate_taus(model)

    with open(WEB_PKL, 'rb') as f:
        warm_web = pickle.load(f)
    warm = run_arm('WARM', warm_web, model, tau_rest, tau_d, t0)

    cold_web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(),
                             debug_level=0, ledger_id='ACC5_COLD')
    cold_web.populate_from_families()
    cold = run_arm('COLD', cold_web, model, tau_rest, tau_d, t0)

    wc, cc = len(warm['closures']), len(cold['closures'])
    wl, cl = warm['first_closure_gen'], cold['first_closure_gen']
    if wc + cc < CLOSURE_FLOOR:
        verdict = ('UNTESTED (combined inherited closures %d < %d '
                   'even under the full protocol — venue claim '
                   'needs revision)' % (wc + cc, CLOSURE_FLOOR))
    elif wc > cc and (cl is None or (wl is not None and wl <= cl)):
        verdict = ('ACCELERATION SUPPORTED: warm %d closures (first '
                   'gen %s) vs cold %d (first gen %s)'
                   % (wc, wl, cc, cl))
    elif cc >= wc and (wl is None or (cl is not None and cl <= wl)):
        verdict = ('NOT SUPPORTED: cold %d (gen %s) ties/beats warm '
                   '%d (gen %s) — capital confers no closure '
                   'advantage' % (cc, cl, wc, wl))
    else:
        verdict = ('PARTIAL: warm %d/g%s vs cold %d/g%s'
                   % (wc, wl, cc, cl))
    print('\nACCELERATION v5 VERDICT: %s' % verdict)
    print('world-sensitive events: warm=%d cold=%d'
          % (warm['world_sensitive_events'],
             cold['world_sensitive_events']))
    out = {'warm': warm, 'cold': cold, 'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

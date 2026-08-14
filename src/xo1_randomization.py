"""XO-1 — the experiment organ's RANDOMIZATION phase (first effector
work; experiment_organ_requirements.md phase 2; card locked at launch
2026-08-13).

THE MOVE: scheduled action bursts — 3 forced-random steps every 37
(prime; state-independent by construction) — injected via a model
WRAPPER for run_world_v3 calls only (no core change; the organ's
"action bid" in its crudest form). A slot-A fire on a forced step is
quasi-do(A): the action that produced it was decoupled from policy
and state. The do/see contrast per nominated link (a,b):
  P(b within W steps | a fired, FORCED) vs
  P(b within W steps | a fired, PASSIVE)
DELTA = P_passive - P_forced. DELTA >> 0: the association rode a
common drive (spontaneous-A carries the driver, forced-A does not) —
link DOWNGRADED. DELTA ~= 0 with both lifts above baseline: the link
SURVIVES randomization (do-consistent at this grain).

POWER (F48's clause, applied at card time): forced steps ~= 8.1% of
~28,800 phase steps ~= 2,300. From XO-0's fire rates, the lag -1
cluster (13/23/27/30/31; fire 0.03-0.08) yields ~70-190 forced
a-fires each — POWERED (floor 30/30 per condition). The long-lag
trigger suspects (16, 20, 24; fire ~0.00) yield ~5 — UNDERPOWERED BY
DESIGN, pre-declared: randomization cannot produce rare antecedents,
only decorrelate common ones; the trigger suspects await XO-2's
windowed manufacture. This card bills the cluster only.

CONTAMINATION stated: run_world_v3's own explore noise overrides
~EXPLORE_RATE of steps in BOTH conditions (passive steps are mostly
policy-driven, forced steps are fully random); the contrast is
diluted, not biased.

EXPECTED INFORMATION: cluster links downgraded -> the mutual-confound
reading (common driver) wins and XO-2 targets the driver; links
survive -> real 1-step couplings exist in the inherited web and XO-2
targets them for subtraction. Either way the organ's next phase is
aimed. No P108 claim here (that is XO-3's bill).

C20: 1 standing venue + wrapper (flagged; wrapper affects only
action choice on scheduled steps). 2 endpoint reads receipts only.
3 schedule is a pure function of the call counter (period 37 prime,
episodes 600 — no resonance). 4 within-link pairing (same link, two
conditions, same run). 5 XO-0's receipted fire rates fund the power
analysis above. 6 per-link floors 30/30; binomial SE reported.
7 single phase; replication after XO-3. 8 population = XO-0's
nominated cluster, fixed before launch; classes labeled."""

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
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, BOOT_SEED)
from replay_overnight_v3 import run_world_v3

B_SEEDS = ((97000, 4), (97001, 3))
GENS = 6
W = 5
PERIOD = 37
BURST = 3
LINKS = [(23, 30), (13, 30), (30, 31), (23, 31), (13, 31), (27, 30),
         (23, 27), (16, 24), (16, 20)]   # cluster + declared-underpowered
COND_FLOOR = 30
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'xo1_randomization.json')


class BurstModel:
    """Forces a random action for BURST steps every PERIOD predict
    calls. Pure wrapper; counter starts at 0 and stays in lockstep
    with web._global_step for run_world_v3 calls."""

    def __init__(self, model, seed):
        self.m = model
        self.rng = np.random.RandomState(seed)
        self.calls = 0

    def predict(self, w):
        act, aux = self.m.predict(w)
        if self.calls % PERIOD < BURST:
            act = self.rng.randint(0, 2,
                                   size=len(act)).astype(act.dtype)
        self.calls += 1
        return act, aux


def main():
    t0 = time.time()
    ro.CHURN_MIN = 2
    print('=== XO-1: scheduled randomization (do/see contrast) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    tau_rest, tau_d = calibrate_taus(model)
    engine = build_engine()
    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(),
                        debug_level=0, ledger_id='XO1')
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
    g0 = web._global_step
    burst = BurstModel(model, seed=90001)
    for g in range(GENS):
        if g > 0:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, burst, engine, web, bank, scan, g,
                             83000 + li * 1000 + g * 17,
                             lived_log, {}, (li, 0))
            gen_windows.append(w)
        web.anneal_all(web._global_step)
        replay_scans(web, scan, g, record)

        def as_court(ws):
            return [(p, classify_behavior_from_features(
                f, tau_rest, tau_d)) for p, f in ws]
        if pending_prev:
            lexicon, _ = ratify_pending(
                lexicon, pending_prev,
                [x for ws in gen_windows[:-1]
                 for x in as_court(ws)][:2000],
                as_court(gen_windows[-1]), ledger)
        lexicon, _, pending_prev = evolve_one_generation(
            lexicon,
            [x for ws in gen_windows[:-1] for x in as_court(ws)],
            as_court(gen_windows[-1]), ledger)
        print('  gen %d done (%.1f min), wrapper calls=%d'
              % (g + 1, (time.time() - t0) / 60, burst.calls))

    assert web._global_step - g0 == burst.calls, \
        'wrapper/global-step lockstep broken'

    def forced(t):
        return ((t - g0 - 1) % PERIOD) < BURST

    active = {sid: s for sid, s in web.slots.items()
              if s.state in ('open', 'closed')}
    fires = {sid: sorted(r.time_step for r in s.ledger.receipts
                         if r.kind == 'fit' and r.sign > 0)
             for sid, s in active.items()}
    total_steps = web._global_step - g0
    rows = []
    for a, b in LINKS:
        fa = fires.get(a, [])
        fb = np.asarray(fires.get(b, []))
        base_b = len(fb) / max(total_steps, 1)
        stats = {'FORCED': [0, 0], 'PASSIVE': [0, 0]}
        for t in fa:
            cond = 'FORCED' if forced(t) else 'PASSIVE'
            stats[cond][0] += 1
            if len(fb) and np.any((fb > t) & (fb <= t + W)):
                stats[cond][1] += 1
        nf, kf = stats['FORCED']
        npp, kp = stats['PASSIVE']
        pf = kf / nf if nf else None
        pp = kp / npp if npp else None
        if nf < COND_FLOOR or npp < COND_FLOOR:
            tag = 'UNTESTED (forced=%d passive=%d)' % (nf, npp)
        else:
            se = float(np.sqrt(pf * (1 - pf) / nf
                               + pp * (1 - pp) / npp))
            delta = pp - pf
            if delta > 2 * se:
                tag = 'DOWNGRADED (confounded: delta=%.3f > 2se=%.3f)' \
                    % (delta, 2 * se)
            elif pf > base_b * W and pp > base_b * W:
                tag = 'SURVIVES (do-consistent: delta=%.3f, both ' \
                    'lifts above baseline %.3f)' % (delta, base_b * W)
            else:
                tag = 'NO-LIFT (association below baseline under ' \
                    'both conditions)'
        rows.append({'a': a, 'b': b, 'n_forced': nf, 'n_passive': npp,
                     'p_forced': pf, 'p_passive': pp,
                     'baseline_w': round(base_b * W, 4), 'tag': tag})
        print('  %s->%s: forced %s/%s passive %s/%s base=%.3f  %s'
              % (a, b, kf, nf, kp, npp, base_b * W, tag))

    out = {'links': rows, 'total_steps': int(total_steps),
           'forced_frac': BURST / PERIOD,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

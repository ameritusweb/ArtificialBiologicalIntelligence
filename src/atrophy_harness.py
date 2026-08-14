"""P86 + P103(atrophy) — induced flow-freeze with inventory intact; P87
rehab contrast rides as pre-registered DESCRIPTIVE (unbilled, first
measurement). (T158 third arrival / T159 fifth arrival; card locked at
launch 2026-08-13.)

THE CLAIMS:
  P86 — freeze the FLOW (structure genesis: compose scans; closure
  admission) while leaving the STOCK (inventory, fit dynamics) intact:
  (i) under STATIC in-distribution worlds the freeze is INVISIBLE to
  service metrics (tightness, calibration) — stock pays the bills;
  (ii) the flow ledger sees it immediately (genesis/closure event
  counts — mechanical, reported, not billed as discovery: this is the
  fifth arrival's world-tag readability clause);
  (iii) under DRIFT the missing vocabulary bills: the frozen arm's
  strain exceeds control's. The conjunction of (i)+(iii) IS
  P103's atrophy signature: the ledger flags the deficit BEFORE the
  service metrics diverge.
  P87 (descriptive this card) — rehabilitation order: after drift, the
  frozen web forks into REHAB-ORDERED (cheapest structure first:
  compose co-fit floor 100 for the first half of rehab, then 50) and
  REHAB-ALL (everything re-enabled at once). Endpoint reported:
  calibration recovery and discharge (genesis+closure) trajectories.
  No verdict billed — floors unknown at first measurement; the billed
  P87 card comes after these magnitudes are on record.

DESIGN — one lived stream, two accountants (+ two rehab forks):
  WARMUP 4 worlds, flow ON both arms, identity asserted.
  STATIC 8 worlds, control flow ON / frozen flow OFF (no compose
  scans; _check_closure disabled on the frozen web INSTANCE — sandbox
  policy, core sov.py untouched). Fit dynamics identical by
  construction.
  DRIFT 4 worlds (fresh seeds, tiers swapped), both arms measured,
  frozen stays frozen.
  REHAB 4 worlds: frozen web deep-copied into the two rehab forks
  (state identity asserted at fork); control continues.

VERDICTS (fixed): P86 SUPPORTED iff (i) static-phase service gap
inside the equivalence band (|d tightness| < 0.005 AND
|d calibration| < 0.01) AND (iii) drift-end calibration error
frozen - control >= 0.01. NOT SUPPORTED iff static gap exceeds the
band (stock does NOT pay — atrophy immediately visible) OR drift gap
<= 0 (the missing structure costs nothing — flow decorative at this
scale). UNTESTED: control static-phase genesis events < 8 (nothing to
freeze — phenomenon absent), or fresh fits < 500 per arm per phase.

C20 (seven): 1 domain — in-dist harness; drift is the treatment.
2 endpoint independence — the freeze writes structure genesis;
endpoints measure geometry-vs-world (fresh distances, own-fit confirm
rates), the legal path; genesis counts are reported as mechanics, not
billed as discovery. 3 exogeneity — freeze/unfreeze at pre-registered
world indices. 4 pairing — one stream, shared model/engine; warmup
identity asserted (receipts+edges); rehab forks state-identical by
deepcopy, asserted. 5 phenomenon strength — floor: control must
produce >= 8 genesis events during static (F21-class runs produced up
to 3/boundary); drift = reseed + tier swap, the standing strongest
shift. 6 sensitivity — composed-structure tightness contribution
O(1e-2) estimated from F21-class webs vs 1e-4 harness noise;
calibration gaps O(0.01) vs 1e-3 noise. 7 genesis rates — genesis is
an endpoint only as a mechanical count; rate design: 8 static
boundaries x up to 3 composes bounds the phenomenon at 24, floor 8.
"""

import copy
import json
import os
import time

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED
from staged_fit_experiment import Accountant
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

WARMUP_WORLDS = [(97700 + i, (4, 3)[i % 2]) for i in range(4)]
STATIC_WORLDS = [(97710 + i, (4, 3)[i % 2]) for i in range(8)]
DRIFT_WORLDS = [(97800 + i, (3, 4)[i % 2]) for i in range(4)]
REHAB_WORLDS = [(97900 + i, (3, 4)[i % 2]) for i in range(4)]
EPISODES = 2
STEPS = 400
TIGHT_BAND = 0.005
CAL_BAND = 0.01
DRIFT_GAP = 0.01
GENESIS_FLOOR = 8
FRESH_FLOOR = 500
ORDERED_FLOOR_HI = 100
ORDERED_FLOOR_LO = 50
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p86_atrophy.json')


class FlowArm(Accountant):
    def __init__(self, name):
        Accountant.__init__(self, name, staged=False, consume=False)
        self.flow = True
        self.compose_floor = 50
        self.compose_cap = 3
        self.genesis_events = 0
        self.closure_events = 0

    def freeze(self):
        self.flow = False
        self.web._check_closure = lambda sid: None   # instance shadow

    def thaw(self, compose_floor):
        self.flow = True
        self.compose_floor = compose_floor
        if '_check_closure' in self.web.__dict__:
            del self.web.__dict__['_check_closure']


def run_phase(worlds, arms, engine, model, base):
    counter = base
    for w_seed, tier in worlds:
        env = TieredEnvironment(seed=w_seed, tier=tier)
        np.random.seed(w_seed * 7)
        env.rng = np.random.RandomState(w_seed * 7 + 1)
        rng = np.random.RandomState(w_seed * 7 + 2)
        bank = LiveReceptorBank()
        for ep in range(EPISODES):
            org = Organism()
            org.reset()
            for step in range(STEPS):
                w = org.get_observation_window()
                act, _ = model.predict(w)
                r = rng.random()
                if r < PROBE_RATE_FLOOR:
                    act = np.zeros_like(act)
                elif r < EXPLORE_RATE:
                    act = rng.randint(0, 2, size=len(act)).astype(
                        act.dtype)
                obs, reward = org.step(act, env, step)
                rv = bank.compute(obs, act, None, reward)
                emb = engine.encoder.embed(engine._core_obs(obs))
                counter += 1
                for a in arms:
                    closed0 = sum(1 for s in a.web.slots.values()
                                  if s.state == 'closed')
                    a.process(rv, emb, obs, reward, counter, ep)
                    closed1 = sum(1 for s in a.web.slots.values()
                                  if s.state == 'closed')
                    if closed1 > closed0:
                        a.closure_events += closed1 - closed0
        for a in arms:
            if not a.flow:
                continue
            web = a.web
            done = 0
            for (x, y), n in sorted(a.cofit.items(),
                                    key=lambda kv: (-kv[1], kv[0])):
                if done >= a.compose_cap or n < a.compose_floor:
                    break
                if (x, y) in a.composed:
                    continue
                sx, sy = web.slots.get(x), web.slots.get(y)
                if (sx is None or sy is None or sx.state != 'open'
                        or sy.state != 'open'):
                    continue
                if web.compose(x, y)[0] >= 0:
                    a.composed.add((x, y))
                    a.genesis_events += 1
                    done += 1
    return counter


def service(arm, window=200):
    tight, n = arm.web.pop_fresh_tightness()
    cal = []
    for s in arm.web.slots.values():
        if s.state not in ('open', 'closed'):
            continue
        recent = [r for r in s.ledger.receipts
                  if r.kind == 'fit'][-window:]
        if len(recent) >= 10:
            confirm = sum(1 for r in recent if r.sign > 0) / len(recent)
            cal.append(abs(s.ledger.certainty - confirm))
    return {'tightness': tight, 'n_fresh': n,
            'calibration': (float(np.mean(cal)) if cal else None)}


def flow_ledger(arm):
    return {'genesis': arm.genesis_events,
            'closures': arm.closure_events,
            'edges': len(arm.web.edges),
            'slots': sum(1 for s in arm.web.slots.values()
                         if s.state in ('open', 'closed'))}


def fork(frozen, name):
    child = FlowArm(name)
    child.web = copy.deepcopy(frozen.web)
    child.cofit = copy.deepcopy(frozen.cofit)
    child.composed = set(frozen.composed)
    child.genesis_events = frozen.genesis_events
    child.closure_events = frozen.closure_events
    child.flow = False
    return child


def main():
    t0 = time.time()
    print('=== P86/P103(atrophy): flow freeze, one stream ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    control = FlowArm('CONTROL')
    frozen = FlowArm('FROZEN')

    print('warmup (4 worlds, flow on both)...')
    c = run_phase(WARMUP_WORLDS, [control, frozen], engine, model, 0)
    sig = [(len(a.web.edges), a.web.get_stats()['total_receipts'])
           for a in (control, frozen)]
    assert sig[0] == sig[1], 'C20 check 4: warmup states diverged'
    print('  identity at freeze: %s' % (sig,))
    for a in (control, frozen):
        a.web.pop_fresh_tightness()
        a.genesis_events = 0
        a.closure_events = 0

    print('static (8 worlds, frozen arm flow OFF)...')
    frozen.freeze()
    c = run_phase(STATIC_WORLDS, [control, frozen], engine, model, c)
    static_svc = {a.name: service(a) for a in (control, frozen)}
    static_flow = {a.name: flow_ledger(a) for a in (control, frozen)}
    print('  static service: %s' % static_svc)
    print('  static flow:    %s' % static_flow)

    print('drift (4 worlds, tier swap + fresh seeds)...')
    c = run_phase(DRIFT_WORLDS, [control, frozen], engine, model, c)
    drift_svc = {a.name: service(a) for a in (control, frozen)}
    drift_flow = {a.name: flow_ledger(a) for a in (control, frozen)}
    print('  drift service: %s' % drift_svc)

    print('rehab (4 worlds, frozen forks; descriptive)...')
    rehab_all = fork(frozen, 'REHAB-ALL')
    rehab_ord = fork(frozen, 'REHAB-ORDERED')
    fsig = (rehab_all.web.get_stats()['total_receipts'],
            rehab_ord.web.get_stats()['total_receipts'],
            frozen.web.get_stats()['total_receipts'])
    assert fsig[0] == fsig[1] == fsig[2], 'fork identity broken'
    rehab_all.thaw(ORDERED_FLOOR_LO)
    rehab_ord.thaw(ORDERED_FLOOR_HI)
    half = len(REHAB_WORLDS) // 2
    c = run_phase(REHAB_WORLDS[:half],
                  [control, rehab_all, rehab_ord], engine, model, c)
    rehab_ord.compose_floor = ORDERED_FLOOR_LO
    run_phase(REHAB_WORLDS[half:],
              [control, rehab_all, rehab_ord], engine, model, c)
    rehab_svc = {a.name: service(a)
                 for a in (control, rehab_all, rehab_ord)}
    rehab_flow = {a.name: flow_ledger(a)
                  for a in (control, rehab_all, rehab_ord)}
    print('  rehab service: %s' % rehab_svc)
    print('  rehab flow:    %s' % rehab_flow)

    # ------------------------------------------------------- verdict
    sc, sf = static_svc['CONTROL'], static_svc['FROZEN']
    dc, df = drift_svc['CONTROL'], drift_svc['FROZEN']
    d_tight = abs(sf['tightness'] - sc['tightness'])
    d_cal = (abs(sf['calibration'] - sc['calibration'])
             if None not in (sf['calibration'], sc['calibration'])
             else None)
    drift_gap = (df['calibration'] - dc['calibration']
                 if None not in (df['calibration'], dc['calibration'])
                 else None)
    fresh_ok = all(v['n_fresh'] >= FRESH_FLOOR
                   for v in list(static_svc.values())
                   + list(drift_svc.values()))
    genesis = static_flow['CONTROL']['genesis'] \
        + static_flow['CONTROL']['closures']

    if genesis < GENESIS_FLOOR or not fresh_ok:
        verdict = ('UNTESTED (floors: control static genesis+closures='
                   '%d/%d fresh_ok=%s)'
                   % (genesis, GENESIS_FLOOR, fresh_ok))
    elif d_cal is None or drift_gap is None:
        verdict = 'UNTESTED (calibration population too small)'
    elif d_tight < TIGHT_BAND and d_cal < CAL_BAND \
            and drift_gap >= DRIFT_GAP:
        verdict = ('SUPPORTED: freeze invisible under static worlds '
                   '(d_tight=%.4f d_cal=%.4f) and billed at drift '
                   '(frozen-control calibration gap %.4f) — the '
                   'atrophy signature: flow ledger flags first, '
                   'service pays later' % (d_tight, d_cal, drift_gap))
    elif d_tight >= TIGHT_BAND or d_cal >= CAL_BAND:
        verdict = ('NOT SUPPORTED (invisibility clause): static gap '
                   'outside band (d_tight=%.4f d_cal=%.4f) — stock '
                   'does not pay' % (d_tight, d_cal))
    else:
        verdict = ('NOT SUPPORTED (cost clause): drift gap %.4f <= 0 '
                   '— missing structure costs nothing at this scale'
                   % drift_gap)
    print('\nP86 VERDICT: %s' % verdict)

    out = {'static_service': static_svc, 'static_flow': static_flow,
           'drift_service': drift_svc, 'drift_flow': drift_flow,
           'rehab_service': rehab_svc, 'rehab_flow': rehab_flow,
           'd_tight_static': d_tight, 'd_cal_static': d_cal,
           'drift_calibration_gap': drift_gap,
           'verdict': verdict,
           'p87_status': 'descriptive only this card (pre-registered)',
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

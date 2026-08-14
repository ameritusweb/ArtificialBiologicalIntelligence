"""P86 v2 — atrophy with POPULATION-RESTRICTED endpoints. (Card locked
at launch 2026-08-13; supersedes v1's endpoints; v1 receipts stand.)

V1 DISPOSITION (results/p86_atrophy.json, on the record): printed
NOT SUPPORTED (static calibration gap 0.063 outside band), CORRECTED
on inspection to VOID-BY-POPULATION-CONFOUND — the ledger-mass
clause's second strike in two days. n_fresh CONTROL 156k vs FROZEN
89k: the treatment (flow) writes the SLOT POPULATION (69 vs 45), and
the v1 endpoints averaged service over each arm's own population, so
the comparison mixed the mechanical consequence of the treatment into
the measured quantity (C20 check 2 violated in retrospect: the freeze
wrote into the endpoint through composition). The v1 'calibration
gap' is the signature of CONTROL'S YOUNG COMPOSED SLOTS being born
uncalibrated (certainty prior 0.5 vs high confirm), not of frozen-arm
service. Two real observations preserved: (i) composed slots are born
poorly calibrated — organs earn their voice (T139) at slot grain,
reported as a v2 secondary; (ii) the rehab forks returned
bit-identical (ordered floor 100 never bound before the 3/boundary
cap) — P87's contrast needs a different lever; P87 remains
descriptive-only.

V2 CHANGES (all else identical to the v1 card, same seeds):
  1. PRIMARY endpoints (tightness, calibration) computed ONLY over
     the SHARED POPULATION: slots existing at the freeze point,
     still open/closed — identical membership in both arms by
     construction.
  2. Whole-web metrics and composed-only metrics reported as
     secondaries (the young-organ calibration trajectory).
  3. Rehab forks kept, descriptive, with the compose cap RAISED to 6
     per boundary during rehab so the ordered floor can bind.

VERDICTS (fixed, on shared-population metrics): P86 SUPPORTED iff
static |d tightness| < 0.005 AND static |d calibration| < 0.01 AND
drift-phase frozen-minus-control gap >= +0.01 on calibration OR
>= +0.005 on tightness (the missing vocabulary may bill on either
channel; direction: frozen worse). NOT SUPPORTED (invisibility): a
static gap outside its band. NOT SUPPORTED (cost): drift gaps below
both gates. UNTESTED: control static genesis+closures < 8, or shared
population < 20 slots with calibration data, or fresh fits < 500 per
arm per phase (shared population).

C20 (seven): as v1 with check 2 REPAIRED — the endpoint population is
now fixed at the freeze point, treatment cannot write membership;
check 5 carries v1's receipt that drift moved tightness (frozen-worse
+0.0074 whole-web at drift vs +0.0028 static — the predicted shape,
now to be measured un-confounded); check 6 — gates sit at the v1
observed magnitudes' scale (tightness gap 3e-3..8e-3, calibration
noise ~1e-3 within fixed population).
"""

import copy
import json
import os
import time

import numpy as np

from replay_overnight import build_engine, BOOT_SEED
from atrophy_harness import (FlowArm, run_phase, fork,
                             WARMUP_WORLDS, STATIC_WORLDS,
                             DRIFT_WORLDS, REHAB_WORLDS,
                             GENESIS_FLOOR, ORDERED_FLOOR_HI,
                             ORDERED_FLOOR_LO)
from train import generate_training_data, train_model

TIGHT_BAND = 0.005
CAL_BAND = 0.01
DRIFT_CAL_GATE = 0.01
DRIFT_TIGHT_GATE = 0.005
SHARED_FLOOR = 20
FRESH_FLOOR = 500
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p86_v2_atrophy.json')


def shared_service(arm, population, t_from, window=200):
    """Tightness and calibration over a FIXED slot population, using
    fit receipts created after t_from (per-phase, per-slot)."""
    dists, cal = [], []
    n_fresh = 0
    for sid in population:
        s = arm.web.slots.get(sid)
        if s is None or s.state not in ('open', 'closed'):
            continue
        phase = [r for r in s.ledger.receipts
                 if r.kind == 'fit' and r.created_at > t_from]
        n_fresh += len(phase)
        pos = [r for r in phase if r.sign > 0 and r.magnitude > 0]
        dists.extend(1.0 - min(r.magnitude, 1.0) for r in pos[-window:])
        recent = phase[-window:]
        if len(recent) >= 10:
            confirm = sum(1 for r in recent if r.sign > 0) / len(recent)
            cal.append(abs(s.ledger.certainty - confirm))
    return {'tightness': (float(np.mean(dists)) if dists else None),
            'calibration': (float(np.mean(cal)) if cal else None),
            'n_fresh': n_fresh, 'n_cal_slots': len(cal)}


def composed_service(arm, population, t_from, window=200):
    comp = [sid for sid, s in arm.web.slots.items()
            if sid not in population and s.state in ('open', 'closed')]
    return shared_service(arm, comp, t_from, window)


def main():
    t0 = time.time()
    print('=== P86 v2: flow freeze, shared-population endpoints ===')
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
    shared = sorted(sid for sid, s in control.web.slots.items()
                    if s.state in ('open', 'closed'))
    print('  identity at freeze: %s shared_population=%d'
          % (sig, len(shared)))
    for a in (control, frozen):
        a.genesis_events = 0
        a.closure_events = 0

    t_static = control.web._global_step
    print('static (8 worlds, frozen arm flow OFF)...')
    frozen.freeze()
    c = run_phase(STATIC_WORLDS, [control, frozen], engine, model, c)
    static_sh = {a.name: shared_service(a, shared, t_static)
                 for a in (control, frozen)}
    static_comp = composed_service(control, shared, t_static)
    static_flow = {a.name: {'genesis': a.genesis_events,
                            'closures': a.closure_events,
                            'slots': sum(1 for s in a.web.slots.values()
                                         if s.state in
                                         ('open', 'closed'))}
                   for a in (control, frozen)}
    print('  static shared: %s' % static_sh)
    print('  static composed-only (control): %s' % static_comp)
    print('  static flow: %s' % static_flow)

    t_drift = control.web._global_step
    print('drift (4 worlds, tier swap + fresh seeds)...')
    c = run_phase(DRIFT_WORLDS, [control, frozen], engine, model, c)
    drift_sh = {a.name: shared_service(a, shared, t_drift)
                for a in (control, frozen)}
    drift_comp = composed_service(control, shared, t_drift)
    print('  drift shared: %s' % drift_sh)
    print('  drift composed-only (control): %s' % drift_comp)

    print('rehab (4 worlds, forks; descriptive; cap 6/boundary)...')
    rehab_all = fork(frozen, 'REHAB-ALL')
    rehab_ord = fork(frozen, 'REHAB-ORDERED')
    rehab_all.thaw(ORDERED_FLOOR_LO)
    rehab_ord.thaw(ORDERED_FLOOR_HI)
    for a in (rehab_all, rehab_ord):
        a.compose_cap = 6
    t_rehab = control.web._global_step
    half = len(REHAB_WORLDS) // 2
    c = run_phase(REHAB_WORLDS[:half],
                  [control, rehab_all, rehab_ord], engine, model, c)
    rehab_ord.compose_floor = ORDERED_FLOOR_LO
    run_phase(REHAB_WORLDS[half:],
              [control, rehab_all, rehab_ord], engine, model, c)
    rehab_sh = {a.name: shared_service(a, shared, t_rehab)
                for a in (control, rehab_all, rehab_ord)}
    rehab_flow = {a.name: {'genesis': a.genesis_events,
                           'closures': a.closure_events}
                  for a in (control, rehab_all, rehab_ord)}
    print('  rehab shared: %s' % rehab_sh)
    print('  rehab flow: %s' % rehab_flow)

    # ------------------------------------------------------- verdict
    sc, sf = static_sh['CONTROL'], static_sh['FROZEN']
    dc, df = drift_sh['CONTROL'], drift_sh['FROZEN']
    ok_pop = (sc['n_cal_slots'] >= SHARED_FLOOR
              and sf['n_cal_slots'] >= SHARED_FLOOR)
    fresh_ok = all(v['n_fresh'] >= FRESH_FLOOR
                   for v in (sc, sf, dc, df))
    genesis = static_flow['CONTROL']['genesis'] \
        + static_flow['CONTROL']['closures']
    d_tight = (abs(sf['tightness'] - sc['tightness'])
               if None not in (sf['tightness'], sc['tightness'])
               else None)
    d_cal = (abs(sf['calibration'] - sc['calibration'])
             if None not in (sf['calibration'], sc['calibration'])
             else None)
    g_cal = (df['calibration'] - dc['calibration']
             if None not in (df['calibration'], dc['calibration'])
             else None)
    g_tight = (df['tightness'] - dc['tightness']
               if None not in (df['tightness'], dc['tightness'])
               else None)

    if genesis < GENESIS_FLOOR or not ok_pop or not fresh_ok:
        verdict = ('UNTESTED (genesis=%d/%d pop_ok=%s fresh_ok=%s)'
                   % (genesis, GENESIS_FLOOR, ok_pop, fresh_ok))
    elif None in (d_tight, d_cal, g_cal, g_tight):
        verdict = 'UNTESTED (endpoint population too small)'
    elif d_tight < TIGHT_BAND and d_cal < CAL_BAND and \
            (g_cal >= DRIFT_CAL_GATE or g_tight >= DRIFT_TIGHT_GATE):
        verdict = ('SUPPORTED: freeze invisible on the shared '
                   'population under static worlds (d_tight=%.4f '
                   'd_cal=%.4f) and billed at drift (cal gap=%.4f '
                   'tight gap=%.4f) — the atrophy signature'
                   % (d_tight, d_cal, g_cal, g_tight))
    elif d_tight >= TIGHT_BAND or d_cal >= CAL_BAND:
        verdict = ('NOT SUPPORTED (invisibility): static shared-'
                   'population gap outside band (d_tight=%.4f '
                   'd_cal=%.4f)' % (d_tight, d_cal))
    else:
        verdict = ('NOT SUPPORTED (cost): drift gaps below gates '
                   '(cal %.4f < %.2f and tight %.4f < %.3f) — '
                   'missing structure costs nothing on the shared '
                   'population at this scale'
                   % (g_cal, DRIFT_CAL_GATE, g_tight,
                      DRIFT_TIGHT_GATE))
    print('\nP86 v2 VERDICT: %s' % verdict)

    out = {'shared_population_n': len(shared),
           'static_shared': static_sh, 'drift_shared': drift_sh,
           'static_composed_control': static_comp,
           'drift_composed_control': drift_comp,
           'static_flow': static_flow,
           'rehab_shared': rehab_sh, 'rehab_flow': rehab_flow,
           'gaps': {'static_d_tight': d_tight, 'static_d_cal': d_cal,
                    'drift_cal_gap': g_cal, 'drift_tight_gap': g_tight},
           'verdict': verdict,
           'p87_status': 'descriptive only (pre-registered)',
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

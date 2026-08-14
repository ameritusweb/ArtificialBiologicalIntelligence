"""P86 v3 — atrophy on the CONSUMPTION channel. (Card locked at launch
2026-08-13; supersedes v2's venue; v1/v2 receipts stand.)

V2 DISPOSITION (results/p86_v2_atrophy.json, on the record): earned
NOT SUPPORTED (cost clause) with a mechanism finding — the shared
population came back BIT-IDENTICAL between CONTROL and FROZEN in
every phase (static 0.15890995649675776 = 0.15890995649675776; drift
and rehab likewise). In the plain-fit sandbox, new structure has NO
CHANNEL through which to serve old structure: composes create slots
and edges that nothing consumes, so the flow is completely decoupled
from stock service. 'Missing structure costs nothing' is true there
BY CONSTRUCTION. The finding: FLOW BILLS ONLY THROUGH CONSUMPTION —
F22's broad-mediator lesson and P76's delta-accounting thesis arriving
from the atrophy side. (Also: rehab fork ordering lever failed to bind
twice — ORDERED_FLOOR must be set from the co-fit distribution, not a
constant; P87 stays descriptive and undesigned at this venue.)

V3 DESIGN: both arms run the S+C policy (staged fringe-ordered
expectations, CONSUME_MODE='expectation' — F32's supported channel).
Now edges are LOAD-BEARING: expectation receipts flow through them
into geometry. CONTROL: flow on (compose scans grow the edge graph).
FROZEN: flow off after warmup (stuck with warmup edges; no new slots,
no new edges, closure checks shadowed). Phases as v1/v2 (same seeds):
warmup 4 / static 8 / drift 4; rehab dropped from this card (P87
awaits its own design). Endpoints on the SHARED at-freeze population
(v2's repair, kept): tightness and calibration per phase; plus
consumption telemetry (expectation receipts per arm per phase — the
mechanical mediator, reported).

VERDICTS (fixed): SUPPORTED iff static |d tightness| < 0.005 AND
static |d calibration| < 0.01 AND drift gap (frozen - control) >=
+0.005 tightness OR >= +0.01 calibration. NOT SUPPORTED
(invisibility): static gap outside band — flow's service shows
immediately, no latent phase. NOT SUPPORTED (cost): drift gaps below
both gates WITH the mediator alive (control expectation receipts
exceed frozen's by >= 100 in drift phase) — structure served nothing
even when consumed. VOID-BY-MEDIATOR: control-minus-frozen
expectation receipts < 100 in drift (the channel never carried the
difference; the venue cannot test the claim). UNTESTED: floors as v2
(genesis >= 8, shared calibration slots >= 20, fresh >= 500).

C20 (seven): as v2 with check 5 now carrying the MEDIATOR floor (the
lesson of this session's three VOIDs: floor every population AND every
channel the claim needs). Check 2: consumption writes geometry through
expectation receipts on BOTH arms; the arms differ only in edge
supply; endpoints measure geometry-vs-world. Check 6: F32 measured
S+C-vs-P+D tightness margins at ~1e-2 scale with ~1050 events;
edge-count differences here (24 vs 72+) are of the scale that moved
sharpness in the density sweep (0.32 -> 0.29 over 34x mass).
"""

import json
import os
import time

import numpy as np

import staged_fit_experiment as sfe
from replay_overnight import build_engine, BOOT_SEED
from atrophy_harness import (WARMUP_WORLDS, STATIC_WORLDS,
                             DRIFT_WORLDS, GENESIS_FLOOR)
from p86_v2 import shared_service, composed_service
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)
from environment import Organism
from environment_tiers import TieredEnvironment
from live_receptors import LiveReceptorBank

sfe.CONSUME_MODE = 'expectation'

TIGHT_BAND = 0.005
CAL_BAND = 0.01
DRIFT_TIGHT_GATE = 0.005
DRIFT_CAL_GATE = 0.01
MEDIATOR_FLOOR = 100
SHARED_FLOOR = 20
FRESH_FLOOR = 500
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p86_v3_atrophy.json')


class ConsumingFlowArm(sfe.Accountant):
    """S+C accountant with a freezable flow (compose scans +
    closure admission)."""

    def __init__(self, name):
        sfe.Accountant.__init__(self, name, staged=True, consume=True)
        self.flow = True
        self.compose_floor = 50
        self.compose_cap = 3
        self.genesis_events = 0

    def freeze(self):
        self.flow = False
        self.web._check_closure = lambda sid: None


def run_phase(worlds, arms, engine, model, base):
    counter = base
    for w_seed, tier in worlds:
        env = TieredEnvironment(seed=w_seed, tier=tier)
        np.random.seed(w_seed * 7)
        env.rng = np.random.RandomState(w_seed * 7 + 1)
        rng = np.random.RandomState(w_seed * 7 + 2)
        bank = LiveReceptorBank()
        for ep in range(2):
            org = Organism()
            org.reset()
            for step in range(400):
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
                    a.process(rv, emb, obs, reward, counter, ep)
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


def main():
    t0 = time.time()
    print('=== P86 v3: atrophy on the consumption channel (S+C) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    control = ConsumingFlowArm('CONTROL')
    frozen = ConsumingFlowArm('FROZEN')

    print('warmup (4 worlds, flow on, consumption on, both)...')
    c = run_phase(WARMUP_WORLDS, [control, frozen], engine, model, 0)
    sig = [(len(a.web.edges), a.web.get_stats()['total_receipts'],
            a.constrains) for a in (control, frozen)]
    assert sig[0] == sig[1], 'C20 check 4: warmup states diverged'
    shared = sorted(sid for sid, s in control.web.slots.items()
                    if s.state in ('open', 'closed'))
    print('  identity at freeze: %s shared=%d' % (sig, len(shared)))
    for a in (control, frozen):
        a.genesis_events = 0
    cons_mark = {a.name: a.constrains for a in (control, frozen)}

    t_static = control.web._global_step
    print('static (8 worlds, frozen flow OFF, both consuming)...')
    frozen.freeze()
    c = run_phase(STATIC_WORLDS, [control, frozen], engine, model, c)
    static_sh = {a.name: shared_service(a, shared, t_static)
                 for a in (control, frozen)}
    static_cons = {a.name: a.constrains - cons_mark[a.name]
                   for a in (control, frozen)}
    cons_mark = {a.name: a.constrains for a in (control, frozen)}
    static_flow = {a.name: {'genesis': a.genesis_events,
                            'edges': len(a.web.edges)}
                   for a in (control, frozen)}
    print('  static shared: %s' % static_sh)
    print('  static consumption: %s flow: %s'
          % (static_cons, static_flow))

    t_drift = control.web._global_step
    print('drift (4 worlds, tier swap + fresh seeds)...')
    run_phase(DRIFT_WORLDS, [control, frozen], engine, model, c)
    drift_sh = {a.name: shared_service(a, shared, t_drift)
                for a in (control, frozen)}
    drift_cons = {a.name: a.constrains - cons_mark[a.name]
                  for a in (control, frozen)}
    print('  drift shared: %s' % drift_sh)
    print('  drift consumption: %s' % drift_cons)

    sc, sf = static_sh['CONTROL'], static_sh['FROZEN']
    dc, df = drift_sh['CONTROL'], drift_sh['FROZEN']
    ok_pop = (sc['n_cal_slots'] >= SHARED_FLOOR
              and sf['n_cal_slots'] >= SHARED_FLOOR)
    fresh_ok = all(v['n_fresh'] >= FRESH_FLOOR
                   for v in (sc, sf, dc, df))
    genesis = static_flow['CONTROL']['genesis']
    mediator = drift_cons['CONTROL'] - drift_cons['FROZEN']
    d_tight = (abs(sf['tightness'] - sc['tightness'])
               if None not in (sf['tightness'], sc['tightness'])
               else None)
    d_cal = (abs(sf['calibration'] - sc['calibration'])
             if None not in (sf['calibration'], sc['calibration'])
             else None)
    g_tight = (df['tightness'] - dc['tightness']
               if None not in (df['tightness'], dc['tightness'])
               else None)
    g_cal = (df['calibration'] - dc['calibration']
             if None not in (df['calibration'], dc['calibration'])
             else None)

    if genesis < GENESIS_FLOOR or not ok_pop or not fresh_ok:
        verdict = ('UNTESTED (genesis=%d/%d pop_ok=%s fresh_ok=%s)'
                   % (genesis, GENESIS_FLOOR, ok_pop, fresh_ok))
    elif None in (d_tight, d_cal, g_tight, g_cal):
        verdict = 'UNTESTED (endpoint population too small)'
    elif mediator < MEDIATOR_FLOOR:
        verdict = ('VOID-BY-MEDIATOR: control-minus-frozen drift '
                   'expectation receipts %d < %d — the consumption '
                   'channel never carried the edge difference'
                   % (mediator, MEDIATOR_FLOOR))
    elif d_tight < TIGHT_BAND and d_cal < CAL_BAND and \
            (g_tight >= DRIFT_TIGHT_GATE or g_cal >= DRIFT_CAL_GATE):
        verdict = ('SUPPORTED: flow-freeze invisible under static '
                   'worlds (d_tight=%.4f d_cal=%.4f), bills at drift '
                   'through consumption (tight gap=%.4f cal gap=%.4f, '
                   'mediator=%d receipts)'
                   % (d_tight, d_cal, g_tight, g_cal, mediator))
    elif d_tight >= TIGHT_BAND or d_cal >= CAL_BAND:
        verdict = ('NOT SUPPORTED (invisibility): static gap outside '
                   'band (d_tight=%.4f d_cal=%.4f) — flow serves '
                   'immediately when consumed' % (d_tight, d_cal))
    else:
        verdict = ('NOT SUPPORTED (cost): drift gaps below gates '
                   '(tight %.4f cal %.4f) with mediator alive (%d) — '
                   'consumed structure still serves nothing'
                   % (g_tight, g_cal, mediator))
    print('\nP86 v3 VERDICT: %s' % verdict)

    out = {'shared_population_n': len(shared),
           'static_shared': static_sh, 'drift_shared': drift_sh,
           'static_consumption': static_cons,
           'drift_consumption': drift_cons,
           'static_flow': static_flow,
           'gaps': {'static_d_tight': d_tight, 'static_d_cal': d_cal,
                    'drift_tight_gap': g_tight,
                    'drift_cal_gap': g_cal, 'mediator': mediator},
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

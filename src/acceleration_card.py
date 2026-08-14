"""ACCELERATION CARD — generalization as capital (task #62; P96's
first partial receipt; carries transfer_rethreshold v2 as its warm
arm). Card locked at launch 2026-08-13.

THE CLAIM (T159 property 1, capital form): a web warm-started with
earned region-A structure learns region B cheaper than a cold web —
the map retains value after the search that built it ends.

ARMS (same boot model, same engine, same B worlds, same seeds — run
sequentially in one process; check 4 clean):
  WARM = the solvent nursery web (data/nursery_web_s98300.pkl: 33
  inherited + 17 rent-solvent composed, earned under the economy-on
  venue, F45) + the full transfer protocol: snapshot the nursery
  distribution (closes the old book), SCOUT region B without fitting,
  rethreshold open slots (battery-sealed C12 repair). The protocol is
  part of the capital — the claim prices capital WITH its handling
  costs.
  COLD = standard boot (33 inherited open slots, generic thresholds).
  No old book exists for a cold web, so no rethreshold — the honest
  statement of the comparison is warm-start PROTOCOL vs cold BOOT.

REGION B: two fresh sandbox-class... no — held-out LINEAGE pair
(seeds 97100 tier-4 / 97101 tier-3, interpret(describe(...))), 8
generations, NO structure dosing (capital test, not strain test),
encoder rebuilt per gen on each arm's own lived log + rebase (the
standing rhythm), anneal_all per gen (the economy runs — F44 parity).

ENDPOINTS (fixed):
  PRIMARY (class-closed per the inspector clause: both arms hold the
  IDENTICAL inherited population; composed slots exist only in WARM
  and are excluded from the comparison): inherited-slot CLOSURES over
  8 gens (count) and FIRST-CLOSURE GENERATION (latency).
  SECONDARY (descriptive): near-miss mass; WARM's solvent-composed
  survival at gen 2 and gen 8 — which is transfer_rethreshold v2's
  endpoint: SELECTIVE vocabulary under the transfer protocol, against
  F44's insolvent baseline (0/72 by gen 1).
  Court engagement was named in the task and is DROPPED from this
  card (no court in this harness path); flagged, not silently cut.

VERDICTS: UNTESTED if combined inherited closures < 2 (venue too
quiet at this horizon). ACCELERATION SUPPORTED iff WARM strictly
exceeds COLD on closures AND ties-or-beats on latency. NOT SUPPORTED
iff COLD ties-or-beats WARM on both. PARTIAL otherwise. v2 endpoint
bills separately: solvent survival >= 50% at gen 2 = rethreshold
MECHANISM VALIDATED on selective stock; < 10% = INSUFFICIENT-DEEPER
(neither solvency nor recalibration explains transfer death).

C20 (eight): 1 domain — standing machinery, warm pickle from the
receipted nursery. 2 endpoint independence — closures admitted by the
corridor gate on fits; the protocol writes thresholds before any B
contact. 3 exogeneity — arms differ only in starting web + protocol;
worlds/seeds/model shared. 4 pairing — same-process, same boot. 5
phenomenon strength — F28: closures occur in lineage venues at these
horizons. 6 sensitivity — 1 closure / 1 gen resolution. 7 rates —
single pair of arms; replication follows a supported verdict. 8
population closure (inspector inventory) — eviction classes:
inherited exempt (both arms), solvent composed evictable (warm only,
excluded from primary endpoint); EMA clocks: warm carries growth-aged
fire rates and balances — THIS IS PART OF THE CAPITAL, stated as
design; caps standard; primary comparison is inherited-only, so both
compared populations are identical at entry by construction.
"""

import json
import os
import pickle
import time

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED, ScanState
from replay_overnight_v3 import run_world_v3
from sov import RETHRESH_MIN_SAMPLES
from staged_fit_experiment import Accountant
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

B_SEEDS = ((97100, 4), (97101, 3))
GENS = 8
INHERITED_MAX = 32
CLOSURE_FLOOR = 2
SURV_VALIDATE = 0.5
SURV_INSUFFICIENT = 0.1
WEB_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'nursery_web_s98300.pkl')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'acceleration_card.json')


def scout(web, model):
    n = 0
    for si, (w_seed, tier) in enumerate(B_SEEDS):
        env = interpret(describe(TieredEnvironment(seed=w_seed,
                                                   tier=tier)))
        np.random.seed(76000 + si)
        env.rng = np.random.RandomState(76001 + si)
        rng = np.random.RandomState(76002 + si)
        bank = LiveReceptorBank()
        org = Organism()
        org.reset()
        for step in range(400):
            w = org.get_observation_window()
            act, _ = model.predict(w)
            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                act = np.zeros_like(act)
            elif r < EXPLORE_RATE:
                act = rng.randint(0, 2, size=len(act)).astype(act.dtype)
            obs, reward = org.step(act, env, step)
            rv = bank.compute(obs, act, None, reward)
            web.observe_activations(rv)
            n += 1
    return n


def run_arm(name, web, model, protocol):
    t0 = time.time()
    print('--- arm %s ---' % name)
    if protocol:
        snap = web.snapshot_activation_dist()
        n_scout = scout(web, model)
        n_adj = web.rethreshold(snap)
        print('  protocol: snapshot=%s scout=%d rethresholded=%d'
              % ('ok' if snap else 'none', n_scout, n_adj))
        assert snap is not None and n_scout >= RETHRESH_MIN_SAMPLES
    solvent0 = sorted(sid for sid, s in web.slots.items()
                      if sid > INHERITED_MAX
                      and s.state in ('open', 'closed'))
    closed_before = {sid for sid, s in web.slots.items()
                     if sid <= INHERITED_MAX and s.state == 'closed'}
    nm0 = sum(s.ledger.near_miss_seen for s in web.slots.values()
              if s.state in ('open', 'closed'))

    engine = build_engine()
    lived_log = []
    scan = ScanState()
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in B_SEEDS]
    closures = []
    surv_curve = []
    for k in range(GENS):
        if k > 0:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)
        bank = LiveReceptorBank()
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            run_world_v3(env, model, engine, web, bank, scan, k,
                         77000 + li * 1000 + k * 17,
                         lived_log, {}, (li, 0))
        web.anneal_all(web._global_step)
        newly = [sid for sid, s in web.slots.items()
                 if sid <= INHERITED_MAX and s.state == 'closed'
                 and sid not in closed_before]
        for sid in newly:
            closures.append({'slot': sid, 'gen': k + 1})
            closed_before.add(sid)
        alive = sum(1 for sid in solvent0
                    if web.slots.get(sid) is not None
                    and web.slots[sid].state in ('open', 'closed'))
        surv_curve.append(alive)
        print('  gen %d (%.1f min): inherited closures=%d solvent '
              'alive=%d/%d'
              % (k + 1, (time.time() - t0) / 60, len(closures),
                 alive, len(solvent0)))
    nm1 = sum(s.ledger.near_miss_seen for s in web.slots.values()
              if s.state in ('open', 'closed'))
    return {'closures': closures,
            'first_closure_gen': (closures[0]['gen'] if closures
                                  else None),
            'near_miss_mass': int(nm1 - nm0),
            'solvent_n0': len(solvent0),
            'solvent_curve': surv_curve}


def main():
    t0 = time.time()
    print('=== ACCELERATION CARD: warm capital vs cold boot ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)

    with open(WEB_PKL, 'rb') as f:
        warm_web = pickle.load(f)
    print('warm web loaded: %d active slots'
          % sum(1 for s in warm_web.slots.values()
                if s.state in ('open', 'closed')))
    warm = run_arm('WARM', warm_web, model, protocol=True)
    cold = run_arm('COLD', Accountant('COLD', staged=False,
                                      consume=False).web,
                   model, protocol=False)

    wc, cc = len(warm['closures']), len(cold['closures'])
    wl, cl = warm['first_closure_gen'], cold['first_closure_gen']
    if wc + cc < CLOSURE_FLOOR:
        verdict = ('UNTESTED (combined inherited closures %d < %d)'
                   % (wc + cc, CLOSURE_FLOOR))
    elif wc > cc and (cl is None or (wl is not None and wl <= cl)):
        verdict = ('ACCELERATION SUPPORTED: warm %d closures (first '
                   'gen %s) vs cold %d (first gen %s) — the map '
                   'retains value' % (wc, wl, cc, cl))
    elif cc >= wc and (wl is None or (cl is not None and cl <= wl)):
        verdict = ('NOT SUPPORTED: cold %d (gen %s) ties/beats warm '
                   '%d (gen %s) — no capital advantage at this '
                   'horizon' % (cc, cl, wc, wl))
    else:
        verdict = ('PARTIAL: warm %d/g%s vs cold %d/g%s (split '
                   'endpoints)' % (wc, wl, cc, cl))

    s0 = warm['solvent_n0']
    g2 = warm['solvent_curve'][1] / s0 if s0 else 0
    g8 = warm['solvent_curve'][-1] / s0 if s0 else 0
    if g2 >= SURV_VALIDATE:
        v2 = ('V2 MECHANISM VALIDATED: solvent survival %.2f at gen '
              '2, %.2f at gen 8 (insolvent baseline 0.00 by gen 1)'
              % (g2, g8))
    elif g2 < SURV_INSUFFICIENT:
        v2 = ('V2 INSUFFICIENT-DEEPER: solvent survival %.2f at gen '
              '2 — neither solvency nor recalibration explains '
              'transfer death' % g2)
    else:
        v2 = 'V2 PARTIAL: solvent survival %.2f at gen 2' % g2
    print('\nACCELERATION VERDICT: %s' % verdict)
    print('RETHRESHOLD V2 VERDICT: %s' % v2)
    print('near-miss mass: warm=%d cold=%d'
          % (warm['near_miss_mass'], cold['near_miss_mass']))

    out = {'warm': warm, 'cold': cold, 'verdict': verdict,
           'v2_verdict': v2,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

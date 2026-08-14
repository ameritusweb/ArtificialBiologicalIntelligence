"""ACCELERATION v2 — the yield endpoint in a venue that yields.
(Card locked at launch 2026-08-13; F46 impl. 2's pre-registered
successor; closes task #62's open endpoint.)

F46's venue was too quiet (1 combined closure in 8 plain gens). This
card moves the comparison to the regime where closures flow at
receipted rates (F28: ~0.222/gen under structure dosing): 12 gens,
structure events at gens 0/4/8 (S=2 per lineage), same held-out pair
(97100/97101). F47's simplification applied: the warm web crosses
NAKED — no scout, no rethreshold — because solvency alone carries
this crossing (17/17 twice).

ARMS: WARM = nursery pickle (33 inherited + 17 solvent composed).
COLD = standard boot. Same model, same worlds, same dose schedule,
sequential in one process.

ENDPOINTS: PRIMARY (class-closed: identical inherited populations at
entry; composed excluded): inherited closures over 12 gens + first-
closure latency. SECONDARY (descriptive): solvent survival under
dosing (F47's crossing was undosed — this is the solvent cohort's
first STRAINED crossing), near-miss mass.

VERDICTS: UNTESTED if combined inherited closures < 3 (the dosed
venue owes ~2.7/arm by F28; below 3 combined means the venue
underdelivered its own receipt). ACCELERATION SUPPORTED iff WARM
strictly exceeds COLD on closures AND ties-or-beats on latency.
NOT SUPPORTED iff COLD ties-or-beats on both. PARTIAL between.

C20 (eight): 1 standing machinery (nursery pickle receipted, dosed
lineage receipted). 2 closures admitted by the corridor gate, not by
anything the arms differ in. 3 dose schedule identical across arms,
pre-registered. 4 same-process shared boot. 5 F28's 0.222/gen. 6
resolution 1 closure / 1 gen. 7 single pair; replication follows
support. 8 inspector inventory: inherited exempt both arms; solvent
composed evictable, warm-only, excluded from primary; warm's aged
EMAs are part of the capital (stated); dose rng shared.
"""

import json
import os
import pickle
import time

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED, ScanState
from replay_overnight_v3 import run_world_v3
from law_structure import mutate_law_structure
from staged_fit_experiment import Accountant
from train import generate_training_data, train_model

B_SEEDS = ((97100, 4), (97101, 3))
GENS = 12
EVENTS = (0, 4, 8)
S = 2
INHERITED_MAX = 32
CLOSURE_FLOOR = 3
WEB_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'nursery_web_s98300.pkl')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'acceleration_v2.json')


def run_arm(name, web, model):
    t0 = time.time()
    print('--- arm %s ---' % name)
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
    rng = np.random.RandomState(6500)
    closures, surv_curve = [], []
    epoch = 0
    for k in range(GENS):
        if k in EVENTS:
            for li in range(len(lineages)):
                for _ in range(S):
                    lineages[li], d = mutate_law_structure(
                        lineages[li], rng)
            epoch += 1
        if k > 0:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)
        bank = LiveReceptorBank()
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            run_world_v3(env, model, engine, web, bank, scan, k,
                         78000 + li * 1000 + k * 17,
                         lived_log, {}, (li, epoch))
        web.anneal_all(web._global_step)
        for sid, s in web.slots.items():
            if sid <= INHERITED_MAX and s.state == 'closed' \
                    and sid not in closed_before:
                closures.append({'slot': sid, 'gen': k + 1})
                closed_before.add(sid)
        alive = sum(1 for sid in solvent0
                    if web.slots.get(sid) is not None
                    and web.slots[sid].state in ('open', 'closed'))
        surv_curve.append(alive)
        print('  gen %d (%.1f min): closures=%d solvent=%d/%d'
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
    print('=== ACCELERATION v2: dosed venue, naked crossing ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    with open(WEB_PKL, 'rb') as f:
        warm_web = pickle.load(f)
    warm = run_arm('WARM', warm_web, model)
    cold = run_arm('COLD', Accountant('COLD', staged=False,
                                      consume=False).web, model)

    wc, cc = len(warm['closures']), len(cold['closures'])
    wl, cl = warm['first_closure_gen'], cold['first_closure_gen']
    if wc + cc < CLOSURE_FLOOR:
        verdict = ('UNTESTED (combined closures %d < %d — dosed '
                   'venue underdelivered its own receipt)'
                   % (wc + cc, CLOSURE_FLOOR))
    elif wc > cc and (cl is None or (wl is not None and wl <= cl)):
        verdict = ('ACCELERATION SUPPORTED: warm %d closures (first '
                   'gen %s) vs cold %d (first gen %s)'
                   % (wc, wl, cc, cl))
    elif cc >= wc and (wl is None or (cl is not None and cl <= wl)):
        verdict = ('NOT SUPPORTED: cold %d (gen %s) ties/beats warm '
                   '%d (gen %s)' % (cc, cl, wc, wl))
    else:
        verdict = ('PARTIAL: warm %d/g%s vs cold %d/g%s'
                   % (wc, wl, cc, cl))
    s0 = warm['solvent_n0']
    print('\nACCELERATION v2 VERDICT: %s' % verdict)
    print('solvent under dosing: %s (of %d)'
          % (warm['solvent_curve'], s0))
    print('near-miss mass: warm=%d cold=%d'
          % (warm['near_miss_mass'], cold['near_miss_mass']))
    out = {'warm': warm, 'cold': cold, 'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

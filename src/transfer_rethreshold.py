"""TRANSFER RETHRESHOLD — does the C12 repair let vocabulary survive
the domain transition? (Mechanism validation, card locked at launch
2026-08-13; F43's closing run.)

BASELINE (the control arm, by receipt): the extinction autopsy —
identical growth seeds, identical phase-B schedule, no rethreshold —
killed 72/72 composed slots, all saturation-class, inside one
generation (results/extinction_autopsy.json). Cross-run comparison is
acceptable here because the predicted effect is categorical (0%
survival vs >= 50%), far above encoder-nondeterminism noise; stated
honestly per check 4.

TREATMENT: phase A growth (same seeds) -> snapshot_activation_dist()
(closes the old book) -> SCOUT PASS: one episode per lineage world
observed WITHOUT fitting (observe_activations — no receipts, no
economy; look before you fire) -> rethreshold(snapshot) (open slots
only) -> phase B exactly as the autopsy (8 gens, structure events at
0/4).

VERDICTS (fixed): MECHANISM VALIDATED iff composed survival >= 50%
at gen 2 AND >= 25% at gen 8 AND surviving-composed median fire_rate
< 0.75 (alive AND selective — survival by recalibration, not by
luck). INSUFFICIENT: survival < 10% at gen 2 (the defect is deeper
than thresholds). PARTIAL between. UNTESTED: composed < 50 at
transition or scout samples < RETHRESH_MIN_SAMPLES.

C20 (eight): 1 domain — standing machinery. 2 endpoint independence
— rethreshold writes thresholds; the endpoint is SURVIVAL under the
untouched rent economy plus the fire-rate band (the world's own
verdict on selectivity). 3 exogeneity — the mechanism is the only
change vs the autopsy. 4 pairing — same seeds, cross-run baseline
by receipt (stated). 5 phenomenon strength — the autopsy's 72/72 is
the phenomenon at maximum strength. 6 sensitivity — categorical
effect predicted; resolution is one slot. 7 rates — single
mechanism-validation run; verdict-grade replication follows if
validated. 8 population closure — the tracked population is fixed
at transition; survival is the endpoint, not an exit path; classes
closed (composed only tracked; trunk reported separately).
"""

import json
import os
import time

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED, ScanState
from replay_overnight_v3 import run_world_v3
from law_structure import mutate_law_structure
from sov import RETHRESH_MIN_SAMPLES
from staged_fit_experiment import Accountant, run_worlds
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

GROW_WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
LINEAGE_SEEDS = ((97000, 4), (97001, 3))
PB_GENS = 8
PB_EVENTS = (0, 4)
S = 2
INHERITED_MAX = 32
COMPOSED_FLOOR = 50
VALIDATE_G2 = 0.5
VALIDATE_G8 = 0.25
INSUFFICIENT_G2 = 0.1
FIRE_BAND = 0.75
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'transfer_rethreshold.json')


def scout(web, engine, model):
    """Observe one episode per lineage world without fitting."""
    n = 0
    for si, (w_seed, tier) in enumerate(LINEAGE_SEEDS):
        corpus = describe(TieredEnvironment(seed=w_seed, tier=tier))
        env = interpret(corpus)
        np.random.seed(74000 + si)
        env.rng = np.random.RandomState(74001 + si)
        rng = np.random.RandomState(74002 + si)
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


def main():
    t0 = time.time()
    print('=== TRANSFER RETHRESHOLD: the C12 repair under fire ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('RETHR', staged=False, consume=False)
    print('phase A: growth (24 worlds)...')
    run_worlds(GROW_WORLDS, [arm], engine, model, 0)
    web = arm.web
    composed0 = sorted(sid for sid, s in web.slots.items()
                       if sid > INHERITED_MAX
                       and s.state in ('open', 'closed'))
    print('  composed at transition: %d' % len(composed0))

    snap = web.snapshot_activation_dist()
    print('  old-book snapshot: %s samples'
          % (None if snap is None else snap['n']))
    n_scout = scout(web, engine, model)
    print('  scout: %d observations (no receipts)' % n_scout)
    n_adj = web.rethreshold(snap)
    print('  rethreshold: %d slots adjusted' % n_adj)

    print('phase B: lineage (8 gens, autopsy schedule)...')
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    epoch = 0
    scan = ScanState()
    rng = np.random.RandomState(6400)
    alive_curve = []
    for k in range(PB_GENS):
        if k in PB_EVENTS:
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
                         75000 + li * 1000 + k * 17,
                         lived_log, {}, (li, epoch))
        web.anneal_all(web._global_step)
        alive = sum(1 for sid in composed0
                    if web.slots.get(sid) is not None
                    and web.slots[sid].state in ('open', 'closed'))
        alive_curve.append(alive)
        print('  gen %d (%.1f min): composed alive %d/%d'
              % (k + 1, (time.time() - t0) / 60, alive,
                 len(composed0)))

    surv_fr = [float(web.slots[sid].ledger.fire_rate)
               for sid in composed0
               if web.slots.get(sid) is not None
               and web.slots[sid].state in ('open', 'closed')]
    frac_g2 = alive_curve[1] / len(composed0) if composed0 else 0
    frac_g8 = alive_curve[-1] / len(composed0) if composed0 else 0
    med_fr = float(np.median(surv_fr)) if surv_fr else None

    if len(composed0) < COMPOSED_FLOOR or n_scout < RETHRESH_MIN_SAMPLES:
        verdict = ('UNTESTED (composed=%d/%d scout=%d/%d)'
                   % (len(composed0), COMPOSED_FLOOR, n_scout,
                      RETHRESH_MIN_SAMPLES))
    elif frac_g2 >= VALIDATE_G2 and frac_g8 >= VALIDATE_G8 \
            and med_fr is not None and med_fr < FIRE_BAND:
        verdict = ('MECHANISM VALIDATED: survival %.2f at gen 2 / '
                   '%.2f at gen 8 vs baseline 0.00, survivor median '
                   'fire rate %.3f (selective) — thresholds as '
                   'quantile commitments carry vocabulary across the '
                   'domain boundary' % (frac_g2, frac_g8, med_fr))
    elif frac_g2 < INSUFFICIENT_G2:
        verdict = ('INSUFFICIENT: survival %.2f at gen 2 — the '
                   'transfer defect is deeper than thresholds'
                   % frac_g2)
    else:
        verdict = ('PARTIAL: g2=%.2f g8=%.2f median_fr=%s'
                   % (frac_g2, frac_g8, med_fr))
    print('\nTRANSFER RETHRESHOLD VERDICT: %s' % verdict)

    out = {'composed_at_transition': len(composed0),
           'slots_rethresholded': n_adj,
           'scout_samples': n_scout,
           'alive_curve': alive_curve,
           'survival_g2': frac_g2, 'survival_g8': frac_g8,
           'survivor_fire_rates': [round(f, 4) for f in surv_fr],
           'baseline': 'extinction_autopsy.json (0/72 by gen 1)',
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

"""EXTINCTION AUTOPSY — which death did the composed vocabulary die?
(Instrument run, 2026-08-13; F41 impl. 5's gate. No claim verdict —
a measurement that decides which mechanism the receipts demand.)

QUESTION: the domain transition killed ~31/33 composed slots (F41).
Two deaths are possible and they demand OPPOSITE mechanisms:
  COLLAPSE death (fire_rate -> 0): the slot's co-fire context
  vanished with its world — ORPHANED. Demands hibernation (dormancy
  extended to open slots) — killing it destroys domain capital the
  organism could reuse on return.
  SATURATION death (fire_rate -> 1): the slot fires on everything in
  the new world — VACUOUS there. Demands exactly what it got: rent
  death. The economy was just.
fire_rate is an EMA frozen at archaization — the terminal value IS
the signature. Classification bands (pre-registered): collapse
< 0.25; saturation > 0.75; mixed between.

DESIGN: phase A (24 worlds, P106 growth, same seeds) -> phase B
(8 lineage gens, structure events at 0/4 — v5 showed most deaths
early). Export for every composed slot: state, terminal fire_rate,
rent_balance, fit_count, lifetime mass; plus inherited-slot fire
rates (context for the exemption-lift audit, which reads this file).

READ-OUT (fixed): the mechanism gate — hibernation is DEMANDED iff
>= 60% of dead composed slots are collapse-class; rent death stands
iff >= 60% are saturation-class; a mixed population demands the
fire-rate fork itself (both paths, keyed by the signature).
"""

import json
import os
import time

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED, ScanState
from replay_overnight_v3 import run_world_v3
from law_structure import mutate_law_structure
from staged_fit_experiment import Accountant, run_worlds
from train import generate_training_data, train_model

GROW_WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
LINEAGE_SEEDS = ((97000, 4), (97001, 3))
PB_GENS = 8
PB_EVENTS = (0, 4)
S = 2
INHERITED_MAX = 32
COLLAPSE_BAND = 0.25
SATURATION_BAND = 0.75
GATE = 0.6
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'extinction_autopsy.json')


def main():
    t0 = time.time()
    print('=== EXTINCTION AUTOPSY: which death? ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('AUTOPSY', staged=False, consume=False)
    print('phase A: growth (24 worlds)...')
    run_worlds(GROW_WORLDS, [arm], engine, model, 0)
    web = arm.web
    composed0 = sorted(sid for sid, s in web.slots.items()
                       if sid > INHERITED_MAX
                       and s.state in ('open', 'closed'))
    print('  composed population at transition: %d' % len(composed0))

    print('phase B: lineage transition (8 gens)...')
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    epoch = 0
    scan = ScanState()
    rng = np.random.RandomState(6400)
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
        print('  gen %d (%.1f min): composed alive %d/%d'
              % (k + 1, (time.time() - t0) / 60, alive,
                 len(composed0)))

    rows = []
    counts = {'collapse': 0, 'saturation': 0, 'mixed': 0}
    for sid in composed0:
        s = web.slots.get(sid)
        if s is None:
            continue
        dead = s.state not in ('open', 'closed')
        fr = float(s.ledger.fire_rate)
        row = {'slot': sid, 'dead': dead, 'state': s.state,
               'fire_rate': round(fr, 4),
               'rent_balance': round(float(s.ledger.rent_balance), 3),
               'fit_count': s.ledger.fit_count,
               'mass': round(float(s.ledger.mass), 2)}
        if dead:
            cls = ('collapse' if fr < COLLAPSE_BAND else
                   'saturation' if fr > SATURATION_BAND else 'mixed')
            row['death_class'] = cls
            counts[cls] += 1
        rows.append(row)
    inherited_fr = {sid: round(float(web.slots[sid].ledger.fire_rate),
                               4)
                    for sid in web.slots
                    if sid <= INHERITED_MAX
                    and web.slots[sid].state in ('open', 'closed')}
    dead_total = sum(counts.values())
    print('  deaths: %d  classes: %s' % (dead_total, counts))

    if dead_total == 0:
        gate = 'NO DEATHS (transition benign this realization)'
    elif counts['collapse'] / dead_total >= GATE:
        gate = ('HIBERNATION DEMANDED: %d/%d collapse-class — the '
                'dead were ORPHANED (context vanished), not vacuous; '
                'open-slot dormancy on fire-rate collapse is the '
                'receipted mechanism' % (counts['collapse'],
                                         dead_total))
    elif counts['saturation'] / dead_total >= GATE:
        gate = ('RENT DEATH STANDS: %d/%d saturation-class — the '
                'dead were VACUOUS in the new world; the economy was '
                'just' % (counts['saturation'], dead_total))
    else:
        gate = ('MIXED POPULATION (%s): the fire-rate fork itself is '
                'demanded — hibernate collapse, kill saturation'
                % counts)
    print('\nAUTOPSY GATE: %s' % gate)

    out = {'composed_rows': rows, 'death_classes': counts,
           'inherited_fire_rates': inherited_fr,
           'gate': gate,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

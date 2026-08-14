"""EXEMPTION-LIFT AUDIT — does the inherited trunk earn its keep?
(Instrument run, 2026-08-13; F41 impl. 7 / task #42's economic arm.
Card locked at launch. Core sov.py untouched: the lift is a
post-anneal sandbox step using the public Archaize operator.)

BACKGROUND: inherited slots are constitutionally exempt from rent
eviction, and the code itself carries the caveat the receipts have
now made load-bearing: "Their survival is therefore NOT evidence of
earning" (sov.py anneal_all). F41 showed the exemption is an
untested immortality privilege that masks the trunk's own transfer
test. This audit measures, in a disposable sandbox, which trunk
slots SELF-FUND under the standard economy when the privilege is
lifted.

DESIGN: phase A (24-world growth, autopsy seeds — inherited balances
enter phase B carrying their earned history; no anneal in the
sandbox phase means credits-only) -> phase B (8 lineage gens,
structure events at 0/4, anneal per gen). LIFT: after each
anneal_all, any INHERITED open slot with rent_balance below the
eviction floor is archaized via web.archaize — exactly the fate a
composed slot meets. Export per inherited slot: died_gen (or
survived), terminal fire_rate, rent_balance trajectory endpoints,
fit_count.

GATES (fixed, descriptive-plus-flag): TRUNK-EARNS iff >= 80% of
inherited slots survive the lifted economy through phase B.
DECORATIVE-FLAG iff < 50% survive — the dead list goes to the
genome audit (#42) as rent-found dead weight. Between: MIXED, the
per-slot list is the deliverable either way. No registry claim is
billed from a single realization; this is the instrument's first
reading, and the design decision (keep/lift/modify the exemption)
remains the user's with this data in hand.
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
from sov import RENT_EVICTION_FLOOR
from staged_fit_experiment import Accountant, run_worlds
from train import generate_training_data, train_model

GROW_WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
LINEAGE_SEEDS = ((97000, 4), (97001, 3))
PB_GENS = 8
PB_EVENTS = (0, 4)
S = 2
INHERITED_MAX = 32
EARN_GATE = 0.8
DECOR_GATE = 0.5
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'exemption_audit.json')


def main():
    t0 = time.time()
    print('=== EXEMPTION-LIFT AUDIT: does the trunk earn? ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('EXAUDIT', staged=False, consume=False)
    print('phase A: growth (24 worlds)...')
    run_worlds(GROW_WORLDS, [arm], engine, model, 0)
    web = arm.web
    inherited = sorted(sid for sid, s in web.slots.items()
                       if sid <= INHERITED_MAX
                       and s.state in ('open', 'closed'))
    entry_balance = {sid: float(web.slots[sid].ledger.rent_balance)
                     for sid in inherited}
    print('  inherited population: %d; entry balances '
          'min/median/max = %.2f / %.2f / %.2f'
          % (len(inherited), min(entry_balance.values()),
             float(np.median(list(entry_balance.values()))),
             max(entry_balance.values())))

    print('phase B: lineage, exemption LIFTED (8 gens)...')
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    epoch = 0
    scan = ScanState()
    rng = np.random.RandomState(6400)
    died_gen = {}
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
        # THE LIFT: inherited open slots meet the same floor as
        # everyone else (public operator; core rule untouched).
        for sid in inherited:
            if sid in died_gen:
                continue
            s = web.slots.get(sid)
            if s is None or s.state != 'open':
                continue
            if s.ledger.rent_balance < RENT_EVICTION_FLOOR:
                web.archaize(sid)
                died_gen[sid] = k + 1
        alive = len(inherited) - len(died_gen)
        print('  gen %d (%.1f min): trunk alive %d/%d'
              % (k + 1, (time.time() - t0) / 60, alive,
                 len(inherited)))

    rows = []
    for sid in inherited:
        s = web.slots.get(sid)
        rows.append({'slot': sid,
                     'name': s.name if s is not None else '?',
                     'died_gen': died_gen.get(sid),
                     'state': s.state if s is not None else 'gone',
                     'fire_rate': (round(float(s.ledger.fire_rate), 4)
                                   if s is not None else None),
                     'entry_balance': round(entry_balance[sid], 3),
                     'final_balance': (round(float(
                         s.ledger.rent_balance), 3)
                         if s is not None else None),
                     'fit_count': (s.ledger.fit_count
                                   if s is not None else None)})
    survived = sum(1 for r in rows if r['died_gen'] is None)
    frac = survived / len(rows) if rows else 0.0
    dead_names = [r['name'] for r in rows if r['died_gen'] is not None]
    if frac >= EARN_GATE:
        gate = ('TRUNK-EARNS: %d/%d (%.2f) survive the lifted '
                'economy — inheritance is economically self-funding '
                'in this venue' % (survived, len(rows), frac))
    elif frac < DECOR_GATE:
        gate = ('DECORATIVE-FLAG: only %d/%d (%.2f) self-fund; dead '
                'list -> genome audit #42: %s'
                % (survived, len(rows), frac, dead_names))
    else:
        gate = ('MIXED: %d/%d (%.2f) self-fund; dead list: %s'
                % (survived, len(rows), frac, dead_names))
    print('\nEXEMPTION AUDIT: %s' % gate)

    out = {'rows': rows, 'survived_frac': frac, 'gate': gate,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

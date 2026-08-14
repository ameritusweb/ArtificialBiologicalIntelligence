"""Closability probe (F24 impl. 3 / impl. 9): measure whether a world
lineage affords closure at the current learner's capacity.

The first instance of the inward-pointed stakeholder instrument: a
property OF A WORLD measured with organism telemetry. Persistence-only
regime (no scans, no court, no shifts): 8 generations, lived-log
encoder rebuilds + rebase, count closure attempts and track the minimum
open-slot radius trend.

Classification (stated in advance):
  CLOSABLE      — >= 1 closure attempt (closed or reopened) in 8 gens
  MARGINAL      — no attempt, but min radius < 0.10 at any generation
                  (within 2x of the closure threshold: the frontier)
  UNCLOSABLE@8  — neither. (A property of world x capacity x horizon,
                  not of the world alone.)

Usage: python closability_probe.py <name> <seedA> <tierA> <seedB> <tierB>
Worlds are CORPUS lineages (describe -> interpret), matching the v4/v5
world source — check 5 receipts are distribution-bound (C20 check 7).
"""

import json
import os
import sys
import time

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import generate_training_data, train_model

from replay_overnight import build_engine, calibrate_taus, ScanState, \
    BOOT_SEED
from replay_overnight_v3 import run_world_v3

GENS = 8


def main(name, sa, ta, sb, tb):
    t0 = time.time()
    print(f'=== CLOSABILITY PROBE {name}: ({sa},t{ta})+({sb},t{tb}) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id=f'PROBE_{name}')
    web.populate_from_families()
    scan = ScanState()
    corpora = [describe(TieredEnvironment(seed=sa, tier=ta)),
               describe(TieredEnvironment(seed=sb, tier=tb))]

    gens_out = []
    for g in range(GENS):
        bank = LiveReceptorBank()
        for li, corpus in enumerate(corpora):
            env = interpret(corpus)
            run_world_v3(env, model, engine, web, bank, scan, g,
                         77000 + li * 1000 + g * 17, lived_log, {},
                         (li, 0))
        web.anneal_all(web._global_step)
        stats = web.get_stats()
        attempts = sum(s.ledger.reopen_count
                       for s in web.slots.values()) + stats['closed']
        radii = sorted(round(s.geometry.radius, 4)
                       for s in web.slots.values()
                       if s.state == 'open' and s.ledger.fit_count >= 3
                       and np.isfinite(s.geometry.radius))
        closed_names = sorted(s.name for s in web.slots.values()
                              if s.state == 'closed')
        gens_out.append({'gen': g, 'closed': stats['closed'],
                         'closed_names': closed_names,
                         'attempts': attempts, 'min_radii': radii[:3]})
        print(f'  gen {g + 1}/{GENS}: closed={stats["closed"]} '
              f'attempts={attempts} minR={radii[:2]} '
              f'({(time.time() - t0) / 60:.1f} min)')
        if g < GENS - 1:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)

    attempts = gens_out[-1]['attempts']
    min_r = min((g['min_radii'][0] for g in gens_out if g['min_radii']),
                default=float('inf'))
    if attempts >= 1:
        cls = 'CLOSABLE'
    elif min_r < 0.10:
        cls = 'MARGINAL'
    else:
        cls = 'UNCLOSABLE@8'
    first_closed = next((g['closed_names'][0] for g in gens_out
                         if g['closed_names']), None)
    out = {'name': name, 'worlds': [[sa, ta], [sb, tb]],
           'classification': cls, 'attempts': attempts,
           'first_closed': first_closed,
           'min_radius_seen': min_r, 'gens': gens_out,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    print(f'first_closed: {first_closed}')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'closability_{name}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'PROBE {name}: {cls} (attempts={attempts}, '
          f'minR={min_r}) saved {path}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
         int(sys.argv[4]), int(sys.argv[5]))

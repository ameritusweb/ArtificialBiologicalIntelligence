"""THE SOLVENT NURSERY — economy-on growth venue, first characterization
+ the P106 economy-on replication. (Card locked at launch 2026-08-13;
F44 impl. 2's growth-venue-parity law enacted; task #66.)

WHAT CHANGES: P106's exact growth (24 worlds, same seeds 98300+i,
compose cap 3/boundary, render at every boundary) with ONE addition —
web.anneal_all(global_step) at every world boundary. Rent is charged,
evictions are live, dormancy checks run: composed slots must EARN their
existence during growth. (Venue novelty flagged per check 1: anneal_all
has never run inside the staged harness; it is a pure economy operation
with no encoder interaction.)

TWO DELIVERABLES, ONE RUN:
  A. NURSERY CHARACTERIZATION (descriptive, the venue's birth
     certificate): composed births vs survivors per boundary, eviction
     counts, survivor fire-rate distribution (the solvency spectrum).
     Expectation from F44/F31: born-vacuous composes die within ~1-2
     boundaries of birth; the standing composed population is small and
     SELECTIVE — the first vocabulary in the program's history that
     earned its existence during growth.
  B. P106 ECONOMY-ON REPLICATION (pre-registered, F44 impl. 2): the
     pigeonhole claim under EARNED crowding. Same endpoint as P106:
     Spearman(per-capita composed-composed collision rate vs composed
     population) over qualifying checkpoints (n_composed >= 4).
     BRANCH 1 (floors met: >= 8 qualifying checkpoints AND final
     composed >= 12): bill the Spearman verdict with P106's gates
     (>= +0.5 & final pairs >= 5 SUPPORTED; <= 0 NOT SUPPORTED).
     BRANCH 2 (floors NOT met — the economy culls too hard): verdict
     UNTESTED-BY-CULLING, and the headline number becomes the SOLVENT
     CROWDING FRACTION: solvent collision pairs / F40's 212 — the
     quantified caveat on F40's economy-free crowd.

ARTIFACT: the final web is pickled to data/nursery_web_s98300.pkl —
the acceleration card (#62) warm-starts from it; rethreshold v2
transfers it. Build once, bill three times.

C20 (eight): 1 domain — staged harness + anneal (novelty flagged
above). 2 endpoint independence — render is a pure read
(hash-asserted); anneal is the treatment's economy, not the render's.
3 exogeneity — n/a, measurement curve (P106's caveat inherited: curve
shape, not causal claim). 4 pairing — n/a. 5 phenomenon strength —
compose events receipted at 3/boundary (F40); vacuous death under
anneal receipted at <= 2 gens (F44/F31). 6 sensitivity — 1 pair / 1
slot resolution. 7 rates — compose cap deterministic; eviction rate is
the measured quantity. 8 population closure (inspector inventory) —
eviction classes: inherited EXEMPT by rule (immortal; excluded from
the collision curve exactly as in P106), composed evictable; EMA
clocks: fire_rate EMA continuous across worlds (same as P106); caps:
compose 3/boundary, ACT_RING passive; the measured population is the
SOLVENT active set at each checkpoint — which is the venue's entire
point, stated as design, not discovered post-run.
"""

import json
import os
import pickle
import time

import numpy as np

from p105_experiment import band_profile, state_hash
from p106_maturity import render_checkpoint, spearman
from replay_overnight import build_engine, BOOT_SEED
from staged_fit_experiment import Accountant, run_worlds
from train import generate_training_data, train_model

WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
INHERITED_MAX = 32
CHECKPOINT_FLOOR = 8
QUALIFY_N = 4
FINAL_COMPOSED_FLOOR = 12
FINAL_PAIRS_FLOOR = 5
SPEARMAN_GATE = 0.5
F40_PAIRS = 212
WEB_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'nursery_web_s98300.pkl')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p_nursery.json')


def main():
    t0 = time.time()
    print('=== SOLVENT NURSERY: economy-on growth (24 worlds) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('NURSERY', staged=False, consume=False)
    curve = []
    births_prev = 0
    c = 0
    for i, w in enumerate(WORLDS):
        c = run_worlds([w], [arm], engine, model, c)
        web = arm.web
        # THE VENUE'S ONE ADDITION: the economy runs during growth.
        web.anneal_all(web._global_step)
        cp = render_checkpoint(web)
        cp['world'] = i + 1
        all_composed_ever = sum(1 for sid in web.slots
                                if sid > INHERITED_MAX)
        cp['composed_births_cum'] = all_composed_ever
        cp['composed_evicted_cum'] = sum(
            1 for sid, s in web.slots.items()
            if sid > INHERITED_MAX and s.state not in ('open', 'closed'))
        curve.append(cp)
        births = all_composed_ever - births_prev
        births_prev = all_composed_ever
        print('  w%02d: born+%d alive=%d evicted_cum=%d cc_pairs=%d '
              'profiles=%d'
              % (i + 1, births, cp['n_composed'],
                 cp['composed_evicted_cum'], cp['pairs_composed'],
                 cp['distinct_profiles']))

    web = arm.web
    solvent_fr = [round(float(s.ledger.fire_rate), 4)
                  for sid, s in web.slots.items()
                  if sid > INHERITED_MAX
                  and s.state in ('open', 'closed')]
    final = curve[-1]

    qual = [cp for cp in curve if cp['n_composed'] >= QUALIFY_N]
    xs = [cp['n_composed'] for cp in qual]
    ys = [cp['pairs_composed'] / cp['n_composed'] for cp in qual]
    rho = spearman(xs, ys) if len(qual) >= 3 else None
    solvent_fraction = final['pairs_composed'] / float(F40_PAIRS)

    if len(qual) >= CHECKPOINT_FLOOR \
            and final['n_composed'] >= FINAL_COMPOSED_FLOOR:
        if rho is not None and rho >= SPEARMAN_GATE \
                and final['pairs_composed'] >= FINAL_PAIRS_FLOOR:
            verdict = ('SUPPORTED (economy-on): pigeonhole binds among '
                       'EARNED vocabulary (rho=%.2f, %d pairs among %d '
                       'solvent composed)'
                       % (rho, final['pairs_composed'],
                          final['n_composed']))
        elif rho is not None and rho <= 0:
            verdict = ('NOT SUPPORTED (economy-on): per-capita '
                       'flat/declining (rho=%.2f) — earned vocabulary '
                       'does not crowd; F40 crowding was unbilled mass'
                       % rho)
        else:
            verdict = ('PARTIAL (economy-on): rho=%s pairs=%d'
                       % (rho, final['pairs_composed']))
    else:
        verdict = ('UNTESTED-BY-CULLING: qualifying=%d/%d final '
                   'composed=%d/%d — the economy holds the solvent '
                   'population below the crowding floors; solvent '
                   'crowding fraction = %d/%d = %.3f of F40'
                   % (len(qual), CHECKPOINT_FLOOR,
                      final['n_composed'], FINAL_COMPOSED_FLOOR,
                      final['pairs_composed'], F40_PAIRS,
                      solvent_fraction))
    print('\nNURSERY / P106-ECON VERDICT: %s' % verdict)
    print('  solvent composed fire rates: %s'
          % (sorted(solvent_fr) if solvent_fr else 'none'))

    os.makedirs(os.path.dirname(WEB_PKL), exist_ok=True)
    with open(WEB_PKL, 'wb') as f:
        pickle.dump(web, f)
    print('  web pickled -> %s' % WEB_PKL)

    out = {'curve': curve,
           'solvent_fire_rates': sorted(solvent_fr),
           'spearman_percapita': rho,
           'solvent_crowding_fraction_of_F40': round(solvent_fraction, 4),
           'verdict': verdict,
           'web_artifact': WEB_PKL,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

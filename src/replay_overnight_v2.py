"""Replay phase REGIME v2 — persistent worlds, pre-registered shifts.

THE CARD (locked at launch; C20 applies). F21 isolated why v1's three
churn-dependent endpoints came back UNTESTED: churn is a function of
WORLD PERSISTENCE, not exposure volume (v1: 4.47M fits, ONE closure
attempt vs the seed-44 baseline's 195). Closure needs a world that
stays; reopening needs a world that then moves. v2 supplies both, at
session scale, with everything else carried over from v1 unchanged
(`replay_overnight.py` — machinery imported, not copied).

REGIME (fixed):
  24 generations in THREE REGIMES of 8 (shifts at g=8 and g=16,
  pre-registered). Within a regime the SAME two worlds (one tier-4, one
  tier-3; seeds 95000+1000*r+i) are RE-LIVED every generation — same
  layout and structure, generation-varied episode randomness (the world
  stays; the life varies). At a shift, both world seeds change: slot
  statistics genuinely move, the 404 windows get their chance. Encoder
  rebuild each generation on the run's own lived log + rebase (v1's
  proven mechanism); scans before rebase; court junction law across
  generations (within a regime, ratification validates on fresh
  EPISODES of the held-out world; across a shift, on genuinely new
  worlds — stated, not hidden).

ENDPOINTS AND VERDICT RULES: identical to v1's card verbatim
(replay_overnight.py docstring + replay_phase_requirements.md):
split-reduces-churn (R<=0.5 / >=0.8, exposure floor 0.5x), P60/P69
(already SUPPORTED in v1 — re-reported, not re-billed), F20 relational
closure (first surviving closure operator-born in {Compose,
Differentiate, Individuate, Abstract} vs inherited; Unify-born closures
REPORTED separately, unbilled — the set was locked before Unify fired
in v1), demand alignment (same floors). NEW OBSERVATIONAL EXPORT
(F21 impl. 4, reported not billed): churn of Unify-born slots — were
the 16-per-run geometry-identity merges synonymy or conflation?

C20 PRE-FLIGHT: checks 1-4 and 6 carry over from v1 unchanged (same
machinery, same endpoints). Check 5 (phenomenon strength): smoke = 6
generations INSIDE regime 0 (persistence only, no shift) must show
closure attempts accumulating (>= 1 attempt, and closed slots forming);
else the regime is iterated before launch.
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from environment_tiers import TieredEnvironment
from environment_descriptive import classify_behavior_from_features
from environment_lexical import (Lexicon, evolve_one_generation,
                                 ratify_pending, append_ledger)
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import generate_training_data, train_model

from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              run_world, replay_scans, context_vocab,
                              verdicts, BOOT_SEED, GENERATIONS,
                              CHURN_EVENTS_FLOOR, PROPOSALS_FLOOR)

REGIME_LEN = 8                 # shifts at g=8 and g=16 (pre-registered)
WORLDS_PER_GEN = 2
TIERS = (4, 3)
V2_SEED_BASE = 95000

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'replay_phase_v2.json')


def world_schedule(g):
    """Same seeds all regime long; both change at each shift."""
    regime = g // REGIME_LEN
    return [(V2_SEED_BASE + 1000 * regime + i, TIERS[i])
            for i in range(WORLDS_PER_GEN)], regime


def main(smoke=False):
    t0 = time.time()
    gens = 6 if smoke else GENERATIONS
    print(f'=== REPLAY PHASE v2 {"SMOKE" if smoke else "FULL"}: '
          f'{gens} generations, regimes of {REGIME_LEN}, '
          f'shifts at {[REGIME_LEN, 2 * REGIME_LEN]} ===')

    print('boot: policy model + engine...')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)
    print(f'frozen taus: rest={tau_rest:.4f} d={tau_d:.4f}')

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id='REPLAY2')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None

    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'generations': [],
              'ratified': [], 'conservation_violations': [],
              'config': {'generations': gens, 'regime_len': REGIME_LEN,
                         'worlds_per_gen': WORLDS_PER_GEN,
                         'seed_base': V2_SEED_BASE}}

    for g in range(gens):
        worlds, regime = world_schedule(g)
        bank = LiveReceptorBank()
        gen_windows = []
        for i, (seed, tier) in enumerate(worlds):
            env = TieredEnvironment(seed=seed, tier=tier)
            # the world stays; the life varies: episode randomness is
            # generation-dependent, layout/structure fixed by seed
            w = run_world(env, model, engine, web, bank, scan, g,
                          seed * 7 + g * 17, lived_log)
            gen_windows.append(w)

        web.anneal_all(web._global_step)
        replay_scans(web, scan, g, record)

        def as_court(ws):
            return [(p, classify_behavior_from_features(f, tau_rest, tau_d))
                    for p, f in ws]
        if pending_prev:
            r_train = [w for ws in gen_windows[:-1]
                       for w in as_court(ws)][:2000]
            r_val = as_court(gen_windows[-1])
            lexicon, ratified = ratify_pending(lexicon, pending_prev,
                                               r_train, r_val, ledger)
            for kind, word, child in ratified:
                record['ratified'].append({'kind': kind, 'word': word,
                                           'child': child, 'gen': g})
        train_c = [w for ws in gen_windows[:-1] for w in as_court(ws)]
        val_c = as_court(gen_windows[-1])
        lexicon, moves, pending = evolve_one_generation(
            lexicon, train_c, val_c, ledger)
        pending_prev = pending
        for p in pending:
            named = (p['discriminator'] if p['kind'] == 'split'
                     else p['word'])
            for c in context_vocab(scan):
                if c in named:
                    record['proposal_counts'][c] += 1

        viol = web.check_conservation_laws()
        if viol:
            record['conservation_violations'].append({'gen': g,
                                                      'violations': viol})
            print(f'  !! conservation violations gen {g}: {viol}')

        stats = web.get_stats()
        closure_attempts = sum(s.ledger.reopen_count
                               for s in web.slots.values()) \
            + stats['closed']
        radii = sorted(round(s.geometry.radius, 4)
                       for s in web.slots.values()
                       if s.state == 'open' and s.ledger.fit_count >= 3
                       and np.isfinite(s.geometry.radius))
        record['generations'].append({
            'gen': g, 'regime': regime,
            'open': stats['open'], 'closed': stats['closed'],
            'archaized': stats['archaized'],
            'receipts': stats['total_receipts'],
            'edges': stats['total_edges'],
            'unassigned_pool': stats['unassigned_pool'],
            'churn_by_slot': stats['churn_by_slot'],
            'closure_attempts': closure_attempts,
            'min_radii': radii[:5],
            'proposals_pending': len(pending)})
        el = (time.time() - t0) / 60
        shift = ' [SHIFT NEXT]' if (g + 1) % REGIME_LEN == 0 else ''
        print(f'gen {g + 1}/{gens} R{regime} ({el:.1f} min): '
              f'open={stats["open"]} closed={stats["closed"]} '
              f'arch={stats["archaized"]} edges={stats["total_edges"]} '
              f'attempts={closure_attempts} pending={len(pending)} '
              f'minR={radii[:3]}{shift}')

        if g < gens - 1:
            engine = build_engine(lived_log)
            n = web.rebase(engine.encoder)
            print(f'  rebased {n} slots into epoch {web._embed_epoch} '
                  f'(log={len(lived_log)})')

    # observational export (unbilled): churn of Unify-born slots
    unify_churn = {s.name: {'reopens': s.ledger.reopen_count,
                            'state': s.state,
                            'fits': s.ledger.fit_count}
                   for s in web.slots.values()
                   if s.origin_operator == 'Unify'}
    unify_closed = [s.name for s in web.slots.values()
                    if s.origin_operator == 'Unify' and s.state == 'closed']

    out = {'record': {k: (dict(v) if isinstance(v, defaultdict) else v)
                      for k, v in record.items()},
           'verdicts': verdicts(web, scan, record),
           'unify_born_churn': unify_churn,
           'unify_born_closed': unify_closed,
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-300:],
           'context_windows': dict(scan.ctx_window_counts),
           'total_windows': scan.total_windows,
           'final_stats': web.get_stats(),
           'smoke': smoke,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    path = RESULTS if not smoke else RESULTS.replace('.json', '_smoke.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print('\n---- verdicts ----')
    for k, v in out['verdicts'].items():
        print(f'  {k}: {v["verdict"] if isinstance(v, dict) else v}')
    print(f'saved {path} ({out["elapsed_min"]} min)')


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)

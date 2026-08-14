"""Law-dose experiment — one arm (F26 impl. 1's pre-registered card,
locked at launch).

THE PREDICTION (fixed when F26 billed): LAW-mutations at furniture-
silent doses (1-4 per lineage per event) will REOPEN rule-Ks while
keeping contact (orphan rate ~0) — the falsification corridor is
enterable by changing DEPTH, not amplitude. The counter-outcome is
equally decisive: if law-mutations at these doses also fail to churn,
the K's content is deeper than the world's parameterization and
closure is effectively permanent in-lineage.

ARM STRUCTURE: identical to dose_sweep.py (same worlds — pair B,
screened CLOSABLE; phase 1 persistence to closure+2 or gen 14; phase 2
= 9 gens with events at phase-2 gens 0, 3, 6) with ONE difference: the
event applies L LAW-mutations per lineage (law_mutations.mutate_law —
timing laws only, zero furniture change, all addresses preserved).
'no-law-line' outcomes are counted, never silent.

DOSES: L in {1, 4}; 2 replicates each (rate design, C20-7).

POOLED VERDICTS (law_dose_analyze.py, same card):
  SUPPORTED   — pooled reopens >= 1 among contacted standing-K events
                AND pooled orphan rate <= 0.5.
  NOT SUPPORTED (deeper-than-parameterization) — 0 reopens across
                >= 4 contacted standing-K events.
  UNTESTED    — < 4 contacted standing-K events pooled.
Secondary (reported): court proposals per phase-2 gen — do LAW
mutations speak to the court below the furniture threshold (~8)?

Usage: python law_dose.py <L> <replicate>
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from environment_descriptive import classify_behavior_from_features
from environment_lexical import (Lexicon, evolve_one_generation,
                                 ratify_pending, append_ledger)
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, context_vocab, BOOT_SEED)
from replay_overnight_v3 import run_world_v3
from law_mutations import mutate_law

LINEAGE_SEEDS = ((97000, 4), (97001, 3))
P1_MAX = 14
P1_HOLD = 2
P2_GENS = 9
DOSE_EVENT_GENS = (0, 3, 6)


def main(L, rep):
    t0 = time.time()
    ro.CHURN_MIN = 2
    print(f'=== LAW DOSE arm L={L} rep={rep} ===')

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id=f'LAW_{L}_{rep}')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]

    epoch = 0
    world_events = []
    per_gen_closed = []
    fit_snapshots = []
    gen_records = []
    law_log = []
    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'ratified': []}
    proposals_by_phase = {'p1': 0, 'p2': 0}

    def one_generation(g, phase):
        nonlocal pending_prev, lexicon
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             20000 + rep * 5000 + li * 1000 + g * 17,
                             lived_log, {}, (li, epoch))
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
        proposals_by_phase[phase] += len(pending)
        for p in pending:
            named = (p['discriminator'] if p['kind'] == 'split'
                     else p['word'])
            for c in context_vocab(scan):
                if c in named:
                    record['proposal_counts'][c] += 1

        stats = web.get_stats()
        closed = {sid for sid, s in web.slots.items()
                  if s.state == 'closed'}
        per_gen_closed.append(sorted(closed))
        fit_snapshots.append({sid: s.ledger.fit_count
                              for sid, s in web.slots.items()})
        attempts = sum(s.ledger.reopen_count
                       for s in web.slots.values()) + stats['closed']
        gen_records.append({'gen': g, 'phase': phase, 'epoch': epoch,
                            'closed': stats['closed'],
                            'assertable': stats['assertable'],
                            'dormant': stats['dormant'],
                            'attempts': attempts,
                            'pending': len(pending)})
        el = (time.time() - t0) / 60
        print(f'L={L} r{rep} gen {g + 1} [{phase}] ({el:.1f} min): '
              f'closed={stats["closed"]} attempts={attempts} '
              f'pending={len(pending)}')
        return closed

    def rebuild():
        nonlocal engine
        engine = build_engine(lived_log)
        web.rebase(engine.encoder)

    def apply_dose(g):
        nonlocal epoch
        standing = [{'sid': sid, 'name': s.name,
                     'family': s.origin_family}
                    for sid, s in web.slots.items()
                    if s.state == 'closed']
        world_events.append({'gen': g, 'standing': standing})
        rng = np.random.RandomState(10000 + rep * 7919 + epoch * 97)
        for li in range(len(lineages)):
            for _ in range(L):
                lineages[li], d = mutate_law(lineages[li], rng)
                law_log.append({'gen': g, 'lineage': li, 'law': d})
        epoch += 1

    g = 0
    closure_gen = None
    while g < P1_MAX:
        closed = one_generation(g, 'p1')
        if closed and closure_gen is None:
            closure_gen = g
        if closure_gen is not None and g - closure_gen >= P1_HOLD:
            g += 1
            break
        rebuild()
        g += 1

    p2_start = g
    for k in range(P2_GENS):
        if k in DOSE_EVENT_GENS:
            apply_dose(p2_start + k)
            print(f'  >>> LAW EVENT at gen {p2_start + k} '
                  f'(standing={len(world_events[-1]["standing"])}): '
                  f'{[x["law"] for x in law_log[-2 * L:]]}')
        rebuild()
        one_generation(p2_start + k, 'p2')

    out = {'dose': str(L), 'replicate': rep,
           'closure_gen_p1': closure_gen,
           'p2_start': p2_start,
           'world_events': world_events,
           'per_gen_closed': per_gen_closed,
           'fit_snapshots': fit_snapshots,
           'gen_records': gen_records,
           'law_log': law_log,
           'proposals_by_phase': proposals_by_phase,
           'proposal_counts': dict(record['proposal_counts']),
           'ratified': record['ratified'],
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-100:],
           'conservation': web.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'law_dose_{L}_r{rep}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print(f'L={L} r{rep} done: closure_p1={closure_gen} '
          f'churn={out["total_churn_events"]} '
          f'p2_proposals={proposals_by_phase["p2"]} '
          f'no_law_lines={sum(1 for x in law_log if x["law"] == "no-law-line")} '
          f'({out["elapsed_min"]} min) -> {path}')


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))

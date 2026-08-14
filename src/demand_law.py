"""Demand alignment at the LAW LAYER — one arm (the card, locked at
launch; F27 impl. 8's pre-registered prediction).

THE QUESTION (booked card 2, third venue — the first fair one): does
the organism's top-churn slot name the same structure as the language's
most-proposed word, when both membranes are read AT THE LAW LAYER —
where F27 showed both concentrate (the organism closes on lawfulness;
the court hears law changes at 8x furniture sensitivity)?

WORLD CHANGES: LAW-STRUCTURE mutations (law_structure.py — laws born,
repealed, species-swapped; the depth rung above F27's parameters).
S=2 per lineage per event; events at phase-2 gens 0, 4, 8; P2=12 gens.
Arm structure otherwise identical to law_dose.py (pair B, P1
persistence to closure+2 or gen 14). R=6 replicates (rate design,
C20-7; the court's law-layer rate ~0.056/gen needs pooled gens).

POOLED ENDPOINTS (demand_law_analyze.py, same card):
1. DEMAND ALIGNMENT (original floors, unchanged since the card was
   booked): pooled churn events >= 10 AND pooled proposals >= 5.
   ALIGNED iff top-churn-slot's argmax-lift context == court's
   most-proposed context. Prediction (F27 impl. 8): ALIGNED.
2. HINGE-VS-STRUCTURE (billable secondary): contacted standing-K
   events under structure mutations — reopens >= 1 locates the hinge
   BELOW law-structure (the hierarchy has a falsification rung
   in-lineage); 0 reopens across >= 4 contacted events puts the K
   above every built stratum (in-lineage absolute so far).
3. Court rate at structure dose vs parameter dose (0.056/gen).
4. Split-reduces-churn rides along (CHURN_MIN=2) if any slot reaches
   2 reopens.

Usage: python demand_law.py <replicate>
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
from law_structure import mutate_law_structure

LINEAGE_SEEDS = ((97000, 4), (97001, 3))
P1_MAX = 14
P1_HOLD = 2
P2_GENS = 12
DOSE_EVENT_GENS = (0, 4, 8)
S = 2                     # structure mutations per lineage per event


def main(rep):
    t0 = time.time()
    ro.CHURN_MIN = 2
    print(f'=== DEMAND@LAW arm rep={rep} (S={S} structure/event) ===')

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id=f'DEMLAW_{rep}')
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
                             15000 + rep * 5000 + li * 1000 + g * 17,
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
        print(f'DL r{rep} gen {g + 1} [{phase}] ({el:.1f} min): '
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
        rng = np.random.RandomState(5000 + rep * 7919 + epoch * 97)
        for li in range(len(lineages)):
            for _ in range(S):
                lineages[li], d = mutate_law_structure(lineages[li], rng)
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
            print(f'  >>> STRUCTURE EVENT at gen {p2_start + k} '
                  f'(standing={len(world_events[-1]["standing"])}): '
                  f'{[x["law"] for x in law_log[-2 * S:]]}')
        rebuild()
        one_generation(p2_start + k, 'p2')

    out = {'replicate': rep,
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
           'splits': record['splits'],
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-200:],
           'context_windows': dict(scan.ctx_window_counts),
           'total_windows': scan.total_windows,
           'conservation': web.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'demand_law_r{rep}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    no_ops = sum(1 for x in law_log if x['law'].startswith('no-op'))
    print(f'DL r{rep} done: closure_p1={closure_gen} '
          f'churn={out["total_churn_events"]} '
          f'p2_proposals={proposals_by_phase["p2"]} no_ops={no_ops} '
          f'({out["elapsed_min"]} min) -> {path}')


if __name__ == '__main__':
    main(int(sys.argv[1]))

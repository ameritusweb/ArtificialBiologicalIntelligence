"""Dose-response sweep — one arm (F25 impl. 1; the card, locked at launch).

THE QUESTION: is there a world-change DOSE that serves both membranes —
large enough to falsify closed accounts (web churn) and to generate
description failures (court proposals), small enough to keep accounts
reachable (low orphaning)? F25 found the extremes: metronome dose
(1 mutation) leaves Ks standing and the court silent; full reseed
(v3) reopens Ks on contact but orphans 3/4 and starves the court a
different way. The sweep measures the three curves between.

DOSES: mutations applied to EACH lineage at every dose event:
  1, 4, 8, 16, or RESEED (dose=-1: replace each lineage's corpus with a
  fresh world's corpus — new seed per event, tier preserved; maximum
  structural change).

ARM STRUCTURE (deterministic, pre-registered):
  PHASE 1 (persistence): no world changes; exit at first closure held
    2 generations, or at gen 14 — whichever first.
  PHASE 2 (dosing): 9 generations; dose events at phase-2 gens 0, 3, 6
    (three events per arm). No closure-paced shifts — dose is the only
    world-change variable. Metronome disabled for the same reason.

RATE DESIGN (C20 check 7): 2 replicates per dose x 5 doses; pooled by
dose_sweep_analyze.py (same card). Worlds: pair B (97000 t4 / 97001
t3), screened CLOSABLE 4/4 census (closability probe receipts).

ENDPOINTS (pooled per dose, all pre-registered):
  reopen rate  — P(K reopens within 3 gens of a dose event | standing K
                 at the event); orphaned events (no contact within 3
                 gens) counted separately, not as failures.
  orphan rate  — P(no contact within 3 gens | standing K at event).
  proposal rate — court proposals per phase-2 generation.
VERDICT (the band rule, fixed):
  BAND EXISTS — >= 1 dose with pooled reopens >= 1 AND pooled
    proposals >= 1 AND orphan rate <= 0.5. The treadmill has a gear.
  NO BAND — no dose satisfies all three while >= 1 dose shows reopens
    and >= 1 dose shows proposals (both responses exist, disjoint
    support): the membranes need different world-change species.
  UNTESTED — no standing K at any dose event across replicates (web
    side unmeasurable), or zero proposals at EVERY dose including
    reseed (court unmeasurable in this harness — F25's silence
    extends to all doses, itself billable as a court-side
    instrument finding).

Usage: python dose_sweep.py <dose|-1> <replicate>
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
from environment_living import mutate
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, context_vocab, BOOT_SEED)
from replay_overnight_v3 import run_world_v3

LINEAGE_SEEDS = ((97000, 4), (97001, 3))     # pair B, screened CLOSABLE
P1_MAX = 14
P1_HOLD = 2
P2_GENS = 9
DOSE_EVENT_GENS = (0, 3, 6)                  # phase-2-relative
RESEED = -1


def main(dose, rep, smoke=False):
    t0 = time.time()
    ro.CHURN_MIN = 2
    label = 'RESEED' if dose == RESEED else str(dose)
    print(f'=== DOSE SWEEP arm dose={label} rep={rep} ===')

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id=f'DOSE_{label}_{rep}')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]

    p1_max = 3 if smoke else P1_MAX
    p2_gens = 4 if smoke else P2_GENS
    dose_events_at = (0, 2) if smoke else DOSE_EVENT_GENS

    epoch = 0
    world_events = []
    per_gen_closed = []
    fit_snapshots = []
    gen_records = []
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
                             40000 + rep * 5000 + li * 1000 + g * 17,
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
        print(f'dose={label} r{rep} gen {g + 1} [{phase}] ({el:.1f} min): '
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
        rng = np.random.RandomState(30000 + rep * 7919 + epoch * 97)
        if dose == RESEED:
            for li, (s0, t0_) in enumerate(LINEAGE_SEEDS):
                new_seed = s0 + 500 + epoch * 37
                lineages[li] = describe(
                    TieredEnvironment(seed=new_seed, tier=t0_))
        else:
            for li in range(len(lineages)):
                for _ in range(dose):
                    lineages[li], _d = mutate(lineages[li], rng)
        epoch += 1

    # ---- phase 1: persistence ----
    g = 0
    closure_gen = None
    while g < p1_max:
        closed = one_generation(g, 'p1')
        if closed and closure_gen is None:
            closure_gen = g
        if closure_gen is not None and g - closure_gen >= P1_HOLD:
            g += 1
            break
        rebuild()
        g += 1

    # ---- phase 2: dosing ----
    p2_start = g
    for k in range(p2_gens):
        if k in dose_events_at:
            apply_dose(p2_start + k)
            print(f'  >>> DOSE EVENT at gen {p2_start + k} '
                  f'(standing={len(world_events[-1]["standing"])})')
        rebuild()
        one_generation(p2_start + k, 'p2')

    out = {'dose': label, 'replicate': rep,
           'closure_gen_p1': closure_gen,
           'p2_start': p2_start,
           'world_events': world_events,
           'per_gen_closed': per_gen_closed,
           'fit_snapshots': fit_snapshots,
           'gen_records': gen_records,
           'proposals_by_phase': proposals_by_phase,
           'proposal_counts': dict(record['proposal_counts']),
           'ratified': record['ratified'],
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-100:],
           'conservation': web.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results',
                        f'dose_sweep_{label}_r{rep}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print(f'dose={label} r{rep} done: closure_p1={closure_gen} '
          f'churn={out["total_churn_events"]} '
          f'p2_proposals={proposals_by_phase["p2"]} '
          f'({out["elapsed_min"]} min) -> {path}')


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]),
         smoke='--smoke' in sys.argv)

"""Replay phase v5 — the metronome: an environment with its own pulse.

THE CARD (locked at launch; C20 SEVEN checks). F24's corrections,
enacted:
  (1) LIVENESS FLOOR — the mutation METRONOME: one mutation per lineage
      every METRONOME_M=6 generations, unconditionally (the
      environment's own metabolism). Closure-paced shifts (live-closure
      rule, 2 extra mutations/lineage) remain as ADDITIONAL punctuation
      — the student's clock modulates the tempo, never owns it.
  (2) RATE DESIGN (C20 check 7a) — R=3 replicates x 18 generations,
      same lineage starting corpora, independent realizations (episode
      streams, webs, encoders). Genesis endpoints are billed as rates/
      pools across replicates by the separate analyzer
      (replay_v5_analyze.py), never as single-run presence.
  (3) SCREENED WORLDS (C20 check 7b) — lineages selected by the
      closability probe (closability_probe.py results cited in the
      analyzer's bill); check 5 re-earned in the corpus-world
      distribution, not carried from direct-env runs.

Both mutation species (metronome tick, closure-paced shift) are WORLD
CHANGES: standing closed Ks at either are shift-test material
(contact-gated F23 form, pooled across replicates by the analyzer).

Per-replicate outputs -> results/replay_phase_v5_r<R>.json. Endpoints,
floors, and verdict rules live in the analyzer docstring (fixed before
launch, same card): shift-test 2x2 pooled; split-reduces-churn
(CHURN_MIN=2, child floor 8000, R<=0.5 / >=0.8) per-replicate,
first-split bills; demand alignment pooled (floors 10 churn / 5
proposals); court engagement rate vs v4's 1/36; orphaning + dormancy
counts (dormancy mechanism live in sov.py).

Usage: python replay_overnight_v5.py <replicate> <seedA> <tierA> \
           <seedB> <tierB>
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
from receptor_eigen_coder import ReceptorEigenCoder, FAMILY_GROUPS
from sov import ConstraintWeb
from train import generate_training_data, train_model

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, context_vocab, verdicts,
                              BOOT_SEED)
from replay_overnight_v3 import run_world_v3

GENERATIONS = 18
METRONOME_M = 6            # environment's own pulse: 1 mutation/lineage
MUT_PER_SHIFT = 2          # closure-paced punctuation (extra)
MIN_DWELL = 4
POST_CLOSURE_HOLD = 2
CHURN_MIN_V5 = 2


def main(rep, sa, ta, sb, tb):
    t0 = time.time()
    ro.CHURN_MIN = CHURN_MIN_V5
    print(f'=== REPLAY v5 replicate {rep}: metronome M={METRONOME_M}, '
          f'lineages ({sa},t{ta})+({sb},t{tb}) ===')

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id=f'REPLAY5_{rep}')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    fam_tracker = {}
    lineages = [describe(TieredEnvironment(seed=sa, tier=ta)),
                describe(TieredEnvironment(seed=sb, tier=tb))]

    epoch = 0                  # world-change counter (either species)
    regime_start = 0
    first_live_gen = None
    world_events = []          # {'gen','kind','standing':[...]}
    per_gen_closed = []
    fit_snapshots = []

    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'generations': [],
              'ratified': [], 'conservation_violations': [],
              'mutations': [],
              'config': {'replicate': rep, 'generations': GENERATIONS,
                         'metronome': METRONOME_M,
                         'mut_per_shift': MUT_PER_SHIFT,
                         'churn_min': CHURN_MIN_V5,
                         'lineages': [[sa, ta], [sb, tb]]}}

    def standing_Ks():
        return [{'sid': sid, 'name': s.name, 'family': s.origin_family,
                 'origin': s.origin_operator, 'live': not s.dormant,
                 'closed_at': s.closed_at}
                for sid, s in web.slots.items() if s.state == 'closed']

    def apply_mutations(g, kind, n_per_lineage):
        nonlocal epoch, regime_start, first_live_gen
        world_events.append({'gen': g, 'kind': kind,
                            'standing': standing_Ks()})
        rng = np.random.RandomState(66000 + rep * 7919 + epoch * 97)
        muts = []
        for li in range(len(lineages)):
            for _ in range(n_per_lineage):
                lineages[li], desc = mutate(lineages[li], rng)
                muts.append({'lineage': li, 'mutation': str(desc)})
        record['mutations'].append({'gen': g, 'kind': kind,
                                    'muts': muts})
        epoch += 1
        regime_start = g + 1
        first_live_gen = None

    for g in range(GENERATIONS):
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             60000 + rep * 5000 + li * 1000 + g * 17,
                             lived_log, fam_tracker, (li, epoch))
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
            record['conservation_violations'].append(
                {'gen': g, 'violations': viol})

        stats = web.get_stats()
        closed = {sid for sid, s in web.slots.items()
                  if s.state == 'closed'}
        live = {sid for sid in closed if not web.slots[sid].dormant}
        per_gen_closed.append({sid: True for sid in closed})
        fit_snapshots.append({sid: s.ledger.fit_count
                              for sid, s in web.slots.items()})
        attempts = sum(s.ledger.reopen_count
                       for s in web.slots.values()) + stats['closed']
        record['generations'].append({
            'gen': g, 'epoch': epoch,
            'open': stats['open'], 'closed': stats['closed'],
            'assertable': stats['assertable'],
            'dormant': stats['dormant'],
            'archaized': stats['archaized'],
            'edges': stats['total_edges'],
            'churn_by_slot': stats['churn_by_slot'],
            'closure_attempts': attempts,
            'proposals_pending': len(pending)})
        el = (time.time() - t0) / 60
        tag = ''

        # closure-paced punctuation (student's clock: modulates)
        if live and first_live_gen is None:
            first_live_gen = g
        elif not live:
            first_live_gen = None
        dwell = g - regime_start + 1
        if (live and first_live_gen is not None
                and g - first_live_gen >= POST_CLOSURE_HOLD
                and dwell >= MIN_DWELL and g < GENERATIONS - 1):
            apply_mutations(g, 'closure-paced', MUT_PER_SHIFT)
            tag = ' [SHIFT closure-paced]'
        # the metronome (environment's own pulse: unconditional)
        elif (g + 1) % METRONOME_M == 0 and g < GENERATIONS - 1:
            apply_mutations(g, 'metronome', 1)
            tag = ' [TICK metronome]'

        print(f'r{rep} gen {g + 1}/{GENERATIONS} ({el:.1f} min): '
              f'closed={stats["closed"]} live={stats["assertable"]} '
              f'dormant={stats["dormant"]} attempts={attempts} '
              f'edges={stats["total_edges"]} pending={len(pending)}{tag}')

        if g < GENERATIONS - 1:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)

    v = verdicts(web, scan, record)
    # P69-restatement instrumentation (2026-08-12): composed-slot
    # economics under net-per-fire pricing (F31's decisive observables)
    composed_econ = [
        {'name': c['name'], 'gen': c['gen'],
         'state': web.slots[c['slot_id']].state,
         'fire_rate': round(web.slots[c['slot_id']].ledger.fire_rate, 4),
         'rent_balance':
         round(web.slots[c['slot_id']].ledger.rent_balance, 4),
         'fits': web.slots[c['slot_id']].ledger.fit_count}
        for c in record['composed'] if c['slot_id'] in web.slots]
    out = {'record': {k: (dict(vv) if isinstance(vv, defaultdict) else vv)
                      for k, vv in record.items()},
           'composed_economics': composed_econ,
           'per_replicate_verdicts': v,
           'world_events': world_events,
           'per_gen_closed': [sorted(d.keys()) for d in per_gen_closed],
           'fit_snapshots': fit_snapshots,
           'fam_tracker': {f'{k[0]}_{k[1]}': [t[0].tolist(), t[1]]
                           for k, t in fam_tracker.items()},
           'slot_families': {sid: s.origin_family
                             for sid, s in web.slots.items()},
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-200:],
           'context_windows': dict(scan.ctx_window_counts),
           'total_windows': scan.total_windows,
           'court_proposals': sum(r['proposals_pending']
                                  for r in record['generations']),
           'dormancy_events': sum(1 for ev in web.etymology
                                  if ev.event_type == 'dormant'),
           'final_stats': web.get_stats(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'replay_phase_v5_r{rep}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    final_attempts = out['record']['generations'][-1]['closure_attempts']
    print(f'r{rep} done: attempts={final_attempts} '
          f'churn={out["total_churn_events"]} '
          f'proposals={out["court_proposals"]} '
          f'({out["elapsed_min"]} min) -> {path}')


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
         int(sys.argv[4]), int(sys.argv[5]))

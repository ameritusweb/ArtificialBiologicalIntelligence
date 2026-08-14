"""The split knob — split-reduces-churn's run (the card, locked at
launch). The LAST unfired pre-registration of the replay phase.

F13 impl. 7 booked the acceptance test; F28 found the knob: structure
dose drives churn (17 events/6 arms at 3 events each), so a LONGER
dosing phase accumulates reopen_count >= CHURN_MIN=2 on a slot and the
Differentiate scan fires. This arm is demand_law.py with P2 stretched:
24 dosing generations, structure events every 3 (8 events/arm), S=2
law-structure mutations per lineage per event. R=4 replicates.

THE DISCRIMINATOR (thresholds locked since the v1 card + v3
re-registration): the first Differentiate in each web targets its
top-churn slot. Matched exposure by RATE WINDOWS:
  parent churn rate  = parent reopens / parent positive fits over the
                       final 8 generations BEFORE the split;
  children churn rate = combined child reopens / combined child fits
                       from the split to run end.
  Exposure floor: children combined fits >= 8000.
  R = child_rate / parent_rate:
    R <= 0.5  -> CONFLATION SUPPORTED (the split named a distinction)
    R >= 0.8  -> NON-STATIONARITY (churn is a world-regime clock;
                 T153 gets its measurement) — reopen timing clustering
                 reported alongside, unbilled
    else      -> INCONCLUSIVE
    floors unmet or no split -> UNTESTED
Pooled: per-web verdicts reported; the acceptance bills on the webs
whose floors are met (majority reading if they disagree).

Usage: python split_knob.py <replicate>
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
P2_GENS = 24        # overridable by CLI arg 2 — the timescale knob
                    # (the F28-addendum law: repeated churn needs 50+
                    # post-closure gens); thresholds never change
DOSE_EVERY = 3
S = 2


def main(rep, p2_gens=P2_GENS):
    t0 = time.time()
    ro.CHURN_MIN = 2
    print(f'=== SPLIT KNOB arm rep={rep}: P2={p2_gens}, structure '
          f'events every {DOSE_EVERY} ===')

    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id=f'KNOB_{rep}')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]

    epoch = 0
    per_gen_reopens = []
    fit_snapshots = []
    gen_records = []
    law_log = []
    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'ratified': []}

    def one_generation(g, phase):
        nonlocal pending_prev, lexicon
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             8000 + rep * 5000 + li * 1000 + g * 17,
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
        for p in pending:
            named = (p['discriminator'] if p['kind'] == 'split'
                     else p['word'])
            for c in context_vocab(scan):
                if c in named:
                    record['proposal_counts'][c] += 1

        stats = web.get_stats()
        per_gen_reopens.append({sid: s.ledger.reopen_count
                                for sid, s in web.slots.items()})
        fit_snapshots.append({sid: s.ledger.fit_count
                              for sid, s in web.slots.items()})
        attempts = sum(s.ledger.reopen_count
                       for s in web.slots.values()) + stats['closed']
        gen_records.append({'gen': g, 'phase': phase,
                            'closed': stats['closed'],
                            'attempts': attempts,
                            'archaized': stats['archaized'],
                            'pending': len(pending)})
        el = (time.time() - t0) / 60
        split_tag = (f' SPLITS={len(record["splits"])}'
                     if record['splits'] else '')
        print(f'KNOB r{rep} gen {g + 1} [{phase}] ({el:.1f} min): '
              f'closed={stats["closed"]} attempts={attempts}'
              f'{split_tag}')
        return {sid for sid, s in web.slots.items()
                if s.state == 'closed'}

    def rebuild():
        nonlocal engine
        engine = build_engine(lived_log)
        web.rebase(engine.encoder)

    def apply_dose(g):
        nonlocal epoch
        rng = np.random.RandomState(2000 + rep * 7919 + epoch * 97)
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
    for k in range(p2_gens):
        if k % DOSE_EVERY == 0:
            apply_dose(p2_start + k)
        rebuild()
        one_generation(p2_start + k, 'p2')

    out = {'replicate': rep,
           'closure_gen_p1': closure_gen,
           'p2_start': p2_start,
           'splits': record['splits'],
           'scan_events': record['scan_events'],
           'per_gen_reopens': per_gen_reopens,
           'fit_snapshots': fit_snapshots,
           'gen_records': gen_records,
           'law_log': law_log,
           'proposal_counts': dict(record['proposal_counts']),
           'ratified': record['ratified'],
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-200:],
           'conservation': web.check_conservation_laws(),
           'slot_names': {sid: s.name for sid, s in web.slots.items()},
           'slot_states': {sid: s.state for sid, s in web.slots.items()},
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'split_knob_r{rep}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print(f'KNOB r{rep} done: splits={len(record["splits"])} '
          f'churn={out["total_churn_events"]} '
          f'({out["elapsed_min"]} min) -> {path}')


if __name__ == '__main__':
    main(int(sys.argv[1]),
         int(sys.argv[2]) if len(sys.argv) > 2 else P2_GENS)

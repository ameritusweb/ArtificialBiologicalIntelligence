"""Replay phase v4 — THE FUSION: closure lifecycle inside living worlds.

THE CARD (locked at launch; C20 applies). F23 forced this design twice
over: full reseed ORPHANS closed accounts (the world change makes them
unreachable — no contact, no falsification), and long persistence
STARVES the court (no description failures, zero proposals in 36 gens).
Both problems have one solution: worlds as MUTATING LINEAGES (EL-2.5's
operator) — small changes to a persistent world, so closed Ks stay
reachable while their statistics genuinely move, and the court gets
fresh structure to fail at describing. The replay phase and the EL arc
fuse by necessity: this is the first run where the organism's web and
the environment's language evolve against the SAME moving lineage.

REGIME (fixed): 36 generations, 2 corpus lineages (tier-4 and tier-3
seed corpora via EL-0 describe; worlds instantiated by interpret()).
Curriculum clock v2 (F23 impl. 2, the corrected register): shift iff
a LIVE closure — closed AND assertable (dormancy mechanism now in
sov.py, DORMANCY_WINDOW=10000) — has stood POST_CLOSURE_HOLD=2 gens
and dwell >= 6; fallback dwell 24. ON SHIFT: MUT_PER_SHIFT=2 mutations
applied to EACH lineage corpus (deterministic rng per shift) — the
world moves, the world remains.

PRE-REGISTERED ENDPOINTS:

A. SHIFT-TEST, CONTACT-GATED (the F23-corrected form, both cells):
   standing LIVE Ks at each shift, families classified by cross-world
   activation variance (median split at analysis):
     variant-family K:   assessed iff >= 12 post-contact fits within
                         the horizon; passes iff reopened within 2 gens
                         of FIRST post-shift contact.
     invariant-family K: assessed iff >= 12 post-shift fits within 4
                         gens; passes iff not reopened in those 4 gens.
   Ks with no post-shift contact in 4 gens = ORPHANED (unassessed;
   counted — the mutation regime predicts this stays near zero, vs
   v3's 3/4 orphaned shifts under full reseed; reported as the
   fusion's own receipt).
   SUPPORTED: every assessed cell majority-correct (>= 1 cell
   assessed). PARTIAL: one of two assessed cells. NOT SUPPORTED:
   assessed cells majority-wrong. UNTESTED: no standing K or nothing
   assessable.

B. SPLIT-REDUCES-CHURN at v3 grain (carried: CHURN_MIN=2, child
   exposure floor 8000 fits, R <= 0.5 conflation / >= 0.8 non-stat).

C. DEMAND ALIGNMENT (floors unchanged: 10 churn events, 5 proposals).
   The mutation regime is its first fair venue: the court has
   something to fail at, the web has churn from real shifts.

D. COURT ENGAGEMENT vs v3 (reported): proposals per generation under
   mutating lineages vs v3's zero — F23 impl. 5's tempo-tension
   receipt.

Also reported: dormancy events (should be rare vs v3), assertive
fraction per generation (now reading stats['assertable'] — the
corrected register), P69 signature (assumed law-grade, not re-billed).

C20: checks 1-4, 6 carried (same machinery; endpoint mechanics as v3
plus contact gating). Check 5: closure under persistence demonstrated
(v2/v3, F22/F23); mutation machinery is EL-2.5's, accepted (F17-F19);
smoke exercises the fused path (interpret-worlds, mutation shifts,
contact analysis) with fallback dwell 2.
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
from replay_overnight_v3 import run_world_v3, family_invariance

FAMILY_NAMES = [name for name, _ in FAMILY_GROUPS]

GENERATIONS = 36
LINEAGE_SEEDS = ((97501, 4), (97502, 3))    # (seed corpus world, tier)
MIN_DWELL = 6
FALLBACK_DWELL = 24
POST_CLOSURE_HOLD = 2
MUT_PER_SHIFT = 2

REOPEN_FAST = 2            # gens from FIRST CONTACT (F23-corrected)
SURVIVE_GENS = 4
MIN_ASSESS_FITS = 12       # the 404 window length: tested, not idle
CHURN_MIN_V4 = 2

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'replay_phase_v4.json')


def main(smoke=False):
    t0 = time.time()
    gens = 6 if smoke else GENERATIONS
    fallback = 2 if smoke else FALLBACK_DWELL
    min_dwell = 1 if smoke else MIN_DWELL
    ro.CHURN_MIN = CHURN_MIN_V4
    print(f'=== REPLAY PHASE v4 {"SMOKE" if smoke else "FULL"} (the '
          f'fusion): {gens} gens, mutating lineages, live-closure '
          f'clock ===')

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
                        ledger_id='REPLAY4')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    fam_tracker = {}

    # the living lineages
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]

    regime = 0
    regime_start = 0
    first_live_gen = None
    shifts = []
    per_gen_closed = []
    fit_snapshots = []
    dormancy_events = 0

    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'generations': [],
              'ratified': [], 'conservation_violations': [],
              'mutations': [],
              'config': {'generations': gens, 'min_dwell': min_dwell,
                         'fallback_dwell': fallback,
                         'post_closure_hold': POST_CLOSURE_HOLD,
                         'mut_per_shift': MUT_PER_SHIFT,
                         'churn_min': CHURN_MIN_V4,
                         'lineage_seeds': list(LINEAGE_SEEDS)}}

    for g in range(gens):
        bank = LiveReceptorBank()
        gen_windows = []
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             98000 + li * 1000 + g * 17, lived_log,
                             fam_tracker, (li, regime))
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
        closed_slots = {sid: s for sid, s in web.slots.items()
                        if s.state == 'closed'}
        live_slots = {sid: s for sid, s in closed_slots.items()
                      if not s.dormant}
        per_gen_closed.append({sid: True for sid in closed_slots})
        fit_snapshots.append({sid: s.ledger.fit_count
                              for sid, s in web.slots.items()})
        dormancy_events = sum(1 for ev in web.etymology
                              if ev.event_type == 'dormant')
        closure_attempts = sum(s.ledger.reopen_count
                               for s in web.slots.values()) \
            + stats['closed']
        total_slots = stats['open'] + stats['closed']
        assertive = stats['assertable'] / max(total_slots, 1)
        record['generations'].append({
            'gen': g, 'regime': regime,
            'open': stats['open'], 'closed': stats['closed'],
            'assertable': stats['assertable'],
            'dormant': stats['dormant'],
            'archaized': stats['archaized'],
            'receipts': stats['total_receipts'],
            'edges': stats['total_edges'],
            'churn_by_slot': stats['churn_by_slot'],
            'closure_attempts': closure_attempts,
            'assertive_fraction': round(assertive, 4),
            'proposals_pending': len(pending)})
        el = (time.time() - t0) / 60

        # ---- curriculum clock v2: LIVE closures only ----
        if live_slots and first_live_gen is None:
            first_live_gen = g
        elif not live_slots:
            first_live_gen = None
        dwell = g - regime_start + 1
        do_shift = ((len(live_slots) >= 1
                     and first_live_gen is not None
                     and g - first_live_gen >= POST_CLOSURE_HOLD
                     and dwell >= min_dwell)
                    or dwell >= fallback)
        tag = ''
        if do_shift and g < gens - 1:
            standing = [{'sid': sid, 'name': s.name,
                         'family': s.origin_family,
                         'origin': s.origin_operator,
                         'live': not s.dormant,
                         'closed_at': s.closed_at}
                        for sid, s in closed_slots.items()]
            shifts.append({'gen': g, 'regime_ending': regime,
                           'dwell': dwell, 'standing': standing,
                           'by_fallback': dwell >= fallback})
            rng = np.random.RandomState(55000 + regime * 97)
            muts = []
            for li in range(len(lineages)):
                for _ in range(MUT_PER_SHIFT):
                    lineages[li], desc = mutate(lineages[li], rng)
                    muts.append({'lineage': li, 'mutation': str(desc)})
            record['mutations'].append({'gen': g, 'muts': muts})
            regime += 1
            regime_start = g + 1
            first_live_gen = None
            tag = (f' [SHIFT -> R{regime}'
                   + (' fallback]' if dwell >= fallback
                      else ' closure-paced]'))
        print(f'gen {g + 1}/{gens} R{record["generations"][-1]["regime"]} '
              f'({el:.1f} min): open={stats["open"]} '
              f'closed={stats["closed"]} live={stats["assertable"]} '
              f'dormant={stats["dormant"]} edges={stats["total_edges"]} '
              f'attempts={closure_attempts} pending={len(pending)}{tag}')

        if g < gens - 1:
            engine = build_engine(lived_log)
            n = web.rebase(engine.encoder)
            print(f'  rebased {n} slots into epoch {web._embed_epoch} '
                  f'(log={len(lived_log)})')

    # ------------- endpoint A: contact-gated shift-test -------------
    invariant = family_invariance(fam_tracker)
    fam_index = {name: i for i, name in enumerate(FAMILY_NAMES)}
    shift_tests = []
    orphaned = 0
    for sh in shifts:
        sg = sh['gen']
        horizon = min(sg + SURVIVE_GENS, gens - 1)
        for K in sh['standing']:
            sid = K['sid']
            # first post-shift contact: first gen with new fits
            contact_gen = None
            for gg in range(sg + 1, horizon + 1):
                if (fit_snapshots[gg].get(sid, 0)
                        > fit_snapshots[gg - 1].get(sid, 0)):
                    contact_gen = gg
                    break
            reopen_gen = None
            for gg in range(sg + 1, gens):
                if sid not in per_gen_closed[gg]:
                    reopen_gen = gg
                    break
            fits_post = (fit_snapshots[horizon].get(sid, 0)
                         - fit_snapshots[sg].get(sid, 0))
            fi = fam_index.get(K['family'], -1)
            cls = ('invariant' if fi in invariant else 'variant') \
                if fi >= 0 else 'unclassified'
            if contact_gen is None:
                orphaned += 1
                shift_tests.append({'shift_gen': sg, 'slot': K['name'],
                                    'class': cls, 'orphaned': True,
                                    'assessed': False, 'passed': None})
                continue
            if cls == 'variant':
                assessed = fits_post >= MIN_ASSESS_FITS
                passed = (assessed and reopen_gen is not None
                          and reopen_gen - contact_gen <= REOPEN_FAST) \
                    if assessed else None
            elif cls == 'invariant':
                assessed = (fits_post >= MIN_ASSESS_FITS
                            and gens - 1 - sg >= SURVIVE_GENS)
                passed = (reopen_gen is None
                          or reopen_gen - sg > SURVIVE_GENS) \
                    if assessed else None
            else:
                assessed, passed = False, None
            shift_tests.append({'shift_gen': sg, 'slot': K['name'],
                                'family': K['family'], 'class': cls,
                                'orphaned': False,
                                'contact_gen': contact_gen,
                                'reopen_gen': reopen_gen,
                                'fits_post': int(fits_post),
                                'assessed': bool(assessed),
                                'passed': passed})
    cells = {c: [t for t in shift_tests
                 if t.get('class') == c and t['assessed']]
             for c in ('variant', 'invariant')}

    def cell_ok(c):
        return (sum(1 for t in c if t['passed']) > len(c) / 2) if c else None
    ok_v, ok_i = cell_ok(cells['variant']), cell_ok(cells['invariant'])
    if not any(sh['standing'] for sh in shifts):
        st_verdict = 'UNTESTED (no standing K at any shift)'
    elif ok_v is None and ok_i is None:
        st_verdict = 'UNTESTED (no assessable K after contact gating)'
    else:
        n_cells = sum(1 for o in (ok_v, ok_i) if o is not None)
        wins = sum(1 for o in (ok_v, ok_i) if o)
        if wins == n_cells:
            st_verdict = ('SUPPORTED (assessed cells): closure survival '
                          'tracks content invariance under contact')
        elif wins >= 1:
            st_verdict = 'PARTIAL: one assessed cell behaves'
        else:
            st_verdict = 'NOT SUPPORTED: assessed cells majority-wrong'

    v = verdicts(web, scan, record)
    v['shift_test'] = {'verdict': st_verdict, 'tests': shift_tests,
                       'orphaned_count': orphaned,
                       'invariant_families': sorted(
                           FAMILY_NAMES[i] for i in invariant)}

    out = {'record': {k: (dict(vv) if isinstance(vv, defaultdict) else vv)
                      for k, vv in record.items()},
           'verdicts': v,
           'shifts': shifts,
           'dormancy_events': dormancy_events,
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-300:],
           'total_windows': scan.total_windows,
           'assertive_curve': [(r['gen'], r['assertive_fraction'])
                               for r in record['generations']],
           'court_engagement': sum(r['proposals_pending']
                                   for r in record['generations']),
           'final_stats': web.get_stats(),
           'smoke': smoke,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    path = RESULTS if not smoke else RESULTS.replace('.json', '_smoke.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print('\n---- verdicts ----')
    for k, vv in out['verdicts'].items():
        print(f'  {k}: {vv["verdict"] if isinstance(vv, dict) else vv}')
    print(f"orphaned standing-Ks: {orphaned} (v3: 3/4 shifts); "
          f"court proposals: {out['court_engagement']} (v3: 0); "
          f"dormancy events: {dormancy_events}")
    print(f'saved {path} ({out["elapsed_min"]} min)')


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)

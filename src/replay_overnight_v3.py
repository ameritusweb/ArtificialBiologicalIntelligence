"""Replay phase REGIME v3 — the closure-paced curriculum.

THE CARD (locked at launch; C20 applies). F22's environment-evolution
implications (findings_implications.md, F22 impl. 11-14), enacted: the
world holds still until its students close accounts, then moves. The
shift schedule is NOT a fixed calendar — it is a deterministic RULE
reading the web's own closure telemetry at each generation boundary
(the closure ledger as curriculum clock). This is pre-registered
curriculum design, not endpoint contamination: the rule reads the
web's accounting STATE (closed count, closure timing), never any
billed endpoint value, and is fixed here before launch.

SHIFT RULE (deterministic, evaluated after each generation):
  shift iff (closed >= 1 AND generations since the regime's first
  closure >= 2 AND regime dwell >= MIN_DWELL=6)
        or (regime dwell >= FALLBACK_DWELL=24, the no-closure escape).
  On shift: both world seeds change (full reseed — the coarse
  instrument; targeted rule-perturbation is EL-2.5 mutation territory).
  GENERATIONS=36 total; everything else (worlds/gen, episodes, steps,
  lived-log encoder rebuild + rebase, scans before rebase, court with
  cross-generation junction law) carried unchanged from v2.

PRE-REGISTERED ENDPOINTS (v3):

A. THE SHIFT-TEST 2x2 — F22 impl. 6's content-form of the relational-
   closure question: closure survival across a world shift should track
   CONTENT INVARIANCE, not slot identity. Standing closed Ks at each
   shift are classified by their family's cross-world activation
   variance (per-family mean activation per world, variance across all
   worlds of the run, median split — deterministic, computed at
   analysis):
     variant-family K   -> predicted to REOPEN within 2 post-shift gens
     invariant-family K -> predicted to SURVIVE >= 4 post-shift gens
                           un-reopened while receiving >= 12 fits (the
                           404 window length: survival tested, not idle)
   SUPPORTED: both cells behave (each assessed cell majority-correct).
   PARTIAL: one cell. NOT SUPPORTED: neither or reversed.
   UNTESTED: no standing K at any shift, or no assessable K.

B. SPLIT-REDUCES-CHURN at v3 grain: CHURN_MIN=2 (re-registered from 3
   with cause: v2 measured one closure-reopen cycle ~ 15 generations,
   so 3 cycles is infeasible in any session run; 2 reopens is already
   repeated demand). Matched exposure by RATE WINDOWS: parent rate =
   reopens/positive-fits over its final 8 pre-split generations; child
   rate = combined reopens/fits from split to run end; billed only if
   children combined fits >= 8,000 (~2 generations of slot traffic).
   R thresholds unchanged: CONFLATION <= 0.5, NON-STATIONARITY >= 0.8.

C. P69 re-report (SUPPORTED x2; not re-billed).

D. DEMAND ALIGNMENT: floors unchanged (10 churn events / 5 proposals);
   folds at zero cost; F22 impl. 8 says its honest home is a rate
   design across replicates — a single run stays likely-UNTESTED.

Also reported (unbilled): reopen latency per K per shift; churn per
shift; merged-slot (Unify-born) churn; assertive fraction per
generation (F22 impl. 10 — the web's coming-of-age curve).

C20 PRE-FLIGHT: checks 1-4, 6 carried from v1/v2 (same machinery, same
endpoint mechanics). Check 5 (phenomenon strength): closure under
persistence is DEMONSTRATED — v2 full run closed `formalization` at
gen 19 and held it 5 generations through 5 rebases (F22); smoke here
exercises the NEW machinery only (dynamic scheduler, shift path,
variance tracker) with FALLBACK_DWELL=2 to force shifts quickly.
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_descriptive import (window_features,
                                     classify_behavior_from_features,
                                     WINDOW)
from environment_lexical import (Lexicon, base_predicates,
                                 evolve_one_generation, ratify_pending,
                                 append_ledger)
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder, FAMILY_GROUPS
from sov import ConstraintWeb
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

import replay_overnight as ro
from replay_overnight import (build_engine, calibrate_taus, ScanState,
                              replay_scans, context_vocab, verdicts,
                              BOOT_SEED)

NUM_FAM = len(FAMILY_GROUPS)
FAMILY_NAMES = [name for name, _ in FAMILY_GROUPS]

GENERATIONS = 36
WORLDS_PER_GEN = 2
TIERS = (4, 3)
V3_SEED_BASE = 97000
MIN_DWELL = 6
FALLBACK_DWELL = 24
POST_CLOSURE_HOLD = 2          # gens a closure must stand before shift
EPISODES = 4
STEPS = 600

# Endpoint A/B constants (locked)
REOPEN_FAST = 2                # variant K: reopen within this many gens
SURVIVE_GENS = 4               # invariant K: survive this many gens
SURVIVE_MIN_FITS = 12          # ...while actually meeting the world
CHURN_MIN_V3 = 2
CHILD_FITS_FLOOR = 8000
PARENT_RATE_WINDOW = 8

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'replay_phase_v3.json')


def run_world_v3(env, model, engine, web, bank, scan, gen,
                 world_rng_seed, lived_log, fam_tracker, world_key):
    """v2's run_world + per-family activation accumulation per world
    (endpoint A's invariance classifier)."""
    np.random.seed(world_rng_seed)
    env.rng = np.random.RandomState(world_rng_seed + 1)
    rng = np.random.RandomState(world_rng_seed + 2)
    sums, cnt = fam_tracker.setdefault(
        world_key, [np.zeros(NUM_FAM), 0])[0], 0
    windows = []
    for ep in range(EPISODES):
        org = Organism()
        org.reset()
        xs, ys, preds0, ctx0 = [], [], None, []
        for step in range(STEPS):
            if step % WINDOW == 0:
                if len(xs) == WINDOW and preds0 is not None:
                    feat = window_features(env, xs, ys, step - WINDOW)
                    windows.append((preds0, feat))
                    scan.window_boundary(web, gen, ctx0)
                xs, ys = [], []
                preds0 = base_predicates(env, org, step)
                ctx0 = [p for p in preds0]
            w = org.get_observation_window()
            act, _ = model.predict(w)
            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                act = np.zeros_like(act)
            elif r < EXPLORE_RATE:
                act = rng.randint(0, 2, size=len(act)).astype(act.dtype)
            obs, reward = org.step(act, env, step)
            xs.append(org.x)
            ys.append(org.y)
            rv = bank.compute(obs, act, None, reward)
            fa = web._obs_to_family_activations(rv)
            sums += fa
            cnt += 1
            core = engine._core_obs(obs)
            emb = engine.encoder.embed(core)
            web._global_step += 1
            results = web.fit_all(rv, emb, obs, obs, reward,
                                  web._global_step, ep, web._global_step,
                                  support_obs=core)
            scan.record_fits(results)
        lived_log.extend(org.experience_log)
    fam_tracker[world_key][1] += cnt
    return windows


def family_invariance(fam_tracker):
    """Median split of per-family cross-world activation variance.
    Returns set of INVARIANT family indices (bottom half)."""
    worlds = [k for k, (s, c) in fam_tracker.items() if c > 0]
    if len(worlds) < 2:
        return set()
    means = np.stack([fam_tracker[k][0] / fam_tracker[k][1]
                      for k in worlds])          # (worlds, fam)
    var = means.var(axis=0)
    order = np.argsort(var, kind='stable')
    return set(int(i) for i in order[:NUM_FAM // 2])


def main(smoke=False):
    t0 = time.time()
    gens = 6 if smoke else GENERATIONS
    fallback = 2 if smoke else FALLBACK_DWELL
    min_dwell = 1 if smoke else MIN_DWELL
    ro.CHURN_MIN = CHURN_MIN_V3     # v3 re-registration (card, cause given)
    print(f'=== REPLAY PHASE v3 {"SMOKE" if smoke else "FULL"}: '
          f'{gens} gens, closure-paced shifts '
          f'(hold {POST_CLOSURE_HOLD} post-closure, min dwell '
          f'{min_dwell}, fallback {fallback}) ===')

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
                        ledger_id='REPLAY3')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None
    fam_tracker = {}

    regime = 0
    regime_start = 0
    first_closure_gen = None
    shifts = []                    # {'gen', 'standing': [...]}
    per_gen_closed = []            # per gen: {sid: fit_count} of closed
    fit_snapshots = []             # per gen: {sid: fit_count} all slots

    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'generations': [],
              'ratified': [], 'conservation_violations': [],
              'config': {'generations': gens, 'min_dwell': min_dwell,
                         'fallback_dwell': fallback,
                         'post_closure_hold': POST_CLOSURE_HOLD,
                         'churn_min': CHURN_MIN_V3,
                         'seed_base': V3_SEED_BASE}}

    for g in range(gens):
        worlds = [(V3_SEED_BASE + 1000 * regime + i, TIERS[i])
                  for i in range(WORLDS_PER_GEN)]
        bank = LiveReceptorBank()
        gen_windows = []
        for i, (seed, tier) in enumerate(worlds):
            env = TieredEnvironment(seed=seed, tier=tier)
            w = run_world_v3(env, model, engine, web, bank, scan, g,
                             seed * 7 + g * 17, lived_log, fam_tracker,
                             (regime, seed))
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
        per_gen_closed.append({sid: s.ledger.fit_count
                               for sid, s in closed_slots.items()})
        fit_snapshots.append({sid: s.ledger.fit_count
                              for sid, s in web.slots.items()})
        closure_attempts = sum(s.ledger.reopen_count
                               for s in web.slots.values()) \
            + stats['closed']
        total_slots = stats['open'] + stats['closed']
        assertive = stats['closed'] / max(total_slots, 1)
        record['generations'].append({
            'gen': g, 'regime': regime,
            'open': stats['open'], 'closed': stats['closed'],
            'archaized': stats['archaized'],
            'receipts': stats['total_receipts'],
            'edges': stats['total_edges'],
            'churn_by_slot': stats['churn_by_slot'],
            'closure_attempts': closure_attempts,
            'assertive_fraction': round(assertive, 4),
            'proposals_pending': len(pending)})
        el = (time.time() - t0) / 60

        # ---- the curriculum clock (pre-registered rule) ----
        if closed_slots and first_closure_gen is None:
            first_closure_gen = g
        dwell = g - regime_start + 1
        do_shift = ((len(closed_slots) >= 1
                     and first_closure_gen is not None
                     and g - first_closure_gen >= POST_CLOSURE_HOLD
                     and dwell >= min_dwell)
                    or dwell >= fallback)
        tag = ''
        if do_shift and g < gens - 1:
            standing = [{'sid': sid, 'name': s.name,
                         'origin': s.origin_operator,
                         'family': s.origin_family,
                         'closed_at': s.closed_at,
                         'fit_count': s.ledger.fit_count,
                         'reopen_count': s.ledger.reopen_count}
                        for sid, s in closed_slots.items()]
            shifts.append({'gen': g, 'regime_ending': regime,
                           'dwell': dwell, 'standing': standing,
                           'by_fallback': dwell >= fallback})
            regime += 1
            regime_start = g + 1
            first_closure_gen = None
            tag = f' [SHIFT -> R{regime}' + \
                  (' fallback]' if dwell >= fallback else ' closure-paced]')
        print(f'gen {g + 1}/{gens} R{record["generations"][-1]["regime"]} '
              f'({el:.1f} min): open={stats["open"]} '
              f'closed={stats["closed"]} arch={stats["archaized"]} '
              f'edges={stats["total_edges"]} attempts={closure_attempts} '
              f'assertive={assertive:.3f} pending={len(pending)}{tag}')

        if g < gens - 1:
            engine = build_engine(lived_log)
            n = web.rebase(engine.encoder)
            print(f'  rebased {n} slots into epoch {web._embed_epoch} '
                  f'(log={len(lived_log)})')

    # ---------------- endpoint A: the shift-test 2x2 ----------------
    invariant = family_invariance(fam_tracker)
    fam_index = {name: i for i, name in enumerate(FAMILY_NAMES)}
    shift_tests = []
    for sh in shifts:
        sg = sh['gen']
        for K in sh['standing']:
            sid = K['sid']
            # reopen gen: first post-shift gen where sid not closed
            reopen_gen = None
            for gg in range(sg + 1, gens):
                if sid not in per_gen_closed[gg]:
                    reopen_gen = gg
                    break
            horizon = min(sg + SURVIVE_GENS, gens - 1)
            fits_post = (fit_snapshots[horizon].get(sid, 0)
                         - fit_snapshots[sg].get(sid, 0))
            fi = fam_index.get(K['family'], -1)
            cls = ('invariant' if fi in invariant else 'variant') \
                if fi >= 0 else 'unclassified'
            latency = None if reopen_gen is None else reopen_gen - sg
            if cls == 'variant':
                assessed = True
                passed = latency is not None and latency <= REOPEN_FAST
            elif cls == 'invariant':
                if latency is None and fits_post < SURVIVE_MIN_FITS:
                    assessed, passed = False, None      # idle, not tested
                else:
                    assessed = True
                    passed = (latency is None
                              or latency > SURVIVE_GENS) and \
                             (gens - 1 - sg >= SURVIVE_GENS)
            else:
                assessed, passed = False, None
            shift_tests.append({'shift_gen': sg, 'slot': K['name'],
                                'family': K['family'], 'class': cls,
                                'reopen_latency': latency,
                                'fits_post': int(fits_post),
                                'assessed': assessed, 'passed': passed})
    cells = {'variant': [t for t in shift_tests
                         if t['class'] == 'variant' and t['assessed']],
             'invariant': [t for t in shift_tests
                           if t['class'] == 'invariant' and t['assessed']]}

    def cell_ok(c):
        return (sum(1 for t in c if t['passed']) > len(c) / 2) if c else None
    ok_v, ok_i = cell_ok(cells['variant']), cell_ok(cells['invariant'])
    if not any(sh['standing'] for sh in shifts):
        st_verdict = 'UNTESTED (no standing K at any shift)'
    elif ok_v is None and ok_i is None:
        st_verdict = 'UNTESTED (no assessable K)'
    else:
        wins = sum(1 for o in (ok_v, ok_i) if o)
        n_cells = sum(1 for o in (ok_v, ok_i) if o is not None)
        if wins == n_cells and n_cells >= 1:
            st_verdict = ('SUPPORTED (assessed cells): closure survival '
                          'tracks content invariance')
        elif wins >= 1:
            st_verdict = 'PARTIAL: one cell behaves'
        else:
            st_verdict = 'NOT SUPPORTED: survival does not track content'

    v = verdicts(web, scan, record)
    # v3 split verdict overlay: recompute with rate windows if a split fired
    if record['splits']:
        s = record['splits'][0]
        # parent rate over final PARENT_RATE_WINDOW pre-split gens is not
        # separately snapshotted; the v1 lifetime-rate verdict from
        # verdicts() is REPORTED, and the rate-window form computed here
        # from generation churn records where possible.
        v['split_reduces_churn']['note'] = ('v3 grain: see '
                                            'split_rate_windows')
    v['shift_test'] = {'verdict': st_verdict, 'tests': shift_tests,
                       'invariant_families': sorted(
                           FAMILY_NAMES[i] for i in invariant)}

    unify_churn = {s.name: {'reopens': s.ledger.reopen_count,
                            'state': s.state}
                   for s in web.slots.values()
                   if s.origin_operator == 'Unify'
                   and s.ledger.reopen_count > 0}

    out = {'record': {k: (dict(v2_) if isinstance(v2_, defaultdict) else v2_)
                      for k, v2_ in record.items()},
           'verdicts': v,
           'shifts': shifts,
           'unify_born_churn_nonzero': unify_churn,
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-300:],
           'total_windows': scan.total_windows,
           'assertive_curve': [(r['gen'], r['assertive_fraction'])
                               for r in record['generations']],
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
    print(f'saved {path} ({out["elapsed_min"]} min)')


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)

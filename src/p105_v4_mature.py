"""P105 v4 — THE CONJUNCTION VENUE: a crowded web in a legislating
world. (Card locked at launch 2026-08-13; the successor F39
pre-registered, made cheap by F40 — 24 sandbox worlds produce 72
composed slots and 212 collision pairs, so the crowding phase costs
25 minutes, not an overnight.)

THE CLAIM (P105, registered): rendering-surfaced gaps the telemetry
has not flagged carry elevated future strain versus matched
controls. F39 established the two requirements (crowded vocabulary,
legislating world) and that no prior venue had both. This card is
the first venue with both.

DESIGN:
  PHASE A (crowding, 24 worlds — P106's growth, same seeds): grow
  the web under the standard compose economy to ~70 composed slots.
  RENDER at phase end (pure read, hash-asserted): collision set A,
  low-margin set B, round-trip set C. TELEMETRY SET (F39 impl. 7
  enacted — info-priced, not existence-based): T = slots with
  near_miss_seen ABOVE THE MEDIAN of active slots (rate-relative;
  the existence flag saturates in strain-bearing regimes and carries
  no information there). N = (A|B|C) - T; matched controls by
  fit_count (family preferred) from the un-rendered, sub-median
  pool; exhaustion amendment stands; matching quality reported
  (mean |delta fits|).
  PHASE B (strain, 12 lineage gens): the web CONTINUES into pair
  B's lineage worlds (the sandbox-to-lineage transition is itself a
  world change; it floods both sides of every pair equally — the
  PAIRED design is what survives flooding). Structure events at
  gens 0/4/8 (S=2 per lineage). Encoder rebuilt on the run's own
  lived log each gen + web.rebase (the standing lineage rhythm).
  Membership FROZEN: no compose scans, no court in phase B (check
  8 by construction).
  ORACLE (per slot over phase B): delta near_miss (primary),
  reopens, negative-fit fraction (reported).

VERDICTS (fixed): UNTESTED if pairs < 8. VOID-BY-ORACLE if pooled
strain events on paired slots < 10 (not expected — the transition
plus doses flood; stated anyway). SUPPORTED iff mean dNM(N) > mean
dNM(controls) AND wins/(wins+losses) >= 0.7. NOT SUPPORTED iff
mean(N) <= mean(controls) with a live oracle — rendering flags
carry no excess strain even where both requirements hold, and the
registered claim takes its wound honestly. PARTIAL between.

C20 (eight): 1 domain — both phases on standing machinery; the
sandbox-to-lineage transition is part of the treatment environment,
identical for all slots. 2 endpoint independence — render is a pure
read (hash-asserted); doses write the world; the oracle reads the
web. 3 exogeneity — dose schedule and phase boundary pre-registered.
4 pairing — within-web matched pairs formed at render, before any
phase-B evidence exists. 5 phenomenon strength — crowding receipted
by F40 (212 pairs, this exact growth); strain receipted by F39's
band runs (34-38k near-miss events) and the budget probe's reseed
row (the transition analog). 6 sensitivity — dNM resolution 1; the
flood puts per-slot dNM in the hundreds; the paired difference is
the measurement. 7 genesis/rates — no genesis endpoints; membership
frozen in phase B. 8 population closure — N, controls, and T fixed
at render; T is rate-relative (the F39 impl. 7 repair); floors
stated; oracle floored.
"""

import json
import os
import time

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe, interpret
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED, ScanState
from replay_overnight_v3 import run_world_v3
from law_structure import mutate_law_structure
from p105_experiment import (band_profile, roundtrip_cos, state_hash,
                             MARGIN_FLOOR, COS_GATE)
from staged_fit_experiment import Accountant, run_worlds, family_of
from train import generate_training_data, train_model

GROW_WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
LINEAGE_SEEDS = ((97000, 4), (97001, 3))
P2_GENS = 12
P2_EVENTS = (0, 4, 8)
S = 2
PAIR_FLOOR = 8
ORACLE_FLOOR = 10
WIN_FRAC = 0.7
NEG_EVENT = 0.05
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p105_v4_mature.json')


def main():
    t0 = time.time()
    print('=== P105 v4: crowded web x legislating world ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    # ---------------- PHASE A: crowding (P106's growth)
    arm = Accountant('P105v4', staged=False, consume=False)
    print('phase A: growing the crowded web (24 worlds)...')
    run_worlds(GROW_WORLDS, [arm], engine, model, 0)
    web = arm.web

    h0 = state_hash(web)
    active = {sid: s for sid, s in web.slots.items()
              if s.state in ('open', 'closed')}
    by_profile = {}
    for sid, s in active.items():
        by_profile.setdefault(band_profile(s), []).append(sid)
    set_A = {sid for p, sids in by_profile.items()
             if p != 'nothing' and len(sids) >= 2 for sid in sids}
    set_B = set()
    for sid, s in active.items():
        if s.state == 'closed':
            th = np.sort(s.geometry.family_thresholds)[::-1]
            if len(th) >= 2 and float(th[0] - th[1]) < MARGIN_FLOOR:
                set_B.add(sid)
    set_C = {sid for sid, s in active.items()
             if roundtrip_cos(s) < COS_GATE}
    nm_median = float(np.median([s.ledger.near_miss_seen
                                 for s in active.values()]))
    set_T = {sid for sid, s in active.items()
             if s.ledger.near_miss_seen > nm_median}
    rendered = set_A | set_B | set_C
    set_N = rendered - set_T
    assert state_hash(web) == h0, 'C20 check 2: render not pure'

    baseline = {sid: {'nm': active[sid].ledger.near_miss_seen,
                      'fits': active[sid].ledger.fit_count,
                      'reopens': active[sid].ledger.reopen_count,
                      'contacts': len([r for r in
                                       active[sid].ledger.receipts
                                       if r.kind == 'fit'])}
                for sid in active}
    pool = [sid for sid in active
            if sid not in rendered and sid not in set_T]
    pairs, dq = [], []
    for sid in sorted(set_N):
        if not pool:
            break
        fam_n = family_of(active[sid])
        fc_n = baseline[sid]['fits']
        pool.sort(key=lambda x: (abs(baseline[x]['fits'] - fc_n),
                                 0 if family_of(active[x]) == fam_n
                                 else 1, x))
        ctl = pool.pop(0)
        pairs.append((sid, ctl))
        dq.append(abs(baseline[ctl]['fits'] - fc_n))
    print('  render: A=%d B=%d C=%d T(median=%d)=%d N=%d pairs=%d '
          'match|dfits|=%.0f'
          % (len(set_A), len(set_B), len(set_C), nm_median,
             len(set_T), len(set_N), len(pairs),
             float(np.mean(dq)) if dq else -1))

    # ---------------- PHASE B: the legislating world
    print('phase B: lineage continuation, structure-dosed (12 gens)')
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    epoch = 0

    # real ScanState (run_world_v3 writes into it) — but replay_scans
    # is never called, so no composes/splits fire: membership frozen.
    scan = ScanState()
    rng = np.random.RandomState(6400)
    for k in range(P2_GENS):
        if k in P2_EVENTS:
            for li in range(len(lineages)):
                for _ in range(S):
                    lineages[li], d = mutate_law_structure(
                        lineages[li], rng)
            epoch += 1
            print('  >>> STRUCTURE EVENT at phase-B gen %d' % k)
        if k > 0:
            engine = build_engine(lived_log)
            web.rebase(engine.encoder)
        bank = LiveReceptorBank()
        for li, corpus in enumerate(lineages):
            env = interpret(corpus)
            run_world_v3(env, model, engine, web, bank, scan, k,
                         75000 + li * 1000 + k * 17,
                         lived_log, {}, (li, epoch))
        web.anneal_all(web._global_step)
        el = (time.time() - t0) / 60
        print('  phase-B gen %d done (%.1f min)' % (k + 1, el))

    # ---------------- oracle
    def strain(sid):
        s = web.slots.get(sid)
        if s is None or s.state not in ('open', 'closed'):
            return None
        b = baseline.get(sid)
        if b is None:
            return None
        fits = [r for r in s.ledger.receipts if r.kind == 'fit']
        post = fits[b['contacts']:]
        neg = (sum(1 for r in post if r.sign < 0) / len(post)
               if post else 0.0)
        return {'d_nm': s.ledger.near_miss_seen - b['nm'],
                'd_reopen': s.ledger.reopen_count - b['reopens'],
                'neg_frac': round(neg, 4)}

    rows, wins, ties, losses, oracle = [], 0, 0, 0, 0
    for nsid, csid in pairs:
        sn, sc = strain(nsid), strain(csid)
        if sn is None or sc is None:
            continue
        rows.append({'novel_slot': nsid, 'control_slot': csid,
                     'novel': sn, 'control': sc,
                     'novel_classes': [k for k, ss in
                                       (('A', set_A), ('B', set_B),
                                        ('C', set_C)) if nsid in ss]})
        oracle += sn['d_nm'] + sc['d_nm'] + sn['d_reopen'] \
            + sc['d_reopen']
        oracle += int(sn['neg_frac'] > NEG_EVENT)
        oracle += int(sc['neg_frac'] > NEG_EVENT)
        if sn['d_nm'] > sc['d_nm']:
            wins += 1
        elif sn['d_nm'] == sc['d_nm']:
            ties += 1
        else:
            losses += 1

    mean_n = (float(np.mean([r['novel']['d_nm'] for r in rows]))
              if rows else None)
    mean_c = (float(np.mean([r['control']['d_nm'] for r in rows]))
              if rows else None)
    decided = wins + losses
    wf = wins / decided if decided else None

    if len(rows) < PAIR_FLOOR:
        verdict = 'UNTESTED (pairs=%d < %d)' % (len(rows), PAIR_FLOOR)
    elif oracle < ORACLE_FLOOR:
        verdict = ('VOID-BY-ORACLE (strain events %d < %d)'
                   % (oracle, ORACLE_FLOOR))
    elif mean_n > mean_c and decided and wf >= WIN_FRAC:
        verdict = ('SUPPORTED: rendering-flagged slots take excess '
                   'strain in the conjunction venue (dNM %.1f vs '
                   '%.1f, wins %.2f, oracle %d)'
                   % (mean_n, mean_c, wf, oracle))
    elif mean_n <= mean_c:
        verdict = ('NOT SUPPORTED: dNM %.1f vs %.1f with live oracle '
                   '(%d events) — the registered claim takes its '
                   'wound in the first venue with both requirements'
                   % (mean_n, mean_c, oracle))
    else:
        verdict = ('PARTIAL: dNM %.1f vs %.1f, wins %.2f < %.2f '
                   '(oracle %d)' % (mean_n, mean_c, wf or 0.0,
                                    WIN_FRAC, oracle))
    print('\nP105 v4 VERDICT: %s' % verdict)

    out = {'sets': {'A': sorted(set_A), 'B': sorted(set_B),
                    'C': sorted(set_C), 'T': sorted(set_T),
                    'N': sorted(set_N)},
           'nm_median_at_render': nm_median,
           'pairs': rows, 'oracle_mass': oracle,
           'means': {'novel_d_nm': mean_n, 'control_d_nm': mean_c,
                     'wins': wins, 'ties': ties, 'losses': losses},
           'match_quality_mean_dfits': (float(np.mean(dq))
                                        if dq else None),
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

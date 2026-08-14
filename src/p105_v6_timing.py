"""P105 v6 — class-closed pairs, time-to-death oracle. (Card locked
at launch 2026-08-13; F41's pre-registered successor.)

V5 DISPOSITION: printed SUPPORTED, corrected VOID-BY-CLASS-CONFOUND
(F41): 20/27 pairs crossed the eviction-exemption class boundary —
immortal controls. The class-fair cell showed ~95% composed
mortality on BOTH sides: binary death saturates; the discriminator,
if any, is TIMING.

V6 CHANGES (same growth, same phase B, same seeds):
  1. CLASS-CLOSED PAIRS: composed-vs-composed only. N* = flagged
     composed (rendering-flagged, telemetry-clean); controls =
     UNFLAGGED composed (sub-median telemetry), matched on
     fit_count, without replacement. Floor: pairs >= 6.
  2. GRADED ORACLE: per-generation tracking; death_gen = first
     phase-B gen at which the slot left the active set (alive at end
     = censored at P2_GENS + 1).
  3. ENDPOINT: flagged composed die EARLIER. Pairwise: N dies
     strictly earlier = win, later = loss, same gen (or both
     censored) = tie. SUPPORTED iff decided pairs >= 5 AND wins /
     decided >= 0.7. NOT SUPPORTED: decided >= 5 and wins/decided
     <= 0.3. UNTESTED: decided < 5 (deaths too simultaneous to rank
     — reported with the death-gen histogram; the venue's lethality
     clock is then itself the finding).

C20 (eight): as v5 with check 8 carrying BOTH same-day clauses
(constant-dominated channels; class-closed pairing). Check 6: death
timing resolution is 1 gen; v5 showed deaths spread across the
early phase (N-alive 27 -> 2 over 12 gens, not one cliff — the
per-gen print is the receipt).
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
from staged_fit_experiment import Accountant, run_worlds
from train import generate_training_data, train_model

GROW_WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
LINEAGE_SEEDS = ((97000, 4), (97001, 3))
P2_GENS = 12
P2_EVENTS = (0, 4, 8)
S = 2
INHERITED_MAX = 32
PAIR_FLOOR = 6
DECIDED_FLOOR = 5
WIN_FRAC = 0.7
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p105_v6_timing.json')


def main():
    t0 = time.time()
    print('=== P105 v6: class-closed pairs, time-to-death ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('P105v6', staged=False, consume=False)
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
    assert state_hash(web) == h0, 'C20 check 2: render not pure'

    composed = {sid for sid in active if sid > INHERITED_MAX}
    flagged = sorted((rendered - set_T) & composed)
    unflagged = [sid for sid in composed
                 if sid not in rendered and sid not in set_T]
    fits = {sid: active[sid].ledger.fit_count for sid in active}
    pairs = []
    pool = list(unflagged)
    for sid in flagged:
        if not pool:
            break
        pool.sort(key=lambda x: (abs(fits[x] - fits[sid]), x))
        pairs.append((sid, pool.pop(0)))
    print('  render: flagged-composed=%d unflagged-composed=%d '
          'pairs=%d' % (len(flagged), len(unflagged), len(pairs)))

    print('phase B: lineage continuation, structure-dosed (12 gens)')
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    epoch = 0
    scan = ScanState()
    rng = np.random.RandomState(6400)
    tracked = sorted({x for p in pairs for x in p})
    death_gen = {}
    for k in range(P2_GENS):
        if k in P2_EVENTS:
            for li in range(len(lineages)):
                for _ in range(S):
                    lineages[li], d = mutate_law_structure(
                        lineages[li], rng)
            epoch += 1
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
        for sid in tracked:
            if sid in death_gen:
                continue
            s = web.slots.get(sid)
            if s is None or s.state not in ('open', 'closed'):
                death_gen[sid] = k + 1
        alive = len(tracked) - len(death_gen)
        print('  phase-B gen %d (%.1f min): tracked alive %d/%d'
              % (k + 1, (time.time() - t0) / 60, alive, len(tracked)))

    censored = P2_GENS + 1
    rows, wins, ties, losses = [], 0, 0, 0
    for nsid, csid in pairs:
        dn = death_gen.get(nsid, censored)
        dc = death_gen.get(csid, censored)
        rows.append({'flagged_slot': nsid, 'control_slot': csid,
                     'flagged_death_gen': dn, 'control_death_gen': dc})
        if dn < dc:
            wins += 1
        elif dn > dc:
            losses += 1
        else:
            ties += 1
    decided = wins + losses
    wf = wins / decided if decided else None
    hist = {}
    for sid in tracked:
        g = death_gen.get(sid, censored)
        hist[g] = hist.get(g, 0) + 1
    print('  death-gen histogram (censored=%d): %s'
          % (censored, dict(sorted(hist.items()))))

    if len(pairs) < PAIR_FLOOR:
        verdict = 'UNTESTED (pairs=%d < %d)' % (len(pairs), PAIR_FLOOR)
    elif decided < DECIDED_FLOOR:
        verdict = ('UNTESTED (decided=%d < %d — deaths too '
                   'simultaneous to rank; the lethality clock is the '
                   'finding)' % (decided, DECIDED_FLOOR))
    elif wf >= WIN_FRAC:
        verdict = ('SUPPORTED: flagged composed die earlier '
                   '(%d/%d decided, %.2f)' % (wins, decided, wf))
    elif wf <= 1 - WIN_FRAC:
        verdict = ('NOT SUPPORTED: controls die earlier or equal '
                   '(%d/%d, %.2f)' % (wins, decided, wf))
    else:
        verdict = 'PARTIAL (%d/%d decided, %.2f)' % (wins, decided, wf)
    print('\nP105 v6 VERDICT: %s' % verdict)

    out = {'pairs': rows, 'death_histogram': hist,
           'flagged_n': len(flagged), 'unflagged_n': len(unflagged),
           'wins': wins, 'ties': ties, 'losses': losses,
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

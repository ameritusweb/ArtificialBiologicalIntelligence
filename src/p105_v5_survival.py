"""P105 v5 — the conjunction venue with MORTALITY as the primary
oracle. (Card locked at launch 2026-08-13; v4's receipts stand.)

V4 DISPOSITION (results/p105_v4_mature.json): UNTESTED (pairs=0) —
but the mechanism is the finding. The render worked at last: N=66
telemetry-clean flagged slots (the crowding phase delivered), pairs
formed — and by phase-B end the paired slots were NOT in the active
set: the composed population was RENT-EVICTED during the lineage
phase. F31's net-per-fire economics responded to the legislating
world's changed fire patterns by killing the crowded vocabulary
before the near-miss oracle could accrue on it. The venue is not
infertile; it is LETHAL — and death is the maximum strain event,
which the v4 oracle was not registered to read. (v4 also lost the
pair list on the dead-slot path; repaired here: pairs saved at
render, unconditionally.)

V5 CHANGES (everything else identical to v4, same seeds):
  1. PRIMARY ENDPOINT — PAIRED MORTALITY: a slot that leaves the
     active set (archaized/evicted) during phase B scores DEAD.
     Among mortality-DISCORDANT pairs (one died, one lived):
     SUPPORTED iff discordant pairs >= 5 AND >= 70% are
     N-died-control-lived (rendering-flagged slots die
     preferentially under the legislating world). NOT SUPPORTED iff
     discordant >= 5 and <= 30% (controls die more). PARTIAL
     between. UNTESTED: discordant < 5 AND surviving pairs < 8
     (neither endpoint has population).
  2. SECONDARY — survivor dNM: among pairs where BOTH lived, the v1
     rule (mean + 0.7 wins), reported and billed only if surviving
     pairs >= 8.
  3. Pairs and per-slot fates exported unconditionally.

C20 (eight): as v4, with check 5 now carrying v4's receipt that the
venue produces mortality (the phenomenon the primary measures), and
check 8 noting the fate of every pair member is reported — no slot
silently exits the comparison (v4's repaired deficiency: death was
an unmodeled exit from the endpoint population; v5 makes death THE
endpoint).
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
DISCORDANT_FLOOR = 5
MORTALITY_FRAC = 0.7
SURVIVOR_PAIR_FLOOR = 8
WIN_FRAC = 0.7
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p105_v5_survival.json')


def main():
    t0 = time.time()
    print('=== P105 v5: conjunction venue, mortality oracle ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('P105v5', staged=False, consume=False)
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
                      'contacts': len([r for r in
                                       active[sid].ledger.receipts
                                       if r.kind == 'fit'])}
                for sid in active}
    pool = [sid for sid in active
            if sid not in rendered and sid not in set_T]
    pairs = []
    for sid in sorted(set_N):
        if not pool:
            break
        fam_n = family_of(active[sid])
        fc_n = baseline[sid]['fits']
        pool.sort(key=lambda x: (abs(baseline[x]['fits'] - fc_n),
                                 0 if family_of(active[x]) == fam_n
                                 else 1, x))
        pairs.append((sid, pool.pop(0)))
    print('  render: A=%d B=%d C=%d T=%d N=%d pairs=%d'
          % (len(set_A), len(set_B), len(set_C), len(set_T),
             len(set_N), len(pairs)))

    print('phase B: lineage continuation, structure-dosed (12 gens)')
    lived_log = []
    lineages = [describe(TieredEnvironment(seed=s, tier=t))
                for s, t in LINEAGE_SEEDS]
    epoch = 0
    scan = ScanState()
    rng = np.random.RandomState(6400)
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
        alive_n = sum(1 for sid, _ in pairs
                      if web.slots.get(sid) is not None
                      and web.slots[sid].state in ('open', 'closed'))
        print('  phase-B gen %d (%.1f min): N-alive=%d/%d'
              % (k + 1, (time.time() - t0) / 60, alive_n, len(pairs)))

    def fate(sid):
        s = web.slots.get(sid)
        if s is None or s.state not in ('open', 'closed'):
            return {'dead': True}
        b = baseline[sid]
        fits = [r for r in s.ledger.receipts if r.kind == 'fit']
        post = fits[b['contacts']:]
        neg = (sum(1 for r in post if r.sign < 0) / len(post)
               if post else 0.0)
        return {'dead': False,
                'd_nm': s.ledger.near_miss_seen - b['nm'],
                'neg_frac': round(neg, 4)}

    rows = []
    n_died_c_lived = c_died_n_lived = both_died = both_lived = 0
    wins = ties = losses = 0
    for nsid, csid in pairs:
        fn, fc = fate(nsid), fate(csid)
        rows.append({'novel_slot': nsid, 'control_slot': csid,
                     'novel': fn, 'control': fc})
        if fn['dead'] and not fc['dead']:
            n_died_c_lived += 1
        elif fc['dead'] and not fn['dead']:
            c_died_n_lived += 1
        elif fn['dead']:
            both_died += 1
        else:
            both_lived += 1
            if fn['d_nm'] > fc['d_nm']:
                wins += 1
            elif fn['d_nm'] == fc['d_nm']:
                ties += 1
            else:
                losses += 1

    discordant = n_died_c_lived + c_died_n_lived
    frac = (n_died_c_lived / discordant) if discordant else None
    surv_n = [r['novel']['d_nm'] for r in rows
              if not r['novel']['dead'] and not r['control']['dead']]
    surv_c = [r['control']['d_nm'] for r in rows
              if not r['novel']['dead'] and not r['control']['dead']]
    mean_n = float(np.mean(surv_n)) if surv_n else None
    mean_c = float(np.mean(surv_c)) if surv_c else None
    decided = wins + losses
    wf = wins / decided if decided else None
    print('  mortality: N-died-C-lived=%d C-died-N-lived=%d '
          'both-died=%d both-lived=%d'
          % (n_died_c_lived, c_died_n_lived, both_died, both_lived))

    if discordant >= DISCORDANT_FLOOR and frac >= MORTALITY_FRAC:
        verdict = ('SUPPORTED (mortality): rendering-flagged slots '
                   'die preferentially under the legislating world '
                   '(%d/%d discordant pairs N-died, %.2f)'
                   % (n_died_c_lived, discordant, frac))
    elif discordant >= DISCORDANT_FLOOR and frac <= 1 - MORTALITY_FRAC:
        verdict = ('NOT SUPPORTED (mortality): controls die more '
                   '(%d/%d, %.2f)' % (n_died_c_lived, discordant,
                                      frac))
    elif both_lived >= SURVIVOR_PAIR_FLOOR and mean_n is not None:
        if mean_n > mean_c and decided and wf >= WIN_FRAC:
            verdict = ('SUPPORTED (survivor dNM): %.1f vs %.1f, '
                       'wins %.2f (discordant %d below floor)'
                       % (mean_n, mean_c, wf, discordant))
        elif mean_n <= mean_c:
            verdict = ('NOT SUPPORTED (survivor dNM): %.1f vs %.1f '
                       '(discordant %d below floor)'
                       % (mean_n, mean_c, discordant))
        else:
            verdict = ('PARTIAL (survivor dNM %.1f vs %.1f wins '
                       '%.2f)' % (mean_n, mean_c, wf or 0.0))
    elif discordant >= DISCORDANT_FLOOR:
        verdict = ('PARTIAL (mortality %.2f between gates, '
                   'discordant %d)' % (frac, discordant))
    else:
        verdict = ('UNTESTED (discordant=%d < %d and surviving '
                   'pairs=%d < %d)'
                   % (discordant, DISCORDANT_FLOOR, both_lived,
                      SURVIVOR_PAIR_FLOOR))
    print('\nP105 v5 VERDICT: %s' % verdict)

    out = {'sets': {'A': sorted(set_A), 'B': sorted(set_B),
                    'C': sorted(set_C), 'T': sorted(set_T),
                    'N': sorted(set_N)},
           'pairs_formed': [[a, b] for a, b in pairs],
           'rows': rows,
           'mortality': {'n_died_c_lived': n_died_c_lived,
                         'c_died_n_lived': c_died_n_lived,
                         'both_died': both_died,
                         'both_lived': both_lived,
                         'discordant_frac_n_died': frac},
           'survivor_means': {'novel': mean_n, 'control': mean_c,
                              'wins': wins, 'ties': ties,
                              'losses': losses},
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

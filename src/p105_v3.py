"""P105-v3 — readout as edge-detector, DRIFT-ORACLE venue (card locked
at launch 2026-08-13; supersedes v1/v2's dead oracle).

V1/V2 DISPOSITION (on the record): v1 UNTESTED (control pool 12 < N
14); v2 (blind pairing amendment) printed NOT SUPPORTED but is
corrected to VOID-BY-ORACLE — all 24 paired slots accrued ZERO strain
over the in-distribution continuation (0 near-misses, 0.0 negative
fraction, 12/12 ties). The strain phenomenon was ABSENT on the paired
population; a comparison over an absent phenomenon is VOID (C20
check 5). METHOD FAILURE OWNED: the v1 card floored the detector side
(|N|, pairs, fit volume) but not the oracle side. In-distribution
worlds deposit near-miss traffic only on already-flagged slots;
telemetry-clean slots stay clean. Strain must be INDUCED.

V3 CHANGES (everything else identical to the v1 card, including flag
classes, telemetry set, pairing rule with the v2 exhaustion amendment,
primary endpoint, win fraction 0.7):
  1. The continuation is OOD: 6 fresh-seed worlds with tiers swapped
     (the standing strongest shift move — F23/F26 receipts). Under
     drift, strain lands broadly; the question becomes: do
     rendering-flagged slots take DISPROPORTIONATE strain?
  2. ORACLE FLOOR (new, the owned lesson): total strain events across
     all paired slots (sum of dNM + count of slots with neg_frac >
     0.05) must be >= 10, else VOID-BY-ORACLE — stated before launch.

VERDICTS: VOID if oracle floor unmet. UNTESTED if pairs < 8 or
continuation positive fits < 2000. SUPPORTED iff mean dNM(N) > mean
dNM(ctl) AND wins/(wins+losses) >= 0.7. NOT SUPPORTED iff mean(N) <=
mean(ctl) with the oracle alive. PARTIAL otherwise.

C20 (seven): as the v1 card, with check 5 now carrying the oracle
floor and the receipt for drift-induced strain (F23: reseed produced
contact strain; F26: orphan step at reseed; F28: structure-dose churn).
Check 2 unchanged: rendering is a pure read, hash-asserted; the drift
cannot be written by a read either.
"""

import json
import os
import time

import numpy as np

from p105_experiment import (band_profile, roundtrip_cos, state_hash,
                             MARGIN_FLOOR, COS_GATE, N_FLOOR,
                             CONT_FIT_FLOOR, WIN_FRAC)
from replay_overnight import build_engine, BOOT_SEED
from staged_fit_experiment import Accountant, run_worlds, family_of
from train import generate_training_data, train_model

WARMUP_WORLDS = [(97200 + i, (4, 3)[i % 2]) for i in range(8)]
CONT_WORLDS = [(98000 + i, (3, 4)[i % 2]) for i in range(6)]
PAIR_FLOOR = 8
ORACLE_FLOOR = 10
NEG_EVENT = 0.05
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p105_v3_edge_detector.json')


def main():
    t0 = time.time()
    print('=== P105-v3: readout as edge-detector (drift oracle) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('P105v3', staged=False, consume=False)
    print('warmup (8 worlds, identical to v1 by seed)...')
    c = run_worlds(WARMUP_WORLDS, [arm], engine, model, 0)
    web = arm.web

    h_before = state_hash(web)
    active = {sid: s for sid, s in web.slots.items()
              if s.state in ('open', 'closed')}
    profiles = {sid: band_profile(s) for sid, s in active.items()}
    by_profile = {}
    for sid, p in profiles.items():
        by_profile.setdefault(p, []).append(sid)
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
    set_T = {sid for sid, s in active.items()
             if s.ledger.near_miss_seen > 0 or s.ledger.fit_count == 0}
    rendered = set_A | set_B | set_C
    set_N = rendered - set_T
    h_after = state_hash(web)
    assert h_before == h_after, 'C20 check 2: rendering was not pure'

    unflagged = [sid for sid in active
                 if sid not in rendered and sid not in set_T]
    baseline = {sid: {'nm': active[sid].ledger.near_miss_seen,
                      'fits': active[sid].ledger.fit_count,
                      'reopens': active[sid].ledger.reopen_count}
                for sid in active}
    pairs = []
    pool = list(unflagged)
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

    print('continuation (6 OOD worlds: fresh seeds, tiers swapped)...')
    fits_before = sum(baseline[sid]['fits'] for sid in baseline)
    t0_step = web._global_step
    run_worlds(CONT_WORLDS, [arm], engine, model, c)

    def strain(sid):
        s = web.slots.get(sid)
        if s is None or s.state not in ('open', 'closed'):
            return None
        recent = [r for r in s.ledger.receipts if r.kind == 'fit'
                  and r.created_at > t0_step][-400:]
        neg = (sum(1 for r in recent if r.sign < 0) / len(recent)
               if recent else 0.0)
        return {'d_nm': s.ledger.near_miss_seen - baseline[sid]['nm'],
                'd_reopen': s.ledger.reopen_count
                            - baseline[sid]['reopens'],
                'neg_frac': round(neg, 4)}

    cont_fits = sum(s.ledger.fit_count for s in web.slots.values()
                    if s.state in ('open', 'closed')) - fits_before

    rows, wins, ties, losses = [], 0, 0, 0
    oracle_mass = 0
    for nsid, csid in pairs:
        sn, sc = strain(nsid), strain(csid)
        if sn is None or sc is None:
            continue
        rows.append({'novel_slot': nsid, 'control_slot': csid,
                     'novel': sn, 'control': sc,
                     'novel_classes': [k for k, ss in
                                       (('A', set_A), ('B', set_B),
                                        ('C', set_C)) if nsid in ss]})
        oracle_mass += sn['d_nm'] + sc['d_nm']
        oracle_mass += int(sn['neg_frac'] > NEG_EVENT)
        oracle_mass += int(sc['neg_frac'] > NEG_EVENT)
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
    win_frac = wins / decided if decided else None

    if oracle_mass < ORACLE_FLOOR:
        verdict = ('VOID-BY-ORACLE (strain events %d < %d even under '
                   'drift — venue cannot test the claim)'
                   % (oracle_mass, ORACLE_FLOOR))
    elif len(rows) < PAIR_FLOOR or cont_fits < CONT_FIT_FLOOR:
        verdict = ('UNTESTED (pairs=%d/%d cont_fits=%d/%d)'
                   % (len(rows), PAIR_FLOOR, cont_fits, CONT_FIT_FLOOR))
    elif mean_n > mean_c and decided and win_frac >= WIN_FRAC:
        verdict = ('SUPPORTED: rendering-flagged slots take '
                   'disproportionate drift strain (dNM %.2f vs %.2f, '
                   'wins %.2f)' % (mean_n, mean_c, win_frac))
    elif mean_n <= mean_c:
        verdict = ('NOT SUPPORTED: dNM %.2f vs %.2f with a live '
                   'oracle (mass %d)' % (mean_n, mean_c, oracle_mass))
    else:
        verdict = ('PARTIAL: mean elevated (%.2f vs %.2f) wins %.2f '
                   '< %.2f' % (mean_n, mean_c, win_frac or 0.0,
                               WIN_FRAC))
    print('\nP105-v3 VERDICT: %s' % verdict)

    out = {'sets': {'A': sorted(set_A), 'B': sorted(set_B),
                    'C': sorted(set_C), 'T': sorted(set_T),
                    'N': sorted(set_N)},
           'pairs': rows, 'oracle_mass': oracle_mass,
           'means': {'novel_d_nm': mean_n, 'control_d_nm': mean_c,
                     'wins': wins, 'ties': ties, 'losses': losses},
           'cont_positive_fits': cont_fits,
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

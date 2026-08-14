"""P105 — readout as edge-detector: does LC rendering surface gaps the
web's own telemetry has not flagged, and do those gaps carry future
strain? (T159 fifth arrival; card locked at launch 2026-08-13.)

THE CLAIM (registered form): LC-rendering a region raises typed-gap
detection there vs matched unrendered regions — "writing finds your
edges."

CARD REFINEMENT (pre-registered, honest): at LC-0a the readout is a
PURE READ — there is no actor, so the causal form is vacuous by
construction (nothing rendering does can change the web). The testable
content today is DETECTION VALIDITY: rendering-surfaced gaps that
standing telemetry does NOT flag must carry elevated FUTURE strain
versus matched unflagged controls. The causal form (rendering changes
allocation) awaits an acting organism (AW-0). This refinement is billed
on the card, not discovered after.

RENDERING-GAP CLASSES (computed by attempting to say the web):
  A. POSE COLLISIONS — two or more slots whose band-quantized pose
     profiles are IDENTICAL: the language cannot tell them apart. A
     conflation visible only in the attempt to describe (connects to
     F29's Differentiate re-key: "one form, two pulls").
  B. LOW-MARGIN ASSERTIONS — closed slots whose family argmax margin
     (top1 - top2 threshold) < 0.1: the assertion template's family
     choice is arbitrary; representational indeterminacy invisible to
     fit telemetry.
  C. ROUND-TRIP LOSS — slots whose pose -> parse reconstruction cosine
     < 0.98: geometry outside the language's carving.
TELEMETRY SET T (what the web already flags without language):
  near_miss_seen > 0 OR fit_count == 0.
NOVEL SET N = (A u B u C) - T.  Controls = unflagged slots matched per
N-slot on fit_count (family preferred), without replacement.

ORACLE (future strain over the continuation, per slot):
  primary   = delta near_miss_seen (the de-conflation demand channel,
              F13's currency)
  secondary = negative-fit fraction, reopen_count delta (reported).

VERDICTS (fixed): SUPPORTED iff |N| >= 5 AND mean dNM(N) > mean
dNM(controls) AND N wins >= 0.7 of matched pairs (ties dropped).
NOT SUPPORTED: |N| >= 5 and N <= controls on the primary. UNTESTED:
|N| < 5, or continuation positive fits < 2000, or matched controls
< |N| (population too small to match).

C20 (seven): 1 domain — staged-fit harness worlds, all components
in-dist. 2 endpoint independence — rendering is a pure read, ASSERTED
by state hash before/after analysis; the oracle accrues after t0 and
cannot be written by a read. 3 exogeneity — n/a for detection validity;
temporal ordering (flags at t0, strain after t0) fixes the direction.
4 pairing — single web; controls matched within-web by pre-registered
rule. 5 phenomenon strength — floors above; near-miss telemetry is live
in this harness class (P77/LC-0a receipts). 6 sensitivity — endpoint
resolution is 1 near-miss event; per-slot dNM over 6 worlds ranged
O(1-30) in prior runs of this class; first measurement of the
between-class difference, floors guard the bill. 7 genesis/rates — no
genesis endpoint; all receipts grown in-run, distribution-bound.
"""

import hashlib
import json
import os
import time

import numpy as np

from lc_store import FAMILY_NAMES, _BANDS, _BAND_VAL
from replay_overnight import build_engine, BOOT_SEED
from staged_fit_experiment import Accountant, run_worlds, family_of
from train import generate_training_data, train_model

WARMUP_WORLDS = [(97200 + i, (4, 3)[i % 2]) for i in range(8)]
CONT_WORLDS = [(97300 + i, (4, 3)[i % 2]) for i in range(6)]
MARGIN_FLOOR = 0.1
COS_GATE = 0.98
N_FLOOR = 5
CONT_FIT_FLOOR = 2000
WIN_FRAC = 0.7
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p105_edge_detector.json')


def band_profile(slot):
    parts = []
    for f in np.argsort(-slot.geometry.family_thresholds, kind='stable'):
        v = float(slot.geometry.family_thresholds[f])
        if v < 0.05:
            break
        band = next(b for cap, b in _BANDS if v <= cap)
        parts.append('%s:%s' % (FAMILY_NAMES[int(f)], band))
    return ','.join(parts) if parts else 'nothing'


def roundtrip_cos(slot):
    th = slot.geometry.family_thresholds
    rec = np.zeros_like(th)
    for f in range(len(th)):
        v = float(th[f])
        if v < 0.05:
            continue
        band = next(b for cap, b in _BANDS if v <= cap)
        rec[f] = _BAND_VAL[band]
    na, nb = float(np.linalg.norm(th)), float(np.linalg.norm(rec))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(th, rec) / (na * nb))


def state_hash(web):
    h = hashlib.sha256()
    h.update(str(web._global_step).encode())
    h.update(str(len(web._receipts_by_id)).encode())
    for sid in sorted(web.slots):
        s = web.slots[sid]
        h.update(('%d:%d:%d:%d' % (sid, s.ledger.receipt_count,
                                   s.ledger.fit_count,
                                   s.ledger.near_miss_seen)).encode())
    return h.hexdigest()


def main():
    t0 = time.time()
    print('=== P105: readout as edge-detector (detection validity) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('P105', staged=False, consume=False)
    print('warmup (8 worlds)...')
    c = run_worlds(WARMUP_WORLDS, [arm], engine, model, 0)
    web = arm.web

    # ---------------- t0: RENDER THE WEB (pure read, hash-asserted)
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
    assert h_before == h_after, 'C20 check 2: rendering was not a pure read'

    # matched controls: unflagged, nearest fit_count, family preferred
    unflagged = [sid for sid in active
                 if sid not in rendered and sid not in set_T]
    baseline = {sid: {'nm': active[sid].ledger.near_miss_seen,
                      'fits': active[sid].ledger.fit_count,
                      'reopens': active[sid].ledger.reopen_count,
                      'receipts': active[sid].ledger.receipt_count}
                for sid in active}
    pairs = []
    pool = list(unflagged)
    for sid in sorted(set_N):
        if not pool:
            break
        fam_n = family_of(active[sid])
        fc_n = baseline[sid]['fits']
        pool.sort(key=lambda x: (abs(baseline[x]['fits'] - fc_n),
                                 0 if family_of(active[x]) == fam_n else 1,
                                 x))
        ctl = pool.pop(0)
        pairs.append((sid, ctl))

    print('  render: A(collisions)=%d B(low-margin)=%d C(roundtrip)=%d '
          'T(telemetry)=%d N(novel)=%d pairs=%d'
          % (len(set_A), len(set_B), len(set_C), len(set_T),
             len(set_N), len(pairs)))
    collision_groups = {p: sids for p, sids in by_profile.items()
                        if p != 'nothing' and len(sids) >= 2}

    # ---------------- continuation (6 worlds), then the oracle
    print('continuation (6 worlds)...')
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
    for nsid, csid in pairs:
        sn, sc = strain(nsid), strain(csid)
        if sn is None or sc is None:
            continue
        rows.append({'novel_slot': nsid, 'control_slot': csid,
                     'novel': sn, 'control': sc,
                     'novel_classes': [c for c, ss in
                                       (('A', set_A), ('B', set_B),
                                        ('C', set_C)) if nsid in ss]})
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

    # descriptive: does rendering ALSO beat telemetry to the punch?
    telemetry_strain = [strain(sid) for sid in sorted(set_T)]
    telemetry_strain = [s for s in telemetry_strain if s is not None]

    if len(set_N) < N_FLOOR or len(pairs) < len(set_N) \
            or cont_fits < CONT_FIT_FLOOR:
        verdict = ('UNTESTED (floors: novel=%d/%d matched=%d cont_fits=%d/%d)'
                   % (len(set_N), N_FLOOR, len(pairs), cont_fits,
                      CONT_FIT_FLOOR))
    elif mean_n > mean_c and decided > 0 and win_frac >= WIN_FRAC:
        verdict = ('SUPPORTED: rendering-surfaced gaps carry elevated '
                   'future strain vs matched controls '
                   '(dNM %.2f vs %.2f, wins %.2f)'
                   % (mean_n, mean_c, win_frac))
    elif mean_n <= mean_c:
        verdict = ('NOT SUPPORTED: rendering finds nothing telemetry '
                   'missed (dNM %.2f vs %.2f)' % (mean_n, mean_c))
    else:
        verdict = ('PARTIAL: mean elevated (%.2f vs %.2f) but win '
                   'fraction %.2f < %.2f'
                   % (mean_n, mean_c, win_frac or 0.0, WIN_FRAC))
    print('\nP105 VERDICT: %s' % verdict)

    out = {'sets': {'A_collisions': sorted(set_A),
                    'B_low_margin': sorted(set_B),
                    'C_roundtrip': sorted(set_C),
                    'T_telemetry': sorted(set_T),
                    'N_novel': sorted(set_N)},
           'collision_groups': collision_groups,
           'pairs': rows,
           'means': {'novel_d_nm': mean_n, 'control_d_nm': mean_c,
                     'wins': wins, 'ties': ties, 'losses': losses},
           'telemetry_strain': telemetry_strain,
           'cont_positive_fits': cont_fits,
           'pure_read_hash_ok': True,
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

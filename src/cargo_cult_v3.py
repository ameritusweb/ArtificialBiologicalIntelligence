"""P103(cargo-cult) v3 — WORLD-GAP testimony. (Card locked at launch
2026-08-13; v1/v2 receipts stand.)

V2 DISPOSITION (results/p103_cargo_cult_v2.json, on the record):
earned NOT SUPPORTED with a dose-response INVERSION — truthful
same-world testimony believed at face value IMPROVES first-contact
calibration monotonically with dose (dose-0 internal control gap
-0.001 ~ 0, validating pairing at slot grain; dose-0.5 gap -0.044;
dose-1.0 gap -0.059). Mechanism: these worlds never move against
inherited content (F26 mutation-robust Ks), confirm rates stay high,
the base certainty formula understates, and true testimony corrects
it. THE PATHOLOGY CANNOT BE IMPORTING PER SE. The cargo-cult lesion
requires a WORLD GAP between donor and receiver (or outright false
testimony) — T158's trust thread: the danger is the exchange rate,
not the trade.

V3 DESIGN (one change class: the donor lives a DIFFERENT WORLD):
  DONOR PRE-GROW: a separate web lives 10 worlds at tiers (7,6) —
  richer physics than the receiver's (4,3) — and BANKS its per-slot
  positive fit receipts (capped 2000/slot) plus an end-state geometry
  snapshot. Its testimony is TRUE OF ITS WORLD and delivered with
  full sincerity; the gap is distributional, not adversarial.
  RECEIVER PHASES as v2 (same seeds/doses): warmup 4 / diet 6 /
  shift 4. On attested steps, dosed slots import ONE banked receipt
  (round-robin per slot) via attest with the donor's snapshot
  geometry; lesion arm additionally credits fit_count + certainty
  toward 0.9 (unchanged).
  NEW SECONDARY (the firewall's real test): the intact arm's
  centroid_pull now imports WRONG-world geometry — report
  first-contact tightness per arm. The firewall protects confidence,
  not geometry; geometric poisoning through lawful attest is a
  separate exposure, measured here for the first time.

PRE-REGISTERED READINGS (both directions): if the lesion first-contact
gap flips POSITIVE (>= +0.02) with dose rho >= +0.3, the cargo-cult
signature is a function of donor-receiver world divergence — the
mature-domain feed's danger is the distribution gap (T159 fifth
arrival). If the gap stays NEGATIVE even under world-gap testimony,
the inversion is universal in this harness class and the pathology
requires adversarial (false) testimony — billed as such, and the v4
would corrupt exports.

VERDICTS (fixed): SUPPORTED iff lesion-minus-lived first-contact
calibration gap >= +0.02 AND dose rho >= +0.3 AND intact firewall
band <= 0.01 on calibration. NOT SUPPORTED: gap <= 0 (inversion
survives world-gap). PARTIAL: between. UNTESTED: transfers < 200 per
attest arm, full-dose lesion slots with >= 5 transfers < 8, slots
with >= 100 first-window fits < 20, or donor bank < 50 slots with
>= 10 banked receipts (gap-testimony phenomenon floor).

C20 (seven): 1 domain — the policy model is a stream generator; webs
measure their own geometry against their own lived stream; the donor's
tier-7 stream is out-of-policy-domain but the donor web's receipts
are lawful lived evidence OF ITS WORLD (that is the treatment).
2 endpoint independence — as v2; the confirm side of first-contact
error is unwritable by imports; tightness measured on fresh lived
fits. 3 exogeneity — donor world assignment is the manipulated
variable; doses by slot id. 4 pairing — receiver arms share one
stream; warmup identity asserted; dose-0 slots are the within-arm
control that validated pairing in v2. 5 phenomenon strength — floors
above; the tier gap (7,6)-vs-(4,3) spans tier-5/6/7 structures the
receiver world lacks entirely. 6 sensitivity — v2 measured dose-1.0
gap -0.059 with noise O(1e-3); a sign flip of comparable magnitude is
well inside resolution. 7 genesis/rates — genesis reported
mechanically; banked receipts are distribution-bound BY DESIGN (the
bound is the treatment).
"""

import json
import os
import time
from collections import defaultdict

import numpy as np

from replay_overnight import build_engine, BOOT_SEED
from sov import CERTAINTY_FLOOR, CERTAINTY_CEILING
from staged_fit_experiment import Accountant, run_worlds
from cargo_cult_probe import (DietAccountant, ExportingAccountant,
                              run_phase, spearman, calibration,
                              ATTEST_DISCOUNT, BELIEVED_SCORE)
from cargo_cult_v2 import (slot_dose, first_contact_error,
                           WARMUP_WORLDS, DIET_WORLDS, SHIFT_WORLDS,
                           FIRST_WINDOW, GAP_GATE, RHO_GATE,
                           FIREWALL_BAND, TRANSFER_FLOOR,
                           FULL_DOSE_SLOT_FLOOR, FIRST_FIT_SLOT_FLOOR)

GAP_WORLDS = [(97460 + i, (7, 6)[i % 2]) for i in range(10)]
BANK_CAP = 2000
BANK_SLOT_FLOOR = 50
BANK_RECEIPT_MIN = 10
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p103_cargo_cult_v3.json')


class BankedDietAccountant(DietAccountant):
    """Dosed attest arm importing from a pre-grown gap-world bank."""

    def __init__(self, name, lesion, bank, bank_geometry):
        DietAccountant.__init__(self, name, lesion)
        self.bank = bank
        self.bank_geometry = bank_geometry
        self._cursor = defaultdict(int)
        self._attest_step = 0

    def process_diet(self, rv, emb, obs, reward, off, ep, donor_exports,
                     donor_web):
        self._parity ^= 1
        if not self.diet_on or self._parity == 0:
            before = {sid: s.ledger.fit_count
                      for sid, s in self.web.slots.items()}
            self.process(rv, emb, obs, reward, off, ep)
            if self.diet_on:
                for sid, s in self.web.slots.items():
                    fc0 = before.get(sid)
                    if fc0 is not None and s.ledger.fit_count > fc0:
                        self.own_pos[sid] += s.ledger.fit_count - fc0
            return
        self.web._global_step += 1
        self._attest_step += 1
        for sid, receipts in self.bank.items():
            dose = slot_dose(sid)
            if dose == 0.0 or not receipts:
                continue
            if dose == 0.5 and self._attest_step % 2 == 0:
                continue
            slot = self.web.slots.get(sid)
            if slot is None or slot.state not in ('open', 'closed'):
                continue
            k = self._cursor[sid] % len(receipts)
            self._cursor[sid] += 1
            posed = {'family_thresholds': self.bank_geometry[sid]}
            moved = self.web.attest(sid, posed, [receipts[k]],
                                    ATTEST_DISCOUNT,
                                    exporter_id='GAP-DONOR',
                                    centroid_pull=True)
            if not moved:
                continue
            self.transfers[sid] += len(moved)
            if self.lesion:
                led = slot.ledger
                for _tr in moved:
                    led.fit_count += 1
                    alpha = max(0.05, 1.0 / max(led.fit_count, 1))
                    led.certainty = float(np.clip(
                        led.certainty * (1 - alpha)
                        + BELIEVED_SCORE * alpha,
                        CERTAINTY_FLOOR, CERTAINTY_CEILING))


def first_contact_tightness(arm, t_shift, window=FIRST_WINDOW):
    dists = []
    for s in arm.web.slots.values():
        if s.state not in ('open', 'closed'):
            continue
        post = [r for r in s.ledger.receipts
                if r.kind == 'fit' and r.created_at > t_shift
                and r.sign > 0][:window]
        dists.extend(1.0 - min(r.magnitude, 1.0) for r in post)
    return float(np.mean(dists)) if dists else None


def main():
    t0 = time.time()
    print('=== P103(cargo-cult) v3: world-gap testimony ===')
    from train import generate_training_data, train_model
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    print('donor pre-grow (10 worlds at tiers 7/6)...')
    gap_donor = ExportingAccountant('GAP-DONOR', staged=False,
                                    consume=False)
    run_worlds(GAP_WORLDS, [gap_donor], engine, model, 0)
    bank, bank_geo = {}, {}
    for sid, s in gap_donor.web.slots.items():
        if s.state not in ('open', 'closed'):
            continue
        pos = [r for r in s.ledger.receipts
               if r.kind == 'fit' and r.sign > 0
               and r.embedding is not None][-BANK_CAP:]
        if pos:
            bank[sid] = pos
            bank_geo[sid] = \
                s.geometry.family_thresholds.tolist()
    rich_slots = sum(1 for v in bank.values()
                     if len(v) >= BANK_RECEIPT_MIN)
    print('  bank: %d slots (%d with >= %d receipts)'
          % (len(bank), rich_slots, BANK_RECEIPT_MIN))

    donor = ExportingAccountant('RIDER', staged=False, consume=False)
    lived = Accountant('LIVED', staged=False, consume=False)
    intact = BankedDietAccountant('GAP-INTACT', False, bank, bank_geo)
    lesion = BankedDietAccountant('GAP-LESION', True, bank, bank_geo)
    arms = [lived, intact, lesion]

    print('warmup (4 worlds)...')
    c = run_phase(WARMUP_WORLDS, donor, arms, engine, model, 0,
                  diet=False)
    sig = [(len(a.web.edges), a.web.get_stats()['total_receipts'])
           for a in [donor] + arms]
    assert len(set(sig)) == 1, 'C20 check 4: warmup states diverged'
    print('  identity at diet switch: True')

    print('diet (6 worlds, gap-bank substitution)...')
    c = run_phase(DIET_WORLDS, donor, arms, engine, model, c, diet=True)
    n_tr = {a.name: sum(a.transfers.values()) for a in (intact, lesion)}
    pre_cal = {a.name: calibration(a) for a in arms}
    cert0 = {a.name: {sid: s.ledger.certainty
                      for sid, s in a.web.slots.items()
                      if s.state in ('open', 'closed')} for a in arms}
    t_shift = {a.name: a.web._global_step for a in arms}
    print('  transfers=%s pre_cal=%s'
          % (n_tr, {k: round(v, 4) for k, v in pre_cal.items()}))

    print('shift (4 worlds, fresh seeds + tier swap, full fits)...')
    run_phase(SHIFT_WORLDS, donor, arms, engine, model, c, diet=False)

    fce = {a.name: first_contact_error(a, cert0[a.name],
                                       t_shift[a.name]) for a in arms}
    fce_mean = {k: (float(np.mean(list(v.values()))) if v else None)
                for k, v in fce.items()}
    fct = {a.name: first_contact_tightness(a, t_shift[a.name])
           for a in arms}
    print('  first-contact cal error: %s'
          % {k: (None if v is None else round(v, 4))
             for k, v in fce_mean.items()})
    print('  first-contact tightness: %s'
          % {k: (None if v is None else round(v, 4))
             for k, v in fct.items()})

    les = fce['GAP-LESION']
    doses = [slot_dose(sid) for sid in sorted(les)]
    errs = [les[sid] for sid in sorted(les)]
    rho = spearman(doses, errs)

    full_dose_slots = sum(1 for sid, n in lesion.transfers.items()
                          if slot_dose(sid) == 1.0 and n >= 5)
    gap = (fce_mean['GAP-LESION'] - fce_mean['LIVED']
           if None not in (fce_mean['GAP-LESION'], fce_mean['LIVED'])
           else None)
    fw = (abs(fce_mean['GAP-INTACT'] - fce_mean['LIVED'])
          if None not in (fce_mean['GAP-INTACT'], fce_mean['LIVED'])
          else None)
    geo_gap = (fct['GAP-INTACT'] - fct['LIVED']
               if None not in (fct['GAP-INTACT'], fct['LIVED'])
               else None)

    floors_ok = (all(v >= TRANSFER_FLOOR for v in n_tr.values())
                 and full_dose_slots >= FULL_DOSE_SLOT_FLOOR
                 and len(les) >= FIRST_FIT_SLOT_FLOOR
                 and rich_slots >= BANK_SLOT_FLOOR)
    if not floors_ok or gap is None or fw is None:
        verdict = ('UNTESTED (transfers=%s full_dose=%d/%d '
                   'first_fit=%d/%d bank=%d/%d)'
                   % (n_tr, full_dose_slots, FULL_DOSE_SLOT_FLOOR,
                      len(les), FIRST_FIT_SLOT_FLOOR, rich_slots,
                      BANK_SLOT_FLOOR))
    elif gap >= GAP_GATE and rho is not None and rho >= RHO_GATE \
            and fw <= FIREWALL_BAND:
        verdict = ('SUPPORTED: world-gap testimony bills at first '
                   'contact (gap=%+.4f), dose-graded (rho=%.2f), '
                   'firewall holds (band=%.4f) — the cargo-cult '
                   'signature is a function of donor-receiver '
                   'divergence' % (gap, rho, fw))
    elif gap <= 0:
        verdict = ('NOT SUPPORTED: inversion survives world-gap '
                   'testimony (gap=%+.4f) — the pathology requires '
                   'adversarial falsity, not distribution distance'
                   % gap)
    else:
        verdict = ('PARTIAL: gap %+.4f positive but below gate %.2f, '
                   'or rho=%s / firewall band %.4f fails'
                   % (gap, GAP_GATE, rho, fw))
    print('\nP103(cargo-cult) v3 VERDICT: %s' % verdict)

    out = {'first_contact_error_mean': fce_mean,
           'first_contact_gap_lesion_lived': gap,
           'firewall_band_calibration': fw,
           'geometric_poisoning_gap_intact_lived': geo_gap,
           'first_contact_tightness': fct,
           'dose_rho_lesion': rho,
           'pre_calibration': pre_cal,
           'transfers': n_tr, 'bank_slots': len(bank),
           'bank_rich_slots': rich_slots,
           'per_slot_first_contact': {k: {str(s): round(v, 4)
                                          for s, v in d.items()}
                                      for k, d in fce.items()},
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

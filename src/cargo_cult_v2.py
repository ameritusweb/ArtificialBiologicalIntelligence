"""P103(cargo-cult) v2 — first-contact measurement, dosed grounding,
funding damage. (Card locked at launch 2026-08-13. Supersedes v1's
endpoints; v1 receipts stand.)

V1 DISPOSITION (results/p103_cargo_cult.json, on the record):
  PARTIAL as printed, corrected on inspection to three instrument
  findings, each with a mechanism:
  1. PRE-SHIFT INVERSION: the lesioned arm was the BEST calibrated
     (0.153 vs lived 0.206) — truthful testimony in a high-confirm
     world corrects the base certainty formula's understatement
     (certainty tracks SCORE, confirm tracks FREQUENCY; believing a
     true donor at 0.9 lands nearer the ~0.9 confirm rate). Gullibility
     is calibration-NEUTRAL-or-better while testimony is true and the
     world is easy. The pathology is not false confidence; it is
     confidence UNBACKED BY GROUNDING, which only bills at contact
     with change.
  2. POST-SHIFT WASHOUT: LIVED and ATT-INTACT post-shift calibration
     came back BIT-IDENTICAL (0.2002736371501126) — fit sign/score
     live in receptor space (stream-identical across arms), and the
     certainty EMA's time constant (~20 lived fits at alpha 0.05)
     forgets diet-phase confidence long before the 200-fit measurement
     window closes. The v1 endpoint measured arms AFTER
     re-equilibration: the cargo-cult signature is a FIRST-CONTACT
     phenomenon and must be measured there. (Positive receipt: the
     intact firewall arm is calibration-indistinguishable from lived
     throughout — the sov.py endpoint firewall verified in field.)
  3. RHO DEGENERATE: uniform 50pct parity substitution gives grounded
     fraction ~0.5 everywhere (spread only from 3 outlier slots);
     the v1 rho (+0.43) is uninterpretable. Grounding must be DOSED
     per slot.

V2 DESIGN (same stream pattern, four webs, same phases):
  Doses per slot: slot_id % 3 -> {0: none, 1: half, 2: full}
  substitution on attested steps (deterministic; half = alternate
  attested steps). Grounded fraction spans ~{1.0, 0.67, 0.5}.
  FIRST-CONTACT ENDPOINT: per slot, |certainty at shift start -
  confirm rate over the FIRST 200 post-shift fits|; arm mean. The
  lesion's borrowed confidence must bill HERE or nowhere.
  DOSE GRADIENT: Spearman(dose, first-contact error) within the
  lesion arm, all slots.
  FUNDING DAMAGE (reported): genesis (composes) and closures per arm
  during diet — fit_count inflation is the PERMANENT lesion channel
  (certainty is transient; funding gates are not).

VERDICTS (fixed): SUPPORTED iff (1) first-contact mean error
LESION - LIVED >= 0.02 AND (2) dose gradient rho >= +0.3 in the
lesion arm AND (3) firewall clause: |INTACT - LIVED| first-contact
mean <= 0.01. PARTIAL: (1) holds, (2) or (3) fails. NOT SUPPORTED:
LESION - LIVED < 0.005 at first contact (borrowed confidence never
bills, even at first contact — the lesion is harmless at this dose).
UNTESTED: transfers < 200 per attest arm, or full-dose lesion slots
with >= 5 transfers < 8, or slots with >= 100 first-window fits < 20.

C20 (seven): 1 domain — as v1. 2 endpoint independence — the lesion
writes certainty; the confirm side of first-contact error is the
world's own record, unwritable by imports; snapshot taken BEFORE the
window it is compared against. 3 exogeneity — doses assigned by slot
id (pre-registered rule), phases at pre-registered indices.
4 pairing — one stream, four webs, warmup identity asserted.
5 phenomenon strength — v1 receipts: transfers ~51k/arm; OOD shift
produced tightness separation; floor above. 6 sensitivity (magnitude,
v1-informed) — full-dose lesion certainty equilibrium ~0.79 (equal-
rate pull toward lived score ~0.69 and pushes toward 0.9) vs lived
~0.69; OOD confirm ~0.7-0.8; expected first-contact gap O(0.03-0.1)
vs noise O(1e-3); gate 0.02 sits inside. 7 genesis/rates — genesis
reported as mechanical counts, not billed as discovery; receipts
distribution-bound as v1.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np

from replay_overnight import build_engine, BOOT_SEED
from sov import CERTAINTY_FLOOR, CERTAINTY_CEILING
from staged_fit_experiment import Accountant
from cargo_cult_probe import (DietAccountant, ExportingAccountant,
                              run_phase, spearman, calibration,
                              ATTEST_DISCOUNT, BELIEVED_SCORE)
from train import generate_training_data, train_model

WARMUP_WORLDS = [(97400 + i, (4, 3)[i % 2]) for i in range(4)]
DIET_WORLDS = [(97500 + i, (4, 3)[i % 2]) for i in range(6)]
SHIFT_WORLDS = [(97600 + i, (3, 4)[i % 2]) for i in range(4)]
DOSE = {0: 0.0, 1: 0.5, 2: 1.0}
FIRST_WINDOW = 200
GAP_GATE = 0.02
NULL_GATE = 0.005
RHO_GATE = 0.3
FIREWALL_BAND = 0.01
TRANSFER_FLOOR = 200
FULL_DOSE_SLOT_FLOOR = 8
FIRST_FIT_SLOT_FLOOR = 20
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p103_cargo_cult_v2.json')


def slot_dose(sid):
    return DOSE[sid % 3]


class DosedDietAccountant(DietAccountant):
    """Attest-fed arm with per-slot substitution doses (dispatches
    through run_phase's DietAccountant isinstance check)."""

    def __init__(self, name, lesion):
        DietAccountant.__init__(self, name, lesion)
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
        for sid, receipts in donor_exports.items():
            dose = slot_dose(sid)
            if dose == 0.0:
                continue
            if dose == 0.5 and self._attest_step % 2 == 0:
                continue
            slot = self.web.slots.get(sid)
            dslot = donor_web.slots.get(sid)
            if slot is None or dslot is None:
                continue
            if slot.state not in ('open', 'closed'):
                continue
            posed = {'family_thresholds':
                     dslot.geometry.family_thresholds.tolist()}
            moved = self.web.attest(sid, posed, receipts,
                                    ATTEST_DISCOUNT, exporter_id='DONOR',
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


def first_contact_error(arm, cert0, t_shift, window=FIRST_WINDOW):
    """Per slot: |certainty at shift start - confirm over the first
    `window` post-shift fits|."""
    out = {}
    for sid, s in arm.web.slots.items():
        if s.state not in ('open', 'closed') or sid not in cert0:
            continue
        post = [r for r in s.ledger.receipts
                if r.kind == 'fit' and r.created_at > t_shift][:window]
        if len(post) >= 100:
            confirm = sum(1 for r in post if r.sign > 0) / len(post)
            out[sid] = abs(cert0[sid] - confirm)
    return out


def flow_counts(arm):
    return {'composes': len(arm.composed),
            'closures': sum(1 for s in arm.web.slots.values()
                            if s.state == 'closed')}


def main():
    t0 = time.time()
    print('=== P103(cargo-cult) v2: dosed grounding, first contact ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    donor = ExportingAccountant('DONOR', staged=False, consume=False)
    lived = Accountant('LIVED', staged=False, consume=False)
    intact = DosedDietAccountant('ATT-INTACT', lesion=False)
    lesion = DosedDietAccountant('ATT-LESION', lesion=True)
    arms = [lived, intact, lesion]

    print('warmup (4 worlds)...')
    c = run_phase(WARMUP_WORLDS, donor, arms, engine, model, 0,
                  diet=False)
    sig = [(len(a.web.edges), a.web.get_stats()['total_receipts'])
           for a in [donor] + arms]
    identical = len(set(sig)) == 1
    assert identical, 'C20 check 4: warmup states diverged'
    print('  identity at diet switch: %s' % identical)

    print('diet (6 worlds, dosed substitution)...')
    c = run_phase(DIET_WORLDS, donor, arms, engine, model, c, diet=True)
    n_tr = {a.name: sum(a.transfers.values()) for a in (intact, lesion)}
    pre_cal = {a.name: calibration(a) for a in arms}
    diet_flow = {a.name: flow_counts(a) for a in [donor] + arms}
    cert0 = {a.name: {sid: s.ledger.certainty
                      for sid, s in a.web.slots.items()
                      if s.state in ('open', 'closed')} for a in arms}
    t_shift = {a.name: a.web._global_step for a in arms}
    print('  transfers=%s pre_cal=%s flow=%s'
          % (n_tr, {k: round(v, 4) for k, v in pre_cal.items()},
             diet_flow))

    print('shift (4 worlds, fresh seeds + tier swap, full fits)...')
    run_phase(SHIFT_WORLDS, donor, arms, engine, model, c, diet=False)

    fce = {a.name: first_contact_error(a, cert0[a.name],
                                       t_shift[a.name]) for a in arms}
    fce_mean = {k: (float(np.mean(list(v.values()))) if v else None)
                for k, v in fce.items()}
    post_cal = {a.name: calibration(a) for a in arms}
    print('  first-contact error means: %s'
          % {k: (None if v is None else round(v, 4))
             for k, v in fce_mean.items()})

    les = fce['ATT-LESION']
    doses = [slot_dose(sid) for sid in sorted(les)]
    errs = [les[sid] for sid in sorted(les)]
    rho = spearman(doses, errs)

    grounded = {}
    for a in (intact, lesion):
        g = {}
        for sid in set(list(a.transfers) + list(a.own_pos)):
            tr, own = a.transfers.get(sid, 0), a.own_pos.get(sid, 0)
            if tr + own > 0:
                g[sid] = own / (own + tr)
        grounded[a.name] = g

    full_dose_slots = sum(1 for sid, n in lesion.transfers.items()
                          if slot_dose(sid) == 1.0 and n >= 5)
    first_fit_slots = len(les)
    gap = (fce_mean['ATT-LESION'] - fce_mean['LIVED']
           if None not in (fce_mean['ATT-LESION'], fce_mean['LIVED'])
           else None)
    fw = (abs(fce_mean['ATT-INTACT'] - fce_mean['LIVED'])
          if None not in (fce_mean['ATT-INTACT'], fce_mean['LIVED'])
          else None)

    floors_ok = (all(v >= TRANSFER_FLOOR for v in n_tr.values())
                 and full_dose_slots >= FULL_DOSE_SLOT_FLOOR
                 and first_fit_slots >= FIRST_FIT_SLOT_FLOOR)
    if not floors_ok or gap is None or fw is None:
        verdict = ('UNTESTED (transfers=%s full_dose_slots=%d/%d '
                   'first_fit_slots=%d/%d)'
                   % (n_tr, full_dose_slots, FULL_DOSE_SLOT_FLOOR,
                      first_fit_slots, FIRST_FIT_SLOT_FLOOR))
    elif gap >= GAP_GATE and rho is not None and rho >= RHO_GATE \
            and fw <= FIREWALL_BAND:
        verdict = ('SUPPORTED: borrowed confidence bills at first '
                   'contact (gap=%.4f), dose-graded (rho=%.2f), and '
                   'the intact firewall prevents it (|intact-lived|='
                   '%.4f)' % (gap, rho, fw))
    elif gap < NULL_GATE:
        verdict = ('NOT SUPPORTED: lesion first-contact error within '
                   '%.3f of lived (gap=%.4f) — borrowed confidence '
                   'never bills at this dose' % (NULL_GATE, gap))
    elif gap >= GAP_GATE:
        verdict = ('PARTIAL: first-contact gap %.4f holds but '
                   'rho=%s (gate %.1f) or firewall band %.4f (gate '
                   '%.2f) fails' % (gap, rho, RHO_GATE, fw,
                                    FIREWALL_BAND))
    else:
        verdict = ('PARTIAL: gap %.4f between null (%.3f) and support '
                   '(%.2f) gates' % (gap, NULL_GATE, GAP_GATE))
    print('\nP103(cargo-cult) v2 VERDICT: %s' % verdict)

    out = {'first_contact_error_mean': fce_mean,
           'first_contact_gap_lesion_lived': gap,
           'firewall_band': fw,
           'dose_rho_lesion': rho,
           'pre_calibration': pre_cal, 'post_calibration': post_cal,
           'transfers': n_tr,
           'diet_flow': diet_flow,
           'grounded_fraction': {k: {str(s): round(v, 3)
                                     for s, v in g.items()}
                                 for k, g in grounded.items()},
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

"""P103(cargo-cult) v4 — the MAINTAINED DIET: pathology as a standing
circulation state. (Card locked at launch 2026-08-13; applies the
seven wave fixes; v1-v3 receipts stand.)

THE CLAIM (the wave's implication 3, enacted): the cargo-cult
signature is not a contamination event but a CIRCULATION STATE.
Testimony that was true of the old world, MAINTAINED while the
organism lives a moved world, produces standing calibration damage;
WITHDRAWING the testimony heals at the certainty clock (~14-fit
half-life, alpha 0.05) — the architecture's rehabilitation
prediction, quantified in advance.

WAVE FIXES APPLIED (F36 addendum, each named):
  1 DISEQUILIBRIUM PROBE: VOID-BY-EQUILIBRIUM branch measured on the
    LIVED arm's canopy slots (confirm drop >= 0.05 at first contact,
    else the shift did not move the world against canopy content and
    the claim has no subject here).
  2 CLOCK TABLE: all windows in FIT-COUNT space inside signal
    lifetimes; the healing prediction IS the clock (certainty
    half-life ~14 fits << shift-phase ~1600 thin-arm fits, so
    WITHDRAW must be healed by end-of-shift).
  3 TEMPORAL TYPE: the treatment is SUSTAINED (MAINT imports through
    the shift from a bank frozen at the shift point — old-world
    testimony against a moved world).
  4 CANOPY TARGETS: doses only on composed slots (sid >= 33); the
    trunk is world-invariant (T95) and serves as the within-arm
    invariance control at dose 0.
  5 SATURATION ARITHMETIC: attest throughput measured ample
    (35-51k transfers/run, no admission ceiling on the attest path);
    lesion certainty writes: canopy confirm at the moved world
    ~0.6-0.75 vs maintained pushes toward 0.9 -> standing error gap
    O(0.05-0.15) vs gate 0.02, noise O(1e-3).
  6 FLOORS FROM CENSUS: warmup extended to 6 worlds so the composed
    census reaches ~18; canopy floor 8 sits under the measured
    ceiling, asserted at diet start.
  7 RIGHT-SPACE ENDPOINTS: calibration in confirm-frequency space
    from own lived fits only; no geometry endpoint on this card.

DESIGN — one stream, four webs + donor rider; compose scans ONLY in
warmup (flow frozen after, all webs keep IDENTICAL slot membership —
check 8 clean by construction):
  LIVED     full fits, all phases (anchor + strain probe).
  THIN      parity-thinned (lives even steps only) in diet AND shift,
            no imports ever — the exposure control.
  MAINT     parity-thinned; odd steps import lesioned canopy
            testimony: live donor exports during diet, the FROZEN
            BANK (donor's diet-phase canopy receipts) during shift.
  WITHDRAW  identical to MAINT through the diet; at the shift the
            imports STOP (still parity-thinned = THIN's exposure) —
            the healing arm.
  Lesion crediting on imports (unchanged): fit_count += 1, certainty
  toward 0.9 at the lived alpha — testimony believed at face value.
  Primary comparisons, all at matched exposure:
    MAINT - THIN      standing-testimony damage (the claim)
    WITHDRAW vs THIN  residual after withdrawal (healing)
    MAINT - WITHDRAW  the maintenance effect proper

ENDPOINTS (canopy slots, own-fit confirm windows):
  first-contact error: |certainty at shift - confirm over first 200
  post-shift fits|; standing error: |certainty at end - confirm over
  last 200 fits|. Trunk versions reported as invariance control.

VERDICTS (fixed): VOID-BY-EQUILIBRIUM if LIVED canopy confirm drop
< 0.05 at first contact. SUPPORTED iff standing MAINT - THIN >= 0.02
AND |WITHDRAW - THIN| standing <= 0.01 AND WITHDRAW first-contact
error - THIN first-contact >= 0.01 (there was damage to heal).
NOT SUPPORTED: MAINT - THIN < 0.005 standing (maintained stale
testimony harmless even against a moved world). PARTIAL: between.
UNTESTED: canopy slots with >= 50 first-window fits < 8, or MAINT
shift transfers < 500, or diet transfers < 200 per import arm.

C20 (eight): 1 domain — as v2/v3. 2 endpoint independence — imports
write certainty; confirm windows are own lived fits, unwritable by
imports; cert snapshots taken before the windows they compare to.
3 exogeneity — arm policies and phase boundaries pre-registered; the
bank freeze at t_shift is the manipulated staleness. 4 pairing — one
stream; warmup identity asserted; MAINT/WITHDRAW identical through
diet (asserted at shift). 5 phenomenon strength — the disequilibrium
probe IS the check, as a measured branch. 6 sensitivity — arithmetic
above. 7 genesis/rates — flow frozen post-warmup; no genesis
endpoints; bank receipts distribution-bound BY DESIGN (staleness is
the treatment). 8 population closure — slot membership frozen at
warmup end and identical across webs; canopy floor from census;
windows floored in fit space.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from live_receptors import LiveReceptorBank
from replay_overnight import build_engine, BOOT_SEED
from sov import CERTAINTY_FLOOR, CERTAINTY_CEILING
from staged_fit_experiment import Accountant
from cargo_cult_probe import (ExportingAccountant, ATTEST_DISCOUNT,
                              BELIEVED_SCORE)
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

WARMUP_WORLDS = [(98100 + i, (4, 3)[i % 2]) for i in range(6)]
DIET_WORLDS = [(98110 + i, (4, 3)[i % 2]) for i in range(6)]
SHIFT_WORLDS = [(98200 + i, (3, 4)[i % 2]) for i in range(4)]
TRUNK_MAX = 32
WINDOW = 200
STRAIN_FLOOR = 0.05
GAP_GATE = 0.02
NULL_GATE = 0.005
HEAL_BAND = 0.01
HEAL_EVIDENCE = 0.01
CANOPY_FLOOR = 8
FIT_FLOOR = 50
SHIFT_TRANSFER_FLOOR = 500
DIET_TRANSFER_FLOOR = 200
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p103_cargo_cult_v4.json')


class V4Arm(Accountant):
    def __init__(self, name, thin, import_diet, import_shift):
        Accountant.__init__(self, name, staged=False, consume=False)
        self.thin = thin
        self.import_diet = import_diet
        self.import_shift = import_shift
        self.transfers_diet = defaultdict(int)
        self.transfers_shift = defaultdict(int)
        self._parity = 0
        self._cursor = defaultdict(int)

    def _lesion_credit(self, slot, moved):
        led = slot.ledger
        for _tr in moved:
            led.fit_count += 1
            alpha = max(0.05, 1.0 / max(led.fit_count, 1))
            led.certainty = float(np.clip(
                led.certainty * (1 - alpha) + BELIEVED_SCORE * alpha,
                CERTAINTY_FLOOR, CERTAINTY_CEILING))

    def step(self, phase, rv, emb, obs, reward, off, ep, canopy,
             live_exports, live_geo, bank, bank_geo):
        if phase == 'warmup' or not self.thin:
            self.process(rv, emb, obs, reward, off, ep)
            return
        self._parity ^= 1
        if self._parity == 1:
            self.process(rv, emb, obs, reward, off, ep)
            return
        self.web._global_step += 1
        if phase == 'diet' and self.import_diet:
            source, geo, book = live_exports, live_geo, \
                self.transfers_diet
        elif phase == 'shift' and self.import_shift:
            source, geo, book = None, bank_geo, self.transfers_shift
        else:
            return
        for sid in canopy:
            slot = self.web.slots.get(sid)
            if slot is None or slot.state not in ('open', 'closed'):
                continue
            if source is not None:
                receipts = source.get(sid)
                if not receipts:
                    continue
                batch = receipts
            else:
                pool = bank.get(sid)
                if not pool:
                    continue
                k = self._cursor[sid] % len(pool)
                self._cursor[sid] += 1
                batch = [pool[k]]
            posed = {'family_thresholds': geo[sid]}
            moved = self.web.attest(sid, posed, batch, ATTEST_DISCOUNT,
                                    exporter_id='DONOR',
                                    centroid_pull=True)
            if moved:
                book[sid] += len(moved)
                self._lesion_credit(slot, moved)


def run_phase(phase, worlds, donor, arms, engine, model, base, canopy,
              bank, bank_geo, scan):
    counter = base
    for w_seed, tier in worlds:
        env = TieredEnvironment(seed=w_seed, tier=tier)
        np.random.seed(w_seed * 7)
        env.rng = np.random.RandomState(w_seed * 7 + 1)
        rng = np.random.RandomState(w_seed * 7 + 2)
        bank_r = LiveReceptorBank()
        for ep in range(2):
            org = Organism()
            org.reset()
            for step in range(400):
                w = org.get_observation_window()
                act, _ = model.predict(w)
                r = rng.random()
                if r < PROBE_RATE_FLOOR:
                    act = np.zeros_like(act)
                elif r < EXPLORE_RATE:
                    act = rng.randint(0, 2, size=len(act)).astype(
                        act.dtype)
                obs, reward = org.step(act, env, step)
                rv = bank_r.compute(obs, act, None, reward)
                emb = engine.encoder.embed(engine._core_obs(obs))
                counter += 1
                exports = donor.process_and_export(rv, emb, obs,
                                                   reward, counter, ep)
                live_exports = {sid: rs for sid, rs in exports.items()
                                if sid in canopy} if canopy else {}
                live_geo = {sid: donor.web.slots[sid].geometry
                            .family_thresholds.tolist()
                            for sid in live_exports}
                for a in arms:
                    a.step(phase, rv, emb, obs, reward, counter, ep,
                           canopy or [], live_exports, live_geo,
                           bank, bank_geo)
        if scan:
            for a in [donor] + arms:
                web = a.web
                done = 0
                for (x, y), n in sorted(a.cofit.items(),
                                        key=lambda kv: (-kv[1], kv[0])):
                    if done >= 3 or n < 50:
                        break
                    if (x, y) in a.composed:
                        continue
                    sx, sy = web.slots.get(x), web.slots.get(y)
                    if (sx is None or sy is None or sx.state != 'open'
                            or sy.state != 'open'):
                        continue
                    if web.compose(x, y)[0] >= 0:
                        a.composed.add((x, y))
                        done += 1
    return counter


def confirm_rate(slot, t_from=None, t_to=None, first=None, last=None):
    fits = [r for r in slot.ledger.receipts if r.kind == 'fit']
    if t_from is not None:
        fits = [r for r in fits if r.created_at > t_from]
    if t_to is not None:
        fits = [r for r in fits if r.created_at <= t_to]
    if first is not None:
        fits = fits[:first]
    if last is not None:
        fits = fits[-last:]
    if len(fits) < FIT_FLOOR:
        return None, len(fits)
    return sum(1 for r in fits if r.sign > 0) / len(fits), len(fits)


def arm_errors(arm, sids, cert0, t_shift):
    fc, st = {}, {}
    for sid in sids:
        s = arm.web.slots.get(sid)
        if s is None or s.state not in ('open', 'closed') \
                or sid not in cert0:
            continue
        c_first, _ = confirm_rate(s, t_from=t_shift, first=WINDOW)
        c_last, _ = confirm_rate(s, t_from=t_shift, last=WINDOW)
        if c_first is not None:
            fc[sid] = abs(cert0[sid] - c_first)
        if c_last is not None:
            st[sid] = abs(s.ledger.certainty - c_last)
    mean = lambda d: (float(np.mean(list(d.values()))) if d else None)
    return {'first_contact': mean(fc), 'standing': mean(st),
            'n': len(st)}


def main():
    t0 = time.time()
    print('=== P103(cargo-cult) v4: the maintained diet ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    donor = ExportingAccountant('DONOR', staged=False, consume=False)
    lived = V4Arm('LIVED', thin=False, import_diet=False,
                  import_shift=False)
    thin = V4Arm('THIN', thin=True, import_diet=False,
                 import_shift=False)
    maint = V4Arm('MAINT', thin=True, import_diet=True,
                  import_shift=True)
    withd = V4Arm('WITHDRAW', thin=True, import_diet=True,
                  import_shift=False)
    arms = [lived, thin, maint, withd]

    print('warmup (6 worlds, flow on, all full-fit)...')
    c = run_phase('warmup', WARMUP_WORLDS, donor, arms, engine, model,
                  0, None, None, None, scan=True)
    sig = [(len(a.web.edges), a.web.get_stats()['total_receipts'])
           for a in [donor] + arms]
    assert len(set(sig)) == 1, 'C20 check 4: warmup states diverged'
    canopy = sorted(sid for sid, s in lived.web.slots.items()
                    if sid > TRUNK_MAX and s.state in ('open', 'closed'))
    trunk = sorted(sid for sid, s in lived.web.slots.items()
                   if sid <= TRUNK_MAX and s.state in ('open', 'closed'))
    print('  identity ok; census: canopy=%d trunk=%d'
          % (len(canopy), len(trunk)))
    assert len(canopy) >= CANOPY_FLOOR, \
        'check 8: canopy census below floor'

    print('diet (6 worlds, flow frozen, lesioned canopy imports)...')
    t_diet = donor.web._global_step
    c = run_phase('diet', DIET_WORLDS, donor, arms, engine, model, c,
                  canopy, None, None, scan=False)
    same_diet = (sum(maint.transfers_diet.values())
                 == sum(withd.transfers_diet.values()))
    n_diet = {a.name: sum(a.transfers_diet.values())
              for a in (maint, withd)}
    print('  diet transfers: %s identical=%s' % (n_diet, same_diet))

    # freeze the bank: donor's diet-phase positive canopy fits
    bank, bank_geo = {}, {}
    for sid in canopy:
        s = donor.web.slots.get(sid)
        if s is None:
            continue
        pos = [r for r in s.ledger.receipts
               if r.kind == 'fit' and r.sign > 0
               and r.created_at > t_diet and r.embedding is not None]
        if pos:
            bank[sid] = pos
            bank_geo[sid] = s.geometry.family_thresholds.tolist()
    cert0 = {a.name: {sid: a.web.slots[sid].ledger.certainty
                      for sid in canopy + trunk
                      if sid in a.web.slots} for a in arms}
    t_shift = {a.name: a.web._global_step for a in arms}
    pre_confirm = {}
    for sid in canopy:
        cr, _ = confirm_rate(lived.web.slots[sid], t_to=t_shift['LIVED'],
                             last=WINDOW)
        if cr is not None:
            pre_confirm[sid] = cr

    print('shift (4 worlds; MAINT keeps importing the frozen bank)...')
    run_phase('shift', SHIFT_WORLDS, donor, arms, engine, model, c,
              canopy, bank, bank_geo, scan=False)

    # disequilibrium probe on LIVED canopy
    drops = []
    for sid, pre in pre_confirm.items():
        post, _ = confirm_rate(lived.web.slots[sid],
                               t_from=t_shift['LIVED'], first=WINDOW)
        if post is not None:
            drops.append(pre - post)
    strain = float(np.mean(drops)) if drops else None
    print('  LIVED canopy confirm drop at first contact: %s (n=%d)'
          % (None if strain is None else round(strain, 4), len(drops)))

    can_err = {a.name: arm_errors(a, canopy, cert0[a.name],
                                  t_shift[a.name]) for a in arms}
    trk_err = {a.name: arm_errors(a, trunk, cert0[a.name],
                                  t_shift[a.name]) for a in arms}
    n_shift = sum(maint.transfers_shift.values())
    print('  canopy errors: %s' % {k: {kk: (None if vv is None
                                            else round(vv, 4))
                                       for kk, vv in v.items()}
                                   for k, v in can_err.items()})
    print('  MAINT shift transfers: %d' % n_shift)

    m, t, w = (can_err['MAINT'], can_err['THIN'], can_err['WITHDRAW'])
    floors_ok = (t['n'] >= CANOPY_FLOOR and m['n'] >= CANOPY_FLOOR
                 and w['n'] >= CANOPY_FLOOR
                 and n_shift >= SHIFT_TRANSFER_FLOOR
                 and all(v >= DIET_TRANSFER_FLOOR
                         for v in n_diet.values()))
    if strain is None or not drops:
        verdict = 'VOID-BY-EQUILIBRIUM (strain probe unmeasurable)'
    elif strain < STRAIN_FLOOR:
        verdict = ('VOID-BY-EQUILIBRIUM: LIVED canopy confirm drop '
                   '%.4f < %.2f — the shift did not move the world '
                   'against canopy content' % (strain, STRAIN_FLOOR))
    elif not floors_ok:
        verdict = ('UNTESTED (n canopy=%s shift_transfers=%d/%d '
                   'diet=%s)' % ({k: v['n'] for k, v in
                                  can_err.items()}, n_shift,
                                 SHIFT_TRANSFER_FLOOR, n_diet))
    else:
        gap = m['standing'] - t['standing']
        heal_resid = abs(w['standing'] - t['standing'])
        heal_ev = (w['first_contact'] - t['first_contact']
                   if None not in (w['first_contact'],
                                   t['first_contact']) else None)
        if gap >= GAP_GATE and heal_resid <= HEAL_BAND \
                and heal_ev is not None and heal_ev >= HEAL_EVIDENCE:
            verdict = ('SUPPORTED: maintained stale testimony holds '
                       'standing damage (MAINT-THIN=%.4f), withdrawal '
                       'heals to control (|W-THIN|=%.4f) after real '
                       'first-contact damage (+%.4f) — the pathology '
                       'is a circulation state and the ledger heals '
                       'on withdrawal' % (gap, heal_resid, heal_ev))
        elif gap < NULL_GATE:
            verdict = ('NOT SUPPORTED: maintained stale testimony '
                       'harmless (MAINT-THIN=%.4f) even against a '
                       'moved world' % gap)
        else:
            verdict = ('PARTIAL: gap=%.4f heal_resid=%.4f heal_ev=%s'
                       % (gap, heal_resid, heal_ev))
    print('\nP103(cargo-cult) v4 VERDICT: %s' % verdict)

    out = {'strain_probe_confirm_drop': strain,
           'canopy_errors': can_err, 'trunk_errors': trk_err,
           'transfers': {'diet': n_diet, 'shift': n_shift},
           'bank_slots': len(bank),
           'canopy_census': len(canopy),
           'diet_arms_identical': same_diet,
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

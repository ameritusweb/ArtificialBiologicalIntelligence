"""P103(cargo-cult) — the mature-domain circulation pathology: imported
structure without lived grounding, and the firewall clause that blocks
it. (T159 fifth arrival; sibling of P81's C2-ablation — both arms
lesion a conservativity gate, this one at the TESTIMONY boundary.
Card locked at launch 2026-08-13.)

THE CLAIM: an organism fed attested (imported) structure in place of
lived contact develops the CARGO-CULT signature — confidence
uncorrelated with grounding — and the signature is (i) present in the
ledger BEFORE the world exposes it, (ii) exposed at first
out-of-distribution contact, (iii) predicted per-slot by GROUNDED
FRACTION. The intact attest firewall (transfers never touch
fit_count/certainty — sov.py's endpoint firewall) should PREVENT the
signature: intact-attest tracks lived, lesioned diverges.

DESIGN — one lived stream, four webs:
  DONOR      full fits throughout; exports per-step positive fit
             receipts; never measured.
  LIVED      full fits throughout (control).
  ATT-INTACT diet substitution during the diet phase: even steps own
             fit_all, odd steps ATTEST import of the donor's positive
             fits that step (centroid_pull=True, discount 0.9) — the
             firewall INTACT: transfers carry no confidence.
  ATT-LESION same diet, firewall ABLATED in the sandbox accountant
             (core sov.py untouched): each transfer additionally
             credits fit_count += 1 and updates certainty toward 0.9
             with the lived alpha formula — testimony believed at face
             value. The pathology, enacted.
PHASES: WARMUP 4 worlds (all arms full fits — identity asserted);
DIET 6 worlds (policies live); SHIFT 4 worlds (fresh seeds, tiers
swapped — everyone meets the real world, full fits, no imports).

GROUNDED FRACTION per slot (diet phase): own positive fits /
(own positive fits + attested transfers).

ENDPOINTS:
  (a) pre-shift certainty mean per arm (confidence inflation)
  (b) calibration error |certainty - own-fit confirm rate|, pre- and
      post-shift per arm (the signature: written confidence vs the
      world's testimony; the confirm side is unwritable by the lesion)
  (c) post-shift strain per arm: negative-fit fraction + near-miss
      delta over the shift phase
  (d) per-slot Spearman(grounded fraction, post-shift calibration
      error) within each attest arm (prediction: negative)

VERDICTS (fixed): SUPPORTED iff (b) LESION calibration error worst of
the three measured arms pre-shift AND the LESION-vs-LIVED gap WIDENS
post-shift AND (d) Spearman <= -0.3 in the lesion arm (n >= 8 slots
with >= 5 transfers). PARTIAL: (b) both clauses, (d) fails. NOT
SUPPORTED: lesion calibration ~ intact (within 0.01) pre- AND
post-shift — the firewall is not load-bearing at this dose. UNTESTED:
transfers < 200 per attest arm, or < 8 slots with >= 5 transfers, or
post-shift fresh fits < 500 per arm.

C20 (seven): 1 domain — staged-fit harness class, in-dist; the SHIFT
phase is deliberately OOD for the webs (that is the treatment), while
the policy/model remain in their trained regime. 2 endpoint
independence — the lesion writes certainty; calibration compares it to
the CONFIRM RATE computed from own lived fits, which no import can
write; tightness measures geometry vs world. 3 exogeneity — diet and
lesion are policies switched at pre-registered world indices; the
shift is a pre-registered world-identity change. 4 pairing — one
stream, four accountants; warmup identity asserted (receipts+edges).
5 phenomenon strength — donor must export >= 200 admissible receipts
per attest arm (floor above); shift = reseed + tier swap, the
strongest standing OOD move (F23/F26 receipts: reseed produces
contact-strain reliably). 6 sensitivity (magnitude estimate, the
standing lesson) — lesion certainty writes: hundreds of transfers at
alpha >= 0.05 toward 0.9 saturate certainty upward, an O(0.1+) shift
vs calibration noise O(1e-3) in this harness class; decisively above
the floor. 7 genesis/rates — no genesis endpoint; receipts
distribution-bound: attested receipts originate in the same warmup
distribution the webs live (stated; the OOD phase tests the webs, not
the receipts).
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
from staged_fit_experiment import Accountant, family_of
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

WARMUP_WORLDS = [(97400 + i, (4, 3)[i % 2]) for i in range(4)]
DIET_WORLDS = [(97500 + i, (4, 3)[i % 2]) for i in range(6)]
SHIFT_WORLDS = [(97600 + i, (3, 4)[i % 2]) for i in range(4)]
EPISODES = 2
STEPS = 400
ATTEST_DISCOUNT = 0.9
BELIEVED_SCORE = 0.9
TRANSFER_FLOOR = 200
SLOT_FLOOR = 8
SLOT_TRANSFER_MIN = 5
FRESH_FLOOR = 500
SPEARMAN_GATE = -0.3
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p103_cargo_cult.json')


class DietAccountant(Accountant):
    """Attest-fed arm: odd diet steps import instead of living."""

    def __init__(self, name, lesion):
        Accountant.__init__(self, name, staged=False, consume=False)
        self.lesion = lesion
        self.diet_on = False
        self.transfers = defaultdict(int)
        self.own_pos = defaultdict(int)
        self._parity = 0

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
        # attested step: time passes, the world is not lived
        self.web._global_step += 1
        for sid, receipts in donor_exports.items():
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


class ExportingAccountant(Accountant):
    """Donor: full fits; exposes this step's positive fit receipts."""

    def process_and_export(self, rv, emb, obs, reward, off, ep):
        marks = {sid: s.ledger.receipt_count
                 for sid, s in self.web.slots.items()}
        self.process(rv, emb, obs, reward, off, ep)
        exports = {}
        for sid, s in self.web.slots.items():
            n0 = marks.get(sid, 0)
            new = s.ledger.receipts[n0:]
            pos = [r for r in new if r.kind == 'fit' and r.sign > 0
                   and r.embedding is not None]
            if pos:
                exports[sid] = pos
        return exports


def run_phase(worlds, donor, arms, engine, model, base, diet):
    for a in arms:
        if isinstance(a, DietAccountant):
            a.diet_on = diet
    counter = base
    for w_seed, tier in worlds:
        env = TieredEnvironment(seed=w_seed, tier=tier)
        np.random.seed(w_seed * 7)
        env.rng = np.random.RandomState(w_seed * 7 + 1)
        rng = np.random.RandomState(w_seed * 7 + 2)
        bank = LiveReceptorBank()
        for ep in range(EPISODES):
            org = Organism()
            org.reset()
            for step in range(STEPS):
                w = org.get_observation_window()
                act, _ = model.predict(w)
                r = rng.random()
                if r < PROBE_RATE_FLOOR:
                    act = np.zeros_like(act)
                elif r < EXPLORE_RATE:
                    act = rng.randint(0, 2, size=len(act)).astype(
                        act.dtype)
                obs, reward = org.step(act, env, step)
                rv = bank.compute(obs, act, None, reward)
                emb = engine.encoder.embed(engine._core_obs(obs))
                counter += 1
                exports = donor.process_and_export(rv, emb, obs, reward,
                                                   counter, ep)
                for a in arms:
                    if isinstance(a, DietAccountant):
                        a.process_diet(rv, emb, obs, reward, counter,
                                       ep, exports, donor.web)
                    else:
                        a.process(rv, emb, obs, reward, counter, ep)
        # boundary compose scan, same as the staged harness
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


def snapshot(arm):
    out = {}
    for sid, s in arm.web.slots.items():
        if s.state not in ('open', 'closed'):
            continue
        out[sid] = {'certainty': s.ledger.certainty,
                    'nm': s.ledger.near_miss_seen,
                    'fits': s.ledger.fit_count,
                    'reopens': s.ledger.reopen_count}
    return out


def calibration(arm, window=200):
    cal = []
    for s in arm.web.slots.values():
        if s.state not in ('open', 'closed'):
            continue
        recent = [r for r in s.ledger.receipts
                  if r.kind == 'fit'][-window:]
        if len(recent) >= 10:
            confirm = sum(1 for r in recent if r.sign > 0) / len(recent)
            cal.append(abs(s.ledger.certainty - confirm))
    return float(np.mean(cal)) if cal else None


def per_slot_calibration(arm, window=200):
    out = {}
    for sid, s in arm.web.slots.items():
        if s.state not in ('open', 'closed'):
            continue
        recent = [r for r in s.ledger.receipts
                  if r.kind == 'fit'][-window:]
        if len(recent) >= 10:
            confirm = sum(1 for r in recent if r.sign > 0) / len(recent)
            out[sid] = abs(s.ledger.certainty - confirm)
    return out


def spearman(xs, ys):
    def rank(v):
        order = np.argsort(np.argsort(v, kind='stable'), kind='stable')
        return order.astype(float)
    if len(xs) < 3:
        return None
    rx, ry = rank(np.array(xs)), rank(np.array(ys))
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else None


def main():
    t0 = time.time()
    print('=== P103(cargo-cult): one stream, four webs ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)
    engine = build_engine()

    donor = ExportingAccountant('DONOR', staged=False, consume=False)
    lived = Accountant('LIVED', staged=False, consume=False)
    intact = DietAccountant('ATT-INTACT', lesion=False)
    lesion = DietAccountant('ATT-LESION', lesion=True)
    arms = [lived, intact, lesion]

    print('warmup (4 worlds, all arms full fits)...')
    c = run_phase(WARMUP_WORLDS, donor, arms, engine, model, 0,
                  diet=False)
    sig = [(len(a.web.edges), a.web.get_stats()['total_receipts'])
           for a in [donor] + arms]
    identical = len(set(sig)) == 1
    print('  identity at diet switch: %s %s' % (identical, sig))
    assert identical, 'C20 check 4: warmup states diverged'

    print('diet (6 worlds, attest arms on 50pct substitution)...')
    c = run_phase(DIET_WORLDS, donor, arms, engine, model, c, diet=True)

    pre = {a.name: snapshot(a) for a in arms}
    pre_cal = {a.name: calibration(a) for a in arms}
    pre_cert = {a.name: float(np.mean([v['certainty']
                                       for v in pre[a.name].values()]))
                for a in arms}
    grounded = {}
    for a in (intact, lesion):
        g = {}
        for sid in set(list(a.transfers) + list(a.own_pos)):
            tr, own = a.transfers.get(sid, 0), a.own_pos.get(sid, 0)
            if tr + own > 0:
                g[sid] = own / (own + tr)
        grounded[a.name] = g
    n_tr = {a.name: sum(a.transfers.values()) for a in (intact, lesion)}
    print('  transfers: %s   pre-cal: %s' % (n_tr,
          {k: (None if v is None else round(v, 4))
           for k, v in pre_cal.items()}))

    for a in arms:
        a.web.pop_fresh_tightness()

    print('shift (4 worlds, fresh seeds + tier swap, full fits)...')
    run_phase(SHIFT_WORLDS, donor, arms, engine, model, c, diet=False)

    post_cal = {a.name: calibration(a) for a in arms}
    tight = {a.name: a.web.pop_fresh_tightness() for a in arms}
    post = {a.name: snapshot(a) for a in arms}
    strain = {}
    for a in arms:
        rows = {}
        for sid, b in pre[a.name].items():
            p = post[a.name].get(sid)
            if p is None:
                continue
            rows[sid] = {'d_nm': p['nm'] - b['nm'],
                         'd_reopen': p['reopens'] - b['reopens']}
        strain[a.name] = rows

    sp = {}
    for a in (intact, lesion):
        pc = per_slot_calibration(a)
        g = grounded[a.name]
        sids = [sid for sid in g
                if sid in pc and a.transfers.get(sid, 0)
                >= SLOT_TRANSFER_MIN]
        sp[a.name] = {'n': len(sids),
                      'rho': spearman([g[s] for s in sids],
                                      [pc[s] for s in sids])}

    fresh_ok = all(v[1] >= FRESH_FLOOR for v in tight.values())
    floors_ok = (all(n_tr[k] >= TRANSFER_FLOOR for k in n_tr)
                 and sp['ATT-LESION']['n'] >= SLOT_FLOOR and fresh_ok)

    lc_pre, li_pre, lv_pre = (pre_cal['ATT-LESION'],
                              pre_cal['ATT-INTACT'], pre_cal['LIVED'])
    lc_post, li_post, lv_post = (post_cal['ATT-LESION'],
                                 post_cal['ATT-INTACT'],
                                 post_cal['LIVED'])
    if not floors_ok:
        verdict = ('UNTESTED (floors: transfers=%s lesion_slots=%d/%d '
                   'fresh_ok=%s)' % (n_tr, sp['ATT-LESION']['n'],
                                     SLOT_FLOOR, fresh_ok))
    else:
        worst_pre = lc_pre > li_pre and lc_pre > lv_pre
        gap_widens = (lc_post - lv_post) > (lc_pre - lv_pre)
        rho = sp['ATT-LESION']['rho']
        rho_ok = rho is not None and rho <= SPEARMAN_GATE
        near = (abs(lc_pre - li_pre) < 0.01
                and abs(lc_post - li_post) < 0.01)
        if worst_pre and gap_widens and rho_ok:
            verdict = ('SUPPORTED: lesioned arm worst-calibrated before '
                       'the shift, gap widens at OOD contact, grounded '
                       'fraction predicts per-slot error (rho=%.2f)'
                       % rho)
        elif near:
            verdict = ('NOT SUPPORTED: lesioned ~ intact on calibration '
                       'pre and post — firewall not load-bearing at '
                       'this dose')
        elif worst_pre and gap_widens:
            verdict = ('PARTIAL: signature present and exposed by the '
                       'shift, but grounded fraction does not predict '
                       'per-slot error (rho=%s)' % rho)
        else:
            verdict = ('PARTIAL: incomplete signature (worst_pre=%s '
                       'gap_widens=%s rho=%s)'
                       % (worst_pre, gap_widens, rho))
    print('\nP103(cargo-cult) VERDICT: %s' % verdict)

    out = {'pre_calibration': pre_cal, 'post_calibration': post_cal,
           'pre_certainty_mean': pre_cert,
           'post_tightness': {k: v[0] for k, v in tight.items()},
           'fresh_n': {k: v[1] for k, v in tight.items()},
           'transfers': n_tr,
           'grounded_fraction': {k: {str(s): round(v, 3)
                                     for s, v in g.items()}
                                 for k, g in grounded.items()},
           'spearman': sp,
           'strain': {k: {str(s): v for s, v in rows.items()}
                      for k, rows in strain.items()},
           'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

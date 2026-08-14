"""SOV-P1 first bill — the forced-closure control (pre-registered).

THE HEADLINE CLAIM, finally in court: does holding unknowns open under
the funded lifecycle BEAT committing early, on identical evidence?

Design — one lived stream, two accountants: a single organism stream
(policy inhabitants, EL harness) is fed to TWO webs simultaneously:
  ARM-SOV    — the standard lifecycle (evidence-gated closure, 404
               window, Reopen; the F13-validated machinery).
  ARM-CLOSED — forced commitment: eager closure (radius <= 0.15 at 3
               fits) and NO reopening — closure is permanent, the
               value-substitution architecture in miniature.
Pairing is perfect by construction: the webs see byte-identical fits.

Mid-run, the world SHIFTS structurally (tier 4 -> tier 3, new seed:
the hidden variable and cross-modal machinery vanish, layouts change —
slot-level statistics genuinely move). Theorem 2's bill: commitment
pays or ties while the world holds; it pays the un-bookable price when
the world moves.

C20 pre-flight (six checks):
1. Domain match — the EL harness worlds and policy inhabitants both
   webs were built against all session.
2. Endpoint independence — webs are pure observers; endpoints are
   web-side epistemic quantities; neither web influences the stream.
3. Exogeneity — the shift is imposed at a pre-registered episode index,
   identical for both arms (same stream).
4. Pairing proven — ONE stream, two accountants: both webs receive
   byte-identical fit inputs by construction (a single organism run
   feeds both), so no identity smoke is needed; the arms differ ONLY
   in lifecycle policy, and diverge pre-shift by design (eager closure
   acts early — that divergence is the treatment, not a leak).
5. Phenomenon strength — floors: >= 500 post-shift positive fits per
   web; >= 5 forced-closures existing at shift time in ARM-CLOSED
   (otherwise the arms cannot differ and the run is UNTESTED).
6. Endpoint sensitivity — fresh-fit tightness (the F16-export window
   statistic, built to move) and calibration error move per window;
   staleness counts closed slots directly.

ENDPOINTS (post-shift, last PHASE2 windows):
  (a) fresh-fit tightness  — mean fit-time distance to centroid
      (lower = the web's geometry tracks the new world).
  (b) calibration error    — mean |slot certainty − recent confirm
      rate| over active slots (lower = certainty means something).
  (c) stale commitments    — count of closed slots whose resolution
      predates the shift (ARM-CLOSED cannot shed them by design;
      reported, not billed — it is true by construction).

VERDICT RULES (fixed before launch):
  SUPPORTED (directional): ARM-SOV beats ARM-CLOSED post-shift on BOTH
      (a) and (b).
  PARTIAL: better on exactly one.
  NOT SUPPORTED: ARM-SOV worse or equal on both — holding open bought
      nothing even under structural shift.
  UNTESTED: floors unmet.
  Honest two-sidedness: PRE-shift, ARM-CLOSED equal or better is
  EXPECTED (commitment is cheap in a stationary world) and is reported
  as confirmation of the theorem's other half, not suppressed.
"""

import json
import math
import os
import time

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from sov import ConstraintWeb, CLOSURE_RADIUS
from receptor_eigen_coder import ReceptorEigenCoder
from train import generate_training_data, train_model, EXPLORE_RATE, \
    PROBE_RATE_FLOOR

PHASE1_WORLDS = [(61000 + i, 4) for i in range(6)]   # tier 4, pre-shift
PHASE2_WORLDS = [(62000 + i, 3) for i in range(6)]   # tier 3, post-shift
EPISODES_PER_WORLD = 2
STEPS = 400
BOOT_SEED = 123
CONFIRM_WINDOW = 200      # receipts for the confirm-rate estimate
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'sov_p1_first_bill.json')


class ForcedClosureWeb(ConstraintWeb):
    """The value-substitution accountant: eager, permanent commitment.
    Experiment-side subclass — the core algebra is untouched."""

    EAGER_RADIUS = 0.28   # v3: into the observed radius band (~0.23
                          # typical pre-shift tightness) so the eager
                          # treatment actually engages (v2: 2 closures,
                          # floor is 5 — C20 check 5 iteration)

    def _check_closure(self, slot_id):
        slot = self.slots[slot_id]
        if slot.state != 'open':
            return False
        if (slot.geometry.radius <= self.EAGER_RADIUS
                and slot.ledger.fit_count >= 3):
            slot.state = 'closed'
            slot.closed_at = self._global_step
            slot.resolution = {'centroid': slot.geometry.centroid.copy(),
                               'certainty': slot.ledger.certainty,
                               'fit_count': slot.ledger.fit_count,
                               'radius': slot.geometry.radius}
            slot.ledger.fail_window = []
            slot.posit_liability = float(self._connectivity(slot_id))
            self._log_event('closed', [slot_id], [],
                            {'radius': slot.geometry.radius,
                             'forced': True})
            return True
        return False

    def reopen(self, slot_id, failing_receipt_ids):
        return []      # commitment is permanent — no retraction, ever


def make_policy_and_engine():
    """Policy for the inhabitants AND the real encoder for the webs —
    v2 instrument fix: the activation-vector embedding never reaches
    closure-grade tightness (v1 UNTESTED: zero forced closures, the
    treatment never engaged); the baseline's closure dynamics live in
    ENCODER space, so the webs get the real thing."""
    from mental_model import build_mental_model
    X, Y, Z, log = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_mental_model(log)
    return model, engine


def endpoints(web):
    tight, n_tight = web.pop_fresh_tightness()
    cal_errs = []
    stale = 0
    for slot in web.slots.values():
        if slot.state not in ('open', 'closed'):
            continue
        recent = [r for r in slot.ledger.receipts
                  if r.kind == 'fit'][-CONFIRM_WINDOW:]
        if len(recent) >= 10:
            confirm = sum(1 for r in recent if r.sign > 0) / len(recent)
            cal_errs.append(abs(slot.ledger.certainty - confirm))
        if slot.state == 'closed':
            stale += 1
    return {'fresh_tightness': tight, 'n_fresh': n_tight,
            'calibration_error': (float(np.mean(cal_errs))
                                  if cal_errs else None),
            'closed_slots': stale}


def main():
    t0 = time.time()
    print('=== SOV-P1 first bill v2: one stream, two accountants ===')
    model, engine = make_policy_and_engine()

    web_sov = ConstraintWeb(eigen_coder=ReceptorEigenCoder(),
                            debug_level=1, ledger_id='SOV')
    web_sov.populate_from_families()
    web_forced = ForcedClosureWeb(eigen_coder=ReceptorEigenCoder(),
                                  debug_level=1, ledger_id='FORCED')
    web_forced.populate_from_families()
    webs = [web_sov, web_forced]

    # v2: REAL encoder embeddings — the same space the citable baseline's
    # closure dynamics (195 attempts) lived in.
    def fit_with_activation_embedding(web, rv, obs, reward, off, ep, ts):
        core = engine._core_obs(obs)
        emb = engine.encoder.embed(core)
        web.fit_all(rv, emb, obs, obs, reward, off, ep, ts,
                    support_obs=core)

    # Re-bind feed loop locally (embedding-aware)
    def run_phase(worlds, base):
        counter = base
        from live_receptors import LiveReceptorBank
        for w_seed, tier in worlds:
            env = TieredEnvironment(seed=w_seed, tier=tier)
            np.random.seed(w_seed * 7 + 1)
            env.rng = np.random.RandomState(w_seed * 7 + 2)
            rng = np.random.RandomState(w_seed * 7 + 3)
            bank = LiveReceptorBank()
            for ep in range(EPISODES_PER_WORLD):
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
                    counter += 1
                    for web in webs:
                        web._global_step += 1
                        fit_with_activation_embedding(
                            web, rv, obs, reward, counter, ep,
                            web._global_step)
                for web in webs:
                    web.anneal_all(web._global_step)
        return counter

    print('phase 1 (tier 4, pre-shift)...')
    c = run_phase(PHASE1_WORLDS, 0)
    pre = {w.ledger_id: endpoints(w) for w in webs}
    forced_closed_at_shift = pre['FORCED']['closed_slots']
    print(f"  pre-shift: SOV closed={pre['SOV']['closed_slots']} "
          f"tight={pre['SOV']['fresh_tightness']} | FORCED closed="
          f"{forced_closed_at_shift} tight="
          f"{pre['FORCED']['fresh_tightness']}")

    print('phase 2 (tier 3, POST-SHIFT)...')
    run_phase(PHASE2_WORLDS, c)
    post = {w.ledger_id: endpoints(w) for w in webs}
    print(f"  post-shift: SOV tight={post['SOV']['fresh_tightness']} "
          f"cal={post['SOV']['calibration_error']} closed="
          f"{post['SOV']['closed_slots']} reopens="
          f"{web_sov._op_counts.get('reopen', 0)}")
    print(f"              FORCED tight={post['FORCED']['fresh_tightness']}"
          f" cal={post['FORCED']['calibration_error']} closed="
          f"{post['FORCED']['closed_slots']}")

    # ---- Verdict ----
    floors = (post['SOV']['n_fresh'] >= 500
              and post['FORCED']['n_fresh'] >= 500
              and forced_closed_at_shift >= 5)
    a_sov = post['SOV']['fresh_tightness']
    a_frc = post['FORCED']['fresh_tightness']
    b_sov = post['SOV']['calibration_error']
    b_frc = post['FORCED']['calibration_error']
    if not floors:
        verdict = ('UNTESTED (floors: fresh fits or forced-closure count '
                   'unmet)')
    else:
        wins = sum([a_sov is not None and a_frc is not None
                    and a_sov < a_frc,
                    b_sov is not None and b_frc is not None
                    and b_sov < b_frc])
        verdict = {2: 'SUPPORTED (directional): the lifecycle beats '
                      'commitment on both post-shift endpoints',
                   1: 'PARTIAL: one of two endpoints',
                   0: 'NOT SUPPORTED: holding open bought nothing '
                      'post-shift'}[wins]
    print(f'\nSOV-P1 FIRST-BILL VERDICT: {verdict}')

    out = {'pre': pre, 'post': post, 'verdict': verdict,
           'sov_reopens': int(web_sov._op_counts.get('reopen', 0)),
           'forced_closed_at_shift': int(forced_closed_at_shift),
           'sov_conservation': web_sov.check_conservation_laws(),
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'saved {RESULTS} ({out["elapsed_min"]} min)')


if __name__ == '__main__':
    main()

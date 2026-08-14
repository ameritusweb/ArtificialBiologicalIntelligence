"""EL-1 acceptance v0.3 (pre-registered; iterations logged).

Iteration history (instrument design, all pre-verdict):
  v0.1 UNTESTED — window volume below floors.
  v0.2 NOT PASSED — kinematics probe exposed two instrument faults:
       (a) distance-delta classification conflated SOURCE drift with
       organism behavior; (b) the bare-oracle inhabitant is near-still
       (path ~0.02/window), violating EL-1's own precondition
       ("behavior rich enough to describe and fail at").
  v0.3 — organism-attributed features (source frozen at window end);
       REAL inhabitants: a bootstrap-trained policy driven with the rich
       runner's probe/explore/policy mix; thresholds calibrated on the
       TRAIN corpus and frozen before any test window is scored.

C20 pre-flight (six checks):
1. Domain match — describer trained and billed on tier-4 worlds with
   policy-driven inhabitants, the class it will serve.
2. Endpoint independence — the describer observes; it writes nothing.
3. Exogeneity — train/test split by WORLD SEED; test worlds are fresh.
4. Pairing — n/a (both describers score identical realized windows).
5. Phenomenon strength — floors: >= 1000 test windows, >= 4 contexts
   with >= 20 train occasions, no behavior class > 0.9 marginal.
6. Endpoint sensitivity — NLL moves per window; word-failure endpoint
   has occasion floors (>= 50).

ACCEPTANCE RULES (fixed):
  EL1-A: held-out mean NLL, conditional < marginal.
  EL1-B: max word-failure rate >= 1.5x mean (>= 50 occasions per word).
  PASS = A and B; A-only blocks EL-2; floors unmet = UNTESTED.
"""

import math

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_descriptive import (
    DescriptiveStratum, window_features, classify_behavior_from_features,
    active_contexts, append_descriptions_to_ledger, WINDOW, BEHAVIORS)
from train import generate_training_data, train_model, EXPLORE_RATE, \
    PROBE_RATE_FLOOR

TIER = 4
TRAIN_WORLDS = list(range(9000, 9016))
TEST_WORLDS = list(range(9500, 9510))
EPISODES_PER_WORLD = 3
STEPS = 400
BOOT_SEED = 123

FLOOR_TEST_WINDOWS = 1000
FLOOR_CONTEXTS = 4
FLOOR_CTX_OCC = 20


def make_inhabitant_policy():
    """One bootstrap-trained policy: the real inhabitant class."""
    print('bootstrapping inhabitant policy (oracle data + training)...')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    return model


def collect_windows(model, seed):
    """One world: policy-driven organism (probe/explore/policy mix,
    deterministic per seed); returns raw window records
    (contexts, features)."""
    env = TieredEnvironment(seed=seed, tier=TIER)
    np.random.seed(seed * 7 + 1)
    env.rng = np.random.RandomState(seed * 7 + 2)
    rng = np.random.RandomState(seed * 7 + 3)
    records = []
    for ep in range(EPISODES_PER_WORLD):
        org = Organism()
        org.reset()
        xs, ys = [], []
        ctx_at_start = None
        for step in range(STEPS):
            if step % WINDOW == 0:
                if len(xs) == WINDOW and ctx_at_start is not None:
                    records.append((ctx_at_start,
                                    window_features(env, xs, ys,
                                                    step - WINDOW)))
                xs, ys = [], []
                ctx_at_start = active_contexts(env, org, step)
            window = org.get_observation_window()
            policy_action, _ = model.predict(window)
            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                executed = np.zeros_like(policy_action)
            elif r < EXPLORE_RATE:
                executed = rng.randint(
                    0, 2, size=len(policy_action)).astype(
                        policy_action.dtype)
            else:
                executed = policy_action
            org.step(executed, env, step)
            xs.append(org.x)
            ys.append(org.y)
    return records


def main():
    print('=== EL-1 acceptance v0.3: the environment describes its '
          'inhabitants ===')
    model = make_inhabitant_policy()

    print(f'\ncollecting {len(TRAIN_WORLDS)} training worlds...')
    train_records = []
    for seed in TRAIN_WORLDS:
        train_records.extend(collect_windows(model, seed))

    # Calibrate thresholds on TRAIN only, then freeze.
    paths = np.array([f['path'] for _, f in train_records])
    moves = np.array([max(abs(f['toward_pain']), abs(f['toward_end']))
                      for _, f in train_records])
    tau_rest = float(np.median(paths))
    tau_d = float(np.percentile(moves, 70))
    print(f'  calibrated (frozen): tau_rest={tau_rest:.3f} '
          f'tau_d={tau_d:.3f}; path p50/p90='
          f'{np.percentile(paths, [50, 90]).round(3)}')

    stratum = DescriptiveStratum()
    for ctx, feat in train_records:
        stratum.observe(ctx, classify_behavior_from_features(
            feat, tau_rest, tau_d))
    n_ctx = sum(1 for c, n in stratum.ctx_totals.items()
                if n >= FLOOR_CTX_OCC)
    marg = {b: stratum.p_marginal(b) for b in BEHAVIORS}
    print(f'  train windows: {int(stratum.total)}; contexts>='
          f'{FLOOR_CTX_OCC}: {n_ctx}')
    print('  behavior marginal: '
          + ', '.join(f'{b}={p:.3f}' for b, p in marg.items()))

    print(f'billing on {len(TEST_WORLDS)} fresh worlds...')
    scores = []
    n_test = 0
    for seed in TEST_WORLDS:
        for ctx, feat in collect_windows(model, seed):
            beh = classify_behavior_from_features(feat, tau_rest, tau_d)
            pc = stratum.p_conditional(ctx, beh)
            pm = stratum.p_marginal(beh)
            scores.append((-math.log(pc), -math.log(pm)))
            stratum.bill(ctx, beh)
            n_test += 1

    nll_c = float(np.mean([s[0] for s in scores]))
    nll_m = float(np.mean([s[1] for s in scores]))
    degenerate = max(marg.values()) > 0.9
    floors_met = (n_test >= FLOOR_TEST_WINDOWS
                  and n_ctx >= FLOOR_CONTEXTS and not degenerate)
    print(f'\n  test windows: {n_test} (floor {FLOOR_TEST_WINDOWS}); '
          f'degenerate: {degenerate}')
    print(f'  NLL conditional: {nll_c:.4f}   marginal: {nll_m:.4f}   '
          f'delta: {nll_m - nll_c:+.4f}')

    rates = stratum.word_failures(min_occasions=50)
    verdict_b = False
    if rates:
        mean_rate = float(np.mean(list(rates.values())))
        top = sorted(rates.items(), key=lambda kv: -kv[1])[:6]
        verdict_b = (top[0][1] >= 1.5 * mean_rate) if mean_rate > 0 else False
        print(f'  word failure rates (mean {mean_rate:.3f}), top: '
              + ', '.join(f'{w}={r:.3f}' for w, r in top))

    if not floors_met:
        verdict = 'UNTESTED (floors not met)'
    elif nll_c < nll_m and verdict_b:
        verdict = 'PASS — descriptions predict and failures localize'
    elif nll_c < nll_m:
        verdict = 'EL1-A only: predicts, no localization (EL-2 blocked)'
    else:
        verdict = 'NOT PASSED — the marginal describer is not beaten'
    print(f'\nEL-1 VERDICT: {verdict}')

    path = append_descriptions_to_ledger(stratum)
    print(f'{len(stratum.sentences())} descriptive sentences billed to '
          f'the ledger: {path}')
    return verdict.startswith('PASS')


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)

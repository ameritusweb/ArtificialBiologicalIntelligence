"""
Step 16 Verification: Sensorimotor Contingency (Detected Emergent)

Tests whether the organism's action patterns reliably produce specific
changes in its own proprioceptive state. This should emerge naturally
from proprioception (Step 14) + efference copy (Step 15) + mental model
without any explicit implementation.
"""

import numpy as np
from collections import defaultdict
from environment import Environment, Organism
from mental_model import action_to_hash

PROPRIO_SLICE = slice(102, 110)
CHANNEL_NAMES = ['speed', 'omega', 'ld0', 'ld1', 'ld2', 'ld3', 'ld4', 'ld5']


def run_diagnostic():
    print("=== Step 16 Verification: Sensorimotor Contingency ===\n")

    print("--- Collecting transitions from 100 episodes ---")
    rng = np.random.RandomState(42)
    all_actions = []
    all_hashes = []
    all_prop_deltas = []

    for ep in range(100):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        prev_proprio = None

        for step in range(300):
            actions = org.compute_optimal_actions(env, step)
            obs, _ = org.step(actions, env, step)
            curr_proprio = obs[102:110].copy()

            if prev_proprio is not None:
                delta = curr_proprio - prev_proprio
                all_actions.append(actions.copy())
                all_hashes.append(action_to_hash(actions))
                all_prop_deltas.append(delta)

            prev_proprio = curr_proprio

    all_actions = np.array(all_actions)
    all_deltas = np.array(all_prop_deltas)
    all_hashes = np.array(all_hashes)
    print(f"  Collected {len(all_deltas):,} transitions")

    # --- Diagnostic 1: Within-group consistency ---
    print("\n--- Diagnostic 1: Within-group consistency (same action -> similar prop change) ---")
    groups = defaultdict(list)
    for i, h in enumerate(all_hashes):
        groups[h].append(i)

    valid_groups = {h: idx for h, idx in groups.items() if len(idx) >= 20}
    print(f"  Action groups with >=20 samples: {len(valid_groups)}")

    within_stds = []
    for h, indices in valid_groups.items():
        group_deltas = all_deltas[indices]
        std_per_channel = group_deltas.std(axis=0)
        within_stds.append(std_per_channel.mean())

    within_stds = np.array(within_stds)
    print(f"  Median within-group std: {np.median(within_stds):.4f}")
    consistent = (within_stds < 0.15).sum()
    print(f"  Groups with std < 0.15: {consistent}/{len(within_stds)} "
          f"({consistent/len(within_stds)*100:.1f}%)")

    # --- Diagnostic 2: Between-group distinctiveness ---
    print("\n--- Diagnostic 2: Between-group distinctiveness (different actions -> different prop changes) ---")
    group_means = []
    for h, indices in valid_groups.items():
        group_means.append(all_deltas[indices].mean(axis=0))
    group_means = np.array(group_means)

    f_ratios = []
    for ch in range(8):
        between_var = group_means[:, ch].var()
        within_vars = []
        for h, indices in valid_groups.items():
            within_vars.append(all_deltas[indices, ch].var())
        within_var = np.mean(within_vars) + 1e-8
        f_ratios.append(between_var / within_var)

    print(f"  F-ratios per channel:")
    high_f = 0
    for i, (name, f) in enumerate(zip(CHANNEL_NAMES, f_ratios)):
        marker = " ***" if f > 2.0 else ""
        print(f"    {name:6s}: {f:.2f}{marker}")
        if f > 2.0:
            high_f += 1
    print(f"  Channels with F > 2.0: {high_f}/8")

    # --- Diagnostic 3: Linear predictability ---
    print("\n--- Diagnostic 3: Linear predictability (actions -> prop deltas) ---")
    X = all_actions.astype(np.float64)
    X_bias = np.hstack([X, np.ones((len(X), 1))])
    Y = all_deltas

    r_squared = []
    for ch in range(8):
        y = Y[:, ch]
        coeffs, residuals, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        ss_res = np.sum((y - X_bias @ coeffs) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-8
        r2 = 1.0 - ss_res / ss_tot
        r_squared.append(r2)

    print(f"  R² per channel:")
    for name, r2 in zip(CHANNEL_NAMES, r_squared):
        marker = " ***" if r2 > 0.3 else ""
        print(f"    {name:6s}: {r2:.3f}{marker}")
    mean_r2 = np.mean(r_squared)
    print(f"  Mean R²: {mean_r2:.3f}")

    # --- Verdict ---
    print("\n=== VERDICT ===")
    checks = [
        ("Within-group consistency (median std < 0.15)",
         np.median(within_stds) < 0.15),
        ("Between-group distinctiveness (F > 2.0 for >=4/8 channels)",
         high_f >= 4),
        ("Linear predictability (mean R² > 0.3)",
         mean_r2 > 0.3),
        ("Speed contingency (R²(speed) > 0.2)",
         r_squared[0] > 0.2),
        ("Rotation contingency (R²(omega) > 0.2)",
         r_squared[1] > 0.2),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  SENSORIMOTOR CONTINGENCY PRESENT BY CONSTRUCTION.")
        print("  The organism's actions reliably produce specific proprioceptive")
        print("  changes. Step 16 is a detected emergent, not a deployment step.")
    elif sum(1 for _, p in checks if p) >= 3:
        print("\n  PARTIAL SENSORIMOTOR CONTINGENCY.")
        print("  Most checks pass. The relationship exists but may need strengthening.")
    else:
        print("\n  SENSORIMOTOR CONTINGENCY NOT DETECTED.")
        print("  Step 16 needs explicit implementation.")

    return all_pass


if __name__ == '__main__':
    run_diagnostic()

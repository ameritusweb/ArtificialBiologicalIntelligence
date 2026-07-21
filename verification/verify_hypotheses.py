"""
Step 13 Verification: Does the mental model maintain multiple competing hypotheses?

Tests whether the CausalMappingStore holds genuinely competing predictions
in ambiguous contexts, or collapses to a single dominant mapping.

A healthy multi-hypothesis system should show:
1. Multiple entries per action_hash with distinct context embeddings
2. When queried in an ambiguous context, top-K results should have
   meaningfully different predicted deltas (not all pointing the same way)
3. The weighted blend should differ from the top-1 prediction
   (if blend == top-1, the system has effectively collapsed)
"""

import numpy as np
from collections import defaultdict
from environment import Environment, Organism
from mental_model import (
    build_mental_model, action_to_hash, CausalMappingStore,
    MentalModelEngine, CORE_OBS_DIM
)


def run_diagnostic():
    print("=== Step 13 Verification: Multiple Hypotheses ===\n")

    # Build experience and mental model
    print("--- Building mental model from 200 episodes ---")
    rng = np.random.RandomState(42)
    global_log = []
    for ep in range(200):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        for step in range(300):
            actions = org.compute_optimal_actions(env, step)
            org.step(actions, env, step)
        global_log.extend(org.experience_log)
    print(f"  Global log: {len(global_log):,} entries")

    engine = build_mental_model(global_log)
    store = engine.store

    # --- Diagnostic 1: Entry diversity per action ---
    print("\n--- Diagnostic 1: Entry diversity per action_hash ---")
    entry_counts = []
    for ah, entries in store.mappings.items():
        entry_counts.append(len(entries))

    entry_counts = np.array(entry_counts)
    print(f"  Action hashes with mappings: {len(entry_counts)}")
    print(f"  Entries per action: min={entry_counts.min()}, "
          f"median={np.median(entry_counts):.0f}, "
          f"mean={entry_counts.mean():.1f}, "
          f"max={entry_counts.max()}")
    multi_entry = (entry_counts > 1).sum()
    print(f"  Actions with >1 entry: {multi_entry}/{len(entry_counts)} "
          f"({multi_entry/len(entry_counts)*100:.1f}%)")

    if multi_entry == 0:
        print("\n  RESULT: COLLAPSED — every action has a single mapping.")
        print("  Multiple hypotheses NOT present. Step 13 needs explicit implementation.")
        return False

    # --- Diagnostic 2: Context diversity within action groups ---
    print("\n--- Diagnostic 2: Context diversity within action groups ---")
    intra_group_sims = []
    diverse_groups = 0
    for ah, entries in store.mappings.items():
        if len(entries) < 2:
            continue
        embs = np.array([e.context_embedding for e in entries])
        sims = embs @ embs.T
        mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
        pairwise = sims[mask]
        avg_sim = float(pairwise.mean()) if len(pairwise) > 0 else 1.0
        intra_group_sims.append(avg_sim)
        if avg_sim < 0.85:
            diverse_groups += 1

    if intra_group_sims:
        sims_arr = np.array(intra_group_sims)
        print(f"  Avg intra-group cosine similarity: {sims_arr.mean():.3f}")
        print(f"  Groups with diverse contexts (avg sim < 0.85): "
              f"{diverse_groups}/{len(sims_arr)} ({diverse_groups/len(sims_arr)*100:.1f}%)")
    else:
        print("  No multi-entry groups to analyze.")

    # --- Diagnostic 3: Prediction divergence in ambiguous contexts ---
    print("\n--- Diagnostic 3: Prediction divergence under top-K retrieval ---")
    divergence_scores = []
    blend_vs_top1_diffs = []

    test_rng = np.random.RandomState(99)
    test_env = Environment(seed=999)
    test_org = Organism()
    test_org.reset(test_rng)

    for step in range(200):
        actions = test_org.compute_optimal_actions(test_env, step)
        obs_before = test_org.history[-1].copy() if len(test_org.history) > 0 else np.zeros(test_org.OBS_DIM)
        test_org.step(actions, test_env, step)

        ah = action_to_hash(actions)
        obs_core = obs_before[:CORE_OBS_DIM]
        embedding = engine.encoder.embed(obs_core)
        results = store.query(ah, embedding, top_k=5)

        if len(results) >= 2:
            deltas = np.array([e.delta for e, _ in results])
            scores = np.array([s for _, s in results])

            pairwise_dists = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    dist = np.linalg.norm(deltas[i] - deltas[j])
                    pairwise_dists.append(dist)
            avg_divergence = np.mean(pairwise_dists) if pairwise_dists else 0.0
            divergence_scores.append(avg_divergence)

            weights = scores / (scores.sum() + 1e-8)
            blend = sum(w * d for w, d in zip(weights, deltas))
            top1 = deltas[0]
            blend_diff = np.linalg.norm(blend - top1)
            blend_vs_top1_diffs.append(blend_diff)

    if divergence_scores:
        div_arr = np.array(divergence_scores)
        diff_arr = np.array(blend_vs_top1_diffs)
        print(f"  Steps with >=2 retrieved mappings: {len(divergence_scores)}/200")
        print(f"  Avg pairwise delta divergence: {div_arr.mean():.4f} "
              f"(std={div_arr.std():.4f})")
        print(f"  Avg blend-vs-top1 difference: {diff_arr.mean():.4f} "
              f"(std={diff_arr.std():.4f})")
        meaningful_divergence = (div_arr > 0.01).sum()
        meaningful_blend = (diff_arr > 0.001).sum()
        print(f"  Steps with meaningful divergence (>0.01): "
              f"{meaningful_divergence}/{len(div_arr)} ({meaningful_divergence/len(div_arr)*100:.1f}%)")
        print(f"  Steps where blend != top1 (>0.001): "
              f"{meaningful_blend}/{len(diff_arr)} ({meaningful_blend/len(diff_arr)*100:.1f}%)")
    else:
        print("  No steps with multiple retrieved mappings.")

    # --- Diagnostic 4: Certainty distribution ---
    print("\n--- Diagnostic 4: Certainty distribution ---")
    all_certs = [e.certainty for entries in store.mappings.values() for e in entries]
    certs = np.array(all_certs)
    print(f"  Total mappings: {len(certs)}")
    print(f"  Certainty: min={certs.min():.3f}, median={np.median(certs):.3f}, "
          f"mean={certs.mean():.3f}, max={certs.max():.3f}")
    low_cert = (certs < 0.3).sum()
    mid_cert = ((certs >= 0.3) & (certs < 0.7)).sum()
    high_cert = (certs >= 0.7).sum()
    print(f"  Low (<0.3): {low_cert} ({low_cert/len(certs)*100:.1f}%)")
    print(f"  Mid (0.3-0.7): {mid_cert} ({mid_cert/len(certs)*100:.1f}%)")
    print(f"  High (>0.7): {high_cert} ({high_cert/len(certs)*100:.1f}%)")

    # --- Verdict ---
    print("\n=== VERDICT ===")
    has_diversity = multi_entry > len(entry_counts) * 0.3
    has_divergence = len(divergence_scores) > 0 and np.mean(divergence_scores) > 0.01
    has_blend_diff = len(blend_vs_top1_diffs) > 0 and np.mean(blend_vs_top1_diffs) > 0.001
    has_cert_spread = (certs.std() > 0.05) if len(certs) > 0 else False

    checks = [
        ("Multi-entry action groups (>30%)", has_diversity),
        ("Meaningful delta divergence in top-K", has_divergence),
        ("Blend differs from top-1", has_blend_diff),
        ("Certainty spread (std > 0.05)", has_cert_spread),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  MULTIPLE HYPOTHESES PRESENT BY CONSTRUCTION.")
        print("  The architecture maintains competing predictions without")
        print("  explicit multi-hypothesis machinery. Step 13 is a detected")
        print("  property, not a deployment step.")
    else:
        print("\n  PARTIAL COLLAPSE DETECTED.")
        print("  Some hypothesis maintenance is present but insufficient.")
        print("  Step 13 may need explicit anti-collapse mechanism.")

    return all_pass


if __name__ == '__main__':
    run_diagnostic()

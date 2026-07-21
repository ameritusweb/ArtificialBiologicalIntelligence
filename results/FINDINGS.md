# Experimental Findings

All results are reproducible by running the corresponding scripts in `src/`.

---

## 1. Receptor Discovery (receptor_discovery.json)

**32 of 46 tested receptors discovered** in the Tier 0 (foundation) environment.

The 29 receptors invariant across all 8 tiers:
coincidence_detection, causal_association, spatial_association, perceptual_similarity, compression_gain, other_detection, empathic_concern, controllability, planning, curiosity, optimism, conflict_detection, stress_detection, dynamic_repetition, functional_similarity, categorical_perception, causal_inference, probabilistic_causation, impulse_override, concept_formation, absence_observation, rule_extraction, ratio_detection, structural_invariance_math, org_boundary_detection, part_whole_detection, functional_organization

4 receptors never emerged at any tier: static_repetition, precedence_detection, completion, quantity_detection

**Key finding:** Complexity reshapes the receptor topology — it doesn't expand it. Different tiers produce different receptors, not more. Discovery count is flat across tiers (31-35).

---

## 2. Evolutionary Sweep (evolutionary_sweep.json, experiment_41_deep_sweep.json)

8 environment tiers × 3 generations × 46 receptor tests.

Tier discovery counts: T0=35, T1=33, T2=35, T3=32, T4=31, T5=31, T6=32, T7=35

Tier-specific receptors reveal what each environment selects for:
- `arousal_regulation`: only Tier 4+ (meta-cognitive environments demand arousal control)
- `chunking`: absent at Tier 0, present at Tier 1+ (complexity demands sequence compression)
- `boundary_detection`: absent at Tier 0-1, present at Tier 2+ (social/instrumental environments have sharp edges)
- `change_detection`: absent at Tier 0-1, present at Tier 2+ (multi-agent environments produce more prediction error)

---

## 3. Cross-Tier Transfer (cross_tier_transfer.json)

5×5 transfer matrix (trained in source tier, evaluated in target tier).

| Source \ Target | T0 | T1 | T2 | T3 | T4 |
|-----------------|------|------|------|------|------|
| T0 | 0.60 | -0.01 | 11.73 | -3.50 | 1.71 |
| T1 | 0.54 | -0.36 | 19.89 | -2.69 | 1.66 |
| T2 | 0.44 | -0.74 | 25.85 | -2.77 | 1.77 |
| T3 | 0.42 | -0.58 | 24.97 | -2.40 | 2.23 |
| T4 | 0.31 | -0.37 | 18.28 | -3.92 | 2.01 |

**Key findings:**
- Social environments (T2) universally benefit from transfer: 11-25x from any source
- Tool environments (T3) resist transfer: negative ratios from every source
- Transfer follows the dependency graph upward, not downward
- T3 (instrumental) is the most portable source topology

---

## 4. Population Evolution (population_evolution.json)

8 organisms competing in the same environment, 5 generations.

| Gen | Avg Fitness | Discovered | Empathy | Behavioral Prediction |
|-----|-----------|------------|---------|----------------------|
| 0 | 17,819 | 17 | 0.482 | 0.000 |
| 1 | 14,273 | 20 | 0.454 | 0.022 |
| 2 | 19,445 | 18 | 0.546 | 0.030 |
| 3 | 19,615 | 18 | 0.508 | 0.016 |
| 4 | 11,863 | 18 | 0.555 | 0.013 |

**Key findings:**
- Empathy trends upward (0.482 → 0.555) across generations
- Behavioral prediction emerged (0.022-0.030) — never appeared in single-organism training
- The social arms race created prediction pressure that static NPC environments couldn't

---

## 5. Topology Bias Inheritance (topology_inheritance.json)

5 generations with three-part transgenerational defense.

| Gen | Accuracy | Convergence Epoch | Discovered | Probe-Validated |
|-----|---------|-------------------|------------|-----------------|
| 0 | 92.9% | 15 | 12 | 12 |
| 1 | 94.4% | 2 | 14 | 12 |
| 2 | 94.9% | 1 | 14 | 13 |
| 3 | 95.1% | 0 | 14 | 12 |
| 4 | 95.3% | 0 | 14 | 13 |

**Key findings:**
- Convergence accelerates from 15 epochs to 0 (immediate competence from inherited topology)
- Discovery grows from 12 to 14 receptors (topology bias surfaces 2 novel receptors)
- 12 receptors stable across all 5 generations (the invariant trunk)
- No canalization detected — the three-part defense works

---

## 6. Scaling Track (scaling_results.json, scaling_seg_3d.json)

Architecture tested across body plans:

| Config | OBS | Actions | Accuracy |
|--------|-----|---------|----------|
| 4 limbs | 130 | 16 | 95.2% |
| 6 limbs | 156 | 22 | 92.8% |
| 8 limbs | 182 | 28 | 92.8% |
| 12 limbs | 234 | 40 | 93.3% |
| 2 segments | 238 | 42 | 92.8% |
| 3D | 158 | 22 | 92.7% |
| Mixed bodies | 182 | 28 | 96.6% |

**Key finding:** Architecture generalizes across body plans without modification. One model handles mixed 4/6/8-limbed bodies at 96.6% accuracy.

---

## 7. Grounded Language (grounding_dictionary.json, grounded_corpus.txt)

- 24 receptor groups mapped to observation indices
- 5 grounded words in shared vocabulary
- 1,013 stable concepts (compressed causal chains)
- 8 fundamental terms grounded in receptor states
- Cultural transmission: +223% reward via mental model replication

**Key finding:** "Love" and "justice" correctly return `grounded: false` — the organism hasn't developed receptors for them. The system knows what it doesn't know because knowledge has addresses.

---

## 8. Training Progression Across Steps

| Step | Accuracy | Pred MSE | Gate | Notable |
|------|---------|---------|------|---------|
| 14 (proprioception) | 97.1% | 0.003 | 0.112 | Largest accuracy jump |
| 23 (intentional signaling) | 97.4% | 0.005 | 0.140 | Highest sustained gate |
| 27 (arbitration) | 95.2% | 0.003 | 0.000 | Arbitration made slow pathway unnecessary |
| 28 (metacognition) | 95.1% | 0.003 | 0.000 | Lowest prediction error |
| 29 (concepts) | 97.0% | 0.004 | 0.126 | Slow pathway reactivated for concept matching |

# The Metabolic Knapsack: Why k_t Oscillates

## The Observation

Across 80 generations of deep time evolution (seed 42), the number of discovered receptors k_t oscillates between 72 and 119. It never climbs to the genome's 200. The mean is 102.2 with std 6.4. Gen-to-gen changes average 5.9 receptors in magnitude, with 39 gains, 34 losses, and 6 unchanged transitions. The trajectory has near-zero autocorrelation (0.14) — it's not trending, it's fluctuating around a mean.

This is not noise. It's a constrained optimization.

## The Knapsack Formulation

Each receptor r has:
- A fitness contribution F(r) when active in environment W
- A metabolic cost alpha_r (the energy/processing budget consumed by maintaining the detector)
- A set of prerequisites P(r) from the DAG (receptors that must be present for r to function)

The topology at any generation is the solution to:

```
maximize  sum_{r in R_t} F(r, W_t)
subject to  sum_{r in R_t} alpha_r <= B
            for all r in R_t: P(r) subset R_t
```

This is a precedence-constrained knapsack problem. It's NP-hard in general but has known approximation structure — and evolution IS an approximate solver (population-based stochastic search with local moves).

## Why k_t Can't Climb

The budget B is finite. Every receptor carried costs alpha_r. When a new receptor activates, it either:
1. Fits within remaining budget (k_t increases)
2. Out-competes an existing receptor whose F(r) dropped (k_t stays constant, composition changes)
3. Cannot fit — all existing receptors have higher F/alpha ratio (k_t stays, new receptor rejected)

The DAG makes this harder: to hold a canopy receptor, you must carry its entire prerequisite chain. A layer-2 receptor with 3 prerequisites costs 4 * alpha, not 1 * alpha. Deep receptors are expensive because prerequisites are carried.

## What the Data Shows

### k_t statistics (80 generations, seed 42)
- Range: 72-119 (47 spread)
- Mean: 102.2, Std: 6.4
- Correlation with fitness: 0.493 (moderate positive — more receptors helps, but isn't everything)

### Gains vs Losses by Layer
| Layer | Gained | Lost | Net |
|-------|--------|------|-----|
| Trunk | 309 | 302 | +7 |
| Branch | 400 | 387 | +13 |
| Canopy | 748 | 730 | +18 |

Canopy receptors turn over fastest (18.7 change state per generation), which is predicted by the knapsack: canopy has the most marginal entries — highest F variance across environments, highest chain cost, most competition for the remaining budget after trunk and branch are allocated.

### Stability by Layer
| Layer | Mean Stability | N |
|-------|---------------|---|
| Trunk | 55.1/80 gens | 38 |
| Branch | 55.5/80 gens | 49 |
| Canopy | 45.5/80 gens | 74 |

Trunk and branch are comparably stable (~69%). Canopy is less stable (~57%). This is predicted: trunk receptors have high F/alpha (cheap, universally useful), so they're rarely displaced. Canopy receptors have variable F (environment-dependent) and high chain cost, so they churn.

### The 18 Invariant Receptors (present in all 80 generations)
These are the receptors that NEVER get displaced — their F/alpha ratio exceeds every competitor in every environment the organism encounters:

| Layer | Receptor | Why invariant |
|-------|----------|---------------|
| T | categorical_compression | Compression is always fitness-positive |
| T | controllability | Knowing what you can affect is always useful |
| T | relational_observation | Observing relationships is always useful |
| B | absence_observation | Detecting what's missing has universal value |
| B | arousal_regulation | Regulating arousal is always fitness-positive |
| B | causal_association | Causation detection is always useful |
| B | compression_gain | Knowing compression quality is always useful |
| B | probabilistic_causation | Calibrated uncertainty is always useful |
| B | ratio_detection | Stable relationships are always useful |
| B | selective_observation | Attention to relevant channels is always useful |
| B | self_model | Self-prediction is always useful |
| B | structural_similarity | Structural matching is always useful |
| C | categorical_perception | Category warping is always useful once formed |
| C | causal_chains | Multi-step reasoning is always useful |
| C | causal_rhythm | Periodic causal events are always useful |
| C | mental_model | Having a predictive model is always useful |
| - | planning | Planning always helps |
| - | prediction | Prediction always helps |

3 trunk, 9 branch, 4 canopy, 2 unlabeled. The invariant set spans all layers. These are the knapsack items that are always in the optimal solution — their value-to-weight ratio is always above the cutoff.

### The 161-vs-119 Gap

Total unique receptors ever seen across 80 gens: 161. Maximum k_t in any single generation: 119. The gap of 42 represents receptors that are fitness-positive in SOME environments but not worth carrying in ALL environments given the budget constraint. They enter when the environment makes them valuable and exit when the environment shifts.

This is exactly the knapsack prediction: under budget pressure, the solution is not the full item set but a rotating subset. The union across solutions (161) exceeds any single solution (119) because different environments have different optimal subsets.

## The Encephalization Prediction

If the knapsack formulation is correct, maximum reachable layer depth should scale with B. Deeper receptors require longer prerequisite chains. Each chain member costs alpha. So:

```
max_reachable_depth ~ B / avg_alpha
```

Vary B across runs. If deeper budgets produce deeper topologies, we've derived encephalization pressure from first principles: brains got bigger because cognitive depth costs metabolic budget, and more budget permits deeper prerequisite chains.

This experiment is running (metabolic_budget_experiment.py, 5 conditions from very_tight to very_generous, 20 generations each).

## The Conservation Law

The key insight from Territory II: alpha_r is not just a cost — it's a conserved quantity. The total metabolic budget B is finite. Every receptor consumes some of it. The topology at any moment is the best allocation of B across available receptors given the current environment and DAG.

This reframes the non-monotonicity of k_t from "the organism gains and loses receptors unpredictably" to "the organism is continuously re-solving a constrained optimization as the fitness landscape shifts." The oscillation IS the optimization. The budget IS the constraint.

And the instrument insight from discuss2.txt: cultural tools (telescopes, notation systems, institutions) shift alpha_r off the organism's metabolic budget onto the artifact. That's why cultural cognition isn't capped the way biological cognition is — instruments launder deep detections into shallow ones, expanding the effective budget without expanding the brain.

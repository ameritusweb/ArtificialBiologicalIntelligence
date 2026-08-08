# Separation of Discriminative and Generative Dimension Selection

## Theorem

For the family of one-new-binary-latent extensions to a generative model, there exist data distributions where the dimension selected by information gain on outcome labels differs from the dimension selected by ELBO maximization, with arbitrarily large gap.

## Setup

Observations o = (x₁, x₂, y) where:

- x₁ ~ ½N(−m, ε²) + ½N(+m, ε²), a symmetric bimodal mixture
- x₂ ~ U[0,1], uniform on the unit interval
- y = 𝟙[x₂ > ½], deterministic function of x₂
- x₁ independent of (x₂, y)

**Current model** (before adding any latent):

The organism models x₁ with a single Gaussian (it hasn't discovered the bimodality):

    p₀(x₁) = N(0, m² + ε²)

This is the moment-matched Gaussian — same mean (0) and same variance (m² + ε²) as the true mixture. It's the best single Gaussian approximation.

The organism models (x₂, y) correctly:

    p₀(x₂, y) = U[0,1] · δ(y − 𝟙[x₂ > ½])

**Two candidate latent extensions:**

- z_A = 𝟙[x₂ > ½]: threshold on x₂, outcome-relevant
- z_B ∈ {0,1}: mode indicator for x₁, outcome-irrelevant

## ELBO Calculation for z_A (x₂ threshold latent)

**Extended model:**

    p_A(x₂, y, z_A) = p(z_A) · p(x₂|z_A) · p(y|z_A)

with:
- p(z_A = 0) = p(z_A = 1) = ½
- p(x₂|z_A = 1) = U[½, 1] (density = 2 on [½, 1])
- p(x₂|z_A = 0) = U[0, ½] (density = 2 on [0, ½])
- p(y|z_A) = δ(y − z_A) (deterministic)

**Optimal posterior:**

Since y determines z_A exactly:

    q*(z_A|o) = δ(z_A − y)

This is a point mass. q* has entropy H(q*) = 0.

**ELBO with optimal q*:**

    ELBO_A = E_q*[log p_A(x₂, y, z_A)] − E_q*[log q*(z_A)]

Term 1 — joint log-likelihood under p_A:

    E_q*[log p_A(x₂, y, z_A)]
    = E_q*[log p(z_A) + log p(x₂|z_A) + log p(y|z_A)]
    = log(½) + log(2) + log(1)
    = −log 2 + log 2 + 0
    = 0

Breakdown:
- log p(z_A) = log(½) = −log 2 (prior cost of the latent)
- log p(x₂|z_A) = log(2) (half-uniform has density 2 within its half)
- log p(y|z_A) = log(1) = 0 (deterministic, always satisfied)

Term 2 — entropy cost:

    E_q*[log q*(z_A)] = 0 (point mass)

**ELBO_A = 0**

**Current ELBO (without z_A):**

    ELBO₀ = E[log p₀(x₂, y)]
    = E[log(1)] + E[log(1)]
    = 0

(Uniform density = 1, deterministic y always matches.)

**Net ELBO gain:**

    ΔELBO_A = ELBO_A − ELBO₀ = 0 − 0 = 0

**The gain is exactly zero.** The log 2 improvement from sharper density on x₂ (half-uniform has density 2 vs uniform's density 1) is exactly cancelled by the log 2 cost of the latent prior p(z_A) = ½.

**Why the cap is log 2 in general:** Even if the current model for x₂ were worse than U[0,1] (e.g., a misspecified Gaussian), the maximum possible improvement from a binary latent is bounded by H(z_A) = log 2, because z_A carries at most log 2 bits. The reconstruction improvement is bounded by the mutual information I(x₂; z_A) ≤ H(z_A) = log 2. So:

    ΔELBO_A ≤ log 2 (in all cases)

and equals 0 when the current model for x₂ is already correct.

## ELBO Calculation for z_B (x₁ mode indicator latent)

**Extended model:**

    p_B(x₁, z_B) = p(z_B) · p(x₁|z_B)

with:
- p(z_B = 0) = p(z_B = 1) = ½
- p(x₁|z_B = 1) = N(+m, ε²)
- p(x₁|z_B = 0) = N(−m, ε²)

**Optimal posterior:**

    q*(z_B = 1|x₁) = N(x₁; m, ε²) / [N(x₁; m, ε²) + N(x₁; −m, ε²)]

For large m/ε, this is ≈ 1 when x₁ > 0 and ≈ 0 when x₁ < 0. Essentially a point mass at the correct mode.

**ELBO with optimal q*:**

    ELBO_B = E_q*[log p_B(x₁, z_B)] − E_q*[log q*(z_B)]

Term 1 — expected joint log-likelihood under p_B:

    E_q*[log p_B(x₁, z_B)]
    = E_q*[log p(z_B) + log p(x₁|z_B)]
    = E_q*[log(½)] + E_q*[log N(x₁; ±m, ε²)]
    = −log 2 + E[log N(x₁; m_correct, ε²)]

where m_correct = +m when z_B = 1 and −m when z_B = 0.

    E[log N(x₁; m_correct, ε²)]
    = E[−½ log(2πε²) − (x₁ − m_correct)²/(2ε²)]
    = −½ log(2πε²) − ½

(because E[(x₁ − m_correct)²|z_B] = ε² under the correct component)

So Term 1 = −log 2 − ½ log(2πε²) − ½

Term 2 — entropy cost:

For large m/ε, q* is nearly a point mass:

    E_q*[log q*(z_B)] ≈ 0

**ELBO_B ≈ −log 2 − ½ log(2πε²) − ½**

**Current ELBO (without z_B):**

    ELBO₀ = E[log p₀(x₁)]
    = E[log N(x₁; 0, m² + ε²)]
    = −½ log(2π(m² + ε²)) − E[x₁²]/(2(m² + ε²))

Since E[x₁²] = m² + ε² (second moment of the mixture):

    ELBO₀ = −½ log(2π(m² + ε²)) − ½

**Net ELBO gain:**

    ΔELBO_B = ELBO_B − ELBO₀
    = [−log 2 − ½ log(2πε²) − ½] − [−½ log(2π(m² + ε²)) − ½]
    = −log 2 − ½ log(ε²) + ½ log(m² + ε²)
    = −log 2 + ½ log((m² + ε²)/ε²)
    = ½ log(1 + m²/ε²) − log 2

**This is unbounded as m/ε → ∞.**

For m = 10, ε = 1: ΔELBO_B ≈ ½ log(101) − log 2 ≈ 2.31 − 0.69 = 1.62 nats
For m = 100, ε = 1: ΔELBO_B ≈ ½ log(10001) − log 2 ≈ 4.61 − 0.69 = 3.92 nats
For m = 1000, ε = 1: ΔELBO_B ≈ ½ log(1000001) − log 2 ≈ 6.91 − 0.69 = 6.22 nats

## Information Gain Calculation

**IG for z_A:**

    IG(y; z_A) = H(y) − H(y|z_A) = log 2 − 0 = log 2

(z_A determines y exactly, so H(y|z_A) = 0)

**IG for z_B:**

    IG(y; z_B) = H(y) − H(y|z_B) = log 2 − log 2 = 0

(z_B is independent of y by construction, so H(y|z_B) = H(y))

## The Separation

| Criterion | Selects | Value |
|---|---|---|
| Information gain (discriminative) | z_A (x₂ threshold) | log 2 |
| ELBO maximization (generative) | z_B (x₁ mode indicator) | ½ log(1 + m²/ε²) − log 2 |

For m/ε > √3 (i.e., m²/ε² > 3), ΔELBO_B > log 2 > ΔELBO_A = 0. The generative criterion strictly prefers z_B. The discriminative criterion strictly prefers z_A. They select different dimensions.

The gap ΔELBO_B − ΔELBO_A = ½ log(1 + m²/ε²) − log 2 is unbounded as m/ε → ∞.

## The Structural Asymmetry

This is not a pathological construction. The asymmetry is structural:

1. **Any outcome-relevant binary latent** contributes at most H(y) ≤ log|Y| to the ELBO, because its value is capped by the entropy of the outcome it predicts. For binary outcomes, this is at most log 2 nats.

2. **Outcome-irrelevant structure** in continuous observations can contribute unbounded ELBO improvement, because continuous distributions can have arbitrarily large gaps between the single-component approximation and the mixture truth.

For every generative family over the full observation vector, there exist distributions where the outcome-irrelevant dimension has larger ELBO gain than the outcome-relevant one. The discriminative criterion always selects the outcome-relevant dimension. The generative criterion may not. ∎

## The Coincidence

On the degenerate generative model p(y) that observes only y:

    max_z ΔELBO(z) = max_z [H(y) − H(y|z)] = max_z IG(y; z)

ELBO maximization and IG maximization are identical. The separation exists only when the generative model covers more than outcomes — when it models the full observation space. The interpolation parameter is the likelihood weight on outcomes vs everything else.

## Implications

The dimension-selection mechanism that optimizes survival outcomes (information gain on fitness-relevant labels) is not equivalent to the mechanism that optimizes world-model quality (ELBO on full observations). They agree when the model cares only about outcomes. They diverge when the model also cares about structure. The divergence is unbounded.

**Correction:** This divergence is population-level, not finite-sample. Both criteria are evaluated on the true distribution. They disagree in the infinite-data limit. The disagreement is about what "better model" means, not about estimation error. See docs/separation_theorem.md for the rigorous proof with exact bounds and no asymptotic error terms.

A system that selects dimensions by survival relevance is performing variational inference on a value-deformed model, not on the world model. The deformation requires the fitness function as input. It is not derived from the generative model. No fixed preference weight β reproduces the mechanism across all environments (Corollary 4.1 in separation_theorem.md), and the meta-model absorption predicts the wrong gate on the separation distribution (Proposition 2).

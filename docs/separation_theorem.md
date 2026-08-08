# Separation of Discriminative and Generative Dimension Selection

**Informal statement.** There is a family of data distributions on which the information-gain criterion and the free-energy criterion select dimensions on *different coordinates* of the observation space, with a free-energy gap that grows without bound. The separation holds at the population level (infinite data), the bounds are exact (no asymptotic error terms), and the two criteria are connected by a one-parameter deformation: information gain is precisely the infinite-preference limit of free-energy selection, and no finite preference weight reproduces it across all environments.

---

## 1. Setup

**Observation space.** O = ℝ × [0,1] × {0,1}, with generic element o = (x₁, x₂, y). Here y is the outcome variable. Fix the base measure

    μ = Leb(ℝ) × Leb([0,1]) × counting({0,1}).

All "densities," cross-entropies, and (differential) entropies below are taken with respect to μ or its marginals; Gibbs' inequality holds for densities with respect to any common base measure, which is the only fact about μ we use.

**Candidate dimensions.** A candidate new dimension is a binary threshold detector

    z_{j,θ} : o ↦ 1[x_j > θ],   j ∈ {1,2},  θ ∈ ℝ.

Write Z for the set of all candidates. This matches the mechanism under study: the discovered dimension *is* its activation function (a threshold on an observed feature).

**Criterion D (discriminative / information gain).** Select

    argmax_{z ∈ Z}  IG(z) := H(y) − H(y | z) = I(y ; z).

**Criterion G (generative / free energy).** Fix a baseline model family M₀ and an extended family M(z) for each candidate, and select

    argmax_{z ∈ Z}  Δ(z) := L(z) − L₀,

where L₀ and L(z) are the best achievable population ELBOs (expected per-observation evidence lower bounds under the true distribution P).

**Model families.**

- Baseline M₀: product models p(x₁)·p(x₂)·p(y) with p(x₁) Gaussian, p(x₂) an arbitrary density on [0,1], p(y) Bernoulli.

- Extended M(z): naive-Bayes models gated by the new latent,

      p(g) · p(x₁ | g) · p(x₂ | g) · p(y | g),   g ∈ {0,1},

  with each conditional in the same family as the baseline (Gaussian / density / Bernoulli), and with the recognition model clamped to the detector: q(g | o) = δ_{z(o)}. Since g is discrete and q is a point mass, the ELBO is

      L(z) = sup_{p ∈ M(z)}  E_P[ log p(z(o)) + log p(x₁ | z(o)) + log p(x₂ | z(o)) + log p(y | z(o)) ].

  This is the gated-meta-model form: the latent gates the generative model, exactly as in the meta-model absorption (p(o, s_active(g))·p(g)), specialized to one candidate gate.

---

## 2. The construction

For parameters m > 0, ε > 0, define P = P_{m,ε} by:

    x₁ ~ ½·N(−m, ε²) + ½·N(+m, ε²)
    x₂ ~ Uniform[0,1]
    y  = 1[x₂ > ½]        (deterministic)
    x₁ ⫫ (x₂, y)

Write r = m²/ε² (squared mode separation). Two features of this distribution do all the work: the outcome is perfectly determined by x₂, and x₁ carries an arbitrarily large amount of structure (as r → ∞) that is completely irrelevant to the outcome.

**Baseline value L₀.** The three blocks are independent, so L₀ decomposes:

- x₁-block: the cross-entropy-optimal Gaussian is the moment-matched one. E[x₁] = 0, Var(x₁) = m² + ε². Value: −½·log(2πe(m² + ε²)).
- x₂-block: the truth (uniform) is in the family. Value: 0.
- y-block: Bernoulli(½) is optimal. Value: −log 2.

So L₀ = −½·log(2πe(m² + ε²)) − log 2.

---

## 3. Lemma 1 (outcome cap)

**Lemma 1.** For every candidate of the form z = f(x₂) (in particular every z_{2,θ}),

    Δ(z) ≤ log 2 = H(y).

*Proof.* Split the ELBO into the x₁-part and the (g, x₂, y)-part.

x₁-part: z = f(x₂) is independent of x₁, so for any conditional family,

    E_P[ log p(x₁ | z) ] = Σ_s P(z = s) · E_P[ log p(x₁ | s) ]  ≤  −½·log(2πe(m² + ε²)),

since the conditional law of x₁ given z = s equals its marginal law, and each term is a Gaussian cross-entropy against that same marginal. The x₁-part therefore cannot exceed its baseline value.

(x₂, y)-part: define the induced function

    p̂(x₂, y) := p(f(x₂)) · p(x₂ | f(x₂)) · p(y | f(x₂)).

Then ∫ Σ_y p̂ dμ = Σ_s p(s) · ∫_{f = s} p(x₂ | s) dx₂ · Σ_y p(y | s) ≤ Σ_s p(s) = 1, so p̂ is a sub-density on the (x₂, y) block. By Gibbs' inequality,

    E_P[ log p̂(x₂, y) ] ≤ E_P[ log f_P(x₂, y) ] = −h(x₂, y) = 0,

because the true (x₂, y)-density with respect to Leb × counting is the indicator 1[y = 1[x₂ > ½]] on [0,1], whose entropy is 0. The baseline value of this block is 0 + (−log 2) = −log 2. Hence the block gain is at most log 2, and the total gain Δ(z) ≤ log 2. ∎

The proof only used that y is discrete and that x₂'s marginal is already perfectly modeled. The cap is H(y) ≤ log|Y| in general — a constant that does not grow with any property of the environment. The same argument applies verbatim if y is a noisy function of x₂ (e.g. y | x₂ ~ Bern(σ(κ(x₂ − ½)))) or multiclass.

**Achievability.** For z = z_{2,½}: take p(g) = Bern(½), p(x₂ | g) uniform on the corresponding half-interval (density 2), p(y | g) deterministic, p(x₁ | g) the baseline Gaussian. The block value is −log 2 + log 2 + 0 = 0. Hence

    Δ(z_{2,½}) = log 2   exactly.

---

## 4. Lemma 2 (unbounded continuous gain)

**Lemma 2.** For the candidate z_B = z_{1,0} = 1[x₁ > 0],

    Δ(z_B) ≥ ½·log(1 + r) − log 2,   for all r > 0.

*Proof.* Exhibit a model: p(g) = Bern(½), p(x₁ | g = s) = N(s·m, ε²) with s ∈ {−1, +1} identified with g ∈ {0,1}, and baseline-optimal p(x₂ | g) = uniform, p(y | g) = Bern(½). The ELBO is

    L(z_B) ≥ −log 2 − E_P[ −log N(x₁ ; sign(x₁)·m, ε²) ] + 0 − log 2
           = −2·log 2 − ½·log(2πε²) − (1/2ε²)·E_P[ (x₁ − sign(x₁)·m)² ].

Bound the quadratic term pointwise. Let c(x₁) ∈ {−1, +1} be the true component from which x₁ was drawn. If sign(x₁) = c(x₁) the two quadratics agree. If sign(x₁) ≠ c(x₁) — say x₁ < 0 was drawn from the +m component — then

    |x₁ + m| = | m − |x₁| | ≤ m + |x₁| = |x₁ − m|,

so the mismatched squared deviation is *smaller* than the matched one, pointwise. (Symmetrically for the other component.) Therefore

    E_P[ (x₁ − sign(x₁)·m)² ] ≤ E_P[ (x₁ − c(x₁)·m)² ] = ε².

Hence L(z_B) ≥ −2·log 2 − ½·log(2πeε²), and

    Δ(z_B) = L(z_B) − L₀ ≥ ½·log( (m² + ε²)/ε² ) − log 2 = ½·log(1 + r) − log 2.  ∎

Note there is no asymptotic error term: the mode-assignment mistakes made by the hard threshold are pointwise *cheaper* than the true assignments, so the bound is exact for every r, not just in the well-separated limit.

---

## 5. Theorem 1 (population-level separation)

**Theorem 1.** For every c > 0, if r ≥ 16·e^{2c} − 1 then on P_{m,ε}:

1. Criterion D selects a threshold on x₂ (namely z_{2,½}, with IG = log 2, the maximum possible), and IG(z) = 0 for every x₁-threshold.
2. Criterion G selects a threshold on x₁, and its advantage over every x₂-threshold is at least c:

       Δ(z_B) − sup_θ Δ(z_{2,θ}) ≥ ½·log(1 + r) − 2·log 2 ≥ c.

Hence the two criteria select dimensions on different coordinates, and the free-energy gap between their selections is unbounded over the family {P_{m,ε}}.

*Proof.* (1) z = f(x₁) is independent of y, so IG = 0 for every x₁-threshold; IG(z_{2,½}) = H(y) − 0 = log 2, which is the maximum achievable by any binary variable. (2) By Lemma 1, every x₂-threshold has Δ ≤ log 2. By Lemma 2, Δ(z_B) ≥ ½·log(1 + r) − log 2. The condition r ≥ 16·e^{2c} − 1 gives ½·log(1 + r) − 2·log 2 ≥ c > 0, so the argmax of Δ over Z lies among x₁-thresholds. ∎

At c = 0 the crossover is r > 15: mode separation m > ε·√15 ≈ 3.9·ε already suffices.

**Remark (population level).** Both criteria are evaluated on the true distribution — this is the infinite-data limit. The criteria do not merely diverge at finite sample sizes and reconcile asymptotically; they *disagree in the limit*. The disagreement is about what "better model" means, not about estimation error.

---

## 6. Proposition 2 (bridge: the meta-model's gradient descent implements Criterion G)

The original question was about gradient descent on a fixed meta-functional F*, not about a selection rule. These coincide.

**Proposition 2 (sketch).** Let Z_fin ⊂ Z be a finite candidate grid. Define the gated meta-model with gate variable γ ∈ Z_fin ∪ {∅}, uniform prior p(γ), and p*(o | γ = z) the extended family M(z) (with γ = ∅ the baseline). Given n i.i.d. observations, under standard regularity (parameters consistently estimable on each branch),

    log p*(data | γ = z) = n·( L₀ + Δ(z) ) + O_P(√n),

so the posterior q*(γ) ∝ p(γ)·p*(data | γ) concentrates on argmax_z Δ(z) as n → ∞. Mean-field / natural-gradient variational descent on F* over q(γ) converges to this posterior. ∎

That is: the gate that gradient descent on the fixed functional F* opens is the Δ-maximizing gate. On P_{m,ε} with r > 15, F*-descent opens an x₁-gate. The mechanism opens an x₂-gate. The meta-model absorption therefore does not merely fail to *explain* the mechanism (the point of Problems 1–4 in the companion document); on this distribution it makes a different *prediction*.

---

## 7. Proposition 3 (coincidence on the outcome-only model)

The quantifier "for any generative model" in the strongest version of the claim is false, and the failure is exactly located.

**Proposition 3.** Let the generative family be over y alone, with z as covariate: models p(y | z) with a free Bernoulli table per value of z, baseline p(y) Bernoulli. Then for every candidate z and every distribution P,

    Δ_y(z) = H(y) − H(y | z) = I(y ; z) = IG(z).

*Proof.* The optimal baseline achieves −H(y); the optimal conditional model achieves −H(y | z); subtract. ∎

So information-gain maximization *is* free-energy minimization — for the degenerate generative model whose entire observable world is the outcome. The separation in Theorem 1 shows it is not free-energy minimization for any model of the full observation vector with enough capacity to notice x₁. The scope of the correct claim is therefore one-sided:

- For the outcome-only family: the criteria coincide on all P.
- For any family over the full vector (satisfying the block-capacity conditions of Sections 1–4): there exist P inverting the ranking by an arbitrarily large margin.

---

## 8. Proposition 4 (the preference deformation and its crossover)

The two endpoints of Section 7 are connected by one parameter. For β ∈ [0, ∞), define the β-deformed criterion by up-weighting the outcome likelihood (equivalently: a preference prior p̃(o) ∝ p(o)·exp(β·U(o)) with U supported on the outcome coordinate; equivalently: tempering p(y | ·)^β):

    Δ_β(z) := Δ_x(z) + β·Δ_y(z),

where Δ_x and Δ_y are the gains on the non-outcome and outcome blocks respectively. β = 1 is Criterion G; the outcome-only model of Proposition 3 is the formal β = ∞.

**Proposition 4.** On P_{m,ε}:

    Δ_β(z_{2,½}) = β·log 2,      Δ_β(z_B) = ½·log(1 + r) − log 2,

so the deformed criterion selects the outcome-aligned dimension if and only if

    β > β*(r) := ½·log₂(1 + r) − 1.

*Proof.* From Section 3, the entire gain of z_{2,½} sits in the (x₂,y) block, and within that block the x₂-part nets to zero (density-2 halves gain log 2, the gate costs log 2), so Δ_x(z_{2,½}) = 0 and Δ_y(z_{2,½}) = log 2. From Section 4, the entire gain of z_B sits in the x₁ block: Δ_y(z_B) = 0. Compare. ∎

**Corollary 4.1 (no fixed preference weight suffices).** β*(r) → ∞ as r → ∞. Hence for every fixed β < ∞ there is an environment (r large) on which the β-deformed free-energy criterion still disagrees with information gain. Conversely, as β → ∞ the deformed ranking converges to argmax Δ_y = argmax I(y ; z), i.e. to Criterion D exactly (the family p(y | z) is saturated, so Δ_y attains I(y ; z)).

**Interpretation.** Information gain on outcome labels is the *lexicographic-preference limit* of variational free-energy selection: infinitely sharp preferences over outcomes, zero residual weight on modeling the world. It is not equivalent to free-energy selection on the world model (Theorem 1), it is not equivalent to free-energy selection under any fixed finite preference sharpness (Corollary 4.1), and it is exactly equivalent at the degenerate limit (Proposition 3). The distance between the discovery mechanism and the variational absorption is therefore not a modeling detail — it is the full range of the preference parameter.

---

## 9. Remarks on scope

**(a) What is and is not proved.** Theorem 1 is a one-sided separation: it refutes "the discovery mechanism's selection is a free-energy selection for the full generative model," and Corollary 4.1 refutes "…for some fixed preference-deformed model." It does not refute — and Proposition 3 shows one cannot refute — "…for *some* generative model," because the outcome-only model realizes the equivalence. The correct summary: the mechanism is variational only relative to a value-deformed model whose deformation must itself be supplied, and no single deformation works across environments.

**(b) The source of the asymmetry.** The cap in Lemma 1 is H(y) ≤ log|Y|: discrete outcome entropy is bounded by the alphabet. The gain in Lemma 2 is a differential-entropy gap, unbounded over distributions on a continuous coordinate. The separation is thus structural (discrete-vs-continuous), not an artifact of the specific mixture: any construction pairing a bounded-alphabet outcome with arbitrarily rich outcome-irrelevant continuous structure will reproduce it.

**(c) Role of the budget.** With no per-dimension cost, a system could eventually add both dimensions and the criteria differ only in order. The separation should be read as a statement about ranking — which dimension is added first, or under a metabolic cost, at all. Since any mechanism with bounded resources operates under such a budget, ranking is the operationally meaningful object, and Proposition 2 shows the meta-model's own dynamics implement the Δ-ranking.

**(d) Robustness.** The deterministic link y = 1[x₂ > ½] is inessential (the cap argument uses only discreteness of y and a well-modeled x₂-marginal); noise in y only lowers the discriminative side's attainable gain. The clamped recognition model matches the mechanism (the dimension is its threshold function); relaxing to soft posteriors changes constants, not the cap/unbounded structure. The naive-Bayes gating matches the meta-model absorption's p(o, s_active(g))·p(g) form.

---

## 10. Open ends

1. **Non-factorized gates.** The extended family here gates blocks conditionally independently. Whether richer gated families (the gate modulating dependencies *between* blocks) can rescue the absorption without smuggling the answer into the prior reduces to the same asymmetry: any bounded-outcome cap versus unbounded continuous slack argument should survive, but this is stated here only for naive-Bayes gating.

2. **Interaction with singular phase transitions.** In singular learning theory, a fixed Bayesian free energy governs discrete transitions between regions of different effective dimensionality without pre-enumerated gates. The transitions it governs are Δ-driven (evidence-driven), so Theorem 1 predicts they will track the generative ranking, not the discriminative one. A distribution of the P_{m,ε} type embedded in a singular model family would test whether outcome-blind phase transitions and outcome-driven dimension discovery can be told apart empirically — this seems to be the sharpest available experiment separating "topology change as free-energy phase transition" from "topology change as discriminative discovery."

3. **The β-schedule.** Corollary 4.1 shows no fixed β reproduces the mechanism. An environment-dependent schedule β(P) ≥ β*(r(P)) does — but β*(P) depends on the un-modeled structure of P, which is precisely what the system does not yet know. Whether a system can estimate its own β* online (i.e., estimate how much outcome-irrelevant structure it is ignoring) without already possessing the dimensions needed to represent that structure looks like the recursive core of the original topology-change problem, restated quantitatively.

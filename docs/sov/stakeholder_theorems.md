# The Stakeholder Theorems
### What Animating a Domain Provably Gains Over Predicting It

**Status:** Formal note, registered 2026-08-10 (T157's mathematical core).
**Companion to:** the Separation Theorem (in-house ancestor), SOV core §8a (options pricing), the Ledgerless Economy, stakeholder_requirements.md (the instance).
**Core claim:** Under bounded representation, consequence-carving and prediction-carving provably diverge, with unbounded gain in constructed cases; the SOV layer adds an optimal-stopping advantage over forced-commitment architectures and a detection-theoretic advantage on regime novelty. The gains exist by theorem; their magnitude in a given domain is what the animated stakeholder measures.

---

## Theorem 1 — The Carving Separation

**Setup.** A stochastic process `X_t` over a state space; an action set `A`; a utility `u(X, a)`; and the load-bearing condition — a **capacity bound**: representations are quotients φ with at most k cells (finitely many receptors; the metabolic knapsack T104 is the in-house form). Define the two bounded learners:

    φ_P^k = argmax_{|φ|≤k}  I( φ(X_t) ; X_{t+1} )                    (predictive carving)
    φ_S^k = argmax_{|φ|≤k}  sup_π  E[ u(X, π(φ(X))) ]                (consequential carving — the D-metric quotient)

and the gain `G(k) = V(φ_S^k) − V(φ_P^k)`.

**Claims.**
(a) `G(k) ≥ 0` (trivial — S optimizes V).
(b) **`G(k)` is unbounded at fixed k** as the domain grows (construction below).
(c) **`G(k) → 0` as `k → ∞`** — an unbounded model contains everything; the separation is a **scarcity phenomenon**. The entire advantage lives where representation is expensive, which is everywhere real.

**The construction for (b) — cap-and-slack, five lines.** Let `X = (X¹, …, Xⁿ, Z)` with independent factors. Each `Xⁱ` is perfectly predictable (`Xⁱ_{t+1} = Xⁱ_t`: one full bit of predictive information each; consequence-irrelevant). `Z` is rare — `P(Z=1) = ε` — with a weak precursor carrying tiny mutual information `δ ≪ 1` bit; missing it costs `L`. A `k=1` predictor spends its cell on some `Xⁱ` (1 bit ≫ δ — this MAXIMIZES its objective) and eats expected loss `εL`. The stakeholder spends its cell on the precursor and hedges. The utility gap is ≈ `εL` with **`L` a free parameter** — unbounded — while the predictive-score gap between the carvings stays bounded by one bit, and the predictor's own loss barely registers the omission (ε-weighted).

**The asymmetry in one sentence:** *prediction weights errors by probability; life weights them by consequence; the two weightings are independent, so a bounded learner optimizing one provably starves the other.* Crashes, regime shifts, and climate transitions are exactly this construction's regime: rare, low-mutual-information, all-stakes.

**Ancestors (named, per the Grassmann discipline):** Blackwell's comparison of experiments (1953) — under capacity bounds there is no universal sufficient statistic; sufficiency is decision-relative. The value-equivalence principle in model-based RL (Grimm et al.) — small value-carved models beat small prediction-carved models. The in-house Separation Theorem — IG and VFE select different dimensions, no fixed β across environments (the receptor-level instance, TNB-ready). SOV-P2b — the same cut at the form level.

**What ERTI adds beyond the theorem:** the argmax defining `φ_S^k` is directly intractable (the representation depends on the policy, which depends on the representation). **The receipt economy is a constructive anytime procedure for it** — carve, bill by lived consequence, keep what pays rent — with the D-metric as its fixed-point criterion. The theorem says the prize exists; the organism is the algorithm that collects it.

## Theorem 2 — The Option Value of Held-Open Slots

**Setup.** Regime shifts arrive at unknown times. Compare: **forced commitment** (every prediction architecture outputs a committed representation each step — closure always) vs the **SOV exercise rule** (close only when `E[restructuring gain] > rent × connectivity`; act meanwhile via robust policies over the feasible set).

**Claim.** Under a shift process with rare large regime changes, forced commitment pays the Reopen cascade — cost proportional to committed connectivity — at every shift, *silently, as corrupted statistics*; the SOV policy pays rent. The regret difference **is the option premium**, and the American-option early-exercise results give the dominance condition: hold while volatility and connectivity are high. Provable under assumptions (a shift process; a Reopen-cost model in connectivity); standard optimal-stopping machinery.

**Empirical silhouette already in the ledger:** CV-P1's final matrix (release pays exactly under global model failure) and F13's 195 honest retractions — the payments a forced-commitment architecture would have made invisibly.

## Theorem 3 — Novelty Identifiability

**Setup.** A monitor must distinguish "high loss because the world is noisy" (variance burst) from "high loss because my vocabulary lacks this regime" (structured novelty). A scalar loss statistic conflates them — the aleatoric/epistemic collapse; T154's unlearned/unsensed conflation stated for monitors.

**Claim.** The structured-404 statistic (the unassigned pool, admission gated by activation coherence) is a test for the *coherence* that defines the alternative hypothesis, to which the loss statistic is blind. Key lemma = D1's noise immunity: noise cannot sustain a slot — no stable profile for the cluster criterion to bind. Hence an ROC separation: detection rates for coherent new regimes that no loss threshold achieves at matched false-alarm rate. Formalizable as a two-hypothesis detection problem; the proof burden is the coherence-blindness of the scalar statistic, which is definitional.

**Domain reading:** for markets and climate the entire question is "new regime or bad month?" — this theorem is the instrument's sharpest edge, and F15's regime-change alarm is its degenerate (aggregate-grain) form.

## What Stays Irreducibly Empirical

The theorems establish existence and locate the gain: it is large exactly where value-relevance and statistical salience misalign — tails, regime boundaries, precursors. They cannot give the magnitude in a particular domain: where salience and relevance happen to align, G is small there. That magnitude is what the animated stakeholder *measures* — the theory says where to look; the organism is the instrument that looks (P79). Market-specific caution: reflexivity — other agents adapt, carvings are per-stakeholder and possibly self-eroding at scale. The theorems concern representation, not alpha.

---

*Registered 2026-08-10. The three gains in one line: a bounded mind that pays rent in consequences learns a different world than one that pays rent in likelihoods — and it knows when to stay uncertain, and it knows when the world has changed.*

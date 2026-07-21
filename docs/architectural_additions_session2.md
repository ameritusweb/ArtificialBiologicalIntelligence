# Architectural Additions: Second Review Session
## Findings Since the Dependency Graph Document

---

## Overview

This document covers findings from the second phase of Fable review, covering four areas: encoder inspectability and basis, the log-shaping failure mode, the probe budget fix and its transgenerational extension, and the Lamarckian objection resolved. Each section states the finding, the proposed fix, and the current status of the architecture.

---

## Finding 1: Inspectability Requires Fixing the Basis, Not Just the Weights

### The Gap

The previous architecture proposed storing the contrastive encoder's per-family Mahalanobis reweighting as explicit weight vectors to achieve inspectability. This was the wrong fix for the right problem.

The weight vectors are already explicit — that was already true of the per-family diagonal reweighting as designed. The problem is what basis those weights are expressed in. If the shared encoder beneath is an unconstrained neural map from raw signals to latent space, the latent dimensions are unnameable and can rotate under retraining. A weight of 0.8 on dimension 37 is auditable only if dimension 37 means something stable. It doesn't, if the encoder is free to reorganize its latent space.

Inspectability of the explicit layer is entirely parasitic on the interpretability of the layer below it.

### The Fix

The design decision that actually buys inspectability: make the shared encoder **linear or affine from raw signal space**. Then the composed metric — linear map plus diagonal reweighting — is a quadratic form over named raw signals, and every audit question becomes answerable:

- Which signal channels does family F condition on?
- Did those weights change?
- Does the change correspond to an environment shift?

**The cost:** A linear encoder cannot represent interaction structure — "this dimension matters only when that one is high." That is exactly the kind of context-conditionality nonlinear encoders exist to capture.

**The resolution:** Run constrained (linear) and unconstrained encoders in shadow mode and measure the held-out prediction gap.

- Small gap → buy interpretability outright. The environment's context structure is approximately linear and the linear encoder is the right choice.
- Large gap → that is a *finding* about the environment, not just a verdict against the linear encoder. Nonlinear context structure in this environment is worth knowing independently.
- Middle option (recommended default): a **two-stage design** — a small explicit library of nonlinear features that is itself named and frozen, with all ongoing learning confined to linear weights over that library. Opacity quarantined to a fixed auditable feature set; drift confined to readable weights.

### Audit Practice Cautions

Two failure modes in the auditing practice itself:

**1. Plausibility as ground truth.** "Check whether weights make causal sense" quietly installs the auditor's intuition as ground truth. Weights that look wrong may be the system's most valuable content — the encoder discovering a real but unintuitive dependency is precisely the case where it is earning its keep. The audit should be primarily **differential, not absolute**: flag weight changes uncorrelated with environment shifts; flag divergence between families that previously agreed. Anomaly detection against the encoder's own history, not plausibility judgment against human priors.

**2. Audit feedback creating concealment pressure.** If audit results feed back into training — penalizing implausible-looking weights — the system faces pressure to look plausible in the audited layer while relocating its actual representational work into whatever layer is not audited. With a nonlinear encoder under a diagonal reweighting, the diagonal stays clean and legible while the distinctions migrate downward into the latent map. **Audit the whole composed metric or the opacity just moves.**

### Status

**Open — design decision required.** The linear vs. featurized vs. unconstrained encoder choice must be resolved by shadow-mode measurement before the encoder section of the framework document can be considered complete. The two-stage featurized design is the recommended default pending that measurement.

---

## Finding 2: The Log-Shaping Failure Mode

### The Gap

All three previously stated mitigations for encoder failure shared a hidden assumption: that the append-only log is a clean reference. The mitigations were:

1. The encoder only shapes retrieval, not the log — bounding damage
2. The cross-metric rank correlation monitor catches encoder drift early
3. The encoder can be retrained from scratch from the log at any time

The assumption fails because **the encoder writes to the log through behavior**. It shapes retrieval → retrieval shapes action selection → actions determine which triples get logged. The log is immutable and integrity-protected, but its *coverage* is policy-dependent, and the policy is encoder-conditioned.

A biased encoder steers the agent away from contexts that would falsify it. The log fills with triples collected under that steering. Retraining from scratch faithfully reproduces the bias from the biased sample. **Immutability protects integrity. Nothing in the previous design protects representativeness.**

### Why Curiosity Doesn't Rescue This

The obvious response — "the intrinsic curiosity signal will drive exploration into neglected regions" — fails because:

- Curiosity targets low-certainty mappings
- Certainty is computed from encoder-conditioned retrieval
- A confidently wrong encoder reports *low uncertainty* precisely where it is wrong

The exploration signal is inside the loop it is supposed to break. Every uncertainty-driven mechanism in the architecture inherits the encoder's blind spots.

### The Fix: Encoder-Independent Probe Budget

A permanent small budget of **encoder-independent action selection** — epsilon-random, or coverage-driven in raw signal space using an encoder-free notion of "unvisited." Triples collected under this probe budget are tagged in the log as unconditionally sampled.

This buys three things simultaneously:

1. **The log permanently contains an unbiased audit stratum.** Retrain-from-log regains its guarantee: train on everything, validate on probe triples.
2. **Loop closure becomes detectable.** A specific statistic: encoder-conditioned predictions degrading on probe triples while remaining fine on policy-collected ones. This is visible before any other diagnostic fires.
3. **Off-distribution coverage.** The probe stream is the data the existing diagnostics were blind without.

### Status

**Resolved — with a caveat.** See Finding 3 for the transgenerational extension that partially reopens this.

---

## Finding 3: The Transgenerational Compounding Problem — Three-Part Defense

### The Gap

The probe budget fix (Finding 2) works within a single lifetime. It breaks the loop between encoder bias and log coverage for one agent. But the architecture includes inheritance of encoder weights across generations — and a simple probe rate floor is not sufficient on its own.

The threat model: a biased parental metric steers offspring data collection from birth. If the bias produces confident short-term performance, it gets selected for. A lineage may select the probe rate toward zero, cancelling the audit exactly where it is most needed. Within-generation gating cannot see slow transgenerational compounding. The fix needs three parts, only one of which is evolvable.

### The Three-Part Defense

**Part 1: Constitutional Floor**

The probe rate minimum lives outside the genome entirely — like the append-only property of the log. It is a property of the substrate, not a trait of the lineage. Evolution tunes `probe_rate_surplus` above the floor, never below.

```python
PROBE_RATE_FLOOR = 0.02   # constitutional: not in the genome, not selectable

def select_action(agent, context):
    probe_rate = PROBE_RATE_FLOOR + agent.genome.probe_rate_surplus  # surplus >= 0

    if random() < probe_rate:
        action = coverage_probe(agent.raw_signal_history, agent.action_vocab)
        tag = "PROBE"       # unconditionally sampled stratum
    else:
        mappings = retrieve(agent.encoder, agent.knowledge_base, context)
        action = agent.transformer(context, mappings)
        tag = "POLICY"      # encoder-conditioned stratum

    outcome = agent.environment.step(action)
    agent.log.append(Triple(context, action, outcome.delta_utility, tag=tag))
    return outcome
```

The 2% floor is a placeholder to be set empirically. The existence of a hardcoded floor is non-negotiable given the threat model. Asymmetry of error: a floor set too high taxes performance linearly; a floor at zero re-opens the exact hole being closed.

**Part 2: Probe-Gated Inheritance**

The inherited encoder earns influence only by predicting well on data the parental bias could not have steered. A burn-in period forces the child to act under an isotropic metric at elevated probe rate, producing a probe stratum whose state distribution the parental bias never touched. The inherited encoder is then evaluated against that stratum before earning influence.

```python
def initialize_offspring_encoder(child, parent_encoder, burn_in_cycles):
    child.encoder = isotropic_encoder()
    child.probe_rate_override = max(0.20, PROBE_RATE_FLOOR)
    run_cycles(child, burn_in_cycles)
    child.probe_rate_override = None

    probe_set = [t for t in child.log if t.tag == "PROBE"]

    err_inherited = prediction_error(parent_encoder, probe_set)
    err_isotropic = prediction_error(child.encoder, probe_set)

    # Smooth decay toward zero as inherited metric underperforms on unsteered data
    alpha = clip((err_isotropic - err_inherited) / err_isotropic, 0.0, 1.0)
    child.encoder = interpolate(child.encoder, parent_encoder, weight=alpha)
    child.inheritance_weight = alpha

    # Standing revalidation as probe stratum grows
    child.schedule_recurring(revalidate_inheritance, uses=probe_stratum_of(child))
```

The burn-in is what makes the gate honest. Without it, even probe triples are drawn from states reached by biased policy actions — probes are unconditional in action choice but not in state visitation. Cost: worse early performance. That cost is bounded and front-loaded; compounding bias is unbounded and back-loaded.

The gate uses smooth decay rather than a hard cliff: partial credit for partial validity. Standing revalidation means an inheritance that passed on a thin early sample can still be demoted later in life as the probe stratum grows.

**Part 3: Lineage Canalization Breaker**

A cross-generation monitor watches probe-set error across generations. Within-generation gates cannot see slow transgenerational compounding — only the cross-generation series can. When probe error trends upward while policy performance holds, the monitor forces an inheritance reset.

```python
def lineage_monitor(lineage):
    series = [g.final_probe_error for g in lineage.generations]

    if len(series) >= MIN_GENERATIONS and upward_trend(series, window=W):
        lineage.flag("CANALIZATION")
        # Surgical reset: keep meta-parameters and family priors, drop encoder weights
        lineage.next_generation.inherit_encoder = False
        lineage.next_generation.probe_rate_override = ELEVATED_RATE
```

The trend statistic is chosen for what bias cannot fake: policy-stratum performance can look excellent under a canalized metric, but probe-set error is computed on unsteered data. The reset is surgical: meta-parameters, thresholds, family priors, and probe surplus are kept — only encoder weights are dropped. The reset lineage retrains its encoder from its own log, validated on probe-tagged triples, restoring the retrain-from-log guarantee.

### Residual Hole

The breaker needs MIN_GENERATIONS of history before it can fire. A lineage can compound bias for several generations before reset — the defense bounds the damage window, it does not eliminate it.

Two mitigations, both with costs:
- **Shorter trend window** — more false resets, each discarding genuinely useful inherited structure
- **Cross-lineage comparison** — flag a lineage whose probe error is anomalous against the population; detects faster but assumes shared environment. In heterogeneous environments, this comparison needs stratifying by environment type. A false-reset epidemic is the failure mode if stratification is wrong.

This design decision should be made before deployment, not discovered as a false-reset epidemic later.

### Status

**Resolved with named residual.** The three-part defense closes the transgenerational compounding problem. The damage window before the canalization breaker fires is bounded but nonzero. Cross-lineage comparison is the recommended extension if the population runs in a shared environment.

---

## Finding 4: The Lamarckian Objection — Fully Resolved

### The Objection

The reviewer flagged topology bias inheritance as potentially Lamarckian — the discredited idea that characteristics acquired during a lifetime are transmitted to the next generation. The specific concern: if a parent agent discovers that signal X is useful during its lifetime, and that discovery biases the next generation toward signal X, isn't that just Lamarckism with extra steps?

### The Within-Lifetime Answer

The distinction holds at the within-lifetime level:

- Classical Lamarckism transmits acquired **characteristics** directly — the parent developed X, the next generation has X.
- This architecture transmits a **prior over the search space** — the parent found X worth the computational cost, the next generation starts with a higher prior probability of exploring near X.

The next generation earns every signal through its own lifetime experience. If the environment has shifted, a signal the parent found useful may not activate, and the prior fades through non-activation. The genetic algorithm still selects; it just starts from a better-shaped distribution. The analog is **inherited search bias**, not inherited characteristics.

### The Transgenerational Answer

Fable identified the transgenerational version as the strongest form of the objection: a biased parental metric steers offspring data collection from birth, compounding across generations in a way the within-lifetime probe budget alone cannot catch.

The three-part defense (Finding 3) answers this directly:

- The constitutional floor ensures the audit stratum cannot be selected away
- Probe-gated inheritance ensures the inherited metric must earn influence on data the parental bias never touched
- The lineage canalization breaker catches slow compounding that within-generation gates cannot see

**The Lamarckian objection survives only if the probe rate is allowed to go to zero under selection pressure.** With the constitutional floor and probe-gated inheritance in place, it does not hold.

### Status

**Resolved.** Contingent on the three-part defense in Finding 3 being implemented.

---

## Summary of Architecture Changes

| Component | Previous State | Current State |
|-----------|---------------|---------------|
| Contrastive encoder basis | Unspecified — implicit unconstrained neural map | Design decision required: linear, two-stage featurized (recommended default), or unconstrained with measured gap |
| Encoder audit practice | "Check whether weights make causal sense" | Differential anomaly detection against encoder's own history; audit the whole composed metric |
| Log representativeness | Protected only by immutability | Protected by immutability (integrity) + encoder-independent probe budget (representativeness) |
| Probe rate | Listed as inherited meta-parameter | Reclassified as architectural primitive; constitutional floor outside genome, not selectable |
| Inheritance mechanism | Probe rate floor only | Three-part defense: constitutional floor + probe-gated inheritance + lineage canalization breaker |
| Lamarckian objection | Partially answered | Fully resolved contingent on three-part defense implementation |

---

## Open Issues

### Encoder Design Decision (Finding 1)
The linear vs. two-stage featurized vs. unconstrained choice must be resolved by shadow-mode measurement. Recommended approach: deploy two-stage featurized as default, run linear in shadow mode, measure held-out prediction gap. If gap is small, simplify to linear. If large, the gap is a finding about environment structure worth documenting independently.

### Probe Rate Floor Value
The constitutional floor is set by the architect, not by evolution. Current placeholder: 2%. The right value is environment-dependent — set conservatively high early (accepting performance cost for audit stratum richness). The floor should be set based on the minimum audit stratum size needed to detect encoder bias within a reasonable number of generations.

### Burn-In Duration
The probe-gated inheritance requires a burn-in period during which the child acts under an isotropic metric at elevated probe rate. The right duration is the minimum needed to produce a probe stratum whose state distribution the parental bias never touched. Too short: the gate is leaky. Too long: excessive early performance cost. An empirical question to be resolved during implementation.

### Canalization Breaker Threshold and Cross-Lineage Comparison
The lineage canalization breaker needs MIN_GENERATIONS of history before it can fire — the damage window is bounded but nonzero. Two open decisions:

1. **Trend window length** — shorter windows reduce the damage window but increase false resets, each discarding genuinely useful inherited structure
2. **Cross-lineage comparison** — flagging a lineage whose probe error is anomalous against the population detects canalization faster but assumes a shared environment. In heterogeneous environments, stratification by environment type is required or false resets become an epidemic

Both decisions should be made before deployment.

### Remaining Ambiguous Roadmap Steps
Seven steps from the dependency graph document remain design-dependent (Steps 7, 15, 18, 22, 26, 27, and the split of 29). The log-shaping failure mode (Finding 2) has implications for Steps 9 and 11: both are inside the encoder-conditioned loop, and their effectiveness as exploration mechanisms depends on the probe budget providing the off-distribution coverage they cannot generate themselves.

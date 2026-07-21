# Observability and Recovery
## Implementation Reference for the Causal Receptor Architecture

---

This document covers the diagnostic stack and recovery protocol for the three coupled learning processes in the architecture: the contrastive encoder, action family structure, and certainty scores. It is an implementation reference, not part of the core conceptual framework.

---

## The Coupling Problem

The three learned components make redundant predictions about the same underlying structure — which context dimensions matter for which actions — and their errors can feed each other:

- A bad encoder mis-conditions retrieval → certainty scores accumulate from wrong contexts
- Corrupted certainty scores bias the outcome profiles that family clustering uses
- Wrong family assignments entrench the bad encoder metric

Each loop individually self-corrects. Whether the coupled system does is an empirical question. The architecture needs a diagnostic for it.

**The coupling also helps.** Because the three processes make redundant predictions about the same latent structure, disagreement between them is a leading indicator of failure — and the raw experience log provides ground truth against which each component can be tested independently.

---

## Early Warning Signals

Listed in order of earliness — each precedes the next in the failure cascade.

### 1. Cross-Metric Rank Correlation

The encoder and action family reweightings are two independent estimates of the same latent fact: which context dimensions matter for which actions. When both are healthy, they should agree — contexts the encoder places close together should be contexts where the family's reweighted metric also sees small distance, for actions in that family.

**Implementation:** Sample context pairs; compute their similarity under the shared encoder and under each family's Mahalanobis reweighting. Track rank correlation per family over time.

**Signal:** Correlation drifting upward or stabilizing high = healthy convergence. Falling correlation = encoder and family structure diverging. One of them is moving away from the truth. This precedes corruption of certainty scores, because it is the divergence that causes mis-conditioning downstream.

### 2. Retrieval Churn

For a fixed probe set of `(context, action)` queries, track how much the retrieved neighbor set changes per unit time.

**Signal:** Healthy learning shows churn decaying as structure settles. Churn that reaccelerates without an environment shift is the oscillatory signature of coupled processes destabilizing — each process chasing the other's moving target. This is visible before global prediction error moves, because prediction error only degrades once churned retrievals have polluted enough certainty updates.

### 3. Certainty–Accuracy Calibration

Bin mappings by certainty score; check whether high-certainty rows actually predict better than low-certainty rows on their next observations.

**Signal:** A healthy system's certainty is calibrated — high certainty means high accuracy. The entrenchment failure mode has a specific signature: certainty keeps climbing while conditional accuracy does not. Confidence accumulates from consistent mis-conditioned retrieval rather than from a consistent world. Miscalibration onset leads global held-out error stalling, because the still-healthy mappings mask the corrupted ones in the aggregate.

This signal is also the most likely to catch the slow-drift failure case (see below).

### 4. Held-Out Prediction Error (Lagging)

Track held-out `(context, action) → Δreceptor` prediction error as the global diagnostic. This is the one measure all three processes jointly serve — monotonic improvement means the coupled system is healthy; stalling means something is wrong.

**Limitation:** This is a lagging indicator. By the time it stalls, a bad equilibrium may already be entrenched. Use as confirmation, not as primary warning.

---

## Attribution

When a warning signal fires, the goal is to identify the upstream failing component before the fault propagates further.

### Default Suspect Ordering

The three processes operate on different timescales:
- Certainty scores update every cycle (fastest)
- Encoder updates over many cycles
- Family structure updates over many splits (slowest)

Slow processes contaminate downward; fast processes mostly don't contaminate up. Therefore:

- **Sudden degradation:** suspect encoder first → families → certainty scores
- **Slow drift:** suspect certainty scores first → families → encoder

### Attribution Tests Against the Raw Log

Each component can be tested against data that did not pass through the others:

**Test the encoder alone:** Compute contrastive loss on raw `(context, action, Δreceptor)` triples directly from the experience log — outcome-similar context pairs, no retrieval or certainty scores involved. If the encoder separates them well, it is not the source of the failure.

**Test family structure alone:** Check whether within-family actions actually share outcome-response profiles in the raw triples, bypassing retrieval. If family assignments correctly group actions with similar context-sensitivity, family structure is not the source.

**If both pass but the composed system fails:** The fault is in the coordination layer — the gating threshold, the update-conditioning logic, or the interface between retrieval and updating. This layer fails at least as often as the components themselves and should be listed as a fourth suspect.

---

## The Genuine Failure Case

If the system entered a bad equilibrium **slowly**, all three components may individually pass their raw-log tests — each is a locally reasonable summary of data collected under a jointly bad policy. Attribution fails because the fault is in the data distribution, not any component.

**Fix:** Not a reset, but exploration pressure. Intrinsic exploration signals must collect off-equilibrium triples — the system needs to act in regions it has been avoiding, see what actually happens, and update from that. The calibration signal (certainty vs. accuracy divergence) is the one most likely to detect this case before it becomes intractable.

---

## Intervention Protocol

When attribution has identified the failing component:

1. **Freeze downstream processes.** Stop certainty updates; pin family assignments. The fault stops propagating while you fix the upstream source.

2. **Retrain the failing component from the raw log.** Because the log is never overwritten, the ground truth is always available. Retraining uses the uncorrupted record of what the organism actually did and what happened.

3. **Selectively recompute downstream state.** Not all derived state is corrupted — only the parts that were conditioned on the bad component.
   - Certainty scores: rows whose conditioning contexts were retrieved under the bad metric get recomputed from the log. Rows untouched by the faulty retrieval regime keep their accumulated certainty.
   - Family assignments: if only a subset of families was affected, recompute those clusters while freezing the rest.

4. **Resume normal operation** and monitor whether the warning signals return to healthy baselines.

### The Reset Dilemma Is Dissolved

The append-only log means there is no genuine dilemma between "lose one component's structure" and "lose everything." Every component's structure is derivable from the log. Resetting any component costs compute, not experience. The organism's history of having acted in the world is always intact.

---

## Checkpointing

Log-recompute costs grow with lifetime. In practice:

- Checkpoint all derived state (encoder weights, family assignments, certainty scores, context embeddings) at regular intervals
- Tag each checkpoint with the log position it was derived from
- Under intervention, replay from the last checkpoint that passes the attribution tests rather than from the beginning of the log

Standard event-sourcing hygiene. Build it in from the start; retrofitting is expensive.

---

## Inherited Diagnostics

Both encoder weights and family structure are inheritable across generations. The diagnostic baselines — what counts as healthy rank correlation, acceptable churn rate, well-calibrated certainty — will vary by environment and should themselves be inherited as priors, tightened by each generation's experience.

The similarity threshold (the bias-variance dial governing update gating) is a meta-parameter stable enough across environments to be a candidate for evolutionary selection. Offspring whose inherited threshold is well-matched to their environment will accumulate useful certainty faster than those starting from scratch.

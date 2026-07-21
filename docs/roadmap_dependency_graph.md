# Roadmap Dependency Graph
## From Numbered Sequence to Structured Graph

---

## The Taxonomy

The 30-step roadmap is not a sequence. It is a dependency graph whose nodes sort into four categories. The numbered order is a linearization of that graph — useful for reading, misleading for building.

### Scheduled Infrastructure
Substrate that either exists or doesn't. Nothing happens on deploy day. Deploy dates are set by the roadmap.

### Gated Promotions
Capabilities earned against a measurement the diagnostic stack already produces. The infrastructure half deploys on a date; the capability half earns influence when a specific metric crosses a threshold. Many roadmap steps are bundles containing both halves — splitting them is the primary value of this taxonomy.

### Detected Emergents
Behaviors that should fall out of machinery already in place. Nothing to build, nothing to promote. The roadmap's job is to notice them. Their *absence* is a falsification signal about the underlying design. Building a module for an expected emergent destroys the evidence that the architecture works.

### Inherited Meta-Parameters
Stable dials the evolutionary layer selects across generations rather than capabilities deployed or earned within a lifetime. These are not infrastructure knobs (they aren't set once) and not promotions (no threshold crossing). They are search targets for the inheritance mechanism.

---

## The Four-Category Sort

### Scheduled Infrastructure

| Step | Name | Notes |
|------|------|-------|
| #2 | Metabolic economy | The compute-cost signal channel. Load-bearing substrate: the mechanism by which every other capability "justifies its cost." |
| #4 | Hierarchical nervous system | Pure architectural restructuring. Fast and slow pathway split. Foundational substrate many later steps hang off. |
| #19 | Sleep and consolidation | The offline recompute/checkpoint-replay phase. The phase is built; the benefit it produces is a separate promotion. |
| #20 | Social utility signals | Signal channels plus the multi-agent environment they presume. Nothing social happens on deploy day. |
| **Infrastructure halves of bundles** | | |
| #1 (half) | Sensory enrichment | Receptor wiring and sensor channels. The policy that uses them is a promotion. |
| #8 (half) | Distance sensing | The sensor channel. A new receptor channel is hardware — it ships. |
| #10 (half) | Prediction / mental model | Schema separation, append-only log, accuracy signal channel. |
| #11 (half) | Curiosity and confidence regulation | The signal channels that make curiosity and confidence exist as utility streams. |
| #12 (half) | Forward simulation | The rollout machinery — a mechanism for running the transformer against the mental model without touching actuators. |
| #16a | Attention (budget allocator) | The retrieval/evaluation allocator that takes a compute budget from #2 and applies an explicit priority function. Deploy-day priority function is top-k cosine — identical to current implicit behavior, changed nothing, risks nothing. |
| #23 (half) | Communication | An actuator that emits signals plus a receptor that receives them. |
| **6b** | Spatial index | See Demand-Triggered Infrastructure below. |

### Gated Promotions

| Step | Name | Gate |
|------|------|------|
| #1 (half) | Sensory enrichment (capability) | Sensor readings earn weight in retrieval and action selection when mappings conditioned on them beat mappings that ignore them on held-out prediction. Shadow-mode from day one. |
| #5 | Temporal association | Delayed-prediction accuracy beats an immediate-only baseline. The delay-distribution field is schema (infra); forming correct delayed links is the promotion. |
| #8 (half) | Distance sensing (capability) | Distance readings earn influence when mappings conditioned on distance beat those ignoring it on held-out prediction. |
| #10 (half) | Prediction (capability) | Agent is "predictive" when certainty-weighted predicted Δutility beats a running-average baseline. Nearly free from the calibration diagnostic. |
| #11 (half) | Confidence regulation (capability) | Earns arbitration weight only when certainty scores are calibrated. **The most dangerous premature promotion in the sequence** — promoting an uncalibrated confidence signal into action arbitration makes the system act on its own miscalibration. |
| #12 (half) | Forward simulation (capability) | Earns authority *per region* where simulated outcomes match realized outcomes. Global promotion on a global average is the entrenchment risk at the planning layer. |
| #13 | Self-model | Self-referential mappings predict the agent's own subsequent signal states better than chance. Almost pure earned capability over existing schema — may not deserve to be a deployment step at all. |
| #14 (half) | Reflex updating (capability) | The write-path from mental model to reflex layer is infra; reflexes *actually improving* is the gate. |
| #16b | Attention (stakes-sensitive) | Stakes-and-state-sensitive prioritization earns influence when it beats top-k cosine on utility-weighted held-out prediction error at fixed budget. Edges in: #2 (budget), #11 (intrinsic-state input). Edges out: #17, #18 (negative-utility salience explains their speed under compute pressure). |
| #21 | Other-model | Exact analog of self-model, aimed at another agent. Gated on predicting their states. |
| #23 (half) | Communication (capability) | Communication exists when a receiver's database holds high-certainty mappings from another agent's emissions to subsequent events. **Directly readable from the knowledge base** — one of the cleanest gates in the sequence. |
| #24 | Compression | Abstracted mappings retain predictive accuracy at lower storage/retrieval cost. Likely shares machinery with #19. |
| #28 | Internal conflict model | Self-model applied to the conflict signal. Gated like other-model. |
| #29 (half) | Concepts and words (compression capability) | Compression capability gated on retained accuracy at lower cost. Distinct from the emergence of stable discrete concepts (see Detected Emergents). |
| #30 | Grounded language | The endpoint. Communication gate (#23) plus the grounding guarantee — emitted signals whose chains terminate in receptor states. |

### Detected Emergents

| Step | Name | Detection Criterion | Absence Means |
|------|------|--------------------|-|
| #3 (half) | Homeostatic regulation (behavior) | The drift signal channels are infra; the agent *learning to service them* is emergent over the metabolic and receptor system. | Metabolic economy or receptor update dynamics are broken. |
| #6a | Spatial conditioning | Directly inspectable: read the learned encoder reweighting and check whether positional dimensions carry weight for relevant action families. Bad-outcome mappings should cluster by position and the metric should reflect it. | Encoder is failing to pick up positional predictive structure — likely an embedding geometry problem. |
| #9 | Active broadband interrogation | Fraction of actions selected primarily for expected certainty gain crosses a threshold; or action distribution measurably shifts toward contexts where retrieved mappings have high variance. | Curiosity signal or mental model is not functioning as intended. Should not be a positioned step — becomes a standing detector that lights up whenever curiosity (#11) and mental model (#10) are both live. |
| #17 (half) | Anticipatory avoidance (behavior) | Agent acts to avoid predicted negative utility *before* it lands, over and above what reflex or direct retrieval would produce. | Prediction (#10) or forward simulation (#12) is not functioning. |
| #29 (half) | Stable concepts (emergence) | Discrete stable concept-units actually appear in the compression layer — not just compression happening, but stable reusable structures. | Environment may not support abstraction at the current level of complexity, or compression machinery is not finding recurring structure. An interesting falsification signal. |

### Inherited Meta-Parameters

These are stable dials the evolutionary inheritance layer optimizes across generations. Not deployed, not promoted. Wrong category if filed as infrastructure or gated promotions.

| Step | Name | Notes |
|------|------|-------|
| #25 | Optimism and goal persistence | A tuned bias/discount parameter, or emergent from confidence + forward simulation sustaining pursuit across delay. Either way, more plausibly an evolutionary-selected meta-parameter than a deployable step. The inheritance layer should be optimizing it rather than the roadmap deploying it. |
| — | Similarity threshold | The bias-variance dial governing update gating — wide early, tightens as contrastive signal accumulates. Stable across environments; candidate for inheritance. |
| — | Split-trigger sensitivity | The threshold at which action families split. Stable meta-parameter; inheritance layer should select over it. |

---

## Demand-Triggered Infrastructure

A variant of scheduled infrastructure whose deploy date is set by a detection criterion rather than a roadmap position. The substrate is built, not earned, but its necessity is environment-dependent.

**Step 6b — Spatial index**

The context vector already includes position coordinates. Spatial conditioning (6a) — "does the current position resemble positions where bad things happened?" — falls out of the contrastive encoder as a detected emergent. No new mechanism needed.

But the encoder actively destroys geometric structure. Its contrastive objective rewards pulling together contexts with similar outcome distributions. Two spatially distant places that are outcome-equivalent get aliased into one embedding region. That's correct for prediction and catastrophic for navigation. The encoder isn't failing to learn geometry — its objective *rewards discarding* geometry wherever outcome-equivalence crosses spatial distance.

Any capability needing metric or compositional queries — what's within radius r, what's the route from here to there, A-connects-to-B therefore a shortcut exists — requires a dedicated spatial index.

**Deploy trigger:** Aliasing errors exceeding threshold — prediction failures concentrated where outcome-similar contexts are spatially distinct. Coordinate-far but embedding-close neighbors producing high residual error. This comparison is cheap: both quantities are already logged.

**Deploy date:** Pulled into existence by #12 (forward simulation) rolling out trajectories through space. Before that, nothing queries geometry.

**Retroactive build:** The append-only log includes position with every triple. When the index ships, it can be built retroactively over the agent's entire spatial history — another view over the log, no early experience wasted by deferring it.

**Edges in:** #8 (distance readings as input), #12 (failure modes demand the index).
**Edges out:** #12 (improves trajectory rollout), #17 (route-based anticipatory avoidance).

**GPS caveat:** Ground-truth position in the context vector is a simulation convenience. If any later roadmap phase moves toward inferred position from sensory flow, 6b changes character entirely — becoming a SLAM-like estimation problem where the aliasing issue reappears at the perceptual level. Abstract the position source from day one; cheap now, painful to retrofit.

---

## Ambiguous Cases — Design Decisions Required

These steps cannot be categorized without a design decision not yet made. Each decision propagates into the dependency structure.

### Step 6 — Spatial Memory ✓ RESOLVED
See above. Split into 6a (detected emergent) and 6b (demand-triggered infrastructure).

### Step 7 — Habituation and Sensitization
**The fork:** Habituation looks like it could fall out for free from certainty saturation — repeated observation → high certainty → reduced novelty/exploration response. If so, detected emergent. If you want the classic asymmetric habituation/sensitization dynamics specifically, that's a built response-modulation rule (infrastructure).

**Decision needed:** Are these emergent from existing update dynamics, or a deliberately shaped rule?

### Step 15 — Multiple Hypothesis Tracking
**The observation:** The schema already stores multiple competing mappings per context with certainty scores. "Tracking multiple hypotheses" may already be present. The only genuinely new capability is *not collapsing prematurely* to the top hypothesis — which is a detected behavior (does the agent keep acting to disambiguate rival high-variance mappings?), not a new mechanism.

**Decision needed:** Is this a step at all, or a property to verify? Possibly should be deleted and replaced with a detection criterion on the existing architecture.

### Step 16 — Attention ✓ RESOLVED
Split into 16a (scheduled infrastructure: budget allocator) and 16b (gated promotion: stakes-sensitive prioritization). The original dimension-weighting reading is redundant with the contrastive encoder and should be deleted. See Gated Promotions above.

### Step 18 — Adaptive Defense Response
**The fork:** Built reflex (infrastructure) or emergent from forward simulation over negative-utility predictions? Plausibly both: a built reflex substrate that reflex-updating (#14) later refines.

**Decision needed:** Is the initial defense response a hardwired reflex (fast pathway, infrastructure) or a learned one? Hardwired is probably correct for "defense" by definition — fast pathway exists precisely for this. Learned refinement via #14 is then a gated promotion on top.

### Step 22 — Deception Detection
**The fork:** Emergent over other-model + communication (fires when another agent's emissions systematically mispredict *and that mismatch itself becomes predictable*) or an explicit mismatch-detector you build?

**Decision needed:** Emergent is more elegant and more falsifiable. Explicit detector is the fallback if the emergent version doesn't appear. Emergent preferred unless the environment requires it earlier than other-model + communication can support it.

### Step 26 — Conflict Signal
**The fork:** A built signal channel (infrastructure) or a detected condition — high variance among candidate actuator activations, readable from the existing action-evaluation step?

**Decision needed:** If detected, it's nearly free and emergent, and #28 (internal conflict model) sits on a detected quantity. If built, it's infrastructure. The detected reading is cheaper and more consistent with the architecture's general principle; built is appropriate if the detected version doesn't fire reliably enough for #28 to depend on.

### Step 27 — Context-Conditioned Arbitration
**The observation:** The architecture is context-conditioned by construction — the entire context-vector retrieval path does this. Either #27 is already present and shouldn't be a step, or it denotes a specific upgrade.

**Most likely reading:** Step 27 *is* the action-family splitting promotion — the point at which different action families learn different arbitration weights for different context regions. If so, it should be renamed and gated on the family split-trigger statistic.

**Decision needed:** Is #27 redundant (delete it) or is it the family-splitting promotion wearing a different name (rename and gate it)?

### Step 29 — Concepts and Words
**The split:** The compression capability is a gated promotion (accuracy retained at lower cost). The emergence of stable discrete concepts is a detected emergent. These are two different things currently bundled in one step.

**Decision needed:** Split into 29a (gated promotion: compression capability) and 29b (detected emergent: stable concept-units appear). Already partially resolved in the sort above.

---

## Dependency Edges

Prerequisites only — not sequence. A step can deploy whenever its prerequisites are met, regardless of number.

```
#2  → #16a (budget source for attention allocator)
#2  → evolutionary layer generally (cost justification)

#4  → #14 (reflex write-path needs hierarchical structure)
#4  → #16a (fast/slow pathway is the scaffold attention builds on)
#4  → #18 (fast pathway is the defense reflex substrate)

#5  → #10 (can't predict before delayed links exist)

#8  → #6b (distance readings as spatial index input)

#10 → #11 (curiosity needs the mental model to query)
#10 → #12 (forward simulation queries the mental model)
#10 → #13 (self-model rows live in the mental model schema)
#10 → #15 (multiple hypotheses are competing rows)
#10 → #17 (anticipation requires prediction)

#11 (curiosity) + #10 → #9 (active interrogation emerges here)
#11 → #16b (intrinsic-state input to attention priority function)

#12 → #6b (trajectory rollout demands spatial index when aliasing errors appear)
#12 → #17 (route-based anticipatory avoidance)
#12 → #18 (learned defense refinement via forward simulation)
#12 → #25 (optimism: sustaining pursuit across delay via simulation)

#13 + #26 → #28 (internal conflict model: self-model applied to conflict signal)

#16b → #17 (negative-utility salience surfaces threat mappings under budget)
#16b → #18 (same mechanism, defense speed)

#19 → #24 (consolidation and compression likely share offline machinery)

#20 (multi-agent environment) → #21 (other-model needs others to model)
#21 → #22 (deception detection needs other-model)

#23 + #24 → #29 (concepts and words: communication + compression)
#29 → #30 (grounded language: concepts plus grounding guarantee)
```

---

## Ordering Contradictions in the Numbered Sequence

Two places where the numbered order contradicts the dependency graph:

**Step 9 before Steps 10 and 11:** Active broadband interrogation is listed before prediction and curiosity, but depends on both. As a detected emergent, it cannot be observed until its prerequisites are promoted. Resolution: remove #9 from the numbered sequence entirely and make it a standing detector that activates whenever #10 and #11 are both live.

**Step 6 before Step 8:** Spatial memory is listed before distance sensing, yet distance readings are a natural input to the spatial index (6b). The 6a branch (detected emergent via encoder) has no dependency on #8 and the ordering is harmless. The 6b branch (spatial index) wants #8 first. Resolution: 6a can stay where it is; 6b deploys after #8 and is demanded by #12.

---

## Rough Tally

| Category | Count |
|----------|-------|
| Scheduled infrastructure (including infra halves of bundles) | ~12 |
| Gated promotions (including promotion halves of bundles) | ~13 |
| Detected emergents | ~5 |
| Inherited meta-parameters | ~3 |
| Demand-triggered infrastructure | 1 (6b) |
| Ambiguous / design-dependent | ~7 |

The numbered sequence of 30 steps is approximately 12 deployments, 13 promotions, 5 detections, 3 inherited dials, and 7 design decisions — wearing a sequence costume.

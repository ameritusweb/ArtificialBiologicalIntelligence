# Theories

## Index of theoretical claims in this research program

Each entry states the claim as a proposition, its evidence status, and where the full argument lives. Claims are grouped by theme but numbered globally for reference.

Evidence status key:
- **Proposed** — stated and falsifiable but untested
- **Partially tested** — some evidence, not conclusive
- **Supported** — experimentally demonstrated in ERTI
- **Revised** — tested and corrected
- **Falsified** — tested and rejected

---

## I. The Starting-Point Critique

**T1. Current AI inverts phylogeny.** Language sits atop hundreds of millions of years of sensorimotor infrastructure. LLMs learn words from beings who had the underlying states without having the states themselves. Starting from language rather than sensation is a fundamental limitation, not a training gap.
*Status: Proposed. Argued: whitepaper Section 1, ERTI_roadmap.*

**T2. Grounding failure is architectural.** The inability to ground LLM concepts is not a data, training, or scaling problem — it is a starting-point problem. You cannot ground a concept that was learned statistically.
*Status: Proposed. Argued: whitepaper Section 1.*

---

## II. The Receptor Topology Thesis

**T3. Capability without receptor is latent.** A capability that has no receptor to trigger it never gets used. The receptor is why the capability exists. Motivation and cognition are the same operation viewed from different angles.
*Status: Proposed. Argued: whitepaper Section 4, genome overview.*

**T4. Intelligence is adequacy to environmental complexity.** Intelligence is not designed; it is selected. The environment decides what's worth sensing; the organism discovers how.
*Status: Proposed. Argued: genome overview, whitepaper Section 1.*

**T5. The input structure itself should evolve.** Rather than specifying what to sense (pixels, tokens), organisms should develop receptors for environmental structure that pays for itself. Topology is the unit of evolution.
*Status: Partially tested. 32/46 receptors discovered; topology inheritance accelerates convergence 15-to-0 epochs. Argued: genome overview, whitepaper Section 11.*

**T6. A receptor is any selected internal state whose activation changes behavior and whose consequences affect whether that state persists.** This definition encompasses pain, curiosity, compression, and conflict without claiming biological identity.
*Status: Proposed (definitional). Argued: ERTI_roadmap.*

**T7. Agency is a continuous variable** determined by receptor, effector, and processing complexity — not a mysterious binary property.
*Status: Proposed. Argued: whitepaper Section 4.*

---

## III. The Serialization Thesis

**T8. Sequential processing is an optimization, not a bottleneck.** Temporal decomposition of simultaneously-available information creates prediction opportunities that parallel processing destroys. Each stage generates an expectation about what the next stage will reveal; the delta is where learning happens.
*Status: Partially tested. ERTI fast/slow pathway is an instance. Argued: serialization_thesis.md Sections 1, 3.*

**T9. Prediction surface scales with stage count.** An n-stage processor generates n expectation-outcome pairs per input. A single-stage parallel processor generates one.
*Status: Proposed. Argued: serialization_thesis.md Section 3.1.*

**T10. Serialization ratchets complexity.** Environmental complexity selects for prediction machinery, which creates more prediction surfaces, which supports richer receptor topologies, which exploits more complex environments.
*Status: Proposed. Argued: serialization_thesis.md Section 3.3.*

**T11. Organisms manufacture prediction error.** They do not just minimize the prediction errors they receive — they engineer their processing architecture to generate more of them. This is the strongest departure from Friston.
*Status: Proposed. Argued: serialization_thesis.md Section 7.*

---

## IV. The Gibson/Enactivism Fork

**T12. Reaction does not require internal models. Prediction does.** Gibson and enactivism correctly describe embodied sensorimotor coupling but cannot explain anticipation, surprise, or learning from expectation violation. These require models that generate expectations before signals arrive.
*Status: Proposed. Argued: serialization_thesis.md Section 2.*

**T13. Affordances are present-tense.** Direct perception tells you what is the case, not what will be. Sensorimotor coupling is a reactive conditional, not a generative prediction. Prediction requires the temporal gap that staging provides.
*Status: Proposed. Argued: serialization_thesis.md Section 2.3.*

---

## V. Per-Receptor Pipeline Architecture

**T14. Every receptor family has its own evolved temporal decomposition strategy.** Pain processes coarse-to-fine-to-contextual. Curiosity processes novelty-to-relevance-to-strategy. The pipeline structure is optimized for each domain's specific prediction structure.
*Status: Proposed. Argued: serialization_thesis.md Section 4, sequential_processing.yaml, perception.yaml.*

**T15. Prediction error minimization is domain-specific, not global.** Contra Friston's single variational objective, prediction operates through heterogeneous temporal decomposition strategies evolved under different selection pressures.
*Status: Proposed. Argued: serialization_thesis.md Section 7.*

**T16. Processing latency should correlate with prediction depth, not computational complexity.** A receptor with 5-stage prediction structure should process more slowly than one with 2 stages, regardless of computational demand. This distinguishes the serialization account from the bottleneck account.
*Status: Proposed (falsifiable). Argued: serialization_thesis.md Section 4.3.*

**T17. The brain's "inefficiencies" are prediction infrastructure.** Slow/fast pathways, distributed processing, hierarchical organization, loops, recurrence — every apparent inefficiency is a prediction opportunity.
*Status: Proposed. Argued: sequential_processing.yaml, perception.yaml.*

---

## VI. Cross-Pipeline Prediction and Binding

**T18. Binding is mutual prediction, not convergence.** Each receptor pipeline generates lateral predictions about what other pipelines will find. The web of mutual prediction IS the integration. No central convergence zone required.
*Status: Proposed. Argued: serialization_thesis.md Section 5.*

**T19. Binding is the absence of cross-pipeline prediction error.** When all pipelines' lateral predictions confirm each other, the percept is unified. When they fail, attention fractures to the violated prediction.
*Status: Proposed. Argued: serialization_thesis.md Section 5.2.*

**T20. Consciousness is recursive mutual prediction between pipelines.** "What it is like" to perceive something is the specific pattern of cross-pipeline predictions — confirmed and violated — generated by that percept.
*Status: Proposed (speculative, three falsification criteria in serialization_thesis.md Section 5.3).*

---

## VII. The Anxiety Derivation

**T21. Anxiety is mechanically predicted by the architecture.** Any system with sequential prediction, a processing-speed receptor, arousal as a response to poor model fit, and arousal's effect on prediction accuracy will exhibit a self-amplifying loop. This is derived, not accommodated post-hoc.
*Status: Proposed. Argued: serialization_thesis.md Section 6.*

**T22. Anxiety is not a malfunction.** Every receptor in the loop reads a real signal and produces the correct response. The pathology is in the loop structure, not in any individual receptor.
*Status: Proposed. Argued: serialization_thesis.md Section 6.1, perception.yaml.*

**T23. Self-damping requires a four-family dependency chain:** metacognition + processing-speed + arousal regulation + conflict detection. CBT trains this receptor. Anxiolytics achieve it pharmacologically.
*Status: Proposed (falsifiable: if self-damping emerges from simpler architecture, the dependency is wrong). Argued: serialization_thesis.md Section 6.2, perception.yaml.*

---

## VIII. Grounding and Architecture

**T24. Grounding is structural, not trained.** In a system where all mental model chains terminate in receptor states, grounding is guaranteed architecturally. No free-floating cognition is possible.
*Status: Partially tested. Grounding dictionary correctly returns false for ungrounded concepts. Argued: whitepaper Section 9.*

**T25. The three-way separation (transformer / mental model / experience log) dissolves grounding, compartmentalization, legibility, unlearning, and safe failure as problems.** These are structural properties of the architecture, not capabilities that need to be trained.
*Status: Partially tested. Argued: whitepaper Section 5.*

**T26. The mental model's value is at training time, not inference time.** The mm_features channels are redundant with what the policy internalized, slightly degrading inference performance. The +223% cultural transmission claim was retracted.
*Status: **Revised** (supported by three controlled decomposition experiments). Argued: FINDINGS.md Section 8.*

---

## IX. Evolutionary Dynamics

**T27. Complexity reshapes topology; it does not expand it.** The number of discovered receptors is roughly constant across tiers (~31-35), but which receptors emerge changes.
*Status: **Supported** (measured across 8 tiers). Argued: FINDINGS.md Section 1.*

**T28. 29 trunk receptors are invariant across all environments.**
*Status: **Supported** (measured). Argued: FINDINGS.md Section 2.*

**T29. Social environments universally benefit from transfer (11-25x); instrumental environments resist it.** Transfer is asymmetric: upward works, downward doesn't.
*Status: **Supported** (5x5 transfer matrix). Argued: FINDINGS.md Section 3.*

**T30. Topology inheritance accelerates convergence and surfaces novel receptors** that unbiased discovery misses.
*Status: **Supported** (5-generation experiment). Argued: FINDINGS.md Section 5.*

**T31. Behavioral prediction emerges in social arms races but not in static environments.** Multi-organism competition creates prediction pressure that single-organism training cannot.
*Status: **Supported** (population evolution experiment). Argued: FINDINGS.md Section 4.*

**T32. An organism's receptor topology is a fossil record of its ancestral environment.** The topology's shape — which families are well-developed — is informative about selection history.
*Status: Partially tested. Tier-specific receptors reveal environment-specific selection. Argued: whitepaper Section 7.*

**T33. Cross-family receptors are most vulnerable to canalization.** If any dependency is incorrectly entrenched, the composite receptor inherits and amplifies the error.
*Status: Partially tested (no canalization detected over 5 generations). Argued: cross_family_dependencies.md.*

---

## X. Dependency Structure of Cognition

**T34. Cognitive receptor evolution follows a DAG with dependency layers.** Trunk receptors (single-family) emerge first; cross-family compositions emerge later; deep canopy receptors (4+ family) emerge last.
*Status: Partially tested. Argued: cross_family_dependencies.md.*

**T35. Social cognition is the integration bottleneck.** Many canopy receptors across multiple families require theory of mind as a prerequisite. Social cognition emergence should trigger a cascade of canopy emergence.
*Status: Proposed. Argued: cross_family_dependencies.md, COMPLETE.md.*

**T36. Regulatory is the most cross-cutting family.** Homeostasis uses whatever works — creating dependency paths into every other family.
*Status: Proposed. Argued: cross_family_dependencies.md.*

**T37. Emergence generation correlates with dependency depth, cross-family breadth, and required environmental complexity.**
*Status: Partially tested (metabolic cost correlates with depth and breadth). Argued: cross_family_dependencies.md, STATUS.md.*

---

## XI. Compression, Concepts, and Language

**T38. Abstraction and bias are the same operation.** The compression receptor rewards simplification regardless of what was discarded. "Fruit" (useful) and "those people" (harmful) differ only in whether discarded information was consequential.
*Status: Proposed. Argued: ERTI_roadmap Step 11.*

**T39. Words are socially stabilized compressions of embodied models.** A concept is a compression that is sufficiently accurate and shared. A word is a concept transmissible via communication receptors. Language both enables precise thought and systematically distorts it.
*Status: Partially tested (1,013 stable concepts measured). Argued: ERTI_roadmap Steps 22, 27.*

**T40. Concepts exist in the world's causal structure, not in the organism's head.** The organism distills them; it does not generate them from nothing.
*Status: Proposed. Argued: whitepaper Section 3.*

---

## XII. Social Cognition

**T41. Theory of mind is the self-model applied to others.** "What would I do in their state?" Once you can model yourself, modeling others is cheap.
*Status: Proposed. Argued: ERTI_roadmap Step 17, COMPLETE.md.*

**T42. Empathy is receptor propagation.** Another organism's distress behavior activates the observer's own pain receptors. Information cost is low; survival benefit is enormous.
*Status: Partially tested (empathy trends upward 0.482-0.555 across 5 generations). Argued: ERTI_roadmap Step 20, FINDINGS.md.*

---

## XIII. The Song Replay Hypothesis

**T43. Earworms are pattern-based homeostatic regulation.** Involuntary song replay is not random memory activation but a regulatory receptor mechanism: stored rhythmic patterns are retrieved to correct internal state.
*Status: Proposed. Argued: song_replay_hypothesis.md.*

**T44. The replayed song correlates with the dysregulated state,** not random: stress triggers slow-rhythm calming songs; low arousal triggers fast-rhythm energizing songs.
*Status: Proposed (falsifiable). Argued: song_replay_hypothesis.md.*

**T45. Song replay peaks at moderate dysregulation** (0.3-0.7 stress range), not at baseline or crisis.
*Status: Proposed (falsifiable, with numerical range). Argued: song_replay_hypothesis.md.*

**T46. Emotional regulation is not a separate module.** It is the standard receptor-retrieval-action loop applied to internal states.
*Status: Proposed. Argued: song_replay_hypothesis.md, COMPLETE.md.*

---

## XIV. Processing Speed as Information

**T47. Processing speed is itself a receptor.** Fast processing = good model-environment fit. Slow processing = poor fit. The speed differential is information about how well your receptor topology handles the current demands.
*Status: Proposed. Argued: perception.yaml, serialization_thesis.md Section 6.1.*

---

## XV. Optimism, Will, and Science

**T48. Optimism is a receptor for a hypothetical world model weighted toward positive receptor states.** Without it, the organism can only navigate away from pain, not toward goals.
*Status: Proposed. Argued: ERTI_roadmap Step 23.*

**T49. Will is optimism plus self-model.** Not reacting to the current field but acting to make a specific imagined future real.
*Status: Proposed. Argued: ERTI_roadmap Step 23.*

**T50. Science is not a cultural invention.** It is what happens when multiple-hypothesis-maintenance becomes fitness-positive under environments where premature commitment is fatal.
*Status: Proposed. Argued: ERTI_roadmap Step 12.*

---

## XVI. Natural Emergence of Serialization

**T51. Physical heterogeneity in processing speed creates timing differences that get exploited as prediction windows.** The magno/parvocellular split and fast/slow pain fibers are instances — evolution inherited timing differences and built prediction machinery around them.
*Status: Partially tested (consistent with known neuroscience). Argued: serialization_thesis.md Section 11.1.*

**T52. In artificial systems, heritable processing schedules over observation channel groups can produce the same outcome.** In time-critical environments, organisms should evolve serialized processing (k=1 to k=3-5) over 50-200 generations. In non-time-critical environments, serialization should not emerge.
*Status: Proposed (experiment designed but not in roadmap). Argued: serialization_thesis.md Section 11.3.*

---

## XVII. Rationalization, Read Policy, and Annealing

**T54. Rationalization is read policy corruption, not memory corruption.** An organism with a pristine append-only experience log can still rationalize at the read layer — downweighting conflict-flagged entries before they are structurally resolved. The three-way separation (transformer / mental model / experience log) provides auditability of rationalization, not immunity.
*Status: The architectural claim (auditability, not immunity) stands. The prescriptive claim — that protecting conflict entries from certainty decay accelerates resolution — is directionally wrong (see T55). Argued: T54_implementation_brief.md.*

**T55. Organisms with read-shielded conflict records discover structural resolutions at a higher rate.** Rationalization (premature diff-collapse) destroys the information where resolution lives.
*Status: **Directionally falsified.** Tested in closed-loop T54 v2 experiment (organisms acting, certainty feeding observation vector, correspondence criterion). Across 6 seeds (2 runs × 3 seeds), the ordering was corrupted > unconstrained > shielded on every seed. Shielded organisms found the fewest resolutions in every run. The shield preserved certainty (0.39 vs 0.37 vs 0.33) but this hindered rather than helped resolution. High certainty on conflict entries = confidently wrong = exactly what needs to be released. The read-shielding prescription was protecting the wrong thing.*

**T57. Certainty release as annealing accelerates conflict resolution.** Loosening commitment to stuck predictions (reducing certainty on conflict entries) frees exploration and produces more correspondence-verified resolutions than either unconstrained or shielded read policies.
*Status: **Supported** (6 seeds, consistent direction, ~6% effect over shielded). Pre-registered as rival to T55. The ordering corrupted (1555.7) > unconstrained (1484.0) > shielded (1465.7) held across all seeds and both runs (pre-fix and post-fix for LP/pain-prediction corrections, confirming the result is not an artifact of either metric). This is the framework's first genuine structural discovery about itself: conflict resolution works by releasing commitment, not by protecting it. Compatible with Friston's prediction-error-minimization (updating the model requires releasing confident predictions that keep generating error), with the ABI-specific addition that this operates at the receptor level through a certainty mechanism.*

---

## XVIII. Epistemic Receptors

**T58. Belief and doubt are distinct receptors, not just certainty values.** Belief fires when certainty is high AND well-calibrated (predictions confirm). Doubt fires when certainty is high AND poorly calibrated (predictions keep surprising). The distinction is between certainty-as-infrastructure and certainty-as-detected-state. Without the receptor, belief/doubt are implicit weightings; with it, they become objects the organism can reason about.
*Status: Proposed. Predicted by the T57 annealing result — the organism had no receptor for the state it was in during annealing. Argued: genome_project/families/epistemic.yaml.*

**T59. Counterfactual salience is a receptor that fires when a non-actual state becomes motivationally loaded** — when "this could happen" becomes "this matters." Distinct from prediction (forward projection), memory (retrieval), and planning (action selection). The firing condition: S is not currently true, S is causally reachable, and the organism is evaluating S against its pain/endorphin model.
*Status: Proposed. Connects to conflict tolerance (Einstein holding Maxwell/Newton), to the annealing result (releasing certainty opens counterfactual space), and to serialization (counterfactual processing is serialization applied to possibility space). Argued: genome_project/families/epistemic.yaml.*

**T60. Epistemic strategy — deliberate management of belief, doubt, and counterfactual states — is the receptor-level mechanism for scientific reasoning (T50).** An organism that can choose when to commit and when to explore navigates the explore/exploit tradeoff at the epistemic level.
*Status: Proposed. The deepest canopy receptor in the Epistemic family. Argued: genome_project/families/epistemic.yaml.*

---

## XIX. Optimization as Receptor

**T62. Optimization is a receptor, not just a process.** It fires when the organism detects that a new solution dominates a prior solution for the same problem — fewer steps, lower cost, better outcomes, fewer exceptions. Requires maintaining competing solutions (multiple_hypotheses) and detecting dominance. An organism that commits to the first working solution never fires this receptor. Bridges to mathematics: optimization detects "this is better"; necessity_detection detects "this must be best."
*Status: Proposed. Argued: genome_project/families/formalization.yaml (form_010b).*

---

## XX. Logic as Receptor Family

**T64. Logic is a receptor family, not a cultural invention.** Transitivity, conjunction, quantification, contradiction, and general valid inference (it_follows) are receptors that fire on inference structure, just as pain fires on tissue damage. The organism that detects "this follows" has a survival advantage: it can act on information it hasn't directly observed.
*Status: Proposed. Argued: genome_project/families/logic.yaml.*

**T65. The it_follows receptor (general valid inference) requires all four specific inference types plus metacognition.** It is a meta-receptor that detects the shared property across transitivity, conjunction, quantification, and contradiction: the conclusion could not be otherwise given the premises. It may be the first receptor that genuinely requires linguistic scaffolding.
*Status: Proposed. Deepest canopy in the Logic family. Argued: genome_project/families/logic.yaml.*

**T66. Contradiction detection is the logical receptor that triggers the T57 annealing mechanism deliberately.** Rather than mechanical certainty decay, contradiction detection identifies when two confident predictions are incompatible and initiates targeted belief revision.
*Status: Proposed. Connects T57 (annealing) to the Logic family. Argued: genome_project/families/logic.yaml.*

---

## XXI. Language Receptors and Semantic Relations

**T67. Naming is a receptor, not a convention.** Concept stabilization via persistent labeling — giving a compressed causal chain a handle that survives across contexts — is a receptor that fires when a concept becomes stable enough to warrant a label. Without naming, concepts blur; with it, they become retrievable, comparable, and composable.
*Status: Proposed. Argued: genome_project/families/language.yaml.*

**T68. Self-talk is the serialization thesis applied to symbolic reasoning.** Internal deliberation works by activating named concepts in sequence, creating prediction opportunities between them. The organism talks to itself to think — not because words are necessary for thought, but because named concepts are more stable anchors for serial deliberation than unnamed ones.
*Status: Proposed. Argued: genome_project/families/language.yaml.*

**T69. Referential grounding is a receptor that detects whether a symbolic connection terminates in actual receptor states.** This is T2 (grounding failure is architectural) stated as a specific receptor: the receptor exists, but in an LLM its inputs are missing. Grounding failure occurs when the referential grounding receptor has no receptor states to terminate in.
*Status: Proposed. Argued: genome_project/families/language.yaml.*

**T70. Semantic relations ("has a", "is a", "causes", "precedes") are the trunk of the Logic family.** You can't build valid inference without first detecting what kind of relationship each link represents. Transitivity holds for "is a" but not always for "has a." The semantic relation receptor tells the logic receptors which inference rules apply.
*Status: Proposed. Argued: genome_project/families/logic.yaml.*

**T71. There are two levels of meaning: semantic relations (what kind of connection) and referential grounding (does the connection terminate in something real).** Both are receptors. Both are necessary. An organism can have semantic relations without referential grounding (structural understanding without experiential verification) or referential grounding without semantic relations (knowing something is real but not what kind of thing it is).
*Status: Proposed. Connects T69 and T70.*

**T72. Translation is a receptor that fires when an unknown symbol maps to a known consequence.** Not learning a new concept, but discovering that something familiar is being referred to through an unfamiliar code. Requires the environment to respond meaningfully to symbol codes (objects that jump when they hear "jump"). Second languages are learned faster than first because the grounded concepts already exist — translation maps new labels onto existing referents rather than building grounding from scratch. Metaphor is translation across domains rather than across vocabularies.
*Status: Proposed. Argued: genome_project/families/language.yaml.*

---

## XXII. Bridging: Recognition to Action

**T73. Mimicry is a receptor that fires when the organism detects correspondence between an observed action and its own motor repertoire.** Not imitation with understanding — pure motor matching that makes cultural transmission fast. Mirror neurons are the biological implementation. Requires proprioception and efference copy to map observed actions onto own body.
*Status: Proposed. Argued: genome_project/families/bridging.yaml.*

**T74. Executability is a receptor that detects whether a current thought has a motor translation available now.** Not planning (future states) but feasibility checking (current state). When executability fires low, the implementation gap is itself informative — it tells the organism what's missing and activates self-augmentation. This is the mechanistic link that makes environmental complexity load-bearing for capability development, not just for recognition.
*Status: Proposed. Argued: genome_project/families/bridging.yaml.*

**T75. The deliberation-to-action chain is: self_talk generates a thought -> executability checks feasibility -> if executable: implement -> if not: augment self or modify environment.** Without the executability receptor, the loop from recognition to action is implicit and recognition can outpace capability indefinitely. With it, the gap is a measurable signal the organism acts on.
*Status: Proposed. Connects Language (self_talk), Agency (planning), Self-Augmentation, and Environmental Augmentation through the executability hub.*

**T76. Shaped absence and missing-piece-located form a directed search cycle that is more specific than curiosity.** Curiosity fires on uncertainty in general. Shaped absence fires on a specific gap in a known pattern. The search it triggers has a target. Missing-piece-located closes the cycle when the specific gap fills. The pair together turn exploration into directed search, which is orders of magnitude more efficient.
*Status: Proposed. Argued: genome_project/families/compression.yaml.*

**T77. Mimicry emerges before executability and translation because it requires less self-model.** Testable ordering prediction: if executability or translation emerges before mimicry in any ERTI tier, the dependency structure needs revision.
*Status: Proposed (falsifiable ordering prediction). Argued: genome_project/families/bridging.yaml.*

---

## XXIII. The Thinking Substrate (MCTS as Cognitive Architecture)

**T78. A thought is a cycle — or cascade of cycles — in which internal receptors dominate the forward-feedback loop.** External receptors fire from the world. Internal receptors fire from the mental model's processing. When the organism responds primarily to its own receptor firings rather than to the world, that is a thought. The participating set of receptors is the *content* of the thought. The number of cycles before the cascade resolves or exits through the motor system is the *depth* of the thought. A reflex is one cycle dominated by external receptors. A thought is multiple cycles dominated by internal ones.
*Status: Proposed (definitional). Argued: README (Core Concepts).*

**T79. Planning is a thought where the mental model is the dominant source of receptor firings — the organism is responding to predicted futures.** The mental model runs `predict_delta` for actions the organism hasn't taken. Predicted states fire internal receptors (optimism, conflict, curiosity). The cascade across candidate actions IS the deliberation. Decision-making is when the cascade exits through the motor system.
*Status: Proposed. Argued: README (Core Concepts).*

**T80. MCTS provides the concrete substrate that makes metacognition tractable.** The tree structure is an architectural component — alongside the mental model and experience log — that records which thinking paths were taken, how often, and with what outcomes. The metacognition receptor has something specific to operate on: not vague introspection but structured visit counts, value estimates, and path divergence data.
*Status: Proposed. Argued: theories.md.*

**T81. The MCTS tree's metadata is itself input to receptors.** Visit count patterns trigger shaped_absence (underexplored regions). UCB scores trigger curiosity (high-uncertainty, high-potential branches). Value convergence triggers completion (the search has found what it was looking for). Path divergence triggers exception_detection. High-value low-visit branches trigger optimization. Repeated low-value high-visit paths trigger contradiction. The tree generates receptor inputs at every level.
*Status: Proposed. Argued: theories.md.*

**T82. The receptor topology is what makes the MCTS evaluation function intrinsic rather than external.** The value of a thinking path is determined by which receptors fire at its terminus and how strongly — not by a designer-specified reward function. This eliminates the ceiling that exists in every system with an externally specified objective: the system can only be as good as whoever specified what counts as good.
*Status: Proposed. Argued: theories.md.*

**T83. The self-modifying loop has no fixed ceiling because each layer feeds the next.** Better receptor topology → better evaluation function → better MCTS search → better thinking paths → richer tree → deeper metacognitive analysis → receptor firings that develop better receptor topology. The loop stabilizes only when the environment stops presenting new structure worth detecting. Since the environmental augmentation family allows the organism to increase environmental complexity, the ceiling rises with the organism.
*Status: Proposed. Argued: theories.md.*

**T84. Receptors are the only cognitive unit that generalizes across every level of the self-modifying loop.** Reward functions are level-specific. Loss functions are task-specific. Utility functions are designer-specified. A receptor — defined as any selected internal state whose activation changes behavior (T6) — fires on conditions whether they arise from the environment, from internal processing, from the MCTS tree, or from analysis of the analysis. This is why the bootstrapping loop is only conceivable with the receptor topology framing.
*Status: Proposed. Argued: theories.md. Depends on T6.*

**T85. Existing AI research has each piece but no bridge between them.** MCTS researchers (DeepMind) treat the tree as a tool with an externally trained evaluation function. Metacognition researchers describe thinking-about-thinking without a computational substrate. Intrinsic motivation researchers (Schmidhuber, Oudeyer) formalize curiosity as a reward bonus — modifying search intensity, not search evaluation or search perception. Self-modifying AI (Gödel machines) allows rewriting with fixed utility. None have a unit that works at every level because none started from the bottom where the question "what is the primitive unit of cognition" is forced.
*Status: Proposed. Argued: theories.md.*

---

## XXIV. The Novel Synthesis Claim

**T63. No existing research program unifies grounded cognition, active inference, and evolutionary receptor topology into a single generative mechanism.** ABI is a novel synthesis across Barsalou, Friston, Gibson, and developmental/evolutionary perspectives.
*Status: Proposed. Argued: whitepaper Section 10.*

---

## Summary

| Status | Count |
|---|---|
| Proposed | 64 |
| Partially tested | 10 |
| Supported | 6 |
| Revised | 1 |
| Falsified | 1 |
| **Total** | **82** |

Seven claims have direct experimental support from ERTI: T27-T31 (evolutionary dynamics), T26 (revised after decomposition), and T57 (annealing — the framework's first structural self-discovery). One claim directionally falsified: T55 (read-shielding accelerates resolution — the opposite was found across 6 seeds). The Epistemic family (T58-T60) is the first family whose existence was predicted by a falsification rather than by theoretical deduction alone. The Thinking Substrate section (T78-T85) identifies MCTS as the mechanism that makes metacognition tractable and the receptor topology as the unit that makes the self-modifying loop possible.

---

*The theories listed here are intended to be falsifiable. When predictions fail, the theory is revised or rejected — not defended. Falsification is the goal, not confirmation.*

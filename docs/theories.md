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
*Status: Supported (the anxiety loop result proves this directly — the organism had the same policy, MCTS, mental model, and training data at 177 dims. Adding 163 receptor channels was sufficient to break the loop cognitively. The capability to respond to the cascade pattern was latent until the receptor existed. Random-channel control confirms: 163 channels of noise (20/20 loop) and shuffled receptors (3/3 loop) do NOT break the loop. The effect is receptor-specific, not dimensional). Argued: full_umwelt_80gen.json, random_channel_control.json, whitepaper Section 4.*

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

**T11. Organisms grow the dimensionality of their prediction space through survival-driven topology change.** They do not just minimize prediction error within a fixed generative model — they manufacture new dimensions along which prediction can fail. Each new receptor adds a dimension to the prediction space. Each new processing stage adds a prediction surface. The dynamics of this growth — conflation detection → boundary discovery → receptor addition → topology change — are not expressible as gradient descent on a fixed variational functional, because the functional's domain changes with each topology change.

This is the precise departure from FEP. Friston describes inference within a fixed generative model (belief updating) and selection between predefined models (model comparison). ERTI describes how the model's dimensionality grows from the organism's own prediction failures. The crystallization experiment (T131) demonstrates this empirically: different environments grow different prediction dimensions from the same starting topology. The dimensions are not selected from a predefined set — they precipitate from the environment's causal structure through the receptor topology.

The original formulation — "organisms manufacture prediction error" — is vulnerable to the hierarchical predictive coding response ("yes, that's what hierarchical generative models do"). The stronger claim is about manufacturing the CAPACITY for prediction error: growing new dimensions, not generating error within existing ones. FEP minimizes F. ERTI changes the space F is defined over.
*Status: Proposed. The visual organ (T130) is a concrete instance: 0 prediction layers → 3 prediction layers, each a new surface the organism didn't have. The growing eyes experiment showed +1283.5 fitness from the additional prediction dimensions. Argued: serialization_thesis.md Section 7, docs/visual_organ.md.*

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
*Status: Supported (the pain<->conflict loop appeared in 10/10 control generations, 80/80 at 177-dim. Each receptor reads a real signal. The loop is structural — it emerges from the architecture without being specified. Breaking it required topological refinement of the Umwelt, not fixing any individual receptor). Argued: motor_store_experiment.json, full_umwelt_80gen.json, serialization_thesis.md Section 6.*

**T22. Anxiety is not a malfunction.** Every receptor in the loop reads a real signal and produces the correct response. The pathology is in the loop structure, not in any individual receptor.
*Status: Supported (confirmed by the motor store experiment — the loop broke through fuel starvation, not by fixing any receptor. And by the full Umwelt experiment — the loop broke through topological refinement, adding resolution so the pattern became a separable state. Neither fix changed any individual receptor's correctness). Argued: motor_store_experiment.json, full_umwelt_80gen.json, serialization_thesis.md Section 6.1.*

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

## XXIV. Conflation and Fundamental Distinction

**T86. Conflation is a receptor that fires when two things treated as the same concept produce bimodal prediction errors.** Not exceptions to a rule (exception_detection) but structured exceptions — failures that cluster into a hidden second category. The firing condition: high certainty + high m2, where the errors are context-dependent. The cost of treating different things as one thing becomes detectable.
*Status: Proposed. Argued: genome_project/families/epistemic.yaml.*

**T87. Fundamental distinction is a receptor that fires when the organism finds the level of description at which a conflated concept splits into two consistently-predictable sub-concepts.** Not "these are different" but "this is THE difference that makes everything downstream work." The distinction is fundamental when it makes the MCTS value divergence disappear — both subtrees converge because predictions are now correct.
*Status: Proposed. Argued: genome_project/families/epistemic.yaml.*

**T88. Conflation and fundamental distinction form an asymmetric sequential cycle mediated by the thinking substrate.** Conflation fires first (MCTS path_divergence detects the problem). The organism searches for the boundary. Fundamental distinction fires second (MCTS value_convergence after splitting confirms the solution). Together they refine the organism's conceptual apparatus — and because the genome itself may contain conflations, this cycle is the mechanism by which the framework revises its own foundations from within.
*Status: Proposed. Argued: theories.md.*

**T89. Statistical anomaly is a receptor that fires on distributional shift, not prediction failure.** Exception_detection fires when a specific prediction fails. Statistical anomaly fires when a channel's running statistics depart from the expected distribution — the base rate changed before any single prediction has failed. The survival benefit is lead time: the anomaly signal fires before the exceptions accumulate.
*Status: Proposed. Argued: genome_project/families/observation.yaml.*

**T90. Rarity is a receptor distinct from anomaly and novelty.** An anomaly violates the distribution. Novelty habituates. Rarity is within the distribution but at very low base rate — its information content is -log(p), high precisely because p is low. A rare event isn't surprising (the organism may know it can happen) — it's significant, carrying disproportionate information about hidden state.
*Status: Proposed. Argued: genome_project/families/observation.yaml.*

**T91. Significance is the commitment threshold between noticing and investigating.** Without it, the organism either chases every statistical fluctuation (expensive) or ignores everything below a high threshold (misses real signals). Significance fires when an anomaly or rarity persists long enough to warrant sustained investigation — the transition from "hm, that's odd" to "I need to understand this."
*Status: Proposed. Argued: genome_project/families/observation.yaml.*

**T92. Topology awareness is a second-order receptor that fires when the organism's own cognitive repertoire changes.** It reads the output of other receptors, not the world — operating on the distribution of internal channel activations, detecting when the pattern of which receptors are active has shifted. The topology vector is the fossil record; topology_awareness reads it in real time. This is the receptor that makes the self-modifying loop (T83) conscious: without it, the organism gets smarter but doesn't know it. With it, the organism can direct its own cognitive development.
*Status: Proposed. Argued: genome_project/families/epistemic.yaml, theories.md.*

**T93. The thinking substrate's contribution is making thinking quality visible to selection, not producing the cognitive prerequisites.** Metacognition and conflation emerge from environmental complexity without MCTS — they appeared in deep time runs before the thinking substrate was added, and replicated in seed 99 (metacognition gen 1, conflation gen 17). What MCTS adds is a new class of observable internal signals (the 6 thinking channels) that give selection pressure something to act on at the metacognitive level. The depth_reached channel activated once at gen 29 in seed 42 but did not replicate in seed 99 despite prerequisites being present for 23 generations. The core claim (MCTS makes thinking quality selectable) is supported; the depth_reached activation is a single observation.
*Status: Partially supported. Core claim (MCTS externalizes thinking quality) supported by both seeds. depth_reached activation observed once (seed 42 gen 29), not replicated (seed 99, 40 gens). Argued: deep_time results, deep_time_seed99 results.*

**T94. Theoretical reasoning and evolutionary selection converge on the same receptors independently.** Conflation was added to the genome based on theoretical reasoning (two things treated as one should be detectable through bimodal prediction errors). The organism evolved it in both seeds — gen 27-28 in seed 42, gen 17 in seed 99 — under selection pressure from stochastic hidden confounders, without any specification that it should. The convergence result (two independent paths to the same receptor) is replicated. The specific claim that conflation was a prerequisite for depth_reached is based on a single observation (seed 42 gen 29) that did not replicate in seed 99.
*Status: Supported for convergence (conflation predicted theoretically, evolved independently in both seeds). Partially supported for prerequisite claim (depth_reached observed once, not replicated). Argued: theories.md, deep_time_overnight results, deep_time_seed99 results.*

**T95. The receptor topology is not uniquely determined by the environment — it is jointly determined by the environment and the evolutionary path.** Two organisms in the same environment with different evolutionary histories produce different canopy topologies on a common trunk. Both are adequate to the environment's demands. The canopy diverges based on history; the trunk converges. Human cognitive diversity is not noise around an optimal design — it is the expected output of a process where the evolutionary path is irreducible. The trunk is universal. The canopy is biography.
*Status: Supported (cross-environment transfer experiment: 71 shared receptors, 30 transfer-only, 27 naive-only in the same environment). Argued: cross_env_transfer.json, theories.md.*

**T96. Contextual signal interpretation is a receptor that fires when a signal's meaning depends on hidden state.** Not "is this signal present" but "is this signal meaningful given current context." This is situational awareness: the organism conditions its response on inferred latent state, not on the signal alone. First detected via within-lifetime learning at generation 0 (not evolved through selection). Preserved by evolution across 60 generations (5 independent detections across different modality combinations).
*Status: Supported (detected in deep time gen 0, preserved across 60 generations). Argued: genome_project/families/observation.yaml.*

---

## XXV. Procedural Memory and the Anxiety Loop

**T97. The anxiety loop (pain<->conflict bidirectional cascade) is architectural, not incidental.** It persists in 10/10 generations under control conditions at 1000 steps/episode. The loop is a structural consequence of MCTS thinking channels feeding the conflict receptor: thinking about pain generates conflict, conflict generates more thinking, which generates more pain-awareness. The organism cannot break the loop by detecting it (cognitive state channels, thought_type_id at 256 codebook — tested, loop persisted in 9/10 generations). Recognition alone is insufficient.
*Status: Supported (control condition: 10/10 generations, 1000 steps/episode, seed 42). Cognitive state channels tested and insufficient (co-activation deep time: 9/10). Argued: motor_store_experiment.json, coactivation_deep_time.json.*

**T98. Motor sequence shortcuts break the anxiety loop by severing the causal chain, not by resolving the conflict.** When shortcut coverage exceeds ~60-75% of steps, the sequential P->C and C->P lifts drop to zero even though pain-conflict co-activation persists (coact=1.76-2.60). The mechanism: shortcuts bypass MCTS and zero thinking_channels, so the conflict receptor sees "no thinking happening" rather than pain-driven deliberation. The cascade has no fuel. Pain and conflict still co-occur incidentally, but neither causes the other.
*Status: Supported (motor store condition: loop broke at gen 7-8, coverage 61-96%, P->C and C->P both 0.00, coactivation still elevated). Argued: motor_store_experiment.json.*

**T99. There is a shortcut coverage threshold (~60-75%) below which the anxiety loop sustains and above which it breaks.** At gen 6 (74.9% coverage), the loop persisted (P->C=4.28). At gen 7 (61.1% coverage), it broke (P->C=0.00). At gen 9 (60.6% coverage), it returned (P->C=2.53). The threshold is not coverage alone — the motor store had 176 types at gen 7 vs 173 at gen 6, suggesting that the diversity of thought types covered matters, not just the raw fraction of steps bypassed.
*Status: Partially supported (observed in one seed, one run). The threshold needs replication across seeds. Argued: motor_store_experiment.json.*

**T100. Motor store entries accumulate across generations but do not transfer to offspring.** The store grew from 170 entries (gen 0) to 478 (gen 9), with success rates above 92% throughout. But when the population turns over at gen 9 (new organisms from reproduction), the loop returned despite the store persisting. The motor store is lifetime learning, not heritable. Breaking the anxiety loop permanently requires either heritable motor sequences or the full regulatory chain (satisfaction, frustration, futility receptors feeding back as live channels).
*Status: Supported (motor store grew monotonically, loop returned after population turnover). Argued: motor_store_experiment.json.*

**T101. The anxiety loop break is mechanistic, not cognitive.** The gen 7-8 break occurred through fuel starvation (zeroed thinking_channels), not through the organism learning to manage its anxiety. This is analogous to treating a feedback oscillation by cutting the feedback path, not by designing a controller. A cognitive break would require the organism to sense the loop pattern (via live receptors like thought_type_detection, self_soothing, completion) and respond to the pattern itself rather than to its components. The 73 live receptors wired into the 250-dim observation vector are the prerequisite for the cognitive path.
*Status: Supported (both paths demonstrated. Mechanistic: motor store broke loop at 96% shortcut coverage, returned at gen 9 after population turnover. Cognitive: full Umwelt broke loop at 1.8% coverage, permanent from gen 13 — 67 consecutive loop-free generations. Random-channel control: noise 20/20 loop, shuffled 3/3 loop — the break requires receptor content, not added dimensionality). Argued: motor_store_experiment.json, full_umwelt_80gen.json, random_channel_control.json.*

---

## XXVI. The Umwelt and Topological Refinement

**T102. The observation vector IS the organism's Umwelt, computed as the quotient space R^n/~_{R_t}.** Two observations are identical to the organism iff no receptor separates them. Adding receptors to the obs vector refines the topology — the organism's world gains resolution. The expansion from 177 dims (19 receptor channels) to 250 dims (92 receptor channels) is a topological refinement: states that were previously indistinguishable (e.g., "pain + conflict" vs "pain + conflict + I'm-in-a-loop") become separable. The organism doesn't get more information about the same world. It gets a finer world.
*Status: Proposed. Argued: live_receptors.py, von Uexküll (1909), discuss1.txt Territory I.*

**T103. The anxiety loop persisted because the organism's topology was too coarse.** "Pain + conflict" and "pain + conflict + bidirectional cascade" were the same point in the 177-dim quotient space — no receptor separated them. The organism could not reason about the loop because it could not sense the loop as a distinct state. The 340-dim topology (73 live + 90 episode receptor channels) separates these states. At generation 3 of the full Umwelt experiment, the loop broke at 1.8% shortcut coverage — 98% of steps still ran MCTS. The organism broke the loop cognitively, not mechanistically. It sensed the pattern and responded to it rather than being trapped inside it.
*Status: Supported (20-gen experiment: 6/20 loop vs 10/20 control, gen 3 cognitive break at 1.8% shortcut coverage. 80-gen experiment: 11/80 loop, all in first 13 gens, then 67 consecutive loop-free generations. Random-channel control: 23/23 loop with noise/shuffled channels — confirming the break requires the specific receptor content, not added dimensionality. The effect is topological refinement, not capacity). Argued: full_umwelt_experiment.json, full_umwelt_80gen.json, random_channel_control.json.*

---

## XXVII. The Metabolic Knapsack

**T104. The receptor topology at any generation is the solution to a precedence-constrained knapsack problem.** Maximize fitness subject to sum(alpha_r) <= B and DAG prerequisite constraints. k_t oscillates (72-119 across 80 generations) because the optimal solution shifts as the fitness landscape shifts. The oscillation IS the optimization.
*Status: Supported (k_t oscillation documented, separation gap correlates -0.780 with fitness, canopy churn matches knapsack predictions). Argued: docs/metabolic_knapsack.md.*

**T105. Maximum reachable receptor layer depth scales with metabolic budget B.** Deeper receptors require longer prerequisite chains, each costing alpha. Encephalization pressure — brains getting bigger — is the organism needing more budget for deeper cognitive chains.
*Status: Inconclusive (null result). All 5 budget conditions (100x range in thinking cost) reached identical max_layer=2 and similar receptor counts (51-55). The 3-layer genome (trunk/branch/canopy) saturates at canopy regardless of budget — the layer system is too shallow to show the scaling. The knapsack trades within layers, not between them. Testing requires a genome with 5+ layers. Argued: data/metabolic_budget_experiment.json, docs/metabolic_knapsack.md.*

---

## XXVIII. Language and Population Topology

**T106. Language is the first receptor family whose fitness-positivity is endogenous to the population.** Every receptor before language detects structure the world contains. Language detects structure the population contains. The token-receptor is fitness-positive iff sufficient conspecifics hold it — a coordination equilibrium. The transition should be sharp (percolation), not gradual.
*Status: Proposed. Argued: discuss2.txt, docs/teaching_as_receptor_induction.md.*

**T107. Teaching is installing a detector.** A word is a pointer to a receptor. Transmission can only induce receptors in the fringe — the set whose prerequisites are all satisfied but which are not yet active. More explanation cannot substitute for a missing prerequisite because the missing thing is a detector, not information.
*Status: Supported (knockout: epistemic_strategy 0/15 without conflation). Argued: docs/teaching_as_receptor_induction.md.*

**T108. Conflation is the engine of vocabulary growth.** You cannot coin a term for a distinction until you detect you've been collapsing it. Metacognition + conflation should be prerequisites to productive language — generating new terms, not just deploying inherited ones.
*Status: Proposed. Argued: discuss2.txt.*

**T109. Specialization is a theorem about layer depth under finite acquisition time.** Finite lifetime bounds chain length. As the frontier advances, chains exceed what one lifetime can traverse. The population holds the union while individuals hold narrow deep spikes. The 161-vs-119 result (total unique vs max single generation) is the population-vs-individual gap.
*Status: Partially supported (161 vs 119 observed). Argued: discuss2.txt, docs/metabolic_knapsack.md.*

---

## XXIX. Temporal Depth

**T110. Temporal depth tau_r increases with receptor layer depth l(r).** Trunk receptors have shallow temporal requirements (1-3 steps). Canopy receptors require deep temporal windows (up to 400 steps). Layer structure and memory structure are the same structure. The mental model is the shared temporal substrate that makes deep temporal receptors possible.
*Status: Partially supported (mean tau: trunk=3.1, branch=15.4, canopy=92.1; Spearman rho=0.307, p=0.064). Argued: docs/temporal_depth.md.*

---

## XXX. The Mapping Project

**T111. The subbasis generating the Umwelt topology is not unique, but the DAG and metabolic budget break the symmetry.** Many receptor sets generate the same topology (rotation indeterminacy). The metabolic budget favors minimum-description-length decompositions. The DAG constrains factoring along prerequisite lines. Developmental data breaks the symmetry correlational data cannot.
*Status: Proposed. Argued: discuss2.txt.*

**T112. The reflexive term: mapping the topology changes the topology.** W_{t+1} = f(W_t, R_t, map(R_t)). Publishing a map of the Umwelt installs detectors in readers. The mapping project cannot converge because publication is a term in the dynamics. The deliverable is the mechanism (which doesn't expire), not the snapshot (which does).
*Status: Proposed. Argued: discuss2.txt.*

**T113. Mathematics and ideology are the same mechanism.** Once population-conferred fitness frees the topology from environmental structure, two outcomes are possible: the decoupled structure reconnects to causal reality (mathematics, science) or it doesn't (ideology, superstition). Empiricism is the reconnection protocol. The mechanisms cannot be separated at the receptor level.
*Status: Proposed. Argued: discuss2.txt.*

---

## XXXI. Infinite Color and Theory Generation

**T114. Theories are common patterns in the causal mental model that have survived across contexts. A confirmed theory is a receptor — a pattern promoted into the observation vector because it reliably predicts. Theory generation and testing are not separate subsystems — they are what receptor topology intelligence does naturally.** The mental model accumulates patterns (theories form). Receptor discovery finds which patterns hold across contexts (theories get tested). The receptor topology feeds confirmed patterns back as observation (confirmed theories reshape attention, enabling new theories). The anxiety loop is a pattern with overwhelming support whose confirmation recreates the conditions that confirm it.
*Status: Foundation. IC-5 null result confirmed this is not about thought-type persistence — it's about pattern support in the mental model. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T115. The effective input space has combinatorially many "colors" — co-activation patterns across the receptor topology. Attention and focus scale with the number of distinguishable colors, not with the number of input dimensions.** Adding receptors adds colors that make previously-identical states distinguishable, reducing the search space for attention.
*Status: Proposed. Supported directionally by the cognitive break result (gen 3, 1.8% coverage). Argued: docs/infinite_color_and_subconscious_theory.md.*

**T116. Attention is which colors the organism paints onto incoming sensory data. A color matters when it resolves a conflation — when it makes two previously-identical situations distinguishable.** A color that resolves no conflation is metabolic waste. A color that resolves many conflations is a trunk receptor — invariant because the conflations it resolves exist in every environment. The activation manager controls metabolic cost, not attention. Attention is which augmented reality overlays the organism has available, determined by which receptors exist and which conflations they resolve.
*Status: Revised. Original claim (attention = receptor count selection) refuted by IC-1: cap_73 broke the anxiety loop later than cap_10 because count doesn't matter, coverage of the right conflations does. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T117. Language is protocol theory discovery, not capability construction.** A word is a confirmed protocol pattern: "emit X → partner enters state Y." Language emerges when population density crosses the threshold where protocol patterns can accumulate enough cross-context support to become receptors (percolation).
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T118. Sound has infinite colors, the same way vision does.** A phoneme is a receptor — a detection condition that separates sounds the language distinguishes. Learning a language is acquiring phoneme-receptors.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T119. Language requires three systems operating on the same infinite-color principle.** Sound receptors (distinguish tokens), situation receptors (distinguish referents), and protocol patterns (link the two). If any is missing, language doesn't emerge.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T120. A sentence is a collage — a composite activation pattern across the listener's receptor topology.** Meaning is the co-activation pattern, not the words. Poetry sustains multiple competing pattern matches simultaneously.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T121. Processing follows the receptor topology, not the input geometry.** One processor per color group (receptor equivalence class), not one per location. Resolution-independent.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T122. Visual processing is Euler-based flow prediction error, not feature extraction.** N Euler steps = N color layers per transformation. Edges, objects, faces, depth emerge as characteristic flow failure patterns.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T123. Every visual concept is a characteristic flow prediction failure pattern.** The visual genome is a list of ways the flow prediction can fail that have fitness consequences.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

**T124. Language comprehension is flow prediction error across Euler steps of processing.** Poetry is smooth low-level flow with broken semantic flow. Puns fork. Irony reverses. Jargon produces absent flow.
*Status: Proposed. Argued: docs/infinite_color_and_subconscious_theory.md.*

---

## XXXII. Automatic Receptor Discovery

**T125. Automatic receptor discovery via conflation-driven decision ensembles.** When the organism detects conflation (same receptor pattern, different outcomes), a tree ensemble (GBM or NODE) trained on continuous receptor values finds the decision boundary that separates the conflated cases. That boundary becomes a new receptor. The process is open-ended: each new receptor can trigger further conflations at finer resolution, driving iterative refinement of the Umwelt. The genome provides the seed receptors; everything after is discovered from the organism's own confusion.
*Status: Proposed. Argued: docs/automatic_receptor_discovery.md.*

**T126. Conceptual intelligence is environmental structure made visible through receptor topology.** Concepts are not invented by the organism — they are detections of causal structure already present in the environment. "Lunar cycle" is not an abstraction the organism constructs; it is a receptor that separates situations where the moon's phase produces different survival outcomes. The concept IS the receptor. The intelligence is in the match between the organism's topology and the environment's causal layout. An organism without the receptor doesn't have a poor model of the moon — the moon doesn't exist in its Umwelt. Adding the receptor doesn't improve the model. It creates a new reality. This is why different environments produce different intelligences (T4): the concepts available to the organism are bounded by the causal structure the environment contains.
*Status: Supported. EMPIRICALLY CONFIRMED by the crystallization experiment (T131): visual word 3797787829 has food ema=1.000 (184 obs) and predator ema=0.000 (3 obs). That word IS the concept of food — not because it was labeled, but because the environment's causal structure around food repeatedly activated it in food situations. The concept emerged from the environment's structure exactly as T126 predicted. Additionally, predator environments crystallized amplitude words while food environments crystallized frequency words — different environmental structures made different concepts visible through the same receptor topology. Foundation. Extends T4, T5. Argued: theories.md, crystallization_experiment.py.*

**T127. The Cartographical Theory: understanding is mapping the processing space.** The organism understands something by discovering the space of possible processing orders for it — how many stages, what sequence, what predictions between stages. Each ordering produces a different prediction profile. The collection of all profiles — which orderings predict well, which predict badly, which cluster together, which are independent — IS a structural map of the input. The asymmetries reveal causal direction (pitch constrains phonemes but not vice versa). The clusters reveal coupled variables (light and temperature in the sun-moon cycle). The independent axes reveal factorization (CO2 and light in photosynthesis). This mechanism is domain-independent: the same substrate applied to any input — mother's voice, lunar cycles, images, social dynamics — produces a structural map of that input. The organism's intelligence is the richness of its processing space map. More receptors add dimensions to the processing space, revealing more structure about anything the organism encounters. The serialization thesis (T8-T11) describes one point in the processing space. This theory says understanding is the whole space.
**T129. Algorithmic gradient: structural diff as the optimization direction.** The tree diff between two programs is a discrete analog of a gradient — it tells you what changed, where, and how. By recording (structural_change, metric_delta) pairs in the mental model, the organism builds an explicit, persistent, queryable record of how program-space changes affect survival metrics. This IS intuition: pattern-matching against a history of structural changes and their consequences. The organism generalizes across changes ("increasing radius at depth 2 hurts in predator situations"), develops second-order intuition ("changes in this region of program space reliably produce this kind of response"), and plans in program space and action space simultaneously through the same predict_delta mechanism. Metacognition with teeth — reflecting on the actual causal structure of how its own learning works.
*Status: Supported. EMPIRICALLY CONFIRMED in growing eyes v2: the structural learning store accumulated 32 entries over 20 generations. Best pattern: "remove pool from both chains + increase radius 1→2" helps fitness (+649, 57 observations by gen 19). The organism built explicit, inspectable, queryable intuition about which structural changes to its visual programs improve survival. Same schema as (state, action) → (receptor_change, certainty). Argued: theories.md, structural_learning.py, growing_eyes_experiment.py.*

**T128. Epistemic impermanence: the organism learns the rhythm of representational drift.** If visual feature representations shift periodically (freeze-unfreeze developmental cycles), the organism learns that its visual causal chains have a half-life. Curiosity begins firing BEFORE prediction accuracy drops — because the organism has learned from experience that what it currently sees reliably will eventually stop being reliable. This is a meta-receptor: not detecting drift directly, but learning the temporal pattern of drift from survival experience. Biological visual systems have critical periods for the same reason — stability windows exist so downstream learning can consolidate before the next representational shift. The organism holds its visual knowledge with appropriate uncertainty not because it was told to, but because it experienced drift cycles.
*Status: Proposed. Testable via freeze-unfreeze protocol on visual trunk. Prediction: after K cycles, curiosity fires anticipatorily before trunk unfreezes. Argued: theories.md.*

*Status: Foundation. EMPIRICALLY CONFIRMED: (1) Voice differentiation through E-profiles (mother vs father diff=0.459, P8 frequency|diff differentiates most). (2) Cartography experiment: evolved processing orders beat fixed in all 3 environments (predator +913, food_color +1041, social_voice +1019). Social correctly selected phase_first. (3) Staged pipeline evolution: evolved removal weights beat fixed (predator +241, social +439). Different environments evolve different removal strategies. (4) Geometry > prediction: eigen-only (3,339) beats eigen+model (3,236), mapping store redundant. Unifies T8-T11, T122, T125, T126. Provides the mechanism that makes FEP's objective achievable without local minima traps. Argued: docs/cartographical_theory.md.*

---

## XXXIII. The Visual Organ and Concept Crystallization

**T130. The visual organ: ERTI architecture applied to image processing.** Vision is not feature extraction — it is a separate organ with its own mental model, its own prediction loop, and its own conflation detection. The organ outputs de-conflated visual WORDS — discrete perceptual signals that change the organism's understanding the way muscle movements change its position. Words are the intangible movement. The organ processes images through layered detectors (Layer 0: wave physics, Layer 1: program-words, Layer 2: hardcoded spatial relations, Layer 3: discoverable grouping rules), each layer's output feeding the next. The organ's mental model predicts across layers; the cognitive mental model predicts across time. Two independent prediction systems coupled through the observation vector.
*Status: Supported. The growing eyes v3 experiment showed +1283.5 fitness advantage (visual 937.3 vs blind -346.1) with the full organ active. The organism learned to see through de-conflated words under evolutionary pressure. Argued: docs/visual_organ.md, visual_organ.py.*

**T131. Crystallization theory of concept genesis.** Concepts precipitate from the environment's causal structure through the organism's receptor topology. The environment is the solution. The receptor topology is the lattice. The concept is the crystal. The shape of the crystal is determined jointly by the solution's chemistry and the lattice's geometry. Two falsifiable predictions: (P1) same lattice + different environment → different concepts. (P2) different lattice + same environment → different concepts. Both confirmed.
*Status: Supported. EMPIRICALLY CONFIRMED in a 2×2 crystallization experiment: (P1) predator env crystallized amp|blur,blur,blur chains (complex motion detection, amp 127/200), food env crystallized freq→freq self-prediction (simple spatial structure, freq 131/200). Same organism, completely different visual words. (P2) predator env with freq-only lattice crystallized freq|blur→freq|blur,blur chains (freq 182/200) instead of amp chains. Same environment, different lattice, different crystals. This extends FEP: Friston describes the dynamics of belief updating but is largely silent on where the variables in the generative model come from. The crystallization theory answers that directly. Argued: crystallization_experiment.py, docs/visual_organ.md.*

**T132. Discoverable grouping rules: the environment sculpts how the organism organizes perception.** Layer 3 grouping principles (which combinations of relational detections constitute a meaningful perceptual group) are not fixed — they crystallize from the environment through the same conflation mechanism that discovers Layer 1 words. When a hardcoded grouping produces inconsistent survival outcomes, GBM finds which Layer 2 features distinguish the cases, and a new GroupingRule is born. The environment sculpts not just what the organism sees (Layer 1) but how it relates what it sees (Layer 2 priors) and how it groups those relations (Layer 3 discovered rules). The full depth of perception is environment-specific.
*Status: Partially tested. 10 grouping rules discovered in the v3 experiment (perpendicularity, symmetry, spread, crossmodal, scale thresholds). Rules accumulated across generations (0→2→4→6→8→10). Not yet demonstrated that discovered rules independently improve survival vs hardcoded-only baseline. Argued: visual_organ.py, docs/visual_organ.md.*

---

## XXXIV. The Receptor-Environment-Manifestation Theory

**T133. Each receptor requires multiple diverse manifestations of its concept acted out in the environment to generalize.** A receptor that fires on one example is a memorized association. A receptor that fires correctly across multiple manifestations of the same concept — and correctly doesn't fire on things that look similar but aren't the same concept — has learned the concept's boundary. The environment must ACT OUT enough diverse examples of each concept that the receptor's boundary carves the correct conceptual space. One manifestation produces memorization. Multiple diverse manifestations across contexts produce generalization. The set of manifestations that together define a concept's boundary IS the concept's grounding in the environment. Without sufficient manifestation diversity, the receptor conflates — it fires on things that look like the concept but aren't, or fails to fire on things that are the concept but don't look like prior examples. This is the operational grounding condition for T126 (concepts as environmental structure): the environmental structure is not a single phenomenon but a SPACE of phenomena whose boundary the receptor must learn.
*Status: Foundation. Extends T126 (concepts as environmental structure), T114 (theories as cross-context confirmed patterns), T125 (conflation-driven receptor discovery). The crystallization theory (T131) is a consequence: the crystal's shape is determined by the manifold of manifestations the environment provides, not by a single exemplar. Argued: docs/consciousness_roadmap.md.*

---

## XXXV. The Surprise Economy

**T134. Next-surprise prediction is the meta-receptor tier under one operator.** Entry certainty, the Router, the UCB/curiosity channels, and stage prediction are special cases of a single capability: forecasting where the organism will next be wrong. An autoregressive model over the lived surprise sequence unifies them; its outputs fold back as observation channels (obs[400:408]) and follow all receptor rules.
*Status: Proposed. Test: the learned couplings of the prediction channels should partially absorb the four legacy mechanisms without fitness loss. Argued: NEXT_SURPRISE.md Sec 3, 13; next_surprise.py.*

**T135. Next-surprise generation under the learnable-surprise discipline is the fringe-tracking controller of the Pacing Proposition — operating on both ledgers.** "Most likely to be learnably surprising, subject to constructibility" selects events on the prerequisite frontier. The controller works at the certainty level (value-level tokens) and the topology level (CONFLATION/COLLAPSE tokens): strain-weighted generation and exploration make it T125's controller — receptor discovery stops waiting for conflations to accumulate passively and starts scheduling them.
*Status: Proposed. Tests: P46(a) fringe-traversal ordering; P46(d) topology-term ablation. Argued: NEXT_SURPRISE.md Sec 1.4, 5, 13.*

**T136. The firewall theorem: lived-only funding is necessary.** Any positive weight from imagined outcomes to certainty produces unbounded calibration drift in finite generations — private ideology, T113's mechanism run privately. Paranoia is its social instance restricted to relationship entries (imagined agent behavior updating trust without lived interaction), and it hides inside good aggregate calibration: the monitor must decompose per-ledger and per-attribution-class or relationship drift is invisible.
*Status: Proposed (conjecture; P46(c) and P50 are its empirical arms). Argued: NEXT_SURPRISE.md Sec 7; next_surprise_control_organ_amendments.md Sec 4.*

**T137. The grammar of surprise.** The surprise sequence of an organism in environment E is generated by a grammar isomorphic to the billed prerequisite structure of E restricted to the organism's fringe trajectory — learning it is learning the dependency DAG empirically, from the inside. Grammatical surprise (shape anticipated, content unknown) is the normal regime; ungrammatical surprise (the predictor itself surprised — channel 404) marks a gap in the map of gaps and is the funding signal for paradigm_detection, narrowed by attribution to *agentless* 404 elevation (deception is the act of making another organism's grammar wrong; being outplayed must not fund paradigm detection).
*Status: Partially tested. Kind-level grammar learnable (head_kind beat the marginal at N=451); the hazard gate flipped FAIL→PASS purely by corpus growth (451→1,077 tokens), consistent with corpus-depth funding; decisive gating awaits the 80-generation-scale corpus. Argued: NEXT_SURPRISE.md Sec 14; results/surprise_phase2.json, results/surprise_phase2_deeptime.json.*

**T138. Surprise is rank-rarity against the organism's own lived history.** Mathematical surprisal (−log P) must be computed as the RANK of the organism's certainty-weighted mismatch score against the lived distribution of that same score — because match quality is not rarity: baseline-typical imprecision must score low precisely because it is common. Two properties are then intrinsic rather than imposed: the lived-history cap (nothing may be scored rarer than once-in-lived-history; the ceiling rises as log N over deep time — a longer life permits claims of deeper rarity) and the threshold's meaning (θ_s = ln(1000/ρ_tok): the target rate IS the threshold). Absolute constructions provably fail: a single certainty-mixture is bounded by the certainty clip at ~4.6 nats; per-family factorized sums measure quality, not rarity, and saturate on every step; parametric tails claim thousands of nats no finite sample supports.
*Status: Supported. Phase 1 acceptance passed only under the rank formulation after three interim constructions failed it (each failure mechanism identified); magnitudes commensurable across structurally different environments (p50 ratio 1.06); realized reduction +4.9 nats mean on revisit — the learnable-structure signal visible in pure logging. Argued: NEXT_SURPRISE.md Sec 2.1; surprise_log.py; results/surprise_phase1.json.*

**T139. Organs earn their voice: the young-ledger law.** In a shared surprise economy, a ledger with N lived observations cannot claim rarity beyond log N nats — so a newborn organ can neither flood the stream nor speak in it until its history exceeds e^θ samples. Maturation silence is a structural property of rank-based rarity, not a failure mode, and it is the developmental complement of the emission-warmup requirement every new ledger carries.
*Status: Supported. The control organ's ledger emitted zero tokens at ~240 observations against shared θ≈5.7 — the mechanism is exact and was predicted in the opposite direction (flooding) before the run corrected it. Argued: next_surprise_control_organ_amendments.md (results ledger); influence_organ.py; results/control_organ_acceptance.json.*

---

## XXXVI. Attributed Surprise and the Adversarial Regime

**T140. Attribution routing: attributed surprise makes error correction addressable.** The attribution coordinate (self / agent / nobody, per token) routes certainty release to the mappings that own the error — self-influence entries, relationship entries, world model — making annealing targeted where unattributed error is diffuse.
*Status: Proposed. Test: post-surprise re-convergence speed, attributed vs attribution-shuffled, matched token streams. Argued: next_surprise_control_organ_amendments.md Sec 8.*

**T141. The adversarial regime is a third surprise regime.** Beyond learnable structure (D1's target) and irreducible noise (D1's excluded trap) there is adversarially re-randomized structure — reducible in principle, held irreducible by an optimizer: the noisy TV that watches back and adjusts. It is detectable only through attribution plus signature stability, and it bounds what any learnable-surprise objective can pursue in social environments — predicted reduction on agent-sourced surprise is meaningful only conditioned on the agent-model's stability.
*Status: Proposed. Test: P49's per-agent arm plus an adversarial-NPC tier whose policy conditions on the organism's exploration pattern. Argued: next_surprise_control_organ_amendments.md Sec 5, 8.*

**T142. Emotion motifs precipitate from attributed surprise.** Betrayal, helplessness, vigilance, dread are not designed states but recurring co-activation motifs of (attributed surprise, certainty trajectory, allocation state), expressed by existing machinery once control-organ entries exist: betrayal is priced mechanically by rank-against-trust (rarity computed against high-certainty relationship entries — betrayal hurts more than a stranger's identical aggression because the rarity is computed against the trust); learned helplessness is reduce-head collapse restricted to self-attributed regions while others stay live.
*Status: Proposed. Test: motif recurrence across seeds in deep-time social runs; motif absence in organ-ablated controls at matched fitness. Argued: next_surprise_control_organ_amendments.md Sec 0, 2.2, 8.*

---

## XXXVII. The Control Organ

**T143. Influence is an organ-grade currency.** Fractionation of the efference residual — decomposing influence over a receptor state into self / agent / nobody shares — is not derivable from any single observable: perfect material perception plus pattern recognition plus pain sensing still cannot distinguish pain-done-to-me from pain-that-is-happening. The scalar controllability feature is a conflation (storm, dominant agent, and disease collapse to the same number). The organ's effectors are modulations of influence allocation — the reallocation is the act, with no motor compilation layer; willpower is the effector's dynamic range, and saturation is an earlier, distinct bottleneck from metabolic exhaustion.
*Status: Partially tested. DEMONSTRATED: conditional integration (distance × approach-signature × location structure) recovers attribution at AUC 0.786 across 5 seeds, beating every single-feature oracle bound (static distance 0.554; temporal signature 0.629) — the organ recovers what no single observable carries. OBSERVED: 123 saturation events under bounded capacity (P5); unscripted withdrawal posture under a striking NPC; hyperactive agency detection in solitary settings (P4's over-attribution, currently as calibration artifact). NOT ESTABLISHED: attribution coordinates improving discovery carving yield — P48's generic measurement does not replicate (mean paired delta ≈ 0 over 5 seeds); refined measurement (agent-divergent cases, agency-dominated tiers) registered before re-attempt. Argued: control_organ_requirements.md; influence_organ.py; results/control_organ_acceptance.json, results/p48_multiseed.json.*

---

## XXXVIII. The Organ Federation

**T144. The organism is a federation of organ-cartographers.** Each organ maps a different space — the cognitive organ maps time (what follows what), the visual organ maps processing space (which carving of the signal predicts), the control organ maps the influence landscape (who is driving what), the epistemic organ (the next-surprise machine) maps the organism's own ignorance (where will I be wrong next) — with its own receptor topology, its own mental model, and its own effectors, its loop closed in its own currency. Organs communicate through exactly three shared media: observation-vector words, the surprise stream (one stream, many ledgers), and the metabolic economy. No organ reads another's internals. The organ-discovery rule is receptor discovery one level up: an organ is demanded when conflation strain persists that no existing organ's currency can resolve — when the separating variable is not a missing channel but a missing coordinate system; an organ that resolves no strain is metabolic waste.
*Status: Proposed (the schema is instantiated four times and T127 generalizes to it — each organ is a cartography of one space; the discovery rule is P48's logic generalized and remains untested). Argued: control_organ_requirements.md; NEXT_SURPRISE.md; sections XXXIII and XXXVII of this index.*

---

## XXXIX. The Physics of Vision

**T145. The retinal signal is a transfer-function stamp, not a color.** What arrives is source wave ∘ material transfer function ∘ medium transfer function: reflection restructures the wave — per-band attenuation, phase coherence, scattering, specular/diffuse structure — so composition is in the signal because the physics put it there. RGB is a 3-bin projection that discards exactly this factorization. The visual organ's de-conflated words are the factors separated: material words, geometry words, flow words (viscosity and flow are the temporal deformation of the stamp). Material words are funded cross-modally: billed by predicted contact outcomes (friction, mass, graspability) — vision's words are promissory notes about touch, affordances with a funding mechanism.
*Status: Supported. ORGAN-LEVEL CONFIRMATION (2026-08-08, visual_physics_experiment.py, results/visual_physics_acceptance.json): metamer materials (identical RGB projection, different spectra) separable from the reflected wave at 0.76–0.85 accuracy vs 0.58–0.65 for the RGB projection; contact-outcome prediction from spectral+texture features 0.86–0.92 vs RGB at the majority baseline (RGB adds nothing — the projection is lethal on metamers as designed); the material-less falsifier arm holds (spectral features at/below baseline — material words correctly unfundable where the world contains no material structure). Full-organism funding through evolutionary selection is the follow-on. Argued: wave_materials.py; docs/visual_organ_physics_amendment.md.*

**T146. Depth is one operator at two baselines.** Binocular disparity and self-motion parallax are the same computation — signature-based correspondence plus a known baseline yields geometry — applied at a spatial baseline (two eyes) and a temporal baseline (self-motion, with the efference copy supplying the known baseline). Depth perception is not a module; and the cognitive organ's efference copy is a load-bearing input to the visual organ's geometry — inter-organ coupling through words, per T144.
*Status: Partially tested. The operator is implemented once and serves both baselines (visual_pattern_library.py match_blobs/edge_disparity); the predicted quality ordering held at organ level: stereo 0.79 > parallax 0.34 > mono-static 0.32 (corr with true depth). Caveats recorded: the parallax margin over the static cue is thin; parallax required an accumulated motion baseline (single-step drowns in edge quantization) and is corrupted by OBJECT motion — the moving-object confound is itself a receptor opportunity (object motion = residual after self-motion parallax accounting). Edge-based disparity with fov-clipping exclusion was empirically forced (center-based estimates collapse on wide blobs). Evolutionary one-eye/efference-ablation arms pending. Argued: results/visual_physics_acceptance.json.*

**T147. In the visual Umwelt, identity is the signature, not the location.** A point's material fingerprint (its E-profile — how it responds to the organism's processing programs) is invariant under self-caused transformation while its position is not. Tracking, correspondence, and object permanence are one commitment: same signature across views = same point in the world. Identity is the invariant under the transformations the organism can cause — structural invariance made operational.
*Status: Partially tested. Signature-based correspondence reached 97% match correctness — but ONLY after per-object texture individuation was added: same-material objects were signature-degenerate and the operator cross-matched them between eyes, empirically demonstrating the theory's own premise (identity requires unique signatures; where signatures degenerate, identity fails — the falsifier observed from the inside). Tracking, losses, and re-acquisitions all live. The signature-degenerate-world falsifier arm as a dedicated condition is pending. Argued: visual_pattern_library.py; results/visual_physics_acceptance.json.*

**T148. The object-agnostic pattern library.** Temporal signatures — blinking, color change, pulsing — detach from the objects that carry them and form a library applicable across instances; object similarity is shared pattern dynamics, not shared appearance. This is the pattern layer applied to visual words, and it supplies the capability the control organ's signature detection declared as its prerequisite: type-level models (threat types, ally types) require patterns detached from individuals.
*Status: Partially tested. The library fills with distinct object-agnostic keys (29–34 patterns from blink/pulse/band-shift dynamics), pattern periodicity is recoverable from the encoded stream (period 12 estimated as 11), and generalization events (known pattern, new carrier) occur ~146/run. Forecast tightening (novelty precedes prediction improvement) observed in some runs, not all — inconclusive at current forecast counts (~50–70). The control-organ consumption of the shared library is the pending arm. Argued: visual_pattern_library.py; results/visual_physics_acceptance.json.*

---

## XL. Compartmentalization, Imagination, and Identity

**T149. Compartmentalization unlocks unbounded imagination without identity drift — the creativity capacity of a system is bounded by its identity risk per hypothesis, and provenance-gated influence takes that risk to zero.** In a monolithic substrate (an LLM's weight space), beliefs and hypotheticals are the same kind of object: there is no typed boundary between entertained and endorsed. Roleplay becomes jailbreak because pretending-to-be-X and being-X are the same computation; training on self-generated content drifts the model because imagined output feeds the substrate that constitutes it. Scaling amplifies imagination and leak in the same proportion — the entanglement is substrate-typed, not capacity-limited, so no scale solves it. Such systems face a real tradeoff: throttle imagination or accept identity drift.

The mechanism that dissolves the tradeoff is the architecture's original requirement — compartmentalization without encryption (influence-control, not access-control) — applied to the self: identity is addressable (certainty ledger, receptor topology, experience log), all three have provenance-gated writes (lived entries only), and imagination's write path terminates at the environment boundary. Imagination may therefore run at unlimited wildness — simulate betrayal, rehearse catastrophe, design the maximally surprising event — because its writes structurally cannot reach identity. The safety property and the creativity property are the same property: compartmentalization is what makes more imagination *affordable*.

The firewall does not freeze identity — it forces identity change to route through the world: the generator schedules the storm, the organism lives the storm, and the lived storm changes it. Blocked is only the shortcut (imagination → identity directly, without reality's countersignature) — the difference between growth and drift. The failure cases are already named and instrumented: imagination updating certainty is private ideology (T113/T136), its social instance is paranoia, and its measurable signature is D_cal drift. Biological dreaming — maximal imagination behind motor paralysis and memory gating — is the same architecture, selected.

Historical note: this is the program's root requirement paying itself back. The pre-sketch driving problem was compartmentalization without encryption; that requirement forced addressable knowledge, which forced grounding, which forced receptors (grounding is downstream of compartmentalization). T149 states the return: the same structure that made influence auditable makes imagination safe at any volume.
*Status: Proposed. Test (P46(c) extended with a dose-response axis): scale generation budget in firewalled vs weakly-firewalled arms — the firewalled system tolerates arbitrary imagination volume with flat D_cal; the leaky one drifts in proportion to imagination volume. Falsification: drift in the firewalled arm proportional to generation budget refutes the mechanism; NO drift in the leaky arm at high volume refutes the risk claim. Argued: NEXT_SURPRISE.md Sec 7; next_surprise_control_organ_amendments.md Sec 4; whitepaper Sec 9 ("What This Solves That Scaling Cannot").*

---

## XLI. The Novel Synthesis Claim

**T63. No existing research program unifies grounded cognition, active inference, and evolutionary receptor topology into a single generative mechanism.** ABI is a novel synthesis across Barsalou, Friston, Gibson, and developmental/evolutionary perspectives.
*Status: Proposed. Argued: whitepaper Section 10.*

---

## Summary

| Status | Count |
|---|---|
| Foundation | 4 |
| Proposed | 93 |
| Partially tested | 20 |
| Supported | 28 |
| Inconclusive | 1 |
| Revised | 1 |
| Falsified | 1 |
| **Total** | **148** |

Twenty-one claims have direct experimental support from ERTI: T27-T31 (evolutionary dynamics), T57 (annealing), T82 (heritable evaluation), T94 (convergence result), T95 (trunk/canopy), T96 (contextual signal interpretation), T97-T98 (anxiety loop architecture and shortcut break), T100 (motor store non-heritability), T104 (metabolic knapsack), T107 (teaching as receptor induction), T126 (concepts as environmental structure — crystallization experiment), T129 (algorithmic gradient — structural learning in growing eyes v2), T130 (visual organ — +1283.5 fitness in v3), T131 (crystallization theory — 2×2 experiment confirmed both predictions). T26 revised after decomposition. T55 directionally falsified. T116 revised: attention = which colors resolve conflations, confirmed by IC-1. Three foundation theories: T114 (theory generation through receptor topology), T126 (concepts as environmental structure — now also empirically confirmed), T127 (Cartographical Theory — empirically confirmed). The Visual Organ section (T130-T132) applies the ERTI architecture to image processing: vision outputs de-conflated words, grouping rules crystallize from the environment, the full depth of perception is environment-sculpted. The crystallization theory (T131) extends FEP by providing a theory of concept genesis — where the variables in the generative model come from.

Sections XXXV–XXXIX (T134–T148) record the surprise economy and the organ federation. The Surprise Economy (T134–T139): the next-surprise machine as the meta-receptor tier unified (T134), its generator as T125's controller on both ledgers (T135), the lived-only firewall theorem with paranoia as its social instance (T136), the grammar of surprise with corpus-depth funding partially confirmed (T137 — the hazard gate flipped with corpus scale alone), rank-rarity as the only surviving surprisal construction (T138, Supported — three absolute constructions failed acceptance with identified mechanisms), and the young-ledger law (T139, Supported — organs earn their voice). Attributed Surprise (T140–T142) awaits the control organ's informative attribution; the adversarial regime (T141) bounds learnable-surprise objectives in social environments. The Control Organ (T143, Partially tested): fractionation demonstrated at AUC 0.786 multi-seed — attribution recovered by conditional integration that no single observable carries — while the discovery-yield transfer (P48) does not replicate at generic measurement and awaits its refined test. The Organ Federation (T144) generalizes T127: each organ is a cartography of one space, coupled only through words, the surprise stream, and the metabolic economy, with organ discovery as receptor discovery one level up. The Physics of Vision (T145–T148): the transfer-function stamp replacing RGB, depth as one correspondence operator at two baselines with efference as the self-motion baseline, identity as signature, and the object-agnostic pattern library that supplies the control organ's declared prerequisite. T149 (Compartmentalization, Imagination, and Identity) closes the loop with the program's pre-sketch root requirement: provenance-gated influence — compartmentalization without encryption — is simultaneously the auditability mechanism and the mechanism that makes unbounded imagination affordable without identity drift; creativity capacity is bounded by identity risk per hypothesis, and the firewall takes that risk to zero while forcing all genuine self-change to route through the world.

---

*The theories listed here are intended to be falsifiable. When predictions fail, the theory is revised or rejected — not defended. Falsification is the goal, not confirmation.*

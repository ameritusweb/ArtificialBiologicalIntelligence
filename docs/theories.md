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
*Status: Foundation. Extends T4 (intelligence as adequacy) and T5 (evolving input structure). Argued: theories.md.*

**T127. The Cartographical Theory: understanding is mapping the processing space.** The organism understands something by discovering the space of possible processing orders for it — how many stages, what sequence, what predictions between stages. Each ordering produces a different prediction profile. The collection of all profiles — which orderings predict well, which predict badly, which cluster together, which are independent — IS a structural map of the input. The asymmetries reveal causal direction (pitch constrains phonemes but not vice versa). The clusters reveal coupled variables (light and temperature in the sun-moon cycle). The independent axes reveal factorization (CO2 and light in photosynthesis). This mechanism is domain-independent: the same substrate applied to any input — mother's voice, lunar cycles, images, social dynamics — produces a structural map of that input. The organism's intelligence is the richness of its processing space map. More receptors add dimensions to the processing space, revealing more structure about anything the organism encounters. The serialization thesis (T8-T11) describes one point in the processing space. This theory says understanding is the whole space.
**T129. Algorithmic gradient: structural diff as the optimization direction.** The tree diff between two programs is a discrete analog of a gradient — it tells you what changed, where, and how. By recording (structural_change, metric_delta) pairs in the mental model, the organism builds an explicit, persistent, queryable record of how program-space changes affect survival metrics. This IS intuition: pattern-matching against a history of structural changes and their consequences. The organism generalizes across changes ("increasing radius at depth 2 hurts in predator situations"), develops second-order intuition ("changes in this region of program space reliably produce this kind of response"), and plans in program space and action space simultaneously through the same predict_delta mechanism. Metacognition with teeth — reflecting on the actual causal structure of how its own learning works.
*Status: Proposed. The mental model stores (state, structural_change) → (metric_delta, certainty) using the same schema as (state, action) → (receptor_change, certainty). Same certainty mechanism, same curiosity, same conflation detection. Argued: theories.md.*

**T128. Epistemic impermanence: the organism learns the rhythm of representational drift.** If visual feature representations shift periodically (freeze-unfreeze developmental cycles), the organism learns that its visual causal chains have a half-life. Curiosity begins firing BEFORE prediction accuracy drops — because the organism has learned from experience that what it currently sees reliably will eventually stop being reliable. This is a meta-receptor: not detecting drift directly, but learning the temporal pattern of drift from survival experience. Biological visual systems have critical periods for the same reason — stability windows exist so downstream learning can consolidate before the next representational shift. The organism holds its visual knowledge with appropriate uncertainty not because it was told to, but because it experienced drift cycles.
*Status: Proposed. Testable via freeze-unfreeze protocol on visual trunk. Prediction: after K cycles, curiosity fires anticipatorily before trunk unfreezes. Argued: theories.md.*

*Status: Foundation. EMPIRICALLY CONFIRMED: (1) Voice differentiation through E-profiles (mother vs father diff=0.459, P8 frequency|diff differentiates most). (2) Cartography experiment: evolved processing orders beat fixed in all 3 environments (predator +913, food_color +1041, social_voice +1019). Social correctly selected phase_first. (3) Staged pipeline evolution: evolved removal weights beat fixed (predator +241, social +439). Different environments evolve different removal strategies. (4) Geometry > prediction: eigen-only (3,339) beats eigen+model (3,236), mapping store redundant. Unifies T8-T11, T122, T125, T126. Provides the mechanism that makes FEP's objective achievable without local minima traps. Argued: docs/cartographical_theory.md.*

---

## XXXIII. The Novel Synthesis Claim

**T63. No existing research program unifies grounded cognition, active inference, and evolutionary receptor topology into a single generative mechanism.** ABI is a novel synthesis across Barsalou, Friston, Gibson, and developmental/evolutionary perspectives.
*Status: Proposed. Argued: whitepaper Section 10.*

---

## Summary

| Status | Count |
|---|---|
| Foundation | 3 |
| Proposed | 87 |
| Partially tested | 14 |
| Supported | 21 |
| Inconclusive | 1 |
| Revised | 1 |
| Falsified | 1 |
| **Total** | **128** |

Fifteen claims have direct experimental support from ERTI: T27-T31 (evolutionary dynamics), T57 (annealing), T82 (heritable evaluation), T94 (convergence result), T95 (trunk/canopy), T96 (contextual signal interpretation), T97-T98 (anxiety loop architecture and shortcut break), T100 (motor store non-heritability), T104 (metabolic knapsack), T107 (teaching as receptor induction). T26 revised after decomposition. T55 directionally falsified. T116 revised: attention = which colors resolve conflations, confirmed by IC-1 (cap_73 broke loop later than cap_10). Three foundation theories: T114 (theory generation through receptor topology), T126 (concepts as environmental structure), T127 (Cartographical Theory — empirically confirmed). The Infinite Color section (T114-T124) unifies theory generation, attention, language, and visual/auditory processing. The Automatic Receptor Discovery section (T125) provides the mechanism for self-refining perception. The Cartographical Theory (T127) — confirmed by voice differentiation, cartography experiment (evolved > fixed in all 3 environments), staged pipeline evolution (different environments evolve different removal strategies), and geometry > prediction (eigen-only beats eigen+model). The no-transformer result (eigen coder surpassed transformer at 1.11 ratio) confirms the policy IS the mental model (T114). The geometry carries more signal than explicit prediction — the mapping store is redundant.

---

*The theories listed here are intended to be falsifiable. When predictions fail, the theory is revised or rejected — not defended. Falsification is the goal, not confirmation.*

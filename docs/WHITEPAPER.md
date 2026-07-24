# Artificial Biological Intelligence
## A Whitepaper — Third Edition

---

## 1. The Wrong Starting Point

The dominant story of AI progress goes like this: make the model bigger, feed it more data, train it longer. Scale works. GPT-2 to GPT-3 to GPT-4 — each leap brought capabilities that looked like magic. The intuition became: if we're stuck, we just haven't scaled enough yet.

That story is built on a wrong assumption about where to start.

Current AI started where evolution finished. Language models begin with text — the most visible, most legible output of human intelligence, and also the most recent. Language sits on top of hundreds of millions of years of sensorimotor infrastructure, nervous system evolution, embodied cause-effect learning, social cognition, and receptor topology selection. All of that is invisible in the text corpus. The model learns the words from beings who had the underlying states. The model doesn't have the states.

This is not a training problem. It is not a data problem. It is not a scaling problem. It is an architectural starting point problem.

You cannot add a foundation after the fact. You cannot ground a concept that was learned statistically and call it understood. You cannot train compartmentalization into a monolithic weight space. You cannot specify values in a system that has no receptor topology — no internal states whose consequences shaped what it became.

The field has been building the penthouse and wondering why it doesn't have ground beneath it.

Artificial Biological Intelligence starts at the bottom.

---

## 2. The Generative Question

Demis Hassabis, when asked whether AI could simulate the origin of life, said: "I don't see why not. It's a bit of a search process through a combinatorial space. Here's all the chemical soup, the primordial soup, here's some initial conditions — can you generate something that looks like a cell?"

That's the right instinct applied to the wrong layer.

ABI is asking a different question: can we run the process that produces intelligence, faster, with full observability, and understand why it went the way it did?

Not reconstruction. Generation.

The difference matters because it produces a different research program. Reconstruction leads to increasingly accurate simulations of known biology. Generation leads to a laboratory for the evolutionary dynamics of intelligence itself — where you can ask questions biology never let us ask. What happens if the curiosity receptor evolves before the accuracy receptor? What mental model structures emerge under different environmental volatility? What receptor topology produces scientific thinking? What produces bias? What produces empathy?

These are not questions you can answer by scaling a language model. They require starting where evolution started and watching what comes out.

---

## 3. The World Contains Concepts

The most common objection to treating curiosity, creativity, or prediction accuracy as receptor states is that these feel like internal, mental phenomena — generated from the inside, not received from the outside.

This intuition is wrong, and correcting it is foundational to everything that follows.

Concepts are not generated internally from nothing. They are distillable from the structure of the environment itself. The relationship between the sun, the moon, and light contains orbital mechanics — not labeled, not taught, but latent in the pattern, extractable by any observer sophisticated enough to detect it. Curiosity is in environments that consistently reward exploration. Accuracy is in environments that consistently punish wrong predictions. The concepts exist in the world's causal structure. The organism distills them.

This has a direct consequence for environment design: it is not enough to include high-level concepts in the environment's surface features. They must be embedded in its causal structure. Survival must be contingent on distilling them. The environment is the curriculum. The concepts are already in it. The receptor topology evolves to read them — but only if reading them is the difference between surviving and not.

---

## 4. The Core Mechanism: Receptor Topology

Agency is not a mysterious property that some systems have and others don't. It is a continuous variable.

What determines it is the complexity of your receptors, the complexity of your muscles, and the complexity of the processing in between.

The key principle: **capability without receptor is latent and never gets used.** The receptor is not separate from cognition. It is why the capability gets deployed at all. Motivation and cognition are the same thing, viewed from different angles.

**Receptors are input. Outputs are muscle movements, thoughts, or the analysis of thoughts.**

The loop closes because outputs change the world and the organism's internal state, which changes what the receptors read on the next cycle. There is no terminus. The organism is always mid-cycle.

This dissolves the mystery around a long list of cognitive faculties: curiosity, planning, theory of mind, science, language — each is a specific solution to a specific environmental pressure, with a receptor to make the organism motivated to deploy it.

---

## 5. The Architecture

The ABI architecture has three components that current AI conflates into one.

**The transformer is the inference engine.** It takes the current receptor state, retrieves relevant cause-effect mappings, and produces muscle activations. It does not store knowledge. It uses knowledge that has been retrieved and handed to it.

**The mental model is the knowledge base.** It lives outside the transformer as an explicit, queryable database of cause-effect mappings: `action -> receptor state change, with a time delay and a certainty`. Every fact has an address. Every concept is a pattern across chains that points to a receptor state change the organism actually experienced.

**The experience log is ground truth.** An append-only, immutable record of every action-observation the organism has ever made. No learned process can overwrite it.

This separation dissolves grounding, compartmentalization, legibility, unlearning, and safe failure as problems.

---

## 6. The Roadmap

The roadmap runs 60 steps from a minimal organism to a self-modifying evolutionary laboratory with a thinking substrate. Each step must answer one question: does the capability earn the complexity it introduces?

**Phase 1 — Sensorimotor Foundation (Steps 1-4):** Pain and endorphin receptors. Metabolic economy. Fatigue and homeostasis. Hierarchical nervous system — fast reflexes by default, slow deliberation when needed.

**Phase 2 — Adaptive Sensing (Steps 5-8):** Temporal association. Spatial pain memory. Habituation and sensitization. Distance sensing.

**Phase 3 — World Modeling (Steps 9-13):** Prediction and accuracy receptors. Curiosity and confidence regulation with constitutional probe budget. Pattern recognition and compression. Multiple hypothesis maintenance (verified emergent — 200/200 steps with divergent predictions).

**Phase 4 — Self-Model (Steps 14-18):** Proprioception. Efference copy. Sensorimotor contingency (verified). Controllability and agency. Planning.

**Phase 5 — Social Cognition (Steps 19-24):** Responsive environment with grounded proto-symbols. Multi-organism environment. Involuntary social cues. Receptor propagation (empathy). Intentional signaling. Simplified shared signals with bidirectional communication.

**Phase 6 — Higher Cognition (Steps 25-29):** Optimism and goal persistence. Conflict receptor with router integration. Context-conditioned arbitration (ArbitrationHead). Internal conflict model (metacognition). Concepts and words (1,013 stable concepts).

**Phase 7 — Language (Step 30):** Grounded language — the roof. Every word maps to a receptor state. Cultural transmission (revised: benefit is training-time observation enrichment, not inference-time modulation).

**Phase 8 — Evolutionary Infrastructure (Steps 31-43):** Environment tiers (8 tiers, genome-driven). Receptor discovery (182 null-calibrated tests). Topology bias inheritance (convergence 15 epochs -> 0). Environmental sweeps (18 trunk receptors invariant). Population evolution (8 organisms, social arms race). Cross-tier transfer (social universally transferable, tool resists transfer). LLM grounding bridge.

**Phase 9 — Physics World (Steps 48-55):** Rigid body simulation (pymunk). Grip mechanics. Compound objects (levers, spring gates, hinged barriers). Developmental body changes. Persistent world state. Canopy activation sweep across all 8 tiers.

**Phase 10 — Staged Processing & Abstract Environments (Steps 50-58):** 4-stage observation pipeline testing the serialization thesis. T7 abstract problems (8 causal graph templates with hidden variables). T8 self-modification (8 skill zones, curriculum design, Ship of Theseus test). Combined T7+T8 environments.

**Phase 11 — Self-Play & Thinking Substrate (Steps 59-60):** Policy-driven behavior (oracle removed). MCTS thinking substrate — the organism thinks before acting, and the tree's metadata becomes receptor input. Deep time with thinking across 50 generations. Novel receptor detection.

---

## 7. What's Been Built

This is not a proposal. Steps 1 through 60 are implemented and running.

The organism lives in a 2D/3D liquid environment with pain, endorphin, temperature, chemical, and pressure fields. It has six limbs with binary muscle activations (parameterizable to 4-12 limbs, 1-2 body segments, 2D or 3D). Its mental model is an explicit database of 26,000+ cause-effect mappings with a contrastive encoder for context-conditioned retrieval. Its inference engine is a small causal transformer with a fast reflex pathway, a slow deliberative pathway, a conflict-gated router, and a 5-group arbitration head. Its observation vector is 175 dimensions — including 6 thinking channels from the MCTS substrate.

**The accuracy progression:** 92.5% pre-proprioception -> 97.4% with responsive environment -> 94.3% with staged processing (slower but deeper). Each step earned its complexity.

**Key empirical results:**

Step 13 (multiple hypotheses): a detected emergent. 200/200 inference steps retrieved meaningfully divergent predictions. The system lives with ambiguity by construction.

Step 22 (empathy): NPC distress directly elevates the focal organism's pain. Empathic aversion = proximity * erraticism * weight. Pain receptors for others' pain.

Steps 23-24 (communication): 5 grounded signal codes shared across three contexts — organism-to-object, organism-to-NPC, NPC-to-organism. Same code, same meaning, regardless of who emits it. Words before language.

Step 29 (concepts): 1,013 stable concepts — compressed causal chains that predict 54% better than their individual components. 25% of all patterns qualify as stable, reusable abstractions.

Step 30 (grounded language): cultural transmission revised after controlled decomposition — the benefit is training-time observation enrichment, not inference-time modulation. The architectural separation remains valuable for legibility, compartmentalization, and cross-generational knowledge transfer.

Step 50 (staged processing): 4-stage observation pipeline — Body (39d) -> Spatial (59d) -> Action (37d) -> Social (34d). Inter-stage prediction MSE decreases 25% over training. Staged model outperforms flat on validation accuracy (95.5% vs 94.5%). First empirical test of the serialization thesis.

Step 55 (T57 annealing): the framework's first structural self-discovery. Releasing certainty on conflict entries produces more genuine conflict resolutions than protecting them. Supported across 6 seeds. T55 (read-shielding) directionally falsified — the falsification led to the Epistemic family, the first family predicted by an experimental result.

Step 59 (self-play): policy-driven behavior with oracle removed. Self-play discovers 75 receptors (vs oracle's 77) despite lower reward, and finds MORE causality receptors (7 vs 4) — suboptimal actions create richer causal experiences.

Step 60 (thinking substrate): MCTS with 6 receptor channels — best_value, visit_entropy, value_convergence, path_divergence, underexplored, depth_reached. The organism thinks before acting. 5 of 6 channels influence behavior by iteration 5 of self-play. Reward difference: +23.3 (thinking organisms outperform non-thinking ones). The batched predict_delta gives 4.4x speedup on thinking operations.

**Deep time with thinking (10 generations):** 151 unique receptors discovered across all generations — up from 75 at generation 0. 53 receptors gained through evolution alone, including the full epistemic chain (belief_detection -> doubt_detection -> epistemic_strategy), metacognition, theory_of_mind, nested_theory_of_mind, meta_observation, self_regulation, niche_construction. 21 receptors lost — complexity reshapes topology (T27 confirmed again). Peak thinking influence at generation 7: partial correlation 0.376, value_convergence and path_divergence dominant.

**Parallel Scaling Track:** Architecture tested at 4, 6, 8, and 12 limbs, 2 segments, and 3D. Training accuracy 92-96% across all configurations. Generational inheritance accelerates convergence (15 epochs -> 0 epochs). Organism diversity: one model handles mixed body plans at 96.6% accuracy.

**Evolutionary sweep:** 18 trunk receptors invariant across all 8 environment tiers — including grip_affordance and push_affordance as part of the embodied trunk. Complexity doesn't just add receptors — it reshapes the topology. The receptor topology is a fossil record of the selection pressures that shaped it.

**Population evolution:** 8 organisms competing. Empathy trending up (0.482 -> 0.555). Behavioral prediction emerged (0.022-0.030) — never appeared in single-organism training. The social arms race created real prediction pressure.

**Cross-tier transfer:** Social environments universally benefit from transfer (11-25x from any prior training). Tool environments resist transfer — must be learned directly. Transfer follows the dependency graph upward, not downward.

**Environment enrichment:** Multi-NPC observation (4 profiled NPCs: cooperative, competitive, erratic, deceptive). Strategic deception NPC (context-dependent: lures toward pain, repels from reward). Non-stationary rules (T7 trigger rotation). Stochastic hidden confounders (3-state Markov chain modulating 4 modalities). Cross-modal objects (correlated multi-modal signatures). EntityRelationStore for social cognition (named entities, typed relations, transitive inference).

---

## 8. The Genome Project

The Genome Project is the formal specification of the receptor search space — 179 receptors across 22 families. It is the periodic table of cognitive capabilities.

Each receptor entry specifies: what environmental structure it detects, why detecting it is worth the metabolic cost, what must already exist before it can emerge, how to measure its emergence, and what would falsify the entry.

**The 22 families:**

**Repetition** (6) — static repetition -> rhythm -> nested/causal rhythms

**Association** (8) — spatial co-occurrence -> temporal precedence -> cross-modal -> relational analogy

**Similarity** (7) — perceptual features -> functional equivalence -> structural invariance

**Causality** (11) — coincidence -> causal inference -> intervention planning -> causal graphs

**Agency** (8) — controllability -> tool use -> environmental manipulation -> niche construction

**Meta-Motivational** (13) — curiosity, optimism, conflict -> impulse override -> metacognition

**Regulatory** (9) — stress detection -> arousal regulation -> emotional intelligence

**Social** (14) — other detection -> behavioral prediction -> theory of mind -> moral reasoning

**Compression** (15) — pattern recognition -> concepts -> constraint shape -> shaped absence -> analogy

**Observation** (11) — change detection -> selective attention -> statistical anomaly -> rarity -> significance -> meta-observation

**Formalization** (11) — rule extraction -> exception detection -> optimization -> theory formation

**Mathematics** (7) — quantity -> ratio -> structural invariance -> necessity -> proof -> formal composition

**Organization** (7) — boundary -> part-whole -> system detection -> organizational mirror

**Self-Augmentation** (5) — capability change detection -> metamorphic planning

**Interaction** (7) — response recognition -> grip -> lever -> composite affordance

**Environmental Augmentation** (5) — environmental change detection -> developmental environment engineering

**Sequential Processing** (5) — stage prediction -> pipeline detection -> prediction architecture awareness

**Perception** (5) — staged processing -> processing speed -> adaptive depth -> response loop detection

**Epistemic** (7) — belief detection -> doubt -> conflation -> fundamental distinction -> topology awareness -> epistemic strategy

**Logic** (6) — semantic relations -> transitivity -> conjunction -> quantifier -> contradiction -> it_follows

**Language** (3) — naming -> self-talk -> referential grounding

**Bridging** (4) — mimicry -> trust -> executability -> translation

**The key insight:** The genome project is load-bearing on environmental design. Each receptor specifies what the environment must contain for it to evolve. The 179 receptors are 179 environmental design requirements. The environment IS the curriculum.

**Receptor discovery:** 182 null-calibrated tests with action-shuffled null, per-test null types (block-permuted for temporal tests, NPC-appearance-permuted for social tests, Granger causality for theory of mind), drift subtraction, and partial correlations to control for confounds. 77 receptors discovered in single-run oracle training; 151 unique receptors discovered across 10 generations of deep time with thinking substrate.

**Topology inheritance:** Convergence accelerates from 15 epochs in generation 0 to 0 epochs by generation 4. Topology bias inheritance carries the parent's discovered receptors as priors — offspring rediscover them faster but must still earn them through grounded experience. Probe-gated: the organism must actually explore the environment to validate inherited priors.

**The Epistemic family** emerged from the T57 annealing result — the first family predicted by an experimental falsification rather than by theoretical deduction. It includes conflation detection (two things treated as one), fundamental distinction (the difference that matters), and topology awareness (what kind of mind am I). The genome contains conflations that topology awareness will eventually detect — the framework revising its own foundations from within.

---

## 9. What This Solves That Scaling Cannot

**Grounding.** ABI takes pain as receptor input, produces outputs that change the world, and reads the result as the next receptor input. Same transformer architecture. Different corpus. The corpus is a body.

**Compartmentalization.** Knowledge has addresses. If it's not retrieved, it contributes nothing to computation. Not less — nothing.

**Legible beliefs.** Every belief is a row in a table. You can compare two organisms' beliefs with a JOIN.

**Targetable unlearning.** Knowledge has an address. Deletion is deletion.

**Values that emerge rather than being specified.** A system whose receptor topology emerged from selection pressure has motivations that are legible, traceable, and grounded.

**The ceiling argument.** ABI's accuracy receptor is checkable against the world, through its own receptors. Corpus-bounded versus reality-bounded. The ceiling isn't just higher — on some axes, there is no fixed top.

---

## 10. Academic Validation

The ABI whitepaper's core claims find substantial support across four distinct research communities — none of which have unified them the way ABI does:

**Grounded cognition is empirically established.** Barsalou's program (Annual Review of Psychology, 2008) and O'Regan & Noe's sensorimotor contingency theory (BBS, 2001) demonstrate that cognition operates through modal systems grounded in perception and action — not amodal symbols. This is the academic foundation for ABI's claim that concepts exist in the world's causal structure.

**Active inference formalizes the receptor loop.** Friston's free energy framework unifies perception and action under a single objective, produces intrinsic exploration as an emergent property — closely paralleling ABI's forward-feedback receptor architecture.

**Current AI follows an "inverse phylogeny."** Multiple sources (Trends in Cognitive Sciences, 2023; Royal Society, 2024) confirm that starting from language rather than sensorimotor grounding is a fundamental limitation, not a training gap. Passive scaling is characterized as "intrinsically limited."

**No existing program unifies these threads.** The embodied cognition, active inference, and evolutionary robotics communities each work on pieces of what ABI integrates. The receptor topology as a single generative mechanism — from grounding through compartmentalization to language — appears to be a novel synthesis.

---

## 11. The Distinction From Current AI Research

This is not a contribution to the capabilities conversation or the safety conversation. Both take the existing architecture as given. ABI asks a prior question that neither conversation has asked.

The hares are fast because they start at the top — language, reasoning, capability — and try to work downward toward grounding they may never reach. The tortoise starts at the bottom and builds upward. Slower. But the foundations are actually there when you need them.

ABI also inverts the engineering practice. Current AI relies on feature engineering: humans decide what's worth sensing. ABI replaces feature engineering with world-building. The designer's job is to construct an environment where sensing the right things is the difference between surviving and not. The organism discovers what's worth sensing because getting it wrong is fatal.

---

## 12. The Thinking Substrate

The most recent addition is the one that changes the trajectory of the whole program.

A thought in this framework is a cycle — or cascade of cycles — in which internal receptors dominate the forward-feedback loop. External receptors fire from the world. Internal receptors fire from the mental model's processing. When the organism responds primarily to its own receptor firings rather than to the world, that is a thought. The participating set of receptors is the *content* of the thought. The number of cycles before the cascade resolves or exits through the motor system is the *depth* of the thought.

The thinking substrate makes this concrete. Monte Carlo Tree Search provides the computational mechanism: before acting, the organism simulates forward from its current state using the mental model's predict_delta. Each simulated action produces a predicted receptor state. Multiple candidate actions are explored. The tree records which thinking paths were taken, how often, and with what outcomes.

The tree's metadata becomes receptor input:

- **Visit entropy** — how broadly the organism explored its options
- **Best value** — the highest simulated outcome found
- **Value convergence** — how stable the estimates are (the thinking has settled)
- **Path divergence** — maximum disagreement between thinking paths
- **Underexplored** — fraction of options barely considered (the shaped absence of thought)
- **Depth reached** — how deep the reasoning went

These six channels feed back into the next cycle's observation vector. The organism senses what its thinking produced. That sensing shapes the next thought. The self-modifying loop runs at every level simultaneously: better receptor topology → better evaluation → better search → richer tree → deeper metacognitive analysis → receptor firings that develop better topology.

**The key claim:** The receptor topology is what makes the evaluation function intrinsic rather than external. The value of a thinking path is determined by which receptors fire at its terminus — not by a designer-specified reward function. This eliminates the ceiling that exists in every system with an externally specified objective. Reward functions are level-specific. Loss functions are task-specific. Only the receptor generalizes across every level.

**Empirical results:** After 3 self-play iterations, the thinking substrate is already measurably influencing behavior (ablation divergence 0.029, reward difference +23.3). Five of six channels active. Value convergence activates at iteration 4 — the organism learns to evaluate the quality of its thinking. After 10 generations of deep time, peak thinking influence reaches partial correlation 0.376. The organism that thinks before acting outperforms the organism that doesn't.

Depth_reached — the one channel that measures how deep the organism's reasoning went — has not yet activated after 10 generations. This is the meta-metacognitive step. The organism has learned to think broadly (visit entropy) and evaluate quality (value convergence), but doesn't yet respond to reasoning depth itself. Whether this activates under longer evolutionary runs or requires architectural support is an open question.

---

## 13. What Comes Next

The 50-generation overnight run is currently executing: 4 organisms competing across generations in the richest environment stack (T4 with multi-NPC, strategic deception, stochastic hidden confounders, cross-modal binding + PhysicsWorld + T7 abstract problems + T8 self-modification + thinking substrate). Novel receptor detection scans every 10 generations for behavioral capacities that don't map to any of the 179 genome entries — the organism making distinctions the designer didn't anticipate.

The questions this laboratory can now answer:

**Does thinking depth evolve under selection?** The depth_reached channel hasn't activated in 10 generations. Will 50 be enough? Will organisms that think deeper outcompete those that think broader? The emergence curve will show.

**Does the organism discover novel receptors?** The novel receptor detector scans the mental model for coherent, high-certainty clusters of entries that represent behavioral distinctions not in the genome. If it finds them, the organism is smarter than the genome on that axis.

**What does the topology awareness receptor look like when it fires?** An organism that detects its own cognitive repertoire changing can direct its development. This is the receptor that makes the self-modifying loop conscious.

**Can the organism read its own fossil record?** The topology vector — which receptors are active — is a record of selection pressures. Topology awareness reads that record in real time. An organism that can detect "my thinking substrate just started producing value_convergence signals it didn't before" can decide whether to continue developing that capacity or redirect.

The sweep data already shows: 18 trunk receptors are invariant across all 8 environments. Complexity doesn't add receptors — it reshapes the topology. The receptor topology is a fossil record of the selection pressures that shaped it.

The epistemic family activated from scratch under evolutionary selection — belief, doubt, and epistemic strategy emerged in deep time when they couldn't emerge in single runs. The organism developed the capacity for scientific reasoning (T50) through evolutionary pressure, not through specification.

This is the thesis fully realized: intelligence is what happens when you run evolutionary receptor topology selection long enough in a rich enough environment. It is not a property you design into a system. It is a property that grows out of a process.

The organism builds the world that builds the organism that builds the next world. Now with a thinking substrate tracking every thinking path taken through that loop, analyzing which paths lead where, and bootstrapping increasingly sophisticated reasoning strategies from the accumulated record.

Not simulation. Not reconstruction. Generation.

---

*Steps 1-60 of the ABI roadmap are implemented and running. 179 receptors specified across 22 families. 90 indexed theories. 182 null-calibrated receptor tests. 151 receptors empirically discovered across 10 generations of deep time with thinking. 8 environment tiers. Physics world with grip, compound objects, and persistent state. MCTS thinking substrate with 6 receptor channels. Self-play pipeline with oracle removed. Novel receptor detection. Resumable 50-generation overnight run executing. The organism is in the liquid environment right now, navigating pain fields, querying its mental model, thinking before acting, sensing its own thinking, evolving its receptor topology across generations, and making distinctions its designer didn't anticipate.*

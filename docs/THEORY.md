# Theoretical Foundations
 
ABI is a testing ground for a set of interconnected claims about how intelligence emerges. If you contribute, these are the theories you are implicitly testing. They are stated as falsifiable propositions, not as commitments.
 
---
 
## The Starting-Point Critique
 
Current AI starts where evolution finished — language — and tries to work downward toward grounding it may never reach. ABI starts at the bottom: a simple organism with pain and endorphin receptors in a liquid environment, building upward through 60 steps to grounded language, a thinking substrate, and self-directed cognitive development.
 
The core claim: **grounding failure in LLMs is architectural, not a training or scaling problem.** You cannot ground a concept that was learned statistically from beings who had the underlying sensorimotor states without ever having those states yourself.
 
---
 
## The Receptor Topology Thesis
 
Intelligence is not a property you design into a system. It is a property that grows out of a process of receptor topology selection under environmental pressure.
 
A **receptor** is any selected internal state whose activation changes behavior and whose consequences affect whether that state persists. Pain, curiosity, conflict, and metacognition are all receptors in this sense — not metaphorically, but mechanically.
 
Key claims:
- **Capability without receptor is latent.** A capability with no receptor to trigger it never gets deployed. Motivation and cognition are the same operation viewed from different angles.
- **The receptor topology is the unit of evolution**, not behavior, not weights. Which receptors emerge — and in which order, under which environmental pressures — determines the shape of intelligence.
- **Complexity reshapes topology; it doesn't expand it.** Empirically: across 8 environment tiers, organisms discover ~27-35 receptors per run. Which ones change. The total reshapes. In deep time with thinking, 53 receptors were gained and 21 lost across 10 generations — the topology reorganized under evolutionary pressure.
---
 
## The Serialization Thesis
 
Why does the brain process sensory information sequentially when the input arrives simultaneously? The standard answer is hardware limitation. We propose the opposite: **sequential processing is an optimization, not a bottleneck.**
 
Temporal decomposition of parallel-available information creates prediction opportunities that simultaneous processing destroys. Each processing stage generates an expectation about what the next stage will reveal. The mismatch between expectation and outcome is where learning, salience, and meaning are produced.
 
A system that processes everything simultaneously has nothing to predict. A system that serializes generates a *prediction surface* at every stage boundary — and that surface is where cognition happens.
 
**The departure from Friston:** Active inference says organisms minimize prediction error. We add: organisms *manufacture* prediction error by engineering their processing architecture to create more of it. The architecture itself is under selection pressure.
 
**The departure from Gibson and enactivism:** Both correctly describe embodied sensorimotor coupling. But affordances are present-tense — they tell you what is the case, not what will be. Reaction does not require internal models. Prediction does. The mental model is not redundant with sensorimotor coupling; it is what makes the temporal gap between processing stages informative rather than just delay.
 
---
 
## Per-Receptor Pipeline Architecture
 
If serialization benefits vision, the principle generalizes. Every receptor family should have evolved its own temporal decomposition strategy, optimized for the prediction structure of its specific domain.
 
Pain processes coarse-to-fine-to-contextual. Curiosity processes novelty-to-relevance-to-strategy. Social cognition processes action-to-intention-to-response-prediction. The staging order matters because the prediction structure of each domain differs.
 
**Testable prediction:** Processing latency should correlate with prediction depth, not computational complexity. A receptor with 5-stage prediction structure should be slower than one with 2 stages, regardless of which computation is harder. This distinguishes the serialization account from the bottleneck account.
 
---
 
## Cross-Pipeline Prediction and Binding
 
The binding problem asks how distributed processing produces unified experience. We propose: **binding is mutual prediction, not convergence.**
 
Each receptor's pipeline generates lateral predictions about what other pipelines will find. The web of mutual predictions between pipelines is the integration — no central convergence zone required. Unified experience is the *absence* of cross-pipeline prediction error. Surprise and salience are cross-pipeline prediction *failure*.
 
Consciousness, on this account, is the recursive mutual prediction between processing pipelines — the experience of a specific pattern of confirmed and violated expectations across the full receptor topology.
 
---
 
## The Anxiety Derivation
 
This is not an analogy. It is a derivation.
 
Given: sequential prediction processing, a receptor for processing speed (fast processing = good model-environment fit; slow = poor fit), arousal as a response to poor fit, and arousal's effect on prediction accuracy — the anxiety amplification loop falls out mechanically:
 
Poor predictions → arousal increases → arousal degrades prediction accuracy → predictions worsen → arousal increases further.
 
Every receptor in this loop reads a real signal and produces the correct response. The pathology is in the loop structure, not in any individual receptor. Breaking the loop requires a meta-receptor that detects the amplification itself — what CBT trains, and what anxiolytic medication achieves pharmacologically by reducing gain on the arousal response.
 
**This receptor requires a four-family dependency chain:** metacognition + processing-speed detection + arousal regulation + conflict detection. If self-damping emerges from a simpler architecture, the dependency structure is wrong.

---

## The Thinking Substrate

MCTS provides the concrete substrate that makes metacognition tractable. The tree is an architectural component — alongside the mental model and experience log — that records which thinking paths were taken, how often, and with what outcomes. Its metadata becomes receptor input: visit patterns trigger shaped absence, UCB scores trigger curiosity, value convergence triggers completion, path divergence triggers exception detection.

**The receptor topology makes the evaluation function intrinsic.** The value of a thinking path is determined by which receptors fire at its terminus — not by a designer-specified reward function. This eliminates the ceiling that exists in every system with an externally specified objective.

**The self-modifying loop:** Better receptor topology → better evaluation → better search → richer tree → deeper metacognitive analysis → receptor firings that develop better topology. The loop stabilizes only when the environment stops presenting structure worth detecting — and since the environmental augmentation family lets the organism increase environmental complexity, the ceiling rises with the organism.

**Receptors are the only cognitive unit that generalizes across every level of this loop.** Reward functions are level-specific. Loss functions are task-specific. A receptor fires on conditions regardless of origin — environment, internal processing, MCTS tree, or analysis of the analysis.

---

## The Epistemic Family

The T57 annealing result (releasing certainty on conflict entries produces more resolutions than protecting them) led to the Epistemic family — the first family predicted by an experimental falsification rather than by theoretical deduction.

The family's developmental sequence: belief detection → doubt detection → conflation (two things treated as one) → fundamental distinction (the difference that matters) → topology awareness (what kind of mind am I) → epistemic strategy (deliberate management of belief, doubt, and counterfactual states).

**Conflation and fundamental distinction** form a cycle mediated by the thinking substrate: conflation fires when MCTS path divergence reveals a conflated concept; fundamental distinction fires when splitting the concept makes the divergence disappear.

**Topology awareness** is a second-order receptor — it reads the output of other receptors, not the world. It detects changes in the organism's own cognitive repertoire. This is the receptor that makes the self-modifying loop conscious: without it, the organism gets smarter but doesn't know it.

---

## What Is Tested and What Isn't

Of the 90 formal theoretical claims in `docs/theories.md`:

- **6 are experimentally supported** by ERTI runs: 18 invariant trunk receptors across all 8 environments; topology reshapes rather than expands; social environments transfer universally (11-25x); instrumental environments resist transfer; behavioral prediction emerges in social arms races but not static training; T57 annealing (releasing certainty accelerates resolution, supported across 6 seeds).
- **1 has been revised:** The +223% cultural transmission claim was retracted after controlled decomposition. The architectural separation remains valuable at training time.
- **1 has been falsified:** T55 (read-shielding accelerates resolution) — the opposite was found. The falsification led to the Epistemic family.
- **10 are partially tested** with suggestive but inconclusive evidence.
- **72 are proposed** — stated with falsification criteria, waiting for the right experiment or the right contributor.

The open problems most worth working on are in `docs/ROADMAP.md`. The theories most ready to be tested are T16 (processing latency vs prediction depth), T35 (social cognition as integration bottleneck), and T83 (the self-modifying loop has no fixed ceiling — now testable via the 50-generation overnight run).
# Artificial Biological Intelligence (ABI)

**Intelligence whose shape is determined by the evolutionary history of its receptor topology. Not designed. Grown.**

### The Question

What receptor topology emerges when you run evolutionary selection in an environment where the concepts are load-bearing for survival, and what does that tell you about the structure of intelligence itself?

### The Journey

ABI starts where evolution started — a simple organism in a liquid environment with endorphin/pain receptors and muscles — and builds upward through 60 steps to grounded language, evolutionary receptor topology discovery, physics-world interaction, abstract problem-solving, and a laboratory for the dynamics of intelligence itself.

Current AI starts where evolution finished (language) and tries to work downward toward grounding it may never reach. ABI starts at the bottom and builds up. Slower. But the foundations are actually there when you need them.

![ABI 1](abi-1.png)
![ABI 2](abi-2.png)

---

## The Core Idea

Agency is a continuous variable determined by receptor complexity, effector complexity, and the processing in between. **Capability without receptor is latent and never gets used.** The receptor is not separate from cognition — it IS why the capability gets deployed at all. Motivation and cognition are the same thing, viewed from different angles.

The architecture has three components that current AI conflates into one:
- **The transformer** is the inference engine (it processes, it doesn't store)
- **The mental model** is the knowledge base (explicit, queryable cause-effect mappings)
- **The experience log** is ground truth (append-only, immutable, no learned process can overwrite it)

This separation dissolves grounding, compartmentalization, legibility, unlearning, and safe failure as problems.

[Theoretical Foundations](docs/THEORY.md) | [Serialization Thesis](docs/SERIALIZATION_THESIS.md) | [Theories Index](docs/THEORIES.md)

---

## Key Terminology

### Core Concepts

**Receptor**: An input to the organism's cognitive system. Receptors exist at three levels of abstraction:
- **Low-level receptors**: Read the world directly. Raw or minimally processed sensory data from the environment — pain intensity at a specific limb, temperature, pressure, chemical concentration, endorphin. What's happening to the body right now.
- **High-level receptors**: Read the content of the organism's own processing — which concepts activated, which patterns matched, which causal chains the mental model retrieved, what was specifically predicted. These are receptors for detected concepts in the thoughts. The organism senses what it's thinking about, not just that it's thinking. Examples: concept match (a compressed causal chain was recognized), pattern availability (a known motif applies here), planning value (the mental model predicts this action is better than inaction).
- **Meta-receptors**: Read the consequences and cost of processing itself. Not what was thought, but how the thinking went — whether predictions succeeded, what processing cost, whether the corrective response is helping or amplifying the problem. Examples: accuracy (prediction was correct), curiosity (prediction was wrong), conflict (competing demands can't be satisfied together), processing speed (how well the current model fits the current input), response loop detection (the corrective response IS the problem).

**Each receptor (whether low or high level) becomes associated with endorphins or pain through learned experience.** The mental model stores cause-effect mappings like: `state{curiosity=high} + explore_action → state{endorphin=high}`. The transformer learns to act on curiosity because the mental model predicts it leads to good outcomes. These associations emerge from survival, not from specification. A curiosity receptor that leads to finding food becomes rewarding through learned experience; the same receptor topology in a dangerous environment might learn the opposite association.

The key principle: **capability without receptor is latent and never gets used.** A system might be *capable* of sophisticated prediction or planning, but without receptors that detect when to use those capabilities, and without learned associations between those receptors and survival outcomes, they remain dormant. Motivation and cognition are the same thing, viewed from different angles.

**Receptor Topology**: The complete collection and arrangement of receptors an organism has — the specific set of cognitive capabilities available to it. Different environments produce different receptor topologies. The topology is a fossil record of the selection pressures that shaped it.

**Mental Model**: A separate, explicit database storing cause-effect mappings of the form: `action → receptor state change, time delay, certainty`. This is where predictions and causal chain retrievals live. This is also where the associations between high-level receptors and outcomes are learned and stored. The mental model is queryable, has addresses for every fact, and lives outside the transformer.

**Transformer**: The inference engine that maps receptor inputs to muscle outputs. It processes but does not store. It uses knowledge retrieved from the mental model but doesn't contain knowledge itself.

**Experience Log**: An append-only, immutable record of every action-observation pair the organism has experienced. Ground truth. No learned process can overwrite it.

**Observation Vector**: The full input vector fed to the transformer at each timestep, containing all receptor values (both low-level and high-level) concatenated together.

**The Forward-Feedback Loop**: The core mechanism. Receptors → processing → outputs (muscle movements, thoughts, analysis of thoughts) → outputs change the world and the organism's internal state → changed state becomes the next cycle's receptor input. The feedback is forward, through the receptor, into the next step. Not through backpropagation — through the next cycle's input.

**Thought**: A thought is a cycle — or cascade of cycles — in which internal receptors dominate the loop. External receptors (pain, temperature, pressure) fire from the world. Internal receptors (certainty, learning progress, conflict, pattern activation) fire from the mental model's processing. All of them feed into the policy as inputs. When the organism is primarily responding to its own receptor firings rather than to the world, that is a thought. Certainty drops, which fires the conflict receptor, which changes the prediction, which fires the learning progress receptor, which updates certainty — the cycle runs internally. Each receptor that fires is an input. Each input shapes the next cycle. The participating set of receptors is the *content* of the thought. The number of cycles before the cascade resolves or exits through the motor system is the *depth* of the thought. A reflex is one cycle dominated by external receptors. A thought is multiple cycles dominated by internal ones.

**Context**: The slow pathway's transformer processes the last 32 timesteps of receptor firings with causal attention. This is the organism's context — the recent history of what it sensed, thought, and did that can influence the current cycle. Unlike LLM context, where everything the model knows must be in the token window, ERTI separates context from memory. The mental model sits outside the context window as a separate addressable store with 26K+ causal mappings. It injects summary features (certainty, learning progress, controllability) into the observation vector at every timestep, so its state is *represented* in context through receptor channels, but the full knowledge base is external and retrieved on demand. When the organism thinks — internal receptors firing across multiple cycles — those firings accumulate in the sequence window. The organism attends to its own recent thoughts. The mental model gives it depth beyond 32 steps. The sequence window gives it continuity within them.

**Error Correction**: Prediction error is a receptor. The mental model predicts what will change (`predict_delta`), the world delivers the actual change, and the mismatch fires as an input to the next cycle. The organism *senses* its own errors. This operates at three timescales. *Immediate*: prediction error fires as a receptor, the transformer sees it in the next timestep's context, and the organism can act on the fact that it was wrong — explore, withdraw, update strategy. Error correction is a thought, not a training step. *Medium-term*: the mental model's certainty mechanism — wrong predictions reduce certainty on the responsible entries, right predictions increase it, so bad mappings lose influence over time. *Long-term*: gradient-based training adjusts policy weights across episodes. The first two timescales operate at runtime — the organism corrects errors while it's living, through the same receptor loop that constitutes thought.

**Planning**: Planning is the mental model running `predict_delta` for actions the organism hasn't taken. The organism asks "what if I do X" and the mental model returns predicted receptor state changes with certainty scores. Those predicted states fire internal receptors — optimism if the predicted outcome is good, conflict if two actions both predict pain, curiosity if the prediction is uncertain. Those receptor firings feed back as input to the next cycle, which can evaluate another candidate action or explore deeper consequences of the first. Planning is a thought where the mental model is the dominant source of receptor firings — the organism is responding to *predicted futures*. The depth of planning is how many cycles of predict-compare-predict the organism runs before committing to motor output.

**Decision-Making**: A decision is when a thought's cascade exits through the motor system. The internal receptor firings converge — one action pathway dominates — and the transformer produces muscle activations. Which receptors participated determines the character of the decision: pain-memory dominance produces risk-aversion, curiosity dominance produces exploration, unresolved conflict produces hesitation. The fast/slow pathway split maps directly onto this — the fast pathway is a decision without planning (one cycle, external receptors in, motor output out), the slow pathway is a decision with planning (the transformer attends to recent internal deliberation across the sequence window), and the Router is deciding whether to plan at all.

**Thinking Substrate (MCTS)**: Monte Carlo Tree Search provides the concrete substrate that makes metacognition tractable. The tree is an architectural component — alongside the mental model and experience log — that records which thinking paths were taken, how often, and with what outcomes. The tree's metadata is itself input to receptors: visit count patterns trigger the shaped_absence receptor (underexplored regions of thought), UCB scores trigger curiosity (high-uncertainty branches worth exploring), value convergence triggers completion (the search has resolved), path divergence triggers exception_detection (something about this thinking path is different from what the pattern predicted), and high-value low-visit branches trigger optimization (there's a better reasoning path the organism hasn't been taking). The receptor topology makes the tree's evaluation function intrinsic — the value of a thinking path is determined by which receptors fire at its terminus, not by a designer-specified objective. This creates a self-modifying loop with no fixed ceiling: better receptor topology → better evaluation → better search → richer tree → deeper metacognitive analysis → receptor firings that develop better topology. The loop stabilizes only when the environment stops presenting structure worth detecting — and since the environmental augmentation family lets the organism increase environmental complexity, the ceiling rises with the organism. Receptors are the only cognitive unit that generalizes across every level of this loop because they fire on conditions regardless of origin — environment, internal processing, tree structure, or analysis of the analysis. Reward functions are level-specific. Loss functions are task-specific. Only the receptor scales.

### Developmental Terms

**Proprioception**: Sensing your own body's position and configuration (where your limbs are, joint angles, body heading).

**Efference Copy**: Internal prediction of the sensory consequences of your own actions. Before executing a muscle command, the system predicts what receptor changes that action should cause. Mismatch between prediction and actual outcome signals external intervention or controllability limits.

**Grounded Language**: Language where every word maps to a receptor state the organism actually experienced. Not statistical word embeddings, but explicit pointers to sensorimotor patterns. "Pain" maps to obs[0:5] firing when limb tips contact pain field sources. "Self" maps to the controllability decomposition. The grounding is inspectable — you can trace any word to the receptor state it refers to.

**Cultural Transmission**: Transfer of knowledge between organisms via mental model replication. One organism's cause-effect database can be copied (not trained) into another organism. The original +223% claim was retracted after controlled decomposition — the benefit is training-time observation enrichment, not inference-time modulation. The architectural separation remains valuable for legibility, compartmentalization, and cross-generational knowledge transfer.

### Evolutionary Terms

**Deep Time Learning**: Learning that happens across generations, where each generation inherits the receptor topology that proved adequate and starts from a richer cognitive foundation than the one before. The task isn't specified — it emerges from what the environment makes load-bearing over sufficient generational depth. Distinct from gradient descent (training runs) and reinforcement learning (episodes).

**Environment Tiers**: Progressively more complex environments (8 levels in current implementation). Each tier is derived from the genome project — the environment must contain the causal structure necessary for specific receptors to evolve. Lower tiers produce simpler receptor topologies; higher tiers produce different (not just more) receptors.

**Topology Bias Inheritance**: Offspring inherit their parent's receptor topology as a *prior*, not hardwired. The offspring must rediscover the receptors through experience, but convergence accelerates dramatically (from 15 training epochs in generation 0 to 0 epochs by generation 4). Evolution of learning mechanisms, not just evolution of behavior.

**Cross-Tier Transfer**: Training an organism in environment tier X, then testing performance in tier Y. Reveals which receptor families are universal (transfer broadly) vs specialized (must be learned in target environment). Result: social skills transfer universally (11-25x), tool use resists transfer.

**Probe-Gated Inheritance**: Topology bias is gated by a constitutional probe budget. The organism must actually probe and explore the environment to validate inherited priors — inheritance accelerates but doesn't bypass the need for grounded experience. The probe rate floor lives outside the genome and cannot be selected to zero.

**Genome Project**: The formal specification of the receptor search space — 179 receptors across 22 families. The periodic table of cognitive capabilities. Each receptor entry specifies what environmental structure it detects, what survival cost the organism pays for missing it, and what must already exist before it can emerge. The genome project is load-bearing on environmental design: the 179 receptors are 179 environmental design requirements.

**Invariant Trunk**: The set of receptors that emerge in every environment regardless of tier or complexity. 18 receptors are invariant across all 8 physics-world tiers — these are the strongest candidates for universal cognitive primitives.

### Key Theoretical Contributions

**The Serialization Thesis**: Sequential processing of simultaneously-available information is not a hardware bottleneck but an evolved optimization — temporal decomposition creates prediction opportunities that parallel processing destroys. Each processing stage generates expectations about what the next stage will reveal; the delta is where learning happens.

**Per-Receptor Pipeline Architecture**: Every receptor family has its own evolved temporal decomposition strategy, optimized for the prediction structure of its specific domain. Pain processes coarse-to-fine-to-contextual; curiosity processes novelty-to-relevance-to-strategy.

**Annealing Discovery (T57)**: The framework's first structural self-discovery. Releasing certainty on conflict entries (annealing) produces more genuine conflict resolutions than protecting them (shielding). Conflict resolution works by releasing commitment, not by protecting it. Supported across 6 seeds, pre-registered as rival to T55 (which was directionally falsified).

---

## What's Implemented

### The Cognitive Sequence (Steps 1-30)
A 6-limbed organism in a 2D liquid environment learns to navigate pain/endorphin fields via a transformer outputting binary muscle activations. Each step earns its complexity from the step below:

| Phase | Steps | What develops |
|-------|-------|---------------|
| Sensorimotor Foundation | 1-4 | Pain/endorphin receptors, metabolic economy, hierarchical nervous system |
| Adaptive Sensing | 5-8 | Temporal association, spatial memory, habituation, distance sensing |
| World Modeling | 9-13 | Causal mental model, curiosity, pattern recognition, multiple hypotheses |
| Self-Model | 14-18 | Proprioception, efference copy, controllability, planning |
| Social Cognition | 19-24 | Proto-symbols, NPC opponent, empathy, intentional signaling, shared vocabulary |
| Higher Cognition | 25-29 | Optimism, conflict receptor, arbitration, metacognition, concepts |
| Language | 30 | Grounded language — every word maps to a receptor state |

### Evolutionary Infrastructure (Steps 31-43)
- **Environment tiers** (8 levels, genome-driven) from simple field navigation to meta-cognitive self-regulation
- **Receptor discovery** — 182 null-calibrated tests with per-test null types (action-shuffled, block-permuted, NPC-appearance-permuted, Granger causality)
- **Topology bias inheritance** — offspring inherit receptor topology priors, probe-gated
- **Population evolution** — 8 competing organisms, social arms race
- **Cross-tier transfer** — 8x8 transfer matrix
- **LLM grounding bridge** — connecting the mental model to language

### Physics World (Steps 48-55)
- **Rigid body simulation** (pymunk) with organism body, limbs, and objects
- **Grip mechanics** — automatic grip on contact + extension, energy cost
- **Compound objects** — levers (pin joints), spring gates, hinged barriers
- **Developmental body changes** — limb growth, receptor sensitivity maturation
- **Persistent world state** — environmental modifications carry across episodes
- **Canopy activation sweep** — receptor discovery across physics world at all 8 tiers

### Staged Observation Processing (Step 50)
- **4-stage pipeline** with inter-stage predictions testing the serialization thesis
- Body Immediate (39 dims) -> Spatial/Temporal (59 dims) -> Action/Agency (37 dims) -> Social/Cognitive (34 dims)
- Inter-stage prediction MSE decreases over training (learnable prediction structure)
- Staged model outperforms flat model on val accuracy (95.5% vs 94.5%)

### Abstract & Self-Modification Environments (Steps 56-58)
- **T7**: 8 causal graph templates, hidden variables, zone consumption order matters
- **T8**: 8 skill zones, 5 difficulty levels, Ship of Theseus test, curriculum design
- **Combined**: abstract problems at varying difficulty with self-directed skill development

### Self-Play & Thinking Substrate (Steps 59-60)
- **Self-play pipeline** — policy drives behavior, oracle removed after bootstrap
- **MCTS thinking substrate** — organism thinks before acting; tree metadata (visit entropy, best value, value convergence, path divergence, underexplored, depth reached) feeds back as 6 receptor channels
- **Thinking influence** — 5 of 6 channels active by self-play iteration 5; +23.3 reward difference over non-thinking organisms
- **Batched predict_delta** — 4.4x speedup on thinking operations

### Deep Time with Thinking (10 generations)
- **151 unique receptors** discovered across all generations (up from 75 at gen 0)
- **53 receptors gained** through evolution alone — including full epistemic chain, metacognition, theory of mind, nested theory of mind, meta-observation, self-regulation, niche construction
- **21 receptors lost** — complexity reshapes topology (T27 confirmed)
- **Peak thinking influence** at generation 7: partial correlation 0.376
- **Novel receptor detection** — scanning mental model for distinctions the genome didn't anticipate

### Closed-Loop Training
- Mental model online during data generation — features computed inline at correct lag
- Eliminates the augmentation pipeline's leakage class entirely
- 7% exploration + 2% null-action probes for counterfactual variation
- EntityRelationStore wired in — observe_npc called every step for social cognition

### Environment Enrichment
- **Multi-NPC observation** — closest of 4 profiled NPCs (cooperative, competitive, erratic, deceptive) visible to organism
- **Strategic deception NPC** — context-dependent lying (signals endorphin near pain, signals pain near endorphin)
- **Non-stationary rules** — T7 trigger signals rotate between phases
- **Stochastic hidden confounders** — 3-state Markov chain modulating 4 modalities simultaneously
- **Cross-modal objects** — sources with correlated pain + temperature + chemical signatures

### The Genome Project (22 families, 179 receptors)
A formal specification of the receptor search space — the periodic table of cognitive capabilities:

| Family | Receptors | From -> To |
|--------|-----------|-----------|
| Repetition | 6 | Static repetition -> causal rhythms |
| Association | 8 | Spatial co-occurrence -> relational analogy |
| Similarity | 7 | Perceptual features -> structural invariance |
| Causality | 11 | Coincidence -> causal graphs |
| Agency | 8 | Controllability -> niche construction |
| Meta-Motivational | 13 | Curiosity -> metacognition |
| Regulatory | 9 | Stress detection -> emotional intelligence |
| Social | 14 | Other detection -> moral reasoning |
| Compression | 15 | Pattern recognition -> constraint shape -> shaped absence -> missing piece located -> analogy |
| Observation | 11 | Change detection -> statistical anomaly -> rarity -> significance -> meta-observation |
| Formalization | 11 | Rule extraction -> optimization -> theory formation |
| Mathematics | 7 | Quantity -> necessity -> proof -> formal composition |
| Organization | 7 | Boundary -> part-whole -> system detection |
| Self-Augmentation | 5 | Capability change -> metamorphic planning |
| Interaction | 7 | Response recognition -> contact response -> grip -> lever -> composite affordance |
| Environmental Augmentation | 5 | Change detection -> developmental environment engineering |
| Sequential Processing | 5 | Stage prediction -> prediction architecture awareness |
| Epistemic | 7 | Belief detection -> doubt -> conflation -> fundamental distinction -> topology awareness -> epistemic strategy |
| Perception | 5 | Staged processing -> response loop detection |
| Logic | 6 | Semantic relations -> transitivity -> conjunction -> quantifier -> contradiction -> it_follows |
| Language | 3 | Naming -> self-talk -> referential grounding |
| Bridging | 4 | Mimicry -> trust -> executability -> translation |

### Key Empirical Results

- **77 receptors discovered** in single-run oracle training; **75 in self-play** (policy-driven, no oracle); **151 unique across 10 generations** of deep time with thinking substrate
- **18 invariant receptors** across all 8 physics-world tiers — including grip_affordance and push_affordance as part of the embodied trunk
- **Complexity reshapes, doesn't expand**: T27 confirmed across tiers, single runs, and deep time (53 gained, 21 lost in 10 generations)
- **Topology inheritance**: convergence accelerates from 15 epochs to 0 across generations
- **Social universally transferable**: any prior training helps social environments (11-25x)
- **Tool use resists transfer**: must be learned directly in the target environment
- **T57 annealing supported** (6 seeds): releasing certainty on conflict entries produces more resolutions than protecting them. First structural self-discovery.
- **T55 directionally falsified**: read-shielding was protecting the wrong thing — the falsification led to the Epistemic family
- **Cultural transmission revised**: +223% claim retracted after decomposition; benefit is training-time observation enrichment, not inference-time modulation
- **Staged processing**: inter-stage prediction MSE decreases 25% over training; staged model outperforms flat on val accuracy
- **Self-play finds richer causality**: 7 causality receptors in self-play vs 4 in oracle — suboptimal actions create more varied causal experiences
- **Thinking substrate load-bearing from iteration 1**: ablation divergence 0.060, reward difference +23.3, 5/6 channels active by iteration 5
- **Epistemic family activated from scratch under evolution**: belief, doubt, epistemic strategy emerged in deep time when they couldn't emerge in single runs
- **Peak thinking influence gen 5**: partial correlation 0.533
- **depth_reached phase transition at gen 29**: metacognition + conflation activated at gen 27-28 as prerequisites; depth became dominant thinking channel one generation later. The organism learned to sense how deep its own reasoning went.
- **Convergence result**: conflation receptor predicted by theoretical reasoning, evolved independently by the organism under selection pressure, confirmed as load-bearing prerequisite for depth_reached. Two independent paths to the same receptor — T40 confirmed at the meta level.

### Theories Index

92 formal theoretical claims indexed in `theories.md`:
- 8 supported by experimental evidence (including the convergence result and contextual signal interpretation)
- 1 revised after controlled decomposition
- 1 directionally falsified (T55, replaced by T57 annealing)
- 72 proposed with falsification criteria

---

## Quick Start

### Requirements
```
Python 3.10+
PyTorch
NumPy
pymunk (for physics world)
```

### Run the organism
```bash
cd src
python environment.py          # Test the organism (all body plans)
python train.py                # Full training pipeline (500 episodes, ~10 min)
python model.py                # Model architecture summary
```

### Self-play with thinking (recommended)
```python
from train import generate_training_data_self_play, train_model
X, Y, Z, log, engine, model = generate_training_data_self_play(
    num_bootstrap=50, num_self_play=75, num_iterations=3,
    steps_per_episode=300, seed=42, use_thinking=True)
```

### Closed-loop training (oracle-driven)
```python
from train import generate_training_data_closed_loop, train_model
X, Y, Z, log, engine = generate_training_data_closed_loop(
    num_bootstrap=100, num_online=400, steps_per_episode=300, seed=42)
model = train_model(X, Y, Z, epochs=30, staged=True)
```

### Run the visualization
Open `visualization/index.html` in a browser after training (loads `src/data/replay.json`).

### Run the laboratory
```bash
python receptor_discovery.py     # Full 182-test receptor battery with null calibration
python environment_tiers.py      # Test all 8 environment tiers
python canopy_sweep.py           # Physics-world receptor sweep across tiers
python run_full_battery.py       # 3-environment comparison (field, physics, T7+T8)
python t54_v2_experiment.py      # T54/T57 rationalization/annealing experiment
python abstract_env.py           # T7 abstract + T8 self-modification environments
python population_evolution.py   # Population evolution (8 organisms)
python cross_tier_transfer.py    # Cross-tier transfer matrix
python self_play_experiment.py   # Self-play vs oracle comparison
python thinking_influence.py     # Thinking substrate influence measurement
python thinking_emergence_curve.py # 10-iteration thinking emergence curve
python deep_time_thinking.py     # Deep time with thinking (10 generations)
python deep_time_overnight.py    # 50-generation overnight run with novel detection
```

### Scale testing
```bash
python scaling.py              # Limb count, segments, 3D, diversity, generational
```

---

## Architecture

```
175-dim observation vector --> HierarchicalPolicy --> 22-bit action vector
           |                        |
     ThinkingTree             +-----+-----+
     (MCTS 24 sims)           |     |     |
     6 receptor channels  FastPath SlowPath Router
           |              (reflex) (transformer)
           |                  |     |     |
           +----------> ArbitrationHead <-+
                        (5 group weights)
                              |
                        Blended output
                              |
                    +---------+---------+
                    |         |         |
              18 muscle  4 emission  Mental Model
              (6x3)     (signals)   (26K+ mappings)
```

### Observation Vector (175 dims)
Pain(6), endorphin(6), temperature(6), chemical(6), pressure(6), fatigue(6), energy(1), temporal aversion(6), receptor gain(6), pain memory(25), distance sensing(16), prediction error(6), mental model features(4), pattern features(2), kinematics(2), limb deviations(6), efference copy(22), agency(3), object proximity(3), object responding(3), NPC obs(12), optimism(2), conflict(3), concepts(2), grip state(6), physics(3), **thinking channels(6)**: best_value, visit_entropy, value_convergence, path_divergence, underexplored, depth_reached

### Staged Processing Pipeline
```
Stage 1 [Body: 39d] --> predict Stage 2 --> Stage 2 [Spatial: 59d] --> predict Stage 3
    --> Stage 3 [Action: 37d] --> predict Stage 4 --> Stage 4 [Social: 34d] --> transformer
```

---

## The Thesis

Intelligence is what happens when you run evolutionary receptor topology selection long enough in a rich enough environment. It is not a property you design into a system. It is a property that grows out of a process.

The organism builds the world that builds the organism that builds the next world. Now with a thinking substrate tracking every thinking path taken through that loop, analyzing which paths lead where, and bootstrapping increasingly sophisticated reasoning strategies from the accumulated record.

The self-modifying loop has no fixed ceiling because each layer feeds the next: better receptor topology → better evaluation → better thinking → richer experience → deeper metacognition → receptor firings that develop better topology. The ceiling rises with the organism.

Not simulation. Not reconstruction. Generation.

---

## Academic Context

The core claims find support across four research communities (see `docs/WHITEPAPER.md` Section 10):
- **Grounded cognition**: Barsalou (2008), O'Regan & Noe (2001)
- **Active inference**: Friston's free energy framework
- **Inverse phylogeny**: Trends in Cognitive Sciences (2023)
- **Embodied cognition**: Phil. Trans. Royal Society B (2024)
- **Ecological realism**: Gibson (1979) — accepted for embodied coupling, diverged on the necessity of internal models
- **Enactivism**: Varela, Thompson, Rosch (1991) — accepted for action-coupled cognition, diverged on prediction requiring internal models

The serialization thesis extends Friston: organisms don't just minimize prediction error, they manufacture prediction opportunities through sequential processing architecture. The per-receptor pipeline claim — that each receptor family has its own evolved temporal decomposition strategy — goes beyond anything in the current embodied cognition literature.

No existing program unifies these threads. The receptor topology as a single generative mechanism — from grounding through compartmentalization to language — appears to be a novel synthesis.

For a formal mathematical treatment of the Receptor-Topological Dynamical System (RTDS), see [docs/ERTI_formalization.tex](docs/ERTI_formalization.tex). Three levels of documentation: intuitive argument ([Whitepaper](docs/WHITEPAPER.md)), empirical record ([Theories](docs/theories.md) + [Roadmap](docs/ROADMAP.md)), formal structure ([Formalization](docs/ERTI_formalization.tex)).

---

## Project Structure

```
abi/
+-- src/                            # Core implementation
|   +-- environment.py              # Organism, NPC, Environment (175-dim obs)
|   +-- model.py                    # HierarchicalPolicy (fast/slow/router/arbitration/staged)
|   +-- train.py                    # Training: augmented, closed-loop, self-play
|   +-- mental_model.py             # Causal mental model (encoder, mappings, patterns, entity store)
|   +-- thinking_substrate.py       # MCTS thinking tree (6 receptor channels)
|   +-- thinking_influence.py       # Ablation + partial correlation measurement
|   +-- thinking_emergence_curve.py # Multi-iteration emergence tracking
|   +-- novel_receptor_detector.py  # Detect distinctions not in the genome
|   +-- physics_world.py            # Pymunk rigid body simulation + grip + compounds
|   +-- environment_tiers.py        # 8 tiered environments (multi-NPC, deception, hidden vars)
|   +-- abstract_env.py             # T7 abstract + T8 self-modification environments
|   +-- receptor_discovery.py       # 182 receptor emergence tests with null calibration
|   +-- deep_time.py                # Evolutionary loop (population + inheritance)
|   +-- deep_time_thinking.py       # Deep time with thinking substrate
|   +-- deep_time_overnight.py      # 50-gen overnight run with checkpoints + resume
|   +-- self_play_experiment.py     # Self-play vs oracle comparison
|   +-- topology_inheritance.py     # Multi-generational topology bias inheritance
|   +-- population_evolution.py     # Population evolution (8 organisms)
|   +-- evolutionary_sweep.py       # Cross-tier evolutionary sweep
|   +-- cross_tier_transfer.py      # Transfer matrix across tiers
|   +-- canopy_sweep.py             # Physics-world receptor sweep
|   +-- run_full_battery.py         # 3-environment receptor comparison
|   +-- t54_v2_experiment.py        # Rationalization/annealing experiment
|   +-- scaling.py                  # Scaling experiments (limbs, segments, 3D)
|   +-- grounding.py                # Grounded language dictionary
|   +-- llm_grounding.py            # LLM grounding bridge
+-- genome_project/                 # Receptor search space specification
|   +-- families/                   # 21 receptor family YAMLs (179 receptors)
|   +-- schemas/                    # Receptor schema definition
|   +-- docs/                       # Cross-family dependencies, overview
+-- theories.md                     # 90 indexed theoretical claims with status
+-- serialization_thesis.md         # The serialization thesis (standalone paper)
+-- visualization/                  # Three.js organism visualization
+-- docs/                           # Whitepaper, roadmap, framework documents
+-- results/                        # Experimental results (JSON + FINDINGS.md)
+-- LICENSE                         # MIT License
```

---

## Contributing

The genome project is designed to be extended. Each receptor entry specifies what environmental structure it detects, what would falsify it, and what must already exist before it can emerge. New receptor families, deeper environment tiers, and empirical tests against the predictions are all welcome.

The theory learns most from its failures. T55 (read-shielding) was directionally falsified and replaced by T57 (annealing) — and that falsification led to the Epistemic family, the first family predicted by an experimental result rather than by theoretical deduction. Which genome predictions don't hold? Which receptors emerge where not predicted? Which never emerge where predicted? Each discrepancy is where the framework needs revision — and revision is growth.

---

## License

MIT License. See [LICENSE](LICENSE).

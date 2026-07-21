# Artificial Biological Intelligence (ABI)

**Intelligence whose shape is determined by the evolutionary history of its receptor topology. Not designed. Grown.**

ABI starts where evolution started — a simple organism in a liquid environment with pain receptors and muscles — and builds upward through 46 steps to grounded language, evolutionary receptor topology discovery, and a laboratory for the dynamics of intelligence itself.

Current AI starts where evolution finished (language) and tries to work downward toward grounding it may never reach. ABI starts at the bottom and builds up. Slower. But the foundations are actually there when you need them.

---

## The Core Idea

Agency is a continuous variable determined by receptor complexity, effector complexity, and the processing in between. **Capability without receptor is latent and never gets used.** The receptor is not separate from cognition — it IS why the capability gets deployed at all. Motivation and cognition are the same thing, viewed from different angles.

The architecture has three components that current AI conflates into one:
- **The transformer** is the inference engine (it processes, it doesn't store)
- **The mental model** is the knowledge base (explicit, queryable cause-effect mappings)
- **The experience log** is ground truth (append-only, immutable, no learned process can overwrite it)

This separation dissolves grounding, compartmentalization, legibility, unlearning, and safe failure as problems.

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

### Evolutionary Infrastructure (Steps 31-40)
- **Environment tiers** (8 levels, genome-driven) from simple field navigation to meta-cognitive self-regulation
- **Receptor discovery** — 46 tests detecting which of 138 genome receptors have emerged
- **Topology bias inheritance** — offspring inherit receptor topology priors, probe-gated
- **Population evolution** — 8 competing organisms, social arms race
- **Cross-tier transfer** — 8x8 transfer matrix showing how receptor topologies port across environments
- **LLM grounding bridge** — connecting the mental model to language

### The Genome Project (16 families, 138 receptors)
A formal specification of the receptor search space — the periodic table of cognitive capabilities:

| Family | Receptors | From → To |
|--------|-----------|-----------|
| Repetition | 6 | Static repetition → causal rhythms |
| Association | 8 | Spatial co-occurrence → relational analogy |
| Similarity | 7 | Perceptual features → structural invariance |
| Causality | 10 | Coincidence → causal graphs |
| Agency | 7 | Controllability → niche construction |
| Meta-Motivational | 10 | Curiosity → metacognition |
| Regulatory | 8 | Stress detection → emotional intelligence |
| Social | 12 | Other detection → moral reasoning |
| Compression | 11 | Pattern recognition → completion |
| Observation | 8 | Change detection → meta-observation |
| Formalization | 10 | Rule extraction → theory formation |
| Mathematics | 7 | Quantity → necessity → proof → formal composition |
| Organization | 7 | Boundary → part-whole → system detection → organizational mirror |
| Self-Augmentation | 5 | Capability change → identity continuity → metamorphic planning |
| Interaction | 6 | Contact response → grip → lever → composite affordance |
| Environmental Augmentation | 5 | Change detection → deliberate complexification → curriculum self-design |

### Key Empirical Results

- **29 invariant receptors** across all 8 environment tiers — the trunk of intelligence
- **1,013 stable concepts** — compressed causal chains the organism actually experienced
- **Cultural transmission**: +223% reward via mental model replication (database copy, not training)
- **Topology inheritance**: convergence accelerates from 15 epochs to 0 across generations
- **Complexity reshapes, doesn't expand**: different tiers produce different receptors, not more
- **Social universally transferable**: any prior training helps social environments (11-25x)
- **Tool use resists transfer**: must be learned directly in the target environment

---

## Quick Start

### Requirements
```
Python 3.10+
PyTorch
NumPy
```

### Run the organism
```bash
cd src
python environment.py          # Test the organism (all body plans)
python train.py                # Full training pipeline (500 episodes, ~10 min)
python model.py                # Model architecture summary
```

### Run the visualization
Open `visualization/index.html` in a browser after training (loads `src/data/replay.json`).

### Run the laboratory
```bash
python receptor_discovery.py   # Discover which receptors have emerged
python environment_tiers.py    # Test all 8 environment tiers
python experiment_41.py        # Evolutionary sweep across tiers
python population_evolution.py # Population evolution (8 organisms)
python cross_tier_transfer.py  # Cross-tier transfer matrix
python grounding.py            # Grounded language analysis
python llm_grounding.py        # LLM grounding bridge
```

### Scale testing
```bash
python scaling.py              # Limb count, segments, 3D, diversity, generational
```

---

## Architecture

```
160-dim observation vector ──► HierarchicalPolicy ──► 22-bit action vector
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              FastPathway      SlowPathway       Router
              (MLP reflex)   (2-layer transformer)  (confidence+energy+conflict)
                    │               │               │
                    └───────► ArbitrationHead ◄──────┘
                           (5 receptor group weights)
                                    │
                              Blended output
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              18 muscle bits   4 emission bits   Mental Model
              (6 limbs × 3)   (signal vocab)   (13K+ mappings)
```

### Observation Vector (160 dims)
Pain(6), endorphin(6), temperature(6), chemical(6), pressure(6), fatigue(6), energy(1), temporal aversion(6), receptor gain(6), pain memory(25), distance sensing(16), prediction error(6), mental model features(4), pattern features(2), proprioception(2+), limb deviations(6), efference copy(22), agency(3), object proximity(3), object responding(3), NPC obs(12), optimism(2), conflict(3), concepts(2)

---

## The Thesis

Intelligence is what happens when you run evolutionary receptor topology selection long enough in a rich enough environment. It is not a property you design into a system. It is a property that grows out of a process.

The organism builds the world that builds the organism that builds the next world.

Not simulation. Not reconstruction. Generation.

---

## Academic Context

The core claims find support across four research communities (see `docs/WHITEPAPER.md` Section 10):
- **Grounded cognition**: Barsalou (2008), O'Regan & Noe (2001)
- **Active inference**: Friston's free energy framework
- **Inverse phylogeny**: Trends in Cognitive Sciences (2023)
- **Embodied cognition**: Phil. Trans. Royal Society B (2024)

No existing program unifies these threads. The receptor topology as a single generative mechanism — from grounding through compartmentalization to language — appears to be a novel synthesis.

---

## Project Structure

```
abi/
├── src/                          # Core implementation
│   ├── environment.py            # Organism, NPC, Environment classes
│   ├── model.py                  # HierarchicalPolicy (fast/slow/router/arbitration)
│   ├── train.py                  # Supervised behavioral cloning pipeline
│   ├── mental_model.py           # Causal mental model (encoder, mappings, patterns)
│   ├── environment_tiers.py      # 8 tiered environments (genome-driven)
│   ├── receptor_discovery.py     # 46 receptor emergence tests
│   ├── topology_inheritance.py   # Multi-generational topology bias inheritance
│   ├── population_evolution.py   # Population evolution (8 organisms)
│   ├── evolutionary_sweep.py     # Cross-tier evolutionary sweep
│   ├── cross_tier_transfer.py    # Transfer matrix across tiers
│   ├── scaling.py                # Scaling experiments (limbs, segments, 3D)
│   ├── grounding.py              # Grounded language dictionary
│   ├── llm_grounding.py          # LLM grounding bridge
│   └── experiment_41.py          # Laboratory experiments
├── genome_project/               # Receptor search space specification
│   ├── families/                 # 16 receptor family YAMLs (138 receptors)
│   ├── schemas/                  # Receptor schema definition
│   └── docs/                     # Cross-family dependencies, overview
├── visualization/                # Three.js organism visualization
│   └── index.html
├── verification/                 # Emergent property verification scripts
├── docs/                         # Whitepaper, roadmap, framework documents
│   ├── WHITEPAPER.md             # Full ABI whitepaper (v2)
│   └── ROADMAP.md                # 46-step roadmap with results
└── LICENSE                       # MIT License
```

---

## Contributing

The genome project is designed to be extended. Each receptor entry specifies what environmental structure it detects, what would falsify it, and what must already exist before it can emerge. New receptor families, deeper environment tiers, and empirical tests against the predictions are all welcome.

The theory learns most from its failures. Which genome predictions don't hold? Which receptors emerge where not predicted? Which never emerge where predicted? Each discrepancy is where the framework needs revision — and revision is growth.

---

## License

MIT License. See [LICENSE](LICENSE).

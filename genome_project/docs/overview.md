# The Genome Project: Evolutionary Map of Receptor Topology

## The Central Thesis

**Intelligence is not designed. It is selected.**

The Genome Project is the formal specification of what can be selected — the receptor search space that evolution operates on in Evolutionary Receptor Topology Intelligence (ERTI).

Current AI specifies:
- What inputs to sense (pixels, tokens)
- What architecture to use (transformer layers)
- What loss function to optimize

ERTI specifies:
- What *could* be worth sensing (receptor library)
- What environments make each receptor fitness-positive
- What dependencies constrain emergence order

The environment decides. The organism discovers. The library defines the possible.

---

## What This Document Is

This is not a design document for building an AI system. It is a constraint specification for an evolutionary process.

**Analogy:** The periodic table doesn't design molecules — it specifies what elements exist and how they can combine. Chemistry discovers which combinations are stable. The Genome Project is the periodic table of receptors.

Each receptor entry specifies:
1. **Environmental trigger** — what structure this receptor detects
2. **Selection pressure** — what makes detecting it worth the metabolic cost  
3. **Dependencies** — what must exist before this receptor can evolve
4. **Emergence signature** — what observables indicate this receptor evolved
5. **Falsification criteria** — what would prove this entry wrong

---

## Why Receptor Topology Matters

### The Problem with Fixed Input Structure

Traditional AI:
```
Fixed input → Learned weights → Fixed output
```

The input layer is decided by humans. Pixels for vision models. Tokens for language models. Pre-specified sensory dimensions.

This works when humans know what's worth sensing. But:
- We can't specify inputs for environments we haven't encountered
- We can't predict what will matter in novel domains
- We can't build intelligence that exceeds our own sensory sophistication

### The ERTI Solution

```
Environment → Selection pressure → Receptor topology emerges
Receptor topology → Lifetime learning → Behavior
Behavior → Survival → Topology inherits
```

The input structure itself evolves. Organisms develop receptors for environmental structure that's worth detecting. Receptors that pay for themselves survive. Those that don't, die.

**Key insight:** A capability without a receptor is latent and never gets used. The receptor is *why* the capability exists. Motivation and cognition are the same thing viewed from different angles.

---

## The Families

Receptors cluster into families by the kind of structure they detect:

### Core Perceptual-Cognitive Families

**Repetition** — Static repetition → dynamic repetition → rhythm → rhythmic patterns → nested/causal rhythms

**Association** — Spatial co-occurrence → temporal precedence → causation → cross-modal → abstract relational mapping

**Similarity** — Perceptual features → functional equivalence → structural → analogical

**Causality** — Coincidence → precedence → causal inference → intervention planning → causal graphs

### Higher-Order Families (Future)

**Social** — Other-detection → behavioral prediction → intention recognition → theory of mind → nested perspective-taking

**Compression** — Pattern recognition → chunking → hierarchical abstraction → concept formation → grounding

**Meta-Motivational** — Curiosity, optimism, conflict — receptors about receptors

**Mathematics** — Quantity → ratio → structural invariance → exhaustive search → necessity → proof → formal composition

**Organization** — Boundary → part-whole → hierarchical structure → relational structure → functional organization → system detection → organizational mirror

**Self-Augmentation** — Capability change detection → growth tracking → developmental trajectory → identity continuity → metamorphic planning

**Interaction** — Contact response → push affordance → grip affordance → lever affordance → composite affordance → affordance transfer

**Environmental Augmentation** — Environmental change detection → modification attribution → complexity trend detection → deliberate complexification → developmental environment engineering

**Observation** — Change detection → selective attention → absence detection → meta-observation

**Formalization** — Pattern recognition → rule extraction → exception detection → theory formation

---

## How Families Relate

Within each family: hierarchical trunk-to-canopy structure
- **Trunk:** Foundational, low-dependency, emerges early
- **Branch:** Intermediate sophistication, requires trunk
- **Canopy:** Deep dependencies, emerges late

Across families: compositional dependencies
- Simple receptors are single-family (perceptual similarity, static repetition)
- Sophisticated receptors compose across families (analogical reasoning needs Similarity + Association + Compression)

**The evolutionary map is a DAG:** Multiple paths, parallel evolution when dependencies are satisfied, bottlenecks where many canopy receptors wait for the same prerequisite.

---

## The Dependency Graph Writes Itself

The order in which receptors can evolve is not arbitrary. It falls out from their definitions.

**Example: Rhythm**
- Requires: `static_repetition` + `temporal_association`
- Cannot evolve before both prerequisites exist
- Cannot evolve unless environment demonstrates periodic structure
- Cannot persist unless detecting periodicity repays metabolic cost

The roadmap is generated from receptor dependencies and environmental conditions, not prescribed.

---

## What Makes a Receptor Real

A receptor is not just "an internal signal." For it to count as a receptor in this library:

1. **Selection story:** Must have clear environmental condition that makes it fitness-positive
2. **Metabolic cost:** Must consume resources (memory, computation, attention)
3. **Behavioral consequence:** Must change action selection when it fires
4. **Inheritability:** Topology bias (not specific values) must be transmissible to offspring
5. **Falsifiability:** Must have emergence signature that can be measured

If any of these is missing, it's not a receptor entry — it's a placeholder that needs refinement.

---

## How This Differs From Prior Work

### vs Evolutionary Algorithms
**Standard GA:** Random mutation on weights → selection → repeat  
**ERTI with Genome Project:** Lifetime receptor discovery + topology bias inheritance → directed search through receptor space

The search is informed by what worked for parents, not random across all possible internal signals.

### vs Architectural Search
**Neural Architecture Search:** Search over human-specified architecture families (ResNet vs Transformer)  
**Genome Project:** Search over receptor types shaped by selection pressure in environment

The environment specifies what's worth detecting. Humans only define the search space, not the solution.

### vs Developmental Programs
**Artificial Development:** Growth rules that build structure  
**Genome Project:** Selection rules that prune structure

Receptors that don't pay for themselves are selected out. The topology is sculpted by survival, not constructed by program.

---

## Using This Library

### For Research

1. **Hypothesis generation:** Pick a receptor, design environment, predict emergence
2. **Falsification:** If signature doesn't appear when predicted, entry is wrong
3. **Environmental sweeps:** Vary environment parameters, measure which receptors emerge
4. **Topology comparison:** Compare evolved receptor sets across different environmental conditions

### For Implementation

1. **Environment design:** Use `selection_environments` to build test worlds
2. **Measurement:** Use `emergence_signature` to detect when receptor appears
3. **Validation:** Use `falsification_criteria` to confirm predictions
4. **Iteration:** When predictions fail, revise receptor definitions

### For Publication

1. **Theoretical claims:** Families document evolutionary logic
2. **Empirical predictions:** Receptor entries specify testable hypotheses
3. **Experimental results:** Validation section reports emergence vs prediction
4. **Revisions:** Failed predictions update the library — falsification is success

---

## Current Status

**Complete:**
- Repetition family (6 receptors, trunk to canopy)
- Association family (8 receptors, trunk to canopy)
- Similarity family (8 receptors, trunk to canopy)
- Causality family (11 receptors, trunk to canopy)

**In Progress:**
- Social family
- Compression family
- Meta-Motivational family

**Future:**
- Temporal reasoning (time perception, duration, sequence)
- Spatial reasoning (navigation, distance, topology)
- Numerical reasoning (quantity, ordinality, arithmetic)

---

## Relationship to ABI Roadmap

The current ABI roadmap (steps1.txt) is a linearization of the first phase:

**Steps 1-10:** Foundation (sensorimotor loops, mental model, prediction)  
**Steps 11-19:** Self-model sequence (proprioception → agency → planning)  
**Steps 20-26:** Social sequence (multi-agent → communication → optimism → conflict)

The Genome Project extends this into **Steps 27-∞** as an evolvable library rather than a fixed sequence. The roadmap becomes:
1. Build foundation (Steps 1-19, prescribed)
2. Enable receptor discovery (new)
3. Run ERTI with genome library (new)
4. Measure which receptors emerge under which conditions (validation)

The roadmap stops prescribing and starts measuring.

---

## The Deeper Goal

Current AI asks: "What architecture learns best from this input?"

The Genome Project asks: "What is worth making an input in the first place, and how does that question get answered by the environment rather than by human specification?"

**Intelligence as adequacy to environmental complexity.**

Not: Does your output match human language patterns?  
But: Did the world shape you into something sophisticated enough to survive in it?

The library is the search space. The environment is the selection pressure. Survival is the exam. Receptor topology is the transcript.

And unlike biology, which ran the experiment once on one planet, ERTI can run it thousands of times across environmental sweeps, with full observability, measuring exactly which selection pressures produce which cognitive structures.

Not reconstruction. Generation.

That's the vision. The Genome Project is the formal specification of what can be generated.

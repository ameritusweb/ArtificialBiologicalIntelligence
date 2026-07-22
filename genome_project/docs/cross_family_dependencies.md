# Cross-Family Dependencies

## Overview

Many sophisticated receptors require inputs from multiple families. This document maps the dependency relationships that cross family boundaries, showing how the families interweave to produce increasingly complex cognition.

The evolutionary map is not a tree — it's a directed acyclic graph (DAG) where higher-tier receptors compose primitive receptors from multiple families.

---

## Key Cross-Family Receptors

### Causal Rhythm (Repetition ← Causality)
**Receptor ID:** `causal_rhythm`  
**Dependencies:**
- `rhythm` (Repetition family)
- `causal_association` (Causality family)

**Why both:** Understanding that one periodic event *causes* another periodic event requires both rhythm detection (recognizing periodicities) and causal inference (understanding that A causes B, not just correlates).

**Emergence condition:** Environment with causally-related rhythms (predator-prey oscillations, resource-triggered migrations).

---

### Cross-Modal Association (Association ← Similarity)
**Receptor ID:** `cross_modal_association`  
**Dependencies:**
- `spatial_association` (Association family)
- `perceptual_similarity` (Similarity family)

**Why both:** Recognizing that stimulus in domain A predicts stimulus in domain B requires associating across domains (association) while recognizing that domain-B predictions benefit from domain-A signals (similarity in outcome-relevance).

**Emergence condition:** Multi-modal sensory environment where channels provide complementary information.

---

### Functional Similarity (Similarity ← Causality)
**Receptor ID:** `functional_similarity`  
**Dependencies:**
- `perceptual_similarity` (Similarity family)
- `causal_association` (Causality family)

**Why both:** Recognizing that two perceptually different stimuli are functionally equivalent requires understanding causation (what outcomes do actions produce) and similarity (these outcomes are equivalent despite different stimuli).

**Emergence condition:** Multiple paths to same outcome.

---

### Abstract Association (Association ← Similarity)
**Receptor ID:** `abstract_association`  
**Dependencies:**
- `causal_association` (Association/Causality family)
- `pattern_recognition` (Compression family)
- `analogical_similarity` (Similarity family)

**Why all three:** Recognizing that relationship between A and B is structurally similar to relationship between C and D requires:
- Understanding relationships themselves (association)
- Detecting patterns across relationships (compression)
- Recognizing structural equivalence (similarity)

**Emergence condition:** Complex environment with recurring relational structures.

---

### Analogical Similarity (Similarity ← Association)
**Receptor ID:** `analogical_similarity`  
**Dependencies:**
- `structural_similarity` (Similarity family)
- `abstract_association` (Association family)

**Why both:** "A is to B as C is to D" requires recognizing structural similarity (similarity family) and mapping relationships across domains (association family).

**Emergence condition:** Cross-domain problem transfer opportunities.

---

### Counterfactual Reasoning (Causality ← Multiple)
**Receptor ID:** `counterfactual_reasoning`  
**Dependencies:**
- `causal_inference` (Causality family)
- `planning` (Foundation - Step 18)
- `self_model` (Foundation - Steps 14-18)

**Why all three:** "What would have happened if I'd done X?" requires:
- Understanding causation (what causes what)
- Ability to simulate alternative actions (planning)
- Model of self as causal agent (self-model)

**Emergence condition:** High opportunity-cost environments.

---

### Hidden Confounder Detection (Causality ← Multiple)
**Receptor ID:** `hidden_confounder_detection`  
**Dependencies:**
- `common_cause_detection` (Causality family)
- `multiple_hypotheses` (Foundation - Step 13)
- `theory_of_mind` (Social family - future)

**Why all:** Inferring unobserved variables requires:
- Understanding common causes (causality)
- Maintaining multiple explanations (multiple hypotheses)
- Often involves inferring hidden internal states of agents (theory of mind)

**Emergence condition:** Social or hidden-state environments.

---

## Dependency Layers

Receptors naturally cluster into dependency layers based on how many families they draw from:

### Layer 0: Foundation (Steps 1-19)
Single-family dependencies or no dependencies beyond foundation.
- All trunk receptors
- Most branch receptors

### Layer 1: Two-Family Composition
- `cross_modal_association` (Association + Similarity)
- `functional_similarity` (Similarity + Causality)
- `causal_rhythm` (Repetition + Causality)
- `intervention_planning` (Causality + Planning)

### Layer 2: Three-Family Composition
- `abstract_association` (Association + Similarity + Compression)
- `counterfactual_reasoning` (Causality + Planning + Self-model)
- `completion` (Compression + Meta-Motivational + Observation)

### Layer 3: Four+ Family Composition (Deep Canopy)
- `causal_graph_reasoning` (Causality + Association + Multiple hypotheses + Planning)
- `analogical_reasoning` (Similarity + Association + Compression + Transfer)
- `exhaustive_search` (Meta-Motivational + Causality + Compression)
- `necessity_detection` (Mathematics + Causality + Compression + Observation) — the apex of mathematical reasoning
- `functional_organization` (Organization + Agency + Causality) — structure that serves a purpose
- `organizational_mirror` (Organization + Meta-Motivational + Similarity) — structure detecting itself
- `cross_pipeline_prediction` (Sequential Processing + Association + Compression) — receptor pipelines predicting each other's states; the mechanism for integrated cognition
- `analogy_receptor` (Compression + Similarity + Sequential Processing) — meta-receptor operating on receptor firing patterns, not environmental input
- `prediction_architecture_awareness` (Sequential Processing + Organization + Self-Augmentation) — organism aware of its own processing pipeline structure
- `processing_speed_receptor` (Perception + Meta-Motivational + Regulatory) — processing cost as model-environment fit signal
- `response_loop_detection` (Perception + Meta-Motivational + Regulatory + Meta-Motivational) — detecting that the corrective response IS the problem; internal self-damping

---

## Evolutionary Implications

**Parallel Evolution:** Layer 0 receptors can evolve in parallel once foundation is complete. No inter-family coordination needed.

**Sequential Constraint:** Layer 1 receptors must wait for their prerequisite Layer 0 receptors from both families. But multiple Layer 1 receptors can still evolve in parallel if their dependencies are satisfied.

**Canopy Convergence:** Layer 2+ receptors require coordination across many families. These emerge late and represent sophisticated cognitive capabilities.

**Prediction:** The generation range for a receptor should correlate with:
1. Its dependency depth (how many layers deep)
2. Its cross-family breadth (how many families it draws from)
3. Environmental complexity required

---

## Testing Cross-Family Emergence

**Methodology:**
1. Identify receptor with cross-family dependencies
2. Verify all prerequisite receptors have emerged
3. Design environment that selects for the target receptor
4. Measure emergence signatures from BOTH families
5. Confirm that emergence doesn't occur if either prerequisite is blocked

**Example: Causal Rhythm**
- Prerequisites: `rhythm` + `causal_association`
- Test environment: Predator-prey oscillations where predator rhythm causes prey rhythm with lag
- Blocking test: If `causal_association` is prevented (non-controllable environment), `causal_rhythm` should NOT emerge even with strong rhythmic structure
- This confirms that both dependencies are necessary, not just sufficient

---

## Implications for ERTI

**Receptor Topology Inheritance:** When offspring inherit receptor topology bias from parents, cross-family receptors are more valuable to inherit because they're expensive to discover independently. An offspring that inherits "analogical_similarity already linked to abstract_association" has a major advantage over one that must discover that composition from scratch.

**Canalization Risk:** Cross-family receptors are most vulnerable to canalization because they depend on multiple learned components. If any dependency becomes incorrectly entrenched, the composite receptor inherits the error and can amplify it.

**Family Balance:** Environments that select strongly for one family but not others create imbalanced topologies. An organism that over-develops causality but under-develops similarity may miss functional equivalences. The receptor topology's *shape* — which families are well-developed — is itself informative about ancestral environment.

---

## Newly Added Cross-Family Receptors

### Agency + Causality

**Tool Use** (`tool_use`, Agency family)
- Dependencies: `controllability` (Agency) + `causal_chains` (Causality)
- Why both: Instrumental causation requires understanding "I control X" AND "X causes Y causes Z"
- Three-way causal chain: my_action → tool_state → environment_change

**Environmental Manipulation** (`environmental_manipulation`, Agency family)
- Dependencies: `tool_use` (Agency) + `planning` (Foundation) + `causal_inference` (Causality)
- Persistent modification requires multi-step causal reasoning about reshaped environments

### Agency + Social

**Distributed Agency** (`distributed_agency`, Agency family)
- Dependencies: `tool_use` (Agency) + `theory_of_mind` (Social) + `intentional_signaling` (Foundation)
- Control via other agents requires treating them as causal instruments with internal states

**Niche Construction** (`niche_construction`, Agency family)
- Dependencies: `environmental_manipulation` (Agency) + `long_term_planning` + `theory_of_mind` (Social)
- Why theory of mind: Must understand that environment shapes offspring receptor topology

### Meta-Motivational + Causality

**Regret** (`regret`, Meta-Motivational family)
- Dependencies: `counterfactual_reasoning` (Causality) + `optimism` (Meta-Motivational)
- Fires on gap between actual outcome and simulated counterfactual outcome

### Meta-Motivational + Multiple Families

**Impulse Override** (`impulse_override`, Meta-Motivational)
- Dependencies: `conflict` (Meta-Motivational) + `optimism` (Meta-Motivational) + `planning` (Foundation)
- Three-way composition: detect conflict, predict better future, act against gradient

**Self-Regulation** (`self_regulation`, Meta-Motivational)
- Dependencies: `impulse_override` + `attention_control` + `internal_conflict_model` (Foundation Step 28)
- Deepest meta receptor — requires full meta-motivational stack

**Metacognition** (`metacognition`, Meta-Motivational)
- Dependencies: `internal_conflict_model` + `self_regulation` (Meta-Motivational) + `theory_of_mind` (Social)
- Why theory of mind: Apply other-modeling template to self

### Compression + Meta-Motivational + Observation

**Completion** (`completion`, Compression family)
- Dependencies: `pattern_recognition` (Compression) + `conflict_detection` (Meta-Motivational) + `absence_observation` (Observation)
- Why all three: Pattern recognition says "this is a known shape." Conflict detection says "the tension resolved." Absence observation says "nothing is missing." All three must agree for the gestalt to register as complete.
- The gap between pattern_recognition and completion IS the drive to continue. An incomplete pattern fires pattern_recognition but NOT completion. The difference is the signal that keeps attention allocated.
- Completion fires -> conflict drops -> curiosity redirects to next open pattern -> the cycle continues. Completion is not a terminal state — it is a receptor state that changes what the next cycle attends to.
- Self-referential: the theory that predicts this receptor was itself recognized as complete when the receptor was identified.

---

## Updated Dependency Layers

### Layer 0: Foundation (Steps 1-19)
Single-family dependencies only.

### Layer 1: Two-Family Composition (18 receptors now)
- Repetition + Causality: `causal_rhythm`
- Association + Similarity: `cross_modal_association`, `abstract_association`
- Similarity + Causality: `functional_similarity`
- Agency + Causality: `tool_use`, `environmental_manipulation`
- Meta-Motivational + Causality: `regret`
- Meta-Motivational + Foundation: `impulse_override`, `attention_control`

### Layer 2: Three-Family Composition (8 receptors now)
- Association + Similarity + Compression: `abstract_association`
- Causality + Planning + Self-model: `counterfactual_reasoning`
- Agency + Planning + Causality: `environmental_manipulation`
- Meta-Motivational + Conflict + Planning: `impulse_override`
- Agency + Tool-use + Theory-of-mind: `distributed_agency`

### Layer 3: Four+ Family Composition (Deep Canopy, 6 receptors)
- `causal_graph_reasoning` (Causality + Association + Multiple hypotheses + Planning)
- `analogical_reasoning` (Similarity + Association + Compression + Transfer)
- `self_regulation` (Meta-Motivational × 3 + Metacognition)
- `niche_construction` (Agency + Planning + Social)
- `metacognition` (Meta-Motivational + Social + Self-model)
- `value_hierarchy` (Meta-Motivational + Arbitration + Long-term planning)

**Pattern discovered:** Layer 3+ receptors increasingly require Social family once it's complete. This suggests social cognition is a major integration bottleneck in the evolutionary graph.

---

## New Family Integration Points

### Agency Family Integrations

**Into Causality:**
- `tool_use` requires `causal_chains`
- `environmental_manipulation` requires `causal_inference`
- All agency receptors build on causality understanding

**Into Meta-Motivational:**
- `agency_salience` is the motivational counterpart to `controllability`
- Bridge between "I can cause X" (agency) and "I want to cause X" (meta-motivational)

**Into Social (future):**
- `distributed_agency` treats other agents as controllable
- `niche_construction` requires understanding offspring will inherit environment

### Meta-Motivational Family Integrations

**Into Causality:**
- `regret` requires `counterfactual_reasoning`
- Meta-motivational receptors fire on predicted/simulated states, not just actual

**Into Social (future):**
- `metacognition` applies theory of mind to self
- `value_hierarchy` enables predictable cooperation

**Into Agency:**
- `impulse_override` enables intentional control despite immediate receptors
- `self_regulation` is meta-control complementing environmental control

---

## Regulatory Family Cross-Dependencies

### Regulatory + Repetition

**Rhythm Entrainment** (`rhythm_entrainment`, Regulatory family)
- Dependencies: `rhythm` (Repetition) + `arousal_regulation` (Regulatory)
- Why both: External rhythms regulate internal arousal by entrainment — requires detecting rhythm AND managing arousal
- Explains: Circadian synchronization, social dance, mother's heartbeat soothing infant

**Pattern-Based Resolution** (`pattern_based_resolution`, Regulatory family)
- Dependencies: `pattern_recognition` (Compression) + `rhythm` (Repetition) + `stress_detection` (Regulatory)
- Why all three: Stored rhythmic patterns retrieved to resolve stress states
- Explains: Song replay phenomenon, comfort behaviors, self-soothing rituals

### Regulatory + Social

**Social Coregulation** (`social_coregulation`, Regulatory family)
- Dependencies: `receptor_propagation` (Step 22, Social) + `pattern_based_resolution` (Regulatory) + `intentional_signaling` (Step 23)
- Why all three: Mutual regulation requires empathy + regulatory patterns + communication
- Explains: Why music is social, group synchronization, emotional contagion

**Emotional Intelligence** (`emotional_intelligence`, Regulatory family)
- Dependencies: `theory_of_mind` (Social) + `social_coregulation` (Regulatory) + `pattern_based_resolution` (Regulatory)
- Why all three: Regulating others' states requires modeling their internal states + knowing regulatory patterns + ability to co-regulate
- Explains: Caregiving, emotional support, empathic regulation

### Regulatory + Meta-Motivational

**Self-Soothing** (`self_soothing`, Regulatory family)
- Dependencies: `pattern_based_resolution` (Regulatory) + `arousal_regulation` (Regulatory) + overlaps with `impulse_override` (Meta-Motivational)
- Bridge: Regulatory is about homeostasis; Meta-Motivational is about control
- Self-soothing requires both: detect arousal mismatch (Regulatory) AND override immediate impulses to execute regulatory pattern (Meta-Motivational)

**Ritual Formation** (`ritual_formation`, Regulatory family)
- Dependencies: `pattern_based_resolution` (Regulatory) + `social_coregulation` (Regulatory) + `value_hierarchy` (Meta-Motivational)
- Why value hierarchy: Rituals become stabilized when regulatory efficacy is prioritized consistently
- Explains: Cultural rituals, sacred behaviors, tradition formation

---

## Updated Dependency Layers with Regulatory

### Layer 0: Foundation (Steps 1-19)
Single-family dependencies only.
- New Regulatory additions: `stress_detection`, `arousal_regulation`

### Layer 1: Two-Family Composition (24 receptors now)
- Repetition + Regulatory: `rhythm_entrainment`
- Compression + Regulatory: `pattern_based_resolution` (also uses Repetition)
- Social + Regulatory: `social_coregulation`
- Meta-Motivational + Regulatory: `self_soothing`, `ritual_formation`
- (Previous entries remain)

### Layer 2: Three-Family Composition (12 receptors now)
- Repetition + Compression + Regulatory: `pattern_based_resolution`
- Social + Regulatory + Meta-Motivational: `ritual_formation`
- Social + Regulatory × 2: `emotional_intelligence`
- (Previous entries remain)

### Layer 3: Four+ Family Composition (Deep Canopy, 8 receptors)
- `emotional_intelligence` (Social + Regulatory × 2 + Meta-Motivational)
- (Previous entries remain)

**New pattern discovered:** Regulatory family bridges ALL other families because homeostasis uses whatever works — rhythm, patterns, social support, meta-control. This makes Regulatory the most cross-cutting family.

---

## The Song Replay Cross-Family Path

**Specific mechanism for involuntary mental song replay:**

```
Stress receptor (Regulatory trunk)
  ↓
Mental model retrieval (Foundation - Step 10)
  ↓
Pattern recognition (Compression - Step 12)
  ↓
Rhythm detection (Repetition family)
  ↓
Pattern-based resolution (Regulatory branch)
  ↓
Song replays internally
  ↓
Stress reduces via entrainment
```

Required receptors from 4 families:
1. `stress_detection` (Regulatory)
2. Mental model with retrieval (Foundation)
3. `pattern_recognition` (Compression)
4. `rhythm` (Repetition)

This is testable NOW in ERTI by measuring whether organisms develop "comfort actions" when stress is moderate.

---

## Future Work

As Social and Compression families are added, expect major new cross-family receptors:
- Theory of mind + Causal inference → Social manipulation prediction
- Compression + Social → Cultural transmission of compressed knowledge
- Meta-Motivational + Social → Empathy, moral reasoning, guilt
- **Regulatory + all families** → Homeostasis mechanisms using all available tools

**Regulatory family as universal integrator:** Because regulation uses whatever works (rhythm, patterns, social support, agency, meta-control), it creates dependency paths into every other family. This predicts Regulatory receptors should show the MOST cross-family composition in the library.

The complete cross-family dependency graph becomes the evolutionary roadmap for ERTI's later phases.

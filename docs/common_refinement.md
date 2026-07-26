# Common Refinement: What Two Organisms Can Communicate About

## The Question

Territory I says two organisms can communicate about exactly those distinctions that survive in the intersection of their topologies. A concept outside the intersection is untranslatable — not because the word is unknown, but because the receptor that would give it meaning isn't in the other organism's topology.

The cross-environment transfer experiment provides the data: one organism evolved in environment A then transferred to B ("transfer"), another evolved from scratch in B ("naive"). Same environment, different evolutionary histories. What topology do they share?

## Data

From `cross_env_transfer.json`: 10 generations each condition, same target environment B.

## Results

### The intersection is large but not total

| Set | Total | Trunk | Branch | Canopy |
|-----|-------|-------|--------|--------|
| **Shared (intersection)** | **134** | 20 | 42 | 58 |
| Transfer-only | 30 | 3 | 11 | 14 |
| Naive-only | 27 | 4 | 9 | 11 |

134 receptors are in both topologies — these are the communicable distinctions. 30 are in the transfer organism but not the naive (untranslatable to the naive). 27 are in the naive but not the transfer (untranslatable the other way). The organisms share ~70% of their receptor topologies and diverge on ~30%.

### Trunk is shared, canopy diverges — as predicted

| Layer | Shared | Total ever seen | Sharing rate |
|-------|--------|-----------------|--------------|
| Trunk | 20 | 27 | 74% |
| Canopy | 58 | 83 | 70% |

Trunk sharing (74%) and canopy sharing (70%) are closer than expected — the canopy is more shared than T95 ("canopy is biography") would strictly predict. This is because both organisms are in the SAME target environment B, so the environment is applying the same selection pressure to both canopies. The divergence comes from their different evolutionary histories providing different prerequisite chains.

### Per-generation Jaccard similarity

| Gen | Naive k_t | Transfer k_t | Intersection | Jaccard |
|-----|-----------|--------------|--------------|---------|
| 0 | 105 | 105 | 78 | 0.591 |
| 4 | 106 | 112 | 88 | 0.677 |
| 9 | 98 | 101 | 71 | 0.555 |

Jaccard oscillates between 0.51 and 0.68 — the shared world fluctuates as both topologies churn under the knapsack constraint. The organisms are never fully aligned and never fully divergent.

### Untranslatable concepts

**Transfer sees but naive cannot:**

The transfer organism brought these from environment A. They're distinctions that A's causal structure made fitness-positive but B's doesn't demand. The naive organism never developed them because B didn't select for them. To the naive organism, states that differ along these dimensions are the same point in its quotient space.

- `regret` — evaluating foregone alternatives (canopy)
- `niche_construction` — building environments for offspring (canopy)
- `long_range_causation` — actions at t affect outcomes at t+N (canopy)
- `distributed_agency` — coordinating with other agents (canopy)
- `counterfactual_salience` — motivationally-loaded possibilities (canopy)
- `analogical_similarity`, `relational_analogy` — structural mapping (canopy)

**Naive sees but transfer cannot:**

The naive organism developed these directly in B. The transfer organism either never had them or lost them during transfer because the prerequisite chain was disrupted.

- `metacognition` — modeling own processing state (canopy)
- `theory_of_mind` — recursive belief attribution (canopy)
- `epistemic_strategy` — deliberate epistemic state management (canopy)
- `it_follows` — general valid inference detection (canopy)
- `language_grounding` — symbol-to-receptor termination (canopy)

This is a striking asymmetry. The transfer organism has richer *causal and planning* receptors (regret, long-range causation, counterfactual). The naive organism has richer *epistemic and social* receptors (metacognition, theory of mind, epistemic strategy). Their canopies are biographies of different selection histories.

## Interpretation

### Communication is bounded by the intersection topology

The 134 shared receptors define what these organisms can communicate about. If one organism tries to convey "regret" (a distinction it makes via the counterfactual comparison receptor), the other has no receptor for it — the concept lands on deaf ears. Not because the word is unknown, but because the distinction the word points to doesn't exist in the receiver's quotient space.

This is von Uexküll's Umwelt made precise: the organisms inhabit overlapping but non-identical worlds. The overlap is where mutual understanding lives. The non-overlap is where miscommunication is structural, not correctable by more explanation.

### The common refinement IS the shared world

In topological terms: the intersection of two topologies is the coarsest topology that both refine. States that are distinguishable in both organisms' worlds are distinguishable in the shared world. States that are distinguishable in only one organism's world collapse in the shared world.

The common refinement of the transfer and naive topologies has 134 receptors — 134 dimensions along which both organisms can agree that two states are different. This is the coordinate system of their shared reality.

### Connection to discuss2.txt

This is the "minimum trunk" question: what is the irreducible common ground without which communication fails entirely? The 18 invariant receptors (from the 80-gen deep time run) are the candidate answer — they're present in every generation under every condition. The 134 shared receptors here are a broader measure: the common ground between two specific evolutionary histories in the same environment.

The specialization theorem predicts that as topologies deepen, the intersection shrinks relative to the union. These organisms have relatively short histories (10 generations each). With 80+ generations of divergent evolution, the naive-only and transfer-only sets would grow while the shared set might shrink — mutual intelligibility degrading as a structural consequence of the frontier's advance.

### Prediction

Run the same experiment with longer evolutionary histories (40+ generations in separate environments before transfer). The prediction: the intersection shrinks, the untranslatable sets grow, and the Jaccard similarity drops below 0.50. If it holds, the framework has derived communication breakdown from topological divergence.

# Environment Search: DARTS-Analogue for Receptor Topology Growth

## The Idea

Use a form of MCTS to search over environment configurations for the one that produces the most novel receptor growth. Not neural architecture search — environment architecture search. The curriculum that teaches the organism what it doesn't yet know how to learn.

The key insight: you don't need a full simulation per rollout. The mental model's `predict_delta` is already a simulator of the current environment. Perturb its inputs during a live simulation to explore counterfactual environments without running new episodes.

## Why This Is Different From Existing Curriculum Learning

Current curriculum learning optimizes for task performance: find the training sequence that maximizes accuracy on a fixed objective. This optimizes for **receptor growth**: find the environment configuration that makes currently-absent receptors fitness-positive. The value function isn't performance. It's distance from the current topology to the genome's specification of what's missing.

The organism isn't just learning in an environment. It's searching for the environment that develops cognitive capabilities it doesn't yet have.

## The Search Space

A continuous vector where each dimension is an environment property and zero means absent.

### Continuous parameters (smooth perturbation)
- Pain/endorphin/temperature/chemical field intensities and gradients
- NPC behavior: erraticism, deception probability, cooperation threshold
- Hidden confounder transition probabilities (how fast hidden state changes)
- Cross-modal correlation strength (how tightly modalities co-vary)
- Rule rotation frequency (non-stationary rule change rate)
- Resource depletion rates
- Predator sweep frequency and intensity

### Discrete additions (parameterized as continuous)
Instead of "add a lever" (discrete), represent as "object with affordance strength X, interaction distance Y, force multiplier Z" (continuous). Set affordance strength to zero = lever absent. Raise gradually = lever fades in. Every discrete addition becomes a point in a continuous space that includes "absent" as the origin.

- Object types: affordance strength, interaction distance, force multiplier
- NPC count: represented as weight on additional NPC observation channels
- Causal graph complexity: number of zones, edge density, hidden variable count
- Skill zone difficulty: scaling parameter per zone

## The Cross-Cut Mechanism

During step t of a live simulation:

1. Organism has observation `x_t` and mental model `M`
2. Perturb `x_t` by modifying specific channels:
   - "What if pain were 1.5x here?"
   - "What if a second NPC were nearby?"
   - "What if the chemical-temperature correlation were stronger?"
3. Query `predict_delta(perturbed_x_t, action)` for each perturbation
4. Run the 182 receptor tests on the perturbed predictions
5. Check: does the perturbation activate receptor tests that the unperturbed doesn't?
6. Score: number of currently-absent receptors that would become detectable

This is one predict_delta call per perturbation per step — not a full simulation. The mental model IS the environment simulator.

## Value Function

The genome provides the target. For each of the 188 receptors:
- If already discovered: value = 0 (no need to search for it)
- If absent: value = receptor's fitness contribution if discovered

The environment perturbation score:

```
Score(perturbation) = Σ_{r ∈ absent_receptors} I(r activates under perturbation) × w_r × certainty
```

Where:
- `I(r activates)` = does the receptor test fire on the perturbed mental model?
- `w_r` = the receptor's predicted fitness contribution (from genome specification)
- `certainty` = predict_delta's certainty for the perturbed query

## Certainty-Weighted Search (The Key Constraint)

The mental model was trained in the current environment. Perturbations too far from the training distribution produce unreliable predictions.

Fix: weight the receptor score by predict_delta's certainty.

```
Effective_score(perturbation) = raw_score × mean_certainty(perturbation)
```

- High-certainty perturbations: full receptor value (the model knows what would happen)
- Low-certainty perturbations: discounted (the model is guessing)
- Very low certainty: near-zero score (too far from training distribution)

The search naturally stays local because distant perturbations have low certainty. The model-exploitation failure mode is prevented by the certainty weighting.

## The Iterative Process

1. **Search**: Find the best small perturbation from the current environment (MCTS/UCB over environment parameters, using cross-cut queries as rollouts)
2. **Apply**: Actually modify the environment with the best perturbation
3. **Adapt**: Run a few generations in the new environment (mental model updates, organisms evolve)
4. **Search again**: From the new environment, find the next best perturbation
5. **Repeat**: The cumulative trajectory walks toward the environment that produces the most novel receptors

This is gradient following in environment space. Each step is small enough that the mental model's predictions are reliable, but the cumulative path walks toward maximal receptor growth.

## MCTS Structure for Environment Search

```
Root: current environment configuration (vector of parameters)
Children: perturbations (increase pain +20%, add NPC, strengthen cross-modal correlation)
UCB: Q(config) + c * sqrt(ln(N_parent) / N(config))
Rollout: cross-cut queries against mental model (fast — no full simulation)
Value: number of absent receptors that activate, weighted by certainty
Backpropagate: update visit counts and mean values
```

The UCB exploration bonus drives the search toward configurations whose receptor consequences are uncertain — where the search doesn't yet know what the perturbation would produce.

## Connection to the RTDS Formalization

Environment search is the fourth evolutionary operator:

1. **Selection** (U_select): which organisms survive
2. **Inheritance** (U_inherit): what offspring start with
3. **Discovery** (U_discover): what new receptors emerge
4. **Environment search** (U_search): what environment to develop in next

```
W_{t+1} = U_search(W_t, R_t, G, M_t)
```

The search operator takes the current environment, current topology, the genome (what's missing), and the mental model (what's predictable), and produces the next environment.

## Connection to Environmental Augmentation Family

The environmental augmentation receptor family (deliberate_complexification, developmental_environment_engineering) describes the organism's ability to modify its own environment. Environment search is the algorithmic mechanism for that ability.

When the environment search operator finds a productive perturbation and the organism's environmental augmentation receptors are active, the organism can apply the perturbation itself — designing its own curriculum. This closes the loop: the organism searches for the environment that develops it, then builds that environment.

## Implementation Path

### Phase 1: Offline search (designer-guided)
- After a deep time run plateaus, run environment search to find what perturbation would unlock the most missing receptors
- Apply the perturbation and continue the deep time run
- Measure: do more receptors emerge? Which ones?

### Phase 2: Online search (organism-guided)
- During a deep time run, the environment search runs as a background process
- Every N generations, the search suggests a perturbation
- The perturbation is applied automatically
- The organism's environmental augmentation receptors fire on the change

### Phase 3: Self-directed search (organism autonomous)
- The organism's own thinking substrate runs environment search
- The MCTS tree has a branch for "what environment should I be in?"
- The organism actively seeks environments that develop capabilities it's missing
- The genome's specification of absent receptors IS the organism's developmental drive

## What This Produces That Nothing Else Does

- **NAS** finds the best architecture for a fixed task
- **Curriculum learning** finds the best training order for a fixed architecture
- **Environment search** finds the best environment for growing the most cognitive capability

The organism builds the world that builds the organism that builds the next world. This is the algorithmic mechanism for the third clause.

## Dependencies

- Mental model with predict_delta (exists)
- Receptor discovery battery (exists, 182 tests)
- Genome specification of missing receptors (exists, 188 - 161 = 27 missing)
- Certainty-weighted predict_delta (exists — certainty is already returned)
- Parameterized environment (partially exists — TieredEnvironment has adjustable parameters)

## Open Questions

- How many perturbation dimensions can the search handle before the space is too large?
- Does the certainty weighting prevent all model-exploitation failures, or are there edge cases?
- Can the search discover environment properties that aren't in the current parameterization?
- What's the right step size for perturbations — too small and the search is slow, too large and predictions are unreliable?
- Should the search tree persist across generations or restart each time?

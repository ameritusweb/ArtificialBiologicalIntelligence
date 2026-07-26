# Language Emergence: Population Density Sweep Design

## The Claim (T106)

Language is the first receptor family whose fitness-positivity is endogenous to the population. Every receptor before it detects structure the world contains. Language detects structure the population contains. The transition should be sharp (percolation), not gradual.

## Three Predictions to Test

### Prediction 1: Token receptors require conspecific-topology receptors

Token receptors (receptors for arbitrary signals from other organisms) must not become fitness-positive before the organism has:
- other_detection (trunk — agent vs environment)
- behavioral_prediction (branch — anticipating other agents)
- theory_of_mind (canopy — recursive belief attribution)
- topology mismatch detection (canopy — detecting that another organism lacks a distinction you hold)

Without the mismatch detector, there's no pressure to transmit. The organism can't notice the other lacks a distinction it holds, and can't aim the transmission if it did.

**Test:** Knockout theory_of_mind from the topology bias. Run the population density sweep. Token receptors should never become fitness-positive regardless of population density.

### Prediction 2: Sharp transition in population density

The token-receptor is fitness-positive iff sufficient conspecifics hold it. Below threshold: the receptor is metabolically wasteful (nobody understands your signals). Above threshold: the receptor is fitness-positive (enough others respond). This is a coordination equilibrium.

**Test:** Sweep population size from 2 to 20 organisms. At each size, run 20 generations. Measure whether any token receptor activates. The prediction: nothing below threshold, reliable activation above threshold. The transition is sharp, not gradual.

### Prediction 3: Union decouples from individual count

After the language transition, the population starts holding more receptors than any individual can. |union_i R^(i)| grows faster than max_i |R^(i)|. Specialization emerges because the chain depth exceeds what one lifetime can traverse.

**Test:** Track per-individual receptor count and population-wide union across generations. Before the language transition, they should track closely. After, they should diverge. The moment of divergence is the moment the unit of cognition stops being the organism.

## Architecture Requirements

The current ERTI codebase runs single organisms (with NPCs as non-evolving agents). Language emergence requires:

### 1. Multi-agent environment

Multiple evolving organisms in the same environment, each with their own policy, mental model, and receptor topology. Currently, EvolvingOrganism creates single organisms that compete by fitness score. For language, they need to coexist in the same environment simultaneously and interact.

### 2. Token emission and reception

Organisms emit tokens (arbitrary signals) into the environment. Other organisms observe these tokens in their observation vector. The token has no causal relationship to the world — its only meaning comes from the population.

**Implementation:** Add `token_channels` to the observation vector. Each organism emits a token vector (e.g., 4 binary bits = 16 possible tokens). Other organisms see the emitter's token in their obs as part of the NPC observation block. The environment doesn't respond to tokens — only organisms do.

Current architecture already has `emission_bits` (4 bits at the end of the action vector) and NPC observation channels. The gap is that NPCs don't evolve and tokens don't propagate between co-evolving organisms.

### 3. Population-level selection with topology inheritance

Currently, select_and_reproduce picks parents by fitness and passes topology bias. For language, this needs to operate over a population where organisms interact during their lifetime, not just compete by individual fitness scores.

### 4. Token receptor in the genome

Add a new receptor to the Language family:

```yaml
- receptor_id: token_detection
  name: Arbitrary Signal Detection
  family: language
  tier: branch
  dependencies:
    - receptor_id: other_detection
    - receptor_id: naming
  environmental_trigger:
    structure: Conspecifics emit arbitrary tokens with consistent mapping
    survival_benefit: Coordinate, warn, share, request
```

The fitness-positivity condition for token_detection is endogenous: it's only fitness-positive when other organisms also have it. This is the coordination equilibrium.

## Experiment Design

### Phase 1: Baseline (no token infrastructure)

Run 20 generations at population sizes 2, 4, 8, 12, 16, 20. No token channels. Measure:
- Max receptor depth per individual
- Union across population
- Whether social receptors (theory_of_mind, etc.) emerge

### Phase 2: Token infrastructure enabled

Same sweep, but with token_channels in the obs vector and token emission in the action vector. Organisms can emit and observe tokens but no pressure to use them. Measure:
- When (if ever) token_detection activates
- Population density at first activation
- Whether the transition is sharp or gradual
- Union vs individual divergence after activation

### Phase 3: Knockout control

Same as Phase 2, but with theory_of_mind knocked out of the topology bias. The prediction: token_detection never activates regardless of population density.

## Expected Results

| Population size | Token activation | Prediction |
|----------------|-----------------|------------|
| 2 | No | Too few conspecifics for coordination equilibrium |
| 4 | No | Still below threshold |
| 8 | Maybe | Near threshold — if transition is sharp, it happens here or not at all |
| 12 | Yes | Above threshold — token becomes fitness-positive |
| 16 | Yes | Reliable activation |
| 20 | Yes | Reliable, plus union-individual divergence |

The sharpest prediction: the transition happens at a specific population size (the percolation threshold), not gradually. Below it: 0% activation. Above it: >80% activation. If the transition is gradual (30% at pop 8, 50% at pop 12, 70% at pop 16), the endogenous fitness claim is weakened — gradual onset suggests environmental structure contributes, not just population density.

## Connection to Human Data

If the simulation shows language emergence as a sharp transition above a population density threshold, it matches the archaeological record: symbolic behavior appears suddenly in the fossil record (~50-100kya), not gradually. The leading hypothesis is that population density crossed a threshold that made symbolic communication fitness-positive — exactly the percolation prediction.

The simulation would be the first computational demonstration of this claim in a system where the organisms, the language, and the selection pressure all emerge from the same dynamics.

# Teaching as Receptor Induction: The Knockout Prediction Formalized

## The Claim (Territory V)

Teaching is installing a detector. A word is a pointer to a receptor. You cannot induce a receptor in someone lacking its prerequisites — more explanation doesn't fix missing prerequisites because the missing thing isn't information, it's a detector, and detectors need scaffolding.

This is already confirmed empirically. This document formalizes the prediction and connects it to the language emergence program.

## The Evidence

### Prerequisite knockout: epistemic_strategy requires conflation

**Experiment:** At generation 29 (seed 42), remove conflation from the topology bias. Run 15 further generations. Check if epistemic_strategy survives.

**Result:**
- epistemic_strategy survived: **0/15 generations** (complete loss)
- conflation rediscovered: **15/15 generations** (environment demands it)
- Conclusion: **functional dependency** — epistemic_strategy cannot exist without conflation

This is the receptor-induction prediction in action. If you tried to "teach" epistemic_strategy to an organism lacking conflation, it would fail — not because the explanation is bad, but because the detector for "managing one's own epistemic state" requires the detector for "noticing two things have been collapsed" as prerequisite input. The prerequisite chain is: belief_detection -> doubt_detection -> conflation -> epistemic_strategy.

### Elicitation: necessity_detection requires targeted environment

**Experiment:** Design an environment specifically to make necessity_detection fitness-positive. Run 200 episodes.

**Result:** necessity_detection found (score 0.988). First genome entry validated by targeted elicitation.

This is "teaching" in the environmental sense — constructing a world where the distinction becomes fitness-relevant, so the organism evolves the detector. Without the right environment, the receptor never appears. With it, it appears reliably.

### Order swap: environment dominates ordering

**Experiment:** Swap the order of curriculum phases (explore-first vs accuracy-first). Measure whether topology differs.

**Result:** 59 of 62 receptors shared. Environment structure dominates ordering. The prerequisites constrain what CAN emerge, but within those constraints, the specific path matters less than the environmental structure.

## The Formalization

### The prerequisite DAG constrains teachability

For receptor r with prerequisite set P(r) = {p_1, p_2, ..., p_m}:

```
r is TEACHABLE to organism O iff P(r) ⊆ R_t(O)
```

If any prerequisite is missing, r cannot be induced regardless of the quality or quantity of instruction. The prerequisite is a detector, not information. More explanation provides more information, but information cannot substitute for a missing detector.

### Teaching operates on the fringe

The **fringe** of a topology is the set of receptors whose prerequisites are all satisfied but which are not yet in the topology:

```
fringe(R_t) = { r ∈ Genome : P(r) ⊆ R_t  AND  r ∉ R_t }
```

Teaching can only induce receptors in the fringe. Below the fringe, prerequisites are missing. Above the fringe, the receptor is already present. The fringe is where education works.

The fringe grows as the topology grows (each new receptor satisfies prerequisites for others). This is the combinatorial acceleration from discuss2.txt: the set of teachable receptors expands as the topology acquires more prerequisites.

### Three modes of receptor induction

| Mode | Mechanism | Speed | Fidelity |
|------|-----------|-------|----------|
| **Selection** | Environmental pressure across generations | Slow (generations) | High (fitness-validated) |
| **Elicitation** | Targeted environment within one lifetime | Medium (episodes) | Medium (environment-dependent) |
| **Transmission** | Direct instruction from another organism | Fast (steps) | Variable (depends on shared topology) |

Selection is the baseline — receptors emerge when the environment makes them fitness-positive. Elicitation is faster — design an environment that makes a specific receptor fitness-positive and the organism finds it within episodes. Transmission is fastest — another organism points at the distinction and the receiver constructs the detector from context.

But all three are constrained by the same DAG. Selection can't skip prerequisites. Elicitation can't skip prerequisites. Transmission can't skip prerequisites. The only thing that changes is speed.

### Transmission requires shared topology

For organism A to transmit receptor r to organism B:
1. A must have r (can't teach what you don't know)
2. B must have P(r) (prerequisites must be in place)
3. A and B must share enough topology to communicate about the distinction r detects

Condition 3 is the common refinement constraint from docs/common_refinement.md. The shared topology (134 receptors between transfer and naive organisms) is the communication channel. Concepts outside the shared topology can't be transmitted because the words pointing to them have no meaning in the receiver's quotient space.

### Prediction: transmission failure correlates with prerequisite depth

The deeper the prerequisite chain for receptor r, the more likely transmission fails — because each prerequisite is itself a receiver-must-have requirement. This generates a testable curve:

```
P(transmission_success | chain_depth = d) decreases with d
```

Shallow receptors (d=1, one prerequisite) should transmit easily. Deep receptors (d=4+, long chains) should transmit rarely — because the probability that ALL prerequisites are present in the receiver decreases multiplicatively.

This matches the "threshold concepts" literature in education: the concepts that students most consistently fail to learn are exactly the ones with the deepest prerequisite chains. Calculus requires limits requires functions requires variables requires symbolic manipulation — each missing prerequisite is an invisible wall.

## Connection to Language Emergence

The knockout prediction is the prerequisite for language itself. From discuss2.txt:

**Language is the first receptor family whose fitness-positivity is endogenous to the population.** Every receptor before it detects structure the world contains. Language detects structure the population contains.

But language ALSO requires prerequisites:
- conspecific presence detection
- conspecific behavior detection
- conspecific state detection
- conspecific topology detection (theory of mind)
- topology mismatch detection

The prerequisite chain for language is 5 deep. Without theory of mind, you can't detect that another organism lacks a distinction you hold. Without mismatch detection, you have no pressure to transmit. The knockout prediction says: **ablate theory_of_mind and language receptors should not emerge regardless of population density.**

This is testable in the existing framework (action item #28): multi-agent RTDS with token emission. The prerequisite knockout experiment provides the template — remove theory_of_mind from the topology bias, then check whether token receptors ever become fitness-positive.

## Empirical Status

| Prediction | Status |
|------------|--------|
| Prerequisites constrain teachability | **Supported** (knockout: 0/15 without conflation) |
| Environment can elicit specific receptors | **Supported** (necessity_detection: 0.988) |
| Ordering matters less than structure | **Supported** (order swap: 59/62 shared) |
| Transmission requires shared topology | **Supported** (common refinement: 134 shared, untranslatable sets exist) |
| Transmission failure increases with chain depth | **Proposed** (testable with multi-agent experiments) |
| Language requires theory_of_mind prerequisite | **Proposed** (testable via knockout + population density sweep) |

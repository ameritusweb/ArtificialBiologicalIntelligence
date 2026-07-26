# The Number Column: Validating the Method on Known Terrain

## The Strategy

Before taking the framework somewhere unmapped, validate it on the best-instrumented column in the whole edifice: number. The attested developmental sequence is independently confirmed across developmental psychology, comparative cognition, and mathematics education. If the framework reconstructs it correctly, the method works on known terrain. Then it can go somewhere nobody has a map.

## The Human Sequence (Known)

| Stage | Capability | Onset | Evidence |
|-------|-----------|-------|----------|
| 1 | Approximate magnitude discrimination | Birth / cross-species | Infants discriminate 8 vs 16 dots. Rats, pigeons, fish do it. Weber's law holds. |
| 2 | Exact counting (small sets) | ~Age 2-3 | Subitizing (1-4 items). No language required for small sets. |
| 3 | Exact counting (large sets) | ~Age 4, language-dependent | Counting words map to quantities. Pirahã (no count words) can't do exact large-number tasks. |
| 4 | Arithmetic | ~Age 5-7 | Addition, subtraction. Requires exact counting as prerequisite. |
| 5 | Algebra / variables | ~Age 11-14 | Symbolic manipulation. The variable concept is a known threshold. |
| 6 | Proof | ~Age 16+ / university | Logical chain verification. Most humans never acquire this. |

The key structural features:
- Stage 1 is cross-species (trunk)
- Stage 3 is language-dependent (branch, requires language family)
- Stages 5-6 require cultural transmission (canopy, requires institutions)
- Each stage is prerequisite to the next — you can't skip

## The ERTI Prediction

The mathematics family in the genome has 7 receptors:

| Receptor | Layer | ERTI prediction | Maps to human stage |
|----------|-------|----------------|-------------------|
| quantity_detection | T | Emerges first, any environment | Stage 1: magnitude discrimination |
| ratio_detection | B | Requires quantity + temporal association | Stage 2-3: stable relationships |
| structural_invariance_math | B | Requires ratio + cross-domain comparison | Stage 4: invariants across operations |
| exhaustive_search | C | Requires structural_invariance | Stage 5: systematic coverage |
| necessity_detection | C | Requires exhaustive_search | Stage 5-6: distinguishing contingent from necessary |
| proof_structure | C | Requires necessity_detection | Stage 6: logical chain verification |
| formal_composition | C | Requires proof_structure | Beyond stage 6: abstract structure combination |

### Prediction 1: Layer depth matches developmental order

The framework predicts that the acquisition order in ERTI (trunk -> branch -> canopy) matches the developmental order in humans (infant -> child -> adolescent -> adult). Specifically:

- quantity_detection (T) should emerge at generation 0 (infant)
- ratio_detection (B) should emerge at generation 1-5 (early childhood)
- structural_invariance_math (B) should emerge at generation 5-15 (middle childhood)
- necessity_detection (C) should emerge at generation 15+ (adolescence)
- proof_structure (C) should emerge late or never in many organisms (most humans never get it)

### Prediction 2: Language-dependence at the exact-counting boundary

In humans, approximate magnitude is pre-linguistic but exact counting of large sets requires count words. The framework predicts this if exact counting maps to a receptor with a Language family prerequisite.

Currently, the genome doesn't explicitly model this dependency — ratio_detection depends on quantity_detection and temporal_association, not on naming. This is a prediction the framework gets WRONG in its current form if it doesn't include the language dependency. Adding a `counting` receptor (branch, depends on quantity_detection + naming) would fix it and generate a testable prediction: organisms without the naming receptor should show approximate magnitude but not exact large-number discrimination.

### Prediction 3: Proof requires cultural transmission

proof_structure is canopy with a deep prerequisite chain. The framework predicts it should be rare — most organisms never reach it because the chain costs too much budget. In humans, most people never acquire the proof concept. This matches: proof is a culturally transmitted deep-canopy receptor that requires institutional scaffolding (education) to induce.

### Prediction 4: The column is ordered, not parallel

The framework predicts that these receptors emerge in a strict sequence, not in parallel. quantity_detection before ratio_detection before structural_invariance_math before necessity_detection before proof_structure. This matches the developmental data: no child learns proof before arithmetic, no child learns arithmetic before counting.

## How to Test in ERTI

### Test A: Single-organism developmental trajectory

Run a single organism through progressive environments with increasing mathematical structure. Track when each math receptor activates. Check:
- Does quantity_detection appear first?
- Does the activation order match the genome's layer predictions?
- Does proof_structure appear late or never?

### Test B: Language knockout

Run two conditions: one with the naming receptor available, one without. Check:
- Does quantity_detection appear in both? (predicted: yes — it's pre-linguistic)
- Does exact counting appear in both? (predicted: no without naming — this is the language dependency)
- Is the gap at the exact-counting boundary, not at approximate magnitude? (predicted: yes)

### Test C: Cultural transmission requirement

Run two conditions: isolated organisms vs population with transmission. Check:
- Do isolated organisms reach proof_structure? (predicted: rarely or never)
- Does a population with transmission reach it more reliably? (predicted: yes)
- Is there a population density threshold? (predicted: yes — the language/transmission percolation from T106)

### Test D: Budget constraint

Run the same mathematical environment at different metabolic budgets. Check:
- Does max math depth scale with B? (predicted: yes — Territory II)
- At what B does proof_structure become reachable? (predicted: high B only — deep chain cost)

## What This Validates

If Tests A-D pass:
1. The layer structure correctly predicts developmental ordering (the DAG is real)
2. The language dependency at the counting boundary matches human data (cross-family prerequisite is real)
3. The cultural transmission requirement for deep concepts matches human data (endogenous fitness is real)
4. The budget constraint on depth matches encephalization data (the knapsack is real)

This is the identification theorem in action: the decomposition makes temporal predictions that a rotated basis would get wrong. If quantity_detection and proof_structure are not separate receptors but a single "math ability" factor, the framework predicts that they always co-occur — which the developmental data flatly contradicts (infants have magnitude discrimination, adults may not have proof).

## What's Missing from the Genome

The current mathematics family has a gap: no explicit `counting` receptor between quantity_detection and ratio_detection. The human data says there should be one, with a language dependency. Adding it:

```
- receptor_id: counting
  name: Exact Enumeration
  family: mathematics
  tier: branch
  dependencies:
    - receptor_id: quantity_detection
    - receptor_id: naming  # Language family — this is the cross-family dependency
  environmental_trigger:
    structure: Discrete countable objects with language-like labels available
```

This generates the sharpest prediction in the column: counting should not emerge without naming, and the Pirahã result (no count words -> no exact large-number discrimination) is a direct consequence of the DAG.

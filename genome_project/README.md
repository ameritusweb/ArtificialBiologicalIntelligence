# The Genome Project: ERTI Receptor Library

## Overview

The Genome Project is the formal specification of the receptor search space for Evolutionary Receptor Topology Intelligence (ERTI). It defines:

- **What receptors CAN evolve** (the search space)
- **What environmental conditions make each receptor fitness-positive** (selection triggers)
- **What dependencies each receptor has** (partial ordering constraints)
- **What signatures indicate emergence** (validation criteria)

This is not a design document. It is a constraint specification. The environment decides which receptors survive. This library defines what the environment has to choose from.

---

## Document Structure

### `/families/`
YAML specifications for each receptor family. One file per family containing all receptors in that family with full metadata.

### `/docs/`
Narrative documentation, publishable as standalone papers or chapters:
- `overview.md` — The evolutionary map thesis
- `family_narratives/` — Readable walk-throughs of each family
- `evolutionary_predictions.md` — Testable predictions about emergence order

### `/schemas/`
Formal schema definitions:
- `receptor_schema.yaml` — Structure for receptor entries
- `family_schema.yaml` — Structure for family definitions
- `validation_schema.yaml` — Structure for emergence signatures

### `/validation/`
Tools and protocols for validating receptor emergence:
- Scripts to measure emergence signatures
- Environment generators for each receptor
- Statistical tests for predicted vs actual emergence order

---

## Receptor Families

### Core Perceptual-Cognitive Families
1. **Repetition** — Static to nested rhythms
2. **Association** — Spatial to abstract relational
3. **Similarity** — Perceptual to analogical
4. **Causality** — Coincidence to counterfactual reasoning

### Higher-Order Families
5. **Social** — Other-detection to nested theory of mind
6. **Compression** — Pattern recognition to hierarchical abstraction
7. **Meta-Motivational** — Curiosity, optimism, conflict (receptors about receptors)

### Foundation (from roadmap Steps 1-19)
Already implemented. These are prerequisites, not evolvable — they're the trunk everything else branches from.

---

## Usage

### For Research
1. Select target receptor from library
2. Generate environment with conditions from `environmental_trigger`
3. Run ERTI simulation with receptor discovery enabled
4. Measure emergence using signature from `emergence_signature`
5. Validate against predicted emergence order

### For Publication
1. Extract family narratives from `/docs/family_narratives/`
2. Compile evolutionary predictions
3. Present validation results against library predictions

### For Implementation
1. Parse YAML specifications
2. Generate receptor detection code from `emergence_signature`
3. Build environments from `selection_environments`
4. Implement receptor topology bias inheritance

---

## Design Principles

1. **Dependencies are generative** — The partial ordering emerges from what each receptor requires as input, not arbitrary sequencing.

2. **Environmental grounding** — Each receptor specifies exactly what environmental structure makes it worth having. No receptor without a selection story.

3. **Falsifiable predictions** — Every receptor includes measurable signatures. If the signature doesn't appear when predicted, the library entry is wrong.

4. **Hierarchical within families** — Trunk → canopy structure within each family reflects increasing sophistication and dependency depth.

5. **Cross-family composition** — Advanced receptors often require inputs from multiple families. The dependency graph is a DAG, not a tree.

6. **Negative space matters** — Some receptors are fitness-negative in certain environments. The library includes these as falsification targets.

---

## Status

- **Phase 1:** Schema definition and first family (Repetition) — IN PROGRESS
- **Phase 2:** All core families specified
- **Phase 3:** Validation tools and environment generators
- **Phase 4:** Narrative documentation for publication
- **Phase 5:** Integration with ERTI simulation

---

## Related Documents

- `../ABI_whitepaper.md` — Thesis and architectural foundation
- `../steps1.txt` — Current roadmap (Steps 1-30)
- `../mental_model_framework_v2.md` — Mental model architecture
- `../ERTI_roadmap.md` — Full evolutionary program

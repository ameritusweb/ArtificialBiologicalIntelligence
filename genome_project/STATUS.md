# Genome Project Status

**Last Updated:** 2026-07-20

---

## Completed

### Core Infrastructure
✅ Directory structure (`families/`, `docs/`, `schemas/`, `validation/`)  
✅ Formal receptor schema (`schemas/receptor_schema.yaml`)  
✅ Project README with usage guide  

### Family Specifications
✅ **Repetition family** — 6 receptors (static_repetition → causal_rhythm)  
✅ **Association family** — 8 receptors (spatial_association → relational_analogy)  
✅ **Similarity family** — 8 receptors (perceptual_similarity → structural_invariance)  
✅ **Causality family** — 11 receptors (coincidence_detection → causal_graph_reasoning)  
✅ **Agency family** — 8 receptors (controllability → niche_construction)  
✅ **Meta-Motivational family** — 13 receptors (curiosity → metacognition)  
✅ **Regulatory family** — 9 receptors (stress_detection → emotional_intelligence)  
✅ **Social family** — 14 receptors (other_detection → moral_reasoning)  
✅ **Compression family** — 12 receptors (pattern_recognition → analogy)  
✅ **Observation family** — 8 receptors (change_detection → meta_observation)  
✅ **Formalization family** — 10 receptors (pattern_recognition → theory_formation)
✅ **Mathematics family** — 7 receptors (quantity_detection → formal_composition)
✅ **Organization family** — 7 receptors (boundary_detection → organizational_mirror)
✅ **Self-Augmentation family** — 5 receptors (capability_change_detection → metamorphic_planning)
✅ **Interaction family** — 6 receptors (contact_response_detection → affordance_transfer)
✅ **Environmental Augmentation family** — 5 receptors (environmental_change_detection → developmental_environment_engineering)

**Total receptors specified:** 138 (with cross-references; ~125 genuinely unique)

### Documentation
✅ Overview document (thesis, families, usage)  
✅ Cross-family dependencies map  
✅ Song replay hypothesis (complete testable prediction)  

---

## Key Insights from Specification Phase

### Receptors Already Present in Current Implementation

Several receptors are **already implicitly implemented** in the ABI roadmap (Steps 1-26):

1. **`perceptual_similarity`** — The contrastive encoder (Step 10) learns this by construction
2. **`functional_similarity`** — Action family discovery (Step 10) groups by outcome, not by action bits
3. **`temporal_association`** — Step 5 from roadmap
4. **`causal_inference`** — Controllability signal (Step 17) + responsive objects (Step 19)
5. **`spatial_association`** — Mental model context encoding includes position
6. **`probabilistic_causation`** — Mental model certainty values are probabilistic causal strengths
7. **`controllability`** — Step 17 from roadmap (Agency family trunk)
8. **`curiosity`** — Step 11 from roadmap (Meta-Motivational family trunk)
9. **`optimism`** — Step 25 from roadmap (Meta-Motivational family trunk)
10. **`conflict`** — Step 26 from roadmap (Meta-Motivational family trunk)

**Implication:** The current architecture already has ~10 receptors from the library operational. This validates the architectural design — these receptors emerged naturally from the mental model + contrastive encoder + self-model structure.

### Testable Right Now

**`rhythm` receptor:**
- Environment: Current sim has sinusoidal field sources (perfect rhythmic structure)
- Prediction: Should emerge by generation 50-90
- Signature: Mental model time delays should cluster at oscillation period
- Test: Measure temporal delay distributions in mental model mappings

**`causal_inference` receptor:**
- Environment: Responsive objects (Step 19 already implemented)
- Prediction: Should be present now
- Signature: High controllability for emission→object response mappings
- Test: Compare controllability values for responsive vs non-responsive mappings

**Validation priority:** These two are immediately testable with current sim.

### Dependency Layers Discovered

**Layer 0 (Foundation):** 15 receptors — single-family, low dependency  
**Layer 1 (Two-family):** 12 receptors — cross-family composition  
**Layer 2 (Three-family):** 4 receptors — sophisticated integration  
**Layer 3 (Four+ family):** 2 receptors — deep canopy

**Evolutionary bottleneck:** Layer 1 receptors all wait for their cross-family prerequisites. Parallel evolution possible within layers but not across them.

### Metabolic Cost Distribution

**Very low:** 3 receptors  
**Low:** 9 receptors  
**Moderate:** 12 receptors  
**High:** 7 receptors  
**Very high:** 2 receptors  

Cost correlates with dependency depth and cross-family breadth as predicted.

---

## In Progress

### All Core Families Complete! ✅

All 8 planned families are now fully specified.

### Documentation to Write
- [ ] Family narrative documents (publishable walk-throughs)
- [ ] Evolutionary predictions compiled from all families
- [ ] Methodology guide for receptor validation

### Validation Tools
- [ ] Rhythm detection validator (testable now)
- [ ] Causal inference validator (testable now)
- [ ] Emergence signature measurement scripts
- [ ] Environment generators for each receptor

---

## Next Steps

### Immediate (This Week)
1. Build `rhythm` validator for current sim
2. Build `causal_inference` validator for Step 19
3. Measure whether these receptors are present

### Near-term (Next Month)
4. Complete Social, Compression, Meta-Motivational families
5. Write family narratives for publication
6. Build environment generators for Layer 1 receptors

### Long-term (3-6 Months)
7. Full ERTI integration (receptor discovery + topology inheritance)
8. Environmental sweep experiments
9. Validation paper: predicted vs actual emergence across environments

---

## Open Questions

### 1. Receptor Granularity
Should the library have hundreds of fine-grained receptors or dozens of coarse families? Current approach: ~10-15 receptors per family, but some (like `rhythmic_pattern` vs `nested_rhythm`) might be better as continuous gradations.

**Decision needed:** Where to draw receptor boundaries?

### 2. Negative Space
Should library include receptors that are **fitness-negative** in certain environments? E.g., "compassion" might be selected against in zero-sum competitive environments.

**Decision needed:** Include anti-receptors or only positive-fitness entries?

### 3. Meta-Receptors
Curiosity, optimism, conflict — these are receptors ABOUT other receptors. Are they a separate family or a tag across families?

**Current approach:** Separate Meta-Motivational family. But might need tagging system for cross-cutting properties.

### 4. Continuous vs Discrete
Some receptors (like similarity detection) are inherently continuous — the question is "how similar?" not "similar or not?" Should library represent this explicitly?

**Current approach:** Discrete receptor entries with `minimum_certainty` thresholds. But might need continuous strength representation.

---

## Dependencies on Other Work

**Blocks:**
- Full ERTI implementation → requires receptor discovery mechanism (not yet built)
- Environmental sweeps → requires environment generators (not yet built)
- Validation → requires measurement scripts (in progress)

**Blocked by:**
- Current roadmap completion (Steps 20-26) → provides test bed for validation
- Mental model instrumentation → need to expose certainty, controllability, temporal delays for measurement

**Parallel:**
- Family specification can continue independently
- Documentation can be written from current specs
- Validation methodology can be designed before implementation

---

## Metrics

**Specification progress:** 89 receptors ✅ COMPLETE  
**Family progress:** 8/8 core families ✅ COMPLETE  
**Documentation progress:** 2/5 core documents complete  
**Validation progress:** 0/33 receptors validated  
**Implementation progress:** ~6/33 receptors already present implicitly  

**Estimated completion:**
- All families specified: 2-3 weeks
- All documentation: 4-6 weeks  
- Validation framework: 6-8 weeks
- First empirical results: 8-12 weeks

---

## How This Fits Into ABI

**Genome Project = Phase 2 of the research program**

**Phase 1 (Current):** Build organism with foundation (Steps 1-26)
- Proves the architecture works
- Validates mental model + contrastive encoder + self-model
- Demonstrates social cognition emergence

**Phase 2 (Genome Project):** Formalize receptor search space
- Specify what CAN evolve
- Define emergence signatures
- Build validation framework
- **We are here**

**Phase 3 (ERTI):** Run evolutionary experiments
- Implement receptor discovery
- Enable topology bias inheritance
- Environmental sweeps
- Measure emergence vs predictions

**Phase 4 (Publication):** Report findings
- Which receptors emerged under which conditions
- Failures and revisions to library
- Evolutionary predictions validated/falsified
- Comparison to biological cognition

The Genome Project bridges Phase 1's proof-of-concept to Phase 3's full evolutionary program.

---

## Notes

This document is a snapshot. As families are completed and validation proceeds, this will be updated to reflect current state.

The library is a living document — receptor definitions will be revised when predictions fail. Falsification is the goal, not confirmation.

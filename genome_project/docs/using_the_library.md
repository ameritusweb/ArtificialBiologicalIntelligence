# Using the Genome Project Library

## Overview

The Genome Project library serves three purposes:

1. **Research tool** — Generate hypotheses, design experiments, validate predictions
2. **Implementation guide** — Build environments and measurement tools  
3. **Publication resource** — Extract narratives and empirical results

This document explains how to use it for each purpose.

---

## For Researchers: Hypothesis Generation

### Workflow

**1. Select a target receptor**

Browse the family YAML files to find a receptor at the appropriate tier:
- Trunk receptors test foundational capabilities
- Branch receptors test composition across trunk receptors
- Canopy receptors test sophisticated multi-family integration

**2. Check dependencies**

```yaml
dependencies:
  - receptor_id: static_repetition
    minimum_certainty: 0.6
  - receptor_id: temporal_association
    minimum_certainty: 0.7
```

Ensure all prerequisite receptors are present (either already implemented or emerged in prior generations).

**3. Design environment**

Use `selection_environments` to build test world:

```yaml
selection_environments:
  - sinusoidal_field_sources
  - periodic_obstacles
  - rhythmic_resource_availability
```

Each entry specifies environmental structure that makes this receptor fitness-positive.

**4. Predict emergence**

```yaml
predicted_emergence_order:
  generation_range: "50-90"
  confidence: high
  conditions: |
    Environment must have at least 30% spatial stability
```

This is your hypothesis: receptor X should emerge in generation range Y under conditions Z.

**5. Define measurement**

```yaml
emergence_signature:
  metrics:
    - metric_name: temporal_delay_clustering
      measurement_method: |
        Histogram of learned time delays should peak at interval T
      expected_range: "70-90% of mass within ±20% of true interval"
      detection_threshold: 0.7
```

Build measurement tool from this specification (see validation/ directory for examples).

**6. Run experiment**

- Build environment with specified structure
- Run ERTI simulation
- Measure emergence signature each generation
- Record when threshold is crossed

**7. Compare prediction to result**

If emergence happens within predicted range → prediction confirmed  
If emergence happens outside range → revise prediction or receptor definition  
If emergence never happens → falsification signal, revise entry

---

## For Implementation: Building Validators

### Example: Rhythm Detector

From `repetition.yaml`, receptor `rhythm`:

```yaml
emergence_signature:
  description: |
    Mental model learns temporal delay distributions that peak at
    recurrence interval T, not uniform distribution.
  metrics:
    - metric_name: temporal_delay_clustering
      measurement_method: |
        Compare mental model certainty growth for contexts seen N times
        vs contexts seen once. Ratio should exceed 1.5 for N >= 5.
      expected_range: "70-90% of mass within ±20% of interval"
      detection_threshold: 0.7
  statistical_test: |
    Chi-square test on temporal delay distribution vs uniform null,
    p < 0.01 for emergence.
```

Implementation (`validation/rhythm_detector.py`):

```python
class RhythmDetector:
    def detect_periodicity(self, delays, expected_period):
        # Extract histogram
        hist, bins = np.histogram(delays, bins=30)
        peak_idx = np.argmax(hist)
        peak_period = (bins[peak_idx] + bins[peak_idx + 1]) / 2

        # Calculate clustering score
        window = peak_period * 0.2  # ±20%
        in_window = sum(delays within window)
        clustering_score = in_window / len(delays)

        # Statistical test
        chi2, p_value = stats.chisquare(hist, uniform)

        # Threshold from spec
        has_rhythm = (clustering_score >= 0.7 and p_value < 0.01)

        return has_rhythm, clustering_score, p_value
```

### Template for New Validators

```python
class ReceptorValidator:
    """Validates emergence of {receptor_name} receptor."""

    def __init__(self, mental_model, environment):
        self.mm = mental_model
        self.env = environment

    def extract_signature_data(self):
        """Extract observable data from mental model state."""
        # Read from mm.store.mappings, mm.encoder, mm.experience_log
        pass

    def test_emergence_criterion(self, data):
        """
        Test whether data meets emergence threshold.

        Returns:
            dict with:
                - receptor_present: bool
                - metric_value: float
                - threshold: float
                - p_value: float (if statistical test)
        """
        pass

    def validate(self):
        """Full validation protocol."""
        data = self.extract_signature_data()
        result = self.test_emergence_criterion(data)

        if result['receptor_present']:
            print(f"{receptor_name} receptor has emerged!")
        else:
            print(f"Not yet emerged: {result['reason']}")

        return result
```

---

## For Publication: Extracting Narratives

### Family Narrative Structure

Each family YAML contains material for a publishable section:

**1. Introduction (from `family_metadata`)**

```yaml
family_metadata:
  description: |
    Receptors that detect recurring structure — from the same stimulus
    appearing in the same place, to complex nested rhythms with causal
    relationships.

  evolutionary_thesis: |
    Repetition detection reduces prediction cost. An organism that
    recognizes recurrence can cache prior outcomes...
```

This becomes the opening of the family section.

**2. Receptor Progression (trunk → canopy)**

For each receptor in order:

```markdown
### Static Repetition

**Environmental trigger:** Same stimulus appears in same location repeatedly.

**Selection pressure:** Any structured environment with stable features.
Prediction cost reduced by recognizing recurrence.

**Emergence signature:** Mental model certainty increases faster for
recurring contexts (1.5-3.0x vs novel contexts).

**Predicted emergence:** Generation 15-30 (high confidence).
```

**3. Dependencies (from cross_family_dependencies.md)**

Show how receptors compose across families:

```markdown
### Cross-Family Composition

Causal Rhythm requires both:
- `rhythm` (Repetition family) — detecting periodicity
- `causal_association` (Causality family) — understanding A causes B

This represents understanding that one periodic event *causes* another,
not just correlates with it.
```

**4. Empirical Results**

After running experiments:

```markdown
### Validation Results

**Rhythm receptor** (predicted gen 50-90, confidence: high)
- Emerged: Generation 67
- Clustering score: 0.81 (threshold: 0.7)
- p < 0.001 vs uniform distribution
- **Prediction confirmed**

**Rhythmic Pattern receptor** (predicted gen 100-150, confidence: low)
- Did not emerge by generation 200
- Environment lacked structured rhythms (every-3rd-pulse variation)
- **Prediction partially confirmed** — absence explained by environment
```

**5. Revisions**

When predictions fail:

```markdown
### Library Revisions

**Receptor:** dynamic_repetition
**Original prediction:** Generation 40-70
**Actual emergence:** Generation 105
**Revision reason:** Underestimated dependency on spatial encoding quality.
New minimum_certainty for spatial_association raised from 0.4 to 0.6.
**Updated prediction:** Generation 70-110
```

---

## Reading the YAML Specs

### Receptor Entry Structure

```yaml
receptor_id: rhythm
name: Temporal Rhythm Detection
family: repetition
tier: branch

dependencies:
  - receptor_id: static_repetition
    minimum_certainty: 0.5

environmental_trigger:
  structure: "Stimulus repeats with regular temporal interval"
  survival_benefit: "Predict WHEN stimulus will recur..."
  metabolic_cost: moderate
  selection_pressure: "Environments with periodic events..."

selection_environments:
  - sinusoidal_field_sources
  - periodic_danger_zones

unlocks:
  - anticipatory_positioning
  - rhythmic_pattern

emergence_signature:
  description: "Mental model learns temporal delay distributions..."
  metrics:
    - metric_name: temporal_delay_clustering
      measurement_method: "Histogram of delays should peak at T..."
      expected_range: "70-90% within ±20%"
      detection_threshold: 0.7

predicted_emergence_order:
  generation_range: "50-90"
  confidence: high
```

### Key Fields Explained

**`tier`** — Evolutionary layer within family:
- trunk: Early, foundational
- branch: Intermediate  
- canopy: Late, sophisticated

**`dependencies`** — Must exist before this receptor can evolve. Cross-family dependencies show composition.

**`metabolic_cost`** — Resource consumption. Higher cost = needs stronger selection pressure.

**`selection_environments`** — Where to test. Use these to build experimental worlds.

**`emergence_signature`** — What to measure. Build validators from this.

**`detection_threshold`** — Quantitative criterion. When metric exceeds this, receptor has emerged.

**`predicted_emergence_order`** — Falsifiable prediction. This is the hypothesis.

**`confidence`** — How certain we are:
- very_high: Strong theory + precedent
- high: Good theory
- medium: Plausible but speculative
- low: Exploratory
- very_low: Highly speculative

---

## Iteration Loop

### When Predictions Confirm

1. Record result in validation section
2. Use as evidence for related predictions (if rhythm emerges as predicted, rhythmic_pattern prediction gains confidence)
3. Publish as validation of library entry

### When Predictions Fail

1. **Diagnose cause:**
   - Environment didn't have required structure?
   - Prerequisite receptor absent?
   - Measurement tool incorrect?
   - Receptor definition wrong?

2. **Classify failure:**
   - **Environmental:** Prediction was conditional on structure that wasn't present
   - **Measurement:** Receptor present but signature was wrong
   - **Definitional:** Receptor concept itself was wrong

3. **Revise:**
   - Environmental failure → update `conditions` field
   - Measurement failure → update `emergence_signature`
   - Definitional failure → revise entire receptor entry

4. **Retest:**
   - Run new experiment with revised prediction
   - Iterate until prediction aligns with observation

**This is the scientific method applied to receptor evolution.**

The library is not a static specification. It's a living document that evolves as we learn what's actually evolvable under what conditions.

---

## Common Pitfalls

### 1. Confusing Correlation with Emergence

**Wrong:** "Certainty increased → receptor emerged"

**Right:** "Certainty increased in the specific pattern predicted by emergence signature, exceeding threshold, with statistical significance"

Always use the full emergence signature, not just one metric.

### 2. Ignoring Environmental Conditions

**Wrong:** "Rhythm didn't emerge → prediction failed"

**Right:** "Rhythm didn't emerge, but environment had no periodic structure. Prediction was conditional on periodicity. Experiment was invalid."

Check environmental conditions before claiming falsification.

### 3. Premature Receptor Definition

**Wrong:** "I think there should be a 'beauty detector' receptor"

**Right:** "In environments with X structure, detecting Y should provide Z survival benefit, costing W metabolically. Emergence signature: A, B, C."

Every receptor needs: environmental trigger, selection story, metabolic cost, emergence signature. If you can't specify these, it's not ready.

### 4. Overfitting to Biology

**Wrong:** "Biology has X, so the library should have X"

**Right:** "Biology evolved X under selection pressure Y. If our environment has Y, we predict X should emerge."

The library is not a copy of biology. It's a prediction about what evolves under what conditions. Biology is one data point, not the definition.

---

## Next Steps

**For your first receptor validation:**

1. Read `repetition.yaml` → pick `rhythm` (testable now)
2. Read `validation/rhythm_detector.py` → understand measurement
3. Run on current simulation → get baseline
4. Check emergence signature → compare to threshold
5. Report result → confirm or revise prediction

**For your first environment design:**

1. Pick a branch-tier receptor (not too easy, not too hard)
2. Read its `selection_environments`
3. Build world with that structure
4. Run ERTI → measure emergence
5. Compare to prediction

**For your first publication:**

1. Pick a complete family (Repetition is good start)
2. Extract narrative from YAML + docs
3. Run validation on all receptors in family
4. Report: predicted vs actual, confirmations and failures
5. Revise library based on results

The library is your experimental protocol. Use it.

# Agency and Meta-Motivational Families: Addition Summary

**Date:** 2026-07-20  
**Added:** 2 new families, 21 receptors  
**Total library:** 6 families, 54 receptors

---

## What Was Added

### Agency Family (8 receptors)

**Thesis:** Intelligence that reshapes its environment rather than merely navigating it.

**Hierarchy:**
```
Trunk:
  controllability (Step 17) — distinguish self-caused from external

Branch:
  agency_salience — motivated to act when control is possible
  tool_use — extend causal reach via instruments
  environmental_manipulation — persistent reshaping

Canopy:
  niche_construction — change environment that changes offspring
  distributed_agency — control via other agents
  long_range_causation — delayed/distant effects
```

**Key insight:** Agency is distinct from Causality. Causality is understanding "A causes B." Agency is exercising that understanding — being *motivated* to produce A to get B, and reshaping the world to make desired B's more achievable.

**Already implemented:** `controllability` (Step 17)

---

### Meta-Motivational Family (13 receptors)

**Thesis:** Receptors about receptors — managing the receptor economy itself.

**Hierarchy:**
```
Trunk:
  curiosity (Step 11) — fires on learning progress
  optimism (Step 25) — fires on predicted better future state
  conflict (Step 26) — fires when receptors demand incompatible actions

Branch:
  impulse_override — resist immediate for predicted better
  attention_control — amplify/dampen receptors by context
  regret — counterfactual outcome aversion

Canopy:
  self_regulation — strategic self-modification
  meta_planning — planning over mental states
  value_hierarchy — stable receptor priorities
  metacognition — modeling own receptor economy
```

**Key insight:** These receptors take OTHER RECEPTORS as input. They're second-order — monitoring and managing the motivational system itself.

**Already implemented:** `curiosity` (Step 11), `optimism` (Step 25), `conflict` (Step 26)

---

## Where They Came From

Your questions:
1. **"Control receptor"** → Became `controllability` (already exists) + `agency_salience` (motivation to exercise control)
2. **"Affect/change environment receptor"** → Became hierarchy from `environmental_manipulation` → `niche_construction`
3. **"Self-control receptor"** → Became hierarchy from `impulse_override` → `self_regulation` → `metacognition`

Each split into multiple receptors at different tiers because they represent distinct evolutionary stages.

---

## Cross-Family Insights

### Agency Integrates With:

**Causality:**
- `tool_use` requires `causal_chains` — three-way causation
- `environmental_manipulation` requires `causal_inference` — reshaping uses causal understanding

**Social (future):**
- `distributed_agency` requires `theory_of_mind` — treat others as controllable tools
- `niche_construction` requires `theory_of_mind` — understand offspring inherit environment

**Meta-Motivational:**
- `agency_salience` bridges controllability (agency) and motivation (meta)
- Both families about *acting* vs *understanding*

### Meta-Motivational Integrates With:

**Causality:**
- `regret` requires `counterfactual_reasoning` — fires on simulated alternatives

**Agency:**
- `impulse_override` enables control despite receptors
- `self_regulation` is meta-control complementing environmental control

**Social (future):**
- `metacognition` applies `theory_of_mind` to self
- `value_hierarchy` enables predictable cooperation

---

## Architectural Validations

**10 receptors already present** in current implementation (Steps 1-26):

**From core families:**
1. `perceptual_similarity` (implicit in contrastive encoder)
2. `functional_similarity` (implicit in action families)
3. `temporal_association` (Step 5)
4. `causal_inference` (Steps 17+19)
5. `spatial_association` (context encoding)
6. `probabilistic_causation` (certainty values)

**From new families:**
7. `controllability` (Step 17 — Agency trunk)
8. `curiosity` (Step 11 — Meta-Motivational trunk)
9. `optimism` (Step 25 — Meta-Motivational trunk)
10. `conflict` (Step 26 — Meta-Motivational trunk)

**Implication:** The roadmap Steps 1-26 already implements trunk receptors from FOUR families (Causality, Similarity, Agency, Meta-Motivational). This validates the architectural design's evolutionary naturalness.

---

## New Conceptual Spaces Opened

### Self-Control as Layered Architecture

**Not a single receptor** but a developmental progression:

1. **Detect conflict** (Step 26) — receptors want different things
2. **Override impulse** — choose against immediate gradient when future is better
3. **Control attention** — amplify/dampen specific receptors by context
4. **Self-regulate** — plan how to put yourself in right mental state
5. **Meta-plan** — sequence mental states across task phases
6. **Form values** — stabilize receptor priorities as identity

Each layer builds on previous. Can't have self-regulation without impulse override. Can't have metacognition without self-regulation.

### Agency as World-Shaping

**Not just "I can cause things"** but a progression:

1. **Detect controllability** (Step 17) — I caused that change
2. **Value agency** — be motivated when control is possible
3. **Use tools** — extend causal reach via instruments
4. **Manipulate environment** — persistent reshaping for future benefit
5. **Construct niche** — shape environment that shapes offspring
6. **Distribute agency** — control via other agents

This is the progression from "I affect my immediate surroundings" to "I engineer evolutionary selection pressures."

---

## Dependency Layer Distribution

**Layer 0 (Foundation):** 4 new receptors
- `controllability`, `curiosity`, `optimism`, `conflict`

**Layer 1 (Two-family):** 8 new receptors
- `agency_salience`, `tool_use`, `impulse_override`, `attention_control`, `regret`, etc.

**Layer 2 (Three-family):** 5 new receptors
- `environmental_manipulation`, `distributed_agency`, `self_regulation`, etc.

**Layer 3 (Four+ family):** 4 new receptors
- `niche_construction`, `metacognition`, `value_hierarchy`, `meta_planning`

**Pattern:** Meta-Motivational family has more canopy-tier receptors than any other family. This makes sense — managing the receptor economy only becomes complex when there's a rich economy to manage.

---

## Testability

### Immediately Testable (Already Implemented)

1. **`controllability`** — Step 17, measure controllability values in mental model
2. **`curiosity`** — Step 11, measure learning progress correlation with action
3. **`optimism`** — Step 25, measure goal persistence through pain
4. **`conflict`** — Step 26, measure slow pathway recruitment correlation

### Testable Soon (Within Current Architecture)

5. **`agency_salience`** — Does action rate correlate with controllability?
6. **`impulse_override`** — Can organism delay gratification when optimism predicts better future?

### Requires Environment Design

- `tool_use` — needs objects that can be manipulated to affect distant targets
- `environmental_manipulation` — needs persistent modifiable structures
- `niche_construction` — needs multi-generation modifiable environments

---

## What This Completes

### Before Today
- 4 families (Repetition, Association, Similarity, Causality)
- 33 receptors
- Focused on perceptual-cognitive primitives

### After Today
- 6 families (+ Agency, + Meta-Motivational)
- 54 receptors
- Covers full range from sensation → action → meta-control

### Still Missing
- **Social family** (other-detection → nested theory of mind) — HIGH PRIORITY
  - Many canopy receptors in Agency and Meta-Motivational require Social
  - `distributed_agency`, `niche_construction`, `metacognition`, `value_hierarchy` all need `theory_of_mind`
- **Compression family** (pattern recognition → concept formation)
  - Step 12 from roadmap is trunk receptor
  - Connects to language (Step 30)

---

## Impact on Evolutionary Map

### New Bottlenecks Identified

**Social family becomes critical integration point:**
- 6 canopy receptors across Agency and Meta-Motivational depend on `theory_of_mind`
- Without Social family, evolutionary progression stalls at Layer 2

**This predicts:** Social cognition emergence should trigger cascade of canopy receptor emergence across other families.

**Biological parallel:** Primate social complexity correlates with general cognitive sophistication. The library now formally predicts this dependency.

### New Testable Predictions

1. **Agency salience should emerge after controllability** (Step 17) in environments with variable controllability
2. **Impulse override requires all three:** `conflict` (Step 26), `optimism` (Step 25), `planning` (Step 18)
3. **Curiosity should correlate with exploration in low-certainty regions** (measurable now)
4. **Value hierarchy should be latest-emerging meta-motivational receptor** — requires stable social environment

---

## Files Added

- `genome_project/families/agency.yaml` — 8 receptors, full specs
- `genome_project/families/meta_motivational.yaml` — 13 receptors, full specs
- Updates to `cross_family_dependencies.md` — new integration points
- Updates to `STATUS.md` — progress metrics

**Lines added:** ~1,200 lines of formal specifications

---

## Next Steps

### Immediate
1. Test curiosity receptor (Step 11) — does exploration correlate with low certainty?
2. Test agency salience — does action rate correlate with controllability?
3. Test impulse override — delayed gratification experiments

### Near-term
1. Complete Social family specification (HIGH PRIORITY)
2. Complete Compression family specification
3. Write family narratives for Agency and Meta-Motivational

### Long-term
1. Implement receptor discovery mechanism for ERTI
2. Environmental sweeps testing Agency receptors
3. Validation paper on meta-motivational emergence

---

## The Deeper Achievement

**You identified a conceptual gap:** The library had receptors for understanding causation but not for *exercising* agency. It had receptors for motivation but not for *managing* motivation.

**The addition of these families completes the architecture:**

- **Perception** (Repetition, Association, Similarity) — detecting structure
- **Understanding** (Causality) — causal reasoning  
- **Acting** (Agency) — exercising control and reshaping world
- **Managing** (Meta-Motivational) — regulating own receptor economy

This is the full cognitive stack: sense → understand → act → self-regulate.

**And the meta-insight:** Self-control isn't one thing. It's a hierarchy from impulse override → attention control → self-regulation → metacognition → value formation. Each tier compositional over the previous.

The Genome Project now has formal specifications for how organisms evolve from reactive navigation to deliberate self-engineering.

That's the completion of Phase 1 of the library.

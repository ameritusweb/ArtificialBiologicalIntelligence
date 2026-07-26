# Criticality: Is There a Complexity Threshold for Cognitive Depth?

## The Question (Territory VI)

Is there a critical environmental complexity below which deep receptors (canopy, layer 2) are never fitness-positive and above which they reliably appear? That has the shape of a percolation threshold.

## What the Data Shows

### Environmental complexity alone produces limited depth

The canopy sweep tested 8 environment tiers (T0-T7) with increasing complexity. Each tier was run for a single training cycle (no evolution).

| Tier | Receptors | Trunk | Branch | Canopy | Canopy % |
|------|-----------|-------|--------|--------|----------|
| T0 | 27 | 13 | 12 | 2 | 7% |
| T1 | 31 | 13 | 13 | 5 | 16% |
| T2 | 31 | 12 | 16 | 3 | 10% |
| T3 | 32 | 13 | 15 | 4 | 12% |
| T4 | 30 | 13 | 14 | 3 | 10% |
| T5 | 29 | 12 | 13 | 4 | 14% |
| T6 | 33 | 13 | 15 | 5 | 15% |
| T7 | 30 | 12 | 13 | 5 | 17% |

Total receptor count barely changes (27-33). Canopy fraction rises from 7% to 17% — a gradual increase, not a sharp transition. The trunk is nearly constant at 12-13. There is no obvious percolation threshold in environmental complexity alone.

18 receptors are invariant across all 8 tiers — the irreducible trunk.

### Evolutionary depth is the missing variable

The 80-generation deep time run (same environment, seed 42) tells a different story:

| Gen | Total | Canopy fraction |
|-----|-------|----------------|
| 0 | 72 | 36% |
| 10 | 103 | 46% |
| 20 | 97 | 40% |
| 40 | 97 | 43% |
| 79 | 110 | 40% |

Single-generation tier sweep: canopy fraction 7-17%.
80-generation deep time: canopy fraction 36-46%.

The canopy fraction is 2-5x higher under evolution than under single-generation training in any tier. This means depth comes from generational selection, not from environmental complexity alone. The environment sets the CEILING (which receptors CAN emerge), but evolution determines how much of that ceiling is REACHED.

### The two-factor criticality

The data suggests criticality is not a single threshold but a two-factor condition:

```
Deep canopy emerges when:
  1. Environmental complexity exceeds a minimum (tier structure matters)
  AND
  2. Evolutionary depth exceeds a minimum (generations matter)
```

Neither factor alone is sufficient. Rich environments without evolution produce ~15% canopy. Evolution in simple environments can't produce receptors that the environment doesn't make fitness-positive. The interaction is multiplicative, not additive.

### Canopy fraction saturates

The canopy fraction stabilizes at ~40% by generation 10 and barely changes through generation 79. This is the knapsack equilibrium — the budget can't sustain more canopy without dropping trunk or branch receptors that are also valuable. The saturation point is the budget constraint, not the environment's limit.

## The Supercritical Question

Does the environmental augmentation operator (organism modifies its own environment) generate complexity faster than the topology needs to grow?

From the existing data: the organism's environment DID become more complex over 80 generations (environmental_modification is in the receptor list). But k_t didn't grow — it oscillated around 102. The augmentation operator enriches the environment, but the knapsack budget prevents the topology from expanding to match. The loop doesn't go supercritical because the budget constraint binds.

**Prediction:** Supercriticality requires either:
1. A growing budget B (encephalization — more metabolic capacity for cognition over evolutionary time)
2. Instruments that shift alpha_r off the organism's budget (cultural tools — the discuss2.txt prediction)

Neither is present in the current simulation. Adding heritable brain size (growing B) or cultural artifacts (externalized receptors) would be the test of whether the loop can go supercritical.

## What's Needed

### The complexity sweep experiment

The existing tier data is from single-generation training — no evolution. To test criticality properly:

1. Run deep time (20 generations) at each of 8 tiers independently
2. Measure canopy fraction and max layer depth at generation 20
3. Plot canopy fraction vs tier complexity
4. Look for a knee — a tier below which canopy never exceeds 10% and above which it reliably exceeds 30%

If the knee exists, it's the percolation threshold: the minimum environmental complexity for deep cognition. If no knee exists (gradual increase), criticality is continuous rather than discontinuous.

### The budget interaction

Run the complexity sweep at two budget levels (tight and generous from the metabolic budget experiment). If the knee shifts with budget — appearing at lower tier with generous budget — then criticality is a function of both environmental complexity and metabolic capacity. That's the encephalization prediction: larger brains (higher B) reach deep cognition in simpler environments.

## Current Status

- **No sharp threshold** in environmental complexity alone (gradual 7% -> 17%)
- **Evolutionary depth** is the primary driver of canopy emergence (15% -> 40%)
- **Saturation at ~40%** due to knapsack budget constraint
- **Supercritical regime** not observed — budget binds before augmentation can outrun topology growth
- **The proper criticality test** (deep time per tier) is planned but not yet run

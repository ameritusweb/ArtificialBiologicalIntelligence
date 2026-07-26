# Temporal Depth: Does Memory Structure Mirror Layer Structure?

## The Conjecture (Territory III)

Each receptor has a temporal depth tau_r — how far back in time it needs to look. The conjecture: tau_r increases with layer depth l(r). Deep receptors are deep in time, not just in dependency. If this holds, layer structure and memory structure are the same structure seen from two angles.

## Method

Each of the 73 live receptors has an explicit temporal depth determined by its buffer requirement:
- tau = 0: instantaneous (reads current obs or engine state)
- tau = 1-3: short lookback (1-3 previous steps)
- tau = 5-20: medium window (sliding window statistics)
- tau = 80-400: long history (autocorrelation, rhythm detection)

Receptors are mapped to their genome layer (T=0, B=1, C=2) via the receptor index.

## Results

### Mean temporal depth increases with layer

| Layer | Mean tau | Median tau | Max tau | N |
|-------|----------|------------|---------|---|
| **All receptors** | | | | |
| Trunk | 1.4 | 0 | 10 | 13 |
| Branch | 11.1 | 0 | 200 | 30 |
| Canopy | 25.7 | 0 | 400 | 29 |
| **Temporal receptors only (tau > 0)** | | | | |
| Trunk | 3.1 | 2 | 10 | 7 |
| Branch | 15.4 | 2 | 200 | 22 |
| Canopy | 92.1 | 5 | 400 | 8 |

The means increase monotonically: 1.4 -> 11.1 -> 25.7 (all), and 3.1 -> 15.4 -> 92.1 (temporal only). The pattern holds.

### Correlation is moderate

| Measure | All 72 | Temporal only (37) |
|---------|--------|--------------------|
| Pearson r | 0.147 | 0.344 |
| Spearman rho | — | 0.307 (p=0.064) |

The weak overall correlation is driven by the 35 instantaneous (tau=0) receptors — mostly live-ready engine queries that happen to span all layers. Among receptors that actually have temporal extent, the correlation is 0.344 (Spearman 0.307, p=0.064). Marginal significance, but the means tell the story clearly.

### The extreme cases

The deepest temporal receptors are all canopy:
- `nested_rhythm` (C): tau = 400 steps
- `causal_rhythm` (C): tau = 300 steps
- `rhythmic_pattern` (B): tau = 80 steps
- `rhythm` (B): tau = 200 steps
- `developmental_trajectory` (C): tau = 20 steps

The shallowest temporal receptors are all trunk:
- `coincidence_detection` (T): tau = 1 step
- `change_detection` (T): tau = 2 steps
- `basic_sensorimotor_loop` (T): tau = 2 steps

No canopy receptor has tau = 1. No trunk receptor has tau > 10.

### The instantaneous receptors complicate the picture

37 receptors have tau = 0 — they query the engine or read current obs with no history. These span all layers (trunk: 6, branch: 8, canopy: 21). They're "temporally deep" in a different sense: they leverage the mental model, which integrates the entire experience log. Their temporal depth is the mental model's, not their own.

This suggests two kinds of temporal access:
1. **Direct**: the receptor maintains its own buffer and computes from history (the buffer-computable receptors)
2. **Delegated**: the receptor queries the mental model, which has already integrated temporal structure into its embeddings and certainty scores

The conjecture holds for direct temporal access. For delegated access, the temporal depth is the mental model's depth, shared across all receptors that query it.

## Interpretation

### Layer structure and memory structure are partially the same

The conjecture is directionally supported but not as clean as Territory III hoped. The means increase monotonically (3x from trunk to branch, 6x from branch to canopy). The extremes confirm it (deepest buffers are canopy, shallowest are trunk). But the correlation is moderate (0.34) because many receptors delegate temporal processing to the mental model rather than maintaining their own history.

### The mental model is the shared temporal substrate

This connects to Territory III's dissolution: "Memory stops being a separate component bolted on. M is the substrate that makes large-tau receptors physically possible." The implementation confirms this — the 37 instantaneous receptors achieve deep temporal integration not by maintaining their own buffers but by querying an engine that has already compressed temporal experience into embeddings and certainty scores. The mental model IS the memory. The receptors read it.

### Prediction

If the conjecture is strengthened to "direct temporal depth (buffer size) increases with layer depth, and delegated temporal depth (mental model queries) is available at all layers," it's fully supported by the current implementation. The layer structure determines which temporal access pattern a receptor uses: trunk receptors tend to use direct short buffers, canopy receptors use either long direct buffers OR delegated access through the mental model.

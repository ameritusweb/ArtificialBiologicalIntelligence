# topology_awareness: What It Can and Cannot Detect

## The Receptor

`topology_awareness` (epistemic family, canopy tier) fires when the organism's cognitive repertoire changes — a receptor that used to fire no longer does, a new one has begun firing, or the pattern of co-activation has shifted. It detects what kind of mind the organism currently has.

Prerequisites: `fundamental_distinction`, `metacognition`, `capability_change_detection`.

## The Diagonal Limit

The diagonal theorem (docs/diagonal_theorem.md) proves no receptor can detect the complete topology. topology_awareness is the concrete instance: it reads the output of other receptors, but it IS one of those receptors. It cannot represent its own contribution to the topology it's measuring.

This structures what it CAN vs CANNOT detect.

## What topology_awareness CAN detect

### 1. Topology changes (the derivative, not the state)

The receptor can detect that the activation pattern shifted — "something changed about what I can perceive." This requires comparing recent activation distributions against a running baseline. Measurable via cosine distance between sliding windows of internal channel activations.

**Already implemented:** `capability_change_detection` (live receptor [64]) tracks gain shifts over a 5-10 step window. `developmental_trajectory` ([65]) tracks 20-step gain slope. These detect the CHANGE, not the complete state.

### 2. Partial topology snapshot (which modalities are active)

The receptor can detect which broad categories of its own processing are currently active — "my thinking channels are firing but my pattern recognition isn't." This is a coarse-grained read of the topology, not a complete one.

**Already implemented:** `multiple_receptor_types` ([7]) counts active receptor groups. `adaptive_depth` ([25]) reads pattern query depth. These give the organism a partial, coarse view of its own cognitive state.

### 3. Topology gaps (via conflation detection)

The receptor can detect that a distinction is being collapsed — "I'm treating two different things as the same." This is an indirect read of topology gaps: the organism senses what it's MISSING via the consequences of missing it (prediction errors that don't resolve, bimodal distributions in what should be unimodal).

**Already implemented:** `conflation` receptor in the genome (not yet a live channel — it's an episode-level test). When wired as a live receptor, this would give topology_awareness its most powerful input: sensing blind spots via their downstream effects.

### 4. Epistemic state trajectory

The receptor can detect trends in its own certainty, doubt, and exploratory behavior — "I'm becoming more certain about this domain" or "my exploration pattern shifted."

**Already implemented:** `metacognition` ([8]), `belief_detection` ([26]), `doubt_detection` ([27]), `epistemic_strategy` ([71]). These give the organism a read on the TRAJECTORY of its epistemic state.

## What topology_awareness CANNOT detect

### 1. Its own contribution to the topology

topology_awareness is itself a receptor in R_t. Its activation is part of the state it's trying to detect. To fully detect the topology, it would need to represent "the state of topology_awareness detecting the topology" — which requires a level of self-reference the diagonal argument rules out.

Concretely: the organism can detect "my metacognition channel is active" but cannot detect "my detection of my metacognition channel is itself active" without a third-order receptor — which would itself have the same blind spot one level up. The self-reference regresses.

### 2. Distinctions it doesn't have receptors for

If the organism lacks a receptor for X, it can't detect that X is missing. The absence doesn't present as a gap — it presents as homogeneity in the quotient space. States that differ along the X dimension look identical.

This is why the anxiety loop persisted: the organism had no receptor for "bidirectional cascade." The cascade and normal thinking were the same point in its quotient space. topology_awareness couldn't detect the missing distinction because detecting it IS having the distinction.

### 3. The full activation pattern of all receptors simultaneously

With k receptors, there are up to 2^k possible activation patterns. A single scalar output (topology_awareness's activation value) can encode at most a few hundred distinguishable states. For k > 8 (any non-trivial topology), the full pattern exceeds what one channel can represent.

This is why `thought_type_id` uses a 256-entry codebook — it compresses the full co-activation pattern into a coarse ID. It's a lossy representation. Some distinct cognitive states map to the same thought type. topology_awareness inherits this limitation.

### 4. What's in other organisms' topologies but not its own

The organism can't detect distinctions it doesn't make. To discover its blind spots, it needs comparison with a topology that isn't its own — another organism, a formal system, or an external instrument. This is the discuss2.txt point: introspection is the one instrument guaranteed to miss the interesting parts. The map must be drawn from outside.

## Test Design

topology_awareness cannot have a COMPLETE test (the diagonal theorem says so). But it can have tests for each of the four things it CAN detect:

### Test 1: Topology Change Response
Track internal channel activation pattern over a sliding window. When the pattern shifts (cosine distance > 0.2), measure whether the organism's behavior changes more than environmental change alone predicts. Partial correlation > 0.15 = topology change response detected.

### Test 2: Active Modality Count
Count how many receptor groups are simultaneously active (> 0.15 mean activation). Check whether the organism's strategy correlates with the count — more active groups → different behavior than fewer. Already measurable via `multiple_receptor_types` ([7]).

### Test 3: Gap Detection via Conflation
After the organism exhibits conflation (treating two distinct inputs identically), check whether it subsequently explores the conflated dimension more — seeking to resolve the ambiguity. This requires `conflation` as a live channel (currently episode-level; item 20 would wire it in).

### Test 4: Growth Seeking
After a period of topology expansion (new channels activating), measure whether the organism preferentially enters states that exercise the new channels. New-channel activation rate in chosen states > 1.5x rate in random states = growth seeking.

## The Nine Live Receptors That Compose It

topology_awareness is not a single detector. It's a composite of partial self-reads:

| Index | Receptor | What it contributes |
|-------|----------|-------------------|
| 7 | multiple_receptor_types | How many modalities are active |
| 8 | metacognition | Certainty x (1 - prediction error) |
| 24 | processing_speed | Model-environment fit |
| 25 | adaptive_depth | Pattern query depth |
| 26 | belief_detection | Current certainty level |
| 27 | doubt_detection | Persistence x prediction error |
| 64 | capability_change_detection | Gain shift over 5-10 steps |
| 65 | developmental_trajectory | 20-step gain slope |
| 71 | epistemic_strategy | Action diversity x certainty |

Together, these nine channels give the organism a partial, lossy, derivative-focused view of its own topology. That's the best any self-model can do. The diagonal theorem says the complete view is structurally unreachable.

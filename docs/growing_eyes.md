# Growing Eyes: Visual Processing from Raw Waves Under Selection Pressure

The organism discovers its own visual processing. No CNN trunk. No learned maps. No imported features. Just raw wave physics, MCTS how-space exploration, and survival pressure.

This document is a firewall against three specific collapses that have happened repeatedly during implementation. Every design decision below includes an explicit check against each collapse.

---

## The Three Collapses

### Collapse 1: Import a Pre-Trained Visual System

**What it looks like:** "Let's use FlatE / a CNN trunk / learned maps to get the features, THEN run the how-space on top."

**Why it happens:** Pre-trained features produce immediate results. The FlatE trunk hits 0.740 on animal classification out of the box. Starting from raw waves produces noise for many generations before anything meaningful emerges. The temptation to "just get it working first" always leads to importing a trunk.

**Why it's wrong:** Importing a trunk imports the representation mismatch problem. The trunk's feature space shifts under training. The mental model's embeddings become unreachable. The organism doesn't own its visual processing — it's renting it from a system trained on a different objective.

**The check:** If the implementation includes `nn.Conv2d` or any learned visual encoder that maps images to features BEFORE the how-space operates, it has collapsed into option 1. The how-space must operate on wave_encode output directly.

### Collapse 2: Separate the Trunk from the Index

**What it looks like:** "Let's add a projection layer / adapter / frozen embedding that translates between the visual features and the mental model."

**Why it happens:** It's an engineering fix for the representation mismatch. Add a thin layer, freeze it, retrain when needed. Clean, professional, solves the immediate problem.

**Why it's wrong:** It adds a component the organism didn't grow. The projection layer is designed, not discovered. It assumes the right representation exists and just needs to be stabilized. The organism should discover what representation works for survival, not have an engineer decide.

**The check:** If the implementation includes any layer whose purpose is "translate between visual features and mental model space," it has collapsed into option 2. The mental model should index directly on the how-space E-profile.

### Collapse 3: Benchmark Against Classification Accuracy

**What it looks like:** "Let's compare against a CNN on 20-class accuracy to prove the approach works."

**Why it happens:** Classification accuracy is a legible, publishable metric. "We matched a CNN" is a clear claim. "The organism survived longer" is harder to evaluate.

**Why it's wrong:** Classification accuracy measures whether the organism can label images. Survival measures whether the organism can use visual information to stay alive. These are different things. A classification benchmark pressures the implementation toward importing a classifier, which is collapse 1.

**The check:** If the primary metric is accuracy on a held-out test set, it has collapsed into option 3. The primary metric must be survival fitness in a visual environment.

---

## The Architecture

### Input: wave_encode on raw pixels

Every frame is converted to a wave representation via `wave_encode(img)`:

```
img [H, W, 3] RGB float
  → HSV conversion
  → amplitude = V (brightness)
  → frequency = F_MIN + H * (F_MAX - F_MIN)  (hue → oscillation frequency)
  → saturation = S
  → wave dict: {'amp': [H,W], 'freq': [H,W], 'sat': [H,W]}
```

This is PHYSICAL. Hue literally is electromagnetic frequency. Brightness literally is amplitude. The wave encoding is not a metaphor — it's the physics of light mapped to a signal processing representation. The encoding is deterministic, parameter-free, and stable. An image produces the same wave dict every time.

**Collapse check:** No `nn.Module` between the image and the wave dict. No learned transformation. `wave_encode` is the only preprocessing.

### Processing: MCTS how-space on the wave dict

The MCTS explores programs on the wave dict. Each program specifies:
- Source component (amp, freq, sat, field)
- Target component to predict
- Operations (pool, blur, sprod, gact, roll, rot)
- Spatial radius
- Phase scheme
- Time parameters

Each simulation evaluates `measure(program, wave_dict)` → ridge R² → how well this processing configuration supports self-prediction on THIS image.

The MCTS produces:
1. **E-profile** [K dims]: distribution of prediction effectiveness across explored programs
2. **Tree analysis** [6 dims]: best_value, visit_entropy, value_convergence, path_divergence, fraction_working, depth_explored

**Collapse check:** The programs operate on `wave_dict`, not on learned maps. `measure()` uses `wave_encode` output directly. No CNN features anywhere in the evaluation.

### Per-class tree accumulation

Each class accumulates a persistent MCTS tree across training images. When a new image of class C arrives:
1. Warm-start from class C's top-K programs (discovered from previous images)
2. Run N simulations, deepening promising branches
3. Update class C's tree with the new results
4. The E-profile captures "how does this image relate to class C's accumulated processing-space knowledge"

By epoch 40, each class has accumulated thousands of simulations of processing-space knowledge. The profile for a new image is not "what programs work on this image" (stateless) but "how does this image fit into what we know about this class" (accumulated).

**Collapse check:** The per-class knowledge is in the MCTS tree (programs and their E-values), not in a learned embedding. No neural network stores the class knowledge.

### Feature vector: E-profile IS the observation

The E-profile + tree analysis vector feeds directly into the organism's observation vector as visual receptor channels. No projection. No adapter. No encoder.

```
Visual receptor channels = E-profile[0:K] + tree_analysis[0:6]
```

These channels are stable because:
- wave_encode is deterministic (same image → same wave dict)
- measure() is deterministic given a seed (same program + same wave dict → same R²)
- The programs evolve slowly (per-class trees accumulate, not reset)

The mental model indexes on these channels. The channels don't shift because there's no learned encoder to drift.

**Collapse check:** The visual channels are the raw E-profile values. No `nn.Linear`, no embedding layer, no normalization beyond clipping to [0,1]. The mental model's causal chains directly reference these channel values.

### Action selection: mental model + MCTS over ACTIONS

The organism's MCTS (the existing ThinkingTree, not the how-space MCTS) explores action sequences using predict_delta on the full observation vector — which now includes visual E-profile channels. The organism decides where to move its 22 muscles based on what its visual processing tells it about the scene.

Two separate MCTS systems:
1. **How-space MCTS**: explores processing programs (what to look at, how to process it)
2. **Action MCTS** (ThinkingTree): explores action sequences (what to do given what you see)

The how-space MCTS runs ONCE per step to produce the visual channels. The action MCTS runs ONCE per step to select actions. They don't interact directly — they communicate through the observation vector.

**Collapse check:** The action MCTS uses predict_delta on the observation vector, not on CNN features. The how-space MCTS produces channels, not classifications.

### Evolution: programs and processing strategy evolve

Heritable parameters:
- `processing_exploration_rate`: how much of the how-space to explore per step
- Per-class best programs (the warm-start seeds) — inherited from parent, mutated

The organism's visual processing improves across generations because:
1. Better programs are inherited (per-class trees carry forward)
2. The mental model accumulates more cross-context patterns involving visual channels
3. The receptor discovery mechanism (T125) finds conflation boundaries in visual channels

**Collapse check:** No trunk weights are inherited. No CNN parameters evolve. Only programs (composable operation chains on wave dicts) and processing strategy parameters evolve.

### Training: survival, not classification

The organism sees rendered frames of its environment — predators, food, obstacles, NPCs. Visual processing produces receptor channels. The mental model predicts outcomes of actions given those channels. Evolution selects organisms whose visual processing and action selection lead to survival.

There is no classification loss. There is no held-out test set. There is no accuracy metric. The only metric is fitness: total reward across episodes.

**Collapse check:** No `CrossEntropyLoss`. No `train/test split` for classification. No `accuracy` computation except as a diagnostic side-channel.

---

## The Representation Mismatch is Impossible

The somatosensory organism's mental model works because the feature space is stable. Pain at limb tip 3 is pain at limb tip 3 across generations.

The visual organism's mental model works for the same reason — but different implementation. The feature space is the E-profile over wave_encode output. wave_encode is deterministic physics. The E-profile measures self-prediction quality of deterministic programs on deterministic input. The feature space doesn't shift because nothing in it is learned.

The organism's understanding deepens through:
- More programs in the per-class trees (richer processing space exploration)
- More causal chains in the mental model (more cross-context support)
- Better programs through evolution (inherited from successful parents)

None of these change the feature space. They change what the organism DOES with the features — which programs it runs, which causal chains it trusts, which actions it takes. The features themselves are stable because they're physics, not parameters.

---

## What This Costs

### Speed
The iprocess FlatE trunk hits 0.740 in 40 epochs (~15 minutes). The wave-encode + how-space approach will take hundreds of generations to approach that accuracy — if accuracy is even the right metric.

### Legibility
FlatE's E-profile is interpretable: "cross-channel ridge R² on 8 learned maps." The how-space E-profile is less legible: "distribution of self-prediction quality across randomly mutated composable programs on HSV wave fields." The programs themselves are interpretable (each one has a describe() method), but the profile as a whole is harder to summarize.

### Comparison
There is no clean number to compare against a CNN. The organism's fitness depends on navigation, not classification. The claim is not "we matched a CNN" but "the organism learned to see by surviving." That's harder to publish but more honest.

---

## What This Gains

### No representation mismatch
The mental model's causal chains never become unreachable. The feature space is physics.

### Genuine discovery
The organism discovers which visual processing matters for survival. It doesn't inherit a human's idea of what matters (edges, textures, objects). It discovers what matters by dying when it gets it wrong.

### Philosophical consistency
The receptor topology IS the visual system. The visual processing isn't bolted on — it's grown from the same mechanism that grows somatosensory receptors, auditory receptors, proprioceptive receptors. One mechanism. No modality-specific design.

### Transferability
Programs discovered for vision can be applied to audio or touch. They're composable operations on wave fields, not CNN filters on pixel grids. An organism that discovers "sprod at radius 2 detects texture structure" in vision has a program that also detects texture structure in audio spectrograms.

---

## Implementation Steps

### Step 1: Raw visual environment

The organism lives in the rendered 64×64 environment (SceneRenderer). Each step:
1. Render the frame
2. wave_encode(frame) → wave dict
3. How-space MCTS search on the wave dict → E-profile + analysis
4. Write to visual receptor channels in the observation vector
5. The rest of the ERTI step loop runs unchanged

No FlatE. No CNN. No learned encoder. Just wave_encode + MCTS how-space.

**Files to modify:** The experiment script. NOT wave_processor.py, NOT the step loop, NOT the mental model. Only the experiment's visual processing section.

### Step 2: Per-class tree accumulation

During training, the how-space MCTS accumulates per-class trees. The organism's visual repertoire grows across episodes and generations.

During evolution, the per-class trees are inherited. Offspring start with their parent's processing-space knowledge, mutated.

**Files to modify:** howspace_mcts.py (already has per-class trees). The experiment script passes class labels during training.

### Step 3: Survival fitness measurement

The organism's fitness is total reward from navigating toward food and away from predators. No classification accuracy. The diagnostic comparison is: "does the organism with visual channels survive better than the organism without?"

Two conditions:
- **Visual**: how-space E-profile channels active
- **Blind**: visual channels zeroed

If the visual organism survives better, it learned to see. The magnitude of the fitness difference measures how much visual information the processing space map captured.

### Step 4: What does the organism see?

After training, inspect the per-class trees:
- Which programs have high E-values for predator-class images?
- Which programs have high E-values for food-class images?
- Do different classes produce different processing space maps?
- Can you interpret what the best programs detect? (e.g., "motion detection" = temporal field program with short dt)

This is the legibility step. The organism's visual processing is interpretable through the programs it discovered — not through the weights of a CNN.

### Step 5: Cross-modal transfer

Take the programs discovered for vision and apply them to audio spectrograms. Do programs that detect "motion" in visual wave fields also detect "rhythm" in auditory wave fields? If so, the processing space is genuinely modality-independent — the programs discover structure, not modality-specific features.

---

## The Firewall

Before implementing ANYTHING, check:

1. Does the implementation include any `nn.Conv2d` or learned visual encoder? → **Collapse 1. Stop.**
2. Does it include any projection/adapter between visual features and mental model? → **Collapse 2. Stop.**
3. Is the primary metric classification accuracy? → **Collapse 3. Stop.**

If all three checks pass, proceed. If any fails, the implementation has drifted from what the framework claims to do.

The organism grows its own eyes. Not imports them, not adapts them, not benchmarks them. Grows them. From raw wave physics. Under survival pressure. Through the Cartographical Theory.

That's the experiment. No shortcuts.

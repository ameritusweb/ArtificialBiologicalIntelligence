# The Quotient Space: Computing the Organism's Umwelt

## What This Is

Territory I says: take each receptor's detection set C_r as a subbasis element. The generated topology T_t on the observation space defines which states the organism can distinguish. The organism lives in R^n/~_{R_t} — the quotient by receptor equivalence. Two states that no receptor separates are the same point in the organism's world.

This document computes that object from 80 generations of deep time data.

## Method

The full continuous quotient R^n/~_{R_t} is infinite-dimensional. But at the resolution of receptor discovery (binary: present or absent), each generation's Umwelt is characterized by its receptor configuration — a binary vector in {0,1}^161 over the 161 receptors ever observed. This is the topology's "signature": which distinctions the organism can make.

The analysis uses:
- An 80 x 161 binary matrix M (generations x receptors)
- SVD for effective dimensionality
- Hamming distance for Umwelt similarity
- Jaccard index for overlap
- Layer-specific decomposition for trunk/canopy dynamics

## Key Results

### Every generation had a unique Umwelt

80 generations, 80 distinct receptor configurations. No two generations inhabited the same quotient space. The organism's world was different at every point in evolutionary time.

### The Umwelt trajectory is low-dimensional

Despite 161 possible receptor dimensions, SVD reveals the trajectory lives in a much smaller space:

| Variance explained | Dimensions needed |
|--------------------|-------------------|
| 90% | 14 |
| 95% | 28 |
| 99% | 54 |

14 dimensions capture 90% of the variation in which receptors are present. This means receptors don't toggle independently — they co-vary in structured groups. The Umwelt moves through a 14-dimensional subspace of the 161-dimensional receptor space.

This is the knapsack prediction: the budget constraint forces correlated selections. When the environment shifts, entire groups of receptors (a prerequisite chain) enter or exit together because the chain must be held as a unit.

### Receptor independence: 99.2% of pairs are independent

Only 98 of 12,880 receptor pairs are strongly correlated (|r| > 0.5). The topology provides approximately 112 independent separations — 112 dimensions along which the organism can distinguish states that differ.

### Consecutive Umwelts differ by ~36 receptors

Mean Hamming distance between consecutive generations: 36.4 receptors (range: 21-81). This means roughly a third of the topology changes every generation. The organism's world doesn't just gain resolution — it reshapes.

### Jaccard similarity flattens quickly

| Generation gap | Mean Jaccard |
|---------------|--------------|
| 1 | 0.699 |
| 5 | 0.680 |
| 10 | 0.663 |
| 20 | 0.656 |
| 40 | 0.657 |

Similarity drops from 0.70 to 0.66 in the first 10 generations, then flatlines. The Umwelt doesn't drift farther and farther from any starting point — it orbits. After about 10 generations, a random pair of Umwelts is as similar as a close pair. This is the knapsack oscillating around its mean solution.

### Layer-specific dynamics

The trunk provides the coarse topology. The canopy provides the refinement. But they churn at comparable per-receptor rates:

| Layer | Total changes (79 transitions) | Receptors | Per receptor per gen |
|-------|-------------------------------|-----------|---------------------|
| Trunk | 450 | 24 | 0.237 |
| Canopy | 1,478 | 74 | 0.253 |

The canopy contributes 3x more total churn because it has 3x more receptors, but per-receptor turnover is similar. Trunk receptors are NOT inherently more stable on a per-receptor basis — they're more stable in aggregate because there are fewer of them and 18 are invariant (present in all 80 generations), which pull the average.

### The 18 invariant receptors are the irreducible common ground

18 receptors are present in every generation. They define the coarsest topology — the maximum quotient that's always shared. Every Umwelt the organism inhabits refines this base. In the language of discuss2.txt, these are the candidate cognitive universals: the trunk that makes communication possible between any two organisms in this system.

### Resolution trajectory by layer

| Gen | Total | Trunk | Branch | Canopy | Trunk fraction |
|-----|-------|-------|--------|--------|----------------|
| 0 | 72 | 14 | 24 | 26 | 0.19 |
| 10 | 103 | 16 | 30 | 47 | 0.16 |
| 20 | 97 | 15 | 33 | 39 | 0.15 |
| 30 | 107 | 15 | 37 | 46 | 0.14 |
| 40 | 97 | 15 | 31 | 42 | 0.15 |
| 50 | 100 | 18 | 32 | 39 | 0.18 |
| 60 | 97 | 16 | 33 | 38 | 0.16 |
| 70 | 98 | 12 | 37 | 39 | 0.12 |

Trunk fraction declines from 0.19 to 0.12 — as the topology grows, the canopy expands faster than the trunk. The organism's world gets finer-grained primarily through canopy additions (biography, not universals).

## Interpretation

### The Umwelt is real and computable

This analysis shows the quotient space R^n/~_{R_t} is not a metaphor. It's a measurable object that changes across generations with specific, quantifiable dynamics: 14-dimensional trajectory, 36-receptor mean distance between consecutive states, Jaccard floor of 0.66.

### The orbit, not the convergence

The Umwelt doesn't converge to a fixed point. It orbits. The Jaccard similarity plateaus at 0.66 — there's no generation where the topology stops changing. This is the knapsack under a shifting fitness landscape: the optimal solution changes when the environment changes, and the environment always changes (partly because the organism changes it).

### The trunk is the coordinate system

The 18 invariant receptors define the frame within which all the variation happens. They're the axes that don't rotate. Everything else — 143 of 161 receptors — is the topology exploring different refinements of that base. Two organisms share a world to the extent they share a topology. The invariant trunk is the shared world.

### What the 250-dim obs vector changes

With 73 live receptors wired in, the organism can now sense 92 of these distinctions in real time (vs 19 before). The quotient space got strictly finer — states that were previously identical (no receptor separated them) are now distinguishable.

The prediction: running deep time with the 250-dim obs vector should show a different trajectory. More separations available per step means more compositional possibilities per thought cycle. Whether this produces deeper topologies, breaks the anxiety loop cognitively, or changes the invariant trunk is the test of T102-T103.

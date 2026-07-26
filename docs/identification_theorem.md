# The Identification Theorem: Why This Decomposition

## The Problem

Many different receptor sets generate the same topology on observation space. Recovering the quotient R^n/~_R does not recover the decomposition into individual receptors. This is factor analysis's rotation indeterminacy in new clothing: the same correlation matrix is consistent with infinitely many factor structures. A century of debate about whether g is real, or whether the Big Five carve cognition at its joints, has never resolved — because correlational data alone cannot distinguish between rotations of the same latent space.

If ERTI can't answer "why these receptors and not some rotation of them," the genome project is a labeling exercise, not a discovery.

## The Theorem

**Theorem (Identification).** Among all subbases generating the same topology T_t on observation space, the RTDS's implemented decomposition is identifiable (unique up to relabeling) under two constraints that factor models lack:

1. **Metabolic budget** (MDL constraint): The total cost sum(alpha_r) <= B is finite. Among subbases generating the same topology, selection favors the one with minimum total cost — the minimum-description-length decomposition. Redundant receptors are metabolically wasteful and get selected out.

2. **Prerequisite DAG** (acquisition order constraint): The decomposition must factor along prerequisite lines. Receptor r_j can only be present if all r_i with r_i < r_j are also present. This constrains which subsets of the subbasis can appear at each developmental stage.

**The key:** rotations don't preserve acquisition order. A rotated basis makes wrong predictions about:
- What children learn when (developmental sequencing)
- Which deficits co-occur (neuropsychological dissociation)
- Which concepts can't be taught without prerequisites (threshold concepts in education)

Developmental data breaks the symmetry that correlational data cannot.

## Proof Sketch

Let S = {C_r1, C_r2, ..., C_rk} and S' = {C_r1', C_r2', ..., C_rk'} be two subbases generating the same topology T on R^n.

**Claim:** If S and S' both satisfy the DAG constraint under the same partial order, and both are locally minimal under the budget constraint, then S = S' up to relabeling.

**Argument:**

(a) The DAG constraint partitions the subbasis into layers: L_0 (no prerequisites), L_1 (prerequisites in L_0 only), L_2 (prerequisites in L_0 ∪ L_1), etc. Both S and S' must have the same layer structure because the acquisition order — which receptors appear first in development — is empirically observable and must agree.

(b) At layer 0, the subbasis elements are the trunk receptors — those that emerge in any structured environment. The budget constraint says the organism carries the cheapest set that generates the required separations at this layer. If two different sets of trunk receptors generate the same layer-0 separations at the same cost, they're equivalent decompositions. But if they predict different developmental onsets (one appears at age 3 months, another at age 6 months), the data distinguishes them.

(c) At each subsequent layer, the new subbasis elements must be consistent with the prerequisites already identified at lower layers. A rotation that mixes layer-0 and layer-1 elements violates the prerequisite structure — it predicts that a layer-1 concept could appear before its layer-0 prerequisite, which developmental data would falsify.

(d) The budget constraint eliminates redundancy within each layer. If two receptors at the same layer are metabolically equivalent and one is a linear combination of the other plus existing receptors, the cheaper one survives. Budget pressure converges to a locally minimal basis at each layer.

The combination of (a)-(d) constrains the decomposition to a unique solution at each layer, given the observable data (developmental order, dissociation structure, acquisition cost).

## Three Empirical Instruments

The theorem says the decomposition is identifiable from three independent data sources, and these must agree:

### 1. Developmental Sequencing

If receptor B never appears in any child before receptor A, that's evidence for A < B in the DAG. Development is the prerequisite structure playing out in observable time.

**In ERTI:** Layered emergence is confirmed — trunk receptors appear at generation 0, branch receptors at generation 1-5, canopy receptors at generation 10+. The ordering replicates across seeds (metacognition and conflation appear before depth_reached in both seed 42 and seed 99).

**In humans:** Spelke's core knowledge systems (objects, agents, number, space) are layer-0 candidates — present in infants and across species. Exact counting arrives with language at ~age 4 (layer-1, language-dependent). Formal proof arrives in adolescence (layer-2+). The ordering is cross-culturally robust.

### 2. Teaching Failure (Threshold Concepts)

A concept that cannot be installed without a prior detector is direct evidence for a DAG edge. The "threshold concepts" literature in education catalogs exactly this: concepts where more explanation doesn't help because the missing thing isn't information, it's a detector.

**In ERTI:** Knockout experiment — epistemic_strategy 0/15 without conflation prerequisite. The prerequisite is not optional and cannot be bypassed by more training data.

**In humans:** Calculus requires limits requires functions requires variables. Students who lack the "function" detector cannot learn calculus regardless of instruction quality. The failure is architectural, not pedagogical.

### 3. Neuropsychological Dissociation

A double dissociation (patient A has receptor X but not Y, patient B has Y but not X) says X and Y are incomparable in the partial order — neither is prerequisite to the other. A reliable single dissociation (X without Y exists, but Y without X does not) says X < Y.

**In ERTI:** The knockout experiment is a controlled single dissociation — conflation without epistemic_strategy exists (15/15), but epistemic_strategy without conflation does not (0/15).

**In humans:** Prosopagnosia (face recognition impaired, object recognition intact) vs visual agnosia (objects impaired, faces intact) is a double dissociation — face_recognition and object_recognition are incomparable in the DAG.

## What Makes This Novel

Factor analysis has rotation indeterminacy because it works from correlational data alone. PCA finds orthogonal axes that maximize variance, but any rotation of those axes explains the data equally well. The Big Five personality traits are one rotation; other rotations (HEXACO, for instance) fit the same correlation matrix.

The RTDS identification theorem breaks this indeterminacy by adding two constraints that correlational methods don't have:

1. **Cost structure** — receptors have metabolic costs, and the organism is under budget pressure. This eliminates redundant decompositions (the organism can't afford to carry two receptors that do the same job).

2. **Temporal structure** — receptors have prerequisites, and the acquisition order is observable. This eliminates rotations that mix layers (a rotated basis that puts a layer-2 concept in layer-0 makes wrong predictions about when children acquire it).

Neither constraint alone is sufficient. Budget without DAG gives you MDL but doesn't fix the rotation. DAG without budget gives you ordering but allows redundancy within each layer. Together, they identify a unique decomposition — the one that explains the developmental data at minimum cost.

## The First Objection and Its Answer

"Aren't your 200 receptors just one of many possible decompositions of cognition?"

Yes — if all you have is the correlation structure. No — if you also have the developmental ordering and the metabolic budget. The genome project's 200 receptors make specific predictions about acquisition order, dissociation structure, and teaching failure. Those predictions are falsifiable. A different decomposition that makes the same predictions at the same cost is an equivalent decomposition by the theorem. A different decomposition that makes different predictions is empirically distinguishable.

The theorem doesn't say the genome project's decomposition is correct. It says the decomposition is *testable* in a way that factor-analytic decompositions are not, because it predicts temporal structure that correlational data cannot access.

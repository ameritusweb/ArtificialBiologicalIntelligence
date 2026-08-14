# SOV Formal Specification
### The Two-Sorted Algebra: Sorts, Lifts, Conservativity, Functoriality, and Boundary Inventory

**Status:** Formal derivation, first articulated 2026-08-09  
**Origin:** Companion to SOV Operator Algebra; formalizes the two-sorted structure discovered there  
**Core claim:** The SOV operator algebra factors into a geometry base and a ledger fiber. Every named operator is a lift from a geometry move to a (geometry, ledger) pair. The three derivation constraints are the conditions on this lift: well-definedness on the base (shape-only computability), conservativity of the fiber (receipt integrity), and functoriality of the lift (provenance closure). The algebra is complete when every required move has a lift satisfying all three conditions, and minimal when no lift can be removed without losing a capability.

---

## 0. Motivation: Why a Formal Spec

The operator document describes what each operator does and why it is forced. This document specifies the algebra's formal structure — the mathematical conditions that make the description precise and checkable.

Three questions the operator document leaves open that the formal spec answers:

1. **What exactly is a "two-sorted algebra"?** The factoring into geometry and ledger sorts needs a precise definition before it can be used to argue minimality.

2. **What are the legal pairings?** Not every geometry move can pair with every fiber move. The formal spec states the composition law — which pairings are admissible and why.

3. **What are the conservation laws?** An algebra without conservation laws cannot state what it preserves. The formal spec derives the conservation laws from the lift conditions.

---

## 1. The Two Sorts

### 1.1 Sort G — Geometry

The geometry sort is the algebra of connector shapes: receptor boundary patterns, relational positions in the dependency graph, feasibility sets, and structural isomorphisms between them.

**Objects of sort G:** Slots (`?X`, `?Y`, `?Z`, ...), known entries (`K`), and constraint expressions (`C`). A slot is a typed position in the dependency graph with a connector geometry — a receptor boundary pattern and a set of relational links to neighboring slots and knowns. A known entry is a slot whose feasibility set has been reduced to a singleton.

**Morphisms of sort G:** Geometry moves — functions from slot configurations to slot configurations that act only on connector geometry, not on receipt histories. A geometry move `g: S → S'` maps a slot (or slot configuration) `S` to a slot configuration `S'` by modifying boundary conditions, relational links, or feasibility sets, without touching any receipt.

**The identity geometry move:** `id_G(?X) = ?X` — the slot unchanged. Pure fiber moves (Archaize, Anneal) have identity geometry moves.

**Composition in G:** Geometry moves compose sequentially. `g₂ ∘ g₁` applies `g₁` first, then `g₂`. Composition is associative; identity is the unit.

### 1.2 Sort L — Ledger

The ledger sort is the algebra of receipt histories: funded structures, provenance chains, certainty weights, and the operations that modify them.

**Objects of sort L:** Receipt histories (`R(?X)` — the set of receipts on slot `?X`), provenance chains (`prov(r)` — the funding trace of receipt `r`), certainty weights (`c(?X)` — the current credence on slot `?X`), and the etymology ledger (`E` — the append-only log of all ledger events).

**Morphisms of sort L:** Ledger moves — functions from receipt configurations to receipt configurations that modify the ledger without directly modifying slot geometries. A ledger move `ℓ: R → R'` maps a receipt configuration to a new receipt configuration by accruing, pooling, partitioning, establishing liens, transferring, retracting, evicting, or annealing.

**The identity ledger move:** `id_L(R) = R` — the receipt history unchanged. Pure geometry moves have identity ledger moves (though in practice every geometry move is paired with at least a trivial ledger annotation).

**Composition in L:** Ledger moves compose sequentially. The ledger is append-only: no ledger move deletes history; all moves add events to the etymology ledger `E`.

---

## 2. The Lift

### 2.1 Definition

A **lift** is a function `lift: G → (G × L)` that takes a geometry move and produces a paired (geometry move, ledger move) — a named operator.

Every named operator in the SOV algebra is a lift:

```
Fit     = lift(boundary-test)      = (boundary-test,    accrue)
Individuate = lift(carve)          = (carve-from-unassigned, pool)   [2026-08-10]
Constrain = lift(feasibility-narrow) = (feasibility-narrow, transitive-accrue)
Retract = lift(compensated-widen)  = (compensated-widen, retract)    [2026-08-10]
Compose = lift(relational-product)  = (relational-product, structural-accrue)
Differentiate = lift(geometry-split) = (geometry-split,   partition)
Unify   = lift(identity-check)     = (identity-check,    pool)
Exclude = lift(divergence-cache)   = (divergence-cache,  accrue-negative)
Abstract = lift(geometry-intersect) = (geometry-intersect, lien)
Posit   = lift(hypothetical-prop)  = (hypothetical-prop,  imagined-∅)
Reopen  = lift(demotion)           = (demotion,          retract)   [derived: Retract at closure]
Pose    = lift(geometry-serialize) = (geometry-serialize, ∅)
Attest  = lift(geometry-match)     = (geometry-match,    discount-transfer)
Bind    = lift(cross-ledger identity check [derived]) = (identity-check, coordination-accrue)  [2026-08-10]
Quote   = lift(geometry-as-data)   = (geometry-as-data,  meta-accrue)
Archaize = lift(id_G)              = (id_G,              evict)
Anneal  = lift(id_G)               = (id_G,              anneal)
```

Where `∅` denotes the trivial (no-op) ledger move, and `imagined-∅` denotes a no-op ledger move with IMAGINED provenance annotation.

### 2.2 The Three Conditions on the Lift

Every lift must satisfy three conditions. These are the formal statements of the three derivation constraints.

**Condition 1 — Well-definedness (Shape-only computability):**

The geometry component of `lift(g)` is well-defined on sort G alone. Formally: for any geometry move `g`, the geometry component `g` of `lift(g)` can be computed given only the connector geometry of the input slot — no receipt history, no slot content, no knowledge of what the slot might resolve to.

> `g(?X) is defined ⟺ geometry(?X) is defined`

This is what "shape-only computability" means precisely: the geometry base is self-contained. The fiber cannot be required to compute the base move.

**Condition 2 — Conservativity (Receipt integrity):**

The ledger component `ℓ` of `lift(g)` is conservative over the geometry move `g`. Formally: the ledger move `ℓ` cannot create funded structure that the geometry move `g` did not earn. 

> `receipts(ℓ(R)) ⊆ earned(g) ∪ R`

Where `earned(g)` is the set of receipts that the geometry move `g` is licensed to generate (determined by the specific geometry move — Fit earns lived-data receipts, Compose earns structural-necessity receipts, etc.), and `R` is the prior receipt history.

The fiber is a conservative extension of the base: it can redistribute, annotate, and transfer receipts, but it cannot manufacture them. The base determines what is earned; the fiber tracks and moves it.

**Condition 3 — Functoriality (Provenance closure):**

The lift is functorial over composition. Formally: the provenance of a composed operator is the composition of the provenances of its components.

> `prov(lift(g₂ ∘ g₁)) = prov(lift(g₂)) ∘ prov(lift(g₁))`

This is why chains are traceable regardless of how many operators were applied. Functoriality means the provenance map commutes with composition — there is no operation on operators that loses provenance. Every funding chain, no matter how long, grounds out in Fit receipts because the functoriality condition propagates the grounding through every composition.

**The conditions jointly:** Condition 1 makes the base autonomous. Condition 2 makes the fiber honest. Condition 3 makes composition safe. A lift that satisfies all three is **admissible**. A lift that violates any one is inadmissible and excluded from the algebra.

---

## 3. Legal Pairings

Not every geometry move can pair with every fiber move. The legal pairings are determined by what the geometry move earns — what the base move licenses the fiber to do.

### 3.1 The Earning Table

| Geometry move | What it earns (fiber license) |
|---|---|
| `boundary-test` (Fit) | Accrual of lived-data receipts; funded by the episode. On a total miss with real activation: an unassigned-pool record (pre-ledger) |
| `carve` (Individuate) | Pool of the consumed cluster as opening receipts, each grounding in its lived log offset; licensed by cluster size ≥ the pre-registered floor |
| `feasibility-narrow` (Constrain) | Transitive accrual from the constraint's provenance |
| `compensated-widen` (Retract) | Retraction event; licensed ONLY by non-IMAGINED failing receipts — the world's testimony (T154's deletion clause inside the algebra) |
| `relational-product` (Compose) | Structural-necessity accrual; funded by the composing slots' receipt histories |
| `geometry-split` (Differentiate) | Partition of prior receipts between children; no new receipts created |
| `identity-check` (Unify) | Pool of both slots' receipt histories; licensed iff identity check succeeds |
| `divergence-cache` (Exclude) | Accrual of a negative receipt; funded by the divergent receipts that failed the check |
| `geometry-intersect` (Abstract) | Lien on the shared sub-pattern of each child's receipts; not a copy |
| `hypothetical-prop` (Posit) | IMAGINED annotation only; no ledger accrual licensed |
| `demotion` (Reopen) | Retraction event; licensed by the 404 receipts that triggered demotion |
| `geometry-serialize` (Pose) | No ledger move licensed; shape only |
| `geometry-match` (Attest) | Discounted transfer of the matched receipts; licensed by D-metric profile overlap |
| `cross-ledger identity check` (Bind) | Coordination-accrue on a jointly-owned interface object; licensed by the Case-2 match AND subsequent dual-grounded coordination events (COORDINATED provenance — the binding mints nothing in either ledger) |
| `geometry-as-data` (Quote) | Meta-accrual; funded by the second-order statistics of receipt arrival on the quoted slot |
| `id_G` (Archaize) | Eviction; licensed by the rent deficit on the slot |
| `id_G` (Anneal) | Certainty weight update; licensed by T57's annealing schedule |

### 3.2 Inadmissible Pairings

A pairing `(g, ℓ)` is inadmissible if `ℓ` requires more than `g` earns. Examples:

- `(boundary-test, pool)` — Fit cannot pool receipt histories; it earns only a single receipt per episode. Attempting to pool on a Fit is inadmissible: it would merge two slots on the basis of a single observation, bypassing the identity check that Unify requires.

- `(hypothetical-prop, accrue)` — Posit cannot accrue to the ledger; hypothetical propagation earns only IMAGINED receipts. Attempting to accrue from a Posit is inadmissible: it is the central failure mode the provenance system exists to prevent — laundering imagined structure into the funded ledger.

- `(id_G, pool)` — A pure identity geometry move cannot pool receipt histories; it earns nothing on the geometry base. Pooling requires a prior identity check (the Unify geometry move). Without the geometry check, pooling two receipt histories is inadmissible: it asserts structural identity without testing it.

- `(geometry-serialize, discount-transfer)` — Pose cannot transfer receipts; it serializes geometry only. Receipt transfer requires a geometry match (the Attest geometry move). Pose without a match is inadmissible.

### 3.3 The Lamarckian Inadmissibility

One inadmissible pairing deserves special treatment because it is biologically and culturally significant:

`(genome-broadcast, accrue-receipts)` — the Lamarckian operator: broadcasting an organism's receipt history (acquired experience) into the genome for inheritance.

Inadmissible by Condition 1: the genome broadcast is a Pose operation (shape-only serialization). Pose earns no ledger move. Accruing receipts through a Pose is inadmissible — it would transfer lived experience across the generational boundary into the inherited structure, which the shape-only computability condition prohibits.

The Weismann barrier is Condition 1 enforced at the germline. Lamarckism is an inadmissible operator. Culture (Attest across generations via externalized logs) is the admissible alternative — it satisfies the geometry-match requirement through shared vocabulary overlap.

---

## 4. Conservation Laws

### 4.1 Receipt Conservation

**Law 1 (No creation):** The total receipts in the system at time T+1 equals the total receipts at time T plus the receipts earned by geometry moves executed at time T.

> `|R(T+1)| = |R(T)| + |earned(G(T))|`

Where `G(T)` is the set of geometry moves executed at time T. No fiber move creates receipts; it can only redistribute, annotate, transfer, or evict them. Accrual (Fit, Compose, Exclude, Abstract, Quote, Reopen) creates new receipts, but only the amount licensed by the geometry move that earned them.

**Law 2 (No destruction):** Receipts are never deleted from the etymology ledger. Eviction (Archaize) removes a slot from active status but preserves its receipt history. Retraction (Reopen) logs a retraction event but does not delete the closure event. Anneal changes certainty weights but does not remove the receipts that set them.

> `E(T+1) ⊇ E(T)` (the etymology ledger is monotonically growing)

The receipts on an archaized slot, a retracted closure, or a deprecated certainty are permanently available for historical reconstruction.

### 4.2 Provenance Conservation

**Law 3 (Grounding):** Every funded receipt in the system has a provenance chain that terminates in a Fit receipt (a lived-data event). No receipt chain can form a cycle (because the provenance DAG is well-founded by Condition 3 — functoriality).

> `∀r ∈ R(T), ∃ chain r → r₁ → r₂ → ... → rₙ where rₙ is a Fit receipt`

IMAGINED receipts (from Posit) are not in `R(T)` — they exist in a separate imagination register with no funding chain to the ledger.

**Law 4 (Attest grounding):** Cross-ledger receipts (from Attest) preserve their grounding. An attested receipt carries the originating agent's ledger identity and the discount factor; its provenance chain terminates in a Fit receipt on the originating agent's ledger. The importing agent's discount does not break the chain — it annotates it.

### 4.3 Geometry Conservation

**Law 5 (Shape persistence):** The connector geometry of a slot is preserved by all fiber moves. Archaize and Anneal have identity geometry moves — they cannot modify slot boundaries. The geometry of an archaized slot is preserved in the etymology ledger even after eviction; it is available for reconstruction.

**Law 6 (Feasibility monotonicity, restated 2026-08-10):** Constrain is monotonically narrowing — it can only reduce a slot's feasibility set, never expand it. Feasibility expands **only** under logged compensation events (Retract, and its derived closure case Reopen), each licensed by non-IMAGINED failing receipts — the world's testimony — and each restoring geometry recorded in the etymology ledger, never generating new geometry. There is no unlicensed widening: the original narrowing events remain logged (compensation, not inversion).

---

## 5. The Boundary-Crossing Inventory

The algebra's perimeter is defined by the boundaries that operators cross. There are five boundary types; each has exactly one admissible crossing direction per named operator.

### 5.1 World → Ledger (inbound lived data)

**Operator: Fit**

Fit is the only operator that carries information from the world (lived experience) into the ledger (funded receipts). Every other operator operates within the ledger or between ledgers. The world/ledger boundary is the most fundamental — it is where funded structure originates.

*Inadmissible crossing:* Any operator that attempts to write world-state directly into the ledger without going through Fit is inadmissible. Generator outputs (effectors) cross ledger→world outbound; their consequences re-enter through Fit inbound. The generator is outside the algebra.

### 5.2 Ledger → World (outbound effector)

**Operator: Generator (outside the algebra)**

The generator carries funded conclusions from the ledger into world actions — writing world states (T153's environment-organism), executing motor commands (ABI organisms), producing utterances (language). The generator is an effector, not an operator in the algebra; it receives funded structure and acts on it, but the acting is not itself a ledger operation.

*Note:* The generator's outputs return to the ledger only through Fit — via the sensory consequences of the action. The loop is: Fit (world→ledger) → operators → generator → world → Fit.

### 5.3 Ledger → Ledger, Geometry (outbound from one agent)

**Operator: Pose**

Pose crosses the boundary from one agent's ledger to another agent's perception, carrying connector geometry only. No receipt crosses this boundary outbound — Pose is shape-only by construction (open slots have no content to leak). The boundary is one-directional per Pose invocation: one agent broadcasts, others receive.

### 5.4 Ledger → Ledger, Receipts (inbound to receiving agent)

**Operator: Attest**

Attest crosses the boundary from one agent's ledger to another, carrying receipts at a D-metric-calibrated discount. The geometry matched by Pose determines which receipts are eligible for transfer; Attest executes the transfer. Receipts enter the importing ledger at the discount rate and anneal under the importer's own billing.

*Note on the Pose/Attest pair:* Pose and Attest are the two halves of a cross-ledger transaction. A complete transaction requires both: Pose opens the geometry channel; Attest closes it with receipts. A Pose without an Attest is a question without an answer — the geometry is broadcast but no receipts arrive. An Attest without a prior Pose is inadmissible — there is no geometry match to license the transfer.

### 5.5 Ledger → Imagination (hypothetical register)

**Operator: Posit (and derived Transpose)**

Posit crosses the boundary from the funded ledger into the imagination register — a hypothetical space where geometry propagation can be computed without any funding consequences. The boundary is enforced by IMAGINED provenance: nothing that exists in the imagination register can cross back into the funded ledger.

*The imagination register is not the ledger.* It is a separate space, maintained offline, that the system can write to freely and read from for planning and targeting — but cannot cite as evidence.

*Transpose crosses the same boundary* (reverse-mode Posit), subject to the same provenance rules and the additional scheduling restriction (replay-slot only).

*Suspend and Counterposit cross the same boundary* (registered 2026-08-10, refined same day): `Suspend(K)` masks a K's testimony in a copy-on-write context — a **modal** operation on the evaluation axis (actual | hypothetical(context)), never a lifecycle transition — and recomputes every dependent structure from its remaining support paths; `Counterposit(K, v′) = Posit(v′) ∘ Suspend(K)` is Pearl's third rung, completed by **abduction-by-replay** (re-fitting a lived window from the append-only log through the context — the log is the background context Pearl's abduction step must otherwise infer). **The support-environment rule:** receipts carry AND/OR justification formulas over their parents; under Suspend(K), a receipt survives iff its formula evaluates true with K's receipts false, and a conclusion survives iff any of its support environments excludes K — evaluated lazily per suspension, never as stored ATMS labels. Descendants with partial independent support survive *discounted*, not erased. Reading the provenance DAG is the register's dependence on the ledger, not a crossing back — nothing IMAGINED funds anything. Same scheduling restriction as Transpose.

### 5.6 Summary Table

| Boundary | Direction | Operator | What crosses | Admissible inversion? |
|---|---|---|---|---|
| World ↔ Ledger | World → Ledger | Fit | Lived-data receipts | No (ledger→world is the generator, outside algebra) |
| World ↔ Ledger | Ledger → World | Generator (external) | Funded conclusions → actions | No (actions return via Fit only) |
| Ledger ↔ Ledger | Outbound geometry | Pose | Connector geometry | Pose is one-directional per invocation |
| Ledger ↔ Ledger | Inbound receipts | Attest | Discounted receipts | Requires prior Pose to license |
| Ledger ↔ Ledger | The interface itself | Bind (2026-08-10) | Nothing crosses — a jointly-owned binding object inhabits the boundary; COORDINATED provenance, dual-grounded | Symmetric; differentiates under strain; archaizes when coordination stops |
| Ledger ↔ Imagination | Ledger → Imagination | Posit, Transpose, Suspend, Counterposit | Hypothetical / counterfactual propagation | No (IMAGINED receipts cannot re-enter ledger) |

---

## 6. The P-Number Registry

The formal spec is the authority on P-number assignment for the SOV document family. Collisions are a coda-discipline failure; this table is the canonical record.

| P-number | Prediction | Source document | Status |
|---|---|---|---|
| P48 | Query-token emergence in co-funded populations | SOV Entailments | Open |
| P49 | Inherited questions: aimed first-exposure exploration | SOV Entailments | Open |
| P50 | Amortization signature: web vs chain inference cost | SOV Entailments | Open |
| P51 | Spectral ripple and hub rule: Laplacian predicts restructuring reach | SOV Entailments | Open |
| P52 | Exclusion-cache signature | SOV Operator Algebra | Open |
| P53 | Transfer holonomy: round-trip slot geometry mismatch confirms curvature | SOV Geometry | Open |
| P54 | Abstract-lattice signature: hierarchical structure predicts transfer | SOV Operator Algebra | Open |
| P55 | Posit as experiment design: hub-targeted generation out-distills random | SOV Operator Algebra | Open |
| P56 | Reopen cost: Posit estimate at closure beats connectivity at retraction | SOV Operator Algebra | Open |
| P57 | Quote enables self-correction: anomaly detection advantage | SOV Operator Algebra | Open |
| P58 | Curvature decreases with understanding | SOV Geometry | Open |
| P59 | Geodesic curriculum advantage | SOV Geometry | Open |
| P60 | Spectral fingerprint predicts transfer cost | SOV Geometry | Open |
| P61 | Re-basis as geometric phase transition | SOV Geometry | Open |
| P62 | Ledger-graded pathology: each ledger fragment mitigates its matched disease component-wise | The Ledgerless Economy | Open |
| P63 | ERTI contrast: ledgered system exhibits none of the six pathologies | The Ledgerless Economy | Open |
| P64 | Assortative Posing: communication yield peaks at intermediate D-metric divergence. **First bill 2026-08-11 (E1): NOT SUPPORTED as registered** — seed 45 VOID-instrument (ladder non-monotone); seed 46 valid: yield rises MONOTONICALLY to the far endpoint (rising limb only). Scoped: within a shared code, incommensurability is unreachable — the falling limb lives at coordinate divergence (cross-encoder / cross-species / T153's organism↔environment case), where the full test moves | SOV Entailments / F16 | Open (rising limb observed; peak untested) |
| P65 | Grammatical 404 tracking: Pose rate tracks structured misprediction, not raw surprise | SOV Entailments | Open |
| P66 | Confluence test: monotone-only propagation is order-independent; non-monotone is order-dependent | SOV Operator Algebra | Open |
| P67 | State-transition coverage: every reachable lifecycle state × required transition has a named operator or stated derivation; an empty cell falsifies completeness, a doubly-occupied cell challenges minimality. Not aesthetics: completeness + enforced conservation is the counterfactual license — the precondition for rung-3 queries being queries rather than guesses (imagination_register.md §7) | SOV Operator Algebra §3 (2026-08-10) | Open |
| P68 | Funded Quote depth scales ~ log(lifetime lived receipts) | Two-Sorted Core TS-4 (renumbered from its "P58") | Open |
| P69 | Composed-slot survival is low and selective; deviation in either direction is diagnostic | Two-Sorted Core §2 (renumbered from its "P60") | **Supported** (2026-08-11, replay phase first bill: 24 composed, survival 0.125, survivor mean fits 89,600 vs evicted 7,314 — F21) **Restated under net-per-fire economics (2026-08-12): survival = selectivity at any volume — survivors fire 0.24-0.28 with cap-pinned balances, evicted fire 1.0 and drain within one generation; total bimodal split (rep 12).** |
| P70 | Reliability-annealed Attest discount out-distills fixed similarity discount | Two-Sorted Core TS-5 (renumbered from its "P61") | Open |
| P71 | Contained counterfactuals: provenance-guided suspension identifies the exact set to sever; ledgerless counterfactuals leak the suspended belief's influence through unsuspended descendants | The Ledgerless Economy / Suspend (2026-08-10) | Open |
| P72 | Calibration by construction: ledger-readout generation produces zero unfunded assertions; a sequence model on the same corpus does not | T155 / language_center_requirements.md | Open |
| P73 | Phrase economics: surviving construction inventory is low and selective; splits validated by naive listeners | T155 / language_center_requirements.md | Open |
| P74 | Analogy speakability: eigen-fingerprint similarity predicts cross-domain clause-family reuse with transfer benefit | T155 / language_center_requirements.md | Open |
| P75 | Resonance precursor: geometry-matched cross-organism slot pairs show co-fit correlation on shared world structure exceeding shuffled-pair baseline, PRESENT IN THE CHANNEL-FREE (ISOLATED) ARM. **First bill 2026-08-11: SUPPORTED, both seeds** (matched ≈0.13 vs shuffled ≈0.007, ~18x, n≈14k/seed; SHARE-arm co-fit numerically identical — the channel adds nothing, world-mediation confirmed by comparison) | Bind (operator doc §1.15) / F16 | **Supported** |
| P76 | Staged fit (Serialization Thesis 2×2): serial stage-wise fitting with edge-derived expectations beats simultaneous fitting on funded structure at matched exposure; the margin grows with edge density, is MEDIATED by enumeration sharpness, and vanishes when receipts are discarded | Serialization Thesis (Entailments §4.6) / replay_phase_requirements.md | **Supported** (2026-08-12, v4: expectation-receipt channel; S+C beats baseline on both endpoints while complete-information consumption of the identical channel HURTS — the staging is the safety structure; fringe sharpness 4/4 — F32) |
| P77 | Earned vs stipulated enumeration: gap-geometry-derived hypothesis spaces beat fixed-vocabulary enumerations on identification-per-receipt at matched exposure, advantage scaling with neighborhood funding — the masked-modeling contrast made billable | Omission Cycle (imagination_register.md §5a) / replay_phase_requirements.md | Open |
| P78 | Meta-prediction pays (T156, the tower's first rung): an SOV′-style carved model over the base web's processing stream out-predicts base-level statistics on the base's restructuring behavior. **First bill 2026-08-10: NOT SUPPORTED at generation grain** (nothing beats the expanding mean; aggregate counts memoryless — the carving too coarse). Per-slot episode-grain full form open | T156 / reflection_tower.md §7 / meta_prediction_p78.py | Open (gen-grain arm: NOT SUPPORTED) |

| P79 | Consequence-carving gain (T157): in an external domain with a lived log and manufactured stakes, consequence-carved receptors find value-relevant dimensions that matched-capacity predictive carving misses, gap concentrated in tail events; falsified by absent gap under demonstrated salience/relevance misalignment or by non-tail concentration | T157 / stakeholder_theorems.md / stakeholder_requirements.md | Open |

*Next available P-number: P80.*

| P80 | Demand allocation tracks discharged-surprise-mass-per-receipt: closure and funding order correlate with per-family discharge opportunity (world-indexed), not raw fit volume — retroactively explains the closure-order census (discharge-first, not rules-first) | T158 / whys.md §23 (2026-08-12) | Open |
| P81 | The counterfeit-discharge lesion: a conservativity-ablated sandboxed web extinguishes its surprise stream while prediction error holds or rises (relief without learning — the epistemic-wirehead signature); the intact web shows coupled decline | T158 (2026-08-12) | Open |
| P82 | Discharge-priced trade: attest acceptance and corroboration yield track the listener's pre-existing surprise stream in the asserted region — assertions are valued as surprise-dischargers | T158 / F28 currency union (2026-08-12) | Open |

| P83 | The falsifiability optimum: surviving explanation-structures cluster near p* = argmax p((1-FIRE_COST_FRAC)-p), and the cluster TRACKS p* when the cost constant moves (rep-12 survivors at 0.24-0.28 vs p*=0.375 is the standing hint) | T158 ext. (2026-08-12) | Open |
| P84 | The eigen fingerprint decodes invoice-state: outstanding per-family near-miss/404 mass is recoverable from the 5-bit structural signature above chance — the geometry IS the why-state (T114 reread's falsifier) | T158 ext. (2026-08-12) | Open |
| P85 | Bedrock is designable: a social-surprise-dominant world (NPC-rich, rule-static) closes social content first — closure order tracks the world's dominant surprise class (P80's decisive form; the census mechanism as engineering capability) | T158 ext. (2026-08-12) | Open |

| P86 | Induced atrophy: freezing why-flow (slot genesis, closure, posing disabled; inventory intact) holds fitness short-term while exploration/pose rates decay and novelty response collapses disproportionately vs a flow-intact control — motivational collapse with knowledge intact, invisible until the world moves | T158 atrophy ext. (2026-08-12) | Supported in ENVIRONMENTAL form (2026-08-13, R=3 pooled, p86_v4: an INTACT organism in a spent world shows TOTAL flow starvation — 0 world-sensitive events/36 gens vs 0.306/gen at the operating band — with exact stock parity, and migration-specific flow restart 3/3 replicates; the freeze form was discharged VOID by F35/F38 — flow bills only through consumption, and the ledger, not the organism, was the patient. F42) |
| P87 | Rehabilitation contrast: after induced atrophy, a closability-margin curriculum of simple high-fertility whys restores settlement then exploration rates faster and more durably than surprise-rich stimulation at matched exposure — the lived protocol as billed verdict | T158 atrophy ext. (2026-08-12) | Open |

| P88 | Reliability-weighted discharge: an importer's surprise reduction from attested whys tracks the exporter's reliability posterior; injected false attests damage the importer in inverse proportion to discount calibration | T158 trust ext. (2026-08-12) | Open |
| P89 | Surprise contagion: a trusted exporter's surprise events open importer invoices along trust edges without direct experience, correlation tracking the reliability posterior — warnings transmit invoices, not information | T158 trust ext. (2026-08-12) | Open |
| P90 | Signed trust: an importer learning SIGNED reliability recovers discharge value from reliably-false exporters (inversion) vs the clip-at-ignore baseline, pays the motive-invoice cost of distrust, and shows the double-bluff vulnerability only adversarial-regime discounting closes | T158 trust ext. (2026-08-12) | Open |

| P91 | Trust-concentration fragility: importers concentrated on one exporter show correlated reopening cascades when the exporter's whys rot; diversified import topologies degrade gracefully — portfolio law for why-inventories | T158 social ext. (2026-08-12) | Open |
| P92 | The audit receptor: a per-region comparator of behavioral competence vs ledger coverage fires on performance-without-account; validated against the known case — EX-0's procedural memory must be its first catch | T158 social ext. (2026-08-12) | Open |
| P93 | Settlement routing: trusted epistemic-state broadcasts (calm/surprised) shift the importer's explore-vs-Attest allocation appropriately, with yield gains over unrouted controls at matched budget | T158 social ext. (2026-08-12) | Open |

| P94 | AW-0 closed loop: a meta-organism traversing its own web (eigen vision, touch-replay, pose) improves base-web quality — closure rate, calibration, gap discharge — over an untraversed control at matched compute | T159 (2026-08-12) | Open |
| P95 | Occlusion ecology: selection-trained inhabitants surviving by enumeration of sealed truths beat the static earned-enumeration operator on identification-per-receipt | T159 (2026-08-12) | Open |
| P96 | The head-to-head: on a receipted structure at matched compute, inhabitation-search recovers consequence-relevant structure that visit-weighted tree/graph search misses, and its map retains value post-search while the tree is discarded | T159 (2026-08-12) | Open |

| P97 | Comparative epistemic anatomy: organisms grown in >=3 receipted idea-worlds evolve an invariant trunk plus domain-specific canopies, topology differences tracking measurable structural differences between corpora | T159 second arrival (2026-08-12) | Open |
| P98 | Historical retrodiction: on a legal corpus with the amendment record held out, later-amended/struck/litigated provisions show elevated strain signals before disclosure — the document audit calibrated against centuries of lived law | T159 second arrival (2026-08-12) | Open |

| P99 | The reflexive retrodiction: the AW organism run on the ABI corpus with the verdict record held out shows elevated strain where the program later falsified, revised, or retracted (T55, T26, rules-first, the +223% claim) — the instrument calibrated against the lived history of its own conception | T159 third arrival (2026-08-12) | Open |

| P100 | The epistemic affordance trunk: trace/replay-class receptors are invariant across >=3 idea-worlds, as grip/push are across physical tiers — embodiment has the same anatomy in worlds of matter and meaning | T159 fourth arrival (2026-08-12) | Open |
| P101 | The designer's exit: evolved operator policies, scan keys, and clock constants outperform the hand-scheduled baseline at matched compute — the program's design debt converted to selection's job | T159 fourth arrival (2026-08-12) | Open |
| P102 | Tri-world circulation: at matched contact budget, all three feeds (base / mature-domain / self) beat every two-feed ablation on world-agreement + funded inventory + discharge per contact | T159 fifth arrival (2026-08-13) | Open |
| P103 | Circulation pathologies are early ledger signatures: severing each feed yields its distinct diagnosable signature (rumination / cargo-cult / atrophy) before behavioral loss — the ledger sees what fitness cannot | T159 fifth arrival (2026-08-13) | Open |
| P104 | The evolved tri-world schedule beats every fixed mix at matched compute and shifts toward self-inhabitation as a function of ledger mass — the human developmental arc as retrodiction | T159 fifth arrival (2026-08-13) | Open |
| P105 | Readout as edge-detector: LC-rendering a region raises typed-gap detection there vs matched unrendered regions — writing finds your edges | T159 fifth arrival (2026-08-13) | Open |
| P106 | The pigeonhole law: collision pairs grow superlinearly with composed-population size — a fixed description language has finite carrying capacity, and vocabulary growth past it is conflation by arithmetic | F39 impl. 6 / T159 fifth-arrival lineage (2026-08-13) | Supported (first bill same day: rho=0.96 over 23 checkpoints, profiles saturate at ~50, 212 pairs among 72 composed — results/p106_maturity.json, F40). Economy-on replication (2026-08-13, p_nursery, F45): PARTIAL — solvent crowding fraction 0.005 (1 pair among 17 earned composed); the arithmetic stands, the timescale stretches ~3-4x; F40's crowd was 99.5% unbilled mass (F44 caveat quantified) |
| P107 | Trajectory-criticality allocation: attention/court priority keyed to certificate-blocking power of open slots beats salience/near-miss-only allocation on endpoint attainment — blocker-removal as an allocation law | T160 (2026-08-13) | Open |
| P108 | The experiment organ: manufacture-loop scheduling (subtraction / isolation / randomization through effectors) earns deconfounded causal structure faster per contact than passive fitting + exploration noise at matched budget | T160 (2026-08-13) | Open |
| P109 | Environment manipulation: an organism licensed to modify its world achieves higher endpoint attainment at lower attention cost than a navigation-only twin, with manipulation concentrated on high-traffic trajectories (the road-building pattern) | T160 (2026-08-13) | Open |
| P110 | The licensing criterion: gating irreversible commitments on endpoint-certificate strength outperforms reversibility-only gating (T152's missing criterion) | T160 (2026-08-13) | Open |
| P111 | Destination-manufacture: minting destination slots priced by expected discharge x onward-fertility sustains higher why-flow than exogenous demand alone; unpriced minting reproduces the conspiracy signature (quest inflation) | T160 vii (2026-08-13) | Open |
| P112 | The constructive hierarchy: reachable-future classes strictly nest across T160's levels in generic venues (ablation collapses capacity in the predicted order; degenerate venues collapse as predicted too) — Pearl's strictness, constructive form | T160 ix (2026-08-13) | Open |

*Registry reconciliation (2026-08-10): this table is the single authority. The Two-Sorted Core's §7 table conflicted with it (both claimed authority); its three orphaned predictions are admitted above as P68–P70, and its "P59" (stored Posit estimate beats retraction-time connectivity) is the same claim as P56 and keeps that number. The Geometry document's inline P57–P60 were shifted one from this registry and have been corrected in place to P58–P61.*

*Assignment rule: P-numbers are assigned in the turn they first appear, by the document that first states the prediction. Subsequent documents that reference a prediction use the original number. Renaming is logged here with the original and new assignment.*

---

## 7. Open Questions

- **What is the composition law for the two sorts?** Section 3 gives the legal pairings (which geometry moves can pair with which fiber moves) but does not give the full composition law — what happens when two named operators are composed in sequence. Does the composition always produce an admissible lift? Are there operator sequences that produce inadmissible intermediate states even when each individual operator is admissible?

- **Is the lift unique?** For a given geometry move, is there exactly one admissible fiber move, or can there be multiple? If there are multiple admissible pairings for the same geometry move, the named operators may not be uniquely determined by the geometry base — there may be a family of valid algebras, differing in their fiber choices.

- **What is the monad structure?** *Resolved (2026-08-10, external review + same-day analysis): the algebra is the Kleisli category of a **graded writer monad** over the etymology monoid, indexed by web state.* A named operator is an arrow `state → state × events`; arrows compose by concatenating the append-only event monoid — the writer monad. Under this identification: **functoriality IS the associativity law** (provenance chains compose because Kleisli composition concatenates logs — coherence comes free); **Law 2 falls out of choosing a free monoid** (no inverses → append-only by construction; compensations-not-inversions is that choice named). But **conservativity is NOT the unit law** — the unit law only makes the trivial fiber an identity, while conservativity forbids the fiber writing events its base move didn't earn. Conservativity is a **grading**: the earning table assigns each geometry move its licensed fiber effects, and admissible operators are the Kleisli arrows whose log lies in their grade — an effect system, with inadmissible pairings as type errors. This division of labor is the healthy outcome: the monad supplies coherence for free; the receipts discipline remains genuinely additional content the monad hosts, not a categorical tautology. Two consequences: (1) the compensation-semantics question ("what quotient of event sequences is the same ledger state?") becomes *presenting the monoid by generators and relations* — the anneal schedule-independence fix was the first relation imposed empirically (`anneal(t₁);anneal(t₂) = anneal(t₂)`), and **P66 is the empirical face of the presentation question**: confluence of the monotone fragment = commutativity relations among monotone generators; (2) the firewall becomes theorem-shaped — see the imagination register entry below. Caveat kept honest: admissibility is state-dependent (the exclusion cache gates Unify; slot states gate operators), so the full structure is a state-indexed graded writer monad, not the textbook instance.

- **Can the algebra be typed?** The current spec treats all slots as untyped — they differ only in their connector geometry. A typed version would assign sort-G types to slots (agent-type, location-type, cause-type, etc.) and require that operators respect typing. This would formalize the wh-word structure (typed open variables) and give the Constrain operator a type-checking semantics. Is typing necessary, or does connector geometry subsume it?

- **What is the semantics of the imagination register?** *Resolved (2026-08-10): the register has its own algebra — specified in the companion document* ***The Imagination Register Algebra*** *(docs/sov/imagination_register.md).* In brief: the register is a second monad over the same geometry sort with a **free (ungraded) fiber** — construction is costless in receipts because nothing constructed is owed to anyone — operating on cheap copy-on-write context webs that support the full geometry sort (imagined Compose, Unify, Constrain, Abstract: the Copernican requirement, since discovering that a rival *unifies* requires performing the unification somewhere). Its scarcity is **metabolic, not epistemic** (replay budget, context count, depth) — a second currency, which is why the firewall sits between the two spaces: epistemic debts cannot be paid in metabolic coin. The firewall, stated categorically: a monad morphism exists from funded to imagined (forget the grading — anything real can be imagined) and **none exists back**; imagination reaches the ledger only through the external composite *register → generator → world → Fit* — reality is the only return arrow, and the Copernican arc is the unique path in the diagram. The register's only lawful outputs are generator directives: structures never exit; questions do. It is T154's fully-reversible rung given its algebra.

- **Does the algebra extend to multi-agent systems?** The current spec handles cross-ledger operations (Pose, Attest) but treats each agent's ledger as a separate object. A multi-agent extension would specify the topology of agent networks, the dynamics of Pose/Attest chains (an agent poses to multiple others; their Attests arrive with different discounts and must be integrated), and the population-level conservation laws that govern communal funding.

---

## 8. The Formal Spec in One Paragraph

The SOV operator algebra is a two-sorted algebra over a geometry base (sort G, the algebra of connector shapes) and a ledger fiber (sort L, the algebra of receipt histories). Named operators are lifts from geometry moves to (geometry, ledger) pairs, subject to three conditions: well-definedness on the base (the geometry move can be computed from connector geometry alone), conservativity of the fiber (the ledger move cannot create receipts the geometry move did not earn), and functoriality of the lift (the provenance of a composed operator is the composition of the provenances of its components). Legal pairings are determined by the earning table — each geometry move licenses specific fiber moves and no others. The algebra has six conservation laws (no receipt creation, ledger monotonicity, provenance grounding, Attest grounding, shape persistence, and feasibility monotonicity-with-licensed-compensation) and five boundary types (world→ledger via Fit, ledger→world via the external generator, ledger↔ledger geometry via Pose, ledger↔ledger receipts via Attest, ledger→imagination via Posit). The named operator set is **sixteen** — {Fit, Individuate, Constrain, Retract, Compose, Differentiate, Unify, Exclude, Abstract, Posit, Pose, Attest, Bind, Quote, Archaize, Anneal} — with {Transfer, Closure, Transpose, Reopen, Suspend, Counterposit, Occlude, Enumerate} derived; the fiber has nine primitives (coordination-accrue added 2026-08-10 as Bind's — the one move whose carrier, the binding, is jointly owned and inhabits the ledger↔ledger interface, funded by the COORDINATED provenance class); receipts carry AND/OR justification formulas, making the ledger a lazy truth-maintenance engine (ATMS with funding). The operator set partitions into monotone ({Fit, Constrain, Exclude, Abstract, Archaize, Anneal, Pose, Attest}) and non-monotone ({Individuate, Retract (and derived Reopen), Differentiate, Unify, Compose, Quote, Posit}) fragments; confluence of the monotone fragment under arbitrary schedules is a **conjecture** pending construction of the common lattice (the Knaster-Tarski argument requires monotone endomaps on one fixed order, not yet established for node-adding, node-evicting, clock-dependent, and cross-ledger operators), with P66 as the escrowed empirical test. The P-number registry runs P48–P70, with P71 the next available assignment.

---

*First articulated in conversation, 2026-08-09. Companion documents: Structured Open Variables (SOV), SOV Entailments, SOV Operator Algebra, SOV Geometry. Status: open — monad structure, typing, and imagination register semantics unresolved.*

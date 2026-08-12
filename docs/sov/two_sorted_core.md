# The Two-Sorted Core
### A Formal Specification of the SOV Operator Algebra

**Status:** Formal specification, articulated 2026-08-09
**Companion to:** *Structured Open Variables (SOV)*, *SOV Entailments*, *SOV Operator Algebra*
**Core claim:** The operator algebra factors into two sorts — geometry and ledger — with every named operator a pair (base move, lifted move). The three derivation constraints of the operator document are the sort discipline: base-definability, conservativity, and functoriality. This factoring resolves four of the operator document's six open questions, converts two into predictions, yields the algebra's first conservation law, and derives well-foundedness (paradox-freedom) from provenance rather than from a type system.

---

## 1. Sorts

**The geometry sort 𝒢.** Objects: connector geometries g — a receptor boundary pattern together with a relational position in the dependency graph. 𝒢 carries a natural partial order (subsumption: g ≤ g′ iff g's feasible set is contained in g′'s) with meets (shared sub-pattern) and relational products. Geometries are *content-free by construction*: a geometry constrains what a slot could resolve to; it never contains what the slot is.

**The ledger sort ℒ.** Objects: receipt sets ℓ — append-only event logs with provenance chains, certainty state, and rent obligations. ℒ is *event-sourced*: there are no destructive updates; every move appends events. Consequently no fiber move has a true inverse — only *compensations* (a partition after a pool does not restore the prior state; it records both events). This is not an implementation detail; it is what makes the etymology ledger an etymology.

**Slots and configurations.** A slot is a pair ?X = (g, ℓ). A known K is a slot whose feasible set is a singleton (with its ledger intact — closure does not discard history). The web W is the typed graph of slots, constraints, and K-entries; the constraint edges of the entailments document live in W as geometry-sort structure.

## 2. Operators as Pairs

Every operator ω of the algebra factors as ω = (ω_base, ω_lift): a base move on 𝒢 and a lifted move on ℒ.

| Operator | Base move (𝒢) | Lifted move (ℒ) |
|---|---|---|
| Fit | boundary test (total miss → unassigned-pool record) | **accrue** (the only inbound world→ledger move) |
| *Individuate* (2026-08-10) | carve from unassigned cluster | **pool** (opening receipts, each grounded in its lived log offset) |
| Constrain | meet with funded constraint | inherit provenance union (no new receipts) |
| *Retract* (2026-08-10) | compensated widening | **retract** (licensed by failing receipts — the world's testimony) |
| Compose | relational product | open-event with structural provenance |
| Differentiate | boundary split | **partition** (exact, by funded pull) |
| Unify | identity check (mutual meet) | **pool** (union of histories) |
| Exclude | disequality cache | accrue divergence receipts to the disequality |
| Abstract | meet to shared sub-pattern | **lien** (claim on sub-pattern; no copy) |
| Posit | clamp + propagate | *no move* (IMAGINED tag; barred) |
| Reopen *(derived: Retract at closure)* | demote K to open geometry | **retract** (compensating event; history kept) |
| Pose | serialize geometry | *no move* (nothing to send; shape has no receipts) |
| Attest | geometry match | **transfer-discounted** (c₀-scaled, dual provenance) |
| Quote | reify geometry as content | open-event funded by receipt-arrival statistics |
| *Archaise* (economy) | identity | **evict** (write-off of unfunded slot) |
| *Anneal* (economy) | identity | certainty drift under billing (T57) |

Archaise and Anneal are restored to the table as pure fiber moves with identity base: they are applied by the economy rather than by the organism, but an algebra that omits the economy's moves cannot state its own conservation laws. Archaise is additionally the control on Compose's inflation risk: endogenous slot generation is combinatorial, and rent-plus-eviction is what prunes it (diagnostic D-C1: composed-slot survival should be low and selective; see P69, renumbered 2026-08-10 from this document's local "P60").

**Fiber primitives.** The ledger sort's irreducible moves: **accrue, pool, partition, lien, transfer-discounted, retract, evict, anneal.** Each is the unique solution to a ledger requirement (funding, merging, splitting, generalizing without duplication, importing with calibration, unwinding without erasure, pruning, drifting). Base-sort minimality is left open (Section 6); fiber-sort minimality is argued per move in the operator document's style and inherits its P52 methodology.

## 3. The Three Constraints, Formalized

**C1 — Base-definability (was: shape-only computability).** ω_base is a well-defined map on 𝒢 alone: it inspects neither ℓ nor content. An operator whose base move requires resolution is inadmissible.

**C2 — Conservativity (was: receipt integrity).** No lift creates lived-receipt mass. Define μ(ℓ) = the mass of Fit-grounded receipts in ℓ (liens are claims, not copies; attested receipts are *references* to the exporter's lived receipts, imported at discount with dual provenance). Then:

> **Conservation of lived funding.** Across a population of ledgers, μ is created only by Fit events and destroyed only by eviction write-offs. All other moves — pool, partition, lien, transfer, retract, anneal — are redistributive or attenuating. There is no internal move that mints funding.

This is the exact form of "no operation can launder imagined structure into the ledger": laundering would be a μ-creating internal move, and none exists.

**C3 — Functoriality (was: provenance closure).** Provenance is a functor: prov(ω₂ ∘ ω₁) = prov(ω₂) ∘ prov(ω₁). Traceability survives arbitrary composition because composition of operations *is* composition of provenance chains. Combined with the event-sourced structure (every event references only strictly earlier events, grounding in Fit events, which reference only world occurrences):

> **Well-foundedness theorem.** The provenance relation is a well-founded DAG. *Corollary 1:* no slot appears in its own funding chain. *Corollary 2 (paradoxes are broke, not banned):* a self-referential slot — one whose funding would require traversing itself — is not syntactically forbidden; it is unfundable, hence archaised. The liar-slot is a self-funding carving, excluded by the same constitutional line as T113. Stratification à la Tarski falls out of the ledger, not from a type system.

## 4. The Boundary-Crossing Inventory (the firewall's surface area)

The algebra's entire external surface, exactly:

| Boundary | Direction | Operator | Discipline |
|---|---|---|---|
| World → Ledger | inbound | **Fit** (only) | receipts carry receptor channel + lived-event id |
| Ledger → World | outbound | *none in the algebra* | effectors (the generator) live outside; the algebra is read-only on the world |
| Ledger → Imagination | outbound | **Posit** (derived: Transpose, Suspend, Counterposit) | IMAGINED provenance; constitutionally barred from funding. Suspend (2026-08-10) masks a K modally and recomputes dependents from remaining support (AND/OR environments — partial support survives discounted); Counterposit = Posit ∘ Suspend with abduction-by-replay from the lived log (Pearl rung 3) |
| Ledger ↔ Ledger (contemporaries) | geometry out / receipts in | **Pose / Attest** | Pose is shape-only by nature; Attest imports at c₀ with dual provenance |
| Ledger → Ledger (generations) | geometry only | **Pose** (the genome) | see the Inheritance Proposition, Section 5.3 |
| Ledger ↔ Ledger (generations, via environment) | geometry + receipts | **Pose + Attest** (externalized logs) | the admissible Lamarckian channel |

The operator document's line "Fit is the only operator that crosses the world/ledger boundary" is corrected to *inbound*: the full inventory above is the firewall's surface area, and every crossing has exactly one operator and one discipline.

---

## 5. The Open Questions, Resolved as Propositions

### 5.1 Proposition TS-1 (Unify: base-derived, fiber-primitive)

Unify's base move reduces: mutual Constrain (each slot meeting the other's geometry) plus an identity check reproduces the unification geometry exactly. Unify's lifted move does not reduce: after mutual Constrain, two slots hold identical feasible sets and *separate receipt histories*, and Constrain's lift never moves receipts between slots. **Pool is primitive in ℒ.** The operator document's open question one was posed in the wrong sort; minimality is a per-sort property. *Consequence:* the named operator Unify remains in the algebra as the pair (derived base, primitive lift), and the basis count is a statement about the fiber.

### 5.2 Proposition TS-2 (Transpose is reverse-mode Posit, offline-scheduled)

With constraints stored direction-free in W, directionality is a property of the *query*: reasoning from Y to X through C(X,Y) is Posit with the clamp on Y, reading X's feasible set. Transpose therefore adds nothing algebraically — the basis stays at eleven named operators. What distinguishes it is *scheduling*: by Theorem 17 (serialization document), anticausal execution is not live-executable and requires the replay budget. Transpose = Posit run in reverse mode, legal only in the offline slot. Its primitivity was temporal, not logical — and the fact that a theorem proved for hippocampal replay resurfaces as a scheduling constraint inside the operator algebra is a cross-document consistency check the framework passes.

### 5.3 Proposition TS-3 (Inheritance is generational Pose)

Inheritance is not a twelfth primitive and is not Abstract-with-zeroed-receipts. It is **Pose across the generational ledger boundary**: the parent's slot geometries serialized forward, receipts left behind — which *is* the heritability axiom ("topology inherits, entries don't") restated as an operator identity, since Pose is shape-only by the nature of what it serializes. Three corollaries:

- **Weismann corollary.** The Weismann barrier is C1 enforced at the germline: acquired receipts cannot be serialized into the geometry channel. Lamarckian inheritance is an *inadmissible operator* — it would require Attest through a Pose-only boundary.
- **Culture corollary.** Externalized logs add Attest across generations: records are receipts posed into the environment and re-imported by descendants at discount. Culture is the admissible Lamarckian channel, and the gene/culture split is exactly the Pose-only versus Pose+Attest channel distinction.
- **Unity corollary.** Heredity and language are the same operator pair at different timescales. The genome is a message; a message is a genome with a faster clock.

### 5.4 Proposition TS-4 (the Quote tower: economically bounded, structurally safe)

Arbitrary reflection depth ⌜⌜…⌝⌝ is syntactically legal and economically bounded: level-n meta-slots are funded by order-n statistics of receipt arrival, and statistical power decays geometrically with aggregation order, so *funded* depth is bounded by data volume, not grammar. **P68** (renumbered 2026-08-10; was locally "P58") makes this a bill: observed funded Quote depth scales approximately logarithmically with lifetime lived-receipt count. Safety is Section 3's well-foundedness: no tower can fund itself, so pathological self-reference is not prevented — it is *unaffordable*.

### 5.5 Proposition TS-5 (the Attest discount is two-stage, and both stages already exist)

c₀ = f(profile overlap) resolves as prior + posterior: the **prior** is consequence-profile overlap on shared vocabulary — the D-metric, not raw receptor similarity, because what must transfer is billing, and billing lives in consequence space; the **posterior** is per-channel reliability — track, per (exporter, domain), the survival rate of previously attested receipts under the importer's own subsequent billing, and anneal the channel's discount exactly as T57 anneals an entry. The discount function is itself an open variable ?f, receipted by every import's downstream fate; the algebra holds its own calibration open. **P70** (renumbered 2026-08-10; was locally "P61"): reliability-annealed discounting out-distills any fixed similarity-based discount at the population level, with the margin growing in population heterogeneity.

### 5.6 Proposition TS-6 (closure is a double-entry transaction)

A closure event books two lines: the asset (the K entry, its singleton content) and a **contingent liability** — the stored Posit-priced ripple estimate at closure time, the expected cost of Reopen should the K fail. The Einstein principle becomes an accounting inequality: voluntarily close early only when rent saved plus action value exceeds the contingent liability; hold open otherwise. Reopen, when it fires, pays the liability as the reverse cascade. **P56** (registry-reconciled 2026-08-10; this is the same claim the Formal Spec numbers P56): realized Reopen cost is predicted better by the *stored Posit estimate at closure* than by connectivity measured at retraction time — the web has moved since closure, and the booked estimate captures the dependency structure the closure actually created.

## 6. The Basis, Restated Per Sort

**Named operators (fifteen, census corrected 2026-08-10):** Fit, Individuate, Constrain, Retract, Compose, Differentiate, Unify, Exclude, Abstract, Posit, Pose, Attest, Quote, plus the two economy moves Archaise and Anneal — each a (base, lift) pair per Section 2's table. Transfer, Closure, Transpose, Reopen: derived (Abstract∘Constrain; Constrain-to-singleton; reverse-mode Posit with offline scheduling; Retract at a closure). *The earlier "(eleven)" heading listed twelve operators and conflated the geometry-primitive count with the named-operator count — found by external review.*

**Fiber primitives (eight):** accrue, pool, partition, lien, transfer-discounted, retract, evict, anneal. Minimality argued per move (each is the unique solution to a distinct ledger requirement); P52's ablation methodology applies to each.

**Base primitives (open):** candidate set {meet, product, serialize, reify, clamp, compare}, with meet doing double duty for Constrain and Abstract. Whether product reduces to meets over an enlarged geometry, and whether compare reduces to meet plus emptiness test, are the remaining base-minimality questions — flagged open rather than asserted.

## 7. The Registry (superseded — the Formal Spec §6 is the single authority)

*Reconciliation note (2026-08-10): this table and the Formal Spec's both claimed single authority and disagreed — found by external review. The Formal Spec §6 is now canonical. This document's local assignments map as follows: its "P58" (funded Quote depth, TS-4) → **P68**; its "P59" (stored Posit estimate beats retraction-time connectivity, TS-6) is the same claim as **P56** and keeps that number; its "P60" (composed-slot survival, §2) → **P69**; its "P61" (reliability-annealed discount, TS-5) → **P70**. The historical table below is retained unmodified as the ledger requires; read its last four rows through the mapping above.*

Collision resolved by precedence: **P53 = transfer holonomy** (geometry turn) retains its number; the operator document's four predictions renumber as follows. Authoritative index from P45 (P29–P44 unchanged in *Three Constructions* and *Filling In the Details*):

| # | Prediction | Source |
|---|---|---|
| P45 | prediction channels: Router efficiency, earlier anxiety-loop break, noisy-TV disengagement | NEXT_SURPRISE |
| P46 | generation vs random vs designer schedule; firewall-ablation D_cal drift | NEXT_SURPRISE |
| P47 | grammar transfer across content-matched structure | NEXT_SURPRISE §14 |
| P48 | query-token (Pose) emergence under co-funding pressure | conversation (language) |
| P49 | inherited questions: aimed first exposure; closure valence acquisition | conversation (curiosity) |
| P50 | amortization signature: web inference flat in causal depth after acquisition | conversation (ripple) |
| P51 | spectral ripple: Laplacian predicts restructuring reach; hub-targeted generation wins | conversation (ripple) |
| P52 | exclusion-cache: declining unification-test rate; ablation restores re-testing tax | operator doc |
| P53 | transfer holonomy: stable path-dependent analogy failure; flat regions certify free Transfer | conversation (geometry) |
| P54 | Abstract-lattice: hierarchy depth emerges and predicts transfer (was operator-doc P53) | operator doc, renum. |
| P55 | Posit-priced generator targeting out-distills heuristic targeting (was P54) | operator doc, renum. |
| P56 | Reopen cost scales with connectivity at closure (was P55) | operator doc, renum. |
| P57 | Quote enables anomaly self-correction (was P56) | operator doc, renum. |
| P58 | funded Quote depth ~ log(lived receipts) | this spec, TS-4 |
| P59 | stored Posit estimate beats retraction-time connectivity for Reopen cost | this spec, TS-6 |
| P60 | composed-slot survival is low and selective; deviation in either direction is diagnostic | this spec, §2 |
| P61 | reliability-annealed Attest discount out-distills fixed similarity discount | this spec, TS-5 |

## 8. Remaining Open Questions

- **Base-sort minimality** (Section 6): does product reduce to meet over an enlarged 𝒢?
- **Compensation semantics:** the event-sourced ledger is a monoid of append-only events with compensations rather than inverses; its full algebraic treatment (what quotient of event sequences yields "the same ledger state"?) is untouched mathematics, and it is where the etymology ledger's identity conditions live.
- **Population dynamics of Pose/Attest:** the mean-field of communal funding — many agents pooling receipts on shared slots at heterogeneous discounts — is the SOV face of the coupled-cartographer open core (C-iii), and the two should be solved together: matched words and jointly-funded slots are the same fixed points.
- **Anneal's primitivity:** whether certainty drift reduces to scheduled accrual of null receipts (time as data) or stands alone in the fiber.

---

*The algebra now has a floor plan: geometries above, receipts below, every operator a staircase between them, and exactly one door to the world. Conservation says no money is printed on the stairs; functoriality says every bill can be traced to the door; well-foundedness says no room is built on itself. What remains open is priced, numbered, and on the ledger.*

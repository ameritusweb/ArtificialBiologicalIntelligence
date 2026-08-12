# SOV Operator Algebra
### The Complete Set of Legal Moves on Structured Open Variables

**Status:** Theoretical derivation, first articulated 2026-08-09  
**Origin:** Companion to Structured Open Variables (SOV) and SOV Entailments  
**Core claim:** The operator set is not designed — it is forced. Three constraints derive every operator: shape-only computability, receipt integrity, and provenance closure. Any operator not satisfying all three is inadmissible. Any move the system needs to make that lacks an operator is a gap in the algebra.

---

## 0. The Three Derivation Constraints

Every operator in this algebra must satisfy all three of the following. These are not preferences — they are the conditions under which the ledger remains honest.

**Shape-only computability.** Operators act on connector geometry — the receptor boundary pattern and relational position of a slot in the dependency graph. They do not act on content. An operator that requires knowing what an unknown *is* in order to apply is inadmissible: it would make open-variable operations contingent on resolution, which collapses the framework back into premature closure.

**Receipt integrity.** No operator creates funded structure that was not earned. Every output of every operator carries a receipt tracing its funding to lived data or to prior funded operations. An operator that generates structure without a receipt chain is inadmissible: it would launder imagined structure into the ledger.

**Provenance closure.** Every operator's output must carry where its funding came from. This is the firewall made algebraic. Without provenance closure, a sequence of operations can obscure the origin of a claim — producing an entry that appears funded but whose receipt chain, if traced, leads back to imagination or to another agent's unverified ledger. Provenance closure means the chain is always traceable, regardless of how many operators were applied.

These three constraints are not independent — and they are not arbitrary. They are the sort discipline of a two-sorted algebra in disguise.

The algebra factors into two sorts: **geometry moves** (constrain a boundary, intersect patterns, compose a relational product, reflect, clamp) and **ledger moves** (accrue, pool, partition, lien, discount-and-transfer, retract). Every named operator is a pair — one geometry move and one ledger move composed together. Differentiate = geometry-split + receipt-partition. Unify = identity-check + receipt-pool. Abstract = geometry-intersection + lien. Attest = geometry-match + discounted-transfer. Reopen = demotion + retraction-event.

The three constraints, restated in sort terms:

- **Shape-only computability** = operators are well-defined on the geometry base. The geometry move can be specified without knowing slot content.
- **Receipt integrity** = the ledger lift is *conservative*. No funding is created that the base geometry move didn't earn. The fiber cannot outrun the base.
- **Provenance closure** = the lift is *compositional* — the provenance of a composed operation is the composition of provenances. This is **functoriality**: the property that makes chains traceable regardless of how many operators were applied. Provenance closure was category theory all along.

Together they define the boundary of the algebra: what is inside is legal; what violates any of the three is outside. Minimality must be argued per sort — geometry primitives and fiber primitives are separate inventories.

---

## 1. The Primitive Operations

### 1.1 Fit — `?X ← data`

**What it does:** Tests incoming data against `?X`'s receptor boundary. The data either falls within the boundary, falls outside, or falls on the boundary itself.

**Outputs:**
- *Match* — data is within boundary; accrues as a positive receipt on `?X`, narrowing its feasible set
- *Mismatch* — data is outside boundary; accrues as a negative receipt, funding Differentiate or boundary revision
- *Boundary case* — data is at the edge; accrues as a receipt funding boundary refinement

**Receipt integrity:** Every output is a receipt. Fit never generates structure — it only updates the funding state of an existing slot.

**Provenance:** Receipts from Fit carry the receptor channel and the episode identifier of the lived event that generated them. They are the most primitive funded objects in the algebra — the only operators whose receipts trace directly to experience rather than to prior operations.

**Relationship to existing machinery:** Already implemented in the receptor topology. Fit is the base case — every other operator either generates slots that Fit can subsequently act on, or reorganizes the receipt structure of slots that Fit has already touched.

**Note:** Fit is the only operator that crosses the world/ledger boundary *inbound* — carrying lived data from the world into the ledger as receipts. The boundary-crossing inventory in full: Fit crosses world→ledger (inbound); Pose and Attest cross ledger↔ledger (peer agents); Posit crosses ledger→imagination (offline, IMAGINED provenance); the generator crosses ledger→world (outbound, effector — rightly outside the algebra). These are the firewall's five surfaces. All five must be stated exactly because together they define the algebra's perimeter.

---

### 1.2 Constrain — `?X ⊃ C`

**What it does:** Adds a structural constraint to `?X`'s connector geometry without resolving it. Narrows the feasible set of `?X` by ruling out any resolution incompatible with `C`, while leaving the slot open.

**Outputs:** `?X` with a tighter connector geometry. The constraint `C` is added to `?X`'s boundary conditions. No receipt is generated — Constrain is funded by whatever funded the constraint `C` itself.

**Receipt integrity:** Constrain is funded transitively. The constraint `C` must itself be a funded structure — either a known (`K`), a prior receipt, or the output of another funded operation. Applying an unfunded constraint is inadmissible.

**Provenance:** The provenance of Constrain's output includes both `?X`'s prior receipt history and the provenance of `C`. The combined chain is the output's funding trace.

**Relationship to other operators:** Constrain is the most general operator. Several others reduce to it or decompose through it:
- The wh-words are pre-loaded Constrain operations: *who* is `?agent ⊃ [agent-role]`
- Outside-in resolution is Constrain iterated to a singleton: the slot closes when Constrain has nothing left to remove
- Transfer decomposes as Abstract followed by Constrain (see Section 1.9)

**Closure as iterated Constrain:** This deserves explicit statement. "Outside-in resolution" — the principle that a slot closes only when the surrounding structure makes closure inevitable — is not a separate principle. It is the operational definition of Constrain exhausting its feasible set. A slot `?X` closes (`?X → K`) when no further application of Constrain can distinguish between any remaining candidates. Closure is not a separate operation. It is the terminal state of Constrain.

---

### 1.3 Compose — `?X ∘ ?Y → ?Z`

**What it does:** Two unknowns in a structural relationship produce a third unknown. The connector geometry of `?Z` is derived from the geometries of `?X` and `?Y` plus the relational operator between them.

**Outputs:** A new open variable `?Z` whose connector geometry is the relational product of `?X` and `?Y`. `?Z` opens automatically when `?X` and `?Y` are in a structural relationship that implies something between or beyond them.

**Receipt integrity:** `?Z` is funded by the structural necessity of the relationship between `?X` and `?Y`. Its opening receipt is: "given `?X` in role R1 and `?Y` in role R2, `?Z` is the slot for whatever connects/mediates/follows." The funding is structural, not empirical — but it is funded, because `?X` and `?Y` themselves carry receipt histories.

**Provenance:** `?Z`'s provenance traces to the provenance of `?X` and `?Y` plus the relational operator that generated it. It does not inherit their receipts — it inherits only the structural necessity that their relationship implies.

**This is the endogenous slot generator.** Compose is how the system produces new unknowns without designer specification. The web grows because existing structure implies new structure — not because someone decided to add a variable.

**Example:** `?cause ∘ ?effect → ?mechanism` opens the slot for whatever connects them. `?organism ∘ ?environment → ?adaptation` opens the slot for the interface. The third slot is not designed — it is composed.

---

### 1.4 Differentiate — `?X → ?X1 | ?X2`

**What it does:** When incoming data pulls `?X` toward two incompatible boundary positions simultaneously — when the connector geometry is being stressed in two directions — the slot splits into two child slots.

**Outputs:** `?X1` and `?X2`, each inheriting a partition of `?X`'s receipt history. The partition is not arbitrary: receipts go to the child whose boundary position they were funding. Neither child starts empty — they inherit.

**Receipt integrity:** The partition of receipts is exact. Every receipt on `?X` is attributed to `?X1` or `?X2` based on which pull it was funding. No receipt is lost; no receipt is double-counted.

**Provenance:** Each child carries its own provenance trace — the subset of `?X`'s receipt history assigned to it. The split event itself is logged in the etymology ledger with the stress pattern that triggered it.

**Relationship to Exclude:** Differentiate is Exclude discovered *inside* one slot. When the system discovers that what it has been treating as one thing is actually two things, it Differentiates. When it tests whether two *separate* slots are one thing and finds they are not, it Excludes. The operations are structurally dual — one working from within a slot outward, the other working from between slots inward.

---

### 1.5 Unify — `?X ≡ ?Y`

**What it does:** Tests whether `?X`'s connector geometry is structurally isomorphic to `?Y`'s — whether they occupy the same relational role in their respective dependency neighborhoods. If yes, merges them into a single slot funded by both receipt histories.

**The isomorphism test:** Two slots are unification candidates when they exhibit the same pattern of relationships to their neighboring nodes in the dependency graph, weighted by receipt density. Not identical receptor boundaries — identical *relational role*. The conflation detector already computes a version of this; Unify runs it prospectively rather than retrospectively.

**Outputs on success:** A single unified slot `?XY` whose connector geometry is the merged geometry of `?X` and `?Y`, and whose receipt history is the union of both. Every constraint either earned independently now applies to both.

**Outputs on failure:** Exclude fires — see Section 1.6. The failed unification is itself funded structure.

**Receipt integrity:** Receipt pooling on unification is the most powerful consequence of the operator. The unified slot is funded by both histories simultaneously. Resolution of the unified slot will be richer than either side could have reached alone.

**The Maxwell case:** Maxwell's recognition that the equations governing electricity and magnetism occupied the same relational role in their respective domains was a unification test. The test succeeded. Receipts pooled. Electromagnetic radiation — the resolution — was funded by both histories.

**Resolution of the Unify vs Constrain question:** Unify *does* reduce to mutual Constrain plus an identity check on the geometry base — each slot constraining the other with its own geometry, the unified slot emerging when both constraints produce the same feasible set. This reduction is exact and complete on the geometry sort. But it leaves two slots with identical feasible sets and *separate receipt histories*. Constrain never moves receipts between slots. **Receipt pooling is an irreducible fiber move.** Unify is derived on the base and primitive on the fiber. The reduction question was posed in the wrong sort — minimality must be argued per sort, and pool is a fiber primitive that no geometry operation can replace.

---

### 1.6 Exclude — `?X ≢ ?Y`

**What it does:** Caches the result of a failed unification test as permanent funded structure. "Whatever `?X` and `?Y` are, they are not one."

**Why this is not just the absence of Unify:** Without caching failed unification tests, the system re-runs them every time the two slots appear in proximity. The re-testing tax grows with the size of the match space. Exclude prunes the match space monotonically — each cached disequality is a constraint that eliminates one candidate unification permanently.

**Outputs:** A funded disequality constraint between `?X` and `?Y`, stored in the match space alongside the receipt history of the divergent receipts that broke the test.

**Receipt integrity:** The disequality is funded by exactly the divergent receipts that failed the unification test. Not by the absence of receipts — by the presence of receipts that pointed in incompatible directions.

**Provenance:** The Exclude entry carries the specific receipts that diverged and the structural positions where the divergence was localized.

**Ancestry:** Disunification, Prolog's `dif/2`, and Waltz filtering — which derived most of its constraint-propagation power from exclusion constraints rather than positive matches.

**Relationship to Differentiate:** Differentiate is Exclude discovered *inside* one slot — the moment when one slot's receipts diverge enough to split it. Exclude is Differentiate between slots that were never unified — the moment when two separate slots' geometries are confirmed as distinct. Structurally dual.

**P52 — the exclusion-cache signature:** An algebra with Exclude shows unification-test rates declining over lifetime on stationary structure while match accuracy holds. Ablating the cache produces the re-testing tax. If the tax does not reappear on ablation, Exclude was not load-bearing and the basis shrinks by one.

---

### 1.7 Abstract — `?X ⊔ ?Y → ?Z`

**What it does:** Takes two slots that are demonstrably different (Exclude has fired between them) yet share partial connector geometry, and opens the superordinate slot holding exactly the shared sub-pattern.

**The lattice dual of Unify:** Unify descends toward the specific — merging two slots that occupy the same role. Abstract ascends toward the general — opening the parent that the shared sub-pattern implies. Together they make the slot space a subsumption lattice: Unify descends, Abstract ascends, and the lattice grows in both directions by receipts.

**Ancestry:** Anti-unification — Plotkin and Reynolds, 1970. Unification computes the most specific unifier; anti-unification computes the least general generalization. The two operations are the lattice-theoretic duals: meet (Unify) and join (Abstract).

**Outputs:** A new slot `?Z` whose connector geometry is the intersection of `?X`'s and `?Y`'s geometries — only the shared sub-pattern, nothing specific to either.

**Receipt integrity:** `?Z` holds a *lien* on the shared sub-pattern of each child's receipts — not a copy. The child receipts are not duplicated; the parent has a claim on the portion of each child's receipt history that pertains to the shared geometry. Nothing is double-funded.

**Taxonomy as the lattice:** Every taxonomic hierarchy is the Abstract lattice grown by receipts. "Mammal" is the Abstract of "dog," "whale," "bat," and many others — the superordinate slot holding exactly the shared connector geometry, funded by a lien on each child's receipt history. The hierarchy was not designed; it was grown upward by accumulated Exclude and Abstract operations.

**Transfer decomposes through Abstract:** The transfer of connector geometry from one domain to another — the analogy operator — decomposes as Abstract followed by Constrain. Abstract extracts the pure relational geometry shared between `?X` and its domain context; Constrain re-specializes that geometry into the new domain. Analogy is join-then-restrict on the lattice.

---

### 1.8 Posit — `do(?X ≐ v)`

**What it does:** Hypothetically clamps a slot to a candidate content `v` and propagates the consequence through the constraint web — the ripple run in content mode, offline, writing nothing to the ledger.

**Pearl's do-operator lifted to unknowns:** Pearl's do-calculus allows reasoning about what *would* happen if a variable were set to a value, without actually intervening. Posit extends this to unresolved slots: "what would the constraint web look like if `?X` resolved to `v`?" The propagation is full and real; the writing is prohibited.

**Why the system cannot function without it:** The generator's targeting rule — resolve the unknown whose closure ripples furthest — requires pricing closures before buying them. P51's expected-restructuring computation is Posit iterated over closure candidates. Without Posit, the system cannot choose which unknown to target; it can only act randomly or follow a fixed schedule.

**Provenance closure — the critical constraint:** Posit outputs carry IMAGINED provenance. They are constitutionally barred from funding the ledger, appearing in receipt chains, or being cited as evidence for any other operation. The imagined/lived distinction is absolute and enforced at the provenance level, not by convention.

**The two-phase ripple:** Section 4 of the entailments document distinguishes shape-phase (what does learning this touch? — free from topology) from content-phase (what do the touched values become? — requires propagation). Posit is the content-phase operator. It purchases the hypothetical propagation without touching the ledger. Reachability (the shape phase) remains free; Posit makes the content phase available at the cost of computation but not at the cost of ledger integrity.

**Relationship to imagination and planning:** Any cognitive operation that reasons about hypotheticals — "what if I did X?", "what would happen if ?Y were the case?" — is Posit. Planning is iterated Posit over action sequences. Hypothesis formation is Posit over candidate resolutions. Scientific modeling is Posit run against the current constraint web to generate predictions.

**Suspend and Counterposit — the counterfactual modes *(registered 2026-08-10; refined same day after second external review; derived, not new primitives)*:**

*The modal axis.* Suspension is not a lifecycle transition — it is an evaluation mode. The state has two axes: **epistemic status** (open | closed | archaized — the ledger's truth) and **evaluation mode** (actual | hypothetical(context)). Within a context, a K's closure is masked and its pre-closure geometry restored; the actual ledger never changes, and the same K can participate in several simultaneous contexts. The state *rationally closed, hypothetically open* is a (closed, hypothetical) coordinate, not a third status.

*Support environments — the ATMS refinement.* "Sever everything funded through K" over-suppresses: with `K∧A→C` and `B→C`, suspending K must leave C standing on B, discounted — remove K's *contribution* and recompute, never blindly erase. This requires justification semantics richer than a parent-ID list: receipts carry AND/OR support formulas (Compose is AND — a mediator needs both parents; independent fits are OR), and a conclusion survives Suspend(K) exactly when at least one of its support environments excludes K. Computed lazily per suspension (full ATMS label sets are worst-case exponential and unnecessary). **This is where the ledger stops being provenance logging and becomes a truth-maintenance engine** — de Kleer's ATMS (1986), given receipts, funding economics, and the imagination firewall.

*The two operations:*

> `Suspend(K)` — mask K's testimony in a copy-on-write context; recompute every dependent structure from its remaining support paths. Output: a non-funding hypothesis package — what is gone, what survives discounted, what stands independently, per-slot context certainties, the exact assumptions suspended — all IMAGINED. Pure Suspend is a *robustness audit*: how much of the web stands on this belief alone?
>
> `Counterposit(K, v′) = Posit(v′) ∘ Suspend(K)` — Pearl's third rung: clamp the alternative and propagate within the context, with certainties taken from the context so the K's implications do not testify at its own trial.

*Abduction-by-replay — why rung 3 is cheap here.* Pearl's counterfactual is abduction → action → prediction; replace-and-propagate alone is only rung 2. Pearl needs the abduction step because ordinary systems discard raw history and must infer the background back from the posterior. This architecture kept it: **the append-only experience log IS the background context.** Abduction here is replay — re-fit a window of actual lived observations through the suspended context and ask what the actual history would have looked like without the K. Its output is exactly the Copernican payload: the *discriminating observations* — lived events only the masked K explained — which is what any rival structure must account for and what the generator should target. Machinery-before-purpose again: C4's log, built for honesty, turns out to be what makes proper counterfactuals a bounded replay instead of a posterior inversion.

*The Copernican arc — the legal route through the firewall:* geocentrism predicted well; no 404s licensed Reopen, and Reopen was the wrong instrument anyway. Suspend the K; Counterposit the alternative; discover in imagination that the rival **unifies what looks disparate**; replay the lived record to find the discriminating observations; let the generator seek them in the world; lived receipts arrive; only then do 404s plus the waiting rival license Retract/Reopen. Imagination never funds the revolution — it identifies which world-facing test might.

*Dogma, defined structurally:* **a closure whose constraint remains active in every hypothetical context in which the closure itself is being evaluated.** The corresponding testable spec — the **anti-dogma property**: suspending K removes all and *only* the conclusions exclusively dependent on K, preserves independently grounded conclusions (discounted by lost support), leaves the actual web bit-identical, and produces no funded receipts. All four clauses are in the operator battery.

*Sort status:* both derived — modal masking plus support recomputation over the existing hypothetical-propagation base; the fiber remains empty (IMAGINED, no ledger write). Replay-gated like Posit and Transpose.

---

### 1.9 Reopen — `K → ?X` *(status revised 2026-08-10: DERIVED — see §1.14 Retract and §2.4)*

**What it does:** Demotes a closed entry (a known, `K`) back to an open slot, with its receipt history intact and its dependents notified. It is the closure case of generalized Retract.

**The missing lifecycle move:** The seven original operators all operated on open slots; closure appeared only as terminal — a one-way door. But a K whose receipts begin failing — channel 404 firing on resolved structure — must be demotable. Without Reopen, premature closure is fatal: once a slot closes wrongly, there is no recovery. With Reopen, closure is a position that can be unwound at market price.

**The cost:** Reopen triggers the restructuring cascade in reverse. Every dependent node that was repriced when `?X` closed must be notified that the closure is being retracted. The cost is proportional to `?X`'s connectivity — exactly as the option value of holding open was proportional to connectivity. The option analysis and the Reopen cost are two faces of the same equation.

**Receipt integrity:** The receipt history is not erased on Reopen. The closed entry's receipts remain in the etymology ledger, now tagged as the receipt history of a retracted closure. The reopened slot inherits this history — it does not start fresh. The ledger is append-only; Reopen adds a retraction event, it does not delete the closure event.

**Provenance:** The Reopen event is logged with the specific 404 receipts that triggered it — the failed predictions on resolved structure that made retraction necessary.

**Kuhn's scientific revolutions as Reopen:** Normal science is Fit — accumulating receipts on open slots and closed entries. Crisis is 404 firing on a K — the paradigm's predictions failing against new data. Revolution is Reopen at a hub — the high-connectivity closed entry retracted, its dependents notified, the restructuring cascade propagating through the web. The cost of revolution scales with the hub's connectivity. Einstein's revolution was expensive because mass, energy, space, and time were all dependents of the closed classical entries that Reopen retracted.

**Relationship to the Einstein principle:** The Einstein principle — hold unknowns open as long as the surrounding structure warrants it — is the policy that minimizes Reopen costs. Reopen is what you pay when the Einstein principle was violated. The two are complementary: one is the preventive policy, the other is the recovery operator.

---

### 1.10 Pose — `Pose(?X)`

**What it does:** Serializes `?X`'s connector geometry and broadcasts it to other agents' ledgers. Shape only — since an open slot has no content, Pose cannot leak content regardless of implementation.

**Why shape-only is guaranteed:** An open variable's connector geometry is its receptor boundary pattern and relational position. It has no content to serialize. Pose is trivially shape-only by the nature of what it serializes. This is not a constraint imposed on Pose — it is a consequence of what open variables are.

**Outputs:** A serialized connector geometry object, broadcastable across ledger boundaries. Readable by any agent whose receptor topology has sufficient overlap to parse the geometry.

**A question is Pose(Constrain(?X)):** The wh-structure of natural language is Pose applied to a Constrain-typed unknown. "Who did it?" is `Pose(?agent ⊃ [agent-role, past-tense, this-event])`. The question serializes the geometry of the unknown and broadcasts it, requesting Attest from any agent whose ledger contains relevant receipts.

**Teaching as scheduled Pose:** Teaching is Pose and Attest scheduled along the learner's fringe — the boundary of the learner's current open slots. The teacher poses the geometry of slots the learner is ready to fund; the learner attests receipts against them. Curriculum is the sequencing of Pose operations along the learner's development.

**P48's query tokens:** In the bootstrapped-language setting, query tokens — the proto-wh-words that emerge before full language — are what Pose looks like when it earns a phoneme. A query token is a Pose operation that has been compressed into a stable signal by the population's receipt dynamics.

---

### 1.11 Attest — `Attest(?X, receipts)`

**What it does:** Exports receipts from one agent's ledger against a slot geometry posed by another agent. The cross-ledger receipt transfer.

**The cross-ledger pair with Pose:** Pose broadcasts geometry; Attest responds with evidence. Together they are the operator substrate of language — the mechanism by which communal funding of shared unknowns is possible.

**Receipt integrity across ledgers — the two-stage discount function:** Attested receipts do not enter the importing ledger at face value. The discount is two-stage:

*Stage 1 — Prior:* Consequence-profile overlap on shared vocabulary, measured in the D-metric (consequential distance), not raw receptor boundary similarity. What must transfer is *billing* — the downstream consequence structure of receipts — and billing lives in consequence space, not observation space. Two agents with similar receptor boundaries but different consequence profiles are poor Attest partners; two agents with different receptor boundaries but matched consequence profiles are good ones.

*Stage 2 — Posterior:* Empirical reliability. Track, per exporter and per domain, the survival rate of previously attested receipts under the importer's own subsequent billing. Anneal the channel's discount exactly as T57 anneals an entry — a channel whose attested receipts consistently survive earns higher weight; one whose receipts consistently fail earns lower weight. The discount function is itself an open variable the algebra can hold: `?f`, receipted by every import's downstream fate, converging under its own annealing.

The discount is not a distrust penalty — it is calibration. The receipt chain remains traceable to the originating lived experience through the discounting agent to the importing ledger.

**Provenance:** Attested receipts carry dual provenance: the original agent's ledger identity and the discount factor applied. The receipt chain is always traceable back to the originating lived experience, through the discounting agent, to the importing ledger.

**Communal funding:** The whole language loop — curiosity shared, questions posed, answers attested, receipts pooling across a population — is Pose and Attest iterated. The parallelization of distillation across a population is this pair operating at scale.

---

### 1.12 Quote — `⌜?X⌝`

**What it does:** Takes a slot's connector geometry *as data* and opens a new slot about it. The geometry of `?X` becomes the content of a new unknown `⌜?X⌝`.

**The reflection operator:** Quote makes the algebra self-referential. The system can hold open variables about its own open variables — unknowns about the structure of its ignorance. This is what meta-unknowns require: "`?X` is a `?Y`-kind of thing" requires `?Y` to range over slot geometries, which requires geometries to be quotable objects.

**Channel 404 as Quote:** When incoming data does not fit any existing slot — ungrammatical surprise — the system cannot generate a Fit receipt. Instead, it Quotes the failure pattern and opens an unknown about its own map of unknowns. The failure pattern becomes the connector geometry of a new meta-slot. The system is now holding an open variable about the structure of what it cannot yet describe.

**The fold, applied to the algebra itself:** Quote is the operator that closes the algebra over itself. Without Quote, the algebra operates on the world but cannot operate on its own operations. With Quote, every slot and every receipt can become the subject of a new slot. The ledger can model its own ledger.

**Funding of meta-slots:** Meta-slots earn receipts from the statistics of receipt arrival — the patterns in how funding flows, where 404s cluster, which slots are systematically underfunded. A slot whose funding rhythm is anomalous is exactly what a Quoted slot detects. Meta-slots are funded by the second-order structure of the receipt economy.

**Relationship to scientific self-correction:** The T-index has been running Quote implicitly from the beginning. Every time a theory falsification opened a new question about *why* the theory failed — not just what to replace it with — that was Quote. "Why did T55 fail in the way it did?" is `⌜T55⌝` — the slot about the slot, funded by the receipt pattern of T55's failures.

---

### 1.13 Individuate — `?new ← U` *(registered 2026-08-10)*

**What it does:** Carves a new first-order slot from a coherent cluster of **unassigned lived observations** — evidence that matched no existing slot's boundary. The lift is (carve, pool): the geometry move derives connector geometry from the cluster's shared activation profile; the fiber move re-homes the cluster's entries as opening receipts, each grounding directly in its lived log offset.

**Why it was missing, and why that mattered:** Compose, Abstract, Quote, and Differentiate all create slots *from existing slots*. Nothing created a slot from evidence no existing geometry could accept. The operator doc previously assigned ungrammatical surprise to Quote — but Quote reifies an existing slot's geometry or receipt statistics, and an unmatched pattern has no slot to quote. A meta-slot *about* the failure stream is not a first-order distinction *carved from it*. The gap was found by external review (2026-08-10), and it is the sharpest possible gap for this program: slot genesis from unassigned evidence **is receptor discovery — growing new eyes (T129)** — the mechanism the entire architecture is built around, already implemented at the organism level (`receptor_discovery.discover`) and already named at the geometry level (re-basis, P61's phase transition). The algebra was missing the program's own signature move.

**Fit generalized, not a new boundary:** Fit remains the only world→ledger crossing. It is generalized to record web-level 404s — observations with real activation that fit nothing — in a bounded **unassigned pool** (pre-ledger records, not receipts). Individuate consumes coherent clusters from that pool. Funding begins at individuation: the opening receipts carry `source_operator='Individuate'` (outside the Law 1 Fit-mass census) and ground in the entries' log offsets.

**Receipt integrity:** Opening receipts are licensed by exactly the lived observations in the cluster — never fewer than the pre-registered minimum cluster size, never by imagination. A carve whose threshold pattern would be empty is refused. Genesis never closes the new slot (radius floored above the closure radius): a distinction is born open.

**Determinism:** Cluster seeds are tried in arrival order; no RNG. Two identical pools individuate identically.

**Fragment:** Non-monotone (creates structure) — replay-gated with the rest of the genesis operators.

---

### 1.14 Retract — `C ↓ ?X` *(registered 2026-08-10)*

**What it does:** Generalized compensation: withdraws a specific piece of funded structure — a constraint on an *open* slot, or a batch of attested imports from one exporter — under license from failing receipts, without touching the rest of the slot's state.

**Why it was missing, and why that mattered:** The algebra could narrow (Constrain, Law 6) and could unwind a *closure* (Reopen), but it could not withdraw one bad constraint from an open slot, revoke an unreliable Attest, or correct boundary drift short of splitting. Fit's own specification promised that mismatch receipts fund "Differentiate **or boundary revision**" — and no boundary-revision operator existed. Found by the same external review.

**The license is the world's testimony:** `Retract` fires only with non-empty, non-IMAGINED failing receipts. This is T154's deletion clause surfacing inside the algebra: funded structure is never removed by taste, schedule, or convenience — only the world's testimony licenses compensation. Law 6 is restated accordingly: *feasibility narrows monotonically except under logged compensation events, and every compensation is licensed by failing receipts.*

**Compensation, not inversion:** The event-sourced ledger has no true inverses (two-sorted core). Retracting a constraint restores the recorded pre-constraint feasible set while the original `constrained` event remains logged; revoking an attestation marks the imported receipts retracted (history preserved), compensates the rent they credited, and bills the exporter's reliability posterior. Every retraction appends a `retraction` receipt parented on the licensing testimony.

**Reopen is now derived:** `Reopen` = Retract at a closure — retract enough closure-defining constraint that the feasible set is no longer a singleton, with dependents notified and the contingent liability paid. It keeps its name (the 404 window calls it directly) but its lift is Retract's: (compensated widening, retraction event).

**Fragment:** Non-monotone (widens feasible sets) — replay-gated.

---

### 1.15 Bind — `?X_A ⋈ ?Y_B` *(registered 2026-08-10; mechanism scheduled post-E1; P75 is its observational precursor)*

**What it does:** Establishes a persistent identification between slots in *different* webs — "these two positions are one" — as a **jointly-owned interface object**, without transmitting anything.

**The de-mystification, first.** Non-local structural identity is real and entirely common-cause. Two lawful sources: (1) **shared world** — each web is a quotient of the same territory (the Erlangen construction), so two slots are "the same position" when they are images of the same world-invariant under two quotient maps; the identity lives in the base, seen from two fibers — no channel required, because the correlation was installed by the common origin; (2) **shared inheritance** — the trunk: slots that were literally one slot, Posed across the generational boundary; T95's invariants are a population-wide standing binding, which is why reference arrives pre-aligned and communication can begin at all. Resonance is two clocks agreeing because they left the same factory. Its objective signature *predates any recognition event*: same-position slots have correlated receipt streams on shared world events with no communication whatsoever — measurable, falsifiable (P75), and cleanest in a channel-free population.

**Base: derived.** The recognition is mutual Pose plus Unify's identity check — Case-2 unknown-to-unknown matching run *across* ledgers, the Maxwell move socialized. Shape-only; open slots have no content to leak.

**Fiber: genuinely new — the ninth fiber primitive, `coordination-accrue`.** Every prior object in the algebra is owned by one ledger. The binding is owned by *neither web alone*: the algebra's first **communal object**, and the first operator whose object IS the boundary (Pose and Attest cross the interface; Bind inhabits it). It cannot carry lived receipts (local by definition) and transfers nothing. It is funded by the provenance class the speech-act inventory already used for declaratives and institutions but never gave a carrier: **COORDINATED** — receipts earned at the interface, dual-grounded in both parties' lived experience of the identification *working* (acting on "same slot" improving mutual prediction; the shared name keeping its promises). Admission is earned, never declared for free; conservativity is intact — the binding mints no funding in either ledger.

**Lifecycle, fully mortal:** a binding under divergent strain **differentiates** — the formal event behind "your ⟨water⟩ isn't quite my ⟨water⟩," which is most of what semantic negotiation is; a binding nobody coordinates through pays rent it cannot cover and archaizes — a dead word. Language is thereby negotiation *and* recognition, with the mix measurable: negotiation probes boundaries where structures genuinely differ (P64's interior); recognition Binds where they were always the same. Predicted stratification (to be numbered when the mechanism is built): bindings form earliest on trunk-adjacent slots, negotiation dominates on the canopy — conversation stratifies along the trunk/canopy axis, recognition at the roots, negotiation at the leaves.

**What a word finally is (T155 hook):** a funded Bind plus a public token. LC constructions that reference bindings rather than private slots are the moment meaning becomes public; P48's query tokens are what stabilizes into them.

**Fragment:** non-monotone (creates communal structure; differentiates). The mechanism build belongs after E1 (it needs the bus and per-organism ledgers); P75's observational precursor needs neither and runs now.

**One line for the ledger:** *transfer moves receipts between webs; resonance discovers that the world already wrote the same receipt twice.*

---

## 2. Derived Operations

Several operations that might appear primitive reduce to compositions of the above. Derivations clarify the basis and prevent redundancy.

### 2.1 Transfer (derived)

**Original framing:** Transfer carries `?X`'s connector geometry into a new domain, opening a corresponding slot there. Stated as a primitive, it appears irreducible.

**Derivation:** Transfer = Abstract(`?X`, domain-context) then Constrain(result, new-domain-context).

Abstract extracts the pure relational geometry that `?X` shares with its domain context — the sub-pattern that would survive re-embedding. Constrain re-specializes that geometry into the new domain by adding the new domain's structural constraints.

The result is a new slot in the new domain whose geometry is the extracted sub-pattern, re-specialized. This is exactly what Transfer was supposed to do — and it decomposes cleanly.

**Consequence:** Analogy is join-then-restrict on the subsumption lattice. The analogical insight is Abstract (find the shared geometry) followed by Constrain (re-embed it in the target domain). The "mysterious" leap of analogy is two funded operations on the lattice.

### 2.2 Closure (derived)

**Original framing:** Closure (`?X → K`) appeared as a terminal state requiring its own operation.

**Derivation:** Closure is Constrain iterated to a singleton feasible set. When Constrain has narrowed `?X`'s feasible set to a single candidate — when no further constraint can distinguish between any remaining resolutions — the slot closes. The K entry is the singleton that remains.

**Consequence:** "Outside-in resolution" is not a principle separate from the operator algebra. It is the operational definition of Constrain exhausting the feasible set. The slot closes when and only when Constrain has nothing left to remove.

### 2.3 Transpose (derived, with scheduling note)

**Original framing:** Transpose (`?X^T`) reverses the directionality of `?X`'s relational position — turning a cause-slot into an effect-slot, or vice versa — enabling direction-free inference through the constraint web.

**Derivation:** With undirected constraints, direction lives in the query, not the edge. Transpose is Posit with the clamped side swapped: clamp Y, read X's feasible set. The geometry of the reverse inference is produced by Posit run in the opposite query direction. No new operator is required.

**Status: derived.** Basis unchanged.

**Scheduling note — the cross-document consistency check:** Transpose is algebraically derived but *temporally* distinguished. Theorem 17 (anticausal execution) surfaces inside the algebra here: reverse-mode Posit — Transpose — can only run in the replay slot, not online. Its apparent primitivity was scheduling, not logic. The algebra and the execution scheduler must agree on this constraint: Transpose is a legal operator composition that carries a scheduling restriction, not an illegal operator and not a free operation.

---

### 2.4 Reopen (derived, 2026-08-10)

**Original framing:** Reopen appeared primitive — the only recovery from premature closure.

**Derivation:** Reopen = Retract applied at a closure: retract enough of the closure-defining constraint that the feasible set ceases to be a singleton. The feasible set is restored from lived receipts (the etymology), dependents are notified, and the contingent liability booked at closure (TS-6) is paid as the reverse cascade.

**Consequence:** Recovery from premature closure and revision of an open slot's boundary are the same operation at different lifecycle stages — one compensation primitive, licensed in both cases by failing receipts. The Einstein principle's recovery cost and the cost of withdrawing a single bad constraint are priced by the same fiber move.

---

## 2a. Economy Operators

Two operators were missing from the original table — not because they are rare, but because they are applied by the *economy* rather than by the organism. An algebra that omits the economy's own moves cannot state its conservation laws. Both are pure fiber moves with identity geometry: they touch only the ledger, leaving the geometry base unchanged.

### 2a.1 Archaize — `∅(?X)`

**What it does:** Evicts an unfunded slot from the active ledger. When an open variable's rent falls to zero — no incoming data feeds it, no other slot depends on it, no active predictions reference it — the economy removes it from the active slot space. The slot's receipt history is retained in the etymology ledger (the log is append-only); only its active status is revoked.

**Why it was missing:** Archaize is in the SOV document's lifecycle description but was absent from the operator table. This is a gap: Archaize is the economy's answer to Compose's one genuine risk.

**Compose's risk:** Compose is the endogenous slot generator — it opens a new slot whenever two existing slots stand in a structural relationship that implies a mediator. This is combinatorially explosive. Every related pair implies a mediator; every mediator can compose with adjacent slots to imply further mediators. Without pruning, the slot space grows without bound.

**Rent plus Archaize is the pruning mechanism.** Composed slots are born with structural funding (the necessity of the relationship that generated them) but must earn empirical funding through Fit receipts to survive. Slots that don't attract receipts pay rent they cannot cover and are archaized. The composed-slot survival rate is the diagnostic: high survival means the composer is too conservative — it's only opening slots that are already obviously funded. Near-zero survival means it's generating spam — opening slots that the environment never validates. Selective, low survival is the target.

**Sort:** Pure fiber move. Geometry: identity (the slot's geometry is not changed — it is simply removed from active status). Ledger: eviction event logged in the etymology ledger with the rent deficit that triggered it.

**Provenance:** The archaism event is a receipt: "slot `?X` evicted at time T, rent deficit R, etymology preserved." The history of what the slot was and why it was removed is permanently available for reconstruction if new evidence reopens the need.

---

### 2a.2 Anneal — `anneal(?X, t)`

**What it does:** Drifts the certainty weight on a slot entry over time under T57's annealing schedule — without any new receipt arriving. Certainty that was earned at time T₀ depreciates by time T unless refreshed by new Fit receipts. Entries that stop earning receipts become less certain; entries that continue earning remain certain.

**Why it was missing:** Anneal is T57's certainty drift, present throughout the corpus but unnamed as an operator. It is the economy's maintenance move — the background process that keeps certainty calibrated to current evidence rather than frozen at historical peak.

**The conservation law Anneal enables:** Without Anneal, a slot closed in the past with high certainty remains maximally certain forever, even if no new receipts have arrived and the world has changed. Anneal enforces that certainty is a *current* quantity, not a historical one. This is the algebra's analogue of radioactive decay — funded certainty decays toward the prior unless actively refreshed.

**Sort:** Pure fiber move. Geometry: identity. Ledger: certainty weight update on slot entries under the annealing schedule. No new structure is created; existing structure is reweighted.

**Relationship to Reopen:** Anneal is the smooth version of Reopen. Anneal gradually reduces certainty on entries whose receipts have stopped arriving; Reopen is the discrete event that fires when Annealed certainty has fallen far enough that channel 404 begins firing on the entry. Anneal makes the transition to Reopen graceful rather than abrupt — it is the warning before the revolution.

---

## 3. The Minimal Basis

The algebra is two-sorted. Minimality is argued per sort.

*Census note (2026-08-10): earlier drafts variously claimed eleven, thirteen, and fourteen named operators, and one list omitted Unify entirely — a counting error found by external review. "Eleven" was the geometry-primitive count migrating into the named-operator count. The corrected census, including the two operators registered 2026-08-10, is below.*

### Geometry base primitives (twelve)

`{Fit-base, Individuate-base (carve), Constrain-base, Retract-base (compensated widening), Compose-base, Exclude-base, Abstract-base, Differentiate-base, Posit-base, Pose-base, Attest-base, Quote-base}`

Unify-base is derived (mutual Constrain + identity check). Reopen-base is derived (Retract at a closure). Transpose is derived (reverse-mode Posit, scheduling-restricted). Transfer is derived (Abstract then Constrain). Closure is derived (Constrain to singleton).

### Fiber primitives (nine)

`{accrue, pool, partition, lien, discount-transfer, retract, evict, anneal, coordination-accrue}`

*coordination-accrue (2026-08-10, Bind's fiber): receipts of the COORDINATED provenance class accrue to a jointly-owned interface object, dual-grounded in both parties' lived coordination events. The one fiber move whose carrier belongs to no single ledger.*

Each is irreducible on the fiber — no geometry operation can move receipts, and no fiber operation can be produced by combining the others:

- **accrue** — Fit's fiber: a receipt enters the ledger
- **pool** — Unify's fiber: two receipt histories merge
- **partition** — Differentiate's fiber: one receipt history splits between children
- **lien** — Abstract's fiber: superordinate holds a claim on children's receipt sub-patterns
- **discount-transfer** — Attest's fiber: receipts cross ledger boundaries at calibrated discount
- **retract** — Reopen's fiber: closure event logged as retracted, dependents notified
- **evict** — Archaize's fiber: slot removed from active status, etymology preserved
- **anneal** — Anneal's fiber: certainty weights drift toward prior under T57's schedule

### Named operator set

Each named operator = one geometry move + one fiber move. **Sixteen named operators** — fourteen cognitive plus two economy — over twelve geometry primitives and nine fiber primitives:

`{Fit, Individuate, Constrain, Retract, Compose, Differentiate, Unify, Exclude, Abstract, Posit, Pose, Attest, Bind, Quote, Archaize, Anneal}`

*(Bind: base derived — mutual Pose + identity check; fiber primitive — coordination-accrue. Registered 2026-08-10; mechanism post-E1.)*

Derived: **Transfer, Closure, Transpose, Reopen, Suspend, Counterposit, Occlude, Enumerate** (the counterfactual pair — modal masking with support recomputation, and Posit ∘ Suspend; and the omission pair — the modal mask's datum-side target, and geometry-indexed hypothesis retrieval: the earned Bayesian enumeration; imagination_register.md §5a).

**Why the named basis is claimed minimal — and how the claim must be tested:** Each named operator is the unique pairing that the system requires; remove any one and a geometry or fiber capability disappears that no other pairing provides. The two economy operators (Archaize, Anneal) state the conservation laws without which the algebra cannot be closed. But minimality-and-completeness is a hypothesis, not a theorem: the universe of "required moves" was previously defined only by example, which is how two genuine gaps (Individuate, Retract) survived until external review. The completeness instrument is now pre-registered as **P67 — the state-transition coverage table**: every reachable lifecycle state × every required transition, each cell naming its operator or stated derivation. An empty cell falsifies completeness; a cell with two independent occupants challenges minimality. Individuate was the empty cell at (no-matching-slot, carve); Retract was the empty cell at (open-with-bad-constraint, revise).

---

## 3a. The Confluence Theorem: One Partition, Three Descriptions

The operator set has a partition that was built for scheduling reasons before it was understood to be a theorem. The theorem makes the partition precise and shows that three apparently separate distinctions are the same cut.

### The Partition

**Monotone operators** — operators that only narrow feasible sets or add constraints, never widen or restructure:

`{Fit, Constrain, Exclude, Abstract, Archaize, Anneal, Pose, Attest}`

These operators are monotone on the subsumption lattice: every application moves slots downward (toward more specific, more constrained, more funded) or holds them in place. They never widen a feasible set, merge receipt histories, split a slot, or retract a closure.

**Non-monotone operators** — operators that widen feasible sets, restructure, create structure, or merge histories:

`{Retract (and derived Reopen), Individuate, Bind, Differentiate, Unify, Compose, Quote, Posit}`

Retract widens a feasible set under testimony license (compensation restores previously-eliminated candidates); Reopen is its closure case. Individuate creates first-order structure from the unassigned pool. Differentiate restructures (one slot becomes two, with partitioned receipt histories). Unify merges receipt histories (pool is an irreducible fiber move that changes the topology of the receipt graph). Compose opens new slots (endogenous generation). Quote makes geometry into data (reflection changes the type of what is being operated on). Posit propagates hypothetically (imagined provenance, offline).

### The Confluence Conjecture *(downgraded from "theorem," 2026-08-10)*

**On the monotone fragment, propagation is conjectured confluent — pending construction of the actual lattice, and tested by P66.**

The intended argument: monotone propagation on a lattice of finite height has a unique least fixed point, reachable from any starting configuration under any update order — Knaster-Tarski (1955), extended to asynchronous schedules by the relaxation results (dataflow analysis, arc-consistency, chaotic iteration — Cousot and Cousot 1977).

What the argument still owes (external review, 2026-08-10): Knaster-Tarski requires monotone endomaps on one fixed complete lattice of finite height, and the "monotone fragment" as listed is not yet shown to live on one — Abstract adds nodes (changes the carrier), Archaize removes active status, Anneal depends on wall-clock time, Attest introduces cross-ledger state, and Fit can trigger closure. Until the common order is constructed and each operator proven monotone over it, schedule-independence is a **design property to be engineered and tested per operator, not a free consequence of the word "monotone."** The proof-in-practice: the first implementation's Anneal was schedule-*dependent* (repeated calls compounded the decay — found and fixed 2026-08-10), a live counterexample inside the allegedly confluent fragment. P66 remains the escrowed empirical test, and its falsification clause already said the honest thing: order-dependence in the monotone fragment means the lattice model is wrong somewhere and the formal spec requires surgery.

**On the non-monotone fragment, propagation is order-dependent.**

Reopen followed by Constrain produces a different result than Constrain followed by Reopen. Two concurrent Unify operations on overlapping slot pairs can produce different merged geometries depending on which fires first. Differentiate interleaved with receipt arrival produces different receipt partitions depending on timing. The fixed point is not unique; the path matters.

### The Triple Identity

The monotone/non-monotone split is identical to two other splits the architecture already maintains for independent reasons:

**Concurrent/serial split:** On the monotone fragment, any propagation schedule — parallel, asynchronous, out-of-order — reaches the same result. Concurrency is free and deterministic on this fragment. On the non-monotone fragment, propagation order is semantically significant. Serialization is required to get a determinate answer.

**Online/offline split:** The architecture already gates non-monotone operators to the replay slot — Reopen's reverse cascade, Theorem 17's anticausal execution, Posit's hypothetical propagation. This was a scheduling decision made for computational and causal reasons. The confluence theorem shows it was also the *correct* decision for semantic reasons: non-monotone operators produce order-dependent results and must be serialized to produce determinate outcomes.

**One partition, three descriptions:**

> monotone = concurrent = online  
> non-monotone = serial = offline (replay-gated)

The architecture built the replay gating before it had the theorem. The fourth time a document written for one level has resurfaced as a constraint at another, and the strongest instance: the scheduling policy and the semantic property are the same cut, derived from opposite directions, meeting in the middle.

### What Confluence Buys

**Concurrency is a theorem, not an assertion.** The claim that the constraint web propagates changes non-spatially and concurrently is not a metaphor or an aspiration on the monotone fragment — it is a proven property of monotone lattice systems. Run the ripple in any order, on any schedule, on any number of parallel processors, and the web converges to the same state.

**Localization is the default.** Most ripples don't propagate globally because the spectral gap of the dependency graph localizes them — the Laplacian's second eigenvalue determines how fast perturbations decay. Global cascades require hub-connectivity *and* non-monotone restructuring (usually Reopen at a hub). These are expensive, rare, and warranted: revolutions, not weather. The healthy design ensures most receipts produce local Constrain updates on the monotone fragment, with occasional hub closures propagating farther but still deterministically.

**The replay slot is the serialization mechanism for non-monotone moves.** Reopen, Differentiate, Unify-with-pooling, Compose, Quote, and Posit must be serialized because their results depend on order. The replay slot is where this serialization happens — the organism processes non-monotone updates in a controlled order, offline, without concurrent interference. The slot is not a performance optimization. It is the semantic enforcement mechanism for the non-monotone fragment.

### The Audited Compression

The corrected statement of what SOV is:

**SOV is a primitive-in-role for reasoning confluently — where monotonicity permits — about funded, non-physical, mostly-local changes of knowledge by knowledge.**

Each qualifier now has a theorem or a receipt behind it:
- *primitive-in-role*: prior to form commitments, implementable in ordinary mathematics
- *confluently*: Knaster-Tarski on the monotone fragment
- *where monotonicity permits*: the non-monotone fragment requires serialization — replay gating
- *funded*: receipt-funded existence; unknowns earn their place
- *non-physical*: the topology is funded adjacency, the metric is the D-metric, the geometry is real but not Euclidean
- *mostly-local*: spectral gap localizes most ripples; global cascades are priced by centrality
- *changes of knowledge by knowledge*: the primitive names the law of change, not the state — the ripple Jacobian, the calculus that was still missing from the custom mathematics stack

---

*Note on P-numbering: P53 is reserved for transfer holonomy (SOV Geometry doc, precedence). The predictions below run P52, P54–P57 to avoid collision.*

**P52 — The exclusion-cache signature.** An algebra with Exclude shows unification-test rates declining over lifetime on stationary structure while match accuracy holds. Ablating the cache produces the re-testing tax back. Falsification: the re-testing tax does not reappear on ablation — Exclude was not load-bearing and the basis shrinks by one.

**P54 — The Abstract-lattice signature.** In a system with both Unify and Abstract, the slot space develops measurable hierarchical structure — a subsumption lattice with statistically significant depth — that is absent in systems with Unify alone. The lattice's depth correlates with the system's generalization performance on transfer tasks. Falsification: Abstract-equipped systems show no latent hierarchy, or hierarchy does not predict transfer.

**P55 — Posit as experiment design.** Generator targeting based on Posit (maximize expected restructuring gain, priced by hypothetical propagation) out-distills random-slot targeting and learnable-surprise-only targeting on topology depth and downstream prediction improvement at matched episode count. Falsification: Posit-targeted generation matches random targeting — the pricing computation adds no value over heuristic selection.

**P56 — Reopen cost: Posit estimate beats connectivity.** The cost of reopening a closed entry — measured as cascade depth and downstream prediction degradation during retraction — is predicted better by the Posit-priced ripple estimate stored at closure time than by connectivity measured at retraction time. This is because the web has moved since closure; the stored estimate was priced against the web-at-closure, which is the relevant structure. Falsification: connectivity at retraction time predicts Reopen cost better than the stored Posit estimate — the web's movement since closure adds more predictive power than the estimate loses.

**P57 — Quote enables self-correction.** Systems with Quote show measurable improvement in detecting and recovering from systematic receipt-pattern anomalies — clustered 404s, underfunded slot regions, recurring misfit patterns — compared to systems without Quote at matched exposure. Falsification: Quote-equipped systems show no advantage in anomaly detection over systems that respond to 404s without opening meta-slots.

**P66 — Confluence test.** Asynchronous propagation restricted to monotone operators ({Fit, Constrain, Exclude, Abstract, Archaize, Anneal, Pose, Attest}) reaches identical fixed points under arbitrary update orders — determinism across schedules. Injecting non-monotone operators ({Reopen, Differentiate, Unify, Compose, Quote, Posit}) asynchronously produces order-dependent divergence. Falsification: monotone-only propagation shows order-dependence — the lattice model of the geometry sort is wrong somewhere and the formal spec requires surgery. Secondary falsification: non-monotone operators produce order-independent results when run asynchronously — in which case the replay gating is a performance decision only, not a semantic one, and the triple identity (monotone = concurrent = online) dissolves.

---

## 5. Open Questions

*Questions resolved since first draft are noted with their resolution.*

- **Does Unify reduce to mutual Constrain?** *Resolved: yes on the geometry base, no on the fiber.* Mutual Constrain plus identity check reproduces the feasibility structure exactly — but leaves separate receipt histories. Pool is an irreducible fiber primitive. Minimality is argued per sort.

- **What is the discount function for Attest?** *Resolved: two-stage.* Stage 1 prior: D-metric consequence-profile overlap. Stage 2 posterior: empirical survival rate of previously attested receipts under the importer's own billing, annealed under T57's schedule. The discount function is itself `?f` — an open variable receipted by every import's downstream fate.

- **Can Quote be applied recursively?** *Resolved: yes, economically bounded.* Arbitrary depth is syntactically legal. Each level is funded by next-order statistics of receipt arrival; statistical power decays geometrically with aggregation order, so funded reflection depth scales logarithmically in lifetime receipts. Pathological self-reference is prevented by provenance closure: the provenance graph is a well-founded DAG (every chain grounds out in Fit receipts), so no slot can appear in its own funding chain. Self-referential paradoxes aren't forbidden — they're broke. Tarski-style stratification from the ledger, not from a type system.

- **What is the interaction between Posit and Reopen?** *Resolved: the Posit estimate survives as a contingent liability.* Closure should record the Posit-priced ripple estimate as a contingent liability alongside the K-asset — double-entry epistemology. The Einstein principle becomes an accounting identity. P55 sharpens: Reopen cost should be predicted better by the stored Posit estimate at closure than by connectivity measured at retraction time, since the web has moved since closure.

- **Is there an operator for slot inheritance across generations?** *Resolved: Pose across the generational boundary.* The genome is a Pose with no possible Attest — geometry broadcast forward, receipts left behind. This is "topology inherits, entries don't" restated as an operator identity. Two corollaries: the Weismann barrier is shape-only computability enforced at the germline (acquired receipts cannot be serialized, so Lamarckism is an inadmissible operator); culture is the channel that adds Attest across generations (externalized logs are cross-generational receipt transfer — the legal Lamarckian channel). Heredity and language: the same operator pair at different timescales.

- **Where does Transpose sit?** *Resolved: derived, scheduling-restricted.* Transpose = reverse-mode Posit (clamp Y, read X). Basis stays at eleven geometry primitives. Scheduling restriction: Transpose can only run in the replay slot (Theorem 17, anticausal execution). Its apparent primitivity was scheduling, not logic.

- **What is the Anneal rate?** T57's annealing schedule is present in the corpus but not derived from first principles here. What determines how fast certainty decays — the receipt arrival rate? The slot's connectivity? The domain's volatility? The rate determines the balance between memory and adaptability and is not yet specified.

- **What is the composition law for the two sorts?** The algebra factors into geometry and fiber, and every named operator is a pair. But what are the legal pairings? Not every geometry primitive can pair with every fiber primitive — some combinations are inadmissible. The composition law (which geometry moves pair with which fiber moves, and what constraints govern the pairing) is the algebra's most important unstated rule.

---

## Appendix: Operator Summary Table

| Operator | Symbol | Geometry move | Fiber move | Fragment | Status | Boundary crossed |
|---|---|---|---|---|---|---|
| Fit | `?X ← data` | boundary test (+ unassigned-pool record on total miss) | accrue | **Monotone** | Primitive | world→ledger (inbound) |
| Individuate | `?new ← U` | carve from unassigned cluster | pool (re-home as opening receipts) | **Non-monotone** | Primitive (2026-08-10) | none |
| Constrain | `?X ⊃ C` | feasibility narrowing | (transitive) | **Monotone** | Primitive | none |
| Retract | `C ↓ ?X` | compensated widening | retract | **Non-monotone** | Primitive (2026-08-10) | none |
| Compose | `?X ∘ ?Y → ?Z` | relational product | structural accrue | **Non-monotone** | Primitive | none |
| Differentiate | `?X → ?X1 \| ?X2` | geometry split | partition | **Non-monotone** | Primitive | none |
| Unify | `?X ≡ ?Y` | identity check (derived) | pool (primitive) | **Non-monotone** | Base derived / fiber primitive | none |
| Exclude | `?X ≢ ?Y` | divergence cache | accrue (negative) | **Monotone** | Primitive | none |
| Abstract | `?X ⊔ ?Y → ?Z` | geometry intersection | lien | **Monotone** | Primitive | none |
| Posit | `do(?X ≐ v)` | hypothetical propagation | (IMAGINED — no ledger write) | **Non-monotone** | Primitive | ledger→imagination |
| Reopen | `K → ?X` | demotion | retract | **Non-monotone** | Derived (Retract at closure) | none |
| Pose | `Pose(?X)` | geometry serialization | (shape only — no receipt) | **Monotone** | Primitive | ledger→ledger (outbound geometry) |
| Attest | `Attest(?X, R)` | geometry match | discount-transfer | **Monotone** | Primitive | ledger→ledger (inbound receipts) |
| Bind | `?X_A ⋈ ?Y_B` | cross-ledger identity check (derived) | coordination-accrue (primitive) | **Non-monotone** | Base derived / fiber primitive (2026-08-10) | ledger↔ledger (the interface itself) |
| Quote | `⌜?X⌝` | geometry-as-data | meta-accrue | **Non-monotone** | Primitive | none |
| Archaize | `∅(?X)` | identity | evict | **Monotone** | Economy (primitive fiber) | none |
| Anneal | `anneal(?X, t)` | identity | anneal | **Monotone** | Economy (primitive fiber) | none |
| Transfer | Abstract; Constrain | extracted geometry; re-specialized | lien; (transitive) | **Monotone** | Derived | none |
| Closure | Constrain* | singleton collapse | (transitive) | **Monotone** | Derived | none |
| Transpose | `?X^T` | reverse-mode Posit | (IMAGINED) | **Non-monotone** | Derived, replay-only | ledger→imagination |
| Suspend | `Suspend(K)` | modal mask + support recomputation (AND/OR environments) | (IMAGINED) | **Non-monotone** | Derived, replay-only (2026-08-10) | ledger→imagination |
| Counterposit | `Posit(v′) ∘ Suspend(K)` | clamp within suspended context | (IMAGINED) | **Non-monotone** | Derived, replay-only (2026-08-10) | ledger→imagination |
| Occlude | `Occlude(datum)` | modal input mask (datum-side Suspend; C4: register-only, input cannot be un-lived) | (IMAGINED) | **Non-monotone** | Derived, replay-only (2026-08-10) | ledger→imagination |
| Enumerate | `Enumerate(gap)` | geometry-indexed retrieval + Posit elimination → earned hypothesis space | (IMAGINED) | **Non-monotone** | Derived, replay-only (2026-08-10) | ledger→imagination |

*Fragment column: Monotone operators propagate confluently under any update order (Knaster-Tarski). Non-monotone operators require serialization — they belong in the replay slot. The monotone/non-monotone split is identical to the concurrent/serial split and the online/offline split: one partition, three descriptions. See Section 3a.*

*Note on boundary crossings: Fit is the only inbound world→ledger crossing. Pose and Attest are the peer ledger↔ledger crossings (geometry out, receipts in). Posit and Transpose cross to the imagination domain (offline, IMAGINED provenance). The generator crosses ledger→world outbound — an effector, outside the algebra. These five surfaces are the firewall's perimeter.*

---

*First articulated in conversation, 2026-08-09. Companion documents: Structured Open Variables (SOV), SOV Entailments. Status: open — basis pending resolution of Unify and Transpose derivation questions.*

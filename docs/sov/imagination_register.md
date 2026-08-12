# The Imagination Register Algebra
### The Second Monad: Free Construction Behind a One-Way Wall

**Status:** Formal companion, articulated 2026-08-10 (external review raised the open question; resolved same day).
**Companion to:** SOV Formal Specification §5.5/§7, SOV Operator Algebra §1.8 (Posit, Suspend, Counterposit), Two-Sorted Core §4.
**Core claim:** The imagination register is not a feature of Posit — it is a second algebra over the same geometry sort, with a free fiber, a metabolic (not epistemic) economy, and exactly one lawful output type. Its relationship to the funded ledger is a one-way monad morphism whose only return path runs through the world. It is T154's fully-reversible rung, given its algebra.

---

## 1. Why the register needs its own algebra

The funded ledger's discipline — conservativity, grounding, the earning table — exists because funded structure makes claims on action. The register makes no claims on anything: nothing in it is owed to the world or to anyone. So its algebra inverts the ledger's central constraint:

> **In the ledger, construction must be earned. In the register, construction is free — and precisely because it is free, nothing constructed there can be kept.**

The Copernican requirement makes the free interior non-optional. Discovering that a rival structure *unifies what looks disparate* requires **performing the unification** — running Unify, Compose, Constrain, Abstract on hypothetical structure and measuring the gain. A register that can only query (return a ripple, a package) but cannot *build* cannot discover rivals; it can only price what already exists. Counterfactual reasoning, planning, hypothesis formation, and play all live here, and all of them are construction.

## 2. The algebra

**Carrier: context webs.** A context `H` is a cheap copy-on-write view of the funded web: spawned from the actual web (optionally through `Suspend(K)` masks and `Counterposit` clamps), mutated freely, discarded without trace or kept as a candidate. Contexts nest (a hypothetical within a hypothetical — the Quote tower's modal sibling), subject to budget.

**Operators: the geometry sort in full, with the trivial fiber.** Within `H`, every geometry move is admissible — boundary tests, constraint narrowing AND widening (no Law 6 inside: hypothesis space is not monotone), relational products, splits, merges, intersections, carving. No move writes a receipt; every structure carries IMAGINED provenance by construction rather than by tag discipline. The fiber is not merely empty — there is *no fiber to lift into*: the register is the geometry sort acting on itself.

**What is inherited from the actual web:** geometry (all of it), certainties *as read-only context inputs* (possibly suspended per the support-environment rule), and the provenance DAG *as a read-only oracle* (Suspend's suspension sets are computed from it — the register depends on the ledger; nothing crosses back).

**Identity with the reversible rung:** every register operation is trivially reversible because nothing is written — discard the context and the excursion closes. This is T151's engine at the cognitive level (closed loops in state space) and T154's rung 1 (imagination, fully reversible, C1-firewalled). The register's algebra IS the algebra of full reversibility: compensation-free, because compensation is only needed where something persists.

## 3. The economy: a second currency

Free in receipts is not free in cost. The register's scarcity is **metabolic**:

- a replay-slot budget (contexts run offline, serialized with the other non-monotone work);
- a live-context cap and a nesting-depth cap;
- per-context operation budgets (a hypothesis that cannot show gain within budget is discarded — imagination has its own archaism, priced in compute).

Two currencies, one wall: the ledger's currency is receipts (epistemic), the register's is budget (metabolic), and **the firewall is the non-convertibility of the two** — epistemic debts cannot be paid in metabolic coin, no matter how much imagining is done. This is why unlimited imagination is affordable (T149): the identity-risk of a hypothesis is zero because its epistemic price is zero; what it costs is only time in the replay slot.

## 4. The boundary, stated categorically

- **Inbound (funded → imagined): a monad morphism exists.** Forget the grading: any funded structure can be imagined; any K can be suspended; the actual web is always a valid context seed. Posit, Transpose, Suspend, Counterposit are this morphism's operational faces.
- **Outbound (imagined → funded): no morphism exists.** There is no internal operation that converts IMAGINED structure into funded structure — not laundering (blocked by conservativity), not gradual promotion, not confidence accumulation. The absence is the firewall.
- **The only return path is external.** Imagination reaches the ledger solely through the composite *register → generator directive → world action → lived consequence → Fit*. Reality is the only arrow from the register back to the ledger, and it is not an arrow of the algebra — the generator and the world are outside (Boundary Inventory §5.2). The Copernican arc is the unique path in the diagram, which is why it is not a workaround but the law.

**The exit rule (the register's entire output interface):** the register's lawful outputs are **generator directives** — discriminating observations (replay_through_context's product), experiment targets (Posit-priced closure candidates, P55), pose candidates (questions worth asking, including the social suspension: "what if we are both wrong?"). Structures never exit. Questions do.

## 5. What the register holds (contents inventory)

- **Transient contexts:** the working spaces of planning, counterfactuals, and hypothesis formation. Discarded by default.
- **Candidate rivals:** contexts retained (within budget) because their measured unification gain exceeds threshold — geocentrism's rival waiting for the phases of Venus. A retained rival is not a belief; it is a standing *reason to look*, i.e., a persistent generator directive.
- **Shared contexts (play):** contexts jointly held across organisms via frame-marked communication (T150's play tag; LC's play readout). The one funded thing about a shared context is the coordination receipt on the *frame* — the agreement to be in it together — never its contents. Games are institutions over imagined stakes: coordination-funded closures whose content receipts are constitutionally zero.
- **The imagination log:** the register's own append-only record (already implemented) — bookkeeping of what was imagined, never evidence of anything.

## 5a. The Omission Cycle — Occlude and Enumerate *(registered 2026-08-10, the user's arrival)*

**The move:** deliberate omission as a cognitive operator. Not waiting for an unknown to emerge from prediction failure — *creating* one by withholding information you possess. The full cycle: deliberate omission → shaped gap (connector geometry inherited from the funded neighborhood) → earned hypothesis space read off that geometry → Posit each candidate → ripple-consistency elimination against receipts → survivors as the funded enumeration → comparison against the withheld truth → the delta bills. Occlusion as a tool, not a limitation.

**Two derived operators, both register-side:**

- **`Occlude(datum-region)`** — Suspend's sibling on the datum axis. The modal mask has two targets: Suspend masks a funded *belief* ("what if I didn't know this?"); Occlude masks a lived *datum* ("what must this be, given everything else?"). **The C4 constraint, constitutional:** the organism cannot un-live its input — the datum's receipt stands in the append-only log; omission is strictly a register operation on a copy-on-write context. The resulting receipt is legal by the same logic that lets MCTS inform action: the *datum* funds the fit; the imagined machinery only sharpened the expectation it was tested against — nothing IMAGINED enters the ledger, the ledger just receives better bills.
- **`Enumerate(gap)`** — the bridge from geometry to probability. The gap's connector geometry runs against the web as a retrieval query (geometry-indexed — the eigen fingerprint doing the same work it does for T155's phrase store); candidates are pruned by Posit-ripple inconsistency with existing receipts; survivors are weighted by the receipt mass of their supporting matches. **The probability space is downstream from the geometry, not prior to it** — a Bayesian enumeration that is earned, not stipulated, and auditable all the way down (the voucher applies). Probability is the residue among geometric survivors, never the frame.

**The targeting policy — the fringe rule, with the Case-3 warning:** omit where the neighborhood is *constrained but unresolved*. Omission where the web knows perfectly is confirmatory (near-zero information); omission where nothing constrains yields a structureless void — no connector geometry, no enumeration derivable (Case 3). The gap "arrives pre-constrained" only if the neighborhood is funded, so the operator's value scales with edge density — the same replay-phase dependency as P76, and the same bootstrap shape: the web must learn structure serially before it can afford to play blind. Attention is this policy running continuously (T116): choosing what to process is choosing what to omit is choosing which enumerations get tested this instant.

**Correspondences (named, per the Grassmann discipline):**
- **Masked modeling** (BERT, MAE, CBOW) is this move with a *stipulated* enumeration — softmax over a predefined vocabulary, no candidate auditable. Its success confirms the delta-amplification mechanism (the Serialization Thesis); its ceiling is the unearned hypothesis space. The SOV version's earned enumeration is precisely the missing half (P77).
- **The testing effect** (retrieval practice beats re-reading): a flashcard is deliberate omission; re-reading manufactures zero deltas. Predicts the effect, its direction, and its boundary condition (no benefit without surrounding structure to derive from).
- **Amodal completion**: perception inferring occluded structure from connector geometry — the involuntary form of the operator.
- **Lineage inside the program**: the attribution instrument (`surprise_attribution.py`, Phase 3) built occlusion as an *experimenter's* tool; the omission cycle is that instrument internalized — the probe becomes an organ, the T153 signature of mechanizing hand-played roles.

**The flywheel:** omission → shaped gap → earned enumeration → sharper receipt → richer web → better-constrained next gap. This is the entailments' compounding loop at the perception timescale — the micro-cycle of the macro-cycle, and the first form of it the organism can crank by choice. Serialization generates unknowns in time; Occlude generates them at will.

**Predictions:** P76 (staged fit — with enumeration sharpness as the measured *mediator* of the edge-density margin) and P77 (earned vs. stipulated enumeration) — registry; experiment specs pinned in replay_phase_requirements.md.

## 6. Relation to the current implementation

`sov.py`'s Posit/Suspend/Counterposit/replay_through_context are **degenerate contexts**: one-shot queries that spawn, evaluate, and discard within a single call, returning packages. They are correct instances of the algebra but do not yet expose the persistent context-web workspace (imagined Compose/Unify with measured unification gain) that the Copernican requirement and the replay-phase build need. That workspace — spawn/build/evaluate/keep-or-discard over copy-on-write webs, budget-metered — is the register's implementation debt, scheduled with the replay-phase build (it shares the replay slot and the non-monotone scheduling).

## 7. The Counterfactual License *(registered 2026-08-10, the user's principle)*

**Completeness of the algebra is not an aesthetic concern — it is the precondition for rung 3 being meaningful at all.** A complete algebra with *enforced* conservation laws means the structure is what it appears to be: every node real, every edge earned, every independence relation preserved. Then counterfactuals are queries. Without it, they are guesses dressed as queries.

This inverts Pearl's own unsolved problem. The ladder tells you what is computable *given* a structural model and is silent on where a trustworthy model comes from — Cartwright's "no causes in, no causes out." Here the model carries its own audit: counterfactual validity stops being an unauditable premise and becomes a computable property of the ledger the query touches. And the license extends to the negative space, which drawn models can never claim: in this algebra **the absences are earned too** — Exclude receipts are funded disequalities, unearnable-so-far is a claim about the world, deletion requires testimony. A suspension set can only be complete over a web whose missing edges were *refused*, not merely never drawn.

**The voucher (implemented in `sov.py`):** every Suspend/Counterposit/replay result ships with a per-query certificate of what it rests on — lived vs. attested support mass (discount-weighted), declared AND formulas vs. defaulted multi-parent ORs (the known under-suppression hole, flagged), embedding-epoch currency of the surface, retracted receipts excluded, and the citation of the last conservation audit. "Can't fully vouch" is per-query metadata, not a global disclaimer: the answer and its warrant travel together.

**The open questions are therefore unvouched query classes, not loose ends:**
- **the composition law** licenses deep imagined derivations (multi-operator chains inside a context — the Copernican workspace); until resolved, long derivations may pass through inadmissible intermediate states, so the law is the workspace build's *safety spec*, to be resolved with or before it;
- **lift uniqueness** licenses cross-organism counterfactual agreement (the E1 stakes): uniqueness is the guarantee that the ledger is a function of the life, not of the accountant;
- **the register's remaining semantics** (gain functional, context composition) license nested and shared counterfactuals — hypothesis towers and play. Single-level suspensions are vouchable now; towers are runnable but uncertified.

*Reflexive note: the lab already runs this epistemology on itself — VOID and UNTESTED are unvouched queries refused a verdict, C20 pre-flight is the vouching procedure at experiment scale, and "falsifications must be earned" is the license applied to claims about what would have happened. The organism inherits the lab's constitution once more.*

## 8. Open questions

- **Context composition:** do contexts form their own category (morphisms = context refinements)? Nesting suggests yes; the budget bounds its depth in practice regardless.
- **The gain functional:** unification gain inside a context is measured against context structure — what exactly is the functional (edge reduction? description length over the context web? restructuring reach)? P55's targeting experiments will constrain it.
- **Shared-context consistency:** when two organisms hold "the same" context, what must actually match — full geometry, or only the frame plus the substructure under discussion? (The D-metric's answer — consequence-profile overlap — is the natural candidate.)
- **Budget policy:** fixed replay fraction vs demand-driven (gap-score-weighted) imagination time — likely another instance of the no-fixed-balance pattern.

---

*Articulated 2026-08-10. The register in one line: a place where everything may be built and nothing may be kept — connected to the world's ledger by exactly one door, and the door is the world.*

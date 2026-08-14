# T160 — Mathematical Formalization
## The Predictable Future: definitions, theorems, and the FEP bridge

*Working draft, 2026-08-13. Companion to THEORIES.md §L (T160 blocks i–viii)
and the registry rows P107–P111. House discipline applies: proved is marked
proved, sketched is marked sketched, conjectured carries a lien. The
Separation Theorem (docs/sov/separation_theorem.md) is used as a standing
formal result.*

---

## 1. Setup

Agent–environment as a controlled Markov process: states $s \in S$, actions
$a \in A$, transition kernel $T(s' \mid s, a)$, policy $\pi$. The agent holds
a model $\hat{T}$ (in the organism: the web — slots, closures, and their
receipts are the model's funded fragment).

**Trajectory plan.** $\tau = (G_0 \to G_1 \to \dots \to G_n)$: a sequence of
stage sets $G_i \subset S$, endpoint $G_n = D$ (the destination). Stages are
defined so that entry events are the conditioning boundaries (semi-Markov at
stage grain). Failure is absorbing.

---

## 2. Definitions

**D1 (Stage probability).**
$p_i(\pi) = P(\text{reach } G_i \text{ before failure} \mid \text{entered }
G_{i-1}, \pi)$.

**D2 (Arrival probability — the series law).**
$$P_\tau(\pi) = \prod_{i=1}^{n} p_i(\pi)$$
Valid for sequential stages with absorbing failure; re-entry/retry structures
fold into the $p_i$ of the induced chain. Log form (the natural
coordinates): the **path hazard**
$$H(\tau) = -\log P_\tau = \sum_i -\log p_i .$$

**D3 (Sustainable state).** A set $\Sigma \subset S$ with holding probability
$q_\Sigma(\pi) = P(s_{t+1} \in \Sigma \mid s_t \in \Sigma, \pi)$ under a
maintenance policy of cost $c(\pi)$. Expected residence $R = (1-q)^{-1}$
(geometric), diverging as $q \to 1$. *Sustainable* means $q \to 1$ at bounded
$c$; maintenance efficiency $\eta = [(1-q)\, c(\pi)]^{-1}$.

**D4 (Judgment error).** The agent holds $\hat{p}_i$. Under **passive
observation**, $\hat{p}_i$ estimates $P(B \mid A)$, which differs from the
interventional $P(B \mid do(A))$ by a confounding bias $b_i$; under
**manufacture** with $n_i$ interventional trials, the bias is removed and the
estimator is Bernoulli. The mean-squared judgment error decomposes:
$$\mathrm{MSE}(\hat{p}_i) = \underbrace{b_i^2}_{\text{confounding}} +
\underbrace{\frac{p_i (1-p_i)}{n_i}}_{\text{sampling}} .$$

**D5 (The T160 objective, stock and flow).** Let $U(\tau)$ be the agent's
uncertainty about arrival (variance of $\hat{P}_\tau$, delta method:
$\mathrm{var}(\log \hat{P}_\tau) \approx \sum_i \mathrm{var}(\hat p_i)/p_i^2$).
The drive is NOT the one-shot minimization of $H + \kappa U$ over a fixed
menu; it is a **production rate** subject to viability:
$$\max \; \frac{d}{dt}\Big[\text{completed arrivals weighted by } W(D)\Big]
\quad \text{s.t.} \quad s_t \in \Sigma_{\text{viable}},$$
where each candidate trajectory is priced by $H(\tau) + \kappa\, U(\tau)$ and
destinations by the worth function $W$ (D6). The generative clause of T160
lives in the $\frac{d}{dt}$: the object is the flow, not the stock.

**D6 (Destination worth — the corollary).**
$$W(D) = \alpha \cdot \mathrm{occ}(D) + \beta \cdot \mathrm{dis}(D) +
\gamma \cdot \Phi(D)$$
with $\mathrm{occ}(D)$ = occupiability: $\exists\, \Sigma \subseteq D$ with
$q_\Sigma \to 1$ at bounded cost (a destination must contain a sustainable
state); $\mathrm{dis}(D)$ = discharge: the hazard retired by arrival (the
trajectory's accumulated $-\log p$ settled — T158's settlement, in these
coordinates); $\Phi(D)$ = **onward fertility**: the trajectory capacity of
$D$, the value of the achievable set $\{\tau' : \tau' \text{ launches from }
D\}$ (an option value; see Open Problems for its exact measure).

---

## 3. Theorems

**Theorem 1 (Blocker-first is gradient ascent). PROVED.**
$\partial P_\tau / \partial p_i = P_\tau / p_i$, which is maximal exactly at
$\arg\min_i p_i$. Under stage-homogeneous improvement costs, optimal effort
allocation works the weakest stage. Moreover the uncertainty is dominated by
the same stages: $\mathrm{var}(\log \hat P_\tau) \approx \sum_i
\mathrm{var}(\hat p_i)/p_i^2$ — small $p_i$, large $\mathrm{var}(\hat p_i)$
dominate. A blocker (low $\hat p$, high variance) therefore dominates BOTH
the value gradient and the estimate's variance: it is attacked first on
either criterion. *Scrum's practice is derived twice.*

**Theorem 2 (The manufacture loop minimizes D4's decomposition). PROVED
(given do-calculus).** Subtraction, isolation, and randomization are the
three identification moves that send $b_i \to 0$ (randomization manufactures
independence when removal is impossible — the RCT identity
$P(B \mid do(A)) = P(B \mid A)$ under $A \perp C$); repetition sends the
sampling term $\to 0$ at rate $1/n_i$; replication **across backgrounds**
certifies that the identified $p_i$ is background-independent (the certificate
quantifies over contexts, so the trials must too). Block (iv) of T160 is
exactly the minimization schedule of D4.

**Theorem 3 (Sustainability is spectral). PROVED (standard).** For the
sub-stochastic kernel $T_\Sigma$ restricted to $\Sigma$, the Perron eigenvalue
$\rho(T_\Sigma)$ is the asymptotic holding rate; residence diverges as
$\rho \to 1$. Sustainability engineering is the control of a Perron
eigenvalue toward 1; maintenance is spectral repair after perturbation.
*(Program note: the eigen-coder reads spectral structure of the constraint
web — an organism could in principle SENSE $\rho$; this is the
orbit-stability receptor of T160 (vi) given a mathematical referent.)*

**Theorem 4 (The T158 bridge is a change of variables). PROVED at
identity level.** $H(\tau) = \sum_i -\log p_i$ is the accumulated surprisal
of the success path. A why that retires stage $i$'s failure mode raises
$p_i$, lowering $-\log p_i$: discharge IS hazard reduction, term by term.
The invoice on stage $i$ is the gap between its current $-\log p_i$ and its
achievable floor. Raising stage probabilities and discharging surprise are
one operation written in two coordinate systems.

**Theorem 5 (The FEP bridge). SKETCHED — the user's conjecture, stated
formally.** Active inference minimizes expected free energy
$$G(\pi) = \underbrace{-\mathbb{E}\big[\log P(o \in C \mid \pi)\big]}_{\text{pragmatic (risk)}}
\; - \; \underbrace{\mathrm{IG}(\pi)}_{\text{epistemic (information gain)}}$$
over policies, with the preference prior $C$ fixed by phenotype. Set
$C = $ "arrival at $D$" and factor the trajectory into stages. Then:
- the **pragmatic term** $= -\log P_\tau(\pi) = H(\tau)$ — the path hazard,
  T160's engineering term;
- the **epistemic term** $=$ expected reduction of $U(\tau)$ — T160's
  calibration term;
so $G(\pi; C{=}D) \cong H(\tau) + \kappa\, U(\tau)$: **T160's per-trajectory
price is the expected-free-energy decomposition under a manufactured goal
prior.** The Separation Theorem slots in exactly here: it proves the two
terms select different dimensions and no fixed $\kappa$ (its $\beta$) works
across environments — which is the formal reason T160 must carry TWO terms
and mix them contextually (Theorem 1 gives the mixing rule at stage grain:
work the term that dominates the blocker).

**Theorem 6 (Level separation — "not competing, different levels").
CONJECTURED with a precise statement; the lien.** Write the full variational
problem $\min G(\pi;\, T,\, C,\, \hat T)$. FEP optimizes the **first
argument only**: policies, given kernel, preferences, and model class. The
three T160 manufacture operations are precisely the OTHER arguments:
1. **Environment manipulation (iii)** edits $T$ — pour probability mass:
   a road is a kernel edit that sets some $p_i \to 1$ structurally;
2. **The manufacture loop (iv)** repairs $\hat T$ — interventional
   identification that passive inference cannot perform (do vs. see);
3. **Destination minting (vii)** manufactures $C$ itself, priced by $W(D)$.
Hence the claim: **FEP is the inner loop of T160.** T160 = the same
functional with every argument opened to optimization, under viability
constraint and a production-rate objective (D5). They cannot compete because
they do not share an optimization variable; they compose because they share
the functional. *Proof obligation: show the composed problem is well-posed
(the inner minimization is continuous in the outer variables) and that the
NESS/viability identification below holds — this is the part that "needs
some work."*

**Corollary 6.1 (FEP's existence claim = D3).** FEP's foundational object —
a system with an attracting set / non-equilibrium steady state — IS a
sustainable state in D3's sense: NESS density $\leftrightarrow$
quasi-stationary distribution of $T_\Sigma$ with $\rho \to 1$. FEP supplies
the viability substrate (the stock); T160 supplies directed trajectories and
the three manufacture levels (the flow). The dark room is dissolved at the
level FEP cannot see: it satisfies the stock condition ($q \approx 1$) and
scores zero on the flow (no arrivals, $\Phi \approx 0$, no discharge).

**Corollary 6.2 (Quest inflation — the corollary's stability condition).**
Destination minting is stable iff minted $W(D)$ is receipt-funded: realized
discharge and realized fertility must track their estimates
($|\hat W - W| $ bounded, the motivation-grain calibration term). Unpriced
minting ($\hat W$ unconstrained by receipts) grows the open-trajectory mass
without settlement — formally the same divergence as the conspiracy
signature: opened invoices per unit time exceeding discharge capacity. The
anti-wirehead guard is a calibration constraint at the $C$-manufacture level.

---

## 4. The level table

| Level | Optimizes | T160 block | FEP status | Human instance |
|---|---|---|---|---|
| 0 | viability (stay in $\Sigma$) | D3 / (i) | THE existence claim (NESS) | homeostasis |
| 1 | policy $\pi$ given $T, C, \hat T$ | riding | expected free energy | skill, navigation |
| 2a | model $\hat T$ (deconfound) | (iv) loop | absent (passive inference conflates do/see) | science |
| 2b | kernel $T$ (manipulate) | (iii) | absent (niche construction, informally noted in FEP lit) | roads, cities, law |
| 3 | preferences $C$ (mint destinations) | (vii) | fixed by phenotype | culture, religion, questions |

The conjecture in one line: **active inference is the Level-1 slice of a
Level-0-through-3 constructor theory, and T160 names Levels 2–3 and the
production-rate objective that spans them.**

---

## 4b. The constructive hierarchy: strict dominance and the Pearl functor
*(T160 block ix; the user's extension, same day.)*

**Dependency chain (strict, generic).** Minting requires manipulation
(a destination must be made occupiable and fertile — kernel work);
manipulation requires deconfounded models (you cannot design a persistent
kernel edit from confounded estimates); deconfounded models require policy
(interventions are executed actions) and viability (an experiment you do not
survive identifies nothing); policy requires viability (there must be an
optimizer). Hence the reachable-future classes nest:
$$\mathcal{F}_{L0} \subsetneq \mathcal{F}_{L1} \subsetneq \mathcal{F}_{L2a}
\subsetneq \mathcal{F}_{L2b} \subsetneq \mathcal{F}_{L3}$$

**Conjecture (Constructive Hierarchy Theorem — CHT's form, constructive
content).** The nesting is strict for GENERIC worlds; it collapses only on
degenerate ones, and the collapses are themselves classifiable: a fully
ballistic world (no manipulable kernel) collapses $L2b \to L2a$; a
one-attractor world collapses $L3 \to$ designation; a confound-free world
collapses $L2a \to L1$ (Pearl's own degenerate case). Empirical anchor:
P112 (ablation order in generic venues; predicted non-collapse failures in
the pre-registered degenerate venues).

**The Pearl functor (the interlock).** Each Pearl level is the epistemic
entry fee of the corresponding constructive level:

| Pearl (what you can know) | T160 (what you can build) | fee's necessity |
|---|---|---|
| L1 association — seeing | L1 riding known trajectories | in-distribution correlations price known paths — FEP's rung |
| L2 intervention — doing | L2a/2b deconfound, then edit the kernel | kernel edits are interventions with persistent effects |
| L3 counterfactuals — imagining | L3 minting destinations | "a place worth being that does not yet exist" IS a counterfactual valuation |

The relation is a functor from query classes to construction classes:
**T160's ladder is Pearl's ladder pushed through the effectors.** The
deepest consequence is the L3 row: destination minting runs on
counterfactual machinery BY MATHEMATICAL NECESSITY — the imagination
register is not an architectural convenience for the corollary's drive but
its formal prerequisite (and the anti-wirehead guard of Cor. 6.2 is then
recognizable as C2 applied at the top rung: counterfactual worth must be
receipt-funded on realization, or minting inflates).

**Positioning, stated plainly.** Pearl's hierarchy classifies epistemic
capacity; this one classifies constructive capacity — what an agent can
build, become, and arrive at. Epistemic capacity is instrumental: the
reason to climb Pearl's ladder is to climb this one. FEP, like classical
statistics, lives on one rung and is blind to the others (Thm 6). Pearl
explains how minds reason; T160 is a candidate for what minds are for.

---

## 5. Open problems (the "needs some work" ledger)

1. **DAG trajectories.** D2 assumes sequential stages; real plans branch and
   join. The series law generalizes to series-parallel reliability networks;
   blocker-first (Thm 1) needs restating on min-cuts: *the blocker of a DAG
   plan is the lowest-probability cut, not the lowest-probability stage.*
2. **The fertility measure.** $\Phi(D)$ as option value needs a committed
   definition: candidate — expected maximal production rate (D5) achievable
   from $D$, which makes $W$ recursive (a destination is worth what it lets
   you produce; Bellman-like; convergence conditions unproven).
3. **Well-posedness of the composed problem** (Thm 6's lien).
4. **Endogenous-$C$ dynamics.** Does destination minting under the
   calibration constraint (Cor. 6.2) converge to a stable portfolio, or
   oscillate? Conjecture: the three-clock economy applied at $C$-grain
   (wrong / orphaned / vacuous destinations) is the stabilizer.
5. **NESS identification rigor** (Cor. 6.1): quasi-stationarity vs. NESS
   density — measure-theoretic care needed; currently an identification at
   the level of objects, not a proof at the level of dynamics.
6. **Empirical anchors:** P107 (blocker-first as allocation law — Thm 1's
   organism test), P108 (Thm 2's), P109 (level 2b), P110 (ratchet gating —
   licensing by $H + \kappa U$ threshold), P111 (Cor. 6.2's inflation
   signature).

---

## 6. Program notes

- The Separation Theorem is the bridge's load-bearing prior result: it
  already proves, in this program's own venues, that the pragmatic and
  epistemic terms are non-fungible and context-mixed — Theorem 5 inherits
  its force from it, and the pair travels well together (TNB-relevant: the
  bridge positions FEP as the inner loop of a constructor-level theory,
  which is a conversation, not a confrontation).
- The organism's mechanisms, placed: closures raise $p_i$ (whys as hazard
  reduction, Thm 4); the manufacture loop is Thm 2 enacted; rethreshold
  keeps $\hat p$ calibrated under distribution change (C12 at the firing
  condition); the conversion organ's licensing criterion (T152) is a
  threshold on $H + \kappa U$ before a ratchet; the court's
  trajectory-criticality key (P107) is Thm 1's gradient on the epistemic
  critical path.

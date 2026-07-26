# Remaining Territory Items (30-36)

## 30. Endogenous Fitness Formalization

**Statement:** The token-receptor r_token is fitness-positive iff:

    E[delta_fitness | r_token active, pop_density >= d*] > alpha_{r_token}

where d* is the percolation threshold — the minimum population density at which enough conspecifics hold the receptor for coordination to pay off.

Below d*: holding r_token costs alpha but produces no coordination benefit. The receptor is metabolically wasteful and gets selected out.

Above d*: holding r_token enables coordination with sufficient conspecifics. The benefit exceeds the cost.

This is qualitatively different from every other receptor's fitness condition, which is:

    E[delta_fitness | r active, W has structure S] > alpha_r

For non-language receptors, fitness-positivity comes from the ENVIRONMENT (W has structure S). For language receptors, it comes from the POPULATION (enough others hold it). The source of selection pressure changed.

**Formal consequence:** The population-level topology R_pop = union_i R^(i) is no longer just a descriptive statistic. It's a CAUSAL variable — the presence of r_token in R_pop directly affects whether r_token is fitness-positive for any individual. The topology of the population feeds back into the selection pressure on individuals. This is the self-referential loop that makes language a phase transition rather than a gradual acquisition.

---

## 31. Trunk Minimization

**Question:** What is the minimum receptor set without which communication fails entirely?

**Candidate answer:** The 18 invariant receptors (present in all 80 generations of deep time, seed 42):
- categorical_compression, controllability, relational_observation (trunk)
- absence_observation, arousal_regulation, causal_association, compression_gain, probabilistic_causation, ratio_detection, selective_observation, self_model, structural_similarity (branch)
- categorical_perception, causal_chains, causal_rhythm, mental_model (canopy)
- planning, prediction (unlabeled)

**Test:** Ablate trunk receptors one at a time. After each ablation, run the cross-environment transfer experiment. Measure the common refinement (intersection topology). When the intersection drops below a critical size, communication fails — the organisms can no longer coordinate or learn from each other.

**Prediction:** Removing any of the 18 invariant receptors will shrink the intersection more than removing any non-invariant receptor. The invariant set IS the minimum common ground, or very close to it.

---

## 32. Cross-Cultural Canopy Variance

**Prediction:** Layer depth predicts cross-cultural variance. Trunk receptors are universal. Canopy receptors are biographical — they depend on evolutionary/cultural history.

**In ERTI:** Already confirmed by the cross-environment transfer experiment. 134 shared (mostly trunk + branch), 30 transfer-only and 27 naive-only (mostly canopy). The canopy diverges based on history.

**In humans:** Spelke's core knowledge (objects, agents, number, space) = trunk. Color terms, spatial frames (absolute vs relative), folk-biological categories = canopy, highly variable across cultures.

**The WEIRD prediction:** Psychology sampled one canopy (Western, Educated, Industrialized, Rich, Democratic) and reported it as trunk. The framework says which findings are at risk of non-replication: the deeper the layer, the more suspect the universality claim. Low-layer findings (object permanence, agent detection) should replicate cross-culturally. High-layer findings (specific cognitive biases, decision-making heuristics) may not.

---

## 33. Instrument Formalization

**Claim:** Instruments lower the layer required to hold a distinction by mapping deep structure onto shallow receptor input.

**Formally:** An instrument I transforms the environment:

    W' = I(W)

such that a distinction detectable only by a layer-L receptor in W becomes detectable by a layer-(L-k) receptor in W'. The instrument performs the detection; the organism only reads the output.

**Examples:**
- A telescope maps distant structure (requires deep spatial reasoning to detect) onto a visible disk (requires only low-layer visual discrimination)
- A thermometer maps temperature gradients onto a number on a dial
- Statistical notation maps distributional structure onto symbolic patterns

**Budget consequence:** Instruments shift alpha_r off the organism's metabolic budget onto the artifact. Cultural cognition isn't capped the way biological cognition is because the budget constraint applies to CARRIED receptors. Instruments externalize the detection, so the organism doesn't pay the metabolic cost.

This is why the human canopy is anomalously deep: not a larger B, but a mechanism for spending less of it per receptor held.

---

## 34. Specialization Theorem

**Statement:** Under finite acquisition time T_life and prerequisite chain cost c(r) = sum of alpha along the chain to r:

    max_depth(individual) = max { l(r) : c(r) <= T_life }

As the frontier advances, max c(r) grows. Eventually c(r) > T_life for the deepest receptors — no single organism can traverse the full chain in one lifetime.

At that point, the topology can only grow if different organisms acquire different chains. The population holds the union. Each individual holds a narrow deep spike.

    |union_i R^(i)| >> max_i |R^(i)|

**Evidence:** The 161-vs-119 result. Total unique receptors across 80 generations (the "population" across time) = 161. Maximum any single generation holds = 119. The gap of 42 is the specialization margin.

**Prediction:** As generation count increases, the gap should widen. With multi-agent populations (item 28), the gap should appear within a single generation — different organisms in the same population holding different deep spikes.

---

## 35. The Reflexive Term

**W_{t+1} = f(W_t, R_t, map(R_t))**

Publishing a map of the topology changes the topology being mapped. Naming a bias installs a bias-detector. A description of conflation IS a conflation-detector — reading about it gives you the receptor.

**Consequences:**
1. The mapping project cannot converge — each publication is a term in the dynamics
2. The deliverable is the mechanism (which doesn't expire), not the snapshot (which does)
3. The map should be organized by layer, not by domain — trunk entries are near-permanent, canopy entries carry a shelf-life

**This is already happening in the simulation.** The episode-level receptor `conflation_ep` was added because we detected conflation in the data and named it. The act of detecting and naming it gave the organism the receptor. The organism's Umwelt gained resolution because we published (coded) the distinction. W_{t+1} = f(W_t, R_t, our_code).

---

## 36. Mathematics vs Ideology

**Claim (T113):** Once population-conferred fitness frees the topology from environmental structure, two outcomes diverge:

**Path A (Mathematics):** The decoupled receptor family constructs distinctions that HAPPEN to correspond to environmental structure. Necessity_detection fires on patterns that hold regardless of perspective. When the decoupled structure reconnects to causal reality, it's called mathematics or science.

**Path B (Ideology):** The decoupled receptor family constructs distinctions that cohere internally and transmit efficiently but detect nothing in the environment's causal structure. The receptor fires reliably. It's fitness-positive (the population sustains it). But it doesn't reconnect.

**The framework says these are the same mechanism.** A population-sustained receptor that has decoupled from environmental structure. The only difference is downstream: does it eventually reconnect? Empiricism is the reconnection protocol — the practice of checking whether population-sustained beliefs correspond to structure the world actually contains.

**This is not a philosophical claim.** It's a structural prediction: in a multi-agent ERTI simulation with language and cultural transmission, both paths should appear. Some culturally-transmitted receptors will correspond to environmental structure (the organisms that hold them do better when tested). Some won't (the organisms do no better or worse — the receptor is fitness-neutral carried by population inertia). The ratio of reconnecting to non-reconnecting cultural receptors is an empirical measurement the simulation can make.

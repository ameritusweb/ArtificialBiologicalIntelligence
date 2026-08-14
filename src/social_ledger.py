"""Social ledger: the Pose/Attest bus between per-organism constraint webs.

E1 protocol (social_ledger_requirements.md): at each episode boundary,
1. resolve pending corroborations from new lived fits,
2. each organism poses its top-Q structured-gap slots (near-miss + 404 +
   starvation — the shape of its ignorance),
3. every other organism matches the posed geometry and attests its recent
   lived receipts; imports arrive at the two-stage discount.

Deterministic throughout — no RNG anywhere on the bus, so a zero-budget bus
(Q=0) leaves every web bit-identical to the ISOLATED arm. The bus writes
web state only; organisms never see it (endpoint firewall: attest may pull
centroids, never radius/certainty/fit_count).
"""

from collections import defaultdict

import numpy as np

# Pre-registered protocol constants (social_ledger_requirements.md §3)
POSE_Q = 3                 # slots posed per organism per episode boundary
MATCH_FLOOR = 0.3          # responder-side geometry-match floor
M_RECEIPTS = 8             # receipts exported per attest
C0_DISCOUNT = 0.5          # base cross-ledger discount
CORR_WINDOW = 400          # lived steps to corroborate an import
CORR_RADIUS_FACTOR = 2.0   # corroboration distance = factor x slot radius
CORR_RADIUS_FALLBACK = 1.0  # when the slot radius is still infinite
GAP_WINDOW = 1000          # lived-step window for the gap score


class PoseAttestBus:

    def __init__(self, blind_families=None, q=POSE_Q, curiosity_quota=0):
        """blind_families: {population_slot: set(family_ids)} — used only to
        LABEL imports as sighted/blind for the yield stat (the masks
        themselves are applied by the runner). q=0 makes the bus a no-op
        (the zero-budget identity arm).

        curiosity_quota (the POSE FLOOR, F16 impl. 8 — the floor family's
        third member): reserve this many of the q pose slots for the top
        STARVED slots (sv > 0 — never-funded holes). Default 0 preserves
        the E1 protocol exactly; E2 pre-registers a nonzero value.
        Without it, struggle signals (renewed every episode) always
        outrank blindness (which generates no evidence): 3,600/3,600 E1
        poses went to struggles."""
        self.blind_families = blind_families or {}
        self.q = q
        self.curiosity_quota = curiosity_quota

        # Earned Attest priors (F16 impl. 10): per-ordered-pair (i<-j)
        # discounts calibrated from measured co-fit — world-mediated
        # alignment replacing the stipulated C0. Empty -> C0 flat (E1
        # protocol). The cofit->prior mapping is an E2 pre-registration
        # decision; calibrate_priors implements the mechanism.
        self.pair_priors = {}

        # Pending corroborations:
        # {importer_slot, exporter_slot, slot_id, emb, import_step,
        #  receipts_len_at_import, sighted}
        self.pending = []

        # Aggregates
        self.imports = defaultdict(int)        # (importer, exporter) -> n
        self.corroborated = defaultdict(int)   # (importer, exporter) -> n
        self.expired = defaultdict(int)        # (importer, exporter) -> n
        self.blind_imports = defaultdict(int)  # importer -> n (coverage)
        self.blind_slots_touched = defaultdict(set)  # importer -> slot ids
        self.pose_log = []                     # P65 observational log
        self.gen_events = []                   # per-boundary summaries

    # ------------------------------------------------------------------
    def episode_boundary(self, webs, generation=-1, episode=-1):
        n_corr, n_exp = self._resolve_corroborations(webs)
        n_poses, n_imports = 0, 0
        if self.q > 0:
            n_poses, n_imports = self._pose_and_attest(
                webs, generation, episode)
        self.gen_events.append({
            'gen': generation, 'ep': episode,
            'poses': n_poses, 'imports': n_imports,
            'corroborated': n_corr, 'expired': n_exp,
        })

    # ------------------------------------------------------------------
    def _resolve_corroborations(self, webs):
        n_corr, n_exp = 0, 0
        still = []
        for p in self.pending:
            web = webs[p['importer']]
            slot = web.slots.get(p['slot_id'])
            if slot is None or slot.state not in ('open', 'closed'):
                n_exp += 1
                self._bill(webs, p, survived=False)
                continue
            outcome = None
            if p['sighted'] and p['emb'] is not None:
                radius = slot.geometry.radius
                if not np.isfinite(radius):
                    radius = CORR_RADIUS_FALLBACK
                tol = CORR_RADIUS_FACTOR * max(radius, 1e-3)
                for r in slot.ledger.receipts[p['receipts_len']:]:
                    if (r.kind == 'fit' and r.sign > 0
                            and r.provenance == 'LIVED'
                            and r.embedding is not None):
                        d = float(np.linalg.norm(
                            np.asarray(r.embedding, dtype=np.float64)
                            - p['emb']))
                        if d <= tol:
                            outcome = True
                            break
            if outcome:
                n_corr += 1
                self._bill(webs, p, survived=True)
            elif web._global_step > p['import_step'] + CORR_WINDOW:
                n_exp += 1
                if p['sighted']:
                    self._bill(webs, p, survived=False)
                # blind imports expire silently: uncorroborable by
                # construction, they must not poison the exporter's
                # reliability posterior.
            else:
                still.append(p)
        self.pending = still
        return n_corr, n_exp

    def _bill(self, webs, p, survived):
        key = (p['importer'], p['exporter'])
        if survived:
            self.corroborated[key] += 1
        else:
            self.expired[key] += 1
        if p['sighted']:
            webs[p['importer']].bill_attest_outcome(
                f"ORG_{p['exporter']}", survived)

    # ------------------------------------------------------------------
    def _pose_and_attest(self, webs, generation, episode):
        n_poses, n_imports = 0, 0
        slots_order = sorted(webs.keys())
        for i in slots_order:
            poser = webs[i]
            scored = []
            starved = []
            for sid, slot in poser.get_open_slots().items():
                g = poser.gap_score(sid, GAP_WINDOW)
                if g['total'] > 0:
                    scored.append((g['total'], sid, g))
                if g['sv'] > 0:
                    starved.append((g['sv'], sid, g))
            # Deterministic: gap desc, slot id asc as tie-break.
            scored.sort(key=lambda x: (-x[0], x[1]))
            selection = scored[:self.q]
            # The pose floor: reserve curiosity_quota slots for the top
            # starved holes ("a hole I cannot fill"), displacing the
            # lowest-ranked struggle poses. Deterministic; no RNG.
            if self.curiosity_quota > 0 and starved:
                starved.sort(key=lambda x: (-x[0], x[1]))
                selected_ids = {sid for _, sid, _ in selection}
                floor_picks = [s for s in starved
                               if s[1] not in selected_ids][
                                   :self.curiosity_quota]
                if floor_picks:
                    keep = max(0, self.q - len(floor_picks))
                    selection = selection[:keep] + [
                        (sv, sid, g) for sv, sid, g in floor_picks]
            for total, sid, g in selection:
                posed = poser.pose(sid)
                n_poses += 1
                sighted = sid not in self._blind_slot_ids(poser, i)
                self.pose_log.append({
                    'gen': generation, 'ep': episode, 'org': i,
                    'slot': sid, 'name': posed['name'],
                    'gap': g, 'sighted': sighted,
                })
                for j in slots_order:
                    if j == i:
                        continue
                    n_imports += self._respond(webs, poser, i, j, sid, posed)
        return n_poses, n_imports

    def _blind_slot_ids(self, web, org):
        """Slot ids whose active threshold families are all blind for org."""
        blind = self.blind_families.get(org, set())
        if not blind:
            return set()
        out = set()
        for sid, slot in web.slots.items():
            active = np.where(slot.geometry.family_thresholds > 0)[0]
            if len(active) and all(int(f) in blind for f in active):
                out.add(sid)
        return out

    def _respond(self, webs, poser, i, j, poser_slot_id, posed):
        responder = webs[j]
        posed_thresh = np.asarray(posed['family_thresholds'])

        # Responder-side match: best own slot against the posed geometry.
        best_sid, best_score = -1, 0.0
        for sid, slot in responder.get_active_slots().items():
            local = slot.geometry.family_thresholds
            overlap = float(np.minimum(posed_thresh, local).sum())
            total = float(np.maximum(posed_thresh, local).sum()) + 1e-8
            score = overlap / total
            if score > best_score or (score == best_score and sid < best_sid):
                best_sid, best_score = sid, score
        if best_sid < 0 or best_score < MATCH_FLOOR:
            return 0

        exported = responder.export_fit_receipts(best_sid, M_RECEIPTS)
        if not exported:
            return 0

        before = len(poser.slots[poser_slot_id].ledger.receipts)
        # Earned prior when calibrated (F16 impl. 10); C0 flat otherwise.
        prior = self.pair_priors.get((i, j), C0_DISCOUNT)
        transferred = poser.attest(
            poser_slot_id, posed, exported, discount=prior,
            exporter_id=f"ORG_{j}", centroid_pull=True)
        if not transferred:
            return 0

        sighted = poser_slot_id not in self._blind_slot_ids(poser, i)
        key = (i, j)
        self.imports[key] += len(transferred)
        if not sighted:
            self.blind_imports[i] += len(transferred)
            self.blind_slots_touched[i].add(poser_slot_id)
        for tr in transferred:
            self.pending.append({
                'importer': i, 'exporter': j,
                'slot_id': poser_slot_id,
                'emb': (None if tr.embedding is None
                        else np.asarray(tr.embedding, dtype=np.float64)),
                'import_step': poser._global_step,
                'receipts_len': before + len(transferred),
                'sighted': sighted,
            })
        return len(transferred)

    # ------------------------------------------------------------------
    def calibrate_priors(self, pair_cofit, shuffled_baseline,
                         floor=0.1, ceil=0.9):
        """Set per-pair Attest priors from MEASURED co-fit (F16 impl. 10):
        the stipulated C0 replaced by world-mediated alignment. Mapping
        (placeholder pending E2 pre-registration): signal fraction
        cofit / (cofit + baseline), clipped to [floor, ceil]. Symmetric:
        both orderings of a pair get the same prior. Deterministic."""
        for key, cf in pair_cofit.items():
            i, j = (int(x) for x in key.split('-'))
            denom = cf + max(shuffled_baseline, 1e-6)
            prior = min(ceil, max(floor, cf / denom if denom > 0 else floor))
            self.pair_priors[(i, j)] = prior
            self.pair_priors[(j, i)] = prior
        return dict(self.pair_priors)

    # ------------------------------------------------------------------
    def yield_matrix(self):
        """Per ordered pair: sighted corroboration yield (P64's endpoint)."""
        out = {}
        for key in set(list(self.corroborated) + list(self.expired)):
            resolved = self.corroborated[key] + self.expired[key]
            if resolved:
                out[f"{key[0]}<-{key[1]}"] = {
                    'corroborated': self.corroborated[key],
                    'resolved': resolved,
                    'yield': self.corroborated[key] / resolved,
                }
        return out

    def stats(self):
        return {
            'total_imports': int(sum(self.imports.values())),
            'imports_by_pair': {f"{k[0]}<-{k[1]}": v
                                for k, v in sorted(self.imports.items())},
            'yield_matrix': self.yield_matrix(),
            'pending': len(self.pending),
            'blind_imports': dict(self.blind_imports),
            'blind_coverage': {k: len(v) for k, v
                               in self.blind_slots_touched.items()},
            'poses': len(self.pose_log),
        }


# ---------------------------------------------------------------------------
# Mask construction (runner-level; the bus only labels sighted/blind)
# ---------------------------------------------------------------------------

def build_masks(pop, blind_w=8, stride=4):
    """Per-organism web-view masks.

    Returns (mask_indices, blind_families):
      mask_indices: {slot: np.int array of receptor indices to zero in the
                     web's view of receptor_channels}
      blind_families: {slot: set of blinded family ids}
    Organism i is web-blind to families {(i*stride + k) % 33 : k < blind_w};
    pairwise family overlap decreases with |i - j| — the divergence ladder.
    """
    from receptor_eigen_coder import FAMILY_GROUPS, NUM_FAMILIES
    mask_indices, blind_families = {}, {}
    for i in range(pop):
        fams = {(i * stride + k) % NUM_FAMILIES for k in range(blind_w)}
        idxs = []
        for f in fams:
            idxs.extend(FAMILY_GROUPS[f][1])
        mask_indices[i] = np.asarray(sorted(idxs), dtype=np.int64)
        blind_families[i] = fams
    return mask_indices, blind_families


class CoFitTracker:
    """P75 observational add-on: the resonance precursor.

    Measures per-generation co-fit correlation between geometry-matched
    slot pairs across organisms (inherited slot i in web A vs inherited
    slot i in web B — identical geometry by construction) against a
    deterministic shuffled-pair baseline (family f vs the next mutually
    sighted family). Records only; touches nothing; runs in ALL arms —
    the prediction's clean form lives in ISOLATED, where there is no
    channel and any matched > shuffled excess is world-mediated.

    Deterministic; no RNG; behavior-invisible.
    """

    def __init__(self, pop, steps_per_episode, blind_families=None):
        from receptor_eigen_coder import NUM_FAMILIES
        self.pop = pop
        self.steps = steps_per_episode
        self.nfam = NUM_FAMILIES
        self.blind = blind_families or {}
        self._episodes = []   # list of {oi: bool matrix (steps x nfam)}
        self._current = {}
        self._current_ep = -1

    def record(self, oi, ep, step, fired_slot_ids):
        if ep != self._current_ep:
            if self._current:
                self._episodes.append(self._current)
            self._current = {}
            self._current_ep = ep
        if oi not in self._current:
            self._current[oi] = np.zeros((self.steps, self.nfam), dtype=bool)
        if step < self.steps:
            for sid in fired_slot_ids:
                if 0 <= sid < self.nfam:   # inherited slots: sid == family
                    self._current[oi][step, sid] = True

    @staticmethod
    def _phi(a, b):
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def _sighted(self, oi):
        return [f for f in range(self.nfam)
                if f not in self.blind.get(oi, set())]

    def end_generation(self):
        if self._current:
            self._episodes.append(self._current)
            self._current = {}
            self._current_ep = -1
        matched, shuffled = [], []
        pair_sums = {}   # per-pair retention (Bind-prior calibration input)
        for epdata in self._episodes:
            orgs = sorted(epdata.keys())
            for a_i in range(len(orgs)):
                for b_i in range(a_i + 1, len(orgs)):
                    i, j = orgs[a_i], orgs[b_i]
                    both = [f for f in self._sighted(i)
                            if f in set(self._sighted(j))]
                    for k, f in enumerate(both):
                        r = self._phi(epdata[i][:, f].astype(float),
                                      epdata[j][:, f].astype(float))
                        if r is not None:
                            matched.append(r)
                            key = f"{i}-{j}"
                            s, n = pair_sums.get(key, (0.0, 0))
                            pair_sums[key] = (s + r, n + 1)
                        # Deterministic shuffle: f vs the NEXT mutually
                        # sighted family (a fixed derangement, no RNG).
                        g = both[(k + 1) % len(both)]
                        if g != f:
                            rs = self._phi(epdata[i][:, f].astype(float),
                                           epdata[j][:, g].astype(float))
                            if rs is not None:
                                shuffled.append(rs)
        self._episodes = []
        return {
            'matched_mean': (round(float(np.mean(matched)), 5)
                             if matched else None),
            'shuffled_mean': (round(float(np.mean(shuffled)), 5)
                              if shuffled else None),
            'n_matched_series': len(matched),
            'n_shuffled_series': len(shuffled),
            'pair_cofit': {k: round(s / n, 5)
                           for k, (s, n) in sorted(pair_sums.items())
                           if n > 0},
        }


def divergence_check(webs, blind_families):
    """Manipulation check: measured D-metric divergence (consequence-profile
    distance) must increase with mask distance |i - j|."""
    profiles = {i: w.consequence_profile() for i, w in webs.items()}
    rows = []
    keys = sorted(webs.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = keys[a], keys[b]
            d_measured = float(np.linalg.norm(profiles[i] - profiles[j]))
            d_mask = len(blind_families.get(i, set())
                         ^ blind_families.get(j, set()))
            rows.append({'pair': f"{i}-{j}", 'mask_dist': abs(i - j),
                         'family_div': d_mask, 'd_metric': d_measured})
    return rows

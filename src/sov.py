"""SOV Operator Algebra — the two-sorted algebra of structured open variables.

Implements the spec from docs/sov/: 15 named operators as (geometry, ledger)
pairs — 13 cognitive (Fit, Individuate, Constrain, Retract, Compose,
Differentiate, Unify, Exclude, Abstract, Posit, Pose, Attest, Quote) + 2
economy (Archaize, Anneal); derived: Transfer, Closure, Transpose, Reopen,
Suspend (modal mask + support recomputation — rationally closed,
hypothetically open), Counterposit = Posit ∘ Suspend (Pearl rung 3, with
abduction-by-replay from the append-only log). Receipts carry AND/OR
justification formulas: the ledger is a truth-maintenance engine (ATMS
with receipts, funding, and the imagination firewall).
6 conservation laws (Law 6 with the testimony-licensed compensation
exception), confluence partition (conjecture under P66).

Audit corrections (2026-08-10):
- Rent has a funding side: positive lived/attested receipts credit rent_balance
  (capped), so eviction is selection, not a fixed TTL.
- Slots keep a ring buffer of raw core-obs support samples; rebase(encoder)
  recomputes centroid/radius after each mental-model rebuild so the feasible
  set never mixes coordinates from unaligned encoder epochs.
- Fit receipts carry a float16 embedding stamped with the embed epoch, making
  Differentiate's partition exact (per-receipt, conservation-asserted).
- Differentiate splits along the principal axis of current-epoch receipt
  embeddings — deterministic, no global RNG.
- Closed slots stay in the Fit stream; a 404 window of systematic misfits
  triggers Reopen, which restores the feasible set from lived receipts.
- Compose/Abstract refuse unfunded parents (conservativity guard).
- Posit writes to a separate imagination register, never the funded etymology.
- Quote grounds only in Fit receipts.
- Attest discount is two-stage: geometry-match prior x per-exporter
  reliability posterior (bill_attest_outcome anneals the posterior).
- Receipt index makes provenance tracing O(1); the etymology has a single
  global event log so Law 2 counts one unit; Law 1 counts positive-sign Fit
  mass only; the Law 3 grounding sample runs whenever the check is called.
- Near-boundary misses accrue negative receipts (strided) so the channel
  that funds Differentiate / boundary revision is not structurally silent.
- Per-slot eigen fingerprint is encoded from the slot's own activation EMA,
  not overwritten with the current observation's global code.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from receptor_eigen_coder import FAMILY_GROUPS, NUM_FAMILIES

EMBED_DIM = 32
ANNEAL_HALFLIFE = 500
CLOSURE_RADIUS = 0.05
FIT_MATCH_THRESHOLD = 0.5
FIT_MIN_SCORE = 0.3
RENT_PER_EDGE = 0.001
COMPOSE_RENT_MULTIPLIER = 2.0
CERTAINTY_FLOOR = 0.01
CERTAINTY_CEILING = 0.99

# Rent economics: positive evidence pays rent (receipt-funded existence).
RENT_CREDIT_PER_FIT = 0.05
RENT_BALANCE_CAP = 1.0
RENT_EVICTION_FLOOR = -1.0

# Information-priced rent (2026-08-12; three receipts demanded it:
# F22 impl. 7 — rent funds contact, not information, making always-on
# mediators rent-immortal; F27/F28 — the court's sensitivity tracks
# informativeness while the web's funding doesn't; F30 impl. 4 — sharp
# expectations cannot buy influence). Rent credit is weighted by the
# slot's firing SELECTIVITY (1 - firing base rate): a slot that fires
# on everything carries no information per fit and earns nothing per
# fit. Mass and Law 1 are untouched — this prices SURVIVAL, not truth.
INFO_RATE_ALPHA = 0.02       # fire-rate EMA horizon ~50 evaluations
# Volume correction (same day, check-6's corollary applied to the first
# attempt): a per-fit DISCOUNT alone cannot starve a high-VOLUME slot —
# income (credit x fires x (1-p)) still swamps time-rent at thousands of
# fires per generation. Each fire therefore also COSTS: net-per-fire =
# credit x (1-p) - cost. Breakeven at p* = 1 - FIRE_COST_FRAC: slots
# firing above p* lose funding PER FIRE (the busier, the faster they
# starve); selective slots earn. Survival now tracks selectivity at any
# volume.
FIRE_COST_FRAC = 0.25        # breakeven fire rate p* = 0.75

# Expectation-as-evidence (2026-08-12, P76's transmission): confirmed
# stage-expectations update geometry at this fixed alpha — deliberately
# ABOVE the fit EMA's 1/n so predicted-and-confirmed evidence can move
# a mass-heavy ledger (the delta-accounting claim as mechanism).
EXPECT_ALPHA = 0.05

# Feasible-set support (raw core-obs ring buffer, re-embeddable on rebase).
SUPPORT_RING = 64

# Near-boundary misfit channel.
BOUNDARY_BAND = 0.8          # activation in [band*thr, thr) is a near miss
NEAR_MISS_STRIDE = 8         # record every k-th near miss per slot

# Closed-slot 404 window -> Reopen.
REOPEN_WINDOW = 12
REOPEN_FAIL_MIN = 8
FAIL_FACTOR = 3.0            # fail when dist > FAIL_FACTOR * closure radius

# Dormancy (F23, 2026-08-11): a closed K the world has stopped touching is
# ORPHANED — no fits, no falsification, invisible to contact-driven truth
# maintenance. It keeps its history but loses assertion rights until
# contact restores it. Assertion = funding + RECENCY OF CONTACT.
DORMANCY_WINDOW = 10000      # steps without contact before a K goes dormant
                             # (~2 generations at current session scale)

# Per-slot eigen fingerprint refresh period (in positive fits).
EIGEN_REFRESH = 32

# Threshold rebase (2026-08-13, F41 impl. 10's receipt — the extinction
# autopsy: 72/72 composed slots died SATURATION-class at a domain
# transition because family thresholds are ABSOLUTE in activation space
# while the activation distribution moved; C12 applied to the firing
# condition itself). OPT-IN: nothing changes unless the harness calls
# snapshot_activation_dist() / rethreshold(). The scout path
# (observe_activations) fills the ring without fits — look before you
# fire: perception without commitment at a domain boundary.
ACT_RING = 2048                # per-family activation history ring
RETHRESH_MIN_SAMPLES = 256     # both distributions must clear this
RETHRESH_Q_LO = 0.01           # quantile clip: preserves "above all
RETHRESH_Q_HI = 0.99           # observed" as top-band, not infinity
RETHRESH_FLOOR = 0.02          # a threshold may never reach zero
                               # (zero fires always — the vacuous death
                               # this mechanism exists to prevent)

# Unassigned-observation pool (the web-level 404 stream Individuate consumes).
UNASSIGNED_RING = 512          # bounded, oldest evicted (deterministic)
UNASSIGNED_MIN_ACTIVATION = 0.25  # dead air is not a distinction
INDIVIDUATE_MIN_CLUSTER = 8    # lived receipts required to carve a slot
INDIVIDUATE_SIMILARITY = 0.6   # profile cosine floor for cluster coherence
INDIVIDUATE_THRESH_FRAC = 0.8  # carved threshold = frac x mean activation
INDIVIDUATE_MIN_FAMILY = 0.1   # families below this mean stay out of the carve

# Pre-compute family sizes for normalized activation
_FAMILY_SIZES = np.array([max(len(indices), 1)
                          for _, indices in FAMILY_GROUPS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Sort G — Geometry
# ---------------------------------------------------------------------------

@dataclass
class ConnectorGeometry:
    family_thresholds: np.ndarray    # (NUM_FAMILIES,) — boundary pattern
    eigen_soft: np.ndarray           # (5,) — spectral fingerprint
    eigen_code: int                  # 5-bit integer
    neighbors: list                  # [(slot_id, edge_type), ...]
    centroid: np.ndarray             # (EMBED_DIM,) — feasible set center
    radius: float                    # feasible set radius (inf = maximally open)
    parent_slot_id: int = -1
    child_slot_ids: list = field(default_factory=list)
    support: list = field(default_factory=list)   # ring of raw core-obs
    activation_ema: np.ndarray = None             # slot-specific receptor EMA

    def copy(self):
        return ConnectorGeometry(
            family_thresholds=self.family_thresholds.copy(),
            eigen_soft=self.eigen_soft.copy(),
            eigen_code=self.eigen_code,
            neighbors=list(self.neighbors),
            centroid=self.centroid.copy(),
            radius=self.radius,
            parent_slot_id=self.parent_slot_id,
            child_slot_ids=list(self.child_slot_ids),
            support=list(self.support),
            activation_ema=(None if self.activation_ema is None
                            else self.activation_ema.copy()),
        )


# ---------------------------------------------------------------------------
# Sort L — Ledger
# ---------------------------------------------------------------------------

@dataclass
class Receipt:
    receipt_id: int
    kind: str              # fit, boundary, structural, negative, meta, lien,
                           # transfer, retraction, eviction, anneal
    source_operator: str   # one of the 13 named operators
    parent_receipt_ids: list
    log_offset: int        # into experience_log; -1 for non-Fit
    episode: int
    time_step: int
    slot_id: int
    channel_indices: list
    family_id: int
    magnitude: float
    sign: int              # +1 match, -1 mismatch, 0 neutral
    provenance: str        # LIVED, IMAGINED, ATTESTED, or COORDINATED
                           # (COORDINATED reserved for Bind: receipts on
                           # jointly-owned interface objects, dual-grounded
                           # in both parties' lived coordination events)
    source_ledger: str
    discount: float = 1.0
    surprise_token_idx: int = -1
    lien_fraction: float = 0.0
    created_at: int = 0
    embedding: np.ndarray = None   # float16 snapshot for exact partition
    embed_epoch: int = -1          # encoder epoch the embedding belongs to
    reward: float = 0.0            # lived reward at the fit (consequence profile)
    justification: str = 'OR'      # support semantics over parents:
                                   # 'OR' (any parent suffices) or 'AND'
                                   # (all required — e.g. Compose). Makes the
                                   # ledger a truth-maintenance engine, not
                                   # just an audit trail.


@dataclass
class EtymologyEvent:
    event_type: str
    step: int
    slot_ids: list
    receipt_ids: list
    detail: dict = field(default_factory=dict)


@dataclass
class SlotLedger:
    receipts: list = field(default_factory=list)
    certainty: float = 0.5
    mass: float = 0.0
    receipt_count: int = 0
    fit_count: int = 0
    last_fit_step: int = -1
    last_activity_step: int = -1
    rent_balance: float = 0.0
    rent_rate: float = 0.0
    rent_multiplier: float = 1.0
    etymology: list = field(default_factory=list)
    near_miss_seen: int = 0
    fail_window: list = field(default_factory=list)  # [(receipt_id, failed)]
    last_anneal_step: int = -1
    reopen_count: int = 0     # closure-churn: the de-conflation demand signal (F13)
    fire_rate: float = 0.5    # firing base-rate EMA (info-priced rent);
                              # 0.5 = uninformative prior


# ---------------------------------------------------------------------------
# Slot = (G, L)
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    slot_id: int
    name: str
    geometry: ConnectorGeometry
    ledger: SlotLedger
    state: str = 'open'          # open, closed, archaized, imagined
    created_at: int = 0
    closed_at: int = -1
    resolution: dict = None
    posit_liability: float = 0.0
    origin_operator: str = 'inherited'
    origin_family: str = ''
    dormant: bool = False        # closed K without recent contact (F23):
                                 # citable history, not citable truth


@dataclass
class ConstraintEdge:
    edge_id: int
    slot_a: int
    slot_b: int
    edge_type: str               # constraint, exclusion, subsumption, composition, lien
    receipt_ids: list = field(default_factory=list)
    strength: float = 1.0


# ---------------------------------------------------------------------------
# ConstraintWeb — the algebra
# ---------------------------------------------------------------------------

class ConstraintWeb:

    def __init__(self, eigen_coder=None, experience_log=None, debug_level=1,
                 ledger_id='COGNITIVE'):
        self.eigen_coder = eigen_coder
        self.experience_log = experience_log
        self.debug_level = debug_level
        self.ledger_id = ledger_id

        self.slots = {}
        self.edges = {}
        self.exclusions = set()

        # Single global etymology (Law 2 counts one unit per event) and a
        # separate imagination register (Posit never touches the ledger).
        self.etymology = []
        self.imagination_log = []

        # Unassigned-observation pool: lived observations that matched NO
        # slot's boundary. Pre-ledger records (not receipts — funding begins
        # when Individuate carves a slot; the opening receipts then ground
        # directly in these entries' log offsets). Bounded ring.
        self.unassigned_pool = []

        # Compensation registry: receipts retracted by Retract (history is
        # preserved — this marks them, it never deletes them).
        self.retracted_receipt_ids = set()

        # Attest posterior: per-exporter reliability, annealed by outcomes.
        self.attest_reliability = {}

        self._next_slot_id = 0
        self._next_edge_id = 0
        self._next_receipt_id = 0
        self._global_step = 0
        self._embed_epoch = 0

        # O(1) provenance tracing
        self._receipts_by_id = {}

        # Fresh-fit tightness (F16's sensitive endpoint): accumulates the
        # fit-time distance of each positive lived fit to its slot's
        # centroid SINCE the last pop — a per-window statistic the radius
        # EMA cannot supply once thousands of fits freeze it.
        self._fresh_dist_sum = 0.0
        self._fresh_dist_n = 0

        # Conservation law counters
        self._total_fit_mass = 0.0
        self._etymology_len = 0
        # Last conservation audit (cited by counterfactual vouchers)
        self._last_audit = {'step': -1, 'clean': None}

        # Per-family activation history ring (threshold rebase; geometry
        # sort only — no receipts, no ledger writes).
        self._fam_act_ring = np.zeros((ACT_RING, NUM_FAMILIES),
                                      dtype=np.float64)
        self._fam_act_n = 0

        self._op_counts = defaultdict(int)

    # --- slot management ---------------------------------------------------

    def create_slot(self, name, geometry, origin_operator='inherited',
                    origin_family=''):
        sid = self._next_slot_id
        self._next_slot_id += 1
        slot = Slot(
            slot_id=sid, name=name,
            geometry=geometry,
            ledger=SlotLedger(),
            created_at=self._global_step,
            origin_operator=origin_operator,
            origin_family=origin_family,
        )
        self.slots[sid] = slot
        self._recompute_rent(sid)
        self._log_event('created', [sid], [],
                        {'origin': origin_operator, 'family': origin_family})
        return sid

    def populate_from_families(self):
        for fam_idx, (fam_name, _) in enumerate(FAMILY_GROUPS):
            thresholds = np.zeros(NUM_FAMILIES, dtype=np.float64)
            thresholds[fam_idx] = FIT_MATCH_THRESHOLD
            geo = ConnectorGeometry(
                family_thresholds=thresholds,
                eigen_soft=np.zeros(5, dtype=np.float64),
                eigen_code=0,
                neighbors=[],
                centroid=np.zeros(EMBED_DIM, dtype=np.float64),
                radius=float('inf'),
            )
            self.create_slot(fam_name, geo, 'inherited', fam_name)

    def get_open_slots(self):
        return {sid: s for sid, s in self.slots.items() if s.state == 'open'}

    def get_active_slots(self):
        return {sid: s for sid, s in self.slots.items()
                if s.state in ('open', 'closed')}

    # --- internal helpers --------------------------------------------------

    def _alloc_receipt_id(self):
        rid = self._next_receipt_id
        self._next_receipt_id += 1
        return rid

    def _alloc_edge_id(self):
        eid = self._next_edge_id
        self._next_edge_id += 1
        return eid

    def _log_event(self, event_type, slot_ids, receipt_ids, detail=None):
        ev = EtymologyEvent(
            event_type=event_type, step=self._global_step,
            slot_ids=list(slot_ids), receipt_ids=list(receipt_ids),
            detail=detail or {},
        )
        self.etymology.append(ev)
        for sid in slot_ids:
            if sid in self.slots:
                self.slots[sid].ledger.etymology.append(ev)
        self._etymology_len = len(self.etymology)

    def _obs_to_family_activations(self, receptor_values):
        activations = np.zeros(NUM_FAMILIES, dtype=np.float64)
        for fam_idx, (_, indices) in enumerate(FAMILY_GROUPS):
            for idx in indices:
                if idx < len(receptor_values):
                    activations[fam_idx] += max(float(receptor_values[idx]), 0.0)
        # Normalize by family size so thresholds are on [0, 1] per-receptor scale
        activations /= _FAMILY_SIZES
        return activations

    def _boundary_test(self, geometry, family_activations):
        active = geometry.family_thresholds > 0
        if not active.any():
            return 0.0, []
        matched = (family_activations >= geometry.family_thresholds) & active
        score = float(matched.sum()) / float(active.sum())
        if score < FIT_MIN_SCORE:
            return 0.0, []
        return score, np.where(matched)[0].tolist()

    def _near_miss_test(self, geometry, family_activations):
        """Largest sub-threshold activation inside the boundary band."""
        active = geometry.family_thresholds > 0
        if not active.any():
            return 0.0, -1
        best, best_fam = 0.0, -1
        for fam in np.where(active)[0]:
            thr = geometry.family_thresholds[fam]
            act = family_activations[fam]
            if BOUNDARY_BAND * thr <= act < thr:
                closeness = act / thr
                if closeness > best:
                    best, best_fam = closeness, int(fam)
        return best, best_fam

    def _add_receipt_to_slot(self, slot, receipt):
        slot.ledger.receipts.append(receipt)
        slot.ledger.receipt_count += 1
        slot.ledger.last_activity_step = receipt.created_at
        self._receipts_by_id[receipt.receipt_id] = receipt
        if receipt.kind == 'fit':
            # Any fit is evidence CONTACT (anneal baseline), but fit_count
            # means POSITIVE lived support: it gates funding (Compose,
            # Abstract, Quote) and closure, where mismatches must not count.
            slot.ledger.last_fit_step = receipt.time_step
            if receipt.sign > 0:
                slot.ledger.fit_count += 1
                slot.ledger.mass += receipt.magnitude * receipt.discount
        # Receipt-funded existence: positive lived/attested evidence pays
        # rent — INFORMATION-PRICED (2026-08-12): credit is weighted by
        # firing selectivity, so an always-on slot earns ~nothing per fit
        # and cannot be rent-immortal (F22's mediators, fixed).
        if receipt.sign > 0 and receipt.kind in ('fit', 'transfer'):
            info_w = max(0.0, 1.0 - slot.ledger.fire_rate)
            net = (RENT_CREDIT_PER_FIT * receipt.magnitude
                   * receipt.discount * (info_w - FIRE_COST_FRAC))
            slot.ledger.rent_balance = min(
                RENT_BALANCE_CAP, slot.ledger.rent_balance + net)

    def _add_edge(self, slot_a, slot_b, edge_type, receipt_ids=None,
                  strength=1.0):
        eid = self._alloc_edge_id()
        edge = ConstraintEdge(
            edge_id=eid, slot_a=slot_a, slot_b=slot_b,
            edge_type=edge_type,
            receipt_ids=receipt_ids or [], strength=strength,
        )
        self.edges[eid] = edge
        if slot_a in self.slots:
            self.slots[slot_a].geometry.neighbors.append((slot_b, edge_type))
            self._recompute_rent(slot_a)
        if slot_b in self.slots:
            self.slots[slot_b].geometry.neighbors.append((slot_a, edge_type))
            self._recompute_rent(slot_b)
        return eid

    def _recompute_rent(self, slot_id):
        """Rent tracks the structural footprint: per-edge rate x the slot's
        multiplier x current connectivity. Recomputed on every topology
        change (creation, edge addition, edge transfer in Unify /
        Differentiate — all of which go through _add_edge)."""
        slot = self.slots[slot_id]
        n = max(len(slot.geometry.neighbors), 1)
        slot.ledger.rent_rate = RENT_PER_EDGE * slot.ledger.rent_multiplier * n

    def _get_neighbors(self, slot_id):
        return [sid for sid, _ in self.slots[slot_id].geometry.neighbors
                if sid in self.slots and self.slots[sid].state in ('open', 'closed')]

    def _connectivity(self, slot_id):
        return len(self._get_neighbors(slot_id))

    def _update_support(self, slot, support_obs):
        if support_obs is None:
            return
        ring = slot.geometry.support
        entry = np.asarray(support_obs, dtype=np.float32).copy()
        if len(ring) < SUPPORT_RING:
            ring.append(entry)
        else:
            ring[slot.ledger.fit_count % SUPPORT_RING] = entry

    def _update_eigen_fingerprint(self, slot, receptor_values):
        """Slot-specific spectral fingerprint from the slot's own activation
        EMA — not the current observation's global code."""
        rv = np.asarray(receptor_values, dtype=np.float64)
        ema = slot.geometry.activation_ema
        if ema is None or ema.shape != rv.shape:
            slot.geometry.activation_ema = rv.copy()
        else:
            slot.geometry.activation_ema = 0.9 * ema + 0.1 * rv
        if (self.eigen_coder is not None
                and slot.ledger.fit_count % EIGEN_REFRESH == 0):
            soft, code, _ = self.eigen_coder.encode(slot.geometry.activation_ema)
            slot.geometry.eigen_soft = soft
            slot.geometry.eigen_code = code

    # -----------------------------------------------------------------------
    # MONOTONE OPERATORS
    # -----------------------------------------------------------------------

    def fit(self, slot_id, receptor_values, embedding, obs_before, obs_after,
            reward, log_offset, episode, time_step,
            _family_activations=None, support_obs=None):
        self._op_counts['fit'] += 1
        slot = self.slots[slot_id]
        if slot.state not in ('open', 'closed'):
            return []

        if _family_activations is None:
            _family_activations = self._obs_to_family_activations(receptor_values)

        score, matched_families = self._boundary_test(
            slot.geometry, _family_activations)

        if score < 1e-6:
            # Near-boundary misfit: fund boundary revision / Differentiate
            # (strided to bound receipt volume).
            if slot.state == 'open':
                closeness, fam = self._near_miss_test(
                    slot.geometry, _family_activations)
                if fam >= 0:
                    slot.ledger.near_miss_seen += 1
                    if slot.ledger.near_miss_seen % NEAR_MISS_STRIDE == 0:
                        nm = Receipt(
                            receipt_id=self._alloc_receipt_id(),
                            kind='boundary',
                            source_operator='Fit',
                            parent_receipt_ids=[],
                            log_offset=log_offset,
                            episode=episode,
                            time_step=time_step,
                            slot_id=slot_id,
                            channel_indices=[fam],
                            family_id=fam,
                            magnitude=closeness,
                            sign=-1,
                            provenance='LIVED',
                            source_ledger='COGNITIVE',
                            created_at=self._global_step,
                        )
                        self._add_receipt_to_slot(slot, nm)
                        self._log_event('boundary_miss', [slot_id],
                                        [nm.receipt_id],
                                        {'closeness': closeness, 'family': fam})
                        return [nm]
            return []

        # Closed slots: the K keeps meeting the world. Consistency with the
        # resolution is the evidence stream that can trigger Reopen.
        if slot.state == 'closed':
            return self._fit_closed(slot, embedding, receptor_values,
                                    matched_families, score, log_offset,
                                    episode, time_step, support_obs)

        sign = 1 if score >= 0.5 else -1
        dominant_fam = matched_families[0] if matched_families else -1

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='fit',
            source_operator='Fit',
            parent_receipt_ids=[],
            log_offset=log_offset,
            episode=episode,
            time_step=time_step,
            slot_id=slot_id,
            channel_indices=matched_families,
            family_id=dominant_fam,
            magnitude=score,
            sign=sign,
            provenance='LIVED',
            source_ledger=self.ledger_id,
            created_at=self._global_step,
            embedding=(None if embedding is None
                       else np.asarray(embedding, dtype=np.float16)),
            embed_epoch=self._embed_epoch,
            reward=float(reward) if reward is not None else 0.0,
        )

        self._add_receipt_to_slot(slot, receipt)
        if sign > 0:
            self._total_fit_mass += score
            self._update_support(slot, support_obs)
            self._update_eigen_fingerprint(slot, receptor_values)

        # Update feasible set: pull centroid toward embedding
        if embedding is not None and slot.state == 'open' and sign > 0:
            n = slot.ledger.fit_count
            if n <= 1:
                slot.geometry.centroid = np.asarray(
                    embedding, dtype=np.float64).copy()
                slot.geometry.radius = 1.0
            else:
                alpha = 1.0 / n
                slot.geometry.centroid = (
                    slot.geometry.centroid * (1 - alpha) + embedding * alpha)
                dist = float(np.linalg.norm(embedding - slot.geometry.centroid))
                slot.geometry.radius = slot.geometry.radius * (1 - alpha) + dist * alpha
                self._fresh_dist_sum += dist
                self._fresh_dist_n += 1

        # Update certainty from boundary match quality
        alpha_c = max(0.05, 1.0 / max(slot.ledger.fit_count, 1))
        slot.ledger.certainty = float(np.clip(
            slot.ledger.certainty * (1 - alpha_c) + score * alpha_c,
            CERTAINTY_FLOOR, CERTAINTY_CEILING))

        self._log_event('fit', [slot_id], [receipt.receipt_id],
                        {'score': score, 'sign': sign})

        # Check closure
        self._check_closure(slot_id)

        if self.debug_level >= 1:
            assert self._total_fit_mass >= 0, "Law 1: negative Fit mass"

        return [receipt]

    def _fit_closed(self, slot, embedding, receptor_values, matched_families,
                    score, log_offset, episode, time_step, support_obs):
        """Fit on a K: accrue the receipt, track resolution consistency,
        Reopen on a systematic 404 pattern."""
        if embedding is None or slot.resolution is None:
            return []
        # Contact restores assertion rights (F23): the world reached the
        # K again, so truth maintenance has jurisdiction again.
        if slot.dormant:
            slot.dormant = False
            self._log_event('reawakened', [slot.slot_id], [],
                            {'idle': self._global_step
                                     - slot.ledger.last_fit_step})
        res_centroid = slot.resolution['centroid']
        dist = float(np.linalg.norm(
            np.asarray(embedding, dtype=np.float64) - res_centroid))
        tol = FAIL_FACTOR * max(slot.geometry.radius, 1e-3)
        failed = dist > tol
        # Fresh-fit tightness for a K measures the same thing it measures
        # for an open slot: the distance between lived reality and the
        # account — here, the FROZEN resolution (a committed account that
        # the world has moved away from shows up as growing distance).
        self._fresh_dist_sum += dist
        self._fresh_dist_n += 1

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='fit',
            source_operator='Fit',
            parent_receipt_ids=[],
            log_offset=log_offset,
            episode=episode,
            time_step=time_step,
            slot_id=slot.slot_id,
            channel_indices=matched_families,
            family_id=matched_families[0] if matched_families else -1,
            magnitude=score,
            sign=-1 if failed else 1,
            provenance='LIVED',
            source_ledger=self.ledger_id,
            created_at=self._global_step,
            embedding=np.asarray(embedding, dtype=np.float16),
            embed_epoch=self._embed_epoch,
        )
        self._add_receipt_to_slot(slot, receipt)
        if not failed:
            self._total_fit_mass += score
            self._update_support(slot, support_obs)
            # Confirming evidence refreshes certainty on the K.
            slot.ledger.certainty = float(np.clip(
                slot.ledger.certainty * 0.95 + score * 0.05,
                CERTAINTY_FLOOR, CERTAINTY_CEILING))

        window = slot.ledger.fail_window
        window.append((receipt.receipt_id, failed))
        if len(window) > REOPEN_WINDOW:
            window.pop(0)

        fails = [rid for rid, f in window if f]
        if len(window) >= REOPEN_WINDOW and len(fails) >= REOPEN_FAIL_MIN:
            self.reopen(slot.slot_id, fails)

        return [receipt]

    def fit_all(self, receptor_values, embedding, obs_before, obs_after,
                reward, log_offset, episode, time_step, support_obs=None):
        fa = self._obs_to_family_activations(receptor_values)
        self._fam_act_ring[self._fam_act_n % ACT_RING] = fa
        self._fam_act_n += 1
        results = []
        any_positive = False
        for sid in list(self.get_active_slots().keys()):
            receipts = self.fit(
                sid, receptor_values, embedding, obs_before, obs_after,
                reward, log_offset, episode, time_step,
                _family_activations=fa, support_obs=support_obs)
            fired = any(r.kind == 'fit' and r.sign > 0 for r in receipts)
            led = self.slots[sid].ledger
            led.fire_rate += INFO_RATE_ALPHA * ((1.0 if fired else 0.0)
                                                - led.fire_rate)
            if receipts:
                results.append((sid, receipts))
                if fired:
                    any_positive = True

        # Fit generalized (Individuate's supply): an observation with real
        # activation that matched NO slot is a web-level 404 — recorded in
        # the unassigned pool as a lived, log-grounded observation.
        if (not any_positive
                and float(fa.max()) >= UNASSIGNED_MIN_ACTIVATION):
            if len(self.unassigned_pool) >= UNASSIGNED_RING:
                self.unassigned_pool.pop(0)
            self.unassigned_pool.append({
                'profile': fa.copy(),
                'embedding': (None if embedding is None
                              else np.asarray(embedding, dtype=np.float16)),
                'embed_epoch': self._embed_epoch,
                'support_obs': (None if support_obs is None
                                else np.asarray(support_obs,
                                                dtype=np.float32).copy()),
                'log_offset': log_offset,
                'episode': episode,
                'time_step': time_step,
                'reward': float(reward) if reward is not None else 0.0,
                'magnitude': float(fa.max()),
            })
        return results

    def constrain(self, slot_id, constraint_receipt_ids):
        self._op_counts['constrain'] += 1
        slot = self.slots[slot_id]
        if slot.state != 'open':
            return []

        # Verify all constraining receipts are grounded
        for rid in constraint_receipt_ids:
            r = self._find_receipt(rid)
            if r is None or r.provenance == 'IMAGINED':
                return []

        old_radius = slot.geometry.radius

        # Each constraint narrows the feasible set
        for rid in constraint_receipt_ids:
            r = self._find_receipt(rid)
            if r is not None and r.slot_id != slot_id:
                source_slot = self.slots.get(r.slot_id)
                if source_slot is not None:
                    # Pull centroid toward the constraining slot's centroid
                    dist = float(np.linalg.norm(
                        slot.geometry.centroid - source_slot.geometry.centroid))
                    if dist > 0 and source_slot.ledger.certainty > 0.3:
                        pull = 0.1 * source_slot.ledger.certainty
                        slot.geometry.centroid += pull * (
                            source_slot.geometry.centroid - slot.geometry.centroid)
                        slot.geometry.radius *= (1.0 - 0.05 * source_slot.ledger.certainty)

        # Law 6: Constrain only narrows. radius_before is recorded so a
        # licensed Retract can compensate this exact narrowing later.
        slot.geometry.radius = min(slot.geometry.radius, old_radius)

        self._log_event('constrained', [slot_id], constraint_receipt_ids,
                        {'radius_before': float(old_radius)})
        self._check_closure(slot_id)
        return []

    def exclude(self, slot_a, slot_b, divergent_receipt_ids):
        self._op_counts['exclude'] += 1
        pair = (min(slot_a, slot_b), max(slot_a, slot_b))
        if pair in self.exclusions:
            return []

        self.exclusions.add(pair)

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='negative',
            source_operator='Exclude',
            parent_receipt_ids=list(divergent_receipt_ids),
            log_offset=-1,
            episode=-1, time_step=-1,
            slot_id=slot_a,
            channel_indices=[],
            family_id=-1,
            magnitude=1.0,
            sign=-1,
            provenance='LIVED',
            source_ledger='COGNITIVE',
            created_at=self._global_step,
        )

        self._add_receipt_to_slot(self.slots[slot_a], receipt)
        self._add_edge(slot_a, slot_b, 'exclusion',
                       [receipt.receipt_id])

        self._log_event('excluded', [slot_a, slot_b], [receipt.receipt_id])
        return [receipt]

    def abstract(self, slot_a, slot_b):
        self._op_counts['abstract'] += 1
        sa, sb = self.slots[slot_a], self.slots[slot_b]

        # Conservativity: the lien license comes from the children's funded
        # receipts. Abstract over unfunded slots is inadmissible.
        if sa.ledger.fit_count < 1 or sb.ledger.fit_count < 1:
            return -1, []

        # Geometry intersection: element-wise minimum of thresholds
        shared = np.minimum(sa.geometry.family_thresholds,
                            sb.geometry.family_thresholds)
        if shared.sum() < 1e-8:
            return -1, []

        geo = ConnectorGeometry(
            family_thresholds=shared,
            eigen_soft=(sa.geometry.eigen_soft + sb.geometry.eigen_soft) / 2.0,
            eigen_code=0,
            neighbors=[],
            centroid=(sa.geometry.centroid + sb.geometry.centroid) / 2.0,
            radius=max(sa.geometry.radius, sb.geometry.radius),
        )

        parent_id = self.create_slot(
            f"abstract({sa.name},{sb.name})", geo,
            origin_operator='Abstract')

        # Lien receipts: parent holds a claim, not a copy
        lien_receipts = []
        for child_id, child in [(slot_a, sa), (slot_b, sb)]:
            for r in child.ledger.receipts:
                if r.kind == 'fit' and r.sign > 0:
                    lr = Receipt(
                        receipt_id=self._alloc_receipt_id(),
                        kind='lien',
                        source_operator='Abstract',
                        parent_receipt_ids=[r.receipt_id],
                        log_offset=r.log_offset,
                        episode=r.episode, time_step=r.time_step,
                        slot_id=parent_id,
                        channel_indices=r.channel_indices,
                        family_id=r.family_id,
                        magnitude=r.magnitude * 0.5,
                        sign=r.sign,
                        provenance=r.provenance,
                        source_ledger=r.source_ledger,
                        lien_fraction=0.5,
                        created_at=self._global_step,
                    )
                    self._add_receipt_to_slot(self.slots[parent_id], lr)
                    lien_receipts.append(lr)

        # Lattice links
        sa.geometry.parent_slot_id = parent_id
        sb.geometry.parent_slot_id = parent_id
        self.slots[parent_id].geometry.child_slot_ids = [slot_a, slot_b]
        self._add_edge(parent_id, slot_a, 'lien',
                       [r.receipt_id for r in lien_receipts])
        self._add_edge(parent_id, slot_b, 'lien')

        self._log_event('abstracted', [parent_id, slot_a, slot_b],
                        [r.receipt_id for r in lien_receipts])
        return parent_id, lien_receipts

    def archaize(self, slot_id):
        self._op_counts['archaize'] += 1
        slot = self.slots[slot_id]
        if slot.state == 'archaized':
            return []

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='eviction',
            source_operator='Archaize',
            parent_receipt_ids=[],
            log_offset=-1, episode=-1, time_step=-1,
            slot_id=slot_id,
            channel_indices=[],
            family_id=-1,
            magnitude=0.0,
            sign=0,
            provenance='LIVED',
            source_ledger='COGNITIVE',
            created_at=self._global_step,
        )
        self._add_receipt_to_slot(slot, receipt)
        slot.state = 'archaized'

        self._log_event('archaized', [slot_id], [receipt.receipt_id],
                        {'rent_balance': slot.ledger.rent_balance})
        return [receipt]

    def anneal(self, slot_id, current_step):
        self._op_counts['anneal'] += 1
        slot = self.slots[slot_id]
        if slot.state not in ('open', 'closed'):
            return []

        # Schedule-independent decay: dt is the increment since the LAST
        # decay application (or last evidence contact), never the full span
        # since the last fit — otherwise repeated anneal calls compound
        # (annealing every 20 steps must equal annealing once at the end).
        last = slot.ledger.last_fit_step
        if last < 0:
            last = slot.created_at
        last = max(last, slot.ledger.last_anneal_step)

        dt = current_step - last
        if dt <= 0:
            return []

        decay = math.exp(-dt / ANNEAL_HALFLIFE)
        old_cert = slot.ledger.certainty
        slot.ledger.certainty = max(CERTAINTY_FLOOR, old_cert * decay)
        slot.ledger.last_anneal_step = current_step

        if abs(old_cert - slot.ledger.certainty) < 1e-6:
            return []

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='anneal',
            source_operator='Anneal',
            parent_receipt_ids=[],
            log_offset=-1, episode=-1, time_step=current_step,
            slot_id=slot_id,
            channel_indices=[],
            family_id=-1,
            magnitude=abs(old_cert - slot.ledger.certainty),
            sign=0,
            provenance='LIVED',
            source_ledger='COGNITIVE',
            created_at=self._global_step,
        )
        self._add_receipt_to_slot(slot, receipt)
        self._log_event('annealed', [slot_id], [receipt.receipt_id],
                        {'old': old_cert, 'new': slot.ledger.certainty, 'dt': dt})
        return [receipt]

    def anneal_all(self, current_step):
        self._global_step = current_step
        results = []
        for sid, slot in list(self.slots.items()):
            if slot.state in ('open', 'closed'):
                r = self.anneal(sid, current_step)
                if r:
                    results.extend(r)
                # Dormancy audit (F23): a K the world has stopped touching
                # loses assertion rights — demoted, not reopened; its
                # history stays citable, its truth does not.
                if (slot.state == 'closed' and not slot.dormant
                        and current_step - slot.ledger.last_fit_step
                        > DORMANCY_WINDOW):
                    slot.dormant = True
                    self._log_event('dormant', [sid], [],
                                    {'last_contact':
                                     slot.ledger.last_fit_step})
                # Rent is charged on open variables only (unknowns pay rent;
                # a K holds structure, not an option position).
                if slot.state == 'open':
                    slot.ledger.rent_balance -= slot.ledger.rent_rate
                    # Inherited trunk slots are constitutionally exempt from
                    # rent eviction: topology inherits, and deletion requires
                    # the world's testimony, not a rent shortfall. Their
                    # survival is therefore NOT evidence of earning.
                    if (slot.ledger.rent_balance < RENT_EVICTION_FLOOR
                            and slot.origin_operator != 'inherited'):
                        self.archaize(sid)
        return results

    def pose(self, slot_id):
        self._op_counts['pose'] += 1
        slot = self.slots[slot_id]
        geo = slot.geometry
        return {
            'slot_id': slot_id,
            'name': slot.name,
            'family_thresholds': geo.family_thresholds.tolist(),
            'eigen_soft': geo.eigen_soft.tolist(),
            'eigen_code': geo.eigen_code,
            'centroid': geo.centroid.tolist(),
            'radius': geo.radius,
            'state': slot.state,
        }

    def attest(self, slot_id, posed_geometry, external_receipts, discount,
               exporter_id=None, centroid_pull=False):
        self._op_counts['attest'] += 1
        slot = self.slots[slot_id]

        # Stage 1 prior: geometry match on the posed connector pattern.
        posed_thresh = np.array(posed_geometry['family_thresholds'])
        local_thresh = slot.geometry.family_thresholds
        overlap = float(np.minimum(posed_thresh, local_thresh).sum())
        total = float(np.maximum(posed_thresh, local_thresh).sum()) + 1e-8
        match_score = overlap / total

        if match_score < 0.1:
            return []

        # Admissibility (conservativity is the operator's job, not the
        # caller's): only positive lived-or-attested Fit evidence may cross
        # the ledger boundary. IMAGINED receipts are constitutionally barred
        # — importing one would launder imagination into funded structure
        # and the grounding trace treats ATTESTED as grounded.
        admissible = [r for r in external_receipts
                      if r.kind == 'fit' and r.sign > 0
                      and r.provenance in ('LIVED', 'ATTESTED')]
        if not admissible:
            return []

        # Stage 2 posterior: per-exporter reliability, annealed by the fate
        # of previously attested receipts (bill_attest_outcome).
        if exporter_id is None:
            exporter_id = admissible[0].source_ledger
        reliability = self.attest_reliability.get(exporter_id, 1.0)

        eff_discount = discount * match_score * reliability

        transferred = []
        for ext_r in admissible:
            tr = Receipt(
                receipt_id=self._alloc_receipt_id(),
                kind='transfer',
                source_operator='Attest',
                parent_receipt_ids=[ext_r.receipt_id],
                log_offset=ext_r.log_offset,
                episode=ext_r.episode,
                time_step=ext_r.time_step,
                slot_id=slot_id,
                channel_indices=ext_r.channel_indices,
                family_id=ext_r.family_id,
                magnitude=ext_r.magnitude * eff_discount,
                sign=ext_r.sign,
                provenance='ATTESTED',
                source_ledger=ext_r.source_ledger,
                discount=eff_discount,
                created_at=self._global_step,
                embedding=ext_r.embedding,
                embed_epoch=ext_r.embed_epoch,
                reward=ext_r.reward,
            )
            self._add_receipt_to_slot(slot, tr)
            transferred.append(tr)

            # Endpoint firewall (social-ledger E1): testimony may aim the
            # feasible set (centroid only, discounted). It must NEVER touch
            # radius, certainty, or fit_count — the world alone narrows.
            if (centroid_pull and slot.state == 'open'
                    and ext_r.embedding is not None):
                emb = np.asarray(ext_r.embedding, dtype=np.float64)
                if emb.shape == slot.geometry.centroid.shape:
                    slot.geometry.centroid += (
                        0.1 * eff_discount * (emb - slot.geometry.centroid))

        self._log_event('attested', [slot_id],
                        [r.receipt_id for r in transferred],
                        {'discount': discount, 'match': match_score,
                         'reliability': reliability,
                         'exporter': str(exporter_id)})
        return transferred

    def bill_attest_outcome(self, exporter_id, survived):
        """Anneal the per-exporter reliability posterior under the importer's
        own billing (TS-5 stage 2)."""
        r = self.attest_reliability.get(exporter_id, 1.0)
        self.attest_reliability[exporter_id] = (
            0.9 * r + 0.1 * (1.0 if survived else 0.0))
        return self.attest_reliability[exporter_id]

    def export_fit_receipts(self, slot_id, m=8):
        """The exporter's side of Attest: the last m positive lived fits."""
        slot = self.slots[slot_id]
        out = [r for r in slot.ledger.receipts
               if r.kind == 'fit' and r.sign > 0 and r.provenance == 'LIVED']
        return out[-m:]

    def gap_score(self, slot_id, window_steps=1000):
        """Structured-gap score for Pose selection (P65's statistic).

        gap = nm + ff + sv
          nm: near-miss boundary receipts in the window (shaped holes)
          ff: negative-sign fit receipts in the window (404s / misfits)
          sv: starvation — an open slot that has never fitted and is in
              rent deficit (the unfundable hole an organism can only
              fill by asking)
        Deterministic; draws no RNG.
        """
        slot = self.slots[slot_id]
        cutoff = self._global_step - window_steps
        nm = 0
        ff = 0
        for r in reversed(slot.ledger.receipts):
            if r.created_at < cutoff:
                break
            if r.kind == 'boundary':
                nm += 1
            elif r.kind == 'fit' and r.sign < 0:
                ff += 1
        sv = 0.0
        if slot.ledger.fit_count == 0 and slot.ledger.rent_balance < 0:
            sv = min(1.0, -slot.ledger.rent_balance)
        return {'nm': nm, 'ff': ff, 'sv': sv,
                'total': float(nm) + float(ff) + sv}

    def consequence_profile(self):
        """Per-family (lived-fit rate, mean reward at fit) — the operational
        D-metric coordinates for the divergence manipulation check."""
        counts = np.zeros(NUM_FAMILIES, dtype=np.float64)
        rewards = np.zeros(NUM_FAMILIES, dtype=np.float64)
        for slot in self.slots.values():
            for r in slot.ledger.receipts:
                if (r.kind == 'fit' and r.sign > 0
                        and r.provenance == 'LIVED' and r.family_id >= 0):
                    counts[r.family_id] += 1
                    rewards[r.family_id] += r.reward
        steps = max(self._global_step, 1)
        rate = counts / steps
        mean_r = np.divide(rewards, counts,
                           out=np.zeros_like(rewards), where=counts > 0)
        return np.concatenate([rate, mean_r])

    # -----------------------------------------------------------------------
    # NON-MONOTONE OPERATORS (replay-gated)
    # -----------------------------------------------------------------------

    def individuate(self, min_cluster=INDIVIDUATE_MIN_CLUSTER,
                    sim_threshold=INDIVIDUATE_SIMILARITY):
        """Slot genesis: carve a new first-order distinction from a coherent
        cluster of unassigned lived observations.

        Registered 2026-08-10. The lift is (carve, pool): the geometry move
        carves connector geometry from the cluster's activation profile; the
        fiber move re-homes the pool entries as opening receipts, each
        grounding DIRECTLY in its lived log offset. Compose/Abstract/Quote
        derive slots from existing slots; Individuate is the only operator
        that creates a slot from evidence no existing geometry could accept
        — the algebra-level name for receptor discovery (growing new eyes).

        Deterministic: seeds are tried in arrival order; no RNG.
        Conservativity: opening receipts carry source_operator='Individuate'
        (outside the Law 1 Fit-mass census) and ground via log_offset.
        """
        self._op_counts['individuate'] += 1
        pool = self.unassigned_pool
        if len(pool) < min_cluster:
            return -1, []

        members = None
        for si in range(len(pool)):
            seed = pool[si]['profile']
            ns = float(np.linalg.norm(seed))
            if ns < 1e-8:
                continue
            got = []
            for i, entry in enumerate(pool):
                pr = entry['profile']
                cos = float(np.dot(pr, seed)
                            / ((np.linalg.norm(pr) + 1e-8) * ns))
                if cos >= sim_threshold:
                    got.append(i)
            if len(got) >= min_cluster:
                members = got
                break
        if members is None:
            return -1, []

        entries = [pool[i] for i in members]
        mean_profile = np.mean([e['profile'] for e in entries], axis=0)
        thresholds = np.where(mean_profile >= INDIVIDUATE_MIN_FAMILY,
                              mean_profile * INDIVIDUATE_THRESH_FRAC, 0.0)
        if thresholds.sum() < 1e-8:
            return -1, []

        embs = [np.asarray(e['embedding'], dtype=np.float64)
                for e in entries
                if e['embedding'] is not None
                and e['embed_epoch'] == self._embed_epoch]
        if embs:
            centroid = np.mean(embs, axis=0)
            radius = float(np.mean(
                [np.linalg.norm(v - centroid) for v in embs]))
            radius = max(radius, CLOSURE_RADIUS)   # genesis never closes
        else:
            centroid = np.zeros(EMBED_DIM, dtype=np.float64)
            radius = float('inf')

        geo = ConnectorGeometry(
            family_thresholds=thresholds,
            eigen_soft=np.zeros(5, dtype=np.float64),
            eigen_code=0,
            neighbors=[],
            centroid=centroid,
            radius=radius,
            support=[e['support_obs'] for e in entries
                     if e['support_obs'] is not None][:SUPPORT_RING],
        )
        dominant = int(np.argmax(mean_profile))
        new_id = self.create_slot(
            f"individuated_f{dominant}", geo, origin_operator='Individuate')
        new_slot = self.slots[new_id]

        opening = []
        for e in entries:
            matched = np.where((thresholds > 0)
                               & (e['profile'] >= thresholds))[0].tolist()
            r = Receipt(
                receipt_id=self._alloc_receipt_id(),
                kind='fit',
                source_operator='Individuate',
                parent_receipt_ids=[],
                log_offset=e['log_offset'],
                episode=e['episode'],
                time_step=e['time_step'],
                slot_id=new_id,
                channel_indices=matched,
                family_id=matched[0] if matched else dominant,
                magnitude=e['magnitude'],
                sign=1,
                provenance='LIVED',
                source_ledger=self.ledger_id,
                created_at=self._global_step,
                embedding=e['embedding'],
                embed_epoch=e['embed_epoch'],
                reward=e['reward'],
            )
            self._add_receipt_to_slot(new_slot, r)
            opening.append(r)

        # Consume the members (event-sourced: the opening receipts ARE the
        # record; the pool is pre-ledger scratch).
        self.unassigned_pool = [p for i, p in enumerate(pool)
                                if i not in set(members)]

        self._log_event('individuated', [new_id],
                        [r.receipt_id for r in opening],
                        {'cluster_size': len(entries),
                         'dominant_family': dominant,
                         'pool_remaining': len(self.unassigned_pool)})
        return new_id, opening

    def retract(self, slot_id, failing_receipt_ids, target='constraint',
                exporter_id=None):
        """Generalized compensation — the Law 6 exception.

        Registered 2026-08-10. Feasibility narrows monotonically EXCEPT
        under logged compensation events, and every compensation must be
        licensed by failing receipts: the world's testimony, never taste.
        Targets:
          'constraint'  — widen an OPEN slot back to its most recent
                          pre-constraint feasible set (compensation, not
                          inverse: the constrained event stays logged)
          'attestation' — revoke imported receipts from one exporter on
                          this slot: rent credit compensated, reliability
                          billed down, receipts marked retracted (kept)
          'closure'     — the derived Reopen case (delegates to reopen())
        """
        self._op_counts['retract'] += 1
        if not failing_receipt_ids:
            return []
        for rid in failing_receipt_ids:
            r = self._find_receipt(rid)
            if r is None or r.provenance == 'IMAGINED':
                return []

        slot = self.slots[slot_id]

        if target == 'closure':
            return self.reopen(slot_id, failing_receipt_ids)

        if target == 'constraint':
            if slot.state != 'open':
                return []
            radius_before = None
            for ev in reversed(slot.ledger.etymology):
                if (ev.event_type == 'constrained'
                        and 'radius_before' in ev.detail):
                    radius_before = ev.detail['radius_before']
                    break
            if radius_before is None:
                return []
            receipt = Receipt(
                receipt_id=self._alloc_receipt_id(),
                kind='retraction',
                source_operator='Retract',
                parent_receipt_ids=list(failing_receipt_ids),
                log_offset=-1, episode=-1, time_step=-1,
                slot_id=slot_id,
                channel_indices=[], family_id=-1,
                magnitude=float(radius_before),
                sign=0,
                provenance='LIVED',
                source_ledger=self.ledger_id,
                created_at=self._global_step,
            )
            self._add_receipt_to_slot(slot, receipt)
            slot.geometry.radius = max(slot.geometry.radius,
                                       float(radius_before))
            self._log_event('retracted_constraint', [slot_id],
                            [receipt.receipt_id],
                            {'restored_radius': float(radius_before)})
            return [receipt]

        if target == 'attestation':
            if exporter_id is None:
                return []
            revoked = []
            credit = 0.0
            for r in slot.ledger.receipts:
                if (r.kind == 'transfer' and r.provenance == 'ATTESTED'
                        and r.receipt_id not in self.retracted_receipt_ids
                        and r.source_ledger == exporter_id):
                    self.retracted_receipt_ids.add(r.receipt_id)
                    revoked.append(r.receipt_id)
                    if r.sign > 0:
                        credit += RENT_CREDIT_PER_FIT * r.magnitude
            if not revoked:
                return []
            slot.ledger.rent_balance -= credit
            self.bill_attest_outcome(exporter_id, survived=False)
            receipt = Receipt(
                receipt_id=self._alloc_receipt_id(),
                kind='retraction',
                source_operator='Retract',
                parent_receipt_ids=list(failing_receipt_ids),
                log_offset=-1, episode=-1, time_step=-1,
                slot_id=slot_id,
                channel_indices=[], family_id=-1,
                magnitude=float(len(revoked)),
                sign=0,
                provenance='LIVED',
                source_ledger=self.ledger_id,
                created_at=self._global_step,
            )
            self._add_receipt_to_slot(slot, receipt)
            self._log_event('retracted_attestation', [slot_id],
                            [receipt.receipt_id],
                            {'exporter': str(exporter_id),
                             'revoked': revoked,
                             'rent_compensated': credit})
            return [receipt]

        return []

    def compose(self, slot_a, slot_b, relation_type='mediates'):
        self._op_counts['compose'] += 1
        sa, sb = self.slots[slot_a], self.slots[slot_b]

        # Conservativity: the structural receipt is funded by the parents'
        # receipt histories. Composing unfunded slots is inadmissible.
        if sa.ledger.fit_count < 1 or sb.ledger.fit_count < 1:
            return -1, []

        # Relational product: combine family thresholds
        combined = (sa.geometry.family_thresholds +
                    sb.geometry.family_thresholds) / 2.0

        geo = ConnectorGeometry(
            family_thresholds=combined,
            eigen_soft=(sa.geometry.eigen_soft + sb.geometry.eigen_soft) / 2.0,
            eigen_code=0,
            neighbors=[],
            centroid=(sa.geometry.centroid + sb.geometry.centroid) / 2.0,
            radius=float('inf'),
        )

        new_id = self.create_slot(
            f"compose({sa.name},{sb.name})", geo,
            origin_operator='Compose')
        new_slot = self.slots[new_id]
        new_slot.ledger.rent_multiplier = COMPOSE_RENT_MULTIPLIER * 2
        self._recompute_rent(new_id)

        # Structural accrue: funded by the parents' POSITIVE lived receipts
        # only — no fallback to mismatch evidence. ONE representative per
        # side, so the flat 'AND' justification expresses the true formula:
        # (support from A) AND (support from B).
        pa = [r.receipt_id for r in sa.ledger.receipts
              if r.kind == 'fit' and r.sign > 0][:1]
        pb = [r.receipt_id for r in sb.ledger.receipts
              if r.kind == 'fit' and r.sign > 0][:1]
        if not pa or not pb:
            return -1, []
        parent_receipt_ids = pa + pb

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='structural',
            source_operator='Compose',
            parent_receipt_ids=parent_receipt_ids,
            log_offset=-1, episode=-1, time_step=-1,
            slot_id=new_id,
            channel_indices=[],
            family_id=-1,
            magnitude=0.1,
            sign=0,
            provenance='LIVED',
            source_ledger='COGNITIVE',
            created_at=self._global_step,
            justification='AND',   # a mediator needs BOTH parents' funding
        )
        self._add_receipt_to_slot(new_slot, receipt)

        self._add_edge(slot_a, new_id, 'composition', [receipt.receipt_id])
        self._add_edge(slot_b, new_id, 'composition', [receipt.receipt_id])

        self._log_event('composed', [new_id, slot_a, slot_b],
                        [receipt.receipt_id], {'relation': relation_type})
        return new_id, [receipt]

    def _principal_axis(self, embeddings):
        """Deterministic top principal component via power iteration."""
        X = np.asarray(embeddings, dtype=np.float64)
        mean = X.mean(axis=0)
        Xc = X - mean
        v = Xc[0].copy()
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            v = np.ones(X.shape[1], dtype=np.float64)
            norm = np.linalg.norm(v)
        v /= norm
        for _ in range(20):
            w = Xc.T @ (Xc @ v)
            n = np.linalg.norm(w)
            if n < 1e-12:
                break
            v = w / n
        spread = float(np.std(Xc @ v))
        return mean, v, spread

    def differentiate(self, slot_id, split_criterion=None):
        self._op_counts['differentiate'] += 1
        slot = self.slots[slot_id]
        if slot.state != 'open':
            return -1, -1, []

        geo_a = slot.geometry.copy()
        geo_b = slot.geometry.copy()
        geo_a.neighbors = []
        geo_b.neighbors = []
        geo_a.support = []
        geo_b.support = []

        axis = None
        axis_mean = None
        if isinstance(split_criterion, dict) and 'split_family' in split_criterion:
            fam = split_criterion['split_family']
            if fam < NUM_FAMILIES:
                geo_a.family_thresholds[fam] *= 1.5
                geo_b.family_thresholds[fam] *= 0.5
        else:
            # Deterministic geometry split along the principal axis of the
            # current-epoch receipt embeddings — the two pulls that stressed
            # the boundary. No global RNG (paired-seed discipline).
            cur = [np.asarray(r.embedding, dtype=np.float64)
                   for r in slot.ledger.receipts
                   if (r.kind == 'fit' and r.sign > 0
                       and r.embedding is not None
                       and r.embed_epoch == self._embed_epoch)]
            if len(cur) < 4:
                # Not enough lived evidence in the current coordinate frame
                # to identify the two pulls: the split is unfunded.
                return -1, -1, []
            axis_mean, axis, spread = self._principal_axis(cur)
            offset = axis * max(spread, 1e-3)
            geo_a.centroid = axis_mean + offset
            geo_b.centroid = axis_mean - offset

        child_a = self.create_slot(
            f"{slot.name}_a", geo_a, origin_operator='Differentiate')
        child_b = self.create_slot(
            f"{slot.name}_b", geo_b, origin_operator='Differentiate')

        # Partition receipts EXACTLY, by the pull each receipt was funding:
        # 1. current-epoch embedding vs the two child centroids (exact);
        # 2. channel overlap against the child threshold geometries;
        # 3. deterministic receipt-id parity (last resort, no receipt lost).
        ga = self.slots[child_a].geometry
        gb = self.slots[child_b].geometry
        receipts_out = []
        n_parent = len(slot.ledger.receipts)
        for r in slot.ledger.receipts:
            target = None
            if (axis is not None and r.embedding is not None
                    and r.embed_epoch == self._embed_epoch):
                emb = np.asarray(r.embedding, dtype=np.float64)
                da = float(np.linalg.norm(emb - ga.centroid))
                db = float(np.linalg.norm(emb - gb.centroid))
                if abs(da - db) > 1e-12:
                    target = child_a if da < db else child_b
            if target is None and r.channel_indices:
                wa = float(sum(ga.family_thresholds[c]
                               for c in r.channel_indices if c < NUM_FAMILIES))
                wb = float(sum(gb.family_thresholds[c]
                               for c in r.channel_indices if c < NUM_FAMILIES))
                if abs(wa - wb) > 1e-12:
                    target = child_a if wa > wb else child_b
            if target is None:
                target = child_a if r.receipt_id % 2 == 0 else child_b

            pr = Receipt(
                receipt_id=self._alloc_receipt_id(),
                kind=r.kind,
                source_operator='Differentiate',
                parent_receipt_ids=[r.receipt_id],
                log_offset=r.log_offset,
                episode=r.episode, time_step=r.time_step,
                slot_id=target,
                channel_indices=r.channel_indices,
                family_id=r.family_id,
                magnitude=r.magnitude,
                sign=r.sign,
                provenance=r.provenance,
                source_ledger=r.source_ledger,
                discount=r.discount,
                created_at=self._global_step,
                embedding=r.embedding,
                embed_epoch=r.embed_epoch,
            )
            self._add_receipt_to_slot(self.slots[target], pr)
            receipts_out.append(pr)

        # Fiber law: partition is exact — no receipt lost, none double-counted.
        assert len(receipts_out) == n_parent, \
            "Differentiate: partition lost or duplicated receipts"

        # Partition the support ring by the same rule (side of the split).
        for i, sup in enumerate(slot.geometry.support):
            if axis is not None:
                # Support entries are raw obs; assign by alternating halves of
                # the ring (re-embedding happens at the next rebase/fit).
                target_geo = ga if i % 2 == 0 else gb
            else:
                target_geo = ga if i % 2 == 0 else gb
            if len(target_geo.support) < SUPPORT_RING:
                target_geo.support.append(sup)

        # Duplicate edges from parent to both children
        for eid, edge in list(self.edges.items()):
            if edge.slot_a == slot_id:
                self._add_edge(child_a, edge.slot_b, edge.edge_type)
                self._add_edge(child_b, edge.slot_b, edge.edge_type)
            elif edge.slot_b == slot_id:
                self._add_edge(edge.slot_a, child_a, edge.edge_type)
                self._add_edge(edge.slot_a, child_b, edge.edge_type)

        slot.state = 'archaized'
        self._log_event('differentiated',
                        [slot_id, child_a, child_b],
                        [r.receipt_id for r in receipts_out],
                        {'criterion': str(split_criterion)[:100]})
        return child_a, child_b, receipts_out

    def unify(self, slot_a, slot_b):
        self._op_counts['unify'] += 1
        sa, sb = self.slots[slot_a], self.slots[slot_b]

        pair = (min(slot_a, slot_b), max(slot_a, slot_b))
        if pair in self.exclusions:
            return -1, []

        # Geometry identity check: cosine of family thresholds
        na = np.linalg.norm(sa.geometry.family_thresholds) + 1e-8
        nb = np.linalg.norm(sb.geometry.family_thresholds) + 1e-8
        cosine = float(np.dot(sa.geometry.family_thresholds,
                              sb.geometry.family_thresholds) / (na * nb))

        if cosine < 0.8:
            divergent = (
                [r.receipt_id for r in sa.ledger.receipts[-3:]] +
                [r.receipt_id for r in sb.ledger.receipts[-3:]])
            self.exclude(slot_a, slot_b, divergent)
            return -1, []

        # Merge geometry — weights sum to 1 so unifying identical slots
        # preserves their geometry exactly. Zero-evidence policy: equal
        # weights when neither side has funded fits.
        total_fit = sa.ledger.fit_count + sb.ledger.fit_count
        if total_fit > 0:
            wa = sa.ledger.fit_count / total_fit
            wb = sb.ledger.fit_count / total_fit
        else:
            wa = wb = 0.5
        merged_geo = ConnectorGeometry(
            family_thresholds=sa.geometry.family_thresholds * wa +
                              sb.geometry.family_thresholds * wb,
            eigen_soft=sa.geometry.eigen_soft * wa + sb.geometry.eigen_soft * wb,
            eigen_code=sa.geometry.eigen_code,
            neighbors=[],
            centroid=sa.geometry.centroid * wa + sb.geometry.centroid * wb,
            radius=max(sa.geometry.radius, sb.geometry.radius),
            support=(list(sa.geometry.support)
                     + list(sb.geometry.support))[:SUPPORT_RING],
        )

        merged_id = self.create_slot(
            f"unify({sa.name},{sb.name})", merged_geo,
            origin_operator='Unify')
        merged = self.slots[merged_id]

        # Pool: union of both receipt histories (fiber primitive)
        pooled = []
        for r in sa.ledger.receipts + sb.ledger.receipts:
            pr = Receipt(
                receipt_id=self._alloc_receipt_id(),
                kind=r.kind,
                source_operator='Unify',
                parent_receipt_ids=[r.receipt_id],
                log_offset=r.log_offset,
                episode=r.episode, time_step=r.time_step,
                slot_id=merged_id,
                channel_indices=r.channel_indices,
                family_id=r.family_id,
                magnitude=r.magnitude,
                sign=r.sign,
                provenance=r.provenance,
                source_ledger=r.source_ledger,
                discount=r.discount,
                created_at=self._global_step,
                embedding=r.embedding,
                embed_epoch=r.embed_epoch,
            )
            self._add_receipt_to_slot(merged, pr)
            pooled.append(pr)

        # Transfer edges
        for eid, edge in list(self.edges.items()):
            if edge.slot_a == slot_a or edge.slot_a == slot_b:
                other = edge.slot_b
                if other not in (slot_a, slot_b):
                    self._add_edge(merged_id, other, edge.edge_type)
            elif edge.slot_b == slot_a or edge.slot_b == slot_b:
                other = edge.slot_a
                if other not in (slot_a, slot_b):
                    self._add_edge(other, merged_id, edge.edge_type)

        sa.state = 'archaized'
        sb.state = 'archaized'

        self._log_event('unified', [merged_id, slot_a, slot_b],
                        [r.receipt_id for r in pooled],
                        {'cosine': cosine})
        return merged_id, pooled

    def reopen(self, slot_id, failing_receipt_ids):
        """DERIVED (2026-08-10): the closure case of generalized Retract —
        retract enough closure-defining constraint that the feasible set is
        no longer a singleton. Kept as a named method because the 404
        window calls it directly; its lift is Retract's (compensated
        widening, retraction event)."""
        self._op_counts['reopen'] += 1
        slot = self.slots[slot_id]
        if slot.state != 'closed':
            return []

        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='retraction',
            source_operator='Reopen',
            parent_receipt_ids=list(failing_receipt_ids),
            log_offset=-1, episode=-1, time_step=-1,
            slot_id=slot_id,
            channel_indices=[],
            family_id=-1,
            magnitude=float(self._connectivity(slot_id)),
            sign=0,
            provenance='LIVED',
            source_ledger='COGNITIVE',
            created_at=self._global_step,
        )
        self._add_receipt_to_slot(slot, receipt)

        slot.state = 'open'
        slot.resolution = None
        slot.closed_at = -1
        slot.dormant = False
        slot.ledger.fail_window = []
        slot.ledger.reopen_count += 1

        # Law 6: the feasible set is restored from lived receipts (the
        # etymology), not generated arbitrarily.
        cur = [np.asarray(r.embedding, dtype=np.float64)
               for r in slot.ledger.receipts
               if (r.kind == 'fit' and r.embedding is not None
                   and r.embed_epoch == self._embed_epoch)]
        if len(cur) >= 2:
            centroid = np.mean(cur, axis=0)
            dists = [float(np.linalg.norm(e - centroid)) for e in cur]
            slot.geometry.centroid = centroid
            slot.geometry.radius = max(float(np.mean(dists)), CLOSURE_RADIUS * 2)
        else:
            slot.geometry.radius = max(slot.geometry.radius * 2.0, 0.5)

        # Notify dependents
        for neighbor_id in self._get_neighbors(slot_id):
            self._log_event('reopen_notify', [neighbor_id], [],
                            {'source': slot_id})

        self._log_event('reopened', [slot_id], [receipt.receipt_id],
                        {'connectivity': self._connectivity(slot_id),
                         'liability': slot.posit_liability})
        return [receipt]

    def quote(self, slot_id):
        self._op_counts['quote'] += 1
        slot = self.slots[slot_id]

        # Meta-geometry: derived from receipt-arrival statistics
        if slot.ledger.fit_count < 3:
            return -1, []

        fit_receipts = [r for r in slot.ledger.receipts
                        if r.kind == 'fit' and r.sign > 0
                        and r.time_step >= 0]
        fit_times = [r.time_step for r in fit_receipts]
        if len(fit_times) < 2:
            return -1, []

        intervals = np.diff(sorted(fit_times)).astype(np.float64)
        arrival_rate = 1.0 / (intervals.mean() + 1e-8)
        arrival_var = float(intervals.var()) if len(intervals) > 1 else 0.0

        meta_thresholds = slot.geometry.family_thresholds.copy() * 0.5
        meta_geo = ConnectorGeometry(
            family_thresholds=meta_thresholds,
            eigen_soft=slot.geometry.eigen_soft.copy(),
            eigen_code=slot.geometry.eigen_code,
            neighbors=[],
            centroid=slot.geometry.centroid.copy(),
            radius=float('inf'),
        )

        meta_id = self.create_slot(
            f"quote({slot.name})", meta_geo, origin_operator='Quote')

        # Meta-accrual grounds in the quoted slot's FIT receipts only —
        # anneal/eviction events are schedule-licensed, not lived evidence.
        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='meta',
            source_operator='Quote',
            parent_receipt_ids=[r.receipt_id for r in fit_receipts[-5:]],
            log_offset=-1, episode=-1, time_step=-1,
            slot_id=meta_id,
            channel_indices=[],
            family_id=-1,
            magnitude=arrival_rate,
            sign=0,
            provenance='LIVED',
            source_ledger='COGNITIVE',
            created_at=self._global_step,
        )
        self._add_receipt_to_slot(self.slots[meta_id], receipt)

        self._add_edge(meta_id, slot_id, 'constraint', [receipt.receipt_id])

        self._log_event('quoted', [meta_id, slot_id], [receipt.receipt_id],
                        {'arrival_rate': arrival_rate, 'arrival_var': arrival_var})
        return meta_id, [receipt]

    def posit(self, slot_id, candidate_value):
        self._op_counts['posit'] += 1
        slot = self.slots[slot_id]
        if slot.state != 'open':
            return {}

        # Read-only propagation over a hypothetical clamp. Nothing here may
        # touch the funded ledger: the record goes to the imagination
        # register, never the etymology (IMAGINED provenance boundary).
        candidate = np.asarray(candidate_value, dtype=np.float64)
        neighbors = self._get_neighbors(slot_id)
        ripple = {}

        for nid in neighbors:
            ns = self.slots[nid]
            dist = float(np.linalg.norm(
                candidate[:len(ns.geometry.centroid)] - ns.geometry.centroid))
            if ns.geometry.radius < float('inf'):
                change = max(0.0, 1.0 - dist / (ns.geometry.radius + 1e-8))
            else:
                change = 0.1
            ripple[nid] = {
                'change': change,
                'current_certainty': ns.ledger.certainty,
                'connectivity': self._connectivity(nid),
            }

        self.imagination_log.append({
            'op': 'posit', 'slot_id': slot_id, 'step': self._global_step,
            'candidate_norm': float(np.linalg.norm(candidate)),
            'ripple_size': len(ripple),
            'provenance': 'IMAGINED',
        })
        return ripple

    # -----------------------------------------------------------------------
    # DERIVED OPERATIONS
    # -----------------------------------------------------------------------

    def transfer(self, slot_id, target_context_slots):
        """Derived: Abstract then Constrain (analogy = join-then-restrict)."""
        geo = self.abstract(slot_id, target_context_slots[0])
        if geo[0] < 0:
            return -1, []
        parent_id = geo[0]
        for ctx_id in target_context_slots[1:]:
            self.constrain(parent_id, [
                r.receipt_id for r in self.slots[ctx_id].ledger.receipts[-3:]
                if r.kind == 'fit'])
        return parent_id, geo[1]

    def transpose(self, slot_id, clamped_neighbor_id, candidate_value):
        """Derived: reverse-mode Posit — clamp the neighbor, read this slot.
        Replay-gated like Posit (IMAGINED provenance, offline only)."""
        ripple = self.posit(clamped_neighbor_id, candidate_value)
        return ripple.get(slot_id, {})

    def _provenance_descendant_slots(self, slot_id):
        """The suspension set: slots holding receipts whose funding chains
        pass through any receipt of slot_id. This is what makes deep
        counterfactuals computable — 'had K been otherwise' must also
        suspend everything K funded, and only the provenance DAG knows
        what that is (contained counterfactuals, P71)."""
        base = {r.receipt_id for r in self.slots[slot_id].ledger.receipts}
        memo = {}

        def funded_through(rid, depth=0):
            if rid in base:
                return True
            if rid in memo:
                return memo[rid]
            if depth > 50:
                return False
            r = self._receipts_by_id.get(rid)
            if r is None or not r.parent_receipt_ids:
                memo[rid] = False
                return False
            out = any(funded_through(p, depth + 1)
                      for p in r.parent_receipt_ids)
            memo[rid] = out
            return out

        result = set()
        for sid, slot in self.slots.items():
            if sid == slot_id:
                continue
            for r in slot.ledger.receipts:
                if r.parent_receipt_ids and funded_through(r.receipt_id):
                    result.add(sid)
                    break
        return result

    def _receipt_survives(self, receipt, masked_slot_id, memo, depth=0):
        """Evaluate a receipt's justification formula with the masked slot's
        receipts false. AND requires all parents to survive; OR requires
        any. Lazy ATMS: computed per suspension, never as stored labels."""
        if receipt.slot_id == masked_slot_id:
            return False
        rid = receipt.receipt_id
        if rid in memo:
            return memo[rid]
        if depth > 50:
            return False
        if (receipt.kind in ('fit', 'boundary') and receipt.log_offset >= 0
                and not receipt.parent_receipt_ids):
            memo[rid] = True
            return True
        if receipt.provenance == 'ATTESTED':
            # Cross-ledger chains live in the exporter's ledger; masking a
            # local slot cannot reach them (Law 4).
            memo[rid] = True
            return True
        if not receipt.parent_receipt_ids:
            out = receipt.kind in ('eviction', 'anneal')
            memo[rid] = out
            return out
        parents = [self._receipts_by_id.get(p)
                   for p in receipt.parent_receipt_ids]
        parents = [p for p in parents if p is not None]
        if not parents:
            memo[rid] = False
            return False
        if receipt.justification == 'AND':
            out = all(self._receipt_survives(p, masked_slot_id, memo,
                                             depth + 1) for p in parents)
        else:
            out = any(self._receipt_survives(p, masked_slot_id, memo,
                                             depth + 1) for p in parents)
        memo[rid] = out
        return out

    def _context_support(self, slot_id, masked_slot_id, memo):
        """Surviving fraction of a slot's support with the masked slot's
        testimony removed. Positive funded mass first; existence receipts
        (structural / lien / meta) when the slot has no funded mass yet."""
        slot = self.slots[slot_id]
        total, surviving = 0.0, 0.0
        ex_total, ex_surv = 0, 0
        for r in slot.ledger.receipts:
            if r.receipt_id in self.retracted_receipt_ids:
                continue
            if r.kind in ('fit', 'transfer') and r.sign > 0:
                w = r.magnitude * r.discount
                total += w
                if self._receipt_survives(r, masked_slot_id, memo):
                    surviving += w
            elif r.kind in ('structural', 'lien', 'meta'):
                ex_total += 1
                if self._receipt_survives(r, masked_slot_id, memo):
                    ex_surv += 1
        if total > 0:
            return surviving / total
        if ex_total > 0:
            return ex_surv / ex_total
        return 1.0

    def _counterfactual_voucher(self, masked_slot_id, surface_slots):
        """The counterfactual license, made computable (registered
        2026-08-10): every counterfactual answer ships with a certificate
        of what it rests on. Rung 3 is only as trustworthy as the audit
        status of the structure the query touches — the voucher makes
        'can't fully vouch' per-query metadata instead of a global
        disclaimer.

        Surface = the masked slot's receipts plus those of every slot whose
        support was actually evaluated (the provenance-descendant set)."""
        n = 0
        lived_mass = 0.0
        attested_mass = 0.0
        declared_and = 0
        defaulted_or_multiparent = 0
        emb_current = 0
        emb_stale = 0
        retracted = 0
        for sid in [masked_slot_id] + sorted(surface_slots):
            slot = self.slots.get(sid)
            if slot is None:
                continue
            for r in slot.ledger.receipts:
                n += 1
                if r.receipt_id in self.retracted_receipt_ids:
                    retracted += 1
                    continue
                if r.kind in ('fit', 'transfer') and r.sign > 0:
                    w = r.magnitude * r.discount
                    if r.provenance == 'ATTESTED':
                        attested_mass += w
                    else:
                        lived_mass += w
                if len(r.parent_receipt_ids) > 1:
                    if r.justification == 'AND':
                        declared_and += 1
                    else:
                        # Under-suppression risk surface: a multi-parent OR
                        # may be a defaulted formula whose true semantics
                        # were AND — queries resting on these are the known
                        # vouching hole.
                        defaulted_or_multiparent += 1
                if r.embedding is not None:
                    if r.embed_epoch == self._embed_epoch:
                        emb_current += 1
                    else:
                        emb_stale += 1
        total_mass = lived_mass + attested_mass
        emb_total = emb_current + emb_stale
        flags = []
        if defaulted_or_multiparent > 0:
            flags.append('or_defaulted_multiparent')
        if attested_mass > 0:
            flags.append('rests_on_testimony')
        if emb_total and emb_current < emb_total:
            flags.append('stale_embeddings_on_surface')
        if self._last_audit['clean'] is not True:
            flags.append('no_clean_audit_on_record')
        return {
            'surface_slots': 1 + len(surface_slots),
            'receipts_examined': n,
            'lived_fraction': (lived_mass / total_mass
                               if total_mass > 0 else 1.0),
            'attested_discounted_mass': round(attested_mass, 6),
            'declared_and': declared_and,
            'defaulted_or_multiparent': defaulted_or_multiparent,
            'embedding_currency': (emb_current / emb_total
                                   if emb_total else 1.0),
            'retracted_excluded': retracted,
            'last_audit_step': self._last_audit['step'],
            'last_audit_clean': self._last_audit['clean'],
            'flags': flags,
        }

    def suspend(self, slot_id):
        """DERIVED modal operator (2026-08-10, refined same day): mask a K's
        testimony in a copy-on-write context and recompute every dependent
        structure from its REMAINING support paths.

        Not a lifecycle transition — an evaluation mode. The state has two
        axes: epistemic status (open|closed|archaized, the ledger's truth)
        and evaluation mode (actual | hypothetical(context)). The K stays
        rationally closed; within the context it is hypothetically open.

        Support semantics (the ATMS refinement): 'suspend everything funded
        through K' over-suppresses — K∧A→C, B→C must leave C standing on B.
        Each dependent slot gets a graded surviving-support fraction from
        its receipts' AND/OR justification formulas; a conclusion is gone
        only when NO support environment excludes K. This is where the
        ledger becomes a truth-maintenance engine rather than an audit
        trail (de Kleer's ATMS, given receipts, funding, and the firewall).

        Returns a non-funding hypothesis package (IMAGINED): what is gone,
        what survives discounted, what stands independently, and per-slot
        context certainties. Pure Suspend (no clamp) is a robustness audit:
        how much of the web stands on this belief alone?
        """
        self._op_counts['suspend'] += 1
        slot = self.slots[slot_id]
        if slot.state not in ('open', 'closed'):
            return {}

        memo = {}
        prefilter = self._provenance_descendant_slots(slot_id)
        gone, discounted, independent = [], {}, []
        context_certainty = {}
        for sid, s in self.slots.items():
            if sid == slot_id or s.state not in ('open', 'closed'):
                continue
            frac = (self._context_support(sid, slot_id, memo)
                    if sid in prefilter else 1.0)
            context_certainty[sid] = 0.5 + (s.ledger.certainty - 0.5) * frac
            if sid in prefilter and frac <= 0.05:
                gone.append(sid)
            elif frac < 0.999:
                discounted[sid] = round(frac, 6)
            else:
                independent.append(sid)

        package = {
            'op': 'suspend',
            'suspended_slot': slot_id,
            'assumptions_suspended': [slot_id],
            'gone': sorted(gone),
            'discounted': discounted,
            'independent_count': len(independent),
            'context_certainty': context_certainty,
            'voucher': self._counterfactual_voucher(slot_id, prefilter),
            'provenance': 'IMAGINED',
        }
        self.imagination_log.append({
            'op': 'suspend', 'slot_id': slot_id, 'step': self._global_step,
            'gone': sorted(gone), 'n_discounted': len(discounted),
            'provenance': 'IMAGINED',
        })
        return package

    def counterposit(self, slot_id, candidate_value):
        """DERIVED (2026-08-10): Counterposit(K, v') = Posit(v') ∘ Suspend(K)
        — the counterfactual proper, Pearl's third rung.

        Suspend masks the K and recomputes support; the clamp then
        propagates the alternative through the context, with each affected
        slot's certainty taken from the CONTEXT (the K's implications do
        not testify at its own trial — the anti-dogma property). The
        Copernican arc runs through here: the package's differing
        predictions direct the generator toward discriminating
        observations; only lived receipts can license Retract/Reopen.
        IMAGINED throughout; replay-gated.
        """
        self._op_counts['counterposit'] += 1
        package = self.suspend(slot_id)
        if not package:
            return {}
        candidate = np.asarray(candidate_value, dtype=np.float64)

        targets = (set(self._get_neighbors(slot_id))
                   | set(package['gone'])
                   | set(package['discounted'].keys()))
        ripple = {}
        for nid in sorted(targets):
            ns = self.slots.get(nid)
            if ns is None or ns.state not in ('open', 'closed'):
                continue
            dist = float(np.linalg.norm(
                candidate[:len(ns.geometry.centroid)] - ns.geometry.centroid))
            if ns.geometry.radius < float('inf'):
                change = max(0.0, 1.0 - dist / (ns.geometry.radius + 1e-8))
            else:
                change = 0.1
            ripple[nid] = {
                'change': change,
                'current_certainty': package['context_certainty'].get(
                    nid, ns.ledger.certainty),
                'connectivity': self._connectivity(nid),
                'suspended': nid in package['gone'],
                'support_fraction': package['discounted'].get(
                    nid, 0.0 if nid in package['gone'] else 1.0),
            }

        package['op'] = 'counterposit'
        package['candidate_norm'] = float(np.linalg.norm(candidate))
        package['ripple'] = ripple
        self.imagination_log.append({
            'op': 'counterposit', 'slot_id': slot_id,
            'step': self._global_step,
            'candidate_norm': package['candidate_norm'],
            'ripple_size': len(ripple),
            'provenance': 'IMAGINED',
        })
        return package

    def replay_through_context(self, masked_slot_id, window):
        """Abduction-by-replay — the rung-3 completion (2026-08-10).

        Pearl's counterfactual needs abduction (infer the background from
        what actually occurred) because ordinary systems discard raw
        history. This architecture kept it: the append-only experience log
        IS the background context. So abduction here is replay — re-fit a
        window of actual lived observations through the masked context and
        ask what the actual history would have looked like without the K.

        window: iterable of dicts with 'receptor_values' (raw vector).
        Returns (IMAGINED, read-only): per-slot context fit counts, how
        many observations only the masked K explained (the discriminating
        observations — what any rival structure must account for), and how
        many become unassigned in the context.
        """
        self._op_counts['replay_context'] += 1
        results = {
            'observations': 0,
            'explained_by_masked': 0,
            'discriminating': 0,
            'unassigned_in_context': 0,
            'context_fit_counts': {},
        }
        # The replay runs against the SUSPENDED context: conclusions gone
        # under suspension cannot testify in it (they were K's implications).
        _ctx = self.suspend(masked_slot_id)
        gone = set(_ctx.get('gone', []))
        results['voucher'] = _ctx.get('voucher')
        masked_geo = self.slots[masked_slot_id].geometry
        active = [(sid, s) for sid, s in self.slots.items()
                  if sid != masked_slot_id and sid not in gone
                  and s.state in ('open', 'closed')]
        for entry in window:
            rv = entry['receptor_values']
            fa = self._obs_to_family_activations(rv)
            results['observations'] += 1
            m_score, _ = self._boundary_test(masked_geo, fa)
            fit_masked = m_score >= 0.5
            any_other = False
            for sid, s in active:
                score, _ = self._boundary_test(s.geometry, fa)
                if score >= 0.5:
                    results['context_fit_counts'][sid] = (
                        results['context_fit_counts'].get(sid, 0) + 1)
                    any_other = True
            if fit_masked:
                results['explained_by_masked'] += 1
                if not any_other:
                    results['discriminating'] += 1
            if not any_other:
                results['unassigned_in_context'] += 1
        self.imagination_log.append({
            'op': 'replay_context', 'slot_id': masked_slot_id,
            'step': self._global_step,
            'observations': results['observations'],
            'discriminating': results['discriminating'],
            'provenance': 'IMAGINED',
        })
        return results

    def _check_closure(self, slot_id):
        slot = self.slots[slot_id]
        if slot.state != 'open':
            return False
        if slot.geometry.radius <= CLOSURE_RADIUS and slot.ledger.fit_count >= 3:
            slot.state = 'closed'
            slot.closed_at = self._global_step
            slot.resolution = {
                'centroid': slot.geometry.centroid.copy(),
                'certainty': slot.ledger.certainty,
                'fit_count': slot.ledger.fit_count,
                'radius': slot.geometry.radius,
            }
            slot.ledger.fail_window = []
            # Book contingent liability (P59)
            slot.posit_liability = float(self._connectivity(slot_id))
            self._log_event('closed', [slot_id], [],
                            {'radius': slot.geometry.radius,
                             'certainty': slot.ledger.certainty,
                             'liability': slot.posit_liability})
            return True
        return False

    def expectation_receipt(self, slot_id, embedding, err):
        """Expectation-as-first-class-evidence (2026-08-12, P76's
        transmission — the diagnosis F30/v2/v3 assembled: the Constrain
        channel saturates ~300 events, an order below ledger inertia).
        A CONFIRMED stage-expectation (the predicted family then
        actually fired, |err| small) bills a transfer-kind LIVED
        receipt and updates geometry toward the ACTUAL lived embedding
        with fixed alpha EXPECT_ALPHA — truth-directed by construction
        (only the observation that arrived is used; the prediction
        merely licenses the boosted weight: predicted-and-confirmed
        evidence is worth more than unpredicted evidence, which is the
        Serialization Thesis's delta-accounting claim as mechanism).
        Mass/Law-1 untouched (transfer kind, non-Fit operator)."""
        self._op_counts['expectation'] += 1
        slot = self.slots.get(slot_id)
        if slot is None or slot.state != 'open' or embedding is None:
            return []
        # Law 3: the receipt grounds through the lived evidence that
        # confirmed the expectation — the slot's own recent positive fit.
        parents = [r.receipt_id for r in slot.ledger.receipts
                   if r.kind == 'fit' and r.sign > 0][-1:]
        if not parents:
            return []          # unconfirmable without lived evidence
        receipt = Receipt(
            receipt_id=self._alloc_receipt_id(),
            kind='transfer',
            source_operator='StagedFit',
            parent_receipt_ids=parents,
            log_offset=-1, episode=-1, time_step=self._global_step,
            slot_id=slot_id,
            channel_indices=[],
            family_id=-1,
            magnitude=max(0.0, 1.0 - float(err)),
            sign=1,
            provenance='LIVED',
            source_ledger=self.ledger_id,
            created_at=self._global_step,
        )
        self._add_receipt_to_slot(slot, receipt)
        emb = np.asarray(embedding, dtype=np.float64)
        slot.geometry.centroid = (slot.geometry.centroid
                                  * (1 - EXPECT_ALPHA)
                                  + emb * EXPECT_ALPHA)
        dist = float(np.linalg.norm(emb - slot.geometry.centroid))
        slot.geometry.radius = (slot.geometry.radius
                                * (1 - EXPECT_ALPHA)
                                + dist * EXPECT_ALPHA)
        self._log_event('expectation', [slot_id],
                        [receipt.receipt_id], {'err': float(err)})
        return [receipt]

    # -----------------------------------------------------------------------
    # OCCLUDE / ENUMERATE — the Omission Cycle (imagination register §5a,
    # registered 2026-08-10; built 2026-08-12 for P77)
    # -----------------------------------------------------------------------

    def occlude(self, slot_id, min_fits=100):
        """DERIVED register operator: deliberate omission — withhold a
        FUNDED slot's content at register level, turning it into a query.
        C4 is inviolate: the funded ledger is untouched; the truth is
        SEALED into the returned context for later verification, never
        deleted. Case-3 guard (P77, pre-registered): omission over an
        unfunded or disconnected region is REFUSED — its enumeration
        failure would be uninformative."""
        self._op_counts['occlude'] += 1
        slot = self.slots.get(slot_id)
        if (slot is None or slot.state not in ('open', 'closed')
                or slot.ledger.fit_count < min_fits
                or self._connectivity(slot_id) < 1):
            return None
        truth = {'family_thresholds':
                 slot.geometry.family_thresholds.copy(),
                 'centroid': slot.geometry.centroid.copy(),
                 'origin_family': slot.origin_family,
                 'dominant_family':
                 int(np.argmax(slot.geometry.family_thresholds))}
        self.imagination_log.append({
            'op': 'occlude', 'slot_id': slot_id,
            'step': self._global_step,
            'fits': slot.ledger.fit_count,
            'connectivity': self._connectivity(slot_id),
            'provenance': 'IMAGINED',
        })
        return {'slot_id': slot_id, 'sealed_truth': truth,
                'occluded_at': self._global_step}

    def enumerate_gap(self, slot_id, walk_steps=3):
        """DERIVED register operator: EARNED enumeration — the candidate
        hypothesis space for an occluded/open slot's content, read from
        its CONNECTOR GEOMETRY alone. Blind to the slot's own geometry by
        construction: uses only the edge graph and the NEIGHBORS' funded
        structure (a receipt-weighted graph diffusion of `walk_steps`
        hops from the gap, certainty-weighted, exclusion-eliminated).
        Returns families ranked by earned score, most plausible first,
        plus the funding census (how many receipts stand behind the
        ranking). An empty/flat result over a funded neighborhood is
        ITSELF evidence (the geometry carries no hypothesis-space
        information — P77's falsifier)."""
        self._op_counts['enumerate'] += 1
        ids = [sid for sid, s in self.slots.items()
               if s.state in ('open', 'closed')]
        idx = {sid: i for i, sid in enumerate(ids)}
        n = len(ids)
        if slot_id not in idx or n < 2:
            return [], 0
        A = np.zeros((n, n))
        for e in self.edges.values():
            ia, ib = idx.get(e.slot_a), idx.get(e.slot_b)
            if ia is not None and ib is not None:
                A[ia][ib] += e.strength
                A[ib][ia] += e.strength
        row = A.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        A = A / row
        p = np.zeros(n)
        p[idx[slot_id]] = 1.0
        mass = np.zeros(n)
        for _ in range(walk_steps):
            p = A.T @ p
            p[idx[slot_id]] = 0.0          # blind to self
            mass += p
        scores = np.zeros(NUM_FAMILIES)
        funding = 0
        for sid, i in idx.items():
            if mass[i] <= 0 or sid == slot_id:
                continue
            s = self.slots[sid]
            fam = int(np.argmax(s.geometry.family_thresholds))
            scores[fam] += mass[i] * s.ledger.certainty
            funding += s.ledger.fit_count
        # Posit-elimination: families excluded against this slot score 0
        for (a, b) in self.exclusions:
            other = b if a == slot_id else (a if b == slot_id else None)
            if other is not None and other in self.slots:
                scores[int(np.argmax(
                    self.slots[other].geometry.family_thresholds))] = 0.0
        order = np.argsort(-scores, kind='stable')
        ranked = [(float(scores[f]), int(f)) for f in order
                  if scores[f] > 0]
        self.imagination_log.append({
            'op': 'enumerate', 'slot_id': slot_id,
            'step': self._global_step,
            'candidates': len(ranked), 'funding': int(funding),
            'provenance': 'IMAGINED',
        })
        return ranked, funding

    # -----------------------------------------------------------------------
    # REBASE — keep feasible-set geometry coherent across encoder epochs
    # -----------------------------------------------------------------------

    def rebase(self, encoder):
        """Recompute every slot's feasible-set geometry under a new encoder.

        The mental-model rebuild retrains the contrastive encoder each
        generation; embeddings from different epochs are unaligned. Because
        slots keep raw core-obs support samples (and Fit receipts carry log
        provenance), the geometry is re-derivable in the new space — the
        etymology makes the web re-basable. Receipts' stored embeddings are
        stamped with the old epoch and excluded from exact partitions until
        refreshed by new fits."""
        self._embed_epoch += 1
        rebased = 0
        for sid, slot in self.slots.items():
            if slot.state not in ('open', 'closed'):
                continue
            sup = slot.geometry.support
            if len(sup) < 2:
                continue
            embs = encoder.embed_batch(np.asarray(sup, dtype=np.float32))
            embs = np.asarray(embs, dtype=np.float64)
            centroid = embs.mean(axis=0)
            dists = np.linalg.norm(embs - centroid, axis=1)
            slot.geometry.centroid = centroid
            radius = float(dists.mean())
            if slot.state == 'open':
                slot.geometry.radius = radius
            else:
                # Closed K: resolution moves to the new coordinates; radius
                # keeps its closure-scale role for the 404 tolerance.
                slot.geometry.radius = max(radius, 1e-3)
                if slot.resolution is not None:
                    slot.resolution['centroid'] = centroid.copy()
            rebased += 1
        self._log_event('rebased', [], [],
                        {'epoch': self._embed_epoch, 'slots': rebased})
        return rebased

    # -----------------------------------------------------------------------
    # THRESHOLD REBASE — keep firing conditions coherent across activation
    # distributions (F41 impl. 10; the extinction autopsy's receipt:
    # 72/72 saturation deaths at a domain transition). C12 applied to the
    # slot's own boundary: a threshold is a QUANTILE commitment, not an
    # absolute price. Geometry sort only — no receipts are minted, no
    # ledger fields touched; Law 1 is safe by construction.
    # -----------------------------------------------------------------------

    def observe_activations(self, receptor_values):
        """Scout path: record family activations WITHOUT fitting — no
        receipts, no boundary tests, no economy. Perception without
        commitment, for filling the ring at a domain boundary before
        rethresholding (a vacuous transplant would otherwise be executed
        by rent inside its first generation of real fits)."""
        fa = self._obs_to_family_activations(receptor_values)
        self._fam_act_ring[self._fam_act_n % ACT_RING] = fa
        self._fam_act_n += 1
        return fa

    def snapshot_activation_dist(self):
        """Sorted per-family copies of the current activation ring —
        the OLD distribution, taken before a domain transition. Taking
        the snapshot CLOSES THE OLD BOOK: the ring resets, so
        everything observed afterward accumulates the new
        distribution alone (rethreshold would otherwise compare
        against a blend of both worlds — caught by the battery on
        first run)."""
        n = min(self._fam_act_n, ACT_RING)
        if n < RETHRESH_MIN_SAMPLES:
            return None
        block = np.sort(self._fam_act_ring[:n].copy(), axis=0)
        self._fam_act_n = 0
        return {'n': int(n), 'sorted': block}

    def rethreshold(self, old_snapshot, include_closed=False):
        """Quantile-preserving threshold rebase: each active threshold's
        percentile under the OLD distribution is mapped to the same
        percentile of the CURRENT ring. Open slots only by default —
        a K's boundary is part of its closed identity; rewriting it
        silently would be a commitment change without the world's
        testimony (C4-adjacent), so closed Ks opt in explicitly."""
        if old_snapshot is None:
            return 0
        n_new = min(self._fam_act_n, ACT_RING)
        if n_new < RETHRESH_MIN_SAMPLES:
            return 0
        old_sorted = old_snapshot['sorted']
        n_old = old_snapshot['n']
        new_block = self._fam_act_ring[:n_new]
        adjusted_slots = []
        for sid, slot in self.slots.items():
            if slot.state == 'closed' and not include_closed:
                continue
            if slot.state not in ('open', 'closed'):
                continue
            th = slot.geometry.family_thresholds
            moved = False
            for fam in np.where(th > 0)[0]:
                q = np.searchsorted(old_sorted[:, fam],
                                    th[fam]) / float(n_old)
                q = float(np.clip(q, RETHRESH_Q_LO, RETHRESH_Q_HI))
                new_t = float(np.quantile(new_block[:, fam], q))
                new_t = max(new_t, RETHRESH_FLOOR)
                if abs(new_t - th[fam]) > 1e-12:
                    th[fam] = new_t
                    moved = True
            if moved:
                adjusted_slots.append(sid)
        self._log_event('rethreshold', adjusted_slots, [],
                        {'n_old': int(n_old), 'n_new': int(n_new),
                         'slots': len(adjusted_slots),
                         'include_closed': bool(include_closed)})
        return len(adjusted_slots)

    # -----------------------------------------------------------------------
    # CONSERVATION LAW ENFORCEMENT
    # -----------------------------------------------------------------------

    def _find_receipt(self, receipt_id):
        return self._receipts_by_id.get(receipt_id)

    def check_conservation_laws(self):
        violations = []

        # Law 1: No creation — only Fit creates lived-receipt mass
        # (positive-sign Fit receipts; mismatch receipts carry no mass).
        original_fit_mass = sum(
            r.magnitude * r.discount
            for s in self.slots.values()
            for r in s.ledger.receipts
            if r.source_operator == 'Fit' and r.kind == 'fit' and r.sign > 0)
        # Relative tolerance: the two sums accumulate in different orders,
        # so float rounding alone exceeds any absolute epsilon once mass
        # reaches ~1e5 (replay run 2026-08-11: 8 false positives at 4.5M
        # fits, both sides printing identical to 4 decimals).
        if (abs(original_fit_mass - self._total_fit_mass)
                > 1e-6 * max(1.0, abs(original_fit_mass))):
            violations.append(
                f"Law 1: original Fit mass {original_fit_mass:.4f} "
                f"!= tracked {self._total_fit_mass:.4f}")

        # Law 2: No destruction — the global etymology grows monotonically.
        if len(self.etymology) < self._etymology_len:
            violations.append(
                f"Law 2: etymology shrunk {self._etymology_len} "
                f"-> {len(self.etymology)}")
        self._etymology_len = len(self.etymology)

        # Law 3: every funded receipt chain grounds in a lived Fit receipt.
        # Deterministic stride sample; O(depth) per trace via the index.
        all_receipts = list(self._receipts_by_id.values())
        if all_receipts:
            stride = max(1, len(all_receipts) // 100)
            for r in all_receipts[::stride]:
                if r.provenance == 'IMAGINED':
                    violations.append(
                        f"Law 3: IMAGINED receipt {r.receipt_id} in ledger")
                    break
                if not self._trace_grounding(r):
                    violations.append(
                        f"Law 3: receipt {r.receipt_id} "
                        f"({r.source_operator}/{r.kind}) ungrounded")
                    break

        # Imagination boundary: the funded etymology never records Posit.
        for ev in self.etymology[-200:]:
            if ev.event_type == 'posited':
                violations.append("Boundary: Posit event in funded etymology")
                break

        self._last_audit = {'step': self._global_step,
                            'clean': not violations}
        return violations

    def _trace_grounding(self, receipt, depth=0):
        if depth > 50:
            return False
        if receipt.kind in ('fit', 'boundary') and receipt.log_offset >= 0:
            return True
        if receipt.provenance == 'ATTESTED':
            # Law 4: the chain continues in the exporter's ledger; dual
            # provenance (source_ledger + discount) annotates the crossing.
            return True
        if not receipt.parent_receipt_ids:
            # Economy moves are licensed by schedule/rent, not lived data.
            return receipt.kind in ('eviction', 'anneal')
        for pid in receipt.parent_receipt_ids:
            parent = self._find_receipt(pid)
            if parent is None:
                return False
            if self._trace_grounding(parent, depth + 1):
                return True
        return False

    # -----------------------------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------------------------

    def get_stats(self):
        slots = self.slots.values()
        return {
            'total_slots': len(self.slots),
            'open': sum(1 for s in slots if s.state == 'open'),
            'closed': sum(1 for s in slots if s.state == 'closed'),
            # F23: assertable = closed AND in contact; dormant Ks hold
            # citable history, not citable truth
            'assertable': sum(1 for s in slots
                              if s.state == 'closed' and not s.dormant),
            'dormant': sum(1 for s in slots
                           if s.state == 'closed' and s.dormant),
            'archaized': sum(1 for s in slots if s.state == 'archaized'),
            'total_receipts': sum(s.ledger.receipt_count for s in slots),
            'total_fit_mass': self._total_fit_mass,
            'total_edges': len(self.edges),
            'exclusions': len(self.exclusions),
            'op_counts': dict(self._op_counts),
            'embed_epoch': self._embed_epoch,
            'etymology_events': len(self.etymology),
            'imagination_events': len(self.imagination_log),
            'unassigned_pool': len(self.unassigned_pool),
            'retracted_receipts': len(self.retracted_receipt_ids),
            # F13's demand ledger: slots ranked by closure churn =
            # de-conflation strain ranking (Differentiate's priority queue)
            'churn_by_slot': {s.name: s.ledger.reopen_count
                              for s in self.slots.values()
                              if s.ledger.reopen_count > 0},
        }

    def pop_fresh_tightness(self):
        """Mean fit-time distance of positive fits since the last pop, then
        reset. The E2-sensitive counterpart of the frozen radius EMA:
        testimony's centroid aim, if real, shows up HERE within the window
        it acts in. Returns (mean, n); (None, 0) when no fits landed."""
        n = self._fresh_dist_n
        mean = (self._fresh_dist_sum / n) if n else None
        self._fresh_dist_sum = 0.0
        self._fresh_dist_n = 0
        return mean, n

    def get_slot_summary(self, slot_id):
        s = self.slots[slot_id]
        return {
            'id': s.slot_id, 'name': s.name, 'state': s.state,
            'certainty': s.ledger.certainty,
            'fit_count': s.ledger.fit_count,
            'radius': s.geometry.radius,
            'connectivity': self._connectivity(slot_id),
            'rent_balance': s.ledger.rent_balance,
            'receipt_count': s.ledger.receipt_count,
        }

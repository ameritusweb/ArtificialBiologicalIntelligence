"""The Replay Phase — the non-monotone fragment at generation boundaries.

THE BUILD booked as experiments_booked.md card 3, with card 2 (cross-
membrane demand alignment) folded into the same instrumented run. Spec:
replay_phase_requirements.md. This docstring IS the finalized card; per
C20 it does not change after launch.

REGIME (why not deep-time T7/T8): closure churn — the demand ledger the
Differentiate scan consumes — is produced by encoder-epoch rebasing plus
regime alternation (F13: 195 attempts / 0 survivors across 49
generational rebuilds). The descriptive court speaks EL-0's grammar,
which covers TieredEnvironment. Demand alignment requires BOTH ledgers
watching the SAME worlds, so this run gives the tiered harness the
generational structure that produces churn: per-generation encoder
rebuild (build_mental_model on fresh lived boot data) + web.rebase, with
tier alternation (4/3) inside each generation supplying the structural
non-stationarity that trips 404 windows. Full machinery, no toys: real
Organism, trained policy, real contrastive encoder, the F13-validated
ConstraintWeb lifecycle, the EL-1/EL-2 court verbatim.

STRUCTURE of one generation g (order fixed; scans run BEFORE rebase so
Differentiate sees current-epoch receipt embeddings):
  1. live: WORLDS_PER_GEN tiered worlds (tiers alternate 4,3,4,3; seeds
     90000+g*100+i), EPISODES x STEPS policy-inhabitant episodes; every
     step the web fits real encoder embeddings (fit_all); every WINDOW
     steps a behavior window is recorded for the court AND per-slot
     reopen deltas are attributed to the window's active contexts.
  2. anneal_all.
  3. REPLAY SCANS (the non-monotone fragment, caps 1 event/scan/boundary):
     a. Individuate over the unassigned pool (genesis).
     b. Differentiate on the top-churn OPEN slot (reopen_count >=
        CHURN_MIN); first split arms the acceptance test.
     c. Compose on the top co-fit pair (co-positive fits >= COFIT_MIN,
        no existing edge, pair not already composed).
     d. Unify on the top threshold-cosine pair (>= UNIFY_COS); the
        operator's own admission gates decide.
     e. Exclude on the top divergent pair (one fits while the other
        fails, count >= DIVERGENT_MIN), funded by sampled divergent
        receipt ids captured when the divergence was lived.
  4. COURT: junction law across generations — gen g's pending proposals
     are ratified on gen g+1's fresh worlds (world-held-out: train
     worlds 0..2, val world 3; ratify r_train worlds 0..1, r_val 2).
  5. conservation check (any violation recorded and printed).
  6. encoder rebuild (generate_training_data seed BOOT_SEED+13*(g+1),
     build_mental_model) + web.rebase — entering gen g+1's frame.

PRE-REGISTERED ENDPOINTS AND VERDICT RULES (fixed before launch):

ACCEPTANCE — split-reduces-churn (F13 impl. 7, the discriminator):
  Target: FIRST Differentiate fires on the top-churn slot (reopen_count
  >= CHURN_MIN = 3). Matched exposure: parent churn rate = pre-split
  reopens / pre-split positive fits; children churn rate = combined
  post-split reopens / combined post-split positive fits (deltas from
  split-time snapshots). Exposure floor: combined post-split child fits
  >= 0.5 x parent pre-split fits. R = child_rate / parent_rate.
    CONFLATION SUPPORTED  iff R <= 0.5   (the split named a distinction)
    NON-STATIONARITY      iff R >= 0.8   (churn is a world-regime clock
                                          -> T153 gets a measurement)
    INCONCLUSIVE          iff 0.5 < R < 0.8
    UNTESTED              iff no split fired or exposure floor unmet.

P60/P69 — composed-slot survival LOW and SELECTIVE:
  Floors: >= 4 Compose events. Survival = state in (open, closed) at run
  end. LOW iff survival fraction < 0.5. SELECTIVE iff survivors' mean
  positive-fit count (post-create) > evicted slots' mean.
    SUPPORTED both / PARTIAL one / NOT SUPPORTED neither / UNTESTED.

F20 RELATIONAL-CLOSURE (pre-registered when F20 billed): among slots
  CLOSED AND STILL CLOSED at run end, the FIRST (earliest closed_at):
    SUPPORTED      iff origin_operator in {Compose, Differentiate,
                       Individuate, Abstract} (relational/meta-level)
    NOT SUPPORTED  iff origin_operator == inherited (positional)
    UNTESTED       iff the class stays empty (F13's empty class
                       persists; reported, still informative).

DEMAND ALIGNMENT (card 2, folded): organism ledger = the top-churn
  slot's reopen events attributed to active contexts, scored by lift
  (context share among that slot's churn events / context base rate
  among all windows). Language ledger = court proposal counts per
  context (substring match of proposed words against the observed
  context vocabulary).
    Primary: ALIGNED iff the top-churn slot's argmax-lift context ==
    the court's most-proposed context. Secondary: Spearman rank
    correlation between per-context churn mass (all slots) and per-
    context proposal counts. Floors: total reopen events >= 10 AND
    total proposals >= 5, else UNTESTED.

C20 PRE-FLIGHT (six checks):
  1. Domain match — web, encoder, court, policy all built against
     tiered worlds this arc (F16-F19 lineage); encoder is the real
     build_mental_model contrastive encoder (SOV-P1 v2 lesson).
  2. Endpoint independence — the court never sees web state; the web
     never sees court output; scans key on web-internal statistics
     only; no endpoint feeds world generation.
  3. Exogeneity — world seeds and tier alternation fixed by formula a
     priori; encoder rebuild seeds fixed by formula.
  4. Pairing — within-run contrasts only: parent vs children at
     matched exposure; survivors vs evicted composed slots. No cross-
     arm pairing needed; no identity smoke needed.
  5. Phenomenon strength — floors above; SMOKE (3 gens, 2 worlds/gen)
     must show >= 1 closure attempt before the full launch, else the
     regime is iterated BEFORE launch (instrument check, not peeking).
  6. Endpoint sensitivity — reopen_count deltas, survival fractions,
     and lift ranks all move by construction; the court's proposal
     machinery moved in 10/10 LL-rate replicates (F19).
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_descriptive import (window_features,
                                     classify_behavior_from_features,
                                     WINDOW)
from environment_lexical import (Lexicon, base_predicates,
                                 evolve_one_generation, ratify_pending,
                                 append_ledger)
from live_receptors import LiveReceptorBank
from mental_model import build_mental_model
from receptor_eigen_coder import ReceptorEigenCoder
from sov import ConstraintWeb
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)

# ---- regime (fixed) ----
# Deep-time's closure dynamics need EXPOSURE CONCENTRATION (F13: attempts
# from gen 1 at ~30k receipts/gen, organisms living long in few worlds)
# — so few worlds, many episodes, and an encoder fed a generous lived-log
# window, not many thin diverse worlds.
GENERATIONS = 24
WORLDS_PER_GEN = 2
TIER_CYCLE = (4, 3)
EPISODES = 4
STEPS = 600
BOOT_SEED = 123
WORLD_SEED_BASE = 90000

# ---- scan thresholds (fixed) ----
CHURN_MIN = 3
COFIT_MIN = 50
UNIFY_COS = 0.95
DIVERGENT_MIN = 50
# Differentiate demand key (F29 impl. 3): 'churn' = the F13 key
# (froth-regime semantics; default, preserves all prior cards);
# 'near_miss' = boundary-channel concentration (durable-regime
# candidate — a slot collecting near-misses is one form under two
# pulls). Future cards set this explicitly.
DIFF_KEY = 'churn'
NEAR_MISS_MIN = 24           # ~3 full near-miss strides of demand

# ---- endpoint floors (fixed) ----
EXPOSURE_FLOOR_FRAC = 0.5
SPLIT_CONFLATION_R = 0.5
SPLIT_NONSTAT_R = 0.8
COMPOSE_FLOOR = 4
CHURN_EVENTS_FLOOR = 10
PROPOSALS_FLOOR = 5

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'replay_phase_overnight.json')


LOG_WINDOW = 19200     # recent lived entries the rebuild trains on (~4 gens)


def build_engine(lived_log=None):
    """The deep-time rebuild, verbatim in mechanism: the contrastive
    encoder retrains on the run's OWN accumulated lived log (F13's
    regime — this is what concentrates the lived distribution's
    embeddings enough for closure dynamics to engage). Gen 0 boots from
    generated episodes because no lived stream exists yet."""
    if lived_log:
        return build_mental_model(lived_log[-LOG_WINDOW:])
    _, _, _, log = generate_training_data(
        num_episodes=8, steps_per_episode=300, seed=BOOT_SEED)
    return build_mental_model(log)


def calibrate_taus(model):
    """Frozen behavior thresholds from probe worlds (EL-1 protocol)."""
    paths, moves = [], []
    for s in range(9000, 9004):
        env = TieredEnvironment(seed=s, tier=4)
        np.random.seed(s * 3)
        env.rng = np.random.RandomState(s * 3 + 1)
        rng = np.random.RandomState(s * 3 + 2)
        org = Organism()
        org.reset()
        xs, ys = [], []
        for step in range(STEPS):
            w = org.get_observation_window()
            act, _ = model.predict(w)
            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                act = np.zeros_like(act)
            elif r < EXPLORE_RATE:
                act = rng.randint(0, 2, size=len(act)).astype(act.dtype)
            org.step(act, env, step)
            xs.append(org.x)
            ys.append(org.y)
            if len(xs) == WINDOW:
                f = window_features(env, xs, ys, step - WINDOW + 1)
                paths.append(f['path'])
                moves.append(max(abs(f['toward_pain']),
                                 abs(f['toward_end'])))
                xs, ys = [], []
    return float(np.median(paths)), float(np.percentile(moves, 70))


class ScanState:
    """Driver-side statistics the scans consume. Web-internal facts only
    (C20 check 2): co-fits, divergences, churn attribution."""

    def __init__(self):
        self.cofit = defaultdict(int)          # (a,b) -> co-positive count
        self.divergent = defaultdict(int)      # (a,b) -> a-pos/b-neg count
        self.div_receipts = defaultdict(list)  # (a,b) -> sampled receipt ids
        self.composed_pairs = set()
        self.churn_events = []                 # {slot, gen, contexts, n}
        self.ctx_window_counts = defaultdict(int)
        self.total_windows = 0
        self._reopen_snap = {}

    def record_fits(self, results):
        pos, neg = [], []
        for sid, receipts in results:
            fits = [r for r in receipts if r.kind == 'fit']
            if any(r.sign > 0 for r in fits):
                pos.append((sid, fits))
            elif fits:
                neg.append((sid, fits))
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                a, b = sorted((pos[i][0], pos[j][0]))
                self.cofit[(a, b)] += 1
        for sid_p, fits_p in pos:
            for sid_n, fits_n in neg:
                key = (min(sid_p, sid_n), max(sid_p, sid_n))
                self.divergent[key] += 1
                if len(self.div_receipts[key]) < 4:
                    self.div_receipts[key].extend(
                        [fits_p[0].receipt_id, fits_n[0].receipt_id])

    def window_boundary(self, web, gen, contexts):
        self.total_windows += 1
        for c in contexts:
            self.ctx_window_counts[c] += 1
        for sid, slot in web.slots.items():
            prev = self._reopen_snap.get(sid, 0)
            cur = slot.ledger.reopen_count
            if cur > prev:
                self.churn_events.append(
                    {'slot': slot.name, 'slot_id': sid, 'gen': gen,
                     'contexts': list(contexts), 'n': cur - prev})
            self._reopen_snap[sid] = cur


def run_world(env, model, engine, web, bank, scan, gen, world_rng_seed,
              lived_log):
    """One world: policy-inhabitant episodes feeding web AND court from
    the SAME trajectory. Returns this world's court windows; appends the
    organisms' experience logs to lived_log (the rebuild's diet)."""
    np.random.seed(world_rng_seed)
    env.rng = np.random.RandomState(world_rng_seed + 1)
    rng = np.random.RandomState(world_rng_seed + 2)
    windows = []
    for ep in range(EPISODES):
        org = Organism()
        org.reset()
        xs, ys, preds0, ctx0 = [], [], None, []
        for step in range(STEPS):
            if step % WINDOW == 0:
                if len(xs) == WINDOW and preds0 is not None:
                    feat = window_features(env, xs, ys, step - WINDOW)
                    windows.append((preds0, feat))
                    scan.window_boundary(web, gen, ctx0)
                xs, ys = [], []
                preds0 = base_predicates(env, org, step)
                ctx0 = [p for p in preds0]
            w = org.get_observation_window()
            act, _ = model.predict(w)
            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                act = np.zeros_like(act)
            elif r < EXPLORE_RATE:
                act = rng.randint(0, 2, size=len(act)).astype(act.dtype)
            obs, reward = org.step(act, env, step)
            xs.append(org.x)
            ys.append(org.y)
            rv = bank.compute(obs, act, None, reward)
            core = engine._core_obs(obs)
            emb = engine.encoder.embed(core)
            web._global_step += 1
            results = web.fit_all(rv, emb, obs, obs, reward,
                                  web._global_step, ep, web._global_step,
                                  support_obs=core)
            scan.record_fits(results)
        lived_log.extend(org.experience_log)
    return windows


def replay_scans(web, scan, gen, record):
    """The non-monotone fragment. One event per scan per boundary."""
    ev = {'gen': gen}

    # a. Individuate — genesis from the unassigned pool
    new_id, receipts = web.individuate()
    if new_id >= 0:
        ev['individuate'] = {'slot': web.slots[new_id].name,
                             'receipts': len(receipts)}

    # b. Differentiate — demand-keyed open slot. DIFF_KEY (F29 impl. 3):
    # 'churn' (reopen_count, the F13 key — froth-regime semantics) or
    # 'near_miss' (boundary-channel concentration: "one form, two
    # pulls" — the durable-regime candidate). Default preserves every
    # prior card; new cards opt in explicitly.
    if DIFF_KEY == 'near_miss':
        cands = [(s.ledger.near_miss_seen, sid)
                 for sid, s in web.slots.items()
                 if s.state == 'open'
                 and s.ledger.near_miss_seen >= NEAR_MISS_MIN]
    else:
        cands = [(s.ledger.reopen_count, sid)
                 for sid, s in web.slots.items()
                 if s.state == 'open'
                 and s.ledger.reopen_count >= CHURN_MIN]
    if cands:
        cands.sort(key=lambda x: (-x[0], x[1]))
        _, sid = cands[0]
        parent = web.slots[sid]
        pre = {'name': parent.name, 'slot_id': sid, 'key': DIFF_KEY,
               'reopens_pre': parent.ledger.reopen_count,
               'near_miss_pre': parent.ledger.near_miss_seen,
               'fits_pre': parent.ledger.fit_count, 'gen': gen}
        ca, cb, _ = web.differentiate(sid)
        if ca >= 0:
            pre['children'] = {
                ca: {'fits_at_split': web.slots[ca].ledger.fit_count,
                     'reopens_at_split': web.slots[ca].ledger.reopen_count},
                cb: {'fits_at_split': web.slots[cb].ledger.fit_count,
                     'reopens_at_split': web.slots[cb].ledger.reopen_count}}
            record['splits'].append(pre)
            ev['differentiate'] = {'parent': parent.name,
                                   'children': [ca, cb]}

    def edged(a, b):
        return any((e.slot_a == a and e.slot_b == b)
                   or (e.slot_a == b and e.slot_b == a)
                   for e in web.edges.values())

    def both_open(a, b):
        return (web.slots.get(a) is not None and web.slots.get(b) is not None
                and web.slots[a].state == 'open'
                and web.slots[b].state == 'open')

    # c. Compose — top co-fit pair
    pairs = sorted(scan.cofit.items(), key=lambda kv: (-kv[1], kv[0]))
    for (a, b), n in pairs:
        if n < COFIT_MIN:
            break
        if (a, b) in scan.composed_pairs or not both_open(a, b) \
                or edged(a, b):
            continue
        new_id, _ = web.compose(a, b)
        if new_id >= 0:
            scan.composed_pairs.add((a, b))
            record['composed'].append(
                {'slot_id': new_id, 'name': web.slots[new_id].name,
                 'gen': gen, 'cofit': n,
                 'fits_at_create': web.slots[new_id].ledger.fit_count})
            ev['compose'] = {'pair': [a, b], 'new': new_id}
        break

    # d. Unify — top threshold-cosine pair; operator gates decide
    open_ids = [sid for sid, s in web.slots.items() if s.state == 'open']
    best = None
    for i in range(len(open_ids)):
        for j in range(i + 1, len(open_ids)):
            a, b = open_ids[i], open_ids[j]
            if (min(a, b), max(a, b)) in web.exclusions or edged(a, b):
                continue
            ta = web.slots[a].geometry.family_thresholds
            tb = web.slots[b].geometry.family_thresholds
            cos = float(np.dot(ta, tb)
                        / ((np.linalg.norm(ta) + 1e-8)
                           * (np.linalg.norm(tb) + 1e-8)))
            if cos >= UNIFY_COS and (best is None or cos > best[0]):
                best = (cos, a, b)
    if best is not None:
        _, a, b = best
        uid, _ = web.unify(a, b)
        if uid >= 0:
            ev['unify'] = {'pair': [a, b], 'new': uid, 'cos': best[0]}

    # e. Exclude — top divergent pair, funded by lived divergent receipts
    dpairs = sorted(scan.divergent.items(), key=lambda kv: (-kv[1], kv[0]))
    for (a, b), n in dpairs:
        if n < DIVERGENT_MIN:
            break
        if not both_open(a, b) or edged(a, b) \
                or (min(a, b), max(a, b)) in web.exclusions:
            continue
        rids = scan.div_receipts[(a, b)][:4]
        if web.exclude(a, b, rids):
            ev['exclude'] = {'pair': [a, b], 'divergences': n}
        break

    if len(ev) > 1:
        record['scan_events'].append(ev)


def context_vocab(scan):
    return sorted(scan.ctx_window_counts.keys())


def verdicts(web, scan, record):
    out = {}

    # ---- acceptance: split-reduces-churn ----
    if not record['splits']:
        out['split_reduces_churn'] = {'verdict': 'UNTESTED (no split fired)'}
    else:
        s = record['splits'][0]
        child_fits = child_reopens = 0
        for cid_s, snap in s['children'].items():
            cid = int(cid_s)
            slot = web.slots.get(cid)
            if slot is None:
                continue
            child_fits += slot.ledger.fit_count - snap['fits_at_split']
            child_reopens += (slot.ledger.reopen_count
                              - snap['reopens_at_split'])
        floor = EXPOSURE_FLOOR_FRAC * s['fits_pre']
        parent_rate = s['reopens_pre'] / max(s['fits_pre'], 1)
        child_rate = child_reopens / max(child_fits, 1)
        detail = {'parent': s['name'], 'split_gen': s['gen'],
                  'parent_rate': parent_rate, 'child_rate': child_rate,
                  'child_fits_post': child_fits,
                  'child_reopens_post': child_reopens,
                  'exposure_floor': floor}
        if child_fits < floor:
            v = 'UNTESTED (child exposure floor unmet)'
        else:
            R = child_rate / max(parent_rate, 1e-12)
            detail['R'] = R
            if R <= SPLIT_CONFLATION_R:
                v = ('CONFLATION SUPPORTED: the split named a real '
                     'distinction (Differentiate funded)')
            elif R >= SPLIT_NONSTAT_R:
                v = ('NON-STATIONARITY: churn is a world-regime clock '
                     '(T153 measurement)')
            else:
                v = 'INCONCLUSIVE (0.5 < R < 0.8)'
        detail['verdict'] = v
        out['split_reduces_churn'] = detail

    # ---- P60/P69: composed-slot survival ----
    comp = record['composed']
    if len(comp) < COMPOSE_FLOOR:
        out['p60_composed_survival'] = {
            'verdict': f'UNTESTED (composed={len(comp)} < {COMPOSE_FLOOR})',
            'composed': len(comp)}
    else:
        surv, evic = [], []
        for c in comp:
            slot = web.slots.get(c['slot_id'])
            fits_post = (0 if slot is None
                         else slot.ledger.fit_count - c['fits_at_create'])
            alive = slot is not None and slot.state in ('open', 'closed')
            (surv if alive else evic).append(fits_post)
        frac = len(surv) / len(comp)
        low = frac < 0.5
        selective = (len(surv) > 0 and len(evic) > 0
                     and float(np.mean(surv)) > float(np.mean(evic)))
        v = {(True, True): 'SUPPORTED: survival low AND selective',
             (True, False): 'PARTIAL: low but not selective',
             (False, True): 'PARTIAL: selective but not low',
             (False, False): 'NOT SUPPORTED'}[(low, selective)]
        out['p60_composed_survival'] = {
            'verdict': v, 'survival_fraction': frac,
            'n_composed': len(comp),
            'survivor_mean_fits': (float(np.mean(surv)) if surv else None),
            'evicted_mean_fits': (float(np.mean(evic)) if evic else None)}

    # ---- F20 relational closure ----
    closed = [(s.closed_at, s) for s in web.slots.values()
              if s.state == 'closed']
    if not closed:
        out['relational_closure'] = {
            'verdict': "UNTESTED (F13's empty class persists: no "
                       "surviving closures)"}
    else:
        closed.sort(key=lambda x: x[0])
        first = closed[0][1]
        relational = first.origin_operator in (
            'Compose', 'Differentiate', 'Individuate', 'Abstract')
        out['relational_closure'] = {
            'verdict': ('SUPPORTED: first surviving closure is '
                        f'{first.origin_operator}-born' if relational else
                        'NOT SUPPORTED: first surviving closure is '
                        'inherited/positional'),
            'first': {'name': first.name,
                      'origin': first.origin_operator,
                      'closed_at': first.closed_at},
            'all_surviving': [{'name': s.name, 'origin': s.origin_operator}
                              for _, s in closed]}

    # ---- demand alignment (folded card 2) ----
    total_churn = sum(e['n'] for e in scan.churn_events)
    total_props = sum(record['proposal_counts'].values())
    if total_churn < CHURN_EVENTS_FLOOR or total_props < PROPOSALS_FLOOR:
        out['demand_alignment'] = {
            'verdict': f'UNTESTED (churn events={total_churn}, '
                       f'proposals={total_props})',
            'churn_events': total_churn, 'proposals': total_props}
    else:
        by_slot = defaultdict(int)
        for e in scan.churn_events:
            by_slot[e['slot']] += e['n']
        top_slot = max(by_slot.items(), key=lambda kv: (kv[1], kv[0]))[0]
        vocab = context_vocab(scan)
        top_events = [e for e in scan.churn_events if e['slot'] == top_slot]
        top_mass = sum(e['n'] for e in top_events)
        lifts = {}
        for c in vocab:
            base = scan.ctx_window_counts[c] / max(scan.total_windows, 1)
            share = (sum(e['n'] for e in top_events if c in e['contexts'])
                     / max(top_mass, 1))
            if base > 0 and share > 0:
                lifts[c] = share / base
        churn_ctx = (max(lifts.items(), key=lambda kv: (kv[1], kv[0]))[0]
                     if lifts else None)
        pc = record['proposal_counts']
        lang_ctx = (max(pc.items(), key=lambda kv: (kv[1], kv[0]))[0]
                    if pc else None)
        # secondary: rank correlation over the shared vocabulary
        all_mass = defaultdict(float)
        for e in scan.churn_events:
            for c in e['contexts']:
                all_mass[c] += e['n']
        common = [c for c in vocab if c in pc or c in all_mass]
        rho = None
        if len(common) >= 3:
            xs = [all_mass.get(c, 0.0) for c in common]
            ys = [pc.get(c, 0) for c in common]
            rx = np.argsort(np.argsort(xs)).astype(float)
            ry = np.argsort(np.argsort(ys)).astype(float)
            if np.std(rx) > 0 and np.std(ry) > 0:
                rho = float(np.corrcoef(rx, ry)[0, 1])
        aligned = (churn_ctx is not None and lang_ctx is not None
                   and churn_ctx == lang_ctx)
        out['demand_alignment'] = {
            'verdict': ('ALIGNED: one shared demand ledger' if aligned
                        else 'MISALIGNED: the membranes name different '
                             'structures'),
            'top_churn_slot': top_slot,
            'top_churn_context': churn_ctx,
            'top_proposed_context': lang_ctx,
            'context_lifts': {k: round(v, 3) for k, v in lifts.items()},
            'proposal_counts': dict(pc),
            'spearman_rho': rho,
            'churn_events': total_churn, 'proposals': total_props}
    return out


def main(smoke=False):
    t0 = time.time()
    gens = 4 if smoke else GENERATIONS
    wpg = WORLDS_PER_GEN          # smoke runs the FULL per-gen config so
                                  # the phenomenon-strength check is real
    print(f'=== REPLAY PHASE {"SMOKE" if smoke else "OVERNIGHT"}: '
          f'{gens} generations x {wpg} worlds ===')

    print('boot: policy model + engine...')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()
    lived_log = []
    tau_rest, tau_d = calibrate_taus(model)
    print(f'frozen taus: rest={tau_rest:.4f} d={tau_d:.4f}')

    web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=0,
                        ledger_id='REPLAY')
    web.populate_from_families()
    scan = ScanState()
    lexicon = Lexicon()
    ledger = []
    pending_prev = None

    record = {'splits': [], 'composed': [], 'scan_events': [],
              'proposal_counts': defaultdict(int), 'generations': [],
              'ratified': [], 'conservation_violations': [],
              'config': {'generations': gens, 'worlds_per_gen': wpg,
                         'episodes': EPISODES, 'steps': STEPS,
                         'churn_min': CHURN_MIN, 'cofit_min': COFIT_MIN,
                         'unify_cos': UNIFY_COS,
                         'divergent_min': DIVERGENT_MIN}}

    for g in range(gens):
        bank = LiveReceptorBank()
        gen_windows = []
        for i in range(wpg):
            tier = TIER_CYCLE[i % len(TIER_CYCLE)]
            seed = WORLD_SEED_BASE + g * 100 + i
            env = TieredEnvironment(seed=seed, tier=tier)
            w = run_world(env, model, engine, web, bank, scan, g,
                          seed * 7, lived_log)
            gen_windows.append(w)

        web.anneal_all(web._global_step)
        replay_scans(web, scan, g, record)

        # court: propose on this gen (world-held-out), ratify LAST gen's
        # pending on THIS gen's fresh worlds (junction law)
        def as_court(ws):
            return [(p, classify_behavior_from_features(f, tau_rest, tau_d))
                    for p, f in ws]
        if pending_prev:
            r_train = [w for ws in gen_windows[:max(1, wpg - 1)]
                       for w in as_court(ws)][:2000]
            r_val = as_court(gen_windows[-1])
            lexicon, ratified = ratify_pending(lexicon, pending_prev,
                                               r_train, r_val, ledger)
            for kind, word, child in ratified:
                record['ratified'].append({'kind': kind, 'word': word,
                                           'child': child, 'gen': g})
        train_c = [w for ws in gen_windows[:max(1, wpg - 1)]
                   for w in as_court(ws)]
        val_c = as_court(gen_windows[-1])
        lexicon, moves, pending = evolve_one_generation(
            lexicon, train_c, val_c, ledger)
        pending_prev = pending
        for p in pending:
            # a split proposal names its structure in the DISCRIMINATOR
            # (the context being carved); a birth names it in the word
            named = (p['discriminator'] if p['kind'] == 'split'
                     else p['word'])
            for c in context_vocab(scan):
                if c in named:
                    record['proposal_counts'][c] += 1

        viol = web.check_conservation_laws()
        if viol:
            record['conservation_violations'].append({'gen': g,
                                                      'violations': viol})
            print(f'  !! conservation violations gen {g}: {viol}')

        stats = web.get_stats()
        closure_attempts = sum(s.ledger.reopen_count
                               for s in web.slots.values()) \
            + stats['closed']
        radii = sorted(round(s.geometry.radius, 4)
                       for s in web.slots.values()
                       if s.state == 'open' and s.ledger.fit_count >= 3
                       and np.isfinite(s.geometry.radius))
        record['generations'].append({
            'gen': g, 'open': stats['open'], 'closed': stats['closed'],
            'archaized': stats['archaized'],
            'receipts': stats['total_receipts'],
            'edges': stats['total_edges'],
            'unassigned_pool': stats['unassigned_pool'],
            'churn_by_slot': stats['churn_by_slot'],
            'closure_attempts': closure_attempts,
            'min_radii': radii[:5],
            'proposals_pending': len(pending)})
        el = (time.time() - t0) / 60
        print(f'gen {g + 1}/{gens} ({el:.1f} min): open={stats["open"]} '
              f'closed={stats["closed"]} arch={stats["archaized"]} '
              f'edges={stats["total_edges"]} '
              f'attempts={closure_attempts} '
              f'pending={len(pending)} minR={radii[:3]}')

        # entering gen g+1's frame: rebuild encoder on the lived log,
        # rebase the web (the deep-time cycle)
        if g < gens - 1:
            engine = build_engine(lived_log)
            n = web.rebase(engine.encoder)
            print(f'  rebased {n} slots into epoch {web._embed_epoch} '
                  f'(log={len(lived_log)})')

    out = {'record': {k: (dict(v) if isinstance(v, defaultdict) else v)
                      for k, v in record.items()},
           'verdicts': verdicts(web, scan, record),
           'total_churn_events': sum(e['n'] for e in scan.churn_events),
           'churn_events': scan.churn_events[-200:],
           'context_windows': dict(scan.ctx_window_counts),
           'total_windows': scan.total_windows,
           'final_stats': web.get_stats(),
           'smoke': smoke,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    path = RESULTS if not smoke else RESULTS.replace('.json', '_smoke.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    append_ledger(ledger)
    print('\n---- verdicts ----')
    for k, v in out['verdicts'].items():
        print(f'  {k}: {v["verdict"] if isinstance(v, dict) else v}')
    print(f'saved {path} ({out["elapsed_min"]} min)')


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)

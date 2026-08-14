"""P76 — staged fit: the Serialization Thesis 2x2 (the card, locked at
launch; second wave, first experiment; pre-registered in
replay_phase_requirements.md since 2026-08-10).

THE CLAIM: processing a single observation's families SERIALLY — each
stage forming an expectation about the next stage's activations from
already-processed stages via constraint edges — beats all-at-once
processing, PROVIDED the expectation receipts are consumed. The margin
is claimed to be delta-accounting, not latency.

DESIGN — one lived stream, four accountants (the SOV-P1 pattern:
byte-identical evidence, arms differ only in bookkeeping policy):
  WARMUP (6 worlds, all webs identical by construction — deterministic
  fits, no RNG in the web): standard fit_all + a compose scan per world
  boundary, so every web enters the treatment with the SAME earned edge
  structure (the edge density P76 always needed — F21-F28's webs).
  POST-SWITCH (6 worlds), the arms:
    P+D — standard fit_all (baseline).
    P+C — fit_all + ALL-AT-ONCE expectation billing: each family's
          activation predicted from ALL other families via edge
          weights; correct predictions consumed (matched accounting
          volume, complete information, NO staging order).
    S+D — staged (fringe-ordered) expectation computation, receipts
          DISCARDED. NOTE, stated honestly: with no consumption this
          web's state is identical to P+D by construction — the cell
          confirms the design (order alone does nothing mechanical),
          it does not test the thesis.
    S+C — staged fringe-ordered expectations, prefix-only information,
          consumed. The thesis cell.
  CONSUMPTION (the lawful mechanism — no new algebra): a correct
  expectation (|pred - actual| < TAU_E on an active family) licenses
  web.constrain(target-family slot, [a recent positive fit receipt of
  the top predicting slot]) — Law 6 narrowing, existing operator,
  battery-covered; capped at 1 per family per CONSTRAIN_STRIDE steps.
  FRINGE RULE: first family = highest edge degree; then argmax total
  edge weight to the processed set (the permutohedron geodesic at
  family grain).

ENDPOINTS (post-switch, world-agreement measures — mechanical
narrowing cannot fake them; the geometry must actually predict the
lived stream):
  (a) fresh-fit tightness (mean fit-time distance, popped per world)
  (b) calibration error (|certainty - recent confirm rate|, SOV-P1's)
  (c) enumeration sharpness per arm (mean |pred - actual| on active
      families) — the claimed MEDIATOR, measured not assumed.

VERDICTS (fixed): SUPPORTED iff S+C beats P+D on BOTH (a) and (b) AND
S+C beats P+C on at least one (staging's prefix constraint wins even
against complete-information accounting). PARTIAL: S+C beats P+D on
both but P+C >= S+C on both (accounting matters, order doesn't — the
thesis holds only at the delta-accounting grain). NOT SUPPORTED:
S+C <= P+D on both. UNTESTED: post-switch positive fits < 500 per web
or constrain events < 50 in S+C (the mechanism never engaged).

C20 (seven): 1 domain — the lineage harness, all components in-dist.
2 endpoint independence — constrain writes geometry; endpoints measure
geometry-vs-WORLD agreement (fresh distances against actual
embeddings), the legal path. 3 exogeneity — policy switch at a
pre-registered world index. 4 pairing — one stream, four accountants;
warmup states identical by construction (asserted at switch).
5 phenomenon strength — warmup must produce >= 30 edges (else the
expectation graph is empty; asserted before treatment). 6 sensitivity
— tightness and calibration moved in SOV-P1 under this harness.
7 genesis/rates — no genesis endpoint (continuous measures); receipts
distribution-bound: edge structure is grown in-run, not imported.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from live_receptors import LiveReceptorBank
from receptor_eigen_coder import ReceptorEigenCoder, FAMILY_GROUPS
from sov import ConstraintWeb, NUM_FAMILIES
from train import (generate_training_data, train_model, EXPLORE_RATE,
                   PROBE_RATE_FLOOR)
from replay_overnight import build_engine, BOOT_SEED

WARMUP_WORLDS = [(96500 + i, (4, 3)[i % 2]) for i in range(6)]
TREAT_WORLDS = [(96600 + i, (4, 3)[i % 2]) for i in range(6)]
EPISODES = 2
STEPS = 400
TAU_E = 0.1
ACTIVE_MIN = 0.25
CONSTRAIN_STRIDE = 50
CONSUME_MODE = 'constrain'   # or 'expectation' (sov.expectation_receipt
                             # — P76-v4's first-class-evidence channel)
EDGE_FLOOR = 30              # check-5 floor; the density SWEEP card
                             # varies this per cell (low cells are the
                             # treatment, not a deficiency)
CONFIRM_WINDOW = 200
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p76_staged_fit.json')


def family_of(slot):
    names = [n for n, _ in FAMILY_GROUPS]
    if slot.origin_family in names:
        return names.index(slot.origin_family)
    return int(np.argmax(slot.geometry.family_thresholds))


class Accountant:
    """One web + one bookkeeping policy over the shared stream."""

    def __init__(self, name, staged, consume):
        self.name = name
        self.staged = staged
        self.consume = consume
        self.web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(),
                                 debug_level=0, ledger_id=name)
        self.web.populate_from_families()
        self.sharp_sum = 0.0
        self.sharp_n = 0
        self.constrains = 0
        self._last_constrain = defaultdict(lambda: -10**9)
        self._W = None
        self._W_step = -10**9
        self.cofit = defaultdict(int)
        self.composed = set()

    def _edge_weights(self):
        if self.web._global_step - self._W_step < 100 and \
                self._W is not None:
            return self._W
        W = np.zeros((NUM_FAMILIES, NUM_FAMILIES))
        fam = {sid: family_of(s) for sid, s in self.web.slots.items()
               if s.state in ('open', 'closed')}
        for e in self.web.edges.values():
            fa, fb = fam.get(e.slot_a), fam.get(e.slot_b)
            if fa is not None and fb is not None and fa != fb:
                W[fa][fb] += e.strength
                W[fb][fa] += e.strength
        self._W = W
        self._W_step = self.web._global_step
        return W

    def _bill_expectation(self, f, pred, actual, fam_slots, emb=None):
        self.sharp_sum += abs(pred - actual)
        self.sharp_n += 1
        if not self.consume:
            return
        if abs(pred - actual) < TAU_E and actual >= ACTIVE_MIN:
            step = self.web._global_step
            if step - self._last_constrain[f] < CONSTRAIN_STRIDE:
                return
            target = fam_slots.get(f)
            if target is None:
                return
            if CONSUME_MODE == 'expectation':
                if emb is not None and self.web.expectation_receipt(
                        target, emb, abs(pred - actual)):
                    self.constrains += 1
                    self._last_constrain[f] = step
                return
            # fund the narrowing with a recent positive lived fit from
            # the strongest-predicting family's slot
            W = self._edge_weights()
            preds = [(W[p][f], p) for p in range(NUM_FAMILIES)
                     if W[p][f] > 0 and p in fam_slots and p != f]
            if not preds:
                return
            src = fam_slots[max(preds)[1]]
            rids = [r.receipt_id for r
                    in self.web.slots[src].ledger.receipts
                    if r.kind == 'fit' and r.sign > 0][-1:]
            if rids:
                self.web.constrain(target, rids)
                self.constrains += 1
                self._last_constrain[f] = step

    def process(self, rv, emb, obs, reward, off, ep):
        web = self.web
        web._global_step += 1
        fa = web._obs_to_family_activations(rv)
        fam_slots = {}
        for sid, s in web.slots.items():
            if s.state == 'open' and s.origin_operator == 'inherited':
                fam_slots[family_of(s)] = sid
        if self.staged:
            W = self._edge_weights()
            degree = W.sum(axis=1)
            order = [int(np.argmax(degree))]
            remaining = set(range(NUM_FAMILIES)) - set(order)
            while remaining:
                scores = {f: sum(W[p][f] for p in order)
                          for f in remaining}
                nxt = max(scores, key=lambda f: (scores[f], -f))
                # expectation from the processed prefix only
                wsum = sum(W[p][nxt] for p in order)
                if wsum > 0:
                    pred = sum(W[p][nxt] * fa[p] for p in order) / wsum
                    self._bill_expectation(nxt, pred, fa[nxt],
                                           fam_slots, emb=emb)
                order.append(nxt)
                remaining.discard(nxt)
        results = web.fit_all(rv, emb, obs, obs, reward, off, ep,
                              web._global_step, support_obs=None)
        pos = [sid for sid, rs in results
               if any(r.kind == 'fit' and r.sign > 0 for r in rs)]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                self.cofit[(min(pos[i], pos[j]),
                            max(pos[i], pos[j]))] += 1
        if not self.staged and self.consume:
            # all-at-once accounting: predict each family from ALL
            # others (complete information, no staging order)
            W = self._edge_weights()
            for f in range(NUM_FAMILIES):
                wsum = sum(W[p][f] for p in range(NUM_FAMILIES)
                           if p != f)
                if wsum > 0:
                    pred = sum(W[p][f] * fa[p]
                               for p in range(NUM_FAMILIES)
                               if p != f) / wsum
                    self._bill_expectation(f, pred, fa[f], fam_slots,
                                           emb=emb)

    def endpoints(self):
        web = self.web
        tight, n = web.pop_fresh_tightness()
        cal = []
        for s in web.slots.values():
            if s.state not in ('open', 'closed'):
                continue
            recent = [r for r in s.ledger.receipts
                      if r.kind == 'fit'][-CONFIRM_WINDOW:]
            if len(recent) >= 10:
                confirm = sum(1 for r in recent if r.sign > 0) \
                    / len(recent)
                cal.append(abs(s.ledger.certainty - confirm))
        return {'tightness': tight, 'n_fresh': n,
                'calibration': (float(np.mean(cal)) if cal else None),
                'sharpness': (self.sharp_sum / self.sharp_n
                              if self.sharp_n else None),
                'constrains': self.constrains,
                'edges': len(web.edges)}


def run_worlds(worlds, arms, engine, model, base):
    counter = base
    for w_seed, tier in worlds:
        env = TieredEnvironment(seed=w_seed, tier=tier)
        np.random.seed(w_seed * 7)
        env.rng = np.random.RandomState(w_seed * 7 + 1)
        rng = np.random.RandomState(w_seed * 7 + 2)
        bank = LiveReceptorBank()
        for ep in range(EPISODES):
            org = Organism()
            org.reset()
            for step in range(STEPS):
                w = org.get_observation_window()
                act, _ = model.predict(w)
                r = rng.random()
                if r < PROBE_RATE_FLOOR:
                    act = np.zeros_like(act)
                elif r < EXPLORE_RATE:
                    act = rng.randint(0, 2, size=len(act)).astype(
                        act.dtype)
                obs, reward = org.step(act, env, step)
                rv = bank.compute(obs, act, None, reward)
                emb = engine.encoder.embed(engine._core_obs(obs))
                counter += 1
                for a in arms:
                    a.process(rv, emb, obs, reward, counter, ep)
        # boundary compose scan: top NEW co-fit pairs (the replay
        # drivers' key), up to 3 per boundary so warmup reaches the
        # C20 edge floor
        for a in arms:
            web = a.web
            done = 0
            for (x, y), n in sorted(a.cofit.items(),
                                    key=lambda kv: (-kv[1], kv[0])):
                if done >= 3 or n < 50:
                    break
                if (x, y) in a.composed:
                    continue
                sx, sy = web.slots.get(x), web.slots.get(y)
                if (sx is None or sy is None or sx.state != 'open'
                        or sy.state != 'open'):
                    continue
                if web.compose(x, y)[0] >= 0:
                    a.composed.add((x, y))
                    done += 1
    return counter


def main():
    t0 = time.time()
    print('=== P76: staged fit 2x2 (one stream, four accountants) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arms = [Accountant('P+D', staged=False, consume=False),
            Accountant('P+C', staged=False, consume=True),
            Accountant('S+D', staged=True, consume=False),
            Accountant('S+C', staged=True, consume=True)]

    print('warmup (policies inert: consumption disabled)...')
    saved = [(a.consume,) for a in arms]
    for a in arms:
        a.consume = False           # warmup: all arms identical
    c = run_worlds(WARMUP_WORLDS, arms, engine, model, 0)
    for a, (cons,) in zip(arms, saved):
        a.consume = cons
        a.sharp_sum, a.sharp_n = 0.0, 0        # sharpness measured
        a.web.pop_fresh_tightness()            # post-switch only
    edges0 = len(arms[0].web.edges)
    fits0 = arms[0].web.get_stats()['total_receipts']
    identical = all(len(a.web.edges) == edges0
                    and a.web.get_stats()['total_receipts'] == fits0
                    for a in arms)
    print(f'  switch: edges={edges0} receipts={fits0} '
          f'arms-identical={identical}')
    assert edges0 >= EDGE_FLOOR, 'C20 check 5: warmup edge floor unmet'
    assert identical, 'C20 check 4: warmup states diverged'

    print('treatment (policies live)...')
    run_worlds(TREAT_WORLDS, arms, engine, model, c)

    ep = {a.name: a.endpoints() for a in arms}
    for k, v in ep.items():
        print(f'  {k}: tight={v["tightness"]:.4f} '
              f'cal={v["calibration"]:.4f} '
              f'sharp={v["sharpness"] if v["sharpness"] is None else round(v["sharpness"], 4)} '
              f'constrains={v["constrains"]} edges={v["edges"]}')

    sc, pd, pc = ep['S+C'], ep['P+D'], ep['P+C']
    floors = (all(v['n_fresh'] >= 500 for v in ep.values())
              and sc['constrains'] >= 50)
    if not floors:
        verdict = 'UNTESTED (fit or constrain floors unmet)'
    else:
        beats_pd = (sc['tightness'] < pd['tightness'],
                    sc['calibration'] < pd['calibration'])
        beats_pc = (sc['tightness'] < pc['tightness'],
                    sc['calibration'] < pc['calibration'])
        if all(beats_pd) and any(beats_pc):
            verdict = ('SUPPORTED: serial-consumed beats parallel on '
                       'both endpoints and beats complete-information '
                       'accounting — the staging itself earns')
        elif all(beats_pd):
            verdict = ('PARTIAL: accounting earns, order does not '
                       '(P+C >= S+C) — the thesis holds only at the '
                       'delta-accounting grain')
        elif not any(beats_pd):
            verdict = ('NOT SUPPORTED: serial-consumed <= parallel '
                       'baseline on both endpoints')
        else:
            verdict = 'PARTIAL: one endpoint of two vs baseline'
    print(f'\nP76 VERDICT: {verdict}')

    out = {'endpoints': ep, 'verdict': verdict,
           'warmup_edges': edges0, 'arms_identical_at_switch': identical,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'saved {RESULTS}')


if __name__ == '__main__':
    main()

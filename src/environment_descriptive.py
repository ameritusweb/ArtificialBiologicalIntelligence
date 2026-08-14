"""EL-1 — the descriptive stratum (T153, phase 1).

The environment-organism's first FALSIFIABLE speech: sentences about
inhabitant behavior — "when energy is low the organism seeks endorphin"
— which can be wrong, because behavior is the one thing in the world the
language does not write. Each sentence is a conditional claim
(context frame -> behavior) billed against realized behavior: confirms
and failures are receipts, appended to the etymology ledger. This is the
environment's sole error signal and sole funding source (requirements
doc, Grammar section).

Version-0 grammar, deliberately minimal and auditable:
- BEHAVIORS: a categorical read of a W-step trajectory window
  (flee-pain / seek-endorphin / rest / roam), deterministic thresholds.
- CONTEXTS: boolean frames at window start (energy low/high, near pain,
  near endorphin, predator sweeping, pulse reward phase, hidden state k).
- SENTENCES: every (context, behavior) conditional with a certainty
  learned as a smoothed lived frequency. Prediction for a window uses
  the active context with the most training observations (deterministic
  tie-break) — one sentence-frame speaks at a time, exactly the billing
  grain the requirements name.
- The MARGINAL DESCRIBER (the environment's MarginalPredictor analog)
  predicts the behavior marginal regardless of context; EL-1's acceptance
  is beating it out-of-sample, with failures LOCALIZING to specific
  words (the word-split trigger's input, EL-2's funding source).

No LLMs, no imported corpora; every number is a lived count.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results')

WINDOW = 10
NEAR_DIST = 4.0
MOVE_EPS = 1.0           # path length below which a window is 'rest' (v0.2)
APPROACH_EPS = 0.6       # net distance change that counts as seek/flee
ENERGY_LOW = 0.45
SMOOTH = 1.0             # Laplace smoothing on sentence certainties

BEHAVIORS = ('flees pain', 'seeks endorphin', 'rests', 'roams')


def _nearest_dist(env, sources, x, y, t):
    if not sources:
        return None
    return min(math.hypot(x - s.position(t)[0], y - s.position(t)[1])
               for s in sources)


def window_features(env, xs, ys, t0):
    """Raw behavioral features of a trajectory window, ORGANISM-
    ATTRIBUTED: source positions are frozen at window end, so distance
    deltas measure what the organism did, never what the drifting
    sources did (the v0.2 classifier's conflation, found by the
    kinematics probe)."""
    t1 = t0 + len(xs) - 1
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    path = sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
               for i in range(len(xs) - 1))

    def org_delta(sources):
        if not sources:
            return 0.0
        d_end = min(math.hypot(x1 - s.position(t1)[0],
                               y1 - s.position(t1)[1]) for s in sources)
        d_start = min(math.hypot(x0 - s.position(t1)[0],
                                 y0 - s.position(t1)[1]) for s in sources)
        return d_start - d_end          # positive = organism approached

    return {'path': path,
            'toward_pain': -org_delta(env.pain_sources),
            'toward_end': org_delta(env.endorphin_sources)}


def classify_behavior_from_features(feat, tau_rest, tau_d):
    """Deterministic priority with TRAIN-CALIBRATED, frozen thresholds:
    flee pain > seek endorphin > rest > roam."""
    if feat['toward_pain'] < -tau_d:
        return 'flees pain'
    if feat['toward_end'] > tau_d:
        return 'seeks endorphin'
    if feat['path'] < tau_rest:
        return 'rests'
    return 'roams'


def active_contexts(env, org, t):
    """Boolean context frames at a window's start. Each is a phrase in the
    seed grammar; each is checkable by any observer of the same world."""
    ctx = []
    x, y = org.x, org.y
    energy = getattr(org, 'energy', 1.0)
    ctx.append('when energy is low' if energy < ENERGY_LOW
               else 'when energy is high')
    d_pain = _nearest_dist(env, env.pain_sources, x, y, t)
    if d_pain is not None and d_pain < NEAR_DIST:
        ctx.append('near a pain source')
    d_end = _nearest_dist(env, env.endorphin_sources, x, y, t)
    if d_end is not None and d_end < NEAR_DIST:
        ctx.append('near an endorphin source')
    for p in getattr(env, 'predator_events', []):
        cycle = t % p.period
        if cycle < p.duration:
            ctx.append('while a predator sweeps')
            break
    for ph in getattr(env, 'anticipatory_phases', []):
        phase = (t % ph['period']) / ph['period']
        if ph['reward_phase_start'] <= phase <= ph['reward_phase_end']:
            ctx.append('in the reward phase of the pulse')
            break
    hv = getattr(env, 'hidden_variable', None)
    if hv is not None:
        ctx.append(f'under hidden state {hv.state(t)}')
    return ctx


class DescriptiveStratum:
    """The environment's conditional claims about its inhabitants, with
    lived-count certainties and per-sentence receipts."""

    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(float))  # ctx->beh->n
        self.ctx_totals = defaultdict(float)
        self.marginal = defaultdict(float)
        self.total = 0.0
        # Billing receipts (test-time): sentence -> [confirms, occasions]
        self.receipts = defaultdict(lambda: [0, 0])

    # -- learning (lived counting) ------------------------------------
    def observe(self, contexts, behavior):
        for c in contexts:
            self.counts[c][behavior] += 1.0
            self.ctx_totals[c] += 1.0
        self.marginal[behavior] += 1.0
        self.total += 1.0

    # -- speaking (the sentences) -------------------------------------
    def sentences(self, min_occasions=20):
        out = []
        for c, beh_counts in sorted(self.counts.items()):
            if self.ctx_totals[c] < min_occasions:
                continue
            for b in BEHAVIORS:
                p = ((beh_counts.get(b, 0.0) + SMOOTH)
                     / (self.ctx_totals[c] + SMOOTH * len(BEHAVIORS)))
                out.append((f"{c} the organism {b}", c, b, p))
        return out

    # -- prediction (one sentence-frame speaks) -----------------------
    def _ctx_kl(self, c):
        """Train-side informativeness: KL(P(B|c) || P(B)) from lived
        counts only — the context whose sentences SAY the most speaks."""
        kl = 0.0
        for b in BEHAVIORS:
            pc = ((self.counts[c].get(b, 0.0) + SMOOTH)
                  / (self.ctx_totals[c] + SMOOTH * len(BEHAVIORS)))
            pm = self.p_marginal(b)
            kl += pc * math.log(pc / pm)
        return kl

    def _best_context(self, contexts, min_occ=20):
        known = [c for c in contexts
                 if self.ctx_totals.get(c, 0) >= min_occ]
        if not known:
            known = [c for c in contexts if self.ctx_totals.get(c, 0) > 0]
        if not known:
            return None
        return max(known, key=lambda c: (self._ctx_kl(c),
                                         self.ctx_totals[c], c))

    def p_conditional(self, contexts, behavior):
        c = self._best_context(contexts)
        if c is None:
            return self.p_marginal(behavior)
        return ((self.counts[c].get(behavior, 0.0) + SMOOTH)
                / (self.ctx_totals[c] + SMOOTH * len(BEHAVIORS)))

    def p_marginal(self, behavior):
        return ((self.marginal.get(behavior, 0.0) + SMOOTH)
                / (self.total + SMOOTH * len(BEHAVIORS)))

    # -- billing (the error signal) -----------------------------------
    def bill(self, contexts, behavior):
        """Test-time receipts: for the speaking context, its four
        sentences get an occasion; the one naming the realized behavior
        confirms; the argmax sentence is the environment's CLAIM and
        fails when a different behavior realizes."""
        c = self._best_context(contexts)
        if c is None:
            return
        claimed = max(BEHAVIORS,
                      key=lambda b: (self.counts[c].get(b, 0.0), b))
        sent = f"{c} the organism {claimed}"
        self.receipts[sent][1] += 1
        if claimed == behavior:
            self.receipts[sent][0] += 1

    def word_failures(self, min_occasions=50):
        """Failure mass aggregated to words — the split trigger's input."""
        word_fail = defaultdict(float)
        word_occ = defaultdict(float)
        for sent, (conf, occ) in self.receipts.items():
            if occ <= 0:
                continue
            for w in set(sent.split()):
                word_fail[w] += (occ - conf)
                word_occ[w] += occ
        rates = {w: word_fail[w] / word_occ[w]
                 for w in word_occ if word_occ[w] >= min_occasions}
        return rates


def append_descriptions_to_ledger(stratum, path=None):
    path = path or os.path.join(RESULTS_DIR, 'el0_etymology.jsonl')
    with open(path, 'a', encoding='utf-8') as fh:
        for sent, c, b, p in stratum.sentences():
            conf, occ = stratum.receipts.get(
                f"{c} the organism "
                f"{max(BEHAVIORS, key=lambda x: (stratum.counts[c].get(x, 0.0), x))}",
                (0, 0))
            fh.write(json.dumps({
                'event': 'description_billed', 'sentence': sent,
                'certainty': round(p, 4),
                'claim_confirms': conf, 'claim_occasions': occ,
            }) + '\n')
    return path

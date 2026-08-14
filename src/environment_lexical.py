"""EL-2 — lexical evolution under receipts (T153, phase 2).

The language revises itself, funded only by its own failed descriptions:

- WORD SPLIT (de-conflation): the word carrying the most failure mass is
  offered conjunction refinements ("under hidden state 0" -> "under
  hidden state 0 when energy is low" / "... when energy is high"); a
  split is ADMITTED only if it reduces held-out NLL on that word's own
  occasions by a pre-registered margin and both children live (occasion
  floors). Polysemy is conflation at the language level.
- WORD MERGE (dead distinctions): two words whose lived behavior
  distributions are indistinguishable (train JSD below floor) merge,
  provided held-out NLL does not degrade. Synonymy is a distinction no
  inhabitant ever lives.
- WORD BIRTH (residual-funded): when the residual (windows every current
  word describes poorly) persists, an unworded predicate from the FORM
  POOL may be named — admitted only if it reduces held-out NLL. The
  forms are fixed; the words are emergent (Anti-Requirements).
- CADENCE: moves happen between generations, never within one; each
  generation lives under a FIXED lexicon (timescale separation).
- ETYMOLOGY: every move appends its receipts — the failed descriptions
  that funded it, the NLL deltas, the occasions. No move by taste.

No LLMs, no imported corpora, no imitation funding: the only currency is
failed description of lived inhabitant behavior.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

from environment_descriptive import (BEHAVIORS, SMOOTH, WINDOW,
                                     window_features,
                                     classify_behavior_from_features,
                                     _nearest_dist)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results')

NEAR_DIST = 4.0
ENERGY_LOW = 0.45

# Pre-registered move rules (fixed before any generation runs)
SPLIT_MIN_OCC = 60          # parent occasions needed to consider a split
SPLIT_CHILD_MIN = 15        # child AND remaining-parent must both live
SPLIT_NLL_GAIN = 0.02       # nats, on the parent word's own occasions,
                            # validated on WORLD-HELD-OUT corpora (v2)
SPLIT_MAX_PARTS = 1         # only refine single-conjunct words (v2:
                            # one refinement per word — no chains)
MERGE_JSD_MAX = 0.01        # distributional indistinguishability floor
MERGE_JACCARD_MIN = 0.9     # v2: TRUE synonymy requires co-extension —
                            # words that fire together, not words whose
                            # different situations merely behave alike
MERGE_MIN_OCC = 40
MERGE_NLL_TOL = 0.005       # held-out degradation tolerance
BIRTH_NLL_GAIN = 0.01       # corpus-wide held-out gain required
RESIDUAL_NLL = 1.2          # a window counts as residual above this

# v3 — THE JUNCTION LAW APPLIED TO WORDS: a move is PROPOSED from one
# generation's receipts and RATIFIED only if its gain survives the NEXT
# generation's entirely fresh worlds (selection on batch one,
# confirmation on batch two — single-candidate, unbiased; the winner's
# curse dies at the border). Unratified proposals are rejected with
# their receipts logged. The lexicon is the slower stratum; nothing
# crosses into it on one generation's evidence.
RATIFY_SPLIT_GAIN = 0.01    # parent-occasion nats on the fresh worlds
RATIFY_BIRTH_GAIN = 0.005   # corpus-wide nats on the fresh worlds


# ---------------------------------------------------------------------------
# Base predicates (the form space). Initially WORDED ones seed the
# lexicon; POOL ones exist as forms but have no word until birth names
# them. The DECOY duplicates near_pain under another name — the merge
# validity check must find and merge it.
# ---------------------------------------------------------------------------

def _dist_point_seg(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def base_predicates(env, org, t):
    """Evaluate ALL base predicates (worded, pool, decoy) at a window
    start. Returns the set of active predicate ids."""
    out = set()
    x, y = org.x, org.y
    energy = getattr(org, 'energy', 1.0)
    out.add('when energy is low' if energy < ENERGY_LOW
            else 'when energy is high')
    d_pain = _nearest_dist(env, env.pain_sources, x, y, t)
    near_pain = d_pain is not None and d_pain < NEAR_DIST
    if near_pain:
        out.add('near a pain source')
        out.add('beside a hurting field')          # DECOY duplicate
    d_end = _nearest_dist(env, env.endorphin_sources, x, y, t)
    if d_end is not None and d_end < NEAR_DIST:
        out.add('near an endorphin source')
    for p in getattr(env, 'predator_events', []):
        if (t % p.period) < p.duration:
            out.add('while a predator sweeps')
            break
    for ph in getattr(env, 'anticipatory_phases', []):
        phase = (t % ph['period']) / ph['period']
        if ph['reward_phase_start'] <= phase <= ph['reward_phase_end']:
            out.add('in the reward phase of the pulse')
            break
    hv = getattr(env, 'hidden_variable', None)
    if hv is not None:
        out.add(f'under hidden state {hv.state(t)}')
    # ---- FORM POOL (unworded until birth) ----
    for b in getattr(env, 'barriers', []):
        if _dist_point_seg(x, y, b.x1, b.y1, b.x2, b.y2) < 2.5:
            out.add('close to a wall')
            break
    for m in getattr(env, 'movable_objects', []):
        if math.hypot(x - m.x, y - m.y) < 2.5:
            out.add('beside a movable thing')
            break
    return out


SEED_WORDS = ['when energy is low', 'when energy is high',
              'near a pain source', 'beside a hurting field',
              'near an endorphin source', 'while a predator sweeps',
              'in the reward phase of the pulse',
              'under hidden state 0', 'under hidden state 1',
              'under hidden state 2']
FORM_POOL = ['close to a wall', 'beside a movable thing']


# ---------------------------------------------------------------------------
# The lexicon: words are conjunctions of base predicate ids (1 or 2).
# ---------------------------------------------------------------------------

class Lexicon:
    def __init__(self, words=None):
        # word name -> frozenset of base predicate ids (all must hold)
        self.words = {w: frozenset([w]) for w in (words or SEED_WORDS)}
        self.merged_into = {}     # old word -> surviving word

    def copy(self):
        lx = Lexicon([])
        lx.words = dict(self.words)
        lx.merged_into = dict(self.merged_into)
        return lx

    def active_words(self, predicate_set):
        return [w for w, parts in self.words.items()
                if parts <= predicate_set]

    def split(self, word, discriminator):
        """word -> (word ∧ D, word ∧ ¬D is expressed as word ∧ other-D
        where base predicates are exhaustive per axis; here we pair with
        the discriminator and its observed complement words)."""
        parts = self.words[word]
        child = f'{word} {discriminator}'
        self.words[child] = parts | frozenset([discriminator])
        return child

    def merge(self, keep, drop):
        del self.words[drop]
        self.merged_into[drop] = keep

    def birth(self, form_id):
        self.words[form_id] = frozenset([form_id])


# ---------------------------------------------------------------------------
# Scoring a lexicon over a corpus of raw windows
# ---------------------------------------------------------------------------

class WordModel:
    """Per-lexicon conditional describer (the EL-1 machinery, keyed by
    lexicon words), trained by counting, scored by NLL, with per-word
    failure receipts."""

    def __init__(self, lexicon):
        self.lx = lexicon
        self.counts = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(float)
        self.marginal = defaultdict(float)
        self.total = 0.0

    def train(self, corpus):
        for preds, beh in corpus:
            for w in self.lx.active_words(preds):
                self.counts[w][beh] += 1.0
                self.totals[w] += 1.0
            self.marginal[beh] += 1.0
            self.total += 1.0
        return self

    def p_marginal(self, b):
        return ((self.marginal.get(b, 0.0) + SMOOTH)
                / (self.total + SMOOTH * len(BEHAVIORS)))

    def _kl(self, w):
        kl = 0.0
        for b in BEHAVIORS:
            pc = ((self.counts[w].get(b, 0.0) + SMOOTH)
                  / (self.totals[w] + SMOOTH * len(BEHAVIORS)))
            kl += pc * math.log(pc / self.p_marginal(b))
        return kl

    def speaking_word(self, preds, min_occ=20):
        active = [w for w in self.lx.active_words(preds)
                  if self.totals.get(w, 0) >= min_occ]
        if not active:
            active = [w for w in self.lx.active_words(preds)
                      if self.totals.get(w, 0) > 0]
        if not active:
            return None
        # Prefer more specific (larger conjunction), then informativeness.
        return max(active, key=lambda w: (len(self.lx.words[w]),
                                          self._kl(w), self.totals[w], w))

    def p_word(self, w, b):
        return ((self.counts[w].get(b, 0.0) + SMOOTH)
                / (self.totals[w] + SMOOTH * len(BEHAVIORS)))

    def nll(self, corpus, only_word=None):
        s, n = 0.0, 0
        for preds, beh in corpus:
            w = self.speaking_word(preds)
            if only_word is not None and w != only_word:
                continue
            p = self.p_word(w, beh) if w else self.p_marginal(beh)
            s += -math.log(p)
            n += 1
        return (s / n if n else None), n

    def failure_mass(self, corpus):
        """Per speaking word: (fails, occasions) — the split trigger."""
        rec = defaultdict(lambda: [0, 0])
        for preds, beh in corpus:
            w = self.speaking_word(preds)
            if w is None:
                continue
            claimed = max(BEHAVIORS,
                          key=lambda b: (self.counts[w].get(b, 0.0), b))
            rec[w][1] += 1
            if claimed != beh:
                rec[w][0] += 1
        return rec

    def dist(self, w):
        return tuple(self.p_word(w, b) for b in BEHAVIORS)


def _jsd(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    def _kl(a, b):
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ---------------------------------------------------------------------------
# One generation of lexical evolution (between-generation moves only)
# ---------------------------------------------------------------------------

def ratify_pending(lexicon, pending, train_corpus, val_corpus, ledger):
    """The junction-law gate (v3): test last generation's proposals on
    THIS generation's fresh worlds. Single candidate each — unbiased.
    Returns (lexicon, ratified_moves)."""
    lx = lexicon
    ratified = []
    for p in pending:
        if p['kind'] == 'split':
            w, d = p['word'], p['discriminator']
            if w not in lx.words or d in lx.merged_into:
                ledger.append({'event': 'split_rejected', 'word': w,
                               'discriminator': d,
                               'receipts': {'reason': 'word gone'}})
                continue
            base_model = WordModel(lx).train(train_corpus)
            base_nll, n0 = base_model.nll(val_corpus, only_word=w)
            trial = lx.copy()
            child = trial.split(w, d)
            tm = WordModel(trial).train(train_corpus)
            nll_new = _nll_on_parent_occasions(tm, base_model, w,
                                               val_corpus)
            gain = ((base_nll - nll_new)
                    if (base_nll is not None and nll_new is not None)
                    else None)
            if (gain is not None and gain >= RATIFY_SPLIT_GAIN
                    and tm.totals.get(child, 0) >= SPLIT_CHILD_MIN
                    and tm.totals.get(w, 0) >= SPLIT_CHILD_MIN):
                lx = trial
                ledger.append({'event': 'word_split_ratified', 'word': w,
                               'child': child,
                               'receipts': {'proposal_gain':
                                            p['gain'],
                                            'ratification_gain':
                                            round(gain, 4)}})
                ratified.append(('split', w, child))
            else:
                ledger.append({'event': 'split_rejected', 'word': w,
                               'discriminator': d,
                               'receipts': {'proposal_gain': p['gain'],
                                            'ratification_gain':
                                            (round(gain, 4)
                                             if gain is not None
                                             else None)}})
        elif p['kind'] == 'birth':
            form = p['word']
            if form in lx.words:
                continue
            base_nll, _ = WordModel(lx).train(train_corpus).nll(val_corpus)
            trial = lx.copy()
            trial.birth(form)
            nll_new, _ = WordModel(trial).train(train_corpus).nll(
                val_corpus)
            gain = base_nll - nll_new
            if gain >= RATIFY_BIRTH_GAIN:
                lx = trial
                ledger.append({'event': 'word_birth_ratified',
                               'word': form,
                               'receipts': {'proposal_gain': p['gain'],
                                            'ratification_gain':
                                            round(gain, 4)}})
                ratified.append(('birth', form, None))
            else:
                ledger.append({'event': 'birth_rejected', 'word': form,
                               'receipts': {'proposal_gain': p['gain'],
                                            'ratification_gain':
                                            round(gain, 4)}})
    return lx, ratified


def evolve_one_generation(lexicon, train_corpus, val_corpus, ledger):
    """Propose at most one split and one birth (PENDING, ratified next
    generation under the junction law); admit at most one merge
    immediately (co-extension synonyms are deletion of duplicates —
    tolerance-checked, low risk). Receipts-only throughout.
    Returns (lexicon, immediate_moves, pending_proposals)."""
    lx = lexicon.copy()
    model = WordModel(lx).train(train_corpus)
    moves = []
    pending = []

    # --- SPLIT: the word with the most failure mass (v2: refine only
    # single-conjunct words; discriminators exclude merged-away names;
    # both child and remaining parent must live) ---
    fails = model.failure_mass(train_corpus)
    candidates = sorted(
        ((f, occ, w) for w, (f, occ) in fails.items()
         if occ >= SPLIT_MIN_OCC
         and len(lx.words.get(w, frozenset())) <= SPLIT_MAX_PARTS),
        key=lambda x: (-x[0], x[2]))
    all_bases = (set(SEED_WORDS) | set(FORM_POOL)) - set(lx.merged_into)
    for f, occ, w in candidates[:3]:
        base_nll, _ = model.nll(val_corpus, only_word=w)
        if base_nll is None:
            continue
        best = None
        for d in sorted(all_bases - lx.words[w]):
            trial = lx.copy()
            child = trial.split(w, d)
            tm = WordModel(trial).train(train_corpus)
            if (tm.totals.get(child, 0) < SPLIT_CHILD_MIN
                    or tm.totals.get(w, 0) < SPLIT_CHILD_MIN):
                continue
            # score the parent's former occasions under the refined lexicon
            nll_new = _nll_on_parent_occasions(tm, model, w, val_corpus)
            if nll_new is None:
                continue
            gain = base_nll - nll_new
            if best is None or gain > best[0]:
                best = (gain, d, child)
        if best and best[0] >= SPLIT_NLL_GAIN:
            gain, d, child = best
            ledger.append({
                'event': 'word_split_proposed', 'word': w,
                'discriminator': d,
                'receipts': {'failures': f, 'occasions': occ,
                             'val_nll_gain': round(gain, 4)}})
            pending.append({'kind': 'split', 'word': w,
                            'discriminator': d, 'gain': round(gain, 4)})
            break

    # --- MERGE: TRUE synonyms only (v2) — co-extension (they fire
    # together) AND distributional identity AND non-degrading held-out.
    # Different situations that merely behave alike stay distinct words.
    model = WordModel(lx).train(train_corpus)
    words = [w for w in lx.words if model.totals.get(w, 0) >= MERGE_MIN_OCC]
    ext = {w: set() for w in words}
    for idx, (preds, _beh) in enumerate(train_corpus):
        for w in words:
            if lx.words[w] <= preds:
                ext[w].add(idx)
    best_pair = None
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a, b = words[i], words[j]
            inter = len(ext[a] & ext[b])
            union = len(ext[a] | ext[b])
            if union == 0 or inter / union < MERGE_JACCARD_MIN:
                continue
            d = _jsd(model.dist(a), model.dist(b))
            if d < MERGE_JSD_MAX and (best_pair is None
                                      or d < best_pair[0]):
                best_pair = (d, a, b)
    if best_pair:
        d, a, b = best_pair
        keep, drop = (a, b) if len(a) <= len(b) else (b, a)
        base_nll, _ = model.nll(val_corpus)
        trial = lx.copy()
        trial.merge(keep, drop)
        nll_merged, _ = WordModel(trial).train(train_corpus).nll(val_corpus)
        if nll_merged is not None and nll_merged <= base_nll + MERGE_NLL_TOL:
            lx = trial
            ledger.append({
                'event': 'word_merge', 'kept': keep, 'dropped': drop,
                'receipts': {'train_jsd': round(d, 6),
                             'val_nll_delta': round(nll_merged - base_nll,
                                                    5)}})
            moves.append(('merge', keep, drop))

    # --- BIRTH: residual-funded naming from the form pool ---
    model = WordModel(lx).train(train_corpus)
    unworded = [f for f in FORM_POOL if f not in lx.words]
    if unworded:
        resid = 0
        for preds, beh in val_corpus:
            w = model.speaking_word(preds)
            p = model.p_word(w, beh) if w else model.p_marginal(beh)
            if -math.log(p) > RESIDUAL_NLL:
                resid += 1
        if resid >= 30:
            base_nll, _ = model.nll(val_corpus)
            best = None
            for form in unworded:
                trial = lx.copy()
                trial.birth(form)
                nll_new, _ = WordModel(trial).train(
                    train_corpus).nll(val_corpus)
                if nll_new is None:
                    continue
                gain = base_nll - nll_new
                if best is None or gain > best[0]:
                    best = (gain, form)
            if best and best[0] >= BIRTH_NLL_GAIN:
                gain, form = best
                ledger.append({
                    'event': 'word_birth_proposed', 'word': form,
                    'receipts': {'residual_windows': resid,
                                 'val_nll_gain': round(gain, 4)}})
                pending.append({'kind': 'birth', 'word': form,
                                'gain': round(gain, 4)})

    return lx, moves, pending


def _nll_on_parent_occasions(trial_model, base_model, parent, corpus):
    """Held-out NLL restricted to windows where the ORIGINAL lexicon's
    speaking word was the parent — the split is billed on the parent's
    own occasions, nothing else."""
    s, n = 0.0, 0
    for preds, beh in corpus:
        if base_model.speaking_word(preds) != parent:
            continue
        w = trial_model.speaking_word(preds)
        p = (trial_model.p_word(w, beh) if w
             else trial_model.p_marginal(beh))
        s += -math.log(p)
        n += 1
    return (s / n) if n else None


def append_ledger(entries, path=None):
    path = path or os.path.join(RESULTS_DIR, 'el0_etymology.jsonl')
    with open(path, 'a', encoding='utf-8') as fh:
        for e in entries:
            fh.write(json.dumps(e) + '\n')
    return path

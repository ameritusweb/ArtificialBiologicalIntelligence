"""EL-2.5 — the living-language loop (T153: generation as utterance).

The coupling the four-finding convergence (F13/F15/F16/F17) demanded:
world generation moves INTO the language, and the language's growth
changes the worlds. The loop:

  corpus (world spoken in EL-0 sentences)
    -> interpret() builds the world            [EL-0, the effector]
    -> inhabitants live in it                  [the one thing not written]
    -> the descriptive stratum bills sentences [EL-1, the error signal]
    -> DESCRIBABLE SLACK funds world-sentences [this module]
    -> richer worlds -> the EL-2 court re-runs [the ratchet test]

**The funding direction (D1-symmetric, pre-registered):** a world's
fitness is its DESCRIBABLE SLACK — the gap between marginal and
conditional NLL of inhabitant behavior under the FIXED seed lexicon:
  slack = NLL_marginal - NLL_conditional
Structure that exists AND is conditionally capturable = learnable
surprise produced. The two degenerate directions are both unprofitable
by construction: a trivial world (behavior marginal-predictable) has a
small gap; a noise world (nothing conditionally capturable) has a small
gap. The mirror trap's generation-side sibling — writing worlds where
your own sentences are cheaply right — is unfundable, because accuracy
per se earns nothing; only CONDITIONAL structure beyond the marginal
does. An entropy floor (no behavior class > 0.9) guards the remaining
corner.

**Mutation is utterance:** world-genotypes are corpora; mutations are
new sentences (add a source / trigger / barrier / cross-modal source;
perturb a source's place) — v0 is append-and-perturb only, preserving
referential integrity (pulse/depletable index references never break).
Every kept mutation carries its fitness receipt into the etymology.

**Timescale separation:** within a language generation the worlds are
fixed; selection is editorial (complexity pressure), never mortal.
"""

import json
import math
import re

import numpy as np

from environment_language import describe, interpret, _f, NUM
from environment_descriptive import (window_features,
                                     classify_behavior_from_features,
                                     WINDOW, BEHAVIORS)
from environment_lexical import (Lexicon, WordModel, base_predicates,
                                 append_ledger)
from environment_tiers import TieredEnvironment
from environment import Organism
from train import EXPLORE_RATE, PROBE_RATE_FLOOR

TIER = 4
EPISODES_PER_EVAL = 3
STEPS = 400
MIN_WINDOWS_PER_WORLD = 90
ENTROPY_CEILING = 0.9        # no behavior class may exceed this share


# ---------------------------------------------------------------------------
# Mutation operators — utterances (append/perturb only, v0)
# ---------------------------------------------------------------------------

def _say_new_source(kind, rng):
    return (f"a {kind} source at ({_f(rng.uniform(3, 17))}, "
            f"{_f(rng.uniform(3, 17))}) drifts by "
            f"({_f(rng.uniform(2.0, 4.0))}, {_f(rng.uniform(2.0, 4.0))}) "
            f"with frequency ({_f(rng.uniform(0.01, 0.04))}, "
            f"{_f(rng.uniform(0.01, 0.04))}) phase "
            f"({_f(rng.uniform(0, 2 * math.pi))}, "
            f"{_f(rng.uniform(0, 2 * math.pi))}) spread "
            f"{_f(2.5 if kind == 'pain' else 3.0)} intensity "
            f"{_f(1.0 if kind == 'pain' else 1.5)}")


def _say_new_trigger(rng):
    return (f"touching ({_f(rng.uniform(3, 17))}, "
            f"{_f(rng.uniform(3, 17))}) within {_f(3.0)} causes "
            f"{'endorphin' if rng.random() < 0.7 else 'pain'} at "
            f"({_f(rng.uniform(3, 17))}, {_f(rng.uniform(3, 17))}) after "
            f"{int(rng.randint(10, 25))} steps with intensity "
            f"{_f(rng.uniform(1.0, 2.0))} for "
            f"{int(rng.randint(20, 40))} steps")


def _say_new_barrier(rng):
    cx, cy = rng.uniform(5, 15), rng.uniform(5, 15)
    ang = rng.uniform(0, math.pi)
    L = rng.uniform(3.0, 6.0)
    x1, y1 = cx - L / 2 * math.cos(ang), cy - L / 2 * math.sin(ang)
    x2, y2 = cx + L / 2 * math.cos(ang), cy + L / 2 * math.sin(ang)
    caps = ['sight and field and movement', 'sight and movement',
            'field'][rng.randint(0, 3)]
    att = {'sight and field and movement': 0.8, 'sight and movement': 0.0,
           'field': 0.6}[caps]
    return (f"a barrier from ({_f(x1)}, {_f(y1)}) to ({_f(x2)}, "
            f"{_f(y2)}) blocks {caps} attenuating {_f(att)}")


def _say_new_crossmodal(rng):
    return (f"a cross-modal source at ({_f(rng.uniform(4, 16))}, "
            f"{_f(rng.uniform(4, 16))}) spread {_f(2.5)} modulates pain "
            f"{_f(rng.uniform(0.3, 1.0))} temperature "
            f"{_f(rng.uniform(0.5, 1.5))} chemical "
            f"{_f(rng.uniform(0.3, 1.2))}")


_SRC_AT = re.compile(rf'^(a (?:pain|endorphin|heat|cold|chemical) source '
                     rf'at )\({NUM}, {NUM}\)(.*)$')


def _perturb_source(corpus, rng):
    idxs = [i for i, s in enumerate(corpus) if _SRC_AT.match(s)]
    if not idxs:
        return None
    i = idxs[rng.randint(0, len(idxs))]
    m = _SRC_AT.match(corpus[i])
    nx = float(np.clip(float(m.group(2)) + rng.uniform(-3, 3), 2, 18))
    ny = float(np.clip(float(m.group(3)) + rng.uniform(-3, 3), 2, 18))
    out = list(corpus)
    out[i] = f"{m.group(1)}({_f(nx)}, {_f(ny)}){m.group(4)}"
    return out, f'moved source {i}'


MUTATIONS = ('add_endorphin', 'add_pain', 'add_trigger', 'add_barrier',
             'add_crossmodal', 'perturb_source')


def mutate(corpus, rng):
    """One utterance: returns (new_corpus, description)."""
    op = MUTATIONS[rng.randint(0, len(MUTATIONS))]
    if op == 'perturb_source':
        r = _perturb_source(corpus, rng)
        if r is not None:
            return r[0], r[1]
        op = 'add_trigger'
    if op == 'add_endorphin':
        return corpus + [_say_new_source('endorphin', rng)], op
    if op == 'add_pain':
        return corpus + [_say_new_source('pain', rng)], op
    if op == 'add_trigger':
        return corpus + [_say_new_trigger(rng)], op
    if op == 'add_barrier':
        return corpus + [_say_new_barrier(rng)], op
    return corpus + [_say_new_crossmodal(rng)], op


# ---------------------------------------------------------------------------
# Living in a spoken world: inhabitants + windows
# ---------------------------------------------------------------------------

def live_in(corpus, model, episode_seed):
    """Build the world from its corpus; run policy inhabitants; return
    raw windows (predicate set, features, episode)."""
    env = interpret(corpus)
    np.random.seed(episode_seed)
    env.rng = np.random.RandomState(episode_seed + 1)
    rng = np.random.RandomState(episode_seed + 2)
    records = []
    for ep in range(EPISODES_PER_EVAL):
        org = Organism()
        org.reset()
        xs, ys = [], []
        preds0 = None
        for step in range(STEPS):
            if step % WINDOW == 0:
                if len(xs) == WINDOW and preds0 is not None:
                    records.append((preds0,
                                    window_features(env, xs, ys,
                                                    step - WINDOW), ep))
                xs, ys = [], []
                preds0 = base_predicates(env, org, step)
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
    return records


def describable_slack(records, tau_rest, tau_d, lexicon=None):
    """The funding signal: NLL_marginal - NLL_conditional on an
    episode-held-out split, under the FIXED lexicon. Returns
    (slack, max_class_share, n_val) — with the entropy guard's
    ingredient reported alongside."""
    lex = lexicon or Lexicon()
    corpus = [(p, classify_behavior_from_features(f, tau_rest, tau_d), ep)
              for p, f, ep in records]
    if len(corpus) < MIN_WINDOWS_PER_WORLD:
        return None, None, len(corpus)
    eps = sorted({ep for _, _, ep in corpus})
    val_ep = eps[-1]
    train = [(p, b) for p, b, ep in corpus if ep != val_ep]
    val = [(p, b) for p, b, ep in corpus if ep == val_ep]
    if len(val) < 20:
        return None, None, len(corpus)
    m = WordModel(lex).train(train)
    s_c, s_m, n = 0.0, 0.0, 0
    for preds, beh in val:
        w = m.speaking_word(preds)
        pc = m.p_word(w, beh) if w else m.p_marginal(beh)
        s_c += -math.log(pc)
        s_m += -math.log(m.p_marginal(beh))
        n += 1
    counts = {b: 0 for b in BEHAVIORS}
    for _, b in train + val:
        counts[b] += 1
    total = sum(counts.values())
    max_share = max(counts.values()) / total if total else 1.0
    return (s_m - s_c) / n, max_share, len(corpus)


# ---------------------------------------------------------------------------
# The generational loop (selection is editorial, never mortal)
# ---------------------------------------------------------------------------

def evolve_worlds(seed_corpora, model, tau_rest, tau_d, generations,
                  ledger, base_seed=50000):
    """Population of world-corpora under describable-slack pressure.
    Deterministic. Returns the final population with fitness history."""
    pop = [list(c) for c in seed_corpora]
    P = len(pop)
    history = []
    for gen in range(generations):
        fits = []
        for i, corpus in enumerate(pop):
            slack, max_share, n = describable_slack(
                live_in(corpus, model, base_seed + gen * 1000 + i * 10),
                tau_rest, tau_d)
            ok = (slack is not None and max_share is not None
                  and max_share <= ENTROPY_CEILING)
            fits.append((slack if ok else -1.0, i, max_share, n))
        fits.sort(key=lambda x: (-x[0], x[1]))
        survivors = [pop[i] for _, i, _, _ in fits[:P // 2]]
        mean_fit = float(np.mean([f for f, _, _, _ in fits
                                  if f > -1.0])) if fits else 0.0
        history.append({'gen': gen, 'mean_slack': round(mean_fit, 4),
                        'best_slack': round(fits[0][0], 4),
                        'worlds_ok': sum(1 for f, _, _, _ in fits
                                         if f > -1.0)})
        # Refill by mutation (utterance), receipts logged
        rng = np.random.RandomState(base_seed + 777 + gen)
        children = []
        for k, parent in enumerate(survivors):
            child, op = mutate(parent, rng)
            children.append(child)
            ledger.append({'event': 'world_utterance', 'gen': gen,
                           'op': op,
                           'receipts': {
                               'parent_rank': k,
                               'parent_slack': round(fits[k][0], 4),
                               'sentences': len(child)}})
        pop = survivors + children
    return pop, history

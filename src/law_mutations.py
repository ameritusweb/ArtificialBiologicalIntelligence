"""The LAWS stratum of the mutation vocabulary (F26 impl. 1).

EL-2.5's original mutation operators edit the world's FURNITURE (add a
source, add a barrier, move a thing). F26 showed no furniture dose can
falsify a rule-account: `formalization`'s content is the world's RULES,
and the corridor law says churn requires change at the K's own depth,
delivered reachably. These operators change the DYNAMICS PARAMETERS of
existing law-bearing sentences — trigger periods, sweep timing, hidden-
state cycling, pulse timing — while preserving every address in the
world (nothing added, nothing removed, nothing moved). T153's decree
stratum, made mutable at the timing-law layer.

Each operator rewrites one sentence in place. Deterministic under the
caller's rng. Returns (corpus, description); description
'no-law-line' when the corpus holds no matching law sentence (counted
by callers, never silent — C20 no-silent-caps).
"""

import re

import numpy as np

_LAWS = (
    ('retime_hidden',
     re.compile(r'^(a hidden state cycles through \d+ states with '
                r'period )(\d+)$'),
     lambda v, rng: max(10, int(v * (2.0 if rng.random() < 0.5
                                     else 0.5)))),
    ('retime_predator',
     re.compile(r'^(a predator sweeps every )(\d+)( steps for \d+ '
                r'steps.*)$'),
     lambda v, rng: max(10, int(v * (2.0 if rng.random() < 0.5
                                     else 0.5)))),
    ('retime_pulse',
     re.compile(r'^(the pain source \d+ pulses with period )(\d+)'
                r'( and amplitude.*)$'),
     lambda v, rng: max(5, int(v * (2.0 if rng.random() < 0.5
                                    else 0.5)))),
    ('reswap',
     re.compile(r'^(the hidden configuration swaps at step )(\d+)$'),
     lambda v, rng: max(20, int(v * (1.5 if rng.random() < 0.5
                                     else 0.66)))),
)


def mutate_law(corpus, rng):
    """Rewrite ONE law parameter in one law-bearing sentence."""
    candidates = []
    for i, line in enumerate(corpus):
        for name, pat, fn in _LAWS:
            m = pat.match(line)
            if m:
                candidates.append((i, name, pat, fn, m))
    if not candidates:
        return list(corpus), 'no-law-line'
    i, name, pat, fn, m = candidates[rng.randint(len(candidates))]
    groups = m.groups()
    old = int(groups[1])
    new = fn(old, rng)
    if new == old:
        new = old * 2
    if len(groups) == 3:
        line2 = f'{groups[0]}{new}{groups[2]}'
    else:
        line2 = f'{groups[0]}{new}'
    out = list(corpus)
    out[i] = line2
    return out, f'{name}:{old}->{new}'

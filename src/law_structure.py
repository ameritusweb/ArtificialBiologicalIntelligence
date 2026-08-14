"""Law-STRUCTURE mutations — the decree stratum's grammar changes
(F27 impl. 1 exit (a); the depth rung above law-parameters).

law_mutations.py re-times existing laws (parameters). These operators
change WHAT KINDS of laws exist: a law is born (an existing thing
acquires dynamics it never had), a law dies (the dynamics vanish, the
thing remains), or a law changes SPECIES (periodic becomes stochastic).
Addresses are preserved wherever a subject exists (pulses attach to
existing pain sources; hidden-state laws have no spatial address to
lose). Every produced sentence matches interpret()'s grammar exactly.

Operators:
  birth_pulse     — a non-pulsing pain source starts pulsing
  kill_pulse      — a pulsing source's pulse law is repealed
  birth_hidden    — the world acquires a periodic hidden state
  kill_hidden     — the hidden-state law is repealed
  species_swap    — periodic hidden ('cycles') -> stochastic
                    ('wanders'), or the reverse
  birth_predator  — a periodic hazard law is enacted
  kill_predator   — the hazard law is repealed

Returns (corpus, description); 'no-op:<reason>' when the chosen
operator has no valid subject (counted by callers, never silent).
"""

import re

_PAIN_SRC = re.compile(r'^a pain source at ')
_PULSE = re.compile(r'^the pain source (\d+) pulses with period ')
_HIDDEN = re.compile(r'^a hidden state cycles through (\d+) states')
_WANDER = re.compile(r'^a second hidden state wanders through (\d+) states$')
_PREDATOR = re.compile(r'^a predator sweeps every ')

_OPS = ('birth_pulse', 'kill_pulse', 'birth_hidden', 'kill_hidden',
        'species_swap', 'birth_predator', 'kill_predator')


def mutate_law_structure(corpus, rng):
    corpus = list(corpus)
    op = _OPS[rng.randint(len(_OPS))]

    n_pain = sum(1 for l in corpus if _PAIN_SRC.match(l))
    pulsed = {int(_PULSE.match(l).group(1))
              for l in corpus if _PULSE.match(l)}
    hidden_idx = [i for i, l in enumerate(corpus) if _HIDDEN.match(l)]
    wander_idx = [i for i, l in enumerate(corpus) if _WANDER.match(l)]
    pred_idx = [i for i, l in enumerate(corpus) if _PREDATOR.match(l)]

    if op == 'birth_pulse':
        free = [i for i in range(n_pain) if i not in pulsed]
        if not free:
            return corpus, 'no-op:all-sources-pulse'
        i = free[rng.randint(len(free))]
        period = int(rng.choice([80, 120, 160]))
        corpus.append(f'the pain source {i} pulses with period '
                      f'{period} and amplitude 0.5')
        return corpus, f'birth_pulse:src{i}@{period}'

    if op == 'kill_pulse':
        lines = [(i, l) for i, l in enumerate(corpus) if _PULSE.match(l)]
        if not lines:
            return corpus, 'no-op:no-pulse-law'
        i, l = lines[rng.randint(len(lines))]
        del corpus[i]
        return corpus, f'kill_pulse:{_PULSE.match(l).group(1)}'

    if op == 'birth_hidden':
        if hidden_idx:
            return corpus, 'no-op:hidden-exists'
        states = int(rng.choice([2, 3, 4]))
        period = int(rng.choice([100, 150, 200]))
        corpus.append(f'a hidden state cycles through {states} states '
                      f'with period {period}')
        return corpus, f'birth_hidden:{states}states@{period}'

    if op == 'kill_hidden':
        if not hidden_idx:
            return corpus, 'no-op:no-hidden-law'
        del corpus[hidden_idx[0]]
        return corpus, 'kill_hidden'

    if op == 'species_swap':
        if hidden_idx:
            i = hidden_idx[0]
            states = _HIDDEN.match(corpus[i]).group(1)
            corpus[i] = (f'a second hidden state wanders through '
                         f'{states} states')
            return corpus, f'species_swap:cycles->wanders({states})'
        if wander_idx:
            i = wander_idx[0]
            states = _WANDER.match(corpus[i]).group(1)
            corpus[i] = (f'a hidden state cycles through {states} '
                         f'states with period 150')
            return corpus, f'species_swap:wanders->cycles({states})'
        return corpus, 'no-op:no-hidden-of-either-species'

    if op == 'birth_predator':
        if pred_idx:
            return corpus, 'no-op:predator-exists'
        period = int(rng.choice([120, 150, 200]))
        corpus.append(f'a predator sweeps every {period} steps for 30 '
                      f'steps at intensity 1.0 speed 2.0')
        return corpus, f'birth_predator:@{period}'

    if op == 'kill_predator':
        if not pred_idx:
            return corpus, 'no-op:no-predator-law'
        del corpus[pred_idx[rng.randint(len(pred_idx))]]
        return corpus, 'kill_predator'

    return corpus, 'no-op:unknown'

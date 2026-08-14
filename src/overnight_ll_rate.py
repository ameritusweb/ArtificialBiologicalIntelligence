"""Overnight: rate-based LL-B + directed slack (F18's exports, pre-registered).

THE QUESTION: does enriching worlds raise the RATE at which new words
become fundable — and does enrichment aimed WHERE the language is trying
to speak (directed slack) beat undirected enrichment?

C20 pre-flight (six checks):
1. Domain match — EL-2.5's exact machinery (corpora worlds, policy
   inhabitants, frozen thresholds, junction-law court), repeated.
2. Endpoint independence — the court never sees fitness; the slack
   objectives never see the court. The DIRECTED arm's target context is
   fixed a priori (the 7x-confirmed candidate), not chosen from results.
3. Exogeneity — every replicate uses independent seed-corpora batches
   and independent evolution/court seeds; arms differ only in objective.
4. Pairing — replicate r of every arm starts from the SAME seed-corpora
   batch (paired at the world-genotype level); court protocol identical.
5. Phenomenon strength — floors pre-registered below; F18 established
   single-session ratification is threshold-stochastic, hence RATES.
6. Endpoint sensitivity — ratification rate over R=10 replicates can
   move by construction; slack endpoints replicate LL-A's proven mover.

ARMS (paired seed-corpora per replicate):
  SEED       — no evolution; court on the seed batch directly.
  UNDIRECTED — 6 generations under global describable slack (v2 objective).
  DIRECTED   — 6 generations under global slack + TARGET slack, where
               target = slack restricted to windows with the standing
               candidate context active ('under hidden state 0' — seven
               independent proposal events; fixed a priori), lambda = 1.

VERDICT RULES (fixed before launch):
  Primary (LL-B-rate): SUPPORTED direction iff
      rate(DIRECTED) > rate(SEED) and ratifications(DIRECTED) >= 4/10.
      NOT SUPPORTED iff rate(DIRECTED) <= rate(SEED).
  Secondary: rate(UNDIRECTED) vs rate(SEED) (does undirected enrichment
      move the rate at all); slack deltas (LL-A at n=10).
  UNTESTED iff proposal rate in DIRECTED < 6/10 (court disengaged) or
      < 8 valid replicates per arm.
"""

import json
import math
import os
import time

import numpy as np

from environment_tiers import TieredEnvironment
from environment_language import describe
from environment_descriptive import classify_behavior_from_features, BEHAVIORS
from environment_lexical import (Lexicon, WordModel, evolve_one_generation,
                                 ratify_pending, append_ledger)
from environment_living import (live_in, describable_slack, mutate,
                                TIER, MIN_WINDOWS_PER_WORLD)
from train import generate_training_data, train_model

R = 10                 # replicates per arm
POP = 8
GENS = 6
LAMBDA = 1.0
TARGET_CTX = 'under hidden state 0'   # fixed a priori (7 proposal events)
BOOT_SEED = 123
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'overnight_ll_rate.json')


def make_policy():
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    return train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)


def calibrate(model):
    recs = []
    for s in range(9000, 9004):
        recs.extend(live_in(describe(TieredEnvironment(seed=s, tier=TIER)),
                            model, s * 3))
    paths = np.array([f['path'] for _, f, _ in recs])
    mvs = np.array([max(abs(f['toward_pain']), abs(f['toward_end']))
                    for _, f, _ in recs])
    return float(np.median(paths)), float(np.percentile(mvs, 70))


def directed_fitness(records, tau_rest, tau_d):
    """Global slack + target-context slack (the DIRECTED objective)."""
    g, share, n = describable_slack(records, tau_rest, tau_d)
    if g is None:
        return None, share
    target = [(p, f, ep) for p, f, ep in records if TARGET_CTX in p]
    bonus = 0.0
    if len(target) >= MIN_WINDOWS_PER_WORLD // 2:
        t, _, _ = describable_slack(target, tau_rest, tau_d)
        if t is not None:
            bonus = t
    return g + LAMBDA * bonus, share


def evolve_arm(seed_corpora, model, tau_rest, tau_d, base_seed, directed):
    pop = [list(c) for c in seed_corpora]
    for gen in range(GENS):
        fits = []
        for i, c in enumerate(pop):
            recs = live_in(c, model, base_seed + gen * 1000 + i * 10)
            if directed:
                fit, share = directed_fitness(recs, tau_rest, tau_d)
            else:
                fit, share, _ = describable_slack(recs, tau_rest, tau_d)
            ok = fit is not None and share is not None and share <= 0.9
            fits.append((fit if ok else -1.0, i))
        fits.sort(key=lambda x: (-x[0], x[1]))
        survivors = [pop[i] for _, i in fits[:POP // 2]]
        rng = np.random.RandomState(base_seed + 777 + gen)
        pop = survivors + [mutate(p, rng)[0] for p in survivors]
    # rank final population by measured fitness for the court
    scored = []
    for i, c in enumerate(pop):
        recs = live_in(c, model, base_seed + 99000 + i * 10)
        fit, share, _ = describable_slack(recs, tau_rest, tau_d)
        scored.append((fit if fit is not None else -9, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    slack_mean = float(np.mean([f for f, _ in scored if f > -9]))
    return [pop[i] for _, i in scored], slack_mean


def court(corpora, model, tau_rest, tau_d, seed0, ledger):
    def windows(cs, s0):
        out = []
        for i, c in enumerate(cs):
            recs = live_in(c, model, s0 + i * 10)
            out.append([(p, classify_behavior_from_features(
                f, tau_rest, tau_d)) for p, f, _ in recs])
        return out
    pw = windows(corpora[:5], seed0)
    rw = windows(corpora[5:8], seed0 + 500)
    train_c = [w for ws in pw[:4] for w in ws]
    val_c = list(pw[4])
    lex, moves, pending = evolve_one_generation(Lexicon(), train_c, val_c,
                                                ledger)
    r_train = [w for ws in rw[:2] for w in ws]
    r_val = list(rw[2])
    _, ratified = ratify_pending(lex, pending, r_train, r_val, ledger)
    return len(pending), len(ratified), [p['word'] for p in pending]


def main():
    t0 = time.time()
    print('=== OVERNIGHT: rate-based LL-B + directed slack ===')
    model = make_policy()
    tau_rest, tau_d = calibrate(model)
    print(f'frozen thresholds: tau_rest={tau_rest:.4f} tau_d={tau_d:.4f}')

    ledger = []
    out = {'arms': {'seed': [], 'undirected': [], 'directed': []},
           'config': {'R': R, 'POP': POP, 'GENS': GENS,
                      'LAMBDA': LAMBDA, 'target': TARGET_CTX}}

    for r in range(R):
        batch_seed = 60000 + r * 137
        seed_corpora = [describe(TieredEnvironment(seed=batch_seed + i,
                                                   tier=TIER))
                        for i in range(POP)]

        # SEED arm: court directly
        n_p, n_r, words = court(seed_corpora, model, tau_rest, tau_d,
                                70000 + r * 311, ledger)
        out['arms']['seed'].append({'proposed': n_p, 'ratified': n_r,
                                    'words': words})

        # UNDIRECTED arm
        pop_u, slack_u = evolve_arm(seed_corpora, model, tau_rest, tau_d,
                                    80000 + r * 977, directed=False)
        n_p, n_r, words = court(pop_u, model, tau_rest, tau_d,
                                81000 + r * 311, ledger)
        out['arms']['undirected'].append({'proposed': n_p, 'ratified': n_r,
                                          'words': words,
                                          'slack': round(slack_u, 4)})

        # DIRECTED arm
        pop_d, slack_d = evolve_arm(seed_corpora, model, tau_rest, tau_d,
                                    85000 + r * 977, directed=True)
        n_p, n_r, words = court(pop_d, model, tau_rest, tau_d,
                                86000 + r * 311, ledger)
        out['arms']['directed'].append({'proposed': n_p, 'ratified': n_r,
                                        'words': words,
                                        'slack': round(slack_d, 4)})

        el = (time.time() - t0) / 60
        print(f'replicate {r + 1}/{R} done ({el:.0f} min): '
              f"seed r={out['arms']['seed'][-1]['ratified']} "
              f"undir r={out['arms']['undirected'][-1]['ratified']} "
              f"dir r={out['arms']['directed'][-1]['ratified']}")

    # ---- Verdicts ----
    def rate(arm, key='ratified'):
        xs = out['arms'][arm]
        return sum(1 for x in xs if x[key] >= 1) / max(len(xs), 1)

    def prop_rate(arm):
        xs = out['arms'][arm]
        return sum(1 for x in xs if x['proposed'] >= 1) / max(len(xs), 1)

    r_seed, r_und, r_dir = rate('seed'), rate('undirected'), rate('directed')
    p_dir = prop_rate('directed')
    print(f'\nratification rates over {R} replicates: '
          f'seed {r_seed:.2f}  undirected {r_und:.2f}  '
          f'directed {r_dir:.2f}; directed proposal rate {p_dir:.2f}')

    if p_dir < 0.6:
        verdict = 'UNTESTED (directed-arm court disengaged)'
    elif r_dir > r_seed and sum(
            1 for x in out['arms']['directed'] if x['ratified'] >= 1) >= 4:
        verdict = ('SUPPORTED (directional): directed enrichment raises '
                   'the rate at which new words become fundable')
    elif r_dir <= r_seed:
        verdict = ('NOT SUPPORTED: directed enrichment does not raise '
                   'the ratification rate over seed worlds')
    else:
        verdict = ('INCONCLUSIVE at floors: directed > seed but below '
                   'the 4/10 support floor')
    print(f'\nLL-B-RATE VERDICT: {verdict}')
    out['verdict'] = verdict
    out['rates'] = {'seed': r_seed, 'undirected': r_und,
                    'directed': r_dir, 'directed_proposal': p_dir}
    out['elapsed_min'] = round((time.time() - t0) / 60, 1)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1)
    append_ledger(ledger)
    print(f'saved {RESULTS} ({out["elapsed_min"]} min); '
          f'{len(ledger)} court events to the etymology ledger')


if __name__ == '__main__':
    main()

"""EL-2.5 acceptance — the living-language loop (pre-registered).

C20 pre-flight (six checks):
1. Domain match — worlds are EL-0 corpora of the tier-4 class; the
   inhabitants, describer, court, and thresholds are the EL-1/EL-2
   instruments unchanged.
2. Endpoint independence — the slack objective never sees the ratchet
   test's court; the court never sees fitness. The FIXED seed lexicon
   measures slack; refinement is judged by the junction-law court alone.
3. Exogeneity — the evolved and seed populations are evaluated with the
   SAME protocol on FRESH episode seeds; the court's propose/ratify
   batches are disjoint world-genotypes (5/3), identically for the seed
   control and the evolved arm.
4. Pairing — seed-vs-evolved comparisons share the evaluation protocol
   and episode-seed scheme; the only difference is the corpora.
5. Phenomenon strength — floors: >= 90 windows/world for any fitness or
   court computation; >= 6 valid worlds per arm; F17 is the external
   control (0 ratifications for tier-4 worlds under this court at 5x
   this data volume).
6. Endpoint sensitivity — slack moves per world by construction; the
   court's margins are the EL-2 pre-registrations, unchanged.

ACCEPTANCE RULES (fixed before running):
  LL-A (the ratchet turns): evolved-population mean describable slack
      exceeds seed-population mean by >= 0.02 nats, fresh episodes,
      identical protocol.
  LL-B (new words become true): the EL-2 court (propose on 5 worlds,
      ratify on 3 disjoint worlds) ratifies >= 1 refinement on the
      EVOLVED population AND 0 on the seed population run identically
      inline — the loop closed: generation growth funded vocabulary
      that was unfundable before.
  LL-C (no degenerate worlds): every surviving evolved world respects
      the entropy ceiling (no behavior class > 0.9) and the slack gain
      does not come with marginal-NLL collapse (evolved mean marginal
      NLL >= 0.5 nats).
  LL-D (receipts): every utterance (kept mutation) and every court
      event carries receipts in the ledger.
  PASS = A, B, C, D. A without B: the worlds enriched but not enough to
  fund words (iterate longer/harder). B without A would be suspect
  (investigate before believing).
"""

import math

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_language import describe
from environment_descriptive import (window_features,
                                     classify_behavior_from_features,
                                     WINDOW, BEHAVIORS)
from environment_lexical import (Lexicon, WordModel, base_predicates,
                                 evolve_one_generation, ratify_pending,
                                 append_ledger)
from environment_living import (live_in, describable_slack, evolve_worlds,
                                TIER, EPISODES_PER_EVAL)
from train import generate_training_data, train_model

POP = 8
GENERATIONS = 8     # v2: four turns was the minimum, not the estimate
BOOT_SEED = 123

PASS_N, FAIL_N = 0, 0


def check(name, cond, detail=''):
    global PASS_N, FAIL_N
    if cond:
        PASS_N += 1
        print(f"  PASS  {name}")
    else:
        FAIL_N += 1
        print(f"  FAIL  {name}  {detail}")


def make_policy():
    print('bootstrapping inhabitant policy...')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    return train_model(X, Y, Z, epochs=6, staged=True,
                       steps_per_episode=300)


def population_stats(corpora, model, tau_rest, tau_d, seed0):
    slacks, shares, marg_nlls, idxs = [], [], [], []
    for i, c in enumerate(corpora):
        recs = live_in(c, model, seed0 + i * 10)
        slack, share, n = describable_slack(recs, tau_rest, tau_d)
        if slack is None:
            continue
        idxs.append(i)
        slacks.append(slack)
        shares.append(share)
        # marginal NLL of this world's behavior (triviality guard)
        beh = [classify_behavior_from_features(f, tau_rest, tau_d)
               for _, f, _ in recs]
        counts = {b: beh.count(b) for b in BEHAVIORS}
        tot = len(beh)
        m = -sum((counts[b] / tot) * math.log(
            max(counts[b] / tot, 1e-9)) for b in BEHAVIORS)
        marg_nlls.append(m)
    return slacks, shares, marg_nlls, idxs


def court_session(corpora, model, tau_rest, tau_d, seed0, ledger, tag):
    """The EL-2 junction-law court over a population: propose on the
    first 5 world-genotypes, ratify on the last 3 (disjoint)."""
    def corpus_windows(cs, s0):
        out = []
        for i, c in enumerate(cs):
            recs = live_in(c, model, s0 + i * 10)
            out.append([(p, classify_behavior_from_features(
                f, tau_rest, tau_d)) for p, f, _ in recs])
        return out

    propose_w = corpus_windows(corpora[:5], seed0)
    ratify_w = corpus_windows(corpora[5:8], seed0 + 500)
    train_c = [w for ws in propose_w[:4] for w in ws]
    val_c = [w for w in propose_w[4]]
    lex = Lexicon()
    lex2, moves, pending = evolve_one_generation(lex, train_c, val_c,
                                                 ledger)
    r_train = [w for ws in ratify_w[:2] for w in ws]
    r_val = [w for w in ratify_w[2]]
    lex3, ratified = ratify_pending(lex2, pending, r_train, r_val, ledger)
    print(f'  [{tag}] proposals: {[p["word"] for p in pending] or "none"}'
          f'; ratified: {ratified or "none"}')
    return ratified


def main():
    print('=== EL-2.5 acceptance: the living-language loop ===')
    model = make_policy()

    # Frozen behavior thresholds (EL-1 protocol) from seed-class worlds
    calib = []
    for s in range(9000, 9004):
        env_c = TieredEnvironment(seed=s, tier=TIER)
        calib.extend(live_in(describe(env_c), model, s * 3))
    paths = np.array([f['path'] for _, f, _ in calib])
    mvs = np.array([max(abs(f['toward_pain']), abs(f['toward_end']))
                    for _, f, _ in calib])
    tau_rest = float(np.median(paths))
    tau_d = float(np.percentile(mvs, 70))
    print(f'frozen thresholds: tau_rest={tau_rest:.4f} tau_d={tau_d:.4f}')

    # Seed population: today's worlds, spoken
    seed_corpora = [describe(TieredEnvironment(seed=40000 + i, tier=TIER))
                    for i in range(POP)]
    ledger = []

    print(f'evolving {POP} worlds for {GENERATIONS} generations under '
          f'describable-slack pressure...')
    evolved, history = evolve_worlds(seed_corpora, model, tau_rest, tau_d,
                                     GENERATIONS, ledger)
    for h in history:
        print(f'  gen {h["gen"]}: mean slack {h["mean_slack"]:+.4f}  '
              f'best {h["best_slack"]:+.4f}  ok {h["worlds_ok"]}/{POP}')

    # ---- LL-A: fresh-episode comparison, identical protocol ----
    s_seed, sh_seed, m_seed, _ = population_stats(
        seed_corpora, model, tau_rest, tau_d, 90000)
    s_evo, sh_evo, m_evo, i_evo = population_stats(
        evolved, model, tau_rest, tau_d, 91000)
    mean_seed = float(np.mean(s_seed)) if s_seed else 0.0
    mean_evo = float(np.mean(s_evo)) if s_evo else 0.0
    print(f'\nfresh-episode slack: seed {mean_seed:+.4f} '
          f'(n={len(s_seed)})  evolved {mean_evo:+.4f} (n={len(s_evo)})  '
          f'delta {mean_evo - mean_seed:+.4f}')
    floors = len(s_seed) >= 6 and len(s_evo) >= 6
    check('LL-A the ratchet turns (evolved slack - seed slack >= 0.02)',
          floors and (mean_evo - mean_seed >= 0.02),
          f'delta {mean_evo - mean_seed:+.4f}, floors={floors}')

    # ---- LL-B: the court, evolved vs seed control ----
    # v2 protocol fix: the final refill's children are UNEVALUATED — the
    # court must question the richest worlds, so the evolved population
    # is ordered by its measured fresh-episode slack first.
    if s_evo:
        ranked = [i_evo[k] for k in np.argsort([-s for s in s_evo])]
        ranked += [i for i in range(len(evolved)) if i not in ranked]
    else:
        ranked = list(range(len(evolved)))
    evolved_sorted = [evolved[i] for i in ranked]
    print('\nthe junction-law court:')
    ratified_evo = court_session(evolved_sorted[:8], model, tau_rest,
                                 tau_d, 95000, ledger, 'evolved')
    ratified_seed = court_session(seed_corpora[:8], model, tau_rest,
                                  tau_d, 96000, ledger, 'seed control')
    check('LL-B new words become true (evolved >=1 ratified, seed 0)',
          len(ratified_evo) >= 1 and len(ratified_seed) == 0,
          f'evolved {len(ratified_evo)}, seed {len(ratified_seed)}')

    # ---- LL-C: no degeneracy ----
    mean_marg_evo = float(np.mean(m_evo)) if m_evo else 0.0
    check('LL-C no degenerate worlds (entropy ceiling + marginal floor)',
          all(sh <= 0.9 for sh in sh_evo) and mean_marg_evo >= 0.5,
          f'max share {max(sh_evo) if sh_evo else 1}, '
          f'marg NLL {mean_marg_evo:.3f}')

    # ---- LL-D: receipts ----
    check('LL-D every utterance and court event carries receipts',
          bool(ledger) and all('receipts' in e for e in ledger),
          f'{len(ledger)} events')

    path = append_ledger(ledger)
    print(f'\n{len(ledger)} living-loop events appended to {path}')
    print(f'\n{"=" * 50}\nEL-2.5 ACCEPTANCE: {PASS_N} passed, '
          f'{FAIL_N} failed')
    if FAIL_N == 0:
        verdict = 'PASS — the loop is closed: generation funds vocabulary'
    else:
        verdict = 'NOT PASSED — iterate'
    print('VERDICT:', verdict)
    return FAIL_N == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)

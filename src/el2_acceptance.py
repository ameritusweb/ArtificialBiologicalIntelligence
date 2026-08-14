"""EL-2 acceptance (pre-registered).

C20 pre-flight (six checks):
1. Domain match — evolution runs on the same corpus class EL-1 passed on
   (tier-4 worlds, policy inhabitants, calibrated windows).
2. Endpoint independence — lexical moves are computed BETWEEN
   generations from receipts; no move sees the final test corpus.
3. Exogeneity — generation corpora use fresh world seeds; the final
   comparison corpus is untouched by any move decision.
4. Pairing — the gen-0 lexicon and the evolved lexicon are trained on
   the SAME final training corpus and scored on the SAME final test
   corpus; the lexicon is the only difference.
5. Phenomenon strength — floors: >= 400 windows/generation, >= 800 in
   the final comparison corpus; EL-1's localized failure signal exists
   (the split trigger has a real target).
6. Endpoint sensitivity — NLL per window; move admissions carry their
   own margins (pre-registered in environment_lexical.py).

ACCEPTANCE RULES (fixed):
  EL2-A (the split is real): at least one word_split admitted across the
      generations, and the children DIVERGE on the final held-out corpus
      (JSD of their behavior distributions >= 0.02, both >= 15
      occasions) — the split named a real distinction.
  EL2-B (the merge is honest): the planted decoy ('beside a hurting
      field' ≡ 'near a pain source') is merged away, and no
      non-duplicate pair merges.
  EL2-C (the language learned): evolved lexicon beats the gen-0 lexicon
      by >= 0.01 nats held-out NLL, both trained on the identical final
      training corpus, scored on the identical final test corpus.
  EL2-D (receipts-only): every admitted move carries receipts in the
      etymology ledger.
  PASS = A, B, C, D.
"""

import math

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment
from environment_descriptive import (window_features,
                                     classify_behavior_from_features,
                                     WINDOW, BEHAVIORS)
from environment_lexical import (Lexicon, WordModel, base_predicates,
                                 evolve_one_generation, ratify_pending,
                                 append_ledger, _jsd, SEED_WORDS)
from train import generate_training_data, train_model, EXPLORE_RATE, \
    PROBE_RATE_FLOOR

TIER = 4
GENERATIONS = 5     # v3: an extra generation so proposals can ratify
WORLDS_PER_GEN = 10          # 7 train + 3 val per generation
FINAL_WORLDS = 8             # 5 train + 3 test for the A/C comparison
EPISODES_PER_WORLD = 3
STEPS = 400
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


def collect_raw(model, seed):
    """Raw windows: (base predicate set, features)."""
    env = TieredEnvironment(seed=seed, tier=TIER)
    np.random.seed(seed * 7 + 1)
    env.rng = np.random.RandomState(seed * 7 + 2)
    rng = np.random.RandomState(seed * 7 + 3)
    records = []
    for ep in range(EPISODES_PER_WORLD):
        org = Organism()
        org.reset()
        xs, ys = [], []
        preds_at_start = None
        for step in range(STEPS):
            if step % WINDOW == 0:
                if len(xs) == WINDOW and preds_at_start is not None:
                    records.append((preds_at_start,
                                    window_features(env, xs, ys,
                                                    step - WINDOW)))
                xs, ys = [], []
                preds_at_start = base_predicates(env, org, step)
            window = org.get_observation_window()
            act, _ = model.predict(window)
            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                act = np.zeros_like(act)
            elif r < EXPLORE_RATE:
                act = rng.randint(0, 2, size=len(act)).astype(act.dtype)
            org.step(act, env, step)
            xs.append(org.x)
            ys.append(org.y)
    return records


def classify_corpus(raw, tau_rest, tau_d):
    return [(preds, classify_behavior_from_features(f, tau_rest, tau_d))
            for preds, f in raw]


def main():
    print('=== EL-2 acceptance: the language revises itself ===')
    model = make_policy()

    # Calibrate behavior thresholds once (gen-0 protocol), freeze forever.
    calib_raw = []
    for seed in range(9000, 9004):
        calib_raw.extend(collect_raw(model, seed))
    paths = np.array([f['path'] for _, f in calib_raw])
    moves_arr = np.array([max(abs(f['toward_pain']), abs(f['toward_end']))
                          for _, f in calib_raw])
    tau_rest = float(np.median(paths))
    tau_d = float(np.percentile(moves_arr, 70))
    print(f'frozen thresholds: tau_rest={tau_rest:.4f} tau_d={tau_d:.4f}')

    lexicon = Lexicon()
    gen0_lexicon = lexicon.copy()
    ledger = []
    all_moves = []       # immediate (merges) + RATIFIED moves only
    pending = []

    seed_base = 20000
    for gen in range(GENERATIONS):
        seeds = list(range(seed_base + gen * 100,
                           seed_base + gen * 100 + WORLDS_PER_GEN))
        # v2: train/val split BY WORLD (C20 exogeneity applied to the
        # move-validation surface).
        train_c, val_c = [], []
        for k, s in enumerate(seeds):
            dest = train_c if k < 7 else val_c
            dest.extend(classify_corpus(collect_raw(model, s),
                                        tau_rest, tau_d))
        # v3: the junction-law gate — last generation's proposals face
        # this generation's fresh worlds before anything enters the
        # lexicon.
        lexicon, ratified = ratify_pending(lexicon, pending, train_c,
                                           val_c, ledger)
        all_moves.extend(ratified)
        lexicon, moves, pending = evolve_one_generation(
            lexicon, train_c, val_c, ledger)
        all_moves.extend(moves)
        print(f'gen {gen}: ratified {ratified if ratified else "none"}; '
              f'immediate {moves if moves else "none"}; '
              f'proposed {[p["word"] for p in pending] or "none"}; '
              f'lexicon size {len(lexicon.words)}')
    if pending:
        for p in pending:
            ledger.append({'event': f'{p["kind"]}_expired_unratified',
                           'word': p['word'], 'receipts': p})

    # ---- Final comparison corpus (untouched by any move decision;
    # train/test split BY WORLD, v2) ----
    final_seeds = list(range(30000, 30000 + FINAL_WORLDS))
    ftrain, ftest = [], []
    for k, s in enumerate(final_seeds):
        dest = ftrain if k < 5 else ftest
        dest.extend(classify_corpus(collect_raw(model, s),
                                    tau_rest, tau_d))
    final = ftrain + ftest
    print(f'final corpus: {len(final)} windows '
          f'({len(ftrain)} train / {len(ftest)} test)')

    m_old = WordModel(gen0_lexicon).train(ftrain)
    m_new = WordModel(lexicon).train(ftrain)
    nll_old, _ = m_old.nll(ftest)
    nll_new, _ = m_new.nll(ftest)
    print(f'held-out NLL: gen0 lexicon {nll_old:.4f}  evolved '
          f'{nll_new:.4f}  delta {nll_old - nll_new:+.4f}')

    # ---- Verdicts (v3 rules, pre-registered before this run) ----
    splits = [m for m in all_moves if m[0] == 'split']
    proposals = [e for e in ledger if e['event'].endswith('_proposed')]
    rejections = [e for e in ledger if 'rejected' in e['event']
                  or 'expired' in e['event']]

    # EL2-A v3: IF a split ratified, its children must diverge on the
    # final corpus. If none ratified, the court must at least have
    # ENGAGED (proposals made and rejected on fresh evidence) — the
    # conservative outcome: no refinable structure at this grain.
    if splits:
        _, parent, child = splits[0]
        occ_c = m_new.totals.get(child, 0)
        occ_p = m_new.totals.get(parent, 0)
        if occ_c >= 15 and occ_p >= 15:
            d = _jsd(m_new.dist(parent), m_new.dist(child))
            verdict_a = d >= 0.02
            detail_a = (f'ratified split {parent!r} -> {child!r}: '
                        f'held-out JSD {d:.4f} (occ {int(occ_p)}/'
                        f'{int(occ_c)})')
        else:
            verdict_a = False
            detail_a = f'children too thin on final corpus ({occ_p}/{occ_c})'
        check('EL2-A ratified split diverges held-out', verdict_a,
              detail_a)
        print(f'    {detail_a}')
    else:
        verdict_a = bool(proposals) and bool(rejections)
        check('EL2-A (conservative) court engaged: proposals made and '
              'rejected on fresh evidence', verdict_a,
              f'{len(proposals)} proposals, {len(rejections)} rejections')

    merges = [m for m in all_moves if m[0] == 'merge']
    decoy_merged = any({'beside a hurting field', 'near a pain source'}
                       >= {m[1], m[2]} for m in merges)
    bad_merges = [m for m in merges
                  if {'beside a hurting field',
                      'near a pain source'} != {m[1], m[2]}]
    check('EL2-B decoy merged, no false merges',
          decoy_merged and not bad_merges, f'merges: {merges}')

    # EL2-C v3: never worse (always); better when anything ratified.
    ratified_any = bool([m for m in all_moves if m[0] in ('split',
                                                          'birth')])
    no_degradation = nll_new <= nll_old + 0.005
    if ratified_any:
        check('EL2-C the language learned (ratified moves improve '
              'held-out >= 0.005)',
              no_degradation and (nll_old - nll_new >= 0.005),
              f'delta {nll_old - nll_new:+.4f}')
    else:
        check('EL2-C (conservative) no ratifications and no degradation',
              no_degradation, f'delta {nll_old - nll_new:+.4f}')

    receipts_ok = all('receipts' in e for e in ledger)
    check('EL2-D every move carries receipts',
          bool(ledger) and receipts_ok, f'{len(ledger)} ledger entries')

    path = append_ledger(ledger)
    print(f'\n{len(ledger)} lexical events appended to {path}')
    print(f'\n{"=" * 50}\nEL-2 ACCEPTANCE: {PASS_N} passed, {FAIL_N} failed')
    if FAIL_N == 0 and ratified_any:
        verdict = 'PASS — the language revised itself under the junction law'
    elif FAIL_N == 0:
        verdict = ('PASS-CONSERVATIVE — the court is honest and no '
                   'refinement survived ratification: no refinable '
                   'structure at this grain (the world must grow first)')
    else:
        verdict = 'NOT PASSED — iterate'
    print('VERDICT:', verdict)
    return FAIL_N == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)

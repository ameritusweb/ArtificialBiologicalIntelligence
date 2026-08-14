"""P78 first bill — meta-prediction at generation grain (T156, the tower's
first rung).

PRE-REGISTRATION (fixed before results were computed):

Data: data/deep_time_overnight_final.json — the seed-44 citable baseline's
processing stream (49 generations). PRIMARY target: next-generation reopen
delta (the base web's restructuring behavior; pre-flight: mean 3.98,
std 3.91, range 0-13, zero-inflated — phenomenon strength PASS). SECONDARY
target (reported, not billed): next-generation discovery count.

Features at generation g (all base-stream, no leakage):
  reopen_delta(g), receipt_delta(g), fit_mass_delta(g), avg_fitness(g),
  num_discovered(g).

Models (all evaluated walk-forward, expanding window, warmup = first 12
transitions, predictions on the remaining ~36):
  A. persistence            — predict target(g+1) = target(g)
  B. expanding mean         — predict the mean of all past targets
  C. AR(1) least squares    — linear fit target(g+1) ~ target(g) on past
  D. SOV'-state (the carved meta-model): each feature median-split into
     {low, high} using PAST data only -> state signature (a quotient of
     processing state, the tower's minimal carving); prediction = mean
     past target following that state; unseen state -> expanding mean.

Endpoint: walk-forward MAE on the primary target.

VERDICT RULES (C13 vocabulary, earned):
  SUPPORTED (directional, first bill) : MAE(D) < MAE(A) and MAE(D) < MAE(B)
  NOT SUPPORTED                       : MAE(D) >= MAE(A) or >= MAE(B)
  UNTESTED                            : fewer than 30 walk-forward
                                        predictions, or target std < 0.5
Notes billed with the verdict regardless of direction: n = 1 run,
generation grain, ~36 predictions — this is the DIRECTIONAL first bill;
the full P78 (predict WHERE reopens cluster, per-slot, episode grain)
requires the churn instrumentation live since 2026-08-10 on a post-E1 run.
AR(1) (model C) is reported for context, not billed (it shares model D's
class of "uses past structure"; the billed contrast is carved-state vs
the two structureless baselines).
"""

import json
import numpy as np

WARMUP = 12
MIN_PREDICTIONS = 30
MIN_TARGET_STD = 0.5


def load_stream(path='data/deep_time_overnight_final.json'):
    with open(path) as f:
        h = json.load(f)['history']
    rows = [r for r in h if r.get('sov')]
    reopen_cum = [r['sov']['reopened'] for r in rows]
    rec_cum = [r['sov']['receipts'] for r in rows]
    mass_cum = [r['sov']['fit_mass'] for r in rows]
    reopen_d = np.diff([0] + reopen_cum).astype(float)
    rec_d = np.diff([0] + rec_cum).astype(float) / 1000.0
    mass_d = np.diff([0] + mass_cum).astype(float) / 1000.0
    fitness = np.array([r['avg_fitness'] for r in rows], float) / 1000.0
    disc = np.array([r['num_discovered'] for r in rows], float)
    feats = np.stack([reopen_d, rec_d, mass_d, fitness, disc], axis=1)
    return feats, reopen_d, disc


def walk_forward(feats, target):
    """Returns dict of model -> list of (prediction, truth)."""
    n = len(target)
    out = {'persistence': [], 'expanding_mean': [], 'ar1': [], 'sov_state': []}
    for t in range(WARMUP, n - 1):
        past_f = feats[:t + 1]
        past_y = target[1:t + 1]          # targets aligned: y[g] = target at g
        hist_pairs_x = target[:t]          # for AR1: x = target(g), y = target(g+1)
        hist_pairs_y = target[1:t + 1]
        truth = target[t + 1]

        # A: persistence
        out['persistence'].append((target[t], truth))
        # B: expanding mean of observed targets so far
        out['expanding_mean'].append((float(np.mean(target[1:t + 1])), truth))
        # C: AR(1) least squares on past pairs
        x, y = hist_pairs_x, hist_pairs_y
        vx = np.var(x)
        if vx > 1e-9:
            b = float(np.cov(x, y, bias=True)[0, 1] / vx)
            a = float(np.mean(y) - b * np.mean(x))
            out['ar1'].append((a + b * target[t], truth))
        else:
            out['ar1'].append((float(np.mean(y)), truth))
        # D: SOV'-state — median split each feature on PAST data only
        med = np.median(past_f[:t], axis=0)          # medians from past states
        sig = tuple((past_f[:t] >= med).astype(int).tolist()[g]
                    for g in range(t))               # per-past-gen signatures
        # build state -> following targets
        table = {}
        for g in range(t):                            # state at g -> target at g+1
            key = tuple((past_f[g] >= med).astype(int))
            table.setdefault(key, []).append(target[g + 1])
        cur_key = tuple((feats[t] >= med).astype(int))
        if cur_key in table:
            pred = float(np.mean(table[cur_key]))
        else:
            pred = float(np.mean(target[1:t + 1]))
        out['sov_state'].append((pred, truth))
    return out


def mae(pairs):
    return float(np.mean([abs(p - y) for p, y in pairs]))


def main():
    feats, reopen_d, disc = load_stream()

    print('=== P78 first bill: meta-prediction at generation grain ===')
    print(f'stream length: {len(reopen_d)} gens; '
          f'primary target std: {reopen_d[1:].std():.2f}')

    results = {}
    for name, target in [('PRIMARY reopen_delta', reopen_d),
                         ('secondary discovered', disc)]:
        wf = walk_forward(feats, target)
        n_pred = len(wf['sov_state'])
        maes = {m: mae(pairs) for m, pairs in wf.items()}
        results[name] = (n_pred, maes)
        print(f'\n--- {name} (walk-forward n={n_pred}) ---')
        for m in ('persistence', 'expanding_mean', 'ar1', 'sov_state'):
            print(f'  {m:15s} MAE = {maes[m]:.4f}')

    # Verdict (primary only, pre-registered rules)
    n_pred, maes = results['PRIMARY reopen_delta']
    target_std = reopen_d[1:].std()
    if n_pred < MIN_PREDICTIONS or target_std < MIN_TARGET_STD:
        verdict = 'UNTESTED (floors not met)'
    elif (maes['sov_state'] < maes['persistence']
          and maes['sov_state'] < maes['expanding_mean']):
        verdict = ('SUPPORTED (directional first bill: carved state model '
                   'beats both structureless baselines)')
    else:
        verdict = ('NOT SUPPORTED at generation grain (carved model fails '
                   'to beat a structureless baseline)')
    print(f'\nP78 first-bill VERDICT: {verdict}')
    print('Scope: n=1 run, generation grain, directional. Full P78 = '
          'per-slot episode grain on instrumented post-E1 runs.')
    return verdict, results


if __name__ == '__main__':
    main()

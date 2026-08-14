"""P106 — the collision maturity curve: conflation as the arithmetic
fate of a growing vocabulary in a fixed language (F39 impl. 5-6, the
pigeonhole law; card locked at launch 2026-08-13; registration as
P106 awaits the user's word — this run is its first bill either way).

THE CLAIM: rendering-flagged collision pairs (slots whose
band-quantized pose profiles are IDENTICAL — the language cannot
tell them apart) grow SUPERLINEARLY with the composed-population
size past a crowding threshold, because the description language's
band space is fixed while the vocabulary grows. Falsifier: per-capita
collision rate flat or declining as the population grows — band
space effectively unbounded at reachable populations, the pigeonhole
never binds, and LC-1's band-refinement operator loses its demand
signal.

DESIGN (measurement curve, no treatment arms): one web (plain-fit
accountant), grown over 24 worlds (staged-harness class, compose
scans up to 3/boundary, co-fit floor 50). RENDER at every world
boundary (pure read, hash-asserted): band profiles over all active
slots; collision pairs counted as sum over profile groups of
C(k,2), reported THREE ways — composed-composed, inherited-involved,
total — because the inherited population is band-distinct at birth
but not guaranteed to stay so under geometry drift (reported, not
assumed). Per checkpoint: n_composed, n_active, distinct profiles,
pairs by class.

ENDPOINT (fixed): per-capita collision rate = composed-composed
pairs / n_composed, over checkpoints with n_composed >= 4.
SUPPORTED iff Spearman(n_composed, per-capita rate) >= +0.5 AND
final composed-composed pairs >= 5 (superlinearity: per-capita rate
rises with crowding). NOT SUPPORTED iff Spearman <= 0 with >= 8
qualifying checkpoints (band space does not bind). PARTIAL between.
UNTESTED: fewer than 8 qualifying checkpoints or final n_composed
< 12.

C20 (eight): 1 domain — staged-harness class, in-dist. 2 endpoint
independence — rendering is a pure read (hash-asserted per
checkpoint); growth is driven by the compose scan, which does not
read profiles. 3 exogeneity — n/a (measurement curve; the
within-trajectory regressor is population size, which time
confounds — stated honestly: this card measures the CURVE SHAPE,
not a causal claim; the pigeonhole argument supplies the mechanism).
4 pairing — n/a (single trajectory). 5 phenomenon strength — F33's
receipt: 16 colliding slots among the composed population at 8
worlds in this harness class; mid-curve existence is receipted.
6 sensitivity — resolution 1 pair; F33 magnitudes sit 3x above the
final floor. 7 genesis/rates — compose events are capped 3/boundary
(deterministic rate); the curve is reported over 24 checkpoints.
8 population closure — the measured population is the web's own
active set at each checkpoint (a growth curve measures a growing
population BY DESIGN; the per-capita endpoint is exactly the
normalization that makes checkpoints comparable; floors above).
"""

import hashlib
import json
import os
import time

import numpy as np

from lc_store import FAMILY_NAMES
from p105_experiment import band_profile, state_hash
from replay_overnight import build_engine, BOOT_SEED
from staged_fit_experiment import Accountant, run_worlds
from train import generate_training_data, train_model

WORLDS = [(98300 + i, (4, 3)[i % 2]) for i in range(24)]
INHERITED_MAX = 32
CHECKPOINT_FLOOR = 8
QUALIFY_N = 4
FINAL_COMPOSED_FLOOR = 12
FINAL_PAIRS_FLOOR = 5
SPEARMAN_GATE = 0.5
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p106_maturity.json')


def render_checkpoint(web):
    h0 = state_hash(web)
    active = {sid: s for sid, s in web.slots.items()
              if s.state in ('open', 'closed')}
    by_profile = {}
    for sid, s in active.items():
        by_profile.setdefault(band_profile(s), []).append(sid)
    cc = ci = 0
    for p, sids in by_profile.items():
        if p == 'nothing' or len(sids) < 2:
            continue
        comp = [x for x in sids if x > INHERITED_MAX]
        inh = [x for x in sids if x <= INHERITED_MAX]
        cc += len(comp) * (len(comp) - 1) // 2
        ci += len(inh) * (len(inh) - 1) // 2 + len(inh) * len(comp)
    n_comp = sum(1 for sid in active if sid > INHERITED_MAX)
    assert state_hash(web) == h0, 'C20 check 2: render not pure'
    return {'n_active': len(active), 'n_composed': n_comp,
            'distinct_profiles': len(by_profile),
            'pairs_composed': cc, 'pairs_inherited_involved': ci,
            'pairs_total': cc + ci}


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    def rank(v):
        return np.argsort(np.argsort(np.asarray(v),
                                     kind='stable'),
                          kind='stable').astype(float)
    rx, ry = rank(xs), rank(ys)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else None


def main():
    t0 = time.time()
    print('=== P106: collision maturity curve (24 worlds) ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    arm = Accountant('P106', staged=False, consume=False)
    curve = []
    c = 0
    for i, w in enumerate(WORLDS):
        c = run_worlds([w], [arm], engine, model, c)
        cp = render_checkpoint(arm.web)
        cp['world'] = i + 1
        curve.append(cp)
        print('  w%02d: composed=%d cc_pairs=%d total_pairs=%d '
              'profiles=%d' % (i + 1, cp['n_composed'],
                               cp['pairs_composed'],
                               cp['pairs_total'],
                               cp['distinct_profiles']))

    qual = [cp for cp in curve if cp['n_composed'] >= QUALIFY_N]
    xs = [cp['n_composed'] for cp in qual]
    ys = [cp['pairs_composed'] / cp['n_composed'] for cp in qual]
    rho = spearman(xs, ys)
    final = curve[-1]

    if len(qual) < CHECKPOINT_FLOOR \
            or final['n_composed'] < FINAL_COMPOSED_FLOOR:
        verdict = ('UNTESTED (qualifying checkpoints=%d/%d, final '
                   'composed=%d/%d)'
                   % (len(qual), CHECKPOINT_FLOOR,
                      final['n_composed'], FINAL_COMPOSED_FLOOR))
    elif rho is not None and rho >= SPEARMAN_GATE \
            and final['pairs_composed'] >= FINAL_PAIRS_FLOOR:
        verdict = ('SUPPORTED: per-capita collision rate rises with '
                   'crowding (rho=%.2f over %d checkpoints; final '
                   '%d pairs among %d composed) — the pigeonhole '
                   'binds' % (rho, len(qual), final['pairs_composed'],
                              final['n_composed']))
    elif rho is not None and rho <= 0:
        verdict = ('NOT SUPPORTED: per-capita rate flat/declining '
                   '(rho=%.2f) — band space does not bind at '
                   'reachable populations' % rho)
    else:
        verdict = ('PARTIAL: rho=%s final_pairs=%d — between gates'
                   % (rho, final['pairs_composed']))
    print('\nP106 VERDICT: %s' % verdict)

    out = {'curve': curve, 'qualifying_checkpoints': len(qual),
           'spearman_percapita': rho, 'verdict': verdict,
           'elapsed_min': round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('saved %s' % RESULTS)


if __name__ == '__main__':
    main()

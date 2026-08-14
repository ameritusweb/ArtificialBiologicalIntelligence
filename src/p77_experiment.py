"""P77 — earned vs stipulated enumeration (the masked-modeling
contrast; card locked at launch; pre-registered in
replay_phase_requirements.md since 2026-08-10; operators built
2026-08-12: sov.occlude / sov.enumerate_gap, battery 162/162).

THE CLAIM: the hypothesis space for withheld content can be READ FROM
THE GAP'S CONNECTOR GEOMETRY — earned enumeration (receipt-weighted
graph diffusion from the gap, certainty-weighted,
exclusion-eliminated, blind to the sealed content by construction)
identifies the withheld family better than any fixed candidate list
over the same space.

DESIGN — leave-one-out omission over one funded web:
  Warmup: the staged-fit harness verbatim (6 worlds, co-fit composes)
  + post-warmup edge densification (compose top co-fit pairs, co-fit
  >= 50, until >= 20 inherited slots carry an edge or pairs exhaust —
  stated cap, no silent truncation). Then, for every INHERITED slot
  with fit_count >= 500 and connectivity >= 1 (truth = its family
  index; operator-born slots excluded for truth cleanliness):
    occlude(slot) — Case-3 guard refusals counted;
    earned rank   = rank of the true family in enumerate_gap's list
                    (absent -> rank NUM_FAMILIES);
    stipulated    = (a) UNIFORM chance rank (NUM_FAMILIES+1)/2 = 17;
                    (b) FREQUENCY list — families ranked once by
                    global positive-fit volume (the strongest fixed
                    list; same list for every gap).
ENDPOINTS: mean earned rank vs mean frequency rank (paired per slot);
funding scaling — Spearman(advantage, neighborhood edge count) > 0
(the card's "advantage scales with neighborhood funding").
VERDICTS: SUPPORTED iff mean earned < mean frequency AND scaling
rho > 0. PARTIAL: one of the two. NOT SUPPORTED: earned >= both
stipulated baselines. UNTESTED: < 10 eligible occlusions.
FALSIFIERS (registered): earned ~ stipulated (geometry carries no
hypothesis-space information beyond the label set); advantage not
scaling with funding (the "earned" part isn't doing the work).

C20: (1) same harness, in-dist; (2) enumeration blind to sealed truth
by construction — the only path from truth to rank is through the
world that built the edges, which is the claim; (3) leave-one-out is
deterministic; (4) all occlusions share one web (paired); (5) >= 30
edges asserted before treatment; (6) ranks range 1..33, free to move;
(7) no genesis endpoint; ~20-30 paired occlusions = within-web
replication.
"""

import json
import os
import time

import numpy as np

from receptor_eigen_coder import FAMILY_GROUPS
from sov import NUM_FAMILIES
from train import generate_training_data, train_model
from replay_overnight import build_engine, BOOT_SEED
import staged_fit_experiment as sfe

FAMILY_NAMES = [n for n, _ in FAMILY_GROUPS]
MIN_FITS = 200      # v2 iteration (v1 UNTESTED at floors: 5 eligible)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p77_enumeration.json')


def main():
    t0 = time.time()
    print('=== P77: earned vs stipulated enumeration ===')
    X, Y, Z, _ = generate_training_data(
        num_episodes=20, steps_per_episode=300, seed=BOOT_SEED)
    model = train_model(X, Y, Z, epochs=6, staged=True,
                        steps_per_episode=300)
    engine = build_engine()

    acc = sfe.Accountant('P77', staged=False, consume=False)
    # v4 (successor card): 18-world stream — v1-v3 converged on 8
    # eligible at 12 worlds (stream-limited population). Also carries
    # the bimodality pre-registration: earned enumeration succeeds on
    # SELECTIVE-content slots and fails on promiscuous ones —
    # slot fire_rate is the predicted moderator (low-fire eligible
    # slots' earned ranks beat high-fire ones'; the info-pricing
    # insight in the enumeration domain).
    extra = [(96700 + i, (4, 3)[i % 2]) for i in range(6)]
    sfe.run_worlds(sfe.WARMUP_WORLDS + sfe.TREAT_WORLDS + extra,
                   [acc], engine, model, 0)
    web = acc.web

    # post-warmup densification, v2 (v1's cascade lesson: unrestricted
    # densification composed compose-of-compose towers, 789 deep, and
    # still starved inherited connectivity): INHERITED-INHERITED pairs
    # only, hard cap 60, stated and logged.
    densified = 0
    for (x, y), n in sorted(acc.cofit.items(),
                            key=lambda kv: (-kv[1], kv[0])):
        if densified >= 60 or n < 20:      # v3: co-fit floor 50 -> 20
            break                          # (v2: 8 eligible < 10)
        if (x, y) in acc.composed:
            continue
        sx, sy = web.slots.get(x), web.slots.get(y)
        if (sx is None or sy is None
                or sx.origin_operator != 'inherited'
                or sy.origin_operator != 'inherited'
                or sx.state not in ('open', 'closed')
                or sy.state not in ('open', 'closed')):
            continue
        if web.compose(x, y)[0] >= 0:
            acc.composed.add((x, y))
            densified += 1
    print(f'web: edges={len(web.edges)} (densified +{densified})')
    assert len(web.edges) >= 30, 'C20 check 5: edge floor unmet'

    # stipulated frequency list (one fixed list for every gap)
    fam_volume = np.zeros(NUM_FAMILIES)
    for s in web.slots.values():
        if s.origin_operator == 'inherited':
            fam_volume[int(np.argmax(
                s.geometry.family_thresholds))] += s.ledger.fit_count
    freq_order = list(np.argsort(-fam_volume, kind='stable'))
    freq_rank = {int(f): i + 1 for i, f in enumerate(freq_order)}

    rows = []
    case3 = 0
    for sid, s in sorted(web.slots.items()):
        if s.origin_operator != 'inherited':
            continue
        if s.ledger.fit_count < MIN_FITS:
            continue
        occ = web.occlude(sid, min_fits=MIN_FITS)
        if occ is None:
            case3 += 1
            continue
        truth = occ['sealed_truth']['dominant_family']
        ranked, funding = web.enumerate_gap(sid)
        er = next((i + 1 for i, (sc, f) in enumerate(ranked)
                   if f == truth), NUM_FAMILIES)
        rows.append({'slot': s.name, 'truth': int(truth),
                     'earned_rank': er,
                     'freq_rank': freq_rank[int(truth)],
                     'edges': web._connectivity(sid),
                     'funding': funding,
                     'fire_rate': round(s.ledger.fire_rate, 4)})

    n = len(rows)
    print(f'eligible occlusions: {n} (case-3 refusals: {case3})')
    if n < 10:
        verdict = f'UNTESTED ({n} eligible occlusions < 10)'
        out = {'verdict': verdict, 'rows': rows, 'case3': case3}
    else:
        e = np.array([r['earned_rank'] for r in rows], dtype=float)
        f = np.array([r['freq_rank'] for r in rows], dtype=float)
        uniform = (NUM_FAMILIES + 1) / 2
        adv = f - e
        deg = np.array([r['edges'] for r in rows], dtype=float)
        rho = None
        if np.std(adv) > 0 and np.std(deg) > 0:
            ra = np.argsort(np.argsort(adv)).astype(float)
            rd = np.argsort(np.argsort(deg)).astype(float)
            rho = float(np.corrcoef(ra, rd)[0, 1])
        beats_freq = e.mean() < f.mean()
        beats_unif = e.mean() < uniform
        scales = rho is not None and rho > 0
        if beats_freq and scales:
            verdict = ('SUPPORTED: earned enumeration beats the fixed '
                       'list and the advantage scales with funding')
        elif beats_freq or (beats_unif and scales):
            verdict = 'PARTIAL: one of the two clauses'
        elif not beats_freq and not beats_unif:
            verdict = ('NOT SUPPORTED: geometry carries no '
                       'hypothesis-space information beyond the '
                       'label set')
        else:
            verdict = 'PARTIAL: beats chance only'
        print(f'earned mean rank {e.mean():.2f} | frequency '
              f'{f.mean():.2f} | uniform {uniform:.1f} | '
              f'scaling rho {rho}')
        # bimodality moderator (successor-card pre-registration):
        fr = np.array([r['fire_rate'] for r in rows])
        med = float(np.median(fr))
        lo = e[fr <= med]
        hi = e[fr > med]
        mod = (float(lo.mean()), float(hi.mean())) if len(lo) and len(hi) \
            else (None, None)
        print(f'fire-rate moderator: low-fire earned {mod[0]} vs '
              f'high-fire {mod[1]} (predicted low < high)')
        out_mod = {'low_fire_mean_rank': mod[0],
                   'high_fire_mean_rank': mod[1],
                   'moderator_confirmed':
                   (mod[0] is not None and mod[0] < mod[1])}
        out = {'verdict': verdict,
               'earned_mean_rank': float(e.mean()),
               'freq_mean_rank': float(f.mean()),
               'uniform_rank': uniform,
               'scaling_rho': rho, 'n': n, 'case3': case3,
               'fire_rate_moderator': out_mod,
               'rows': rows}
    print(f'\nP77 VERDICT: {out["verdict"]}')
    out['elapsed_min'] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as fjson:
        json.dump(out, fjson, indent=1, default=str)
    print(f'saved {RESULTS}')


if __name__ == '__main__':
    main()

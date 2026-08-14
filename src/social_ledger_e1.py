"""E1: ledger split + Pose/Attest — arm runner.

Arms (social_ledger_requirements.md §4):
  isolated  — per-organism webs, masked views, bus OFF
  share     — per-organism webs, masked views, bus ON
  communal  — ONE shared web, contributions masked per organism (the
              pre-split accidental design, kept as an explicit anchor)

Usage:
  python social_ledger_e1.py --arm share --seed 45
  python social_ledger_e1.py --identity --seed 45     # pre-flight smoke
  python social_ledger_e1.py --analyze                # verdict table

Pre-flight discipline (C20):
  - identity mode MUST pass before any billed arm: SHARE with Q=0 must be
    bit-identical to ISOLATED (fitness traces AND per-web receipt counts).
  - behavior is arm-invariant by construction (webs are observers); the
    analyzer asserts fitness-trace equality across arms at matched seed and
    VOIDs the comparison if it fails.
  - divergence manipulation check runs per generation; P64 is
    VOID-instrument if measured D does not track mask distance.

Endpoint probes that run_overnight performs but E1 skips (both read-only,
zero behavioral effect): thinking-influence measurement, novel-receptor
detection. Everything behavior-bearing is kept: oracle bootstrap, policy
retraining, mental-model rebuilds, per-organism discovery, selection and
reproduction, the full rich stack.
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np

from deep_time_overnight import run_generation_rich
from deep_time import EvolvingOrganism, select_and_reproduce
from mental_model import build_mental_model
from receptor_discovery import discover, calibrate_null_thresholds
from model import compute_obs_indices
from train import train_model, generate_training_data
from thinking_substrate import ThinkingTree
from sov import ConstraintWeb
from receptor_eigen_coder import ReceptorEigenCoder
from social_ledger import (PoseAttestBus, CoFitTracker, build_masks,
                           divergence_check)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

POP = 6
GENS = 20
EPISODES = 5
STEPS = 200
BOOTSTRAP_EPISODES = 20
EPOCHS_PER_GEN = 8
TIER = 4
BLIND_W = 8
STRIDE = 4

# Pre-registered billing floors (§5)
FLOOR_IMPORTS = 500
FLOOR_CORROBORATIONS = 100
FLOOR_POSES_PER_GRADE = 30


def _seed_everything(seed):
    """Global-RNG pairing discipline (the P45 lesson)."""
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _make_webs(arm, masks_families):
    if arm == 'communal':
        web = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=1,
                            ledger_id='COMMUNAL')
        web.populate_from_families()
        return None, web
    webs = {}
    for i in range(POP):
        w = ConstraintWeb(eigen_coder=ReceptorEigenCoder(), debug_level=1,
                          ledger_id=f'ORG_{i}')
        w.populate_from_families()
        webs[i] = w
    return webs, None


def _sighted_slot_ids(web, blind_fams):
    out = set()
    for sid, slot in web.slots.items():
        active = np.where(slot.geometry.family_thresholds > 0)[0]
        if not len(active):
            continue
        if any(int(f) not in blind_fams for f in active):
            out.add(sid)
    return out


def _web_endpoints(web, blind_fams):
    """Per-web epistemic endpoints on SIGHTED slots (E1-DEMAND)."""
    sighted = _sighted_slot_ids(web, blind_fams)
    radii, certs, closures = [], [], 0
    for sid in sighted:
        slot = web.slots[sid]
        if slot.state == 'closed':
            closures += 1
            certs.append(slot.ledger.certainty)
        elif slot.state == 'open':
            if np.isfinite(slot.geometry.radius):
                radii.append(slot.geometry.radius)
            certs.append(slot.ledger.certainty)
    return {
        'closures': closures,
        'mean_open_radius': float(np.mean(radii)) if radii else None,
        'mean_certainty': float(np.mean(certs)) if certs else None,
        'reopens': web._op_counts.get('reopen', 0),
    }


def run_arm(arm, seed, gens=GENS, quiet_bus=False, tag=None):
    """One full arm run. quiet_bus: SHARE infrastructure with Q=0 (the
    zero-budget identity configuration)."""
    _seed_everything(seed)
    t0 = time.time()

    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    rng = np.random.RandomState(seed)

    mask_indices, blind_families = build_masks(POP, BLIND_W, STRIDE)
    webs, communal_web = _make_webs(arm, blind_families)

    bus = None
    if arm == 'share':
        bus = PoseAttestBus(blind_families=blind_families,
                            q=0 if quiet_bus else 3)

    # P75 add-on: co-fit tracker runs in ALL arms (the clean prediction
    # lives in ISOLATED — matched > shuffled with no channel at all).
    cofit = CoFitTracker(POP, STEPS, blind_families)

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(POP)]
    world_state = None
    cumulative_log = []
    cumulative_windows, cumulative_targets, cumulative_next_pain = [], [], []
    history = []

    label = tag or f"{arm}{'_q0' if quiet_bus else ''}"
    print(f"=== E1 arm={label} seed={seed} pop={POP} gens={gens} ===")

    # --- Generation 0: oracle bootstrap (engine None — no SOV, parity
    # with the baseline) -------------------------------------------------
    print("--- Generation 0 (bootstrap) ---")
    world_state, _, _, _ = run_generation_rich(
        organisms, 0, model=None, engine=None, tree=None, rng=rng,
        world_state=world_state, steps_per_episode=STEPS,
        num_episodes=BOOTSTRAP_EPISODES, tier=TIER, use_oracle=True)
    for evo_org in organisms:
        cumulative_log.extend(evo_org.experience_log)

    print("  Training initial policy...")
    X, Y, Z, _ = generate_training_data(
        num_episodes=BOOTSTRAP_EPISODES, steps_per_episode=STEPS, seed=seed)
    model = train_model(X, Y, Z, epochs=EPOCHS_PER_GEN, staged=True,
                        steps_per_episode=STEPS)
    cumulative_windows.extend([w for w in X])
    cumulative_targets.extend([t for t in Y])
    cumulative_next_pain.extend([p for p in Z])

    print("  Building mental model...")
    engine = build_mental_model(cumulative_log)
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24,
                        max_depth=3)
    null_thresh = calibrate_null_thresholds(
        cumulative_log[:30000], engine, num_shuffles=5)

    for evo_org in organisms:
        if len(evo_org.experience_log) >= 100:
            mt = evo_org.body_params.get('merge_threshold')
            org_engine = build_mental_model(evo_org.experience_log,
                                            merge_threshold=mt)
            results = discover(evo_org.experience_log, org_engine,
                               threshold_overrides=null_thresh,
                               log_provenance='oracle')
            evo_org.discovered_receptors = results['discovered']

    # --- Generations 1..G ----------------------------------------------
    for gen in range(1, gens + 1):
        print(f"\n--- Gen {gen}/{gens} ({(time.time()-t0)/60:.0f}min, "
              f"arm={label}) ---")
        organisms = select_and_reproduce(organisms, POP, rng)
        for i, o in enumerate(organisms):
            o.organism_id = f"gen{gen}_{i}"

        # Route webs for this arm
        if arm == 'communal':
            engine.constraint_web = communal_web
            engine.constraint_webs = None
            sov_webs = None
        else:
            engine.constraint_web = None
            sov_webs = webs

        world_state, windows, targets, next_pain = run_generation_rich(
            organisms, gen, model=model, engine=engine, tree=tree, rng=rng,
            world_state=world_state, steps_per_episode=STEPS,
            num_episodes=EPISODES, tier=TIER, use_oracle=False,
            sov_webs=sov_webs, sov_masks=mask_indices, sov_bus=bus,
            sov_cofit=cofit)

        for evo_org in organisms:
            cumulative_log.extend(evo_org.experience_log)
        if windows:
            cumulative_windows.extend(windows)
            cumulative_targets.extend(targets)
            cumulative_next_pain.extend(next_pain)

        Xa = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Ya = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Za = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        if len(Xa) >= 100:
            model = train_model(Xa, Ya, Za, epochs=EPOCHS_PER_GEN,
                                staged=True, steps_per_episode=STEPS)

        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)
        # Rebase every carried web against the fresh encoder (one epoch for
        # all webs — epoch counters stay aligned, which exact partition and
        # cross-web embedding comparisons rely on).
        if arm == 'communal':
            engine.constraint_web = communal_web
            communal_web.rebase(engine.encoder)
        else:
            engine.constraint_web = None
            for w in webs.values():
                w.rebase(engine.encoder)

        # Discovery (behavior-bearing via selection) — kept, full organism
        gen_discovered = set()
        for evo_org in organisms:
            if len(evo_org.experience_log) >= 100:
                mt = evo_org.body_params.get('merge_threshold')
                org_engine = build_mental_model(evo_org.experience_log,
                                                merge_threshold=mt)
                results = discover(evo_org.experience_log, org_engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='policy')
                evo_org.discovered_receptors = results['discovered']
                gen_discovered.update(results['discovered'])

        rec = {
            'generation': gen,
            'avg_fitness': round(float(np.mean(
                [o.fitness for o in organisms])), 1),
            'best_fitness': round(float(max(
                o.fitness for o in organisms)), 1),
            'fitness_per_org': [round(float(o.fitness), 1)
                                for o in organisms],
            'num_discovered': len(gen_discovered),
            'elapsed_min': round((time.time() - t0) / 60, 1),
        }

        if arm == 'communal':
            stats = communal_web.get_stats()
            rec['sov'] = {'COMMUNAL': {
                'stats': {k: stats[k] for k in
                          ('open', 'closed', 'archaized', 'total_receipts',
                           'unassigned_pool')},
                'violations': communal_web.check_conservation_laws(),
            }}
        else:
            rec['sov'] = {}
            for i, w in webs.items():
                stats = w.get_stats()
                rec['sov'][f'ORG_{i}'] = {
                    'stats': {k: stats[k] for k in
                              ('open', 'closed', 'archaized',
                               'total_receipts', 'unassigned_pool')},
                    'endpoints': _web_endpoints(w, blind_families[i]),
                    'violations': w.check_conservation_laws(),
                }
            rec['divergence'] = divergence_check(webs, blind_families)
        if bus is not None:
            rec['bus'] = bus.stats()
        rec['cofit'] = cofit.end_generation()

        history.append(rec)
        viols = [v for s in rec['sov'].values()
                 for v in s.get('violations', [])]
        print(f"  fitness avg={rec['avg_fitness']} best={rec['best_fitness']}"
              f" | conservation={'PASS' if not viols else viols}")
        if bus is not None:
            b = rec['bus']
            print(f"  bus: poses={b['poses']} imports={b['total_imports']} "
                  f"pending={b['pending']} "
                  f"blind_coverage={b['blind_coverage']}")

    out = {
        'arm': label, 'seed': seed, 'pop': POP, 'gens': gens,
        'blind_w': BLIND_W, 'stride': STRIDE,
        'protocol': {'q': (bus.q if bus else 0), 'match_floor': 0.3,
                     'm_receipts': 8, 'c0': 0.5, 'corr_window': 400},
        'history': history,
        'final_bus': bus.stats() if bus is not None else None,
        'pose_log_size': len(bus.pose_log) if bus is not None else 0,
        'elapsed_min': round((time.time() - t0) / 60, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'social_ledger_e1_{label}_s{seed}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    # Pose log saved separately (P65 observational data, can be large)
    if bus is not None and bus.pose_log:
        with open(os.path.join(
                RESULTS_DIR,
                f'social_ledger_e1_{label}_s{seed}_poses.json'), 'w') as f:
            json.dump(bus.pose_log, f)
    print(f"\nSaved {path} ({out['elapsed_min']}min)")
    return out


# ---------------------------------------------------------------------------
def run_identity_smoke(seed, gens=2):
    """Pre-flight: SHARE with Q=0 must be BIT-IDENTICAL to ISOLATED."""
    print("=== IDENTITY SMOKE: isolated vs share(Q=0) ===")
    a = run_arm('isolated', seed, gens=gens, tag='identity_isolated')
    b = run_arm('share', seed, gens=gens, quiet_bus=True,
                tag='identity_shareq0')

    ok = True
    for ra, rb in zip(a['history'], b['history']):
        if ra['fitness_per_org'] != rb['fitness_per_org']:
            print(f"FAIL: fitness diverges at gen {ra['generation']}: "
                  f"{ra['fitness_per_org']} vs {rb['fitness_per_org']}")
            ok = False
        sa = {k: v['stats'] for k, v in ra['sov'].items()}
        sb = {k: v['stats'] for k, v in rb['sov'].items()}
        if sa != sb:
            print(f"FAIL: web stats diverge at gen {ra['generation']}")
            print(f"  isolated: {sa}")
            print(f"  share_q0: {sb}")
            ok = False
    print(f"\nIDENTITY SMOKE: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
def analyze():
    """Verdict table per §5. Verdicts are earned: floors and identity
    checks gate every claim (SUPPORTED / NOT SUPPORTED / VOID / UNTESTED)."""
    import glob
    files = glob.glob(os.path.join(RESULTS_DIR, 'social_ledger_e1_*.json'))
    runs = {}
    for p in files:
        if p.endswith('_poses.json'):
            continue
        with open(p) as f:
            r = json.load(f)
        runs[(r['arm'], r['seed'])] = r
    if not runs:
        print("No E1 results found.")
        return

    seeds = sorted({s for (_, s) in runs})
    print(f"Runs: {sorted(runs.keys())}\n")

    for seed in seeds:
        iso = runs.get(('isolated', seed))
        sha = runs.get(('share', seed))
        if not (iso and sha):
            continue
        print(f"=== seed {seed} ===")

        # Arm-invariance assertion (VOID gate)
        invariant = all(
            ra['fitness_per_org'] == rb['fitness_per_org']
            for ra, rb in zip(iso['history'], sha['history']))
        print(f"behavior arm-invariant: {'YES' if invariant else 'NO — VOID'}")
        if not invariant:
            continue

        # Manipulation check: measured D vs mask distance (Spearman-ish:
        # mean d_metric per mask-distance grade must be increasing)
        div = sha['history'][-1].get('divergence', [])
        by_grade = {}
        for row in div:
            by_grade.setdefault(row['mask_dist'], []).append(row['d_metric'])
        grades = sorted(by_grade)
        means = [float(np.mean(by_grade[g])) for g in grades]
        monotone = all(means[k] <= means[k + 1] + 1e-12
                       for k in range(len(means) - 1))
        print(f"divergence ladder (mask_dist -> mean D): "
              f"{dict(zip(grades, [round(m, 4) for m in means]))} "
              f"monotone={'YES' if monotone else 'NO'}")

        # E1-DEMAND endpoints
        def endpoint_totals(run):
            last = run['history'][-1]['sov']
            closures = sum(v['endpoints']['closures']
                           for v in last.values() if 'endpoints' in v)
            radii = [v['endpoints']['mean_open_radius']
                     for v in last.values()
                     if v.get('endpoints', {}).get('mean_open_radius')
                     is not None]
            certs = []
            for rec in run['history']:
                for v in rec['sov'].values():
                    c = v.get('endpoints', {}).get('mean_certainty')
                    if c is not None:
                        certs.append(c)
            return (closures,
                    float(np.mean(radii)) if radii else None,
                    float(np.mean(certs)) if certs else None)

        ic, ir, iauc = endpoint_totals(iso)
        sc, sr, sauc = endpoint_totals(sha)
        print(f"E1-DEMAND  closures: share={sc} iso={ic} | "
              f"sighted radius: share={sr and round(sr,4)} "
              f"iso={ir and round(ir,4)} | "
              f"certainty: share={sauc and round(sauc,4)} "
              f"iso={iauc and round(iauc,4)}")

        # P64 floors + yield curve by mask distance
        fb = sha.get('final_bus') or {}
        total_imports = fb.get('total_imports', 0)
        ym = fb.get('yield_matrix', {})
        total_corr = sum(v['corroborated'] for v in ym.values())
        floors_met = (total_imports >= FLOOR_IMPORTS
                      and total_corr >= FLOOR_CORROBORATIONS)
        print(f"floors: imports={total_imports}/{FLOOR_IMPORTS} "
              f"corroborations={total_corr}/{FLOOR_CORROBORATIONS} "
              f"-> {'MET' if floors_met else 'NOT MET (P64 UNTESTED)'}")
        if ym:
            by_dist = {}
            for key, v in ym.items():
                i, j = key.split('<-')
                d = abs(int(i) - int(j))
                by_dist.setdefault(d, {'c': 0, 'r': 0})
                by_dist[d]['c'] += v['corroborated']
                by_dist[d]['r'] += v['resolved']
            curve = {d: round(by_dist[d]['c'] / by_dist[d]['r'], 4)
                     for d in sorted(by_dist) if by_dist[d]['r'] > 0}
            print(f"P64 yield curve (mask_dist -> yield): {curve}")

        # P75 — resonance precursor, CLEAN form: the channel-free arm.
        def pooled_cofit(run):
            m_sum = m_n = s_sum = s_n = 0.0
            for rec in run['history']:
                c = rec.get('cofit') or {}
                if c.get('matched_mean') is not None:
                    m_sum += c['matched_mean'] * c['n_matched_series']
                    m_n += c['n_matched_series']
                if c.get('shuffled_mean') is not None:
                    s_sum += c['shuffled_mean'] * c['n_shuffled_series']
                    s_n += c['n_shuffled_series']
            return ((m_sum / m_n if m_n else None), int(m_n),
                    (s_sum / s_n if s_n else None), int(s_n))

        m, mn, s, sn = pooled_cofit(iso)
        floor_met = mn >= 200 and sn >= 200
        print(f"P75 (ISOLATED, channel-free): matched={m and round(m,4)} "
              f"(n={mn}) vs shuffled={s and round(s,4)} (n={sn}) "
              f"-> {'UNTESTED (floor)' if not floor_met else ('matched > shuffled' if (m or 0) > (s or 0) else 'NOT SUPPORTED direction')}")
        print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['isolated', 'share', 'communal'])
    ap.add_argument('--seed', type=int, default=45)
    ap.add_argument('--gens', type=int, default=GENS)
    ap.add_argument('--identity', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    args = ap.parse_args()

    if args.analyze:
        analyze()
    elif args.identity:
        ok = run_identity_smoke(args.seed)
        sys.exit(0 if ok else 1)
    elif args.arm:
        run_arm(args.arm, args.seed, gens=args.gens)
    else:
        ap.print_help()

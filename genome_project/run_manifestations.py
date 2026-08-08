"""
Manifestation Test Runner

Runs the organism in each manifestation environment, measures receptor activation,
and compares against anti-manifestation environments as negative controls.

Usage:
    python -m genome_project.run_manifestations                    # Run all
    python -m genome_project.run_manifestations rhythm             # Run one receptor
    python -m genome_project.run_manifestations --stage 1          # Run all stage 1
    python -m genome_project.run_manifestations --coverage         # Show coverage report
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash, CausalMappingStore, MentalModelEngine
from mental_model import train_contrastive_encoder
from genome_project.manifester import ManifestationRegistry, RECIPES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def run_organism_in_env(env, steps=300, seed=42, num_episodes=3):
    """Run a basic organism in the given environment and collect logs.

    Returns (log, engine) where engine is a MentalModelEngine built from the log.
    """
    rng = np.random.RandomState(seed)
    all_logs = []

    for ep in range(num_episodes):
        org = Organism()
        org.x = rng.uniform(3, 17)
        org.y = rng.uniform(3, 17)
        env.reset(seed=seed + ep * 100)

        npc = None
        npcs = getattr(env, 'npcs', [])
        if npcs:
            npc = npcs[0]
            npc.reset(rng)

        for t in range(steps):
            tips = org.get_limb_tips()
            pain, endo = env.get_field_values(tips, t)
            temps = env.get_temperature_values(tips, t)
            chem = env.get_chemical_values(tips, t)
            press = env.get_pressure_values(tips)

            gx, gy = env.get_combined_gradient(org.x, org.y, t)
            action = _gradient_to_action(org, gx, gy, rng)
            emission = action[18:22] if len(action) >= 22 else np.zeros(4)

            for obj in getattr(env, 'responsive_objects', []):
                obj.update(org.x, org.y, emission, t)

            if hasattr(env, 'step_extras'):
                env.step_extras(org.x, org.y, t)

            obs_before = org.history[-1].copy() if len(org.history) > 0 else np.zeros(org.OBS_DIM)

            mm_features = np.zeros(4)
            pattern_features = np.zeros(2)
            agency_features = np.zeros(3)

            org.step(action, env, t, mm_features=mm_features,
                     pattern_features=pattern_features,
                     agency_features=agency_features,
                     npc=npc)

            obs_after = org.history[-1].copy() if len(org.history) > 0 else np.zeros(org.OBS_DIM)

            reward = float(np.mean(endo) - np.mean(pain))

            all_logs.append({
                'step': t,
                'episode': ep,
                'obs_before': obs_before,
                'obs_after': obs_after,
                'action': action,
                'reward': reward,
                'x': org.x, 'y': org.y,
                'pain_mean': float(np.mean(pain)),
                'endo_mean': float(np.mean(endo)),
                'temp_mean': float(np.mean(temps)),
                'chem_mean': float(np.mean(chem)),
                'press_mean': float(np.mean(press)),
            })

    engine = _build_engine_from_log(all_logs)
    return all_logs, engine


def _build_engine_from_log(log, obs_dim=96):
    """Build a mental model engine from the collected log."""
    if len(log) < 10:
        store = CausalMappingStore()
        return MentalModelEngine(None, store)

    global_log = []
    for entry in log:
        obs_b = entry['obs_before']
        obs_a = entry['obs_after']
        action = entry['action']
        if isinstance(obs_b, list):
            obs_b = np.array(obs_b, dtype=np.float32)
        if isinstance(obs_a, list):
            obs_a = np.array(obs_a, dtype=np.float32)
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        global_log.append({
            'obs_before': obs_b[:obs_dim],
            'obs_after': obs_a[:obs_dim],
            'action': action,
            'reward': entry.get('reward', 0.0),
        })

    try:
        encoder = train_contrastive_encoder(global_log, obs_dim=obs_dim, epochs=5)
    except Exception:
        encoder = None

    store = CausalMappingStore()
    if encoder is not None:
        store.build_from_log(global_log, encoder)

    return MentalModelEngine(encoder, store)


def _gradient_to_action(org, gx, gy, rng, explore_rate=0.07):
    """Convert environment gradient to action vector with exploration."""
    action = np.zeros(22, dtype=np.float32)

    if rng.random() < explore_rate:
        action[:18] = rng.randint(0, 2, size=18).astype(np.float32)
        action[18:22] = rng.randint(0, 2, size=4).astype(np.float32)
        return action

    angle = np.arctan2(gy, gx) if abs(gx) + abs(gy) > 0.01 else rng.uniform(0, 2 * np.pi)
    for i in range(6):
        limb_angle = org.BASE_ANGLES[i] + org.heading
        diff = angle - limb_angle
        extend = np.cos(diff) > 0
        action[i * 3] = 1.0 if extend else 0.0
        action[i * 3 + 1] = 1.0 if np.sin(diff) > 0.3 else 0.0
        action[i * 3 + 2] = 1.0 if np.sin(diff) < -0.3 else 0.0

    return action


def compute_manifestation_scores(log, engine):
    """Compute basic receptor-relevant scores from a run log."""
    if not log:
        return {}

    N = len(log)
    pain_series = np.array([e['pain_mean'] for e in log])
    endo_series = np.array([e['endo_mean'] for e in log])
    temp_series = np.array([e['temp_mean'] for e in log])
    chem_series = np.array([e['chem_mean'] for e in log])

    scores = {}

    scores['mean_pain'] = float(np.mean(pain_series))
    scores['mean_endorphin'] = float(np.mean(endo_series))
    scores['pain_variance'] = float(np.var(pain_series))
    scores['endo_variance'] = float(np.var(endo_series))
    scores['temp_variance'] = float(np.var(temp_series))
    scores['chem_variance'] = float(np.var(chem_series))

    if N > 10:
        pain_delta = np.abs(np.diff(pain_series))
        scores['mean_pain_change'] = float(np.mean(pain_delta))
        scores['max_pain_change'] = float(np.max(pain_delta))
    else:
        scores['mean_pain_change'] = 0.0
        scores['max_pain_change'] = 0.0

    if N > 100:
        for lag in [50, 75, 100, 125]:
            if lag < N:
                pain_norm = (pain_series - pain_series.mean()) / (pain_series.std() + 1e-8)
                ac = np.corrcoef(pain_norm[:-lag], pain_norm[lag:])[0, 1]
                if not np.isnan(ac):
                    scores[f'pain_autocorr_lag{lag}'] = float(abs(ac))

    all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
    if all_entries:
        scores['mm_entries'] = len(all_entries)
        scores['mm_mean_certainty'] = float(np.mean([e.certainty for e in all_entries]))
        high = [e for e in all_entries if e.count >= 5]
        low = [e for e in all_entries if e.count <= 2]
        if high and low:
            scores['certainty_ratio'] = float(
                np.mean([e.certainty for e in high]) /
                (np.mean([e.certainty for e in low]) + 1e-8))
    else:
        scores['mm_entries'] = 0
        scores['mm_mean_certainty'] = 0.0

    x_coords = [e['x'] for e in log]
    y_coords = [e['y'] for e in log]
    scores['spatial_range_x'] = float(max(x_coords) - min(x_coords))
    scores['spatial_range_y'] = float(max(y_coords) - min(y_coords))
    scores['mean_x'] = float(np.mean(x_coords))
    scores['mean_y'] = float(np.mean(y_coords))

    return scores


def run_receptor_manifestations(receptor_id, seed=42, steps=None, verbose=True):
    """Run all manifestation and anti-manifestation environments for one receptor."""
    registry = ManifestationRegistry()
    envs = registry.get_environments(receptor_id)
    anti_envs = registry.get_anti_environments(receptor_id)

    if not envs:
        if verbose:
            print(f"  No recipes for {receptor_id}")
        return None

    results = {
        'receptor_id': receptor_id,
        'manifestations': [],
        'anti_manifestations': [],
    }

    for cfg in envs:
        ep_steps = steps or cfg.steps_per_episode
        if verbose:
            print(f"  Running {receptor_id}/{cfg.manifestation_id} "
                  f"({ep_steps} steps x {cfg.num_episodes} episodes)...")
        t0 = time.time()
        env = cfg.build(seed=seed)
        log, engine = run_organism_in_env(env, steps=ep_steps,
                                          seed=seed, num_episodes=cfg.num_episodes)
        scores = compute_manifestation_scores(log, engine)
        elapsed = time.time() - t0

        results['manifestations'].append({
            'id': cfg.manifestation_id,
            'scores': scores,
            'elapsed_s': round(elapsed, 2),
            'steps': ep_steps,
            'episodes': cfg.num_episodes,
        })
        if verbose:
            print(f"    pain={scores.get('mean_pain', 0):.3f} "
                  f"endo={scores.get('mean_endorphin', 0):.3f} "
                  f"pain_var={scores.get('pain_variance', 0):.4f} "
                  f"mm_entries={scores.get('mm_entries', 0)} "
                  f"({elapsed:.1f}s)")

    for cfg in anti_envs:
        ep_steps = steps or cfg.steps_per_episode
        if verbose:
            print(f"  Running {receptor_id}/{cfg.manifestation_id} (ANTI) "
                  f"({ep_steps} steps)...")
        t0 = time.time()
        env = cfg.build(seed=seed)
        log, engine = run_organism_in_env(env, steps=ep_steps,
                                          seed=seed, num_episodes=cfg.num_episodes)
        scores = compute_manifestation_scores(log, engine)
        elapsed = time.time() - t0

        results['anti_manifestations'].append({
            'id': cfg.manifestation_id,
            'scores': scores,
            'elapsed_s': round(elapsed, 2),
            'is_anti': True,
        })
        if verbose:
            print(f"    (anti) pain={scores.get('mean_pain', 0):.3f} "
                  f"endo={scores.get('mean_endorphin', 0):.3f} "
                  f"pain_var={scores.get('pain_variance', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run manifestation environments')
    parser.add_argument('receptors', nargs='*', help='Specific receptor IDs to run')
    parser.add_argument('--stage', type=int, help='Run all receptors at this stage')
    parser.add_argument('--coverage', action='store_true', help='Show coverage report')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=None, help='Override steps per episode')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    args = parser.parse_args()

    registry = ManifestationRegistry()

    if args.coverage:
        report = registry.coverage_report()
        print(f"Manifestation Environment Coverage")
        print(f"===================================")
        print(f"Total manifestation YAMLs: {report['total_yamls']}")
        print(f"Receptors with recipes:    {report['with_recipes']}")
        print(f"Coverage:                  {report['coverage_pct']}%")
        print()

        by_stage = defaultdict(list)
        for r in report['covered']:
            info = registry.get_yaml_info(r)
            stage = info['stage'] if info else 0
            by_stage[stage].append(r)
        for stage in sorted(by_stage):
            print(f"Stage {stage}: {len(by_stage[stage])} receptors")
            for r in by_stage[stage]:
                envs = registry.get_environments(r)
                anti = registry.get_anti_environments(r)
                print(f"  {r}: {len(envs)} env, {len(anti)} anti")
        return

    if args.receptors:
        target_receptors = args.receptors
    elif args.stage is not None:
        target_receptors = []
        for r in registry.list_receptors_with_recipes():
            info = registry.get_yaml_info(r)
            if info and info.get('stage') == args.stage:
                target_receptors.append(r)
    else:
        target_receptors = registry.list_receptors_with_recipes()

    print(f"Running {len(target_receptors)} receptor manifestation batteries...")
    print()

    all_results = {}
    for receptor_id in target_receptors:
        print(f"[{receptor_id}]")
        result = run_receptor_manifestations(
            receptor_id, seed=args.seed, steps=args.steps)
        if result:
            all_results[receptor_id] = result
        print()

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        outpath = os.path.join(RESULTS_DIR, 'manifestation_results.json')
        with open(outpath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {outpath}")

    passed = sum(1 for r in all_results.values() if r['manifestations'])
    total = len(all_results)
    print(f"\nSummary: {passed}/{total} receptors had successful runs")


if __name__ == '__main__':
    main()

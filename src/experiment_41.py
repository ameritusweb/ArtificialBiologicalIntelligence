import os
import json
import time
import numpy as np
import torch
from environment import Organism, NPC
from environment_tiers import create_environment
from model import HierarchicalPolicy, compute_obs_indices
from mental_model import build_mental_model
from train import (augment_with_mental_model, augment_with_patterns,
                   augment_with_agency, augment_with_concepts, DEVICE)
from receptor_discovery import build_tests, discover
from topology_inheritance import BIAS_FACTOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_deep_sweep(tiers=None, generations=3, episodes=100):
    if tiers is None:
        tiers = [0, 1, 2, 3, 4]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 41: Deep Sweep — {len(tiers)} tiers x {generations} gen x 40 tests")
    print(f"  Deepened environments + expanded receptor discovery")
    print(f"{'='*70}")

    idx = compute_obs_indices(6)
    base_tests = build_tests()
    print(f"  Receptor tests: {len(base_tests)}")

    sweep = {}
    for tier in tiers:
        print(f"\n  {'='*55}")
        print(f"  TIER {tier} (deepened)")
        print(f"  {'='*55}")

        parent_topology = None
        parent_weights = None
        tier_history = []

        for gen in range(generations):
            print(f"    Gen {gen}...", end=" ", flush=True)
            seed = tier * 100000 + gen * 10000
            rng = np.random.RandomState(seed)

            all_windows, all_targets, all_next_pain = [], [], []
            global_log = []

            for ep in range(episodes):
                env = create_environment(tier=tier, seed=rng.randint(0, 100000))
                org = Organism()
                org.reset(rng)
                npc = NPC()
                npc.reset(rng)
                episode_pain = []

                for step in range(300):
                    npc.step(env, step)
                    env.step_tier(org.x, org.y, step)
                    optimal = org.compute_optimal_actions(env, step, npc=npc)
                    window = org.get_observation_window()
                    all_windows.append(window.copy())
                    all_targets.append(optimal.copy())
                    obs, reward = org.step(optimal, env, step, npc=npc)
                    npc.receive_signal(optimal[org.NUM_LIMBS * 3:], org.x, org.y)
                    episode_pain.append(obs[0:org.NUM_LIMBS].copy())

                for i in range(300):
                    all_next_pain.append(episode_pain[min(i+1, 299)])
                global_log.extend(org.experience_log)

            X = np.array(all_windows, dtype=np.float32)
            Y = np.array(all_targets, dtype=np.float32)
            Z = np.array(all_next_pain, dtype=np.float32)

            engine = build_mental_model(global_log[:min(50000, len(global_log))],
                                        core_obs_dim=idx['core_obs_dim'])

            X = augment_with_mental_model(X, global_log, engine, 300,
                                          idx['core_obs_dim'], idx['mm_start'])
            X = augment_with_patterns(X, global_log, engine, 300,
                                      idx['core_obs_dim'], idx['pattern_start'])
            X = augment_with_agency(X, global_log, engine, 300,
                                    idx['core_obs_dim'], idx['agency_start'])
            cs = idx['obs_dim'] - 2
            X = augment_with_concepts(X, global_log, engine, 300,
                                      idx['core_obs_dim'], cs)

            num_pain = Z.shape[1]
            model = HierarchicalPolicy(
                obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
                energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict'],
                num_pain_channels=num_pain, num_limbs=6
            ).to(DEVICE)

            if parent_weights is not None:
                curr = model.state_dict()
                for k in curr:
                    if k in parent_weights and curr[k].shape == parent_weights[k].shape:
                        curr[k] = 0.8 * parent_weights[k] + 0.2 * curr[k]
                model.load_state_dict(curr)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
            bce = torch.nn.BCEWithLogitsLoss()
            N = X.shape[0]
            indices = np.arange(N)
            final_acc = 0.0

            for epoch in range(10):
                np.random.shuffle(indices)
                model.train()
                total_acc, n_b = 0, 0
                for start in range(0, N, 256):
                    batch_idx = indices[start:start+256]
                    x_b = torch.FloatTensor(X[batch_idx]).to(DEVICE)
                    y_b = torch.FloatTensor(Y[batch_idx]).to(DEVICE)
                    result = model(x_b)
                    loss = bce(result['blended'], y_b)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    with torch.no_grad():
                        preds = (torch.sigmoid(result['blended']) > 0.5).float()
                        total_acc += (preds == y_b).float().mean().item()
                    n_b += 1
                scheduler.step()
                final_acc = total_acc / n_b

            biased = None
            if parent_topology is not None:
                biased = {}
                for i, test in enumerate(base_tests):
                    if i < len(parent_topology) and parent_topology[i] == 1:
                        biased[test.receptor_id] = test.threshold * BIAS_FACTOR

            log_sample = global_log[:min(50000, len(global_log))]
            results = discover(log_sample, engine, threshold_overrides=biased)
            parent_topology = results['topology_vector']
            parent_weights = {k: v.clone() for k, v in model.state_dict().items()}

            tier_history.append({
                'generation': gen,
                'accuracy': round(final_acc, 4),
                'discovered': results['discovered'],
                'discovered_count': len(results['discovered']),
                'scores': {k: round(v, 4) for k, v in results['scores'].items()},
            })
            print(f"acc={final_acc:.3f}  discovered={len(results['discovered'])}")

        sweep[f'tier_{tier}'] = tier_history

    receptor_ids = [t.receptor_id for t in base_tests]

    print(f"\n\n{'='*70}")
    print(f"  DEEP SWEEP EMERGENCE MATRIX (40 tests, deepened environments)")
    print(f"{'='*70}")
    header = f"  {'Receptor':<30}" + "".join(f"{'T'+str(t):>8}" for t in tiers)
    print(header)
    print(f"  {'-'*30}" + "-"*8*len(tiers))

    emergence = {}
    for rid in receptor_ids:
        row = {}
        for tier in tiers:
            key = f'tier_{tier}'
            first_gen = None
            for g in sweep[key]:
                if rid in g['discovered']:
                    first_gen = g['generation']
                    break
            row[f'tier_{tier}'] = first_gen
        emergence[rid] = row

        cells = ""
        for tier in tiers:
            val = row[f'tier_{tier}']
            cells += f"{'G'+str(val):>8}" if val is not None else f"{'--':>8}"
        print(f"  {rid:<30}{cells}")

    invariant = [rid for rid in receptor_ids
                 if all(emergence[rid].get(f'tier_{t}') is not None for t in tiers)]
    tier_specific = {}
    for rid in receptor_ids:
        present = [t for t in tiers if emergence[rid].get(f'tier_{t}') is not None]
        absent = [t for t in tiers if emergence[rid].get(f'tier_{t}') is None]
        if present and absent:
            tier_specific[rid] = {'present': present, 'absent': absent}
    never = [rid for rid in receptor_ids
             if all(emergence[rid].get(f'tier_{t}') is None for t in tiers)]

    counts = {t: sweep[f'tier_{t}'][-1]['discovered_count'] for t in tiers}

    print(f"\n  ANALYSIS")
    print(f"  {'='*55}")
    print(f"  Invariant (all tiers):  {len(invariant)} receptors")
    print(f"  Tier-specific:          {len(tier_specific)} receptors")
    print(f"  Never emerged:          {len(never)} receptors")
    print(f"  Counts by tier:         {counts}")
    print(f"\n  Invariant: {invariant}")
    print(f"  Never:     {never}")
    for rid, info in tier_specific.items():
        print(f"  {rid}: present={info['present']} absent={info['absent']}")

    original_never = {'static_repetition', 'precedence_detection',
                      'causal_association', 'curiosity'}
    now_found = original_never - set(never)
    still_missing = original_never & set(never)
    print(f"\n  COMPARISON WITH ORIGINAL SWEEP (step 34)")
    print(f"  Previously never emerged, NOW found: {sorted(now_found) if now_found else 'none'}")
    print(f"  Still missing:                       {sorted(still_missing)}")

    new_tests = set(receptor_ids) - {'static_repetition', 'rhythm', 'coincidence_detection',
        'precedence_detection', 'causal_association', 'spatial_association',
        'temporal_association', 'perceptual_similarity', 'pattern_recognition',
        'compression_gain', 'other_detection', 'empathic_concern', 'controllability',
        'planning', 'curiosity', 'optimism', 'conflict_detection', 'stress_detection',
        'change_detection'}
    new_found = [rid for rid in new_tests
                 if any(emergence[rid].get(f'tier_{t}') is not None for t in tiers)]
    print(f"  New tests (step 35) that emerged: {sorted(new_found)}")

    return {
        'sweep': {k: v for k, v in sweep.items()},
        'emergence': emergence,
        'invariant': invariant,
        'tier_specific': tier_specific,
        'never': never,
        'counts': counts,
    }


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    results = run_deep_sweep(tiers=[0, 1, 2, 3, 4, 5, 6, 7], generations=3, episodes=100)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} min")

    path = os.path.join(DATA_DIR, 'experiment_41_deep_sweep.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")
    print("\nExperiment 41 complete.")

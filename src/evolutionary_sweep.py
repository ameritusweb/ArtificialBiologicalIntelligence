import os
import json
import time
import numpy as np
import torch
from environment import Organism, NPC
from environment_tiers import create_environment
from model import HierarchicalPolicy, compute_obs_indices
from mental_model import build_mental_model, action_to_hash
from train import (augment_with_mental_model, augment_with_patterns,
                   augment_with_agency, augment_with_concepts, DEVICE)
from receptor_discovery import build_tests, discover
from topology_inheritance import BIAS_FACTOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def generate_tiered_data(tier, num_episodes=100, steps_per_episode=300, seed=0):
    rng = np.random.RandomState(seed)
    all_windows, all_targets, all_next_pain = [], [], []
    global_log = []

    for ep in range(num_episodes):
        env = create_environment(tier=tier, seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        episode_pain = []

        for step in range(steps_per_episode):
            npc.step(env, step)
            env.step_tier(org.x, org.y, step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(optimal.copy())
            obs, reward = org.step(optimal, env, step, npc=npc)
            npc.receive_signal(optimal[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:org.NUM_LIMBS].copy())

        for i in range(steps_per_episode):
            if i + 1 < steps_per_episode:
                all_next_pain.append(episode_pain[i + 1])
            else:
                all_next_pain.append(episode_pain[i])
        global_log.extend(org.experience_log)

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    return X, Y, Z, global_log


def run_tier_generations(tier, n_generations=3, episodes=100):
    print(f"\n  {'='*55}")
    print(f"  TIER {tier}: {n_generations} generations, {episodes} episodes")
    print(f"  {'='*55}")

    idx = compute_obs_indices(6)
    base_tests = build_tests()
    parent_topology = None
    parent_weights = None
    tier_history = []

    for gen in range(n_generations):
        print(f"    Gen {gen}...", end=" ", flush=True)
        seed = tier * 100000 + gen * 10000

        X, Y, Z, log = generate_tiered_data(tier, episodes, 300, seed)
        engine = build_mental_model(log, core_obs_dim=idx['core_obs_dim'])

        X = augment_with_mental_model(X, log, engine, 300,
                                      idx['core_obs_dim'], idx['mm_start'])
        X = augment_with_patterns(X, log, engine, 300,
                                  idx['core_obs_dim'], idx['pattern_start'])
        X = augment_with_agency(X, log, engine, 300,
                                idx['core_obs_dim'], idx['agency_start'])
        cs = idx['obs_dim'] - 2
        X = augment_with_concepts(X, log, engine, 300,
                                  idx['core_obs_dim'], cs)

        num_pain = Z.shape[1] if Z.ndim > 1 else 6
        model = HierarchicalPolicy(
            obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
            energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict'],
            num_pain_channels=num_pain, num_limbs=6
        ).to(DEVICE)

        if parent_weights is not None:
            prev_state = parent_weights
            curr_state = model.state_dict()
            for key in curr_state:
                if key in prev_state and curr_state[key].shape == prev_state[key].shape:
                    curr_state[key] = 0.8 * prev_state[key] + 0.2 * curr_state[key]
            model.load_state_dict(curr_state)

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

        results = discover(log, engine, threshold_overrides=biased)
        parent_topology = results['topology_vector']
        parent_weights = {k: v.clone() for k, v in model.state_dict().items()}

        tier_history.append({
            'generation': gen,
            'accuracy': round(final_acc, 4),
            'discovered': results['discovered'],
            'discovered_count': len(results['discovered']),
            'scores': results['scores'],
        })
        print(f"acc={final_acc:.3f}  discovered={len(results['discovered'])}: "
              f"{results['discovered']}")

    return tier_history


def run_sweep(tiers=None, generations=3, episodes=100):
    if tiers is None:
        tiers = [0, 1, 2, 3, 4]

    print(f"\n{'='*70}")
    print(f"  EVOLUTIONARY SWEEP: {len(tiers)} tiers x {generations} generations")
    print(f"{'='*70}")

    sweep_results = {}
    for tier in tiers:
        sweep_results[f'tier_{tier}'] = run_tier_generations(tier, generations, episodes)

    base_tests = build_tests()
    receptor_ids = [t.receptor_id for t in base_tests]

    print(f"\n\n{'='*70}")
    print(f"  EMERGENCE MATRIX")
    print(f"{'='*70}")
    header = f"  {'Receptor':<25}" + "".join(f"{'T'+str(t):>8}" for t in tiers)
    print(header)
    print(f"  {'-'*25}" + "-"*8*len(tiers))

    emergence_matrix = {}
    for rid in receptor_ids:
        row = {}
        for tier in tiers:
            key = f'tier_{tier}'
            first_gen = None
            for g in sweep_results[key]:
                if rid in g['discovered']:
                    first_gen = g['generation']
                    break
            row[f'tier_{tier}'] = first_gen
        emergence_matrix[rid] = row

        cells = ""
        for tier in tiers:
            val = row[f'tier_{tier}']
            cells += f"{'G'+str(val):>8}" if val is not None else f"{'—':>8}"
        print(f"  {rid:<25}{cells}")

    tier_invariant = [rid for rid in receptor_ids
                      if all(emergence_matrix[rid].get(f'tier_{t}') is not None for t in tiers)]
    tier_specific = {}
    for rid in receptor_ids:
        present_at = [t for t in tiers if emergence_matrix[rid].get(f'tier_{t}') is not None]
        absent_at = [t for t in tiers if emergence_matrix[rid].get(f'tier_{t}') is None]
        if present_at and absent_at:
            tier_specific[rid] = {'present': present_at, 'absent': absent_at}

    counts_by_tier = {}
    for tier in tiers:
        key = f'tier_{tier}'
        final_gen = sweep_results[key][-1]
        counts_by_tier[tier] = final_gen['discovered_count']

    print(f"\n  ANALYSIS")
    print(f"  {'='*55}")
    print(f"  Tier-invariant (trunk): {tier_invariant}")
    print(f"  Tier-specific receptors:")
    for rid, info in tier_specific.items():
        print(f"    {rid}: present at tiers {info['present']}, absent at {info['absent']}")
    print(f"  Discovery count by tier: {counts_by_tier}")
    print(f"  Complexity scaling: {'YES' if counts_by_tier.get(max(tiers), 0) >= counts_by_tier.get(min(tiers), 0) else 'NO'}")

    return {
        'sweep_results': {k: v for k, v in sweep_results.items()},
        'emergence_matrix': emergence_matrix,
        'tier_invariant': tier_invariant,
        'tier_specific': tier_specific,
        'counts_by_tier': counts_by_tier,
    }


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()

    results = run_sweep(tiers=[0, 1, 2, 3, 4], generations=3, episodes=100)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} min")

    path = os.path.join(DATA_DIR, 'evolutionary_sweep.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")
    print("\nStep 34 complete. The full ERTI roadmap is realized.")

import os
import json
import time
import numpy as np
import torch
from environment import Environment, Organism, NPC
from environment_tiers import create_environment
from model import HierarchicalPolicy, compute_obs_indices
from mental_model import build_mental_model
from train import (augment_with_mental_model, augment_with_patterns,
                   augment_with_agency, augment_with_concepts, DEVICE)
from receptor_discovery import discover
from topology_inheritance import BIAS_FACTOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def generate_tier_data(tier, num_episodes=100, steps=300, seed=0):
    rng = np.random.RandomState(seed)
    all_windows, all_targets, all_next_pain = [], [], []
    global_log = []
    total_reward = 0.0

    for ep in range(num_episodes):
        env = create_environment(tier=tier, seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        episode_pain = []

        for step in range(steps):
            npc.step(env, step)
            env.step_tier(org.x, org.y, step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(optimal.copy())
            obs, reward = org.step(optimal, env, step, npc=npc)
            npc.receive_signal(optimal[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:org.NUM_LIMBS].copy())
            total_reward += reward

        for i in range(steps):
            all_next_pain.append(episode_pain[min(i+1, steps-1)])
        global_log.extend(org.experience_log)

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    return X, Y, Z, global_log, total_reward / num_episodes


def train_in_tier(tier, generations=3, episodes=100):
    idx = compute_obs_indices(6)
    parent_weights = None

    for gen in range(generations):
        X, Y, Z, log, avg_reward = generate_tier_data(tier, episodes, 300,
                                                        seed=tier*100000 + gen*10000)
        engine = build_mental_model(log[:min(50000, len(log))],
                                    core_obs_dim=idx['core_obs_dim'])
        X = augment_with_mental_model(X, log, engine, 300,
                                      idx['core_obs_dim'], idx['mm_start'])
        X = augment_with_patterns(X, log, engine, 300,
                                  idx['core_obs_dim'], idx['pattern_start'])
        X = augment_with_agency(X, log, engine, 300,
                                idx['core_obs_dim'], idx['agency_start'])
        cs = idx['obs_dim'] - 2
        X = augment_with_concepts(X, log, engine, 300, idx['core_obs_dim'], cs)

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
        bce = torch.nn.BCEWithLogitsLoss()
        N = X.shape[0]
        indices = np.arange(N)
        for epoch in range(10):
            np.random.shuffle(indices)
            model.train()
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

        parent_weights = {k: v.clone() for k, v in model.state_dict().items()}

    topology = discover(log[:min(50000, len(log))], engine)
    return parent_weights, topology['discovered'], avg_reward


def evaluate_in_tier(tier, model_weights, episodes=50):
    idx = compute_obs_indices(6)
    model = HierarchicalPolicy(
        obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
        energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict'],
        num_pain_channels=6, num_limbs=6
    ).to(DEVICE)
    if model_weights is not None:
        model.load_state_dict(model_weights)
    model.eval()

    total_reward = 0.0
    rng = np.random.RandomState(9999 + tier)

    for ep in range(episodes):
        env = create_environment(tier=tier, seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        for step in range(300):
            npc.step(env, step)
            env.step_tier(org.x, org.y, step)
            window = org.get_observation_window()
            actions, _ = model.predict(window)
            obs, reward = org.step(actions, env, step, npc=npc)
            npc.receive_signal(actions[org.NUM_LIMBS * 3:], org.x, org.y)
            total_reward += reward

    return total_reward / episodes


def oracle_baseline_in_tier(tier, episodes=50):
    total_reward = 0.0
    rng = np.random.RandomState(9999 + tier)

    for ep in range(episodes):
        env = create_environment(tier=tier, seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        for step in range(300):
            npc.step(env, step)
            env.step_tier(org.x, org.y, step)
            actions = org.compute_optimal_actions(env, step, npc=npc)
            obs, reward = org.step(actions, env, step, npc=npc)
            npc.receive_signal(actions[org.NUM_LIMBS * 3:], org.x, org.y)
            total_reward += reward

    return total_reward / episodes


def run_transfer_matrix(tiers=None):
    if tiers is None:
        tiers = [0, 1, 2, 3, 4]

    print(f"\n{'='*70}")
    print(f"  CROSS-TIER TRANSFER MATRIX")
    print(f"{'='*70}")

    trained = {}
    for src in tiers:
        print(f"\n  Training in Tier {src} (3 generations)...", flush=True)
        weights, discovered, train_reward = train_in_tier(src, generations=3, episodes=100)
        trained[src] = {'weights': weights, 'discovered': discovered, 'train_reward': train_reward}
        print(f"    Reward: {train_reward:.1f}, Discovered: {len(discovered)}")

    print(f"\n  Computing oracle baselines...", flush=True)
    baselines = {}
    for tgt in tiers:
        baselines[tgt] = oracle_baseline_in_tier(tgt, episodes=50)
        print(f"    Tier {tgt} oracle: {baselines[tgt]:.1f}")

    print(f"\n  Evaluating transfers...", flush=True)
    matrix = {}
    for src in tiers:
        matrix[src] = {}
        for tgt in tiers:
            reward = evaluate_in_tier(tgt, trained[src]['weights'], episodes=50)
            baseline = baselines[tgt]
            ratio = reward / baseline if abs(baseline) > 1.0 else 0.0
            matrix[src][tgt] = {
                'reward': round(reward, 1),
                'baseline': round(baseline, 1),
                'ratio': round(ratio, 3),
            }
        src_str = " ".join(f"{matrix[src][t]['ratio']:>7.3f}" for t in tiers)
        print(f"    T{src} -> [{src_str}]")

    print(f"\n  {'TRANSFER MATRIX (ratio: transferred / oracle baseline)':}")
    print(f"  {'Source':>8}" + "".join(f"{'T'+str(t):>8}" for t in tiers))
    print(f"  {'-'*8}" + "-"*8*len(tiers))
    for src in tiers:
        cells = "".join(f"{matrix[src][t]['ratio']:>8.3f}" for t in tiers)
        print(f"  {'T'+str(src):>8}{cells}")

    print(f"\n  ANALYSIS")
    for src in tiers:
        for tgt in tiers:
            if src != tgt:
                r = matrix[src][tgt]['ratio']
                rev = matrix[tgt][src]['ratio']
                if abs(r - rev) > 0.1:
                    direction = "helps more" if r > rev else "helps less"
                    print(f"    Asymmetry: T{src}->T{tgt} ({r:.3f}) vs T{tgt}->T{src} ({rev:.3f}) — T{src} {direction}")

    helps = [(src, tgt, matrix[src][tgt]['ratio'])
             for src in tiers for tgt in tiers if src != tgt and matrix[src][tgt]['ratio'] > 1.0]
    hurts = [(src, tgt, matrix[src][tgt]['ratio'])
             for src in tiers for tgt in tiers if src != tgt and matrix[src][tgt]['ratio'] < 0.8]
    print(f"\n    Transfer helps ({len(helps)} pairs): "
          + ", ".join(f"T{s}->T{t}({r:.2f})" for s, t, r in helps[:5]))
    print(f"    Transfer hurts ({len(hurts)} pairs): "
          + ", ".join(f"T{s}->T{t}({r:.2f})" for s, t, r in hurts[:5]))

    return {
        'matrix': {f'T{s}->T{t}': matrix[s][t] for s in tiers for t in tiers},
        'baselines': {f'T{t}': baselines[t] for t in tiers},
        'trained_discoveries': {f'T{t}': trained[t]['discovered'] for t in tiers},
    }


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    results = run_transfer_matrix(tiers=[0, 1, 2, 3, 4])
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} min")

    path = os.path.join(DATA_DIR, 'cross_tier_transfer.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")
    print("\nStep 38 complete.")

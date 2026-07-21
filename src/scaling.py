import os
import json
import time
import numpy as np
import torch
from environment import Environment, Organism, NPC
from model import HierarchicalPolicy, compute_obs_indices
from mental_model import build_mental_model, action_to_hash
from train import (generate_training_data, augment_with_mental_model,
                   augment_with_patterns, augment_with_agency, train_model,
                   run_inference_episode, run_baseline_episode, DEVICE)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def run_scaled_pipeline(num_limbs=6, num_episodes=100, steps_per_episode=300,
                        epochs=15, seed=0, num_segments=1, dims=2):
    idx = compute_obs_indices(num_limbs, num_segments, dims)
    print(f"\n{'='*60}")
    print(f"  SCALING: {num_limbs}L x {num_segments}S x {dims}D | obs={idx['obs_dim']} | actions={idx['num_actions']}")
    print(f"{'='*60}")

    rng = np.random.RandomState(seed)
    all_windows, all_targets, all_next_pain = [], [], []
    global_log = []

    for ep in range(num_episodes):
        env = Environment(seed=rng.randint(0, 100000))
        total_limbs = num_segments * num_limbs
        org = Organism(num_limbs=num_limbs, num_segments=num_segments, dims=dims)
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        episode_pain = []
        em_start = total_limbs * 3 + (num_segments - 1) * 2
        for step in range(steps_per_episode):
            npc.step(env, step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(optimal.copy())
            obs, reward = org.step(optimal, env, step, npc=npc)
            npc.receive_signal(optimal[em_start:], org.x, org.y)
            episode_pain.append(obs[0:total_limbs].copy())
        for i in range(steps_per_episode):
            if i + 1 < steps_per_episode:
                all_next_pain.append(episode_pain[i + 1])
            else:
                all_next_pain.append(episode_pain[i])
        global_log.extend(org.experience_log)
        if (ep + 1) % 50 == 0:
            print(f"    Generated {ep+1}/{num_episodes} episodes")

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    print(f"  Data: X={X.shape}, Y={Y.shape}")

    print("  Building mental model...")
    engine = build_mental_model(global_log, core_obs_dim=idx['core_obs_dim'])
    stats = engine.get_stats()
    print(f"    Mappings: {stats['total_mappings']:,}")

    print("  Augmenting...")
    X = augment_with_mental_model(X, global_log, engine, steps_per_episode,
                                  idx['core_obs_dim'], idx['mm_start'])
    X = augment_with_patterns(X, global_log, engine, steps_per_episode,
                              idx['core_obs_dim'], idx['pattern_start'])
    X = augment_with_agency(X, global_log, engine, steps_per_episode,
                            idx['core_obs_dim'], idx['agency_start'])

    print(f"  Training ({epochs} epochs)...")
    model = train_model(X, Y, Z, epochs=epochs, batch_size=256, lr=1e-3,
                        num_limbs=num_limbs, num_segments=num_segments, dims=dims)

    print("  Evaluating...")
    total_limbs = num_segments * num_limbs
    em_start = total_limbs * 3 + (num_segments - 1) * 2
    baselines, model_rewards = [], []
    for i in range(3):
        env = Environment(seed=2000 + i)
        eval_rng = np.random.RandomState(2000 + i)
        org = Organism(num_limbs=num_limbs, num_segments=num_segments, dims=dims)
        org.reset(eval_rng)
        npc_b = NPC()
        npc_b.reset(eval_rng)
        total_b = 0.0
        for step in range(300):
            npc_b.step(env, step)
            actions = org.compute_optimal_actions(env, step, npc=npc_b)
            _, reward = org.step(actions, env, step, npc=npc_b)
            npc_b.receive_signal(actions[em_start:], org.x, org.y)
            total_b += reward
        baselines.append(total_b)

    for i in range(3):
        env = Environment(seed=2000 + i)
        eval_rng = np.random.RandomState(2000 + i)
        frames, model_reward = run_inference_episode(model, env, 300, eval_rng, engine,
                                                      num_limbs=num_limbs,
                                                      num_segments=num_segments,
                                                      dims=dims)
        model_rewards.append(model_reward)

    action_entropy = float(-np.mean(Y * np.log(Y + 1e-8) + (1-Y) * np.log(1-Y + 1e-8)))

    result = {
        'num_limbs': num_limbs,
        'num_segments': num_segments,
        'dims': dims,
        'obs_dim': idx['obs_dim'],
        'num_actions': idx['num_actions'],
        'baseline_reward': round(np.mean(baselines), 1),
        'model_reward': round(np.mean(model_rewards), 1),
        'mappings': stats['total_mappings'],
        'action_entropy': round(action_entropy, 4),
    }
    print(f"  Result: baseline={result['baseline_reward']:.1f}  "
          f"model={result['model_reward']:.1f}  "
          f"mappings={result['mappings']:,}  entropy={result['action_entropy']:.4f}")
    return result


def experiment_1_limb_scaling():
    print("\n" + "="*70)
    print("  EXPERIMENT 1: Limb Count Scaling")
    print("="*70)
    results = []
    for nl in [4, 6, 8, 12]:
        r = run_scaled_pipeline(num_limbs=nl, num_episodes=100, epochs=15)
        results.append(r)
    print("\n  --- Limb Scaling Summary ---")
    print(f"  {'Limbs':>5} {'OBS':>5} {'Actions':>7} {'Baseline':>10} {'Model':>10} {'Mappings':>10} {'Entropy':>8}")
    for r in results:
        print(f"  {r['num_limbs']:>5} {r['obs_dim']:>5} {r['num_actions']:>7} "
              f"{r['baseline_reward']:>10.1f} {r['model_reward']:>10.1f} "
              f"{r['mappings']:>10,} {r['action_entropy']:>8.4f}")
    return results


def experiment_5_generational(num_limbs=6, generations=3, episodes_per_gen=100):
    print("\n" + "="*70)
    print("  EXPERIMENT 5: Generational Inheritance")
    print("="*70)
    idx = compute_obs_indices(num_limbs)
    gen_results = []
    prev_model = None

    for gen in range(generations):
        print(f"\n  --- Generation {gen} ---")
        rng = np.random.RandomState(gen * 1000)
        all_windows, all_targets, all_next_pain = [], [], []
        global_log = []

        for ep in range(episodes_per_gen):
            env = Environment(seed=rng.randint(0, 100000))
            org = Organism(num_limbs=num_limbs)
            org.reset(rng)
            npc = NPC()
            npc.reset(rng)
            episode_pain = []
            for step in range(300):
                npc.step(env, step)
                optimal = org.compute_optimal_actions(env, step, npc=npc)
                window = org.get_observation_window()
                all_windows.append(window.copy())
                all_targets.append(optimal.copy())
                obs, reward = org.step(optimal, env, step, npc=npc)
                npc.receive_signal(optimal[num_limbs * 3:], org.x, org.y)
                episode_pain.append(obs[0:num_limbs].copy())
            for i in range(300):
                all_next_pain.append(episode_pain[min(i+1, 299)])
            global_log.extend(org.experience_log)

        X = np.array(all_windows, dtype=np.float32)
        Y = np.array(all_targets, dtype=np.float32)
        Z = np.array(all_next_pain, dtype=np.float32)

        engine = build_mental_model(global_log, core_obs_dim=idx['core_obs_dim'])
        X = augment_with_mental_model(X, global_log, engine, 300,
                                      idx['core_obs_dim'], idx['mm_start'])
        X = augment_with_patterns(X, global_log, engine, 300,
                                  idx['core_obs_dim'], idx['pattern_start'])
        X = augment_with_agency(X, global_log, engine, 300,
                                idx['core_obs_dim'], idx['agency_start'])

        model = HierarchicalPolicy(
            obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
            energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict']
        ).to(DEVICE)

        if prev_model is not None:
            prev_state = prev_model.state_dict()
            curr_state = model.state_dict()
            for key in curr_state:
                if key in prev_state and curr_state[key].shape == prev_state[key].shape:
                    curr_state[key] = 0.8 * prev_state[key] + 0.2 * curr_state[key]
            model.load_state_dict(curr_state)
            print(f"    Inherited weights from gen {gen-1} (0.8 parent + 0.2 random)")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)
        bce = torch.nn.BCEWithLogitsLoss()
        N = X.shape[0]
        indices = np.arange(N)
        epoch_accs = []

        for epoch in range(15):
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
            epoch_accs.append(total_acc / n_b)

        final_acc = epoch_accs[-1]
        convergence_epoch = next((i for i, a in enumerate(epoch_accs) if a > 0.93), 15)
        print(f"    Gen {gen}: acc={final_acc:.3f}  converged_at_epoch={convergence_epoch}")

        gen_results.append({
            'generation': gen,
            'final_accuracy': round(final_acc, 4),
            'convergence_epoch': convergence_epoch,
            'epoch_accs': [round(a, 4) for a in epoch_accs],
        })
        prev_model = model

    print("\n  --- Generational Summary ---")
    for r in gen_results:
        print(f"    Gen {r['generation']}: acc={r['final_accuracy']:.4f}  "
              f"converge@epoch={r['convergence_epoch']}")
    return gen_results


def experiment_4_diversity(num_episodes=100, epochs=15):
    print("\n" + "="*70)
    print("  EXPERIMENT 4: Organism Diversity")
    print("="*70)
    limb_options = [4, 6, 8]
    max_obs = 13 * max(limb_options) + 78
    max_actions = 3 * max(limb_options) + 4
    rng = np.random.RandomState(42)
    all_windows, all_targets = [], []
    per_type = {nl: {'count': 0, 'reward': 0.0} for nl in limb_options}

    for ep in range(num_episodes):
        nl = limb_options[ep % len(limb_options)]
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism(num_limbs=nl)
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        ep_reward = 0.0
        for step in range(300):
            npc.step(env, step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            padded_window = np.zeros((32, max_obs), dtype=np.float32)
            padded_window[:, :org.OBS_DIM] = window
            padded_target = np.zeros(max_actions, dtype=np.float32)
            padded_target[:org.NUM_ACTIONS] = optimal
            all_windows.append(padded_window)
            all_targets.append(padded_target)
            obs, reward = org.step(optimal, env, step, npc=npc)
            npc.receive_signal(optimal[nl * 3:], org.x, org.y)
            ep_reward += reward
        per_type[nl]['count'] += 1
        per_type[nl]['reward'] += ep_reward

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)

    print(f"  Data: X={X.shape}, Y={Y.shape} (padded to max obs/actions)")
    for nl, info in per_type.items():
        avg = info['reward'] / max(info['count'], 1)
        print(f"    {nl}-limbed: {info['count']} episodes, avg_reward={avg:.1f}")

    idx = compute_obs_indices(max(limb_options))
    model = HierarchicalPolicy(
        obs_dim=max_obs, num_actions=max_actions,
        energy_obs_index=idx['energy'], conflict_obs_index=max_obs - 1
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    N = X.shape[0]
    indices = np.arange(N)

    for epoch in range(epochs):
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
        acc = total_acc / n_b
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  acc={acc:.3f}")

    print(f"  Diversity model final acc: {acc:.3f}")
    return {'final_accuracy': round(acc, 4), 'body_types': limb_options}


def experiment_2_segments():
    print("\n" + "="*70)
    print("  EXPERIMENT 2: Body Segments")
    print("="*70)
    results = []
    for ns in [1, 2]:
        r = run_scaled_pipeline(num_limbs=6, num_episodes=100, epochs=15,
                                num_segments=ns)
        results.append(r)
    print("\n  --- Segment Scaling Summary ---")
    print(f"  {'Segs':>4} {'OBS':>5} {'Actions':>7} {'Baseline':>10} {'Model':>10}")
    for r in results:
        print(f"  {r.get('num_segments',1):>4} {r['obs_dim']:>5} {r['num_actions']:>7} "
              f"{r['baseline_reward']:>10.1f} {r['model_reward']:>10.1f}")
    return results


def experiment_3_3d():
    print("\n" + "="*70)
    print("  EXPERIMENT 3: 3D Environment")
    print("="*70)
    results = []
    for d in [2, 3]:
        r = run_scaled_pipeline(num_limbs=6, num_episodes=100, epochs=15,
                                dims=d)
        results.append(r)
    print("\n  --- 2D vs 3D Summary ---")
    print(f"  {'Dims':>4} {'OBS':>5} {'Baseline':>10} {'Model':>10}")
    for r in results:
        print(f"  {r.get('dims',2):>4} {r['obs_dim']:>5} "
              f"{r['baseline_reward']:>10.1f} {r['model_reward']:>10.1f}")
    return results


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}
    t0 = time.time()

    results['limb_scaling'] = experiment_1_limb_scaling()
    results['segments'] = experiment_2_segments()
    results['3d'] = experiment_3_3d()
    results['diversity'] = experiment_4_diversity()
    results['generational'] = experiment_5_generational()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ALL SCALING EXPERIMENTS COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    path = os.path.join(DATA_DIR, 'scaling_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")

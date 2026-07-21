import os
import json
import time
import numpy as np
import torch
from environment import Environment, Organism, NPC
from model import HierarchicalPolicy, compute_obs_indices
from mental_model import build_mental_model, action_to_hash
from train import (augment_with_mental_model, augment_with_patterns,
                   augment_with_agency, augment_with_concepts, DEVICE)
from receptor_discovery import build_tests, discover

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
BIAS_FACTOR = 0.6
ELEVATED_PROBE_RATE = 0.05
BURN_IN_EPISODES = 50


def generate_data_with_probes(num_episodes=100, steps_per_episode=300, seed=0,
                               probe_rate=0.02):
    rng = np.random.RandomState(seed)
    all_windows, all_targets, all_next_pain = [], [], []
    global_log = []

    for ep in range(num_episodes):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        episode_pain = []
        inference_rng = np.random.RandomState(rng.randint(0, 100000))

        effective_probe = ELEVATED_PROBE_RATE if ep < BURN_IN_EPISODES else probe_rate

        for step in range(steps_per_episode):
            npc.step(env, step)
            is_probe = inference_rng.random() < effective_probe

            if is_probe:
                actions = inference_rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                actions = org.compute_optimal_actions(env, step, npc=npc)

            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(actions.copy())
            obs, reward = org.step(actions, env, step, npc=npc)
            npc.receive_signal(actions[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:org.NUM_LIMBS].copy())

            entry = org.experience_log[-1]
            entry['is_probe'] = is_probe

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


def compute_biased_thresholds(parent_topology, base_tests):
    overrides = {}
    for i, test in enumerate(base_tests):
        if i < len(parent_topology) and parent_topology[i] == 1:
            overrides[test.receptor_id] = test.threshold * BIAS_FACTOR
    return overrides


def check_canalization(gen_history):
    if len(gen_history) < 3:
        return False
    recent = gen_history[-3:]
    probe_counts = [g['probe_validated_count'] for g in recent]
    policy_counts = [g['discovered_count'] for g in recent]
    probe_declining = probe_counts[-1] < probe_counts[-2] < probe_counts[-3]
    policy_stable = policy_counts[-1] >= policy_counts[-3] - 1
    return probe_declining and policy_stable


def run_generational_experiment(n_generations=5, episodes_per_gen=100, tier=0):
    print(f"\n{'='*70}")
    print(f"  TOPOLOGY BIAS INHERITANCE: {n_generations} generations, {episodes_per_gen} episodes each")
    print(f"{'='*70}")

    idx = compute_obs_indices(6)
    base_tests = build_tests()
    gen_history = []
    parent_topology = None
    parent_weights = None

    for gen in range(n_generations):
        print(f"\n  --- Generation {gen} ---")
        seed = gen * 10000

        X, Y, Z, global_log = generate_data_with_probes(
            num_episodes=episodes_per_gen, steps_per_episode=300, seed=seed)
        print(f"    Data: {X.shape[0]:,} samples, log: {len(global_log):,}")

        probe_log = [e for e in global_log if e.get('is_probe', False)]
        print(f"    Probe entries: {len(probe_log):,} ({len(probe_log)/len(global_log)*100:.1f}%)")

        engine = build_mental_model(global_log, core_obs_dim=idx['core_obs_dim'])
        stats = engine.get_stats()
        print(f"    Mappings: {stats['total_mappings']:,}")

        X = augment_with_mental_model(X, global_log, engine, 300,
                                      idx['core_obs_dim'], idx['mm_start'])
        X = augment_with_patterns(X, global_log, engine, 300,
                                  idx['core_obs_dim'], idx['pattern_start'])
        X = augment_with_agency(X, global_log, engine, 300,
                                idx['core_obs_dim'], idx['agency_start'])
        cs = idx['obs_dim'] - 2
        X = augment_with_concepts(X, global_log, engine, 300,
                                  idx['core_obs_dim'], cs)

        model = HierarchicalPolicy(
            obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
            energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict'],
            num_pain_channels=6, num_limbs=6
        ).to(DEVICE)

        if parent_weights is not None:
            prev_state = parent_weights
            curr_state = model.state_dict()
            for key in curr_state:
                if key in prev_state and curr_state[key].shape == prev_state[key].shape:
                    curr_state[key] = 0.8 * prev_state[key] + 0.2 * curr_state[key]
            model.load_state_dict(curr_state)
            print(f"    Inherited weights from gen {gen-1}")

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

        biased_thresholds = compute_biased_thresholds(parent_topology, base_tests) \
            if parent_topology is not None else None
        policy_results = discover(global_log, engine, threshold_overrides=biased_thresholds)
        discovered_count = len(policy_results['discovered'])

        probe_results = discover(probe_log, engine) if len(probe_log) > 1000 else \
            {'discovered': [], 'topology_vector': [0] * len(base_tests)}
        probe_validated = set(policy_results['discovered']) & set(probe_results['discovered'])
        probe_validated_count = len(probe_validated)

        canalized = check_canalization(gen_history)
        if canalized:
            print(f"    CANALIZATION DETECTED — resetting topology bias")
            parent_topology = None
        else:
            parent_topology = policy_results['topology_vector']

        parent_weights = {k: v.clone() for k, v in model.state_dict().items()}

        gen_record = {
            'generation': gen,
            'final_accuracy': round(final_acc, 4),
            'convergence_epoch': convergence_epoch,
            'discovered_count': discovered_count,
            'discovered': policy_results['discovered'],
            'probe_validated_count': probe_validated_count,
            'probe_validated': sorted(probe_validated),
            'canalization_reset': canalized,
            'topology_vector': policy_results['topology_vector'],
        }
        gen_history.append(gen_record)

        print(f"    Acc: {final_acc:.3f}  Converge: epoch {convergence_epoch}")
        print(f"    Discovered: {discovered_count}  Probe-validated: {probe_validated_count}")
        print(f"    Topology: {policy_results['topology_vector']}")
        if canalized:
            print(f"    ** Bias reset for next generation **")

    print(f"\n  {'='*60}")
    print(f"  GENERATIONAL SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Gen':>4} {'Acc':>7} {'Conv':>5} {'Disc':>5} {'Probe':>6} {'Reset':>6}")
    for g in gen_history:
        print(f"  {g['generation']:>4} {g['final_accuracy']:>7.3f} {g['convergence_epoch']:>5} "
              f"{g['discovered_count']:>5} {g['probe_validated_count']:>6} "
              f"{'YES' if g['canalization_reset'] else '':>6}")

    stable_receptors = set(gen_history[0]['discovered'])
    for g in gen_history[1:]:
        stable_receptors &= set(g['discovered'])
    print(f"\n  Stable across all generations: {sorted(stable_receptors)}")

    novel = set()
    for g in gen_history[1:]:
        novel |= set(g['discovered']) - set(gen_history[0]['discovered'])
    print(f"  Novel (not in gen 0): {sorted(novel)}")

    return gen_history


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    history = run_generational_experiment(n_generations=5, episodes_per_gen=100)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} min")

    path = os.path.join(DATA_DIR, 'topology_inheritance.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved: {path}")
    print("\nStep 33 complete.")

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from environment import Environment, Organism, NPC
from model import HierarchicalPolicy, ENERGY_OBS_INDEX, compute_obs_indices
from mental_model import build_mental_model, action_to_hash

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALPHA = 0.3
BETA = 0.5
GAMMA = 0.1
DELTA = 0.2
PROBE_RATE_FLOOR = 0.02


def generate_training_data(num_episodes=500, steps_per_episode=300, seed=0):
    rng = np.random.RandomState(seed)
    all_windows = []
    all_targets = []
    all_next_pain = []
    global_log = []

    for ep in range(num_episodes):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        episode_pain = []

        for step in range(steps_per_episode):
            npc.step(env, step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(optimal.copy())
            obs, reward = org.step(optimal, env, step, npc=npc)
            npc.receive_signal(optimal[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:6].copy())

        for i in range(steps_per_episode):
            if i + 1 < steps_per_episode:
                all_next_pain.append(episode_pain[i + 1])
            else:
                all_next_pain.append(episode_pain[i])

        global_log.extend(org.experience_log)

        if (ep + 1) % 100 == 0:
            print(f"  Generated {ep + 1}/{num_episodes} episodes "
                  f"({(ep + 1) * steps_per_episode:,} samples, "
                  f"log: {len(global_log):,})")

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    return X, Y, Z, global_log


def augment_with_mental_model(X, global_log, engine, steps_per_episode=300,
                              core_obs_dim=96, mm_start=96):
    N = X.shape[0]
    all_obs = np.array([e['obs_after'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_actions = [e['action'] for e in global_log]
    print(f"    Computing MM features for {len(all_obs):,} observations...")
    mm_fam, mm_qual = engine.get_context_features_batch(all_obs)

    print(f"    Computing MM certainty for {len(all_obs):,} observations...")
    mm_cert = np.zeros(N, dtype=np.float32)
    mm_lp = np.zeros(N, dtype=np.float32)
    for i in range(N):
        _, cert, n_ret = engine.predict_delta(all_obs[i], all_actions[i])
        mm_cert[i] = cert
        mm_lp[i] = (1.0 - cert) if n_ret > 0 else 1.0
        if (i + 1) % 50000 == 0:
            print(f"      Processed {i+1:,}/{N:,}")

    num_episodes = N // steps_per_episode
    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        ep_fam = mm_fam[ep_start:ep_start + steps_per_episode]
        ep_qual = mm_qual[ep_start:ep_start + steps_per_episode]
        ep_cert = mm_cert[ep_start:ep_start + steps_per_episode]
        ep_lp = mm_lp[ep_start:ep_start + steps_per_episode]
        for s in range(steps_per_episode):
            window_idx = ep_start + s
            history_len = min(s + 1, 32)
            offset = 32 - history_len
            first_step = s - history_len + 1
            X[window_idx, offset:32, mm_start] = ep_fam[first_step:s + 1]
            X[window_idx, offset:32, mm_start+1] = ep_qual[first_step:s + 1]
            X[window_idx, offset:32, mm_start+2] = ep_cert[first_step:s + 1]
            X[window_idx, offset:32, mm_start+3] = ep_lp[first_step:s + 1]
    return X


def augment_with_patterns(X, global_log, engine, steps_per_episode=300,
                          core_obs_dim=96, pattern_start=100):
    N = X.shape[0]
    all_obs = np.array([e['obs_after'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_hashes = [action_to_hash(e['action']) for e in global_log]
    num_episodes = N // steps_per_episode

    print(f"    Computing pattern features for {N:,} observations...")
    pat_avail = np.zeros(N, dtype=np.float32)
    pat_cert = np.zeros(N, dtype=np.float32)

    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        for s in range(1, steps_per_episode):
            idx = ep_start + s
            prev_hash = all_hashes[idx - 1]
            pa, pc = engine.query_pattern(prev_hash, all_obs[idx])
            pat_avail[idx] = pa
            pat_cert[idx] = pc

    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        ep_pa = pat_avail[ep_start:ep_start + steps_per_episode]
        ep_pc = pat_cert[ep_start:ep_start + steps_per_episode]
        for s in range(steps_per_episode):
            window_idx = ep_start + s
            history_len = min(s + 1, 32)
            offset = 32 - history_len
            first_step = s - history_len + 1
            X[window_idx, offset:32, pattern_start] = ep_pa[first_step:s + 1]
            X[window_idx, offset:32, pattern_start+1] = ep_pc[first_step:s + 1]
    return X


def augment_with_concepts(X, global_log, engine, steps_per_episode=300,
                          core_obs_dim=96, concept_start=158):
    N = X.shape[0]
    all_obs = np.array([e['obs_after'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_hashes = [action_to_hash(e['action']) for e in global_log]
    num_episodes = N // steps_per_episode

    print(f"    Computing concept features for {N:,} observations...")
    cm = np.zeros(N, dtype=np.float32)
    cq = np.zeros(N, dtype=np.float32)

    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        for s in range(1, steps_per_episode):
            idx = ep_start + s
            match, quality = engine.query_concept(all_hashes[idx - 1], all_obs[idx])
            cm[idx] = match
            cq[idx] = quality

    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        ep_cm = cm[ep_start:ep_start + steps_per_episode]
        ep_cq = cq[ep_start:ep_start + steps_per_episode]
        for s in range(steps_per_episode):
            window_idx = ep_start + s
            history_len = min(s + 1, 32)
            offset = 32 - history_len
            first_step = s - history_len + 1
            X[window_idx, offset:32, concept_start] = ep_cm[first_step:s + 1]
            X[window_idx, offset:32, concept_start+1] = ep_cq[first_step:s + 1]
    return X


def augment_with_agency(X, global_log, engine, steps_per_episode=300,
                        core_obs_dim=96, agency_start=132):
    N = X.shape[0]
    all_obs_b = np.array([e['obs_before'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_obs_a = np.array([e['obs_after'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_actions = [e['action'] for e in global_log]

    print(f"    Computing agency features for {N:,} observations...")
    ag_ctrl = np.zeros(N, dtype=np.float32)
    ag_ext = np.zeros(N, dtype=np.float32)
    ag_plan = np.zeros(N, dtype=np.float32)

    for i in range(N):
        ctrl, ext, plan = engine.compute_agency_features(all_obs_b[i], all_actions[i], all_obs_a[i])
        ag_ctrl[i] = ctrl
        ag_ext[i] = ext
        ag_plan[i] = plan
        if (i + 1) % 50000 == 0:
            print(f"      Processed {i+1:,}/{N:,}")

    num_episodes = N // steps_per_episode
    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        ep_ctrl = np.zeros(steps_per_episode, dtype=np.float32)
        ep_ext = np.zeros(steps_per_episode, dtype=np.float32)
        ep_plan = np.zeros(steps_per_episode, dtype=np.float32)
        ep_ctrl[1:] = ag_ctrl[ep_start:ep_start + steps_per_episode - 1]
        ep_ext[1:] = ag_ext[ep_start:ep_start + steps_per_episode - 1]
        ep_plan[1:] = ag_plan[ep_start:ep_start + steps_per_episode - 1]
        for s in range(steps_per_episode):
            window_idx = ep_start + s
            history_len = min(s + 1, 32)
            offset = 32 - history_len
            first_step = s - history_len + 1
            X[window_idx, offset:32, agency_start] = ep_ctrl[first_step:s + 1]
            X[window_idx, offset:32, agency_start+1] = ep_ext[first_step:s + 1]
            X[window_idx, offset:32, agency_start+2] = ep_plan[first_step:s + 1]
    return X


def train_model(X, Y, Z, epochs=30, batch_size=256, lr=1e-3, num_limbs=6,
                num_segments=1, dims=2):
    idx = compute_obs_indices(num_limbs, num_segments, dims)
    num_pain = Z.shape[1] if Z.ndim > 1 else 6
    total_limbs = idx.get('total_limbs', num_limbs)
    model = HierarchicalPolicy(
        obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
        energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict'],
        num_pain_channels=num_pain, num_limbs=total_limbs
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    bce = nn.BCEWithLogitsLoss()

    N = X.shape[0]
    indices = np.arange(N)
    best_loss = float('inf')

    for epoch in range(epochs):
        np.random.shuffle(indices)
        model.train()
        totals = {'loss': 0, 'acc': 0, 'gate': 0, 'conf': 0, 'fast_acc': 0, 'pred_mse': 0}
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = torch.FloatTensor(X[batch_idx]).to(DEVICE)
            y_batch = torch.FloatTensor(Y[batch_idx]).to(DEVICE)
            z_batch = torch.FloatTensor(Z[batch_idx]).to(DEVICE)

            result = model(x_batch)

            blend_loss = bce(result['blended'], y_batch)
            fast_loss = bce(result['fast_logits'], y_batch)
            slow_loss = bce(result['slow_logits'], y_batch)

            with torch.no_grad():
                fast_preds = (torch.sigmoid(result['fast_logits']) > 0.5).float()
                fast_accuracy = (fast_preds == y_batch).float().mean(dim=1, keepdim=True)
            conf_loss = F.mse_loss(result['confidence'], fast_accuracy)

            energy = x_batch[:, -1, idx['energy']:idx['energy'] + 1]
            energy_weight = 1.0 + (1.0 - energy)
            metabolic_loss = (result['gate'] * energy_weight).mean()

            pred_loss = F.mse_loss(result['predicted_next_pain'], z_batch)

            total_loss = (blend_loss
                          + ALPHA * (fast_loss + slow_loss)
                          + BETA * conf_loss
                          + GAMMA * metabolic_loss
                          + DELTA * pred_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(result['blended']) > 0.5).float()
                acc = (preds == y_batch).float().mean().item()

            totals['loss'] += total_loss.item()
            totals['acc'] += acc
            totals['gate'] += result['gate'].mean().item()
            totals['conf'] += result['confidence'].mean().item()
            totals['fast_acc'] += fast_accuracy.mean().item()
            totals['pred_mse'] += pred_loss.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / n_batches for k, v in totals.items()}

        if avg['loss'] < best_loss:
            best_loss = avg['loss']
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'best_model.pt'))

        print(f"  Epoch {epoch + 1:2d}/{epochs}  loss={avg['loss']:.4f}  "
              f"acc={avg['acc']:.3f}  gate={avg['gate']:.3f}  "
              f"conf={avg['conf']:.3f}  fast_acc={avg['fast_acc']:.3f}  "
              f"pred_mse={avg['pred_mse']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_model.pt'), weights_only=True))
    return model


def run_inference_episode(model, env, steps=300, rng=None, mental_model=None,
                          num_limbs=6, num_segments=1, dims=2):
    org = Organism(num_limbs=num_limbs, num_segments=num_segments, dims=dims)
    org.reset(rng)
    npc = NPC()
    npc.reset(rng)
    frames = []
    total_reward = 0.0
    prev_predicted_pain = None
    prev_learning_progress = 0.0
    prev_mm_certainty = 0.0
    prev_action_hash = 0
    prev_controllability = 0.0
    prev_external_change = 0.0
    prev_planning_value = 0.0
    inference_rng = np.random.RandomState(rng.randint(0, 100000) if rng else 42)

    for step in range(steps):
        npc.step(env, step)
        window = org.get_observation_window()
        obs_before = org.history[-1].copy() if len(org.history) > 0 else np.zeros(org.OBS_DIM)

        is_probe = inference_rng.random() < PROBE_RATE_FLOOR
        if is_probe:
            actions = inference_rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            pathway_info = {'gate_value': 0.0, 'confidence': 0.0,
                            'used_slow': False,
                            'predicted_next_pain': [0.0] * org.NUM_LIMBS}
        else:
            actions, pathway_info = model.predict(window)

        mm_features = None
        pattern_features = None
        if mental_model is not None:
            mm_fam, mm_qual = mental_model.get_context_features(obs_before)
            mm_features = (mm_fam, mm_qual, prev_mm_certainty, prev_learning_progress)
            if mental_model.pattern_store is not None:
                pa, pc = mental_model.query_pattern(prev_action_hash, obs_before)
                pattern_features = (pa, pc)

        if mental_model is not None:
            cm, cq = mental_model.query_concept(prev_action_hash, obs_before)
            org.concept_match = cm
            org.concept_quality = cq

        obs, reward = org.step(
            actions, env, step,
            slow_pathway_active=pathway_info['used_slow'],
            gate_value=pathway_info['gate_value'],
            predicted_pain=prev_predicted_pain,
            mm_features=mm_features,
            pattern_features=pattern_features,
            agency_features=(prev_controllability, prev_external_change, prev_planning_value),
            npc=npc,
        )
        npc.receive_signal(actions[org.NUM_LIMBS * 3:], org.x, org.y)
        prev_predicted_pain = np.array(pathway_info['predicted_next_pain'])
        prev_action_hash = action_to_hash(actions)

        frame = org.get_frame_data(env, step, npc=npc)
        frame['gv'] = pathway_info['gate_value']
        frame['cf'] = pathway_info['confidence']
        frame['pw'] = int(pathway_info['used_slow'])
        frame['pp'] = pathway_info.get('predicted_next_pain', [0.0] * org.NUM_LIMBS)
        frame['arb'] = pathway_info.get('arb_weights', [1.0] * 5)
        frame['is_probe'] = int(is_probe)

        if mental_model is not None:
            ctrl, ext_ch, plan_v = mental_model.compute_agency_features(obs_before, actions, obs)
            prev_controllability = ctrl
            prev_external_change = ext_ch
            prev_planning_value = plan_v

            quality = mental_model.get_prediction_quality(obs_before, actions, obs)
            lp, cert = mental_model.compute_learning_progress(obs_before, actions, obs, reward)
            prev_learning_progress = lp
            prev_mm_certainty = cert
            frame['mm_mse'] = round(quality['mse'], 6)
            frame['mm_pain_mse'] = round(quality['pain_mse'], 6)
            frame['mm_retrieved'] = quality['num_retrieved']
            frame['mm_certainty'] = round(cert, 4)
            frame['learning_progress'] = round(lp, 4)
            frame['ct'] = round(ctrl, 4)
            frame['ec'] = round(ext_ch, 4)
            frame['pv'] = round(plan_v, 4)

        frames.append(frame)
        total_reward += reward

    return frames, total_reward


def export_replay(model, num_episodes=5, steps=300, mental_model=None):
    replay = {
        'organism': Organism().to_dict(),
        'episodes': [],
    }

    for i in range(num_episodes):
        seed = 1000 + i
        env = Environment(seed=seed)
        rng = np.random.RandomState(seed)
        frames, total_reward = run_inference_episode(model, env, steps, rng, mental_model)

        slow_count = sum(1 for f in frames if f.get('pw', 0) == 1)
        slow_pct = slow_count / len(frames) * 100
        avg_gate = sum(f.get('gv', 0) for f in frames) / len(frames)
        avg_conf = sum(f.get('cf', 0) for f in frames) / len(frames)

        mm_str = ""
        if mental_model is not None:
            avg_mm_mse = sum(f.get('mm_mse', 0) for f in frames) / len(frames)
            avg_mm_cert = sum(f.get('mm_certainty', 0) for f in frames) / len(frames)
            mm_str = f"  mm_mse={avg_mm_mse:.4f}  mm_cert={avg_mm_cert:.3f}"

        replay['episodes'].append({
            'episode_id': i,
            'total_reward': round(total_reward, 2),
            'environment': env.to_dict(),
            'frames': frames,
        })
        print(f"  Episode {i}: reward={total_reward:.1f}  "
              f"slow={slow_pct:.0f}%  gate={avg_gate:.3f}  conf={avg_conf:.3f}"
              f"{mm_str}")

    path = os.path.join(DATA_DIR, 'replay.json')
    with open(path, 'w') as f:
        json.dump(replay, f, separators=(',', ':'))
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {path} ({size_mb:.1f} MB)")


def run_baseline_episode(env, steps=300, rng=None):
    org = Organism()
    org.reset(rng)
    npc = NPC()
    npc.reset(rng)
    total_reward = 0.0
    for step in range(steps):
        npc.step(env, step)
        actions = org.compute_optimal_actions(env, step, npc=npc)
        _, reward = org.step(actions, env, step, npc=npc)
        npc.receive_signal(actions[org.NUM_LIMBS * 3:], org.x, org.y)
        total_reward += reward
    return total_reward


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    idx = compute_obs_indices(6)

    print("=== Generating training data ===")
    X, Y, Z, global_log = generate_training_data(num_episodes=500, steps_per_episode=300)
    print(f"  Dataset: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    print(f"  Global experience log: {len(global_log):,} entries")
    print(f"  Label balance: {Y.mean():.3f} (fraction of 1s)")

    print("\n=== Building mental model ===")
    engine = build_mental_model(global_log, core_obs_dim=idx['core_obs_dim'])
    stats = engine.get_stats()
    print(f"  Mappings: {stats['total_mappings']:,}, "
          f"Actions: {stats['num_action_patterns']}, "
          f"Avg certainty: {stats['avg_certainty']:.3f}, "
          f"Avg count: {stats['avg_count']:.1f}, "
          f"Encoder params: {stats['encoder_params']:,}")

    mm = idx['mm_start']
    pt = idx['pattern_start']
    ag = idx['agency_start']

    print("\n=== Augmenting training data with mental model context ===")
    X = augment_with_mental_model(X, global_log, engine,
                                  core_obs_dim=idx['core_obs_dim'], mm_start=mm)
    print(f"  MM familiarity range: [{X[:, -1, mm].min():.3f}, {X[:, -1, mm].max():.3f}]")
    print(f"  MM quality range:     [{X[:, -1, mm+1].min():.3f}, {X[:, -1, mm+1].max():.3f}]")
    print(f"  MM certainty range:   [{X[:, -1, mm+2].min():.3f}, {X[:, -1, mm+2].max():.3f}]")
    print(f"  Learning progress:    [{X[:, -1, mm+3].min():.3f}, {X[:, -1, mm+3].max():.3f}]")

    print("\n=== Augmenting with pattern features ===")
    X = augment_with_patterns(X, global_log, engine,
                              core_obs_dim=idx['core_obs_dim'], pattern_start=pt)
    print(f"  Pattern available:    [{X[:, -1, pt].min():.3f}, {X[:, -1, pt].max():.3f}]")
    print(f"  Pattern certainty:    [{X[:, -1, pt+1].min():.3f}, {X[:, -1, pt+1].max():.3f}]")

    print("\n=== Augmenting with agency features ===")
    X = augment_with_agency(X, global_log, engine,
                            core_obs_dim=idx['core_obs_dim'], agency_start=ag)
    print(f"  Controllability:      [{X[:, -1, ag].min():.3f}, {X[:, -1, ag].max():.3f}]")
    print(f"  External change:      [{X[:, -1, ag+1].min():.3f}, {X[:, -1, ag+1].max():.3f}]")
    print(f"  Planning value:       [{X[:, -1, ag+2].min():.3f}, {X[:, -1, ag+2].max():.3f}]")

    cs = idx['obs_dim'] - 2
    print("\n=== Augmenting with concept features ===")
    X = augment_with_concepts(X, global_log, engine,
                              core_obs_dim=idx['core_obs_dim'], concept_start=cs)
    print(f"  Concept match:        [{X[:, -1, cs].min():.3f}, {X[:, -1, cs].max():.3f}]")
    print(f"  Concept quality:      [{X[:, -1, cs+1].min():.3f}, {X[:, -1, cs+1].max():.3f}]")

    if engine.pattern_store is not None:
        cstats = engine.pattern_store.get_concept_stats()
        print(f"  Stable concepts: {cstats['num_stable_concepts']} / {cstats['total_patterns']}")
        print(f"  Avg concept quality: {cstats['avg_concept_quality']:.3f}")

    print("\n=== Training hierarchical policy ===")
    model = train_model(X, Y, Z, epochs=30, batch_size=256, lr=1e-3)

    print("\n=== Baseline (optimal gradient actions) ===")
    for i in range(3):
        env = Environment(seed=1000 + i)
        rng = np.random.RandomState(1000 + i)
        baseline = run_baseline_episode(env, 300, rng)
        print(f"  Optimal episode {i}: reward={baseline:.1f}")

    print("\n=== Model inference (with mental model) ===")
    for i in range(3):
        env = Environment(seed=1000 + i)
        rng = np.random.RandomState(1000 + i)
        frames, model_reward = run_inference_episode(model, env, 300, rng, mental_model=engine)
        slow_count = sum(1 for f in frames if f.get('pw', 0) == 1)
        avg_mm = sum(f.get('mm_mse', 0) for f in frames) / len(frames)
        print(f"  Model  episode {i}: reward={model_reward:.1f}  "
              f"slow={slow_count/len(frames)*100:.0f}%  mm_mse={avg_mm:.4f}")

    print("\n=== Exporting replay data ===")
    export_replay(model, num_episodes=5, steps=300, mental_model=engine)
    print("\nDone!")

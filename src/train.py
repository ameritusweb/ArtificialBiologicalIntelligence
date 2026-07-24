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
EPSILON = 0.15
PROBE_RATE_FLOOR = 0.02


EXPLORE_RATE = 0.07


def generate_training_data_physics(num_episodes=200, steps_per_episode=300, seed=0,
                                    compound_objects=True, developmental=False):
    """Generate training data with physics world active."""
    from physics_world import PhysicsWorld
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

        pw = PhysicsWorld(env, org, num_objects=3, seed=rng.randint(0, 100000))
        if compound_objects:
            pw.add_compound_objects()

        episode_pain = []
        for step in range(steps_per_episode):
            npc.step(env, step)
            if developmental:
                pw.apply_developmental_changes(step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(optimal.copy())
            r = rng.random()
            if r < 0.02:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < EXPLORE_RATE:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = optimal
            pw.apply_organism_forces(executed)
            pw.check_grips(executed)
            pw.step()
            obs, reward = org.step(executed, env, step, npc=npc)
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:6].copy())

        for i in range(steps_per_episode):
            if i + 1 < steps_per_episode:
                all_next_pain.append(episode_pain[i + 1])
            else:
                all_next_pain.append(episode_pain[i])
        global_log.extend(org.experience_log)

        if (ep + 1) % 50 == 0:
            print(f"  Physics: {ep + 1}/{num_episodes} episodes "
                  f"({(ep + 1) * steps_per_episode:,} samples)")

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    return X, Y, Z, global_log


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
            r = rng.random()
            if r < 0.02:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < EXPLORE_RATE:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = optimal
            obs, reward = org.step(executed, env, step, npc=npc)
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
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


def _scatter_to_windows(X, ep_features, steps_per_episode, start_col, num_episodes):
    """Write per-step feature arrays into observation windows with correct lag.

    All feature arrays are pre-lagged by 1: ep_features[s] contains the feature
    value observable at decision time for step s (i.e., computed from step s-1's
    outcome). Step 0 gets zeros (no prior observation to compute from).

    This is the single place where the lag convention is enforced. All augmentation
    functions must pre-lag their features before calling this.
    """
    num_cols = len(ep_features)
    for ep in range(num_episodes):
        ep_start = ep * steps_per_episode
        for s in range(steps_per_episode):
            window_idx = ep_start + s
            history_len = min(s + 1, 32)
            offset = 32 - history_len
            first_step = s - history_len + 1
            for c, feat in enumerate(ep_features):
                X[window_idx, offset:32, start_col + c] = feat[ep_start + first_step:ep_start + s + 1]


def augment_with_mental_model(X, global_log, engine, steps_per_episode=300,
                              core_obs_dim=96, mm_start=96):
    N = X.shape[0]
    all_obs = np.array([e['obs_after'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_actions = [e['action'] for e in global_log]
    print(f"    Computing MM features for {len(all_obs):,} observations...")
    mm_fam_raw, mm_qual_raw = engine.get_context_features_batch(all_obs)

    print(f"    Computing MM certainty for {len(all_obs):,} observations...")
    mm_cert_raw = np.zeros(N, dtype=np.float32)
    mm_lp_raw = np.zeros(N, dtype=np.float32)
    for i in range(N):
        predicted, cert, n_ret = engine.predict_delta(all_obs[i], all_actions[i])
        mm_cert_raw[i] = cert
        if n_ret > 0 and i > 0 and (i % steps_per_episode) != 0:
            actual_delta = all_obs[i] - all_obs[i - 1]
            mse = float(np.mean((predicted - actual_delta[:len(predicted)]) ** 2))
            mm_lp_raw[i] = float(np.clip(mse / (mse + 0.01), 0.0, 1.0))
        else:
            mm_lp_raw[i] = 1.0
        if (i + 1) % 50000 == 0:
            print(f"      Processed {i+1:,}/{N:,}")

    num_episodes = N // steps_per_episode
    mm_fam = np.zeros(N, dtype=np.float32)
    mm_qual = np.zeros(N, dtype=np.float32)
    mm_cert = np.zeros(N, dtype=np.float32)
    mm_lp = np.zeros(N, dtype=np.float32)
    for ep in range(num_episodes):
        s = ep * steps_per_episode
        e = s + steps_per_episode
        mm_fam[s + 1:e] = mm_fam_raw[s:e - 1]
        mm_qual[s + 1:e] = mm_qual_raw[s:e - 1]
        mm_cert[s + 1:e] = mm_cert_raw[s:e - 1]
        mm_lp[s + 1:e] = mm_lp_raw[s:e - 1]

    _scatter_to_windows(X, [mm_fam, mm_qual, mm_cert, mm_lp],
                        steps_per_episode, mm_start, num_episodes)
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
            pa, pc = engine.query_pattern(prev_hash, all_obs[idx - 1])
            pat_avail[idx] = pa
            pat_cert[idx] = pc

    _scatter_to_windows(X, [pat_avail, pat_cert],
                        steps_per_episode, pattern_start, num_episodes)
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
            match, quality = engine.query_concept(all_hashes[idx - 1], all_obs[idx - 1])
            cm[idx] = match
            cq[idx] = quality

    _scatter_to_windows(X, [cm, cq],
                        steps_per_episode, concept_start, num_episodes)
    return X


def augment_with_agency(X, global_log, engine, steps_per_episode=300,
                        core_obs_dim=96, agency_start=132):
    N = X.shape[0]
    all_obs_b = np.array([e['obs_before'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_obs_a = np.array([e['obs_after'][:core_obs_dim] for e in global_log], dtype=np.float32)
    all_actions = [e['action'] for e in global_log]

    print(f"    Computing agency features for {N:,} observations...")
    ag_ctrl_raw = np.zeros(N, dtype=np.float32)
    ag_ext_raw = np.zeros(N, dtype=np.float32)
    ag_plan_raw = np.zeros(N, dtype=np.float32)

    for i in range(N):
        ctrl, ext, plan = engine.compute_agency_features(all_obs_b[i], all_actions[i], all_obs_a[i])
        ag_ctrl_raw[i] = ctrl
        ag_ext_raw[i] = ext
        ag_plan_raw[i] = plan
        if (i + 1) % 50000 == 0:
            print(f"      Processed {i+1:,}/{N:,}")

    num_episodes = N // steps_per_episode
    ag_ctrl = np.zeros(N, dtype=np.float32)
    ag_ext = np.zeros(N, dtype=np.float32)
    ag_plan = np.zeros(N, dtype=np.float32)
    for ep in range(num_episodes):
        s = ep * steps_per_episode
        e = s + steps_per_episode
        ag_ctrl[s + 1:e] = ag_ctrl_raw[s:e - 1]
        ag_ext[s + 1:e] = ag_ext_raw[s:e - 1]
        ag_plan[s + 1:e] = ag_plan_raw[s:e - 1]

    _scatter_to_windows(X, [ag_ctrl, ag_ext, ag_plan],
                        steps_per_episode, agency_start, num_episodes)
    return X


def _mixed_action_loss(logits, targets, num_continuous, bce_fn):
    if num_continuous > 0:
        cont_loss = F.mse_loss(logits[:, :num_continuous], targets[:, :num_continuous])
        bin_loss = bce_fn(logits[:, num_continuous:], targets[:, num_continuous:])
        return cont_loss + bin_loss
    return bce_fn(logits, targets)


def generate_training_data_closed_loop(num_bootstrap=100, num_online=400,
                                       steps_per_episode=300, seed=0):
    """Generate training data with mental model online during episodes.

    Phase 1 (bootstrap): collect data with no mental model (same as before).
    Phase 2 (online): build mental model from bootstrap, then run episodes
    with mm_features/pattern/agency/concepts computed live and the mental
    model updating after each step. This closes the loop: the organism's
    observation vector reflects the mental model's current state, and the
    mental model learns from the organism's actions.

    The augmentation functions become unnecessary — features are computed
    inline at the correct lag (obs_before, prev_* variables).
    """
    from mental_model import build_mental_model, action_to_hash

    rng = np.random.RandomState(seed)
    all_windows = []
    all_targets = []
    all_next_pain = []
    global_log = []

    total_episodes = num_bootstrap + num_online

    for ep in range(num_bootstrap):
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
            r = rng.random()
            if r < 0.02:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < EXPLORE_RATE:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = optimal
            obs, reward = org.step(executed, env, step, npc=npc)
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:6].copy())

        for i in range(steps_per_episode):
            if i + 1 < steps_per_episode:
                all_next_pain.append(episode_pain[i + 1])
            else:
                all_next_pain.append(episode_pain[i])
        global_log.extend(org.experience_log)

        if (ep + 1) % 100 == 0:
            print(f"  Bootstrap: {ep + 1}/{num_bootstrap} episodes")

    print(f"  Bootstrap complete: {len(global_log)} log entries")
    print("  Building mental model from bootstrap...")
    engine = build_mental_model(global_log)
    print(f"  Store: {engine.store.total_count} mappings")

    for ep in range(num_online):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        episode_pain = []

        prev_predicted_pain = None
        prev_mm_certainty = 0.0
        prev_learning_progress = 0.0
        prev_action_hash = 0
        prev_controllability = 0.0
        prev_external_change = 0.0
        prev_planning_value = 0.0

        for step in range(steps_per_episode):
            npc.step(env, step)
            obs_before = org.history[-1].copy() if len(org.history) > 0 else np.zeros(org.OBS_DIM)

            mm_fam, mm_qual = engine.get_context_features(obs_before)
            mm_features = (mm_fam, mm_qual, prev_mm_certainty, prev_learning_progress)

            pattern_features = None
            if engine.pattern_store is not None:
                pa, pc = engine.query_pattern(prev_action_hash, obs_before)
                pattern_features = (pa, pc)

            cm, cq = engine.query_concept(prev_action_hash, obs_before)
            org.concept_match = cm
            org.concept_quality = cq

            optimal = org.compute_optimal_actions(env, step, npc=npc)
            window = org.get_observation_window()
            all_windows.append(window.copy())
            all_targets.append(optimal.copy())

            r = rng.random()
            if r < 0.02:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < EXPLORE_RATE:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = optimal

            obs, reward = org.step(
                executed, env, step,
                predicted_pain=prev_predicted_pain,
                mm_features=mm_features,
                pattern_features=pattern_features,
                agency_features=(prev_controllability, prev_external_change, prev_planning_value),
                npc=npc,
            )
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:6].copy())

            ctrl, ext_ch, plan_v = engine.compute_agency_features(obs_before, executed, obs)
            prev_controllability = ctrl
            prev_external_change = ext_ch
            prev_planning_value = plan_v

            pred, cert, _ = engine.predict_delta(obs_before, executed)
            lp, cert_after = engine.compute_learning_progress(obs_before, executed, obs, reward)
            prev_mm_certainty = cert_after

            if hasattr(engine, 'observe_npc'):
                engine.observe_npc(obs)
            prev_learning_progress = lp

            prev_predicted_pain = obs_before[:6] + pred[:6] if len(pred) >= 6 else obs_before[:6].copy()
            prev_action_hash = action_to_hash(executed)

        for i in range(steps_per_episode):
            if i + 1 < steps_per_episode:
                all_next_pain.append(episode_pain[i + 1])
            else:
                all_next_pain.append(episode_pain[i])
        global_log.extend(org.experience_log)

        if (ep + 1) % 100 == 0:
            print(f"  Online: {ep + 1}/{num_online} episodes "
                  f"(store: {engine.store.total_count} mappings)")
            if engine.pattern_store is not None:
                print("  Rebuilding pattern store...")
                engine.pattern_store.build_from_log(
                    global_log, engine.encoder, engine.store,
                    steps_per_episode=steps_per_episode)
                ps = engine.pattern_store.get_stats()
                print(f"    Patterns: {ps['total_patterns']}, "
                      f"avg_gain: {ps['avg_compression_gain']:.3f}")

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    return X, Y, Z, global_log, engine


def generate_training_data_self_play(num_bootstrap=50, num_self_play=200,
                                      num_iterations=3, steps_per_episode=300,
                                      epochs_per_iter=15, seed=0, staged=True,
                                      use_thinking=True):
    """Generate training data via self-play: policy drives behavior, not oracle.

    Phase 1 (bootstrap): Small oracle dataset to get an initial policy.
    Phase 2 (self-play iterations): Train policy, then run episodes where the
    policy's own actions drive behavior. The mental model learns from what the
    policy actually did. Retrain on self-generated data. Repeat.

    When use_thinking=True, the organism runs MCTS (thinking substrate) before
    acting. Tree metadata becomes receptor input: visit patterns, value
    convergence, path divergence, underexplored branches.
    """
    from thinking_substrate import ThinkingTree
    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    obs_dim = idx['obs_dim']
    num_actions = idx['num_actions']

    print(f"Self-play: {num_bootstrap} bootstrap + {num_iterations}x{num_self_play} self-play")

    # Phase 1: Bootstrap from oracle
    print("\n--- Phase 1: Bootstrap (oracle) ---")
    all_windows, all_targets, all_next_pain = [], [], []
    global_log = []

    for ep in range(num_bootstrap):
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
            r = rng.random()
            if r < 0.02:
                executed = np.zeros(num_actions, dtype=np.int32)
            elif r < EXPLORE_RATE:
                executed = rng.randint(0, 2, size=num_actions).astype(np.int32)
            else:
                executed = optimal
            obs, reward = org.step(executed, env, step, npc=npc)
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
            episode_pain.append(obs[0:6].copy())

        for i in range(steps_per_episode):
            next_p = episode_pain[i + 1] if i + 1 < steps_per_episode else episode_pain[i]
            all_next_pain.append(next_p)
        global_log.extend(org.experience_log)

    print(f"  Bootstrap: {len(all_windows)} samples from {num_bootstrap} episodes")

    # Train initial policy
    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    print(f"  Training initial policy ({epochs_per_iter} epochs)...")
    model = train_model(X, Y, Z, epochs=epochs_per_iter, staged=staged,
                        steps_per_episode=steps_per_episode)

    # Build mental model from bootstrap
    print("  Building mental model from bootstrap...")
    engine = build_mental_model(global_log)
    print(f"  Store: {engine.store.total_count} mappings")

    tree = ThinkingTree(num_actions=idx['num_actions'],
                        max_simulations=32, max_depth=4) if use_thinking else None

    # Phase 2: Self-play iterations
    for iteration in range(num_iterations):
        print(f"\n--- Phase 2: Self-play iteration {iteration + 1}/{num_iterations} ---")

        sp_windows, sp_targets, sp_next_pain = [], [], []
        sp_log = []
        iter_rewards = []

        for ep in range(num_self_play):
            env = Environment(seed=rng.randint(0, 100000))
            org = Organism()
            org.reset(rng)
            npc = NPC()
            npc.reset(rng)
            episode_pain = []
            episode_reward = 0.0

            prev_predicted_pain = None
            prev_mm_certainty = 0.0
            prev_learning_progress = 0.0
            prev_action_hash = 0
            prev_controllability = 0.0
            prev_external_change = 0.0
            prev_planning_value = 0.0

            for step in range(steps_per_episode):
                npc.step(env, step)
                obs_before = org.history[-1].copy() if len(org.history) > 0 else np.zeros(obs_dim)

                mm_fam, mm_qual = engine.get_context_features(obs_before)
                mm_features = (mm_fam, mm_qual, prev_mm_certainty, prev_learning_progress)

                pattern_features = None
                if engine.pattern_store is not None:
                    pa, pc = engine.query_pattern(prev_action_hash, obs_before)
                    pattern_features = (pa, pc)

                cm, cq = engine.query_concept(prev_action_hash, obs_before)
                org.concept_match = cm
                org.concept_quality = cq

                if tree is not None:
                    thinking_analysis = tree.think(obs_before, engine)
                    org.thinking_channels = thinking_analysis

                # Policy decides the action — no oracle
                window = org.get_observation_window()
                policy_action, _ = model.predict(window)

                # Oracle provides the training target (what SHOULD have been done)
                optimal = org.compute_optimal_actions(env, step, npc=npc)

                sp_windows.append(window.copy())
                sp_targets.append(optimal.copy())

                # Exploration on the policy's action
                r = rng.random()
                if r < PROBE_RATE_FLOOR:
                    executed = np.zeros(num_actions, dtype=np.int32)
                elif r < EXPLORE_RATE:
                    executed = rng.randint(0, 2, size=num_actions).astype(np.int32)
                else:
                    executed = policy_action

                obs, reward = org.step(
                    executed, env, step,
                    predicted_pain=prev_predicted_pain,
                    mm_features=mm_features,
                    pattern_features=pattern_features,
                    agency_features=(prev_controllability, prev_external_change, prev_planning_value),
                    npc=npc,
                )
                npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
                episode_pain.append(obs[0:6].copy())
                episode_reward += reward

                ctrl, ext_ch, plan_v = engine.compute_agency_features(obs_before, executed, obs)
                prev_controllability = ctrl
                prev_external_change = ext_ch
                prev_planning_value = plan_v

                pred, cert, _ = engine.predict_delta(obs_before, executed)
                lp, cert_after = engine.compute_learning_progress(obs_before, executed, obs, reward)
                prev_mm_certainty = cert_after

                if hasattr(engine, 'observe_npc'):
                    engine.observe_npc(obs)
                prev_learning_progress = lp

                prev_predicted_pain = obs_before[:6] + pred[:6] if len(pred) >= 6 else obs_before[:6].copy()
                prev_action_hash = action_to_hash(executed)

            for i in range(steps_per_episode):
                next_p = episode_pain[i + 1] if i + 1 < steps_per_episode else episode_pain[i]
                sp_next_pain.append(next_p)
            sp_log.extend(org.experience_log)
            iter_rewards.append(episode_reward)

            if (ep + 1) % 50 == 0:
                print(f"  Self-play: {ep + 1}/{num_self_play} episodes, "
                      f"avg reward: {np.mean(iter_rewards[-50:]):.1f}")

        global_log.extend(sp_log)

        # Combine bootstrap + all self-play data so far
        all_windows.extend(sp_windows)
        all_targets.extend(sp_targets)
        all_next_pain.extend(sp_next_pain)

        X = np.array(all_windows, dtype=np.float32)
        Y = np.array(all_targets, dtype=np.float32)
        Z = np.array(all_next_pain, dtype=np.float32)

        print(f"  Retraining policy on {len(X)} samples ({epochs_per_iter} epochs)...")
        model = train_model(X, Y, Z, epochs=epochs_per_iter, staged=staged,
                            steps_per_episode=steps_per_episode)

        # Rebuild mental model with all data
        print("  Rebuilding mental model...")
        engine = build_mental_model(global_log)
        if engine.pattern_store is not None:
            engine.pattern_store.build_from_log(
                global_log, engine.encoder, engine.store,
                steps_per_episode=steps_per_episode)

        print(f"  Store: {engine.store.total_count} mappings, "
              f"avg reward: {np.mean(iter_rewards):.1f}")

    return X, Y, Z, global_log, engine, model


def train_model(X, Y, Z, epochs=30, batch_size=256, lr=1e-3, num_limbs=6,
                num_segments=1, dims=2, staged=False,
                steps_per_episode=300, val_fraction=0.15, continuous_actions=False):
    idx = compute_obs_indices(num_limbs, num_segments, dims, continuous_actions=continuous_actions)
    num_pain = Z.shape[1] if Z.ndim > 1 else 6
    nc = idx.get('num_continuous', 0)
    model = HierarchicalPolicy(
        obs_dim=idx['obs_dim'], num_actions=idx['num_actions'],
        energy_obs_index=idx['energy'], conflict_obs_index=idx['conflict'],
        num_pain_channels=num_pain, num_limbs=num_limbs,
        staged=staged, num_segments=num_segments, dims=dims,
        num_continuous=nc
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    bce = nn.BCEWithLogitsLoss()

    N = X.shape[0]
    num_episodes = N // steps_per_episode
    num_val_eps = max(1, int(num_episodes * val_fraction))
    ep_order = np.arange(num_episodes)
    np.random.RandomState(0).shuffle(ep_order)
    val_eps = set(ep_order[:num_val_eps].tolist())

    train_mask = np.ones(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    for ep in val_eps:
        s = ep * steps_per_episode
        e = s + steps_per_episode
        train_mask[s:e] = False
        val_mask[s:e] = True

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"    Train: {len(train_idx):,} samples ({num_episodes - num_val_eps} episodes)  "
          f"Val: {len(val_idx):,} samples ({num_val_eps} episodes)")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        model.train()
        totals = {'loss': 0, 'acc': 0, 'gate': 0, 'conf': 0, 'fast_acc': 0, 'pred_mse': 0,
                  'stage_mse': 0}
        n_batches = 0

        for start in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[start:start + batch_size]
            x_batch = torch.FloatTensor(X[batch_idx]).to(DEVICE)
            y_batch = torch.FloatTensor(Y[batch_idx]).to(DEVICE)
            z_batch = torch.FloatTensor(Z[batch_idx]).to(DEVICE)

            result = model(x_batch)

            blend_loss = _mixed_action_loss(result['blended'], y_batch, nc, bce)
            fast_loss = _mixed_action_loss(result['fast_logits'], y_batch, nc, bce)
            slow_loss = _mixed_action_loss(result['slow_logits'], y_batch, nc, bce)

            with torch.no_grad():
                if nc > 0:
                    fast_bin_preds = (torch.sigmoid(result['fast_logits'][:, nc:]) > 0.5).float()
                    fast_accuracy = (fast_bin_preds == y_batch[:, nc:]).float().mean(dim=1, keepdim=True)
                else:
                    fast_preds = (torch.sigmoid(result['fast_logits']) > 0.5).float()
                    fast_accuracy = (fast_preds == y_batch).float().mean(dim=1, keepdim=True)
            conf_loss = F.mse_loss(result['confidence'], fast_accuracy)

            energy = x_batch[:, -1, idx['energy']:idx['energy'] + 1]
            energy_weight = 1.0 + (1.0 - energy)
            metabolic_loss = (result['gate'] * energy_weight).mean()

            pred_loss = F.mse_loss(result['predicted_next_pain'], z_batch)

            inter_stage_loss = torch.tensor(0.0, device=DEVICE)
            if 'stage_predictions' in result:
                for pred, actual in result['stage_predictions']:
                    inter_stage_loss = inter_stage_loss + F.mse_loss(pred, actual)
                inter_stage_loss = inter_stage_loss / len(result['stage_predictions'])

            total_loss = (blend_loss
                          + ALPHA * (fast_loss + slow_loss)
                          + BETA * conf_loss
                          + GAMMA * metabolic_loss
                          + DELTA * pred_loss
                          + EPSILON * inter_stage_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                if nc > 0:
                    bin_preds = (torch.sigmoid(result['blended'][:, nc:]) > 0.5).float()
                    acc = (bin_preds == y_batch[:, nc:]).float().mean().item()
                else:
                    preds = (torch.sigmoid(result['blended']) > 0.5).float()
                    acc = (preds == y_batch).float().mean().item()

            totals['loss'] += total_loss.item()
            totals['acc'] += acc
            totals['gate'] += result['gate'].mean().item()
            totals['conf'] += result['confidence'].mean().item()
            totals['fast_acc'] += fast_accuracy.mean().item()
            totals['pred_mse'] += pred_loss.item()
            totals['stage_mse'] += inter_stage_loss.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / n_batches for k, v in totals.items()}

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            val_n = 0
            for vs in range(0, len(val_idx), batch_size):
                vi = val_idx[vs:vs + batch_size]
                xv = torch.FloatTensor(X[vi]).to(DEVICE)
                yv = torch.FloatTensor(Y[vi]).to(DEVICE)
                zv = torch.FloatTensor(Z[vi]).to(DEVICE)
                rv = model(xv)
                vl = _mixed_action_loss(rv['blended'], yv, nc, bce).item()
                if nc > 0:
                    va = ((torch.sigmoid(rv['blended'][:, nc:]) > 0.5).float() == yv[:, nc:]).float().mean().item()
                else:
                    va = ((torch.sigmoid(rv['blended']) > 0.5).float() == yv).float().mean().item()
                val_loss += vl
                val_acc += va
                val_n += 1
            val_loss /= max(val_n, 1)
            val_acc /= max(val_n, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'best_model.pt'))

        stage_str = f"  stage_mse={avg['stage_mse']:.4f}" if avg['stage_mse'] > 0 else ""
        print(f"  Epoch {epoch + 1:2d}/{epochs}  loss={avg['loss']:.4f}  "
              f"acc={avg['acc']:.3f}  val_acc={val_acc:.3f}  "
              f"gate={avg['gate']:.3f}  "
              f"pred_mse={avg['pred_mse']:.4f}{stage_str}  "
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
        actions, pathway_info = model.predict(window)
        if is_probe:
            actions = inference_rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)

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

    cs = idx['concept_start']
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

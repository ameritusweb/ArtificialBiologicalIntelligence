"""Track thinking influence across 10 self-play iterations.

Measures ablation divergence, reward difference, and per-channel
influence after each iteration to build the emergence curve.
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from model import compute_obs_indices, HierarchicalPolicy
from mental_model import build_mental_model, action_to_hash
from thinking_substrate import ThinkingTree
from thinking_influence import measure_thinking_influence
from train import train_model, EXPLORE_RATE, PROBE_RATE_FLOOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SEED = 42


def run_emergence_curve(num_bootstrap=30, num_self_play=50,
                        num_iterations=10, steps_per_episode=200,
                        epochs_per_iter=10, measure_every=1):
    from thinking_substrate import ThinkingTree

    rng = np.random.RandomState(SEED)
    idx = compute_obs_indices()
    obs_dim = idx['obs_dim']
    num_actions = idx['num_actions']

    curve = []

    # Phase 1: Bootstrap
    print("=" * 60)
    print(f"EMERGENCE CURVE: {num_bootstrap} bootstrap + {num_iterations}x{num_self_play} self-play")
    print("=" * 60)

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

    print(f"  Bootstrap: {len(all_windows)} samples")

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    Z = np.array(all_next_pain, dtype=np.float32)
    model = train_model(X, Y, Z, epochs=epochs_per_iter, staged=True,
                        steps_per_episode=steps_per_episode)

    engine = build_mental_model(global_log)
    tree = ThinkingTree(num_actions=num_actions, max_simulations=32, max_depth=4)

    # Measure iteration 0 (bootstrap only, no thinking exposure)
    print(f"\n--- Measuring iteration 0 (bootstrap) ---")
    m0 = measure_thinking_influence(model, engine, num_episodes=10,
                                     steps_per_episode=150, seed=99)
    m0['iteration'] = 0
    curve.append(m0)
    _print_row(m0)

    # Phase 2: Self-play iterations with thinking
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*40} Iteration {iteration}/{num_iterations} {'='*40}")

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
                obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                mm_fam, mm_qual = engine.get_context_features(obs_before)
                mm_features = (mm_fam, mm_qual, prev_mm_certainty, prev_learning_progress)

                pattern_features = None
                if engine.pattern_store is not None:
                    pa, pc = engine.query_pattern(prev_action_hash, obs_before)
                    pattern_features = (pa, pc)

                cm, cq = engine.query_concept(prev_action_hash, obs_before)
                org.concept_match = cm
                org.concept_quality = cq

                thinking_analysis = tree.think(obs_before, engine)
                org.thinking_channels = thinking_analysis

                window = org.get_observation_window()
                policy_action, _ = model.predict(window)
                optimal = org.compute_optimal_actions(env, step, npc=npc)

                sp_windows.append(window.copy())
                sp_targets.append(optimal.copy())

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

        global_log.extend(sp_log)
        all_windows.extend(sp_windows)
        all_targets.extend(sp_targets)
        all_next_pain.extend(sp_next_pain)

        print(f"  Avg reward: {np.mean(iter_rewards):.1f}")

        X = np.array(all_windows, dtype=np.float32)
        Y = np.array(all_targets, dtype=np.float32)
        Z = np.array(all_next_pain, dtype=np.float32)
        print(f"  Retraining on {len(X)} samples...")
        model = train_model(X, Y, Z, epochs=epochs_per_iter, staged=True,
                            steps_per_episode=steps_per_episode)

        print("  Rebuilding mental model...")
        engine = build_mental_model(global_log)
        if engine.pattern_store is not None:
            engine.pattern_store.build_from_log(
                global_log, engine.encoder, engine.store,
                steps_per_episode=steps_per_episode)

        if iteration % measure_every == 0:
            print(f"\n--- Measuring iteration {iteration} ---")
            m = measure_thinking_influence(model, engine, num_episodes=10,
                                            steps_per_episode=150, seed=99)
            m['iteration'] = iteration
            m['train_reward'] = float(np.mean(iter_rewards))
            curve.append(m)
            _print_row(m)

    # Summary
    print("\n" + "=" * 60)
    print("EMERGENCE CURVE SUMMARY")
    print("=" * 60)
    print(f"{'Iter':>4} {'Divergence':>11} {'Reward+':>8} {'PartCorr':>9} "
          f"{'BestVal':>8} {'Entropy':>8} {'Converg':>8} {'PathDiv':>8} "
          f"{'Unexpl':>8} {'Depth':>8}")
    print("-" * 105)
    for m in curve:
        ch = m['channel_influence']
        print(f"{m['iteration']:>4} {m['ablation_divergence']:>11.4f} "
              f"{m['reward_difference']:>+8.1f} {m['thinking_action_partial_corr']:>9.4f} "
              f"{ch['best_value']:>8.4f} {ch['visit_entropy']:>8.4f} "
              f"{ch['value_convergence']:>8.4f} {ch['path_divergence']:>8.4f} "
              f"{ch['underexplored']:>8.4f} {ch['depth_reached']:>8.4f}")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'thinking_emergence_curve.json'), 'w') as f:
        json.dump(curve, f, indent=2)
    print(f"\nSaved to {os.path.join(DATA_DIR, 'thinking_emergence_curve.json')}")

    return curve


def _print_row(m):
    ch = m['channel_influence']
    print(f"  Divergence: {m['ablation_divergence']:.4f}, "
          f"Reward+: {m['reward_difference']:+.1f}, "
          f"PartCorr: {m['thinking_action_partial_corr']:.4f}")
    print(f"  Channels: best={ch['best_value']:.3f} entropy={ch['visit_entropy']:.3f} "
          f"converg={ch['value_convergence']:.3f} div={ch['path_divergence']:.3f} "
          f"unexpl={ch['underexplored']:.3f} depth={ch['depth_reached']:.3f}")


if __name__ == '__main__':
    curve = run_emergence_curve(
        num_bootstrap=30,
        num_self_play=50,
        num_iterations=10,
        steps_per_episode=200,
        epochs_per_iter=10,
        measure_every=1,
    )

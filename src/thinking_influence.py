"""Measure when the thinking substrate starts influencing behavior.

Two measurements:
1. Ablation: run same policy with thinking channels live vs zeroed.
   Behavioral divergence > 0 means the policy learned to use them.
2. Partial correlation: do thinking channels predict action after
   controlling for environmental state?

Run after each self-play iteration to track the emergence curve.
"""

import numpy as np
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash
from thinking_substrate import ThinkingTree, NUM_THINKING_CHANNELS
from model import compute_obs_indices


def measure_thinking_influence(model, engine, num_episodes=20,
                               steps_per_episode=200, seed=0):
    """Run episodes with and without thinking, compare behavior.

    Returns a dict with:
      - ablation_divergence: mean action divergence (live vs zeroed)
      - reward_difference: mean reward difference (live - zeroed)
      - thinking_action_corr: partial correlation of thinking channels
                              with action, controlling for pain/endorphin
      - channel_influence: per-channel correlation with action change
    """
    idx = compute_obs_indices()
    obs_dim = idx['obs_dim']
    num_actions = idx['num_actions']
    thinking_start = idx['thinking_start']
    rng = np.random.RandomState(seed)

    tree = ThinkingTree(num_actions=num_actions, max_simulations=32, max_depth=4)

    live_rewards = []
    ablated_rewards = []
    divergences = []

    thinking_vals_all = []
    action_vals_all = []
    pain_vals_all = []
    endo_vals_all = []

    for ep in range(num_episodes):
        env_seed = rng.randint(0, 100000)

        # --- Run with thinking LIVE ---
        env = Environment(seed=env_seed)
        org = Organism()
        org.reset(np.random.RandomState(env_seed))
        npc = NPC()
        npc.reset(np.random.RandomState(env_seed))

        live_actions = []
        live_reward = 0.0

        for step in range(steps_per_episode):
            npc.step(env, step)
            obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

            thinking_analysis = tree.think(obs_before, engine)
            org.thinking_channels = thinking_analysis

            window = org.get_observation_window()
            action, _ = model.predict(window)
            obs, reward = org.step(action, env, step, npc=npc)
            live_reward += reward
            live_actions.append(action.copy())

            thinking_vals_all.append(thinking_analysis.copy())
            action_vals_all.append(np.array(action, dtype=float))
            pain_vals_all.append(float(np.mean(obs[:6])))
            endo_vals_all.append(float(np.mean(obs[6:12])))

        live_rewards.append(live_reward)

        # --- Run with thinking ABLATED (same env, same seed) ---
        env = Environment(seed=env_seed)
        org = Organism()
        org.reset(np.random.RandomState(env_seed))
        npc = NPC()
        npc.reset(np.random.RandomState(env_seed))

        ablated_actions = []
        ablated_reward = 0.0

        for step in range(steps_per_episode):
            npc.step(env, step)
            org.thinking_channels = np.zeros(NUM_THINKING_CHANNELS)

            window = org.get_observation_window()
            action, _ = model.predict(window)
            obs, reward = org.step(action, env, step, npc=npc)
            ablated_reward += reward
            ablated_actions.append(action.copy())

        ablated_rewards.append(ablated_reward)

        # Compute per-episode action divergence
        min_len = min(len(live_actions), len(ablated_actions))
        ep_divergence = np.mean([
            np.mean(np.abs(np.array(live_actions[i], dtype=float) -
                           np.array(ablated_actions[i], dtype=float)))
            for i in range(min_len)
        ])
        divergences.append(ep_divergence)

    # --- Partial correlation: thinking → action | (pain, endorphin) ---
    thinking_arr = np.array(thinking_vals_all)
    action_arr = np.array(action_vals_all)
    pain_arr = np.array(pain_vals_all)
    endo_arr = np.array(endo_vals_all)

    action_scalar = np.mean(action_arr, axis=1)
    n = len(action_scalar)

    channel_influence = np.zeros(NUM_THINKING_CHANNELS)
    partial_corr = 0.0

    if n >= 30:
        ext = np.column_stack([pain_arr, endo_arr, np.ones(n)])

        for ch in range(NUM_THINKING_CHANNELS):
            tc = thinking_arr[:, ch]
            if np.std(tc) < 1e-8:
                continue
            try:
                beta_a = np.linalg.lstsq(ext, action_scalar, rcond=None)[0]
                resid_a = action_scalar - ext @ beta_a
                beta_t = np.linalg.lstsq(ext, tc, rcond=None)[0]
                resid_t = tc - ext @ beta_t

                if np.std(resid_a) > 1e-8 and np.std(resid_t) > 1e-8:
                    c = np.corrcoef(resid_a, resid_t)[0, 1]
                    if not np.isnan(c):
                        channel_influence[ch] = abs(float(c))
            except Exception:
                pass

        partial_corr = float(np.max(channel_influence))

    results = {
        'ablation_divergence': float(np.mean(divergences)),
        'ablation_divergence_std': float(np.std(divergences)),
        'reward_live': float(np.mean(live_rewards)),
        'reward_ablated': float(np.mean(ablated_rewards)),
        'reward_difference': float(np.mean(live_rewards) - np.mean(ablated_rewards)),
        'thinking_action_partial_corr': partial_corr,
        'channel_influence': {
            'best_value': float(channel_influence[0]),
            'visit_entropy': float(channel_influence[1]),
            'value_convergence': float(channel_influence[2]),
            'path_divergence': float(channel_influence[3]),
            'underexplored': float(channel_influence[4]),
            'depth_reached': float(channel_influence[5]),
        },
        'num_episodes': num_episodes,
    }

    return results


def track_emergence(model, engine, iterations_completed,
                    num_episodes=10, steps_per_episode=100, seed=0):
    """Run after each self-play iteration to build the emergence curve.

    Returns one row of the curve: (iteration, divergence, partial_corr, reward_diff).
    When divergence rises above null baseline, thinking is influencing behavior.
    """
    results = measure_thinking_influence(
        model, engine, num_episodes=num_episodes,
        steps_per_episode=steps_per_episode, seed=seed)

    print(f"  Iteration {iterations_completed}:")
    print(f"    Ablation divergence: {results['ablation_divergence']:.4f} "
          f"(+/- {results['ablation_divergence_std']:.4f})")
    print(f"    Reward: live={results['reward_live']:.1f}, "
          f"ablated={results['reward_ablated']:.1f}, "
          f"diff={results['reward_difference']:.1f}")
    print(f"    Partial corr: {results['thinking_action_partial_corr']:.4f}")
    ch = results['channel_influence']
    top = sorted(ch.items(), key=lambda x: -x[1])[:3]
    print(f"    Top channels: {', '.join(f'{k}={v:.3f}' for k, v in top)}")

    return results


if __name__ == '__main__':
    from train import generate_training_data_self_play, train_model

    print("=== Thinking Influence Emergence Experiment ===\n")
    print("Training self-play with thinking substrate...\n")

    X, Y, Z, log, engine, model = generate_training_data_self_play(
        num_bootstrap=30, num_self_play=50, num_iterations=3,
        steps_per_episode=200, epochs_per_iter=10, seed=42,
        use_thinking=True)

    print("\n=== Measuring thinking influence ===\n")
    results = measure_thinking_influence(model, engine, num_episodes=15,
                                          steps_per_episode=150, seed=99)

    print(f"Ablation divergence: {results['ablation_divergence']:.4f}")
    print(f"Reward difference:   {results['reward_difference']:.1f}")
    print(f"Partial correlation: {results['thinking_action_partial_corr']:.4f}")
    print(f"\nPer-channel influence:")
    for name, val in results['channel_influence'].items():
        bar = '#' * int(val * 50)
        print(f"  {name:<20} {val:.4f} {bar}")

    if results['ablation_divergence'] > 0.01:
        print(f"\nThinking IS influencing behavior (divergence > 0.01)")
    else:
        print(f"\nThinking NOT yet influencing behavior (divergence <= 0.01)")

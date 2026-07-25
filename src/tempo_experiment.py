"""Environment Tempo Experiment.

Two populations, same genome, same seed:
  SLOW: hidden confounder transitions every 500 steps, stable fields,
        NPC moves slowly. Deep thinking has time to pay off.
  FAST: hidden confounder transitions every 20 steps, volatile fields,
        NPC moves rapidly. Deep thinking is wasted — state changes
        before the organism can act on it.

Predictions:
  - SLOW produces higher evolved thinking_budget
  - SLOW produces depth_reached activation
  - FAST produces lower thinking_budget
  - FAST produces no depth_reached
  - This is T52 applied to thinking depth
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from environment_tiers import TieredEnvironment, StochasticHiddenVariable
from mental_model import build_mental_model, action_to_hash
from receptor_discovery import discover, calibrate_null_thresholds
from model import compute_obs_indices
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from thinking_influence import measure_thinking_influence
from deep_time import EvolvingOrganism, select_and_reproduce

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TempoEnvironment(TieredEnvironment):
    """TieredEnvironment with controllable tempo."""

    def __init__(self, seed=None, tier=4, tempo='slow'):
        super().__init__(seed=seed, tier=tier)
        self.tempo = tempo

        if hasattr(self, 'stochastic_hidden') and self.stochastic_hidden is not None:
            if tempo == 'slow':
                # Stay in each state much longer
                self.stochastic_hidden.transition_probs = np.full(
                    (3, 3), 0.005 / 2)
                np.fill_diagonal(self.stochastic_hidden.transition_probs, 0.995)
            elif tempo == 'fast':
                # Change state rapidly
                self.stochastic_hidden.transition_probs = np.full(
                    (3, 3), 0.15 / 2)
                np.fill_diagonal(self.stochastic_hidden.transition_probs, 0.85)

        # Adjust NPC speed
        for npc in self.npcs:
            if tempo == 'slow':
                npc.vx *= 0.3
                npc.vy *= 0.3
            elif tempo == 'fast':
                npc.vx *= 2.0
                npc.vy *= 2.0

        # Adjust field source movement
        if tempo == 'fast':
            for src in self.pain_sources + self.endorphin_sources:
                src.omega_x *= 3.0
                src.omega_y *= 3.0


def run_tempo(tempo_name, num_generations=30, population_size=4,
              num_episodes=5, steps_per_episode=200,
              epochs_per_gen=8, seed=42):
    """Run deep time in a specific tempo environment."""
    print(f"\n{'='*50} {tempo_name.upper()} TEMPO {'='*50}")

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']

    organisms = [EvolvingOrganism(f"{tempo_name}_gen0_{i}")
                 for i in range(population_size)]
    history = []
    cumulative_log = []

    # Bootstrap
    print("  Bootstrapping...")
    X, Y, Z, boot_log = generate_training_data(
        num_episodes=20, steps_per_episode=steps_per_episode, seed=seed)
    model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                        steps_per_episode=steps_per_episode)
    cumulative_log.extend(boot_log)
    cumulative_windows = list(X)
    cumulative_targets = list(Y)
    cumulative_next_pain = list(Z)

    engine = build_mental_model(cumulative_log)
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=4)

    null_thresh = calibrate_null_thresholds(cumulative_log[:30000], engine, num_shuffles=3)

    for gen in range(num_generations):
        env_seed = rng.randint(0, 100000)
        env = TempoEnvironment(seed=env_seed, tier=4, tempo=tempo_name)

        for evo_org in organisms:
            org = evo_org.create_organism(rng)

            # Set per-organism thinking params
            budget = int(evo_org.body_params.get('thinking_budget', 24))
            tree.max_simulations = budget
            v_keys = {k: v for k, v in evo_org.body_params.items()
                      if k.startswith('v_')}
            tree.v_weights = v_keys if v_keys else None
            thinking_cost = float(evo_org.body_params.get('thinking_cost', 0.001))

            npc = NPC()
            npc.reset(rng)

            for ep in range(num_episodes):
                for step in range(steps_per_episode):
                    env.step_tier(org.x, org.y, step)
                    active_npc = env.get_closest_npc(org.x, org.y) or npc
                    active_npc.step(env, step)

                    obs_before = org.history[-1].copy() if org.history else np.zeros(idx['obs_dim'])

                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)

                    if gen == 0:
                        actions = org.compute_optimal_actions(env, step, npc=active_npc)
                        executed = actions
                    else:
                        window = org.get_observation_window()
                        policy_action, _ = model.predict(window)
                        optimal = org.compute_optimal_actions(env, step, npc=active_npc)
                        cumulative_windows.append(window.copy())
                        cumulative_targets.append(optimal.copy())
                        executed = policy_action

                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(num_actions, dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=num_actions).astype(np.int32)

                    obs, reward = org.step(executed, env, step, npc=active_npc)
                    evo_org.fitness += reward

                ep_pain = [e['obs_after'][0:6].copy() for e in org.experience_log[-steps_per_episode:]]
                for i in range(len(ep_pain)):
                    next_p = ep_pain[i + 1] if i + 1 < len(ep_pain) else ep_pain[-1]
                    cumulative_next_pain.append(next_p)

            cumulative_log.extend(org.experience_log)

        # Retrain
        X = np.array(cumulative_windows[-60000:], dtype=np.float32)
        Y = np.array(cumulative_targets[-60000:], dtype=np.float32)
        Z = np.array(cumulative_next_pain[-60000:], dtype=np.float32)
        if len(X) >= 100 and gen > 0:
            model = train_model(X, Y, Z, epochs=epochs_per_gen, staged=True,
                                steps_per_episode=steps_per_episode)

        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)

        # Discovery + measurement every 5 gens
        if gen % 5 == 0 or gen == num_generations - 1:
            gen_discovered = set()
            if len(log_slice) >= 500:
                results = discover(log_slice, engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='oracle')
                gen_discovered = set(results['discovered'])

            ti = measure_thinking_influence(model, engine, num_episodes=5,
                                             steps_per_episode=100, seed=99)

            avg_budget = float(np.mean([o.body_params.get('thinking_budget', 24)
                                        for o in organisms]))
            avg_cost = float(np.mean([o.body_params.get('thinking_cost', 0.001)
                                      for o in organisms]))
            avg_merge = float(np.mean([o.body_params.get('merge_threshold', 0.9)
                                       for o in organisms]))

            rec = {
                'generation': gen,
                'tempo': tempo_name,
                'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
                'num_discovered': len(gen_discovered),
                'avg_thinking_budget': round(avg_budget, 1),
                'avg_thinking_cost': round(avg_cost, 6),
                'avg_merge_threshold': round(avg_merge, 4),
                'thinking_pcorr': round(ti['thinking_action_partial_corr'], 4),
                'depth_reached': round(ti['channel_influence'].get('depth_reached', 0), 4),
            }
            history.append(rec)
            print(f"  Gen {gen}: fitness={rec['avg_fitness']:.0f}  "
                  f"receptors={rec['num_discovered']}  "
                  f"budget={rec['avg_thinking_budget']:.0f}  "
                  f"pcorr={rec['thinking_pcorr']:.4f}  "
                  f"depth={rec['depth_reached']:.4f}")

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"{tempo_name}_gen{gen+1}_{i}"

    return history


if __name__ == '__main__':
    SEED = 42
    NUM_GENS = 30

    slow = run_tempo('slow', num_generations=NUM_GENS, seed=SEED)
    fast = run_tempo('fast', num_generations=NUM_GENS, seed=SEED)

    # Comparison
    print("\n" + "=" * 60)
    print("TEMPO COMPARISON")
    print("=" * 60)

    print(f"\n{'Gen':>4} {'Slow Budget':>12} {'Fast Budget':>12} "
          f"{'Slow Depth':>11} {'Fast Depth':>11}")
    print("-" * 55)
    for s, f in zip(slow, fast):
        print(f"{s['generation']:>4} {s['avg_thinking_budget']:>12.0f} "
              f"{f['avg_thinking_budget']:>12.0f} "
              f"{s['depth_reached']:>11.4f} {f['depth_reached']:>11.4f}")

    slow_final_budget = slow[-1]['avg_thinking_budget']
    fast_final_budget = fast[-1]['avg_thinking_budget']
    slow_depth_ever = any(r['depth_reached'] > 0.01 for r in slow)
    fast_depth_ever = any(r['depth_reached'] > 0.01 for r in fast)

    print(f"\nFinal thinking budget: slow={slow_final_budget:.0f}, fast={fast_final_budget:.0f}")
    print(f"Depth ever activated:  slow={slow_depth_ever}, fast={fast_depth_ever}")

    if slow_final_budget > fast_final_budget:
        print("\nCONFIRMED: Slow environment selects for higher thinking budget")
    else:
        print("\nNOT CONFIRMED: Fast environment did not select for lower budget")

    if slow_depth_ever and not fast_depth_ever:
        print("CONFIRMED: Depth activates in slow but not fast (T52)")
    elif slow_depth_ever and fast_depth_ever:
        print("PARTIAL: Depth activates in both (tempo not the bottleneck)")
    else:
        print("NOT CONFIRMED: Depth didn't activate in either")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'tempo_experiment.json'), 'w') as f:
        json.dump({
            'slow': slow,
            'fast': fast,
            'slow_final_budget': slow_final_budget,
            'fast_final_budget': fast_final_budget,
            'slow_depth_ever': slow_depth_ever,
            'fast_depth_ever': fast_depth_ever,
        }, f, indent=2)
    print(f"\nSaved to data/tempo_experiment.json")

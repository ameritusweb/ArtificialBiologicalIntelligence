"""Order-Swap Experiment.

Two environments, same genome, same seed:
  A: Exploration pays first (high exploration bonus early, prediction accuracy
     bonus later). Curiosity before accuracy.
  B: Prediction accuracy pays first (accurate predictions rewarded early,
     exploration bonus later). Accuracy before curiosity.

Question: do different mature topologies emerge?
This tests whether the evolutionary path — which capability pays first —
determines the canopy topology. The whitepaper's own question, never run.
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash
from receptor_discovery import discover, calibrate_null_thresholds
from model import compute_obs_indices
from train import train_model, EXPLORE_RATE, PROBE_RATE_FLOOR
from deep_time import EvolvingOrganism, select_and_reproduce

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class OrderedEnvironment(Environment):
    """Environment where reward structure changes over generations."""

    def __init__(self, seed=None, order='explore_first', generation=0,
                 switch_generation=10):
        super().__init__(seed=seed)
        self.order = order
        self.generation = generation
        self.switch_generation = switch_generation

    def get_phase_weights(self):
        """Return (exploration_weight, accuracy_weight) based on phase."""
        if self.generation < self.switch_generation:
            if self.order == 'explore_first':
                return 2.0, 0.3
            else:
                return 0.3, 2.0
        else:
            if self.order == 'explore_first':
                return 0.3, 2.0
            else:
                return 2.0, 0.3


def compute_reward_with_order(org, env, engine, obs_before, action, obs_after, base_reward):
    """Modify reward based on environment order."""
    explore_w, accuracy_w = env.get_phase_weights()

    # Exploration component: reward for visiting new states
    obs_delta = float(np.linalg.norm(obs_after[:30] - obs_before[:30]))
    explore_reward = explore_w * obs_delta * 0.5

    # Accuracy component: reward for correct predictions
    accuracy_reward = 0.0
    if engine is not None:
        pred, cert, n = engine.predict_delta(obs_before, action)
        if n > 0:
            actual = obs_after[:len(pred)] - obs_before[:len(pred)]
            error = float(np.mean(np.abs(pred - actual)))
            accuracy_reward = accuracy_w * max(0.0, 1.0 - error) * 0.5

    return base_reward + explore_reward + accuracy_reward


def run_order(order_name, num_generations=20, population_size=4,
              num_episodes=10, steps_per_episode=200,
              switch_generation=10, seed=42):
    """Run one ordering condition."""
    print(f"\n{'='*40} {order_name} {'='*40}")
    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()

    organisms = [EvolvingOrganism(f"{order_name}_gen0_{i}")
                 for i in range(population_size)]
    history = []
    cumulative_log = []

    engine = None
    null_thresh = None

    for gen in range(num_generations):
        env_seed = rng.randint(0, 100000)

        for evo_org in organisms:
            org = evo_org.create_organism(rng)
            env = OrderedEnvironment(seed=env_seed, order=order_name,
                                     generation=gen,
                                     switch_generation=switch_generation)
            npc = NPC()
            npc.reset(rng)

            for ep in range(num_episodes):
                for step in range(steps_per_episode):
                    npc.step(env, step)
                    actions = org.compute_optimal_actions(env, step, npc=npc)
                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(idx['num_actions'], dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=idx['num_actions']).astype(np.int32)
                    else:
                        executed = actions
                    obs_before = org.history[-1].copy() if org.history else np.zeros(idx['obs_dim'])
                    obs, base_reward = org.step(executed, env, step, npc=npc)
                    order_reward = compute_reward_with_order(
                        org, env, engine, obs_before, executed, obs, base_reward)
                    evo_org.fitness += order_reward

            cumulative_log.extend(org.experience_log)

        # Build/rebuild engine
        log_slice = cumulative_log[-60000:]
        engine = build_mental_model(log_slice)

        # Discovery
        if gen == 0 or gen % 5 == 0 or gen == num_generations - 1:
            if null_thresh is None:
                null_thresh = calibrate_null_thresholds(log_slice[:30000], engine, num_shuffles=3)

            gen_discovered = set()
            if len(log_slice) >= 500:
                results = discover(log_slice, engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='oracle')
                gen_discovered = set(results['discovered'])
                for evo_org in organisms:
                    evo_org.discovered_receptors = results['discovered']

            rec = {
                'generation': gen,
                'order': order_name,
                'phase': 'early' if gen < switch_generation else 'late',
                'avg_fitness': round(float(np.mean([o.fitness for o in organisms])), 1),
                'num_discovered': len(gen_discovered),
                'discovered': sorted(gen_discovered),
            }
            history.append(rec)
            print(f"  Gen {gen} ({rec['phase']}): fitness={rec['avg_fitness']:.0f}  "
                  f"receptors={rec['num_discovered']}")

        # Reproduce
        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org in enumerate(organisms):
                org.organism_id = f"{order_name}_gen{gen+1}_{i}"

    return history


if __name__ == '__main__':
    SEED = 42
    NUM_GENS = 20
    SWITCH = 10

    explore_first = run_order('explore_first', num_generations=NUM_GENS,
                               switch_generation=SWITCH, seed=SEED)
    accuracy_first = run_order('accuracy_first', num_generations=NUM_GENS,
                                switch_generation=SWITCH, seed=SEED)

    # Comparison
    print("\n" + "=" * 60)
    print("ORDER-SWAP COMPARISON")
    print("=" * 60)

    ef_final = set(explore_first[-1]['discovered'])
    af_final = set(accuracy_first[-1]['discovered'])
    both = ef_final & af_final
    ef_only = ef_final - af_final
    af_only = af_final - ef_final

    print(f"\nExplore-first final: {len(ef_final)} receptors")
    print(f"Accuracy-first final: {len(af_final)} receptors")
    print(f"Both: {len(both)}")
    print(f"Explore-first only ({len(ef_only)}): {sorted(ef_only)}")
    print(f"Accuracy-first only ({len(af_only)}): {sorted(af_only)}")

    ef_cumul = set()
    af_cumul = set()
    for r in explore_first:
        ef_cumul.update(r['discovered'])
    for r in accuracy_first:
        af_cumul.update(r['discovered'])
    print(f"\nCumulative: explore-first={len(ef_cumul)}, accuracy-first={len(af_cumul)}")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'order_swap.json'), 'w') as f:
        json.dump({
            'explore_first': explore_first,
            'accuracy_first': accuracy_first,
            'both': sorted(both),
            'explore_only': sorted(ef_only),
            'accuracy_only': sorted(af_only),
        }, f, indent=2)
    print(f"\nSaved to data/order_swap.json")

"""Deep Time Learning: the full evolutionary loop.

Generation N:
  1. Organisms compete in a persistent physics world
  2. Mental models build from experience
  3. Receptor discovery runs on accumulated logs
  4. Fitness-proportional selection chooses parents
  5. Offspring inherit:
     - Receptor topology bias (discovery thresholds)
     - The modified world state (niche construction)
     - Developmental body parameters (heritable variation)
  6. The environment that Generation N reshaped IS Generation N+1's starting conditions

This is the loop the individual pieces were built for.
"""

import os
import json
import copy
import numpy as np
from collections import defaultdict
from environment import Environment, Organism, NPC
from physics_world import PhysicsWorld
from mental_model import build_mental_model, action_to_hash
from receptor_discovery import discover, build_tests
from model import compute_obs_indices

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')


class EvolvingOrganism:
    """An organism with heritable traits."""

    def __init__(self, organism_id, parent_bias=None, body_params=None):
        self.organism_id = organism_id
        self.topology_bias = parent_bias or {}
        self.body_params = body_params or {
            'num_limbs': 6,
            'growth_rate': 1.0,
            'pain_sensitivity': 1.0,
        }
        self.organism = None
        self.fitness = 0.0
        self.experience_log = []
        self.discovered_receptors = []

    def create_organism(self, rng):
        self.organism = Organism(
            num_limbs=self.body_params['num_limbs'],
        )
        self.organism.reset(rng)
        return self.organism

    def mutate_body(self, rng):
        child_params = dict(self.body_params)
        child_params['growth_rate'] = np.clip(
            child_params['growth_rate'] + rng.normal(0, 0.05), 0.5, 1.5)
        child_params['pain_sensitivity'] = np.clip(
            child_params['pain_sensitivity'] + rng.normal(0, 0.05), 0.5, 2.0)
        return child_params


def run_generation(organisms, env_seed, world_state=None,
                   steps_per_episode=300, num_episodes=10, rng=None):
    """Run one generation: all organisms compete in the same persistent world."""
    if rng is None:
        rng = np.random.RandomState()

    env = Environment(seed=env_seed)
    ref_org = organisms[0].create_organism(rng)
    pw = PhysicsWorld(env, ref_org, num_objects=4, seed=env_seed)
    pw.add_compound_objects(['lever', 'spring_gate', 'hinged_barrier'])
    pw.enable_persistence()

    if world_state is not None:
        pw.restore_world_state(world_state)

    all_logs = {org.organism_id: [] for org in organisms}

    for ep in range(num_episodes):
        for evo_org in organisms:
            org = evo_org.create_organism(rng)
            ref_org = org
            pw.org = org
            org.physics_mode = True

            npc = NPC()
            npc.reset(rng)

            episode_reward = 0.0
            for step in range(steps_per_episode):
                npc.step(env, step)
                pw.apply_developmental_changes(step)
                actions = org.compute_optimal_actions(env, step, npc=npc)

                r = rng.random()
                if r < 0.02:
                    executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
                elif r < 0.07:
                    executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
                else:
                    executed = actions

                pw.apply_organism_forces(executed)
                pw.check_grips(executed)
                pw.step()
                obs, reward = org.step(executed, env, step, npc=npc)
                npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
                episode_reward += reward

            evo_org.fitness += episode_reward
            all_logs[evo_org.organism_id].extend(org.experience_log)

    for evo_org in organisms:
        evo_org.experience_log = all_logs[evo_org.organism_id]

    final_world_state = pw.save_world_state()
    return final_world_state


def discover_receptors_for_organism(evo_org):
    """Run receptor discovery on an organism's accumulated experience."""
    if len(evo_org.experience_log) < 100:
        return [], {}

    engine = build_mental_model(evo_org.experience_log)
    results = discover(evo_org.experience_log, engine, log_provenance='oracle')
    evo_org.discovered_receptors = results['discovered']

    bias = {}
    for receptor_id in results['discovered']:
        bias[receptor_id] = results['scores'].get(receptor_id, 0.5)
    return results['discovered'], bias


def select_and_reproduce(organisms, num_offspring, rng):
    """Fitness-proportional selection with topology bias inheritance."""
    fitnesses = np.array([max(0.01, org.fitness) for org in organisms])
    probs = fitnesses / fitnesses.sum()

    offspring = []
    for i in range(num_offspring):
        parent = organisms[rng.choice(len(organisms), p=probs)]

        child_bias = dict(parent.topology_bias)
        for receptor_id in parent.discovered_receptors:
            child_bias[receptor_id] = child_bias.get(receptor_id, 0.5) * 0.6

        child_body = parent.mutate_body(rng)

        child = EvolvingOrganism(
            organism_id=f"gen{i}",
            parent_bias=child_bias,
            body_params=child_body,
        )
        offspring.append(child)

    return offspring


def run_deep_time(num_generations=5, population_size=6, num_episodes=10,
                  steps_per_episode=300, seed=0):
    """Run the full deep time loop."""
    print("=== Deep Time Learning ===\n")
    print(f"  Generations: {num_generations}")
    print(f"  Population: {population_size}")
    print(f"  Episodes/gen: {num_episodes}")
    print(f"  Steps/episode: {steps_per_episode}\n")

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    world_state = None
    history = []

    for gen in range(num_generations):
        print(f"--- Generation {gen} ---")

        env_seed = rng.randint(0, 100000)
        world_state = run_generation(
            organisms, env_seed, world_state=world_state,
            steps_per_episode=steps_per_episode,
            num_episodes=num_episodes, rng=rng,
        )

        gen_discovered = set()
        for evo_org in organisms:
            discovered, bias = discover_receptors_for_organism(evo_org)
            gen_discovered.update(discovered)
            evo_org.topology_bias.update(bias)

        fitnesses = [org.fitness for org in organisms]
        avg_fitness = np.mean(fitnesses)
        best_fitness = max(fitnesses)
        total_receptors = len(gen_discovered)
        total_log = sum(len(org.experience_log) for org in organisms)

        print(f"  Fitness: avg={avg_fitness:.0f}  best={best_fitness:.0f}")
        print(f"  Discovered: {total_receptors} receptors across population")
        print(f"  Log entries: {total_log}")
        print(f"  World persisted: {world_state is not None}")

        bias_sizes = [len(org.topology_bias) for org in organisms]
        print(f"  Topology bias: avg={np.mean(bias_sizes):.0f} receptors inherited")

        gen_record = {
            'generation': gen,
            'avg_fitness': round(avg_fitness, 1),
            'best_fitness': round(best_fitness, 1),
            'discovered': sorted(gen_discovered),
            'num_discovered': total_receptors,
            'population_size': len(organisms),
            'total_log_entries': total_log,
            'avg_bias_size': round(np.mean(bias_sizes), 1),
        }
        history.append(gen_record)

        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            print(f"  Reproduced: {population_size} offspring with inherited bias + mutated bodies")
        print()

    print("=== DEEP TIME SUMMARY ===\n")
    for rec in history:
        print(f"  Gen {rec['generation']}: fitness={rec['avg_fitness']:8.0f}  "
              f"receptors={rec['num_discovered']:2d}  "
              f"bias={rec['avg_bias_size']:.0f}")

    all_discovered_ever = set()
    for rec in history:
        all_discovered_ever.update(rec['discovered'])
    print(f"\n  Total unique receptors across all generations: {len(all_discovered_ever)}")

    gen0 = set(history[0]['discovered'])
    gen_last = set(history[-1]['discovered'])
    gained = gen_last - gen0
    lost = gen0 - gen_last
    if gained:
        print(f"  Gained (not in Gen 0, in final): {sorted(gained)}")
    if lost:
        print(f"  Lost (in Gen 0, not in final): {sorted(lost)}")

    fitness_trend = [r['avg_fitness'] for r in history]
    if len(fitness_trend) >= 2:
        if fitness_trend[-1] > fitness_trend[0]:
            print(f"\n  Fitness trend: IMPROVING ({fitness_trend[0]:.0f} -> {fitness_trend[-1]:.0f})")
        else:
            print(f"\n  Fitness trend: DECLINING ({fitness_trend[0]:.0f} -> {fitness_trend[-1]:.0f})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'deep_time_results.json'), 'w') as f:
        json.dump({
            'history': history,
            'all_discovered': sorted(all_discovered_ever),
            'gained': sorted(gained),
            'lost': sorted(lost),
        }, f, indent=2)
    print(f"\n  Results saved to data/deep_time_results.json")
    return history


if __name__ == '__main__':
    run_deep_time(num_generations=5, population_size=4,
                  num_episodes=5, steps_per_episode=200, seed=42)

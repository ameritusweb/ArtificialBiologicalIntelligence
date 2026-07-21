import os
import json
import time
import math
import numpy as np
import torch
from collections import defaultdict
from environment import Environment, Organism, NPC
from model import HierarchicalPolicy, compute_obs_indices
from mental_model import build_mental_model, action_to_hash
from train import (augment_with_mental_model, augment_with_patterns,
                   augment_with_agency, augment_with_concepts, DEVICE)
from receptor_discovery import build_tests, discover
from topology_inheritance import BIAS_FACTOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class OrganismAsNPC:
    MAX_SPEED = 3.0
    MAX_ANG_SPEED = 2.0
    SIGNAL_RANGE = 8.0
    REPEL_SIGNAL = (1, 1, 0, 0)
    CALM_SIGNAL = (1, 0, 1, 0)

    def __init__(self, organism):
        self.org = organism
        self.signal_active = False
        self.signal_type = None
        self.signal_timer = 0
        self.signal_source_x = 0.0
        self.signal_source_y = 0.0
        self._prev_speed = 0.0
        self._erraticism = 0.0

    @property
    def x(self): return self.org.x
    @property
    def y(self): return self.org.y
    @property
    def heading(self): return self.org.heading
    @property
    def vx(self): return self.org.vx
    @property
    def vy(self): return self.org.vy
    @property
    def omega(self): return self.org.omega

    def speed_val(self):
        return math.sqrt(self.org.vx**2 + self.org.vy**2)

    def distance_to(self, px, py):
        return math.sqrt((self.org.x - px)**2 + (self.org.y - py)**2)

    @property
    def erraticism(self):
        self._erraticism = 0.9 * self._erraticism + 0.1 * abs(self.org.omega)
        return self._erraticism

    @property
    def acceleration(self):
        spd = self.speed_val()
        acc = (spd - self._prev_speed) / 0.05
        self._prev_speed = spd
        return acc

    @property
    def emission_bits(self):
        em_start = self.org.NUM_LIMBS * 3 + (self.org.num_segments - 1) * 2
        return self.org.last_actions[em_start:]

    def receive_signal(self, emission_bits, org_x, org_y):
        dist = math.sqrt((self.org.x - org_x)**2 + (self.org.y - org_y)**2)
        if dist > self.SIGNAL_RANGE:
            return
        signal = tuple(int(b) for b in emission_bits)
        if signal == self.REPEL_SIGNAL or signal == self.CALM_SIGNAL:
            self.signal_active = True
            self.signal_type = 'repel' if signal == self.REPEL_SIGNAL else 'calm'
            self.signal_timer = 10
            self.signal_source_x = org_x
            self.signal_source_y = org_y

    def to_dict(self):
        return {
            'x': round(self.org.x, 3), 'y': round(self.org.y, 3),
            'heading': round(self.org.heading, 3),
            'speed': round(self.speed_val(), 3),
            'accel': round(self.acceleration, 3),
            'omega': round(self.org.omega, 3),
            'erratic': round(self._erraticism, 3),
            'signal': self.signal_type if self.signal_active else None,
            'signal_timer': self.signal_timer if self.signal_active else 0,
            'emission': [int(b) for b in self.emission_bits],
        }


def find_nearest(org_idx, population):
    org = population[org_idx]
    best_dist, best_idx = float('inf'), -1
    for j, other in enumerate(population):
        if j == org_idx:
            continue
        d = math.sqrt((org.x - other.x)**2 + (org.y - other.y)**2)
        if d < best_dist:
            best_dist = d
            best_idx = j
    return best_idx


def run_population_episode(env, population, adapters, steps=300):
    pop_size = len(population)
    rewards = [0.0] * pop_size
    all_logs = [[] for _ in range(pop_size)]

    for step in range(steps):
        actions_list = []
        nearest_list = []

        for i, org in enumerate(population):
            ni = find_nearest(i, population)
            nearest_list.append(ni)
            actions = org.compute_optimal_actions(env, step, npc=adapters[ni])
            actions_list.append(actions)

        for i, org in enumerate(population):
            ni = nearest_list[i]
            obs, reward = org.step(actions_list[i], env, step, npc=adapters[ni])
            rewards[i] += reward

        em_start = population[0].NUM_LIMBS * 3 + (population[0].num_segments - 1) * 2
        for i in range(pop_size):
            emission = actions_list[i][em_start:]
            for j in range(pop_size):
                if i != j:
                    adapters[j].receive_signal(emission, population[i].x, population[i].y)

        for i, org in enumerate(population):
            all_logs[i].extend(org.experience_log[-1:])

    return rewards, all_logs


def run_population_experiment(n_generations=5, pop_size=8, episodes_per_gen=50,
                               steps=300, tier=0):
    print(f"\n{'='*70}")
    print(f"  POPULATION EVOLUTION: {pop_size} organisms, {n_generations} generations")
    print(f"{'='*70}")

    idx = compute_obs_indices(6)
    base_tests = build_tests()
    gen_history = []
    parent_weights = None
    parent_topology = None

    for gen in range(n_generations):
        print(f"\n  --- Generation {gen} ---")
        rng = np.random.RandomState(gen * 7777)
        combined_log = []
        fitness_scores = []

        for ep in range(episodes_per_gen):
            env = Environment(seed=rng.randint(0, 100000))
            population = []
            adapters = []
            for i in range(pop_size):
                org = Organism()
                org.reset(rng)
                population.append(org)
                adapters.append(OrganismAsNPC(org))

            rewards, logs = run_population_episode(env, population, adapters, steps)

            for i in range(pop_size):
                if ep == 0:
                    fitness_scores.append(rewards[i])
                else:
                    fitness_scores[i] += rewards[i]
                combined_log.extend(logs[i])

            if (ep + 1) % 25 == 0:
                print(f"    Episode {ep+1}/{episodes_per_gen}")

        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        min_fitness = np.min(fitness_scores)
        fitness_var = np.var(fitness_scores)

        print(f"    Population: avg={avg_fitness:.1f} max={max_fitness:.1f} "
              f"min={min_fitness:.1f} var={fitness_var:.0f}")
        print(f"    Combined log: {len(combined_log):,} entries")

        ranked = np.argsort(fitness_scores)[::-1]
        top_half = ranked[:pop_size // 2]

        weighted_log = []
        for i in range(pop_size):
            rank = list(ranked).index(i)
            weight = 2.0 if rank < pop_size // 2 else 1.0
            org_entries = [e for e in combined_log[i * steps * episodes_per_gen:
                                                    (i + 1) * steps * episodes_per_gen]]
            n_samples = int(len(org_entries) * weight)
            if org_entries:
                indices = np.random.choice(len(org_entries),
                                          min(n_samples, len(org_entries)), replace=True)
                for idx_val in indices:
                    weighted_log.append(org_entries[idx_val])

        if not weighted_log:
            weighted_log = combined_log

        engine = build_mental_model(weighted_log[:min(50000, len(weighted_log))],
                                    core_obs_dim=idx['core_obs_dim'])
        stats = engine.get_stats()
        print(f"    Mental model: {stats['total_mappings']:,} mappings")

        biased_thresholds = None
        if parent_topology is not None:
            biased_thresholds = {}
            for i, test in enumerate(base_tests):
                if i < len(parent_topology) and parent_topology[i] == 1:
                    biased_thresholds[test.receptor_id] = test.threshold * BIAS_FACTOR

        disc_results = discover(weighted_log[:min(50000, len(weighted_log))],
                               engine, threshold_overrides=biased_thresholds)

        emission_counts = defaultdict(int)
        for e in combined_log[:10000]:
            em = tuple(int(b) for b in e['action'][18:22]) if len(e['action']) >= 22 else (0,0,0,0)
            if sum(em) > 0:
                emission_counts[em] += 1

        parent_topology = disc_results['topology_vector']

        gen_record = {
            'generation': gen,
            'avg_fitness': round(avg_fitness, 1),
            'max_fitness': round(max_fitness, 1),
            'min_fitness': round(min_fitness, 1),
            'fitness_variance': round(fitness_var, 1),
            'discovered_count': len(disc_results['discovered']),
            'discovered': disc_results['discovered'],
            'signal_usage': {str(k): v for k, v in sorted(emission_counts.items(),
                            key=lambda x: x[1], reverse=True)[:5]},
            'social_scores': {
                'other_detection': disc_results['scores'].get('other_detection', 0),
                'empathic_concern': disc_results['scores'].get('empathic_concern', 0),
                'behavioral_prediction': disc_results['scores'].get('behavioral_prediction', 0),
            },
        }
        gen_history.append(gen_record)

        print(f"    Discovered: {len(disc_results['discovered'])} receptors")
        print(f"    Social: other={gen_record['social_scores']['other_detection']:.3f} "
              f"empathy={gen_record['social_scores']['empathic_concern']:.3f} "
              f"prediction={gen_record['social_scores']['behavioral_prediction']:.3f}")
        print(f"    Signals: {dict(list(emission_counts.items())[:3])}")

    print(f"\n  {'='*60}")
    print(f"  POPULATION EVOLUTION SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Gen':>4} {'Avg':>8} {'Max':>8} {'Disc':>5} {'Other':>7} {'Empathy':>8} {'Predict':>8}")
    for g in gen_history:
        print(f"  {g['generation']:>4} {g['avg_fitness']:>8.1f} {g['max_fitness']:>8.1f} "
              f"{g['discovered_count']:>5} "
              f"{g['social_scores']['other_detection']:>7.3f} "
              f"{g['social_scores']['empathic_concern']:>8.3f} "
              f"{g['social_scores']['behavioral_prediction']:>8.3f}")

    single_org_disc = 20
    pop_disc = gen_history[-1]['discovered_count']
    print(f"\n  Single-organism discovery: {single_org_disc} receptors")
    print(f"  Population discovery:     {pop_disc} receptors")
    if pop_disc > single_org_disc:
        novel = set(gen_history[-1]['discovered']) - set(gen_history[0]['discovered'])
        print(f"  Novel from population pressure: {sorted(novel)}")

    return gen_history


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    history = run_population_experiment(n_generations=5, pop_size=8,
                                        episodes_per_gen=50, steps=300)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} min")

    path = os.path.join(DATA_DIR, 'population_evolution.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"  Saved: {path}")
    print("\nStep 36 complete.")

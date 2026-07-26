"""Multi-Agent Infrastructure.

Multiple evolving organisms in the same environment, each with their own
policy, mental model, and receptor topology. Organisms observe each other
as conspecifics — position, heading, emission tokens, distress. Tokens
propagate between organisms: one organism's emission_bits appear in
another's observation vector.

This is the infrastructure for language emergence (T106), population
density sweeps, and the specialization theorem (T109).
"""

import math
import numpy as np
from environment import Environment, Organism, NPC
from model import compute_obs_indices
from mental_model import build_mental_model, action_to_hash


class AgentProxy:
    """Wraps an Organism to look like an NPC to other organisms.

    Other organisms observe this agent through their _get_npc_obs method.
    The proxy exposes the same interface as NPC: x, y, heading, vx, vy,
    omega, erraticism, emission_bits, speed_val(), distance_to().
    """

    def __init__(self, organism):
        self.org = organism
        self.x = organism.x
        self.y = organism.y
        self.heading = organism.heading
        self.vx = organism.vx
        self.vy = organism.vy
        self.omega = organism.omega
        self.prev_speed = 0.0
        self.erraticism = 0.0
        self.emission_bits = np.zeros(4, dtype=int)
        self.acceleration = 0.0

    def sync(self):
        """Sync proxy state from the organism after a step."""
        prev_speed = math.sqrt(self.vx**2 + self.vy**2)
        self.x = self.org.x
        self.y = self.org.y
        self.heading = self.org.heading
        self.vx = self.org.vx
        self.vy = self.org.vy
        self.omega = self.org.omega
        curr_speed = math.sqrt(self.vx**2 + self.vy**2)
        self.acceleration = curr_speed - prev_speed
        self.erraticism = min(1.0, abs(self.acceleration) + abs(self.omega) * 0.3)
        self.prev_speed = curr_speed

    def set_emission(self, action):
        """Extract emission bits from an organism's action vector."""
        L = self.org.NUM_LIMBS
        num_joints = self.org.num_segments - 1
        joint_end = L * 3 + num_joints * 2
        if len(action) > joint_end:
            self.emission_bits = action[joint_end:joint_end + 4].astype(int)

    def speed_val(self):
        return math.sqrt(self.vx**2 + self.vy**2)

    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)

    def step(self, env, time_step):
        pass

    def receive_signal(self, emission_bits, sender_x, sender_y):
        pass


class MultiAgentEnvironment:
    """Manages N organisms in a shared environment.

    Each organism sees the nearest other organism through the NPC
    observation block (12 dims). Token emission from one organism
    becomes token observation for others.
    """

    def __init__(self, env, organisms, rng=None):
        self.env = env
        self.organisms = organisms
        self.proxies = [AgentProxy(org) for org in organisms]
        self.rng = rng or np.random.RandomState()
        self.N = len(organisms)

    def get_nearest_proxy(self, organism_idx):
        """Return the AgentProxy of the nearest other organism."""
        org = self.organisms[organism_idx]
        best_dist = float('inf')
        best_proxy = None
        for j, proxy in enumerate(self.proxies):
            if j == organism_idx:
                continue
            d = math.sqrt((org.x - proxy.x)**2 + (org.y - proxy.y)**2)
            if d < best_dist:
                best_dist = d
                best_proxy = proxy
        return best_proxy

    def step_all(self, actions_list, time_step, **kwargs):
        """Step all organisms simultaneously.

        actions_list: list of action arrays, one per organism.
        Returns list of (obs, reward) tuples.
        """
        results = []
        for i, (org, action) in enumerate(zip(self.organisms, actions_list)):
            nearest = self.get_nearest_proxy(i)
            obs, reward = org.step(action, self.env, time_step, npc=nearest, **kwargs)
            results.append((obs, reward))

        # Sync proxies AFTER all organisms have stepped
        for i, (org, action) in enumerate(zip(self.organisms, actions_list)):
            self.proxies[i].sync()
            self.proxies[i].set_emission(action)

        return results

    def reset_all(self, rng=None):
        """Reset all organisms to random positions."""
        r = rng or self.rng
        for org in self.organisms:
            org.reset(r)
        for proxy in self.proxies:
            proxy.sync()


class PopulationManager:
    """Manages a population of evolving organisms across generations.

    Handles:
    - Creating organisms with heritable traits
    - Running episodes with multi-agent interaction
    - Fitness-proportional selection and reproduction
    - Topology bias inheritance
    - Token propagation tracking
    """

    def __init__(self, population_size, seed=42):
        from deep_time import EvolvingOrganism
        self.population_size = population_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.idx = compute_obs_indices()

        self.evo_organisms = [
            EvolvingOrganism(f"pop_{i}") for i in range(population_size)
        ]
        self.generation = 0

        # Token tracking
        self.token_events = []
        self.token_receptor_active = [False] * population_size

    def create_organisms(self):
        """Create Organism instances for all population members."""
        organisms = []
        for evo_org in self.evo_organisms:
            org = evo_org.create_organism(self.rng)
            organisms.append(org)
        return organisms

    def run_generation(self, model, engine, tree, cog_detector,
                       receptor_bank=None, episode_bank=None,
                       activation_mgr=None,
                       num_episodes=5, steps_per_episode=1000):
        """Run one generation with all organisms interacting."""
        from live_receptors import LiveReceptorBank
        from thinking_substrate import ThinkingTree
        from train import EXPLORE_RATE, PROBE_RATE_FLOOR

        organisms = self.create_organisms()
        env = Environment(seed=self.rng.randint(0, 100000))
        multi_env = MultiAgentEnvironment(env, organisms, self.rng)

        num_actions = self.idx['num_actions']
        core_obs_dim = self.idx['core_obs_dim']
        obs_dim = self.idx['obs_dim']
        num_thinking_channels = self.idx.get('num_thinking_channels', 6)

        # Per-organism state
        prev_action_hashes = [0] * self.N
        per_org_banks = []
        for i in range(self.N):
            if receptor_bank is not None:
                bank = LiveReceptorBank()
                per_org_banks.append(bank)
            else:
                per_org_banks.append(None)

        gen_token_emissions = 0
        gen_token_receptions = 0

        for ep in range(num_episodes):
            multi_env.reset_all(self.rng)
            for bank in per_org_banks:
                if bank is not None:
                    bank.reset()

            for step in range(steps_per_episode):
                actions_list = []

                for i, (evo_org, org) in enumerate(zip(self.evo_organisms, organisms)):
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    # Cognitive state
                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hashes[i])
                    org.thought_type_id = tt
                    org.concept_id = ca

                    # Thinking
                    budget = int(evo_org.body_params.get('thinking_budget', 24))
                    tree.max_simulations = budget
                    v_keys = {k: v for k, v in evo_org.body_params.items()
                              if k.startswith('v_')}
                    tree.v_weights = v_keys if v_keys else None
                    thinking_cost = float(evo_org.body_params.get('thinking_cost', 0.001))

                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)

                    # Action selection
                    if self.generation == 0:
                        nearest = multi_env.get_nearest_proxy(i)
                        actions = org.compute_optimal_actions(env, step, npc=nearest)
                        executed = actions
                    else:
                        window = org.get_observation_window()
                        policy_action, _ = model.predict(window)
                        executed = policy_action

                    r = self.rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(num_actions, dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = self.rng.randint(0, 2, size=num_actions).astype(np.int32)

                    actions_list.append(executed)

                # Step all organisms simultaneously
                results = multi_env.step_all(actions_list, step)

                # Post-step processing
                for i, ((obs, reward), executed) in enumerate(zip(results, actions_list)):
                    self.evo_organisms[i].fitness += reward
                    prev_action_hashes[i] = action_to_hash(executed)

                    # Live receptors
                    if per_org_banks[i] is not None:
                        rc = per_org_banks[i].compute(obs, executed, engine, reward)
                        if activation_mgr is not None:
                            activation_mgr.update_live(rc)
                            rc, _ = activation_mgr.apply_mask(rc, org.episode_receptor_channels)
                        organisms[i].receptor_channels = rc

                    # Track token emissions
                    emission = executed[organisms[i].NUM_LIMBS * 3:]
                    if len(emission) >= 4 and np.sum(emission[:4]) > 0:
                        gen_token_emissions += 1

                    # Check if any organism received a token (nearest proxy has emission)
                    nearest = multi_env.get_nearest_proxy(i)
                    if nearest is not None and np.sum(nearest.emission_bits) > 0:
                        dist = nearest.distance_to(organisms[i].x, organisms[i].y)
                        if dist < 8.0:
                            gen_token_receptions += 1

            # Episode-level receptors
            if episode_bank is not None:
                for i, org in enumerate(organisms):
                    erc = episode_bank.compute(org.experience_log, engine)
                    org.episode_receptor_channels = erc

            # Collect logs
            for evo_org, org in zip(self.evo_organisms, organisms):
                evo_org.experience_log.extend(org.experience_log)

        stats = {
            'generation': self.generation,
            'population_size': self.N,
            'token_emissions': gen_token_emissions,
            'token_receptions': gen_token_receptions,
            'fitnesses': [float(eo.fitness) for eo in self.evo_organisms],
            'avg_fitness': float(np.mean([eo.fitness for eo in self.evo_organisms])),
        }

        return organisms, stats

    def select_and_reproduce(self):
        """Fitness-proportional selection with topology bias inheritance."""
        from deep_time import select_and_reproduce
        self.evo_organisms = select_and_reproduce(
            self.evo_organisms, self.population_size, self.rng)
        self.generation += 1
        for i, evo_org in enumerate(self.evo_organisms):
            evo_org.organism_id = f"gen{self.generation}_{i}"
            evo_org.experience_log = []
            evo_org.fitness = 0.0

    def get_population_topology(self):
        """Return the union and per-individual receptor sets."""
        individual_sets = []
        for evo_org in self.evo_organisms:
            individual_sets.append(set(evo_org.discovered_receptors))
        union = set().union(*individual_sets) if individual_sets else set()
        intersection = set.intersection(*individual_sets) if individual_sets and all(individual_sets) else set()
        return {
            'union': sorted(union),
            'intersection': sorted(intersection),
            'union_size': len(union),
            'intersection_size': len(intersection),
            'individual_sizes': [len(s) for s in individual_sets],
            'max_individual': max(len(s) for s in individual_sets) if individual_sets else 0,
        }

    @property
    def N(self):
        return self.population_size

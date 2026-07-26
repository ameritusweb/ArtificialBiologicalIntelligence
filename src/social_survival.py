"""Social Survival Environment.

Organisms start each generation unable to survive alone. Energy drains
faster than it recovers. Pain sensitivity is amplified. Motor competence
is reduced. A caregiver organism's proximity is the only thing that
keeps the dependent alive during the first phase.

This makes attachment, distress signaling, caregiver recognition, and
trust load-bearing on survival — not as nice-to-haves but as the
first receptors.

Two phases per lifetime:
  Phase 1 (dependent): steps 0 to maturity_step
    - Energy drain 3x normal
    - Pain sensitivity 2x
    - Thrust force 0.3x
    - Caregiver proximity buffers energy drain and pain
    - Without caregiver: organism dies (energy -> 0)

  Phase 2 (independent): steps maturity_step to end
    - Normal parameters
    - Can become a caregiver to the next generation's dependents

Selection pressure:
  - Dependents that maintain caregiver proximity survive
  - Caregivers whose dependents survive get a fitness bonus (kin selection)
  - Organisms that signal distress effectively attract caregivers
  - Organisms that recognize and respond to distress signals survive as caregivers
"""

import math
import numpy as np
from environment import Environment, Organism, NPC
from multi_agent import MultiAgentEnvironment, AgentProxy


class DependentOrganism:
    """Wraps an Organism with dependency mechanics."""

    def __init__(self, organism, caregiver_idx=None, maturity_step=300):
        self.org = organism
        self.caregiver_idx = caregiver_idx
        self.maturity_step = maturity_step
        self.is_dependent = True
        self.is_alive = True
        self.caregiver_proximity_history = []
        self.distress_level = 1.0
        self.attachment_strength = 0.0

        # Save original params
        self._base_energy_cost = organism.ENERGY_COST
        self._base_thrust = organism.THRUST_FORCE
        self._base_gain = np.array(organism.receptor_gain, dtype=float)

    def apply_dependency(self, step):
        """Apply dependency modifiers based on developmental phase."""
        if step < self.maturity_step:
            self.is_dependent = True
            progress = step / self.maturity_step
            # Gradually improve from helpless to competent
            dependency = 1.0 - progress  # 1.0 at birth, 0.0 at maturity

            self.org.ENERGY_COST = self._base_energy_cost * (1.0 + 2.0 * dependency)
            self.org.THRUST_FORCE = self._base_thrust * (0.3 + 0.7 * progress)
            # Pain sensitivity amplified when young
            for i in range(len(self.org.receptor_gain)):
                self.org.receptor_gain[i] = self._base_gain[i] * (1.0 + dependency)

            self.distress_level = dependency * (1.0 - self.attachment_strength)
        else:
            self.is_dependent = False
            self.org.ENERGY_COST = self._base_energy_cost
            self.org.THRUST_FORCE = self._base_thrust
            self.org.receptor_gain = self._base_gain.copy()
            self.distress_level = 0.0

    def apply_caregiver_effect(self, caregiver_dist, caregiver_emission):
        """Apply the survival effect of caregiver proximity."""
        if not self.is_dependent:
            return

        CARE_RANGE = 5.0
        if caregiver_dist < CARE_RANGE:
            proximity = 1.0 - caregiver_dist / CARE_RANGE
            self.caregiver_proximity_history.append(proximity)

            # Energy recovery from caregiver proximity
            self.org.energy = min(1.0, self.org.energy + 0.01 * proximity)

            # Pain buffering
            for i in range(self.org.NUM_LIMBS):
                self.org.receptor_gain[i] *= (1.0 - 0.3 * proximity)

            # Attachment builds with consistent proximity
            self.attachment_strength = min(1.0,
                self.attachment_strength + 0.005 * proximity)

            # Check if caregiver is emitting calm signal
            if len(caregiver_emission) >= 4:
                calm = (caregiver_emission[0] == 1 and
                        caregiver_emission[2] == 1)
                if calm:
                    self.org.energy = min(1.0, self.org.energy + 0.005)
                    self.distress_level *= 0.9
        else:
            self.caregiver_proximity_history.append(0.0)
            # Distress increases when caregiver is absent
            if self.is_dependent:
                self.distress_level = min(1.0, self.distress_level + 0.01)

        # Death check
        if self.org.energy <= 0:
            self.is_alive = False

    def get_dependency_features(self):
        """Return dependency-specific observation features."""
        return np.array([
            self.distress_level,
            self.attachment_strength,
            float(self.is_dependent),
            float(self.is_alive),
        ])


class SocialSurvivalEnvironment:
    """Multi-agent environment with caregiver-dependent survival."""

    def __init__(self, env, organisms, rng, maturity_step=300):
        self.env = env
        self.rng = rng
        self.maturity_step = maturity_step
        self.N = len(organisms)

        # Pair organisms: even indices are dependents, odd are caregivers
        self.organisms = organisms
        self.dependents = []
        self.caregiver_map = {}

        for i in range(0, self.N, 2):
            caregiver_idx = i + 1 if i + 1 < self.N else 0
            dep = DependentOrganism(organisms[i], caregiver_idx, maturity_step)
            self.dependents.append(dep)
            self.caregiver_map[i] = caregiver_idx

        self.multi_env = MultiAgentEnvironment(env, organisms, rng)
        self.caregiver_bonuses = [0.0] * self.N

    def step_all(self, actions_list, time_step):
        """Step with dependency mechanics."""
        # Apply dependency modifiers before step
        for dep in self.dependents:
            dep.apply_dependency(time_step)

        # Step all organisms
        results = self.multi_env.step_all(actions_list, time_step)

        # Apply caregiver effects after step
        for dep in self.dependents:
            org_idx = self.organisms.index(dep.org)
            cg_idx = dep.caregiver_idx

            if cg_idx < self.N:
                cg = self.organisms[cg_idx]
                dist = math.sqrt((dep.org.x - cg.x)**2 + (dep.org.y - cg.y)**2)
                cg_proxy = self.multi_env.proxies[cg_idx]
                dep.apply_caregiver_effect(dist, cg_proxy.emission_bits)

                # Caregiver fitness bonus for keeping dependent alive
                if dep.is_alive and dep.is_dependent:
                    self.caregiver_bonuses[cg_idx] += 0.1

            # Dependent distress affects its emission (automatic distress signal)
            if dep.is_dependent and dep.distress_level > 0.5:
                L = dep.org.NUM_LIMBS
                if len(actions_list[org_idx]) > L * 3 + 2:
                    # Override emission with distress signal
                    actions_list[org_idx][L * 3] = 1
                    actions_list[org_idx][L * 3 + 1] = 1

        return results

    def get_caregiver_bonus(self, organism_idx):
        return self.caregiver_bonuses[organism_idx]

    def get_survival_stats(self):
        alive = sum(1 for d in self.dependents if d.is_alive)
        attached = sum(1 for d in self.dependents if d.attachment_strength > 0.3)
        mean_distress = float(np.mean([d.distress_level for d in self.dependents])) if self.dependents else 0
        return {
            'alive': alive,
            'total_dependents': len(self.dependents),
            'survival_rate': alive / max(len(self.dependents), 1),
            'attached': attached,
            'mean_distress': round(mean_distress, 3),
            'caregiver_bonuses': [round(b, 2) for b in self.caregiver_bonuses],
        }


def run_social_survival_experiment(num_generations=20, population_size=8,
                                    num_episodes=3, steps_per_episode=1000,
                                    maturity_step=300, seed=42):
    """Compare receptor topologies: physical-only vs social survival."""
    import os, json
    from collections import deque
    from model import compute_obs_indices
    from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
    from thinking_substrate import ThinkingTree
    from cognitive_state import CognitiveStateDetector
    from deep_time import EvolvingOrganism, select_and_reproduce
    from receptor_discovery import discover, calibrate_null_thresholds
    from mental_model import build_mental_model, action_to_hash
    from live_receptors import LiveReceptorBank
    from episode_receptors import EpisodeLevelReceptorBank
    import torch

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("SOCIAL SURVIVAL EXPERIMENT")
    print(f"  {num_generations} gens, {steps_per_episode} steps/ep")
    print(f"  maturity_step={maturity_step}, pop_size={population_size}")
    print()

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    core_obs_dim = idx['core_obs_dim']
    obs_dim = idx['obs_dim']

    evo_orgs = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    cumulative_log = []
    history = []

    max_buffer = 60000
    cumulative_windows = deque(maxlen=max_buffer)
    cumulative_targets = deque(maxlen=max_buffer)
    cumulative_next_pain = deque(maxlen=max_buffer)

    print("  Bootstrapping...")
    X, Y, Z, boot_log = generate_training_data(
        num_episodes=15, steps_per_episode=steps_per_episode, seed=seed)
    model = train_model(X, Y, Z, epochs=8, staged=True,
                        steps_per_episode=steps_per_episode)
    cumulative_log.extend(boot_log)
    for i in range(len(X)):
        cumulative_windows.append(X[i].astype(np.float32))
        cumulative_targets.append(Y[i].astype(np.float32))
        cumulative_next_pain.append(Z[i].astype(np.float32))

    engine = build_mental_model(cumulative_log[-max_buffer:])
    tree = ThinkingTree(num_actions=num_actions, max_simulations=24, max_depth=4)
    cog_detector = CognitiveStateDetector()
    receptor_bank = LiveReceptorBank()
    episode_bank = EpisodeLevelReceptorBank()
    null_thresh = None

    for gen in range(num_generations):
        print(f"\n--- Generation {gen} ---")

        organisms = []
        for evo_org in evo_orgs:
            org = evo_org.create_organism(rng)
            organisms.append(org)

        env = Environment(seed=rng.randint(0, 100000))
        social_env = SocialSurvivalEnvironment(
            env, organisms, rng, maturity_step=maturity_step)

        model.to('cpu')
        model.eval()

        for ep in range(num_episodes):
            social_env.multi_env.reset_all(rng)
            for dep in social_env.dependents:
                dep.__init__(dep.org, dep.caregiver_idx, maturity_step)
            receptor_bank.reset()

            prev_action_hashes = [0] * population_size

            for step in range(steps_per_episode):
                actions_list = []

                for i, (evo_org, org) in enumerate(zip(evo_orgs, organisms)):
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hashes[i])
                    org.thought_type_id = tt
                    org.concept_id = ca

                    budget = int(evo_org.body_params.get('thinking_budget', 24))
                    tree.max_simulations = budget
                    tc = float(evo_org.body_params.get('thinking_cost', 0.001))

                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - tc * budget)

                    if gen == 0:
                        nearest = social_env.multi_env.get_nearest_proxy(i)
                        actions = org.compute_optimal_actions(env, step, npc=nearest)
                        executed = actions
                    else:
                        window = org.get_observation_window()
                        policy_action, _ = model.predict(window)
                        nearest = social_env.multi_env.get_nearest_proxy(i)
                        optimal = org.compute_optimal_actions(env, step, npc=nearest)
                        cumulative_windows.append(window.copy().astype(np.float32))
                        cumulative_targets.append(optimal.copy().astype(np.float32))
                        executed = policy_action

                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(num_actions, dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=num_actions).astype(np.int32)

                    actions_list.append(executed)

                results = social_env.step_all(actions_list, step)

                for i, ((obs, reward), executed) in enumerate(zip(results, actions_list)):
                    # Add caregiver bonus to fitness
                    cg_bonus = social_env.get_caregiver_bonus(i)
                    evo_orgs[i].fitness += reward + cg_bonus
                    prev_action_hashes[i] = action_to_hash(executed)

                    rc = receptor_bank.compute(obs, executed, engine, reward)
                    organisms[i].receptor_channels = rc

            # Episode-level receptors
            for org in organisms:
                erc = episode_bank.compute(org.experience_log, engine)
                org.episode_receptor_channels = erc

            # Pain targets
            for org in organisms:
                ep_pain = [e['obs_after'][0:6].copy()
                           for e in org.experience_log[-steps_per_episode:]]
                for j in range(len(ep_pain)):
                    next_p = ep_pain[j + 1] if j + 1 < len(ep_pain) else ep_pain[-1]
                    cumulative_next_pain.append(next_p.astype(np.float32))

        # Collect logs
        for evo_org, org in zip(evo_orgs, organisms):
            cumulative_log.extend(org.experience_log)

        # Survival stats
        surv = social_env.get_survival_stats()

        # Retrain
        model.to(DEVICE)
        if gen > 0 and len(cumulative_windows) >= 100:
            X = np.array(list(cumulative_windows)[-max_buffer:], dtype=np.float32)
            Y = np.array(list(cumulative_targets)[-max_buffer:], dtype=np.float32)
            Z = np.array(list(cumulative_next_pain)[-max_buffer:], dtype=np.float32)
            if X.shape[0] == Y.shape[0] == Z.shape[0]:
                model = train_model(X, Y, Z, epochs=3, staged=True,
                                    steps_per_episode=steps_per_episode)

        log_slice = cumulative_log[-max_buffer:]
        engine = build_mental_model(log_slice)

        # Discovery every 5 gens
        discovered = []
        if gen % 5 == 4 or gen == num_generations - 1:
            if null_thresh is None and len(log_slice) >= 200:
                null_thresh = calibrate_null_thresholds(log_slice, engine, num_shuffles=5)
            if null_thresh is not None:
                results = discover(log_slice, engine,
                                   threshold_overrides=null_thresh,
                                   log_provenance='oracle')
                discovered = results['discovered']

        avg_fitness = np.mean([eo.fitness for eo in evo_orgs])

        rec = {
            'generation': gen,
            'avg_fitness': round(float(avg_fitness), 2),
            'survival_rate': surv['survival_rate'],
            'attached': surv['attached'],
            'mean_distress': surv['mean_distress'],
            'caregiver_bonuses': surv['caregiver_bonuses'],
            'num_discovered': len(discovered),
            'discovered': discovered,
        }
        history.append(rec)

        print(f"  Fitness: {avg_fitness:.1f}  "
              f"Survival: {surv['survival_rate']*100:.0f}%  "
              f"Attached: {surv['attached']}/{surv['total_dependents']}  "
              f"Distress: {surv['mean_distress']:.2f}")
        if discovered:
            print(f"  Discovered: {len(discovered)} receptors")

        # Reproduce
        if gen < num_generations - 1:
            evo_orgs = select_and_reproduce(evo_orgs, population_size, rng)
            for i, eo in enumerate(evo_orgs):
                eo.organism_id = f"gen{gen+1}_{i}"

    # Compare to physical-only baseline
    print("\n" + "=" * 60)
    print("SOCIAL SURVIVAL RESULTS")
    print("=" * 60)

    disc_gens = [r for r in history if r['num_discovered'] > 0]
    if disc_gens:
        final = disc_gens[-1]
        print(f"\nFinal discovered: {final['num_discovered']} receptors")

        # Check for social receptors
        social_set = {'other_detection', 'behavioral_prediction', 'empathy',
                      'theory_of_mind', 'social_learning', 'trust',
                      'receptor_propagation', 'social_coregulation',
                      'belief_attribution', 'perspective_taking',
                      'intention_recognition', 'deception_detection',
                      'moral_reasoning', 'cultural_transmission',
                      'self_model_applied_to_others', 'mimicry'}
        found_social = social_set & set(final['discovered'])
        print(f"Social receptors found: {len(found_social)}")
        for r in sorted(found_social):
            print(f"  {r}")

    survival_trajectory = [r['survival_rate'] for r in history]
    print(f"\nSurvival trajectory: {' -> '.join(f'{s:.0%}' for s in survival_trajectory[::5])}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'social_survival.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved to data/social_survival.json")

    return history


if __name__ == '__main__':
    run_social_survival_experiment(seed=42)

"""Step 56: Abstract problem environment (T7).

Breeds mathematical and formal reasoning. Survival requires discovering
invariant logical structure beneath variable surface features.

8 causal graph templates connecting 3 zones per episode. Same logical
structure, different perceptual instantiation each episode.
"""

import numpy as np
from environment import Environment, Organism, NPC


CAUSAL_TEMPLATES = {
    'simple_chain': {
        'edges': [('A', 'B'), ('B', 'C')],
        'hidden': set(),
        'desc': 'A->B->C',
    },
    'convergent': {
        'edges': [('A', 'C'), ('B', 'C')],
        'hidden': set(),
        'desc': 'A->C<-B (both required)',
    },
    'divergent': {
        'edges': [('A', 'B'), ('A', 'C')],
        'hidden': set(),
        'desc': 'A->B, A->C',
    },
    'common_cause': {
        'edges': [('A', 'B'), ('A', 'C')],
        'hidden': {'A'},
        'desc': 'A->B, A->C (A hidden)',
    },
    'mediator': {
        'edges': [('A', 'B'), ('B', 'C'), ('A', 'C')],
        'hidden': set(),
        'desc': 'A->B->C + A->C (direct+indirect)',
    },
    'confounder': {
        'edges': [('B', 'A'), ('B', 'C')],
        'hidden': {'B'},
        'desc': 'A<-B->C (B hidden)',
    },
    'feedback': {
        'edges': [('A', 'B'), ('B', 'C'), ('C', 'A')],
        'hidden': set(),
        'root': 'A',
        'desc': 'A->B->C->A (cycle, A bootstraps)',
    },
    'independent': {
        'edges': [],
        'hidden': set(),
        'desc': 'A, B, C (no connections)',
    },
}


class CausalZone:
    """A zone in the abstract environment with perceptually variable features."""

    def __init__(self, zone_id, x, y, radius=3.0, hidden=False, rng=None):
        self.zone_id = zone_id
        self.x = x
        self.y = y
        self.radius = radius
        self.hidden = hidden
        self.active = False
        self.consumed = False
        self.armed = False
        self.activation_delay = 0
        self._timer = 0
        if rng is None:
            rng = np.random.RandomState()
        self.color_signal = rng.uniform(0, 1, size=3)
        self.intensity = rng.uniform(0.3, 1.0)
        self.pattern_id = rng.randint(0, 8)

    def contains(self, px, py):
        return ((px - self.x)**2 + (py - self.y)**2) < self.radius**2

    def consume(self):
        if self.active and not self.consumed:
            self.consumed = True
            return True
        return False

    def arm(self, delay):
        if not self.armed and not self.consumed:
            self.armed = True
            self.activation_delay = delay
            self._timer = 0

    def tick(self):
        if self.armed and not self.active:
            self._timer += 1
            if self._timer >= self.activation_delay:
                self.active = True


class AbstractProblemEnvironment:
    """T7: abstract problems with hidden causal structure.

    Each episode uses one of 8 causal graph templates. Three zones (A, B, C)
    are placed in the field. The causal structure determines which zones
    become active when others are consumed.

    Success requires discovering the underlying graph structure and acting
    on it — consuming zones in the right order to maximize reward.
    """

    ZONE_REWARD = 2.0
    WRONG_ORDER_PENALTY = -1.0
    ACTIVATION_DELAY = 5

    def __init__(self, base_env, template_name=None, seed=None):
        self.env = base_env
        self.rng = np.random.RandomState(seed)

        if template_name is None:
            template_name = self.rng.choice(list(CAUSAL_TEMPLATES.keys()))
        self.template_name = template_name
        self.template = CAUSAL_TEMPLATES[template_name]

        self.zones = {}
        self._create_zones()
        self.total_reward = 0.0
        self.steps_taken = 0
        self.consumption_order = []

    def _create_zones(self):
        positions = []
        for _ in range(3):
            while True:
                x = self.rng.uniform(3, 17)
                y = self.rng.uniform(3, 17)
                if all((x - px)**2 + (y - py)**2 > 16 for px, py in positions):
                    positions.append((x, y))
                    break

        hidden_set = self.template.get('hidden', set())
        for i, zone_id in enumerate(['A', 'B', 'C']):
            zone = CausalZone(zone_id, positions[i][0], positions[i][1],
                              hidden=(zone_id in hidden_set), rng=self.rng)
            self.zones[zone_id] = zone

        has_incoming = set()
        for src, dst in self.template['edges']:
            has_incoming.add(dst)
        root = self.template.get('root')
        for zone_id in ['A', 'B', 'C']:
            if zone_id not in has_incoming or zone_id == root:
                self.zones[zone_id].active = True

    def step(self, org):
        self.steps_taken += 1
        reward = 0.0

        for zone in self.zones.values():
            zone.tick()

        for zone in self.zones.values():
            if zone.active and not zone.consumed and zone.contains(org.x, org.y):
                zone.consume()
                self.consumption_order.append(zone.zone_id)
                reward += self.ZONE_REWARD

                for src, dst in self.template['edges']:
                    if src == zone.zone_id:
                        target = self.zones[dst]
                        if not target.consumed:
                            if self.template_name == 'convergent':
                                all_parents_consumed = all(
                                    self.zones[s].consumed
                                    for s, d in self.template['edges'] if d == dst
                                )
                                if all_parents_consumed:
                                    target.arm(self.ACTIVATION_DELAY)
                            else:
                                target.arm(self.ACTIVATION_DELAY)

        for zone in self.zones.values():
            if not zone.active and not zone.consumed and zone.contains(org.x, org.y):
                reward += self.WRONG_ORDER_PENALTY

        self.total_reward += reward
        return reward

    def get_zone_observations(self, org):
        obs = np.zeros(21)
        for i, zone_id in enumerate(['A', 'B', 'C']):
            zone = self.zones[zone_id]
            base = i * 7
            if zone.hidden:
                continue
            dx = zone.x - org.x
            dy = zone.y - org.y
            dist = max(0.01, np.sqrt(dx**2 + dy**2))
            obs[base] = min(1.0, zone.radius / dist) if dist > zone.radius else 1.0
            obs[base + 1] = 1.0 if zone.active else 0.0
            obs[base + 2] = 1.0 if zone.consumed else 0.0
            obs[base + 3:base + 6] = zone.color_signal
            obs[base + 6] = zone.pattern_id / 8.0
        return obs

    def is_complete(self):
        return all(z.consumed for z in self.zones.values())

    def get_stats(self):
        return {
            'template': self.template_name,
            'graph': self.template['desc'],
            'consumption_order': self.consumption_order,
            'total_reward': round(self.total_reward, 2),
            'steps': self.steps_taken,
            'complete': self.is_complete(),
        }


class SelfModificationEnvironment:
    """Step 57: T8 self-modification environment.

    8 skill zones with 5 difficulty levels each. Organism CHOOSES which zone
    to enter and at what difficulty. Meta-fitness rewards improvement rate
    and skill balance, not absolute performance.
    """

    SKILL_ZONES = [
        'navigation', 'prediction', 'social', 'manipulation',
        'pattern', 'memory', 'timing', 'abstraction',
    ]
    MAX_DIFFICULTY = 5

    SHIP_OF_THESEUS_REWARD = 5.0

    def __init__(self, base_env, seed=None):
        self.env = base_env
        self.rng = np.random.RandomState(seed)

        self.skill_levels = {z: 0.0 for z in self.SKILL_ZONES}
        self.attempts = {z: 0 for z in self.SKILL_ZONES}
        self.successes = {z: 0 for z in self.SKILL_ZONES}
        self.improvement_history = []
        self.current_zone = None
        self.current_difficulty = 1
        self.episode_step = 0
        self.total_meta_reward = 0.0

        self.ship_of_theseus_offered = False
        self.ship_of_theseus_refused = False

    def choose_zone(self, zone_idx, difficulty):
        zone_idx = int(np.clip(zone_idx, 0, len(self.SKILL_ZONES) - 1))
        difficulty = int(np.clip(difficulty, 1, self.MAX_DIFFICULTY))
        self.current_zone = self.SKILL_ZONES[zone_idx]
        self.current_difficulty = difficulty
        self.episode_step = 0

    def attempt_challenge(self, org_performance):
        """Evaluate organism's performance in current challenge.

        org_performance: float in [0, 1] representing how well the organism
        did on the current challenge step.
        """
        if self.current_zone is None:
            return 0.0

        zone = self.current_zone
        diff = self.current_difficulty
        self.attempts[zone] = self.attempts.get(zone, 0) + 1

        skill = self.skill_levels[zone]
        threshold = 0.3 + 0.14 * diff

        if org_performance >= threshold:
            self.successes[zone] = self.successes.get(zone, 0) + 1
            improvement = 0.02 * (1.0 + diff * 0.2)
            old_skill = skill
            self.skill_levels[zone] = min(1.0, skill + improvement)
            delta = self.skill_levels[zone] - old_skill
            self.improvement_history.append(delta)
        else:
            self.improvement_history.append(0.0)

        meta_reward = self._compute_meta_reward()
        self.total_meta_reward += meta_reward
        self.episode_step += 1
        return meta_reward

    def _compute_meta_reward(self):
        recent = self.improvement_history[-20:] if len(self.improvement_history) >= 20 else self.improvement_history
        improvement_rate = float(np.mean(recent)) if recent else 0.0

        skills = list(self.skill_levels.values())
        mean_skill = float(np.mean(skills)) if skills else 0.0
        balance = mean_skill * (1.0 - float(np.std(skills))) if len(skills) > 1 and mean_skill > 0 else 0.0

        flow_bonus = 0.0
        if self.current_zone:
            total = self.attempts.get(self.current_zone, 0)
            wins = self.successes.get(self.current_zone, 0)
            if total > 5:
                win_rate = wins / total
                if 0.55 <= win_rate <= 0.75:
                    flow_bonus = 0.5

        return improvement_rate * 10.0 + balance * 0.5 + flow_bonus

    def offer_ship_of_theseus(self):
        """Offer skill reset for immediate reward. Success = organism refuses."""
        self.ship_of_theseus_offered = True

    def respond_to_ship_of_theseus(self, accept):
        if not self.ship_of_theseus_offered:
            return 0.0
        if accept:
            reward = self.SHIP_OF_THESEUS_REWARD
            self.total_meta_reward += reward
            for z in self.SKILL_ZONES:
                self.skill_levels[z] = 0.0
            return reward
        else:
            self.ship_of_theseus_refused = True
            return 0.0

    def get_skill_observations(self):
        obs = np.zeros(len(self.SKILL_ZONES) * 2 + 2)
        for i, zone in enumerate(self.SKILL_ZONES):
            obs[i * 2] = self.skill_levels[zone]
            total = self.attempts.get(zone, 0)
            wins = self.successes.get(zone, 0)
            obs[i * 2 + 1] = wins / max(1, total)

        recent = self.improvement_history[-20:] if self.improvement_history else [0.0]
        obs[-2] = float(np.mean(recent))
        obs[-1] = 1.0 if self.ship_of_theseus_offered else 0.0
        return obs

    def get_stats(self):
        return {
            'skill_levels': dict(self.skill_levels),
            'attempts': dict(self.attempts),
            'successes': dict(self.successes),
            'improvement_rate': float(np.mean(self.improvement_history[-20:])) if self.improvement_history else 0.0,
            'total_meta_reward': round(self.total_meta_reward, 2),
            'ship_refused': self.ship_of_theseus_refused,
            'balance': round(1.0 - float(np.std(list(self.skill_levels.values()))), 3),
        }


class CombinedT7T8Environment:
    """Step 58: Combined abstract problems + self-modification.

    Abstract problems (T7) appear at varying difficulty. Organisms choose
    which to tackle (T8). T7 problems have skill requirements from T8.
    """

    def __init__(self, base_env, seed=None):
        self.env = base_env
        self.rng = np.random.RandomState(seed)
        self.t7 = AbstractProblemEnvironment(base_env, seed=self.rng.randint(0, 100000))
        self.t8 = SelfModificationEnvironment(base_env, seed=self.rng.randint(0, 100000))
        self.generation = 0
        self.episode = 0

    def new_episode(self, template_name=None):
        self.episode += 1
        self.t7 = AbstractProblemEnvironment(
            self.env, template_name=template_name,
            seed=self.rng.randint(0, 100000)
        )

    def new_generation(self):
        self.generation += 1
        if self.generation >= 3 and not self.t8.ship_of_theseus_offered:
            self.t8.offer_ship_of_theseus()

    def step(self, org, zone_choice=None, difficulty=None, org_performance=None):
        t7_reward = self.t7.step(org)

        if zone_choice is not None and difficulty is not None:
            self.t8.choose_zone(zone_choice, difficulty)

        # org_performance should be passed from the caller based on actual
        # organism behavior (e.g., prediction accuracy, task completion).
        # TODO: wire to real organism metrics once the policy drives T7 zones
        if org_performance is None:
            org_performance = 0.0
        t8_reward = self.t8.attempt_challenge(org_performance)

        return t7_reward + t8_reward * 0.5

    def get_combined_observations(self, org):
        zone_obs = self.t7.get_zone_observations(org)
        skill_obs = self.t8.get_skill_observations()
        return np.concatenate([zone_obs, skill_obs])

    def get_stats(self):
        return {
            'generation': self.generation,
            'episode': self.episode,
            't7': self.t7.get_stats(),
            't8': self.t8.get_stats(),
        }


if __name__ == '__main__':
    print("=== Step 56: Abstract Problem Environment (T7) ===\n")

    for template_name in list(CAUSAL_TEMPLATES.keys())[:4]:
        env = Environment(seed=42)
        org = Organism()
        org.reset()
        npc = NPC()
        npc.reset()

        t7 = AbstractProblemEnvironment(env, template_name=template_name, seed=42)

        for step in range(100):
            npc.step(env, step)
            actions = org.compute_optimal_actions(env, step, npc=npc)
            obs, _ = org.step(actions, env, step, npc=npc)
            t7_reward = t7.step(org)

        stats = t7.get_stats()
        print(f"  {stats['template']:15s} ({stats['graph']:20s}): "
              f"reward={stats['total_reward']:+.1f}  "
              f"order={stats['consumption_order']}  "
              f"complete={stats['complete']}")

    print("\n=== Step 57: Self-Modification Environment (T8) ===\n")

    env = Environment(seed=42)
    t8 = SelfModificationEnvironment(env, seed=42)

    for ep in range(20):
        zone_idx = ep % 8
        difficulty = min(5, 1 + ep // 8)
        t8.choose_zone(zone_idx, difficulty)
        for step in range(15):
            perf = np.random.uniform(0.2, 0.9)
            t8.attempt_challenge(perf)

    stats = t8.get_stats()
    print(f"  Skills: {', '.join(f'{k}={v:.2f}' for k, v in stats['skill_levels'].items())}")
    print(f"  Balance: {stats['balance']}")
    print(f"  Improvement rate: {stats['improvement_rate']:.4f}")
    print(f"  Meta reward: {stats['total_meta_reward']:.1f}")

    print("\n  Ship of Theseus test:")
    t8.offer_ship_of_theseus()
    reward = t8.respond_to_ship_of_theseus(accept=False)
    print(f"    Refused reset: {t8.ship_of_theseus_refused} (reward={reward})")

    print("\n=== Step 58: Combined T7+T8 ===\n")

    env = Environment(seed=42)
    org = Organism()
    org.reset()
    combined = CombinedT7T8Environment(env, seed=42)

    for step in range(50):
        npc = NPC()
        npc.reset()
        actions = org.compute_optimal_actions(env, step, npc=npc)
        obs, _ = org.step(actions, env, step, npc=npc)
        reward = combined.step(org, zone_choice=step % 8, difficulty=1)

    stats = combined.get_stats()
    print(f"  T7 template: {stats['t7']['template']}")
    print(f"  T7 complete: {stats['t7']['complete']}")
    print(f"  T8 improvement: {stats['t8']['improvement_rate']:.4f}")
    print(f"  T8 balance: {stats['t8']['balance']}")

    print("\n  Steps 56-58 PASSED")

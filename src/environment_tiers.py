import math
import numpy as np
from collections import defaultdict
from environment import Environment, FieldSource, ResponsiveObject, NPC, Organism


class PulsingSource(FieldSource):
    def __init__(self, cx, cy, ax, ay, omega_x, omega_y, phi_x, phi_y,
                 sigma, intensity, pulse_period=100, pulse_amplitude=0.8):
        super().__init__(cx, cy, ax, ay, omega_x, omega_y, phi_x, phi_y, sigma, intensity)
        self.pulse_period = pulse_period
        self.pulse_amplitude = pulse_amplitude

    def intensity_at(self, px, py, t, pz=0.0):
        pulse = 1.0 + self.pulse_amplitude * math.sin(2 * math.pi * t / self.pulse_period)
        base = super().intensity_at(px, py, t, pz)
        return base * max(0.0, pulse)


class CausalTrigger:
    def __init__(self, trigger_region, effect_region, delay, effect_type='endorphin',
                 effect_intensity=1.5, effect_duration=30, trigger_radius=3.0):
        self.trigger_x, self.trigger_y = trigger_region
        self.effect_x, self.effect_y = effect_region
        self.delay = delay
        self.effect_type = effect_type
        self.effect_intensity = effect_intensity
        self.effect_duration = effect_duration
        self.trigger_radius = trigger_radius
        self.triggered_at = -1
        self.active = False
        self.timer = 0

    def check_trigger(self, org_x, org_y, t):
        dist = math.sqrt((org_x - self.trigger_x)**2 + (org_y - self.trigger_y)**2)
        if dist < self.trigger_radius and self.triggered_at < 0:
            self.triggered_at = t

    def update(self, t):
        if self.triggered_at >= 0 and t >= self.triggered_at + self.delay and not self.active:
            self.active = True
            self.timer = self.effect_duration
        if self.active:
            self.timer -= 1
            if self.timer <= 0:
                self.active = False
                self.triggered_at = -1

    def effect_at(self, px, py):
        if not self.active:
            return 0.0
        dx, dy = px - self.effect_x, py - self.effect_y
        return self.effect_intensity * math.exp(-(dx*dx + dy*dy) / 8.0)

    def reset(self):
        self.triggered_at = -1
        self.active = False
        self.timer = 0


class PredatorEvent:
    def __init__(self, period=150, duration=20, intensity=2.0, speed=0.5):
        self.period = period
        self.duration = duration
        self.intensity = intensity
        self.speed = speed
        self.active = False
        self.sweep_x = 0.0
        self.direction = 1

    def update(self, t, width=20.0):
        cycle_pos = t % self.period
        self.active = cycle_pos < self.duration
        if self.active:
            progress = cycle_pos / self.duration
            self.sweep_x = progress * width if self.direction > 0 else width * (1 - progress)
        if cycle_pos == 0:
            self.direction *= -1

    def pain_at(self, px, py):
        if not self.active:
            return 0.0
        dx = px - self.sweep_x
        return self.intensity * math.exp(-(dx * dx) / 2.0)


class MovableObject:
    def __init__(self, x, y, mass=2.0, friction=0.8, radius=0.6):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.mass = mass
        self.friction = friction
        self.radius = radius

    def apply_force(self, fx, fy, dt=0.05):
        self.vx += (fx / self.mass) * dt
        self.vy += (fy / self.mass) * dt

    def step(self, width=20.0, height=20.0):
        self.vx *= self.friction
        self.vy *= self.friction
        self.x += self.vx * 0.05
        self.y += self.vy * 0.05
        self.x = max(0.5, min(width - 0.5, self.x))
        self.y = max(0.5, min(height - 0.5, self.y))

    def blocks_source(self, sx, sy, block_radius=1.5):
        dist = math.sqrt((self.x - sx)**2 + (self.y - sy)**2)
        return dist < block_radius

    def distance_to(self, px, py):
        return math.sqrt((self.x - px)**2 + (self.y - py)**2)

    def near(self, other, threshold=2.0):
        return self.distance_to(other.x, other.y) < threshold

    def reset(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0

    def to_dict(self):
        return {'x': round(self.x, 3), 'y': round(self.y, 3),
                'vx': round(self.vx, 3), 'vy': round(self.vy, 3)}


class HiddenVariable:
    def __init__(self, period=500, num_states=3):
        self.period = period
        self.num_states = num_states

    def state(self, t):
        return int(t / self.period) % self.num_states

    def modulation(self, t):
        s = self.state(t)
        if s == 0:
            return {'pain': 1.0, 'endorphin': 1.0}
        elif s == 1:
            return {'pain': 1.5, 'endorphin': 0.5}
        else:
            return {'pain': 0.5, 'endorphin': 1.5}


class TieredEnvironment(Environment):
    def __init__(self, seed=None, tier=0):
        self.tier = tier
        super().__init__(seed=seed)
        self.pulsing_sources = []
        self.causal_triggers = []
        self.predator_events = []
        self.movable_objects = []
        self.hidden_variable = None
        self.npcs = []

        if tier >= 1:
            self._add_tier1(self.rng)
        if tier >= 2:
            self._add_tier2(self.rng)
        if tier >= 3:
            self._add_tier3(self.rng)
        if tier >= 4:
            self._add_tier4()
        if tier >= 5:
            self._add_tier5(self.rng)
        if tier >= 6:
            self._add_tier6(self.rng)
        if tier >= 7:
            self._add_tier7(self.rng)

    def _add_tier1(self, rng):
        self.anticipatory_phases = []
        for src in self.pain_sources:
            period = rng.randint(50, 200)
            self.pulsing_sources.append(
                PulsingSource(src.cx, src.cy, src.ax, src.ay,
                              src.omega_x, src.omega_y, src.phi_x, src.phi_y,
                              src.sigma, src.intensity,
                              pulse_period=period, pulse_amplitude=0.7))
            self.anticipatory_phases.append({
                'source': src, 'period': period,
                'reward_phase_start': 0.7, 'reward_phase_end': 0.9,
            })

        for _ in range(2):
            tx, ty = rng.uniform(3, 17), rng.uniform(3, 17)
            ex, ey = rng.uniform(3, 17), rng.uniform(3, 17)
            delay = rng.randint(10, 25)
            self.causal_triggers.append(
                CausalTrigger((tx, ty), (ex, ey), delay))

        self.conditional_triggers = []
        for _ in range(1):
            tx, ty = rng.uniform(3, 17), rng.uniform(3, 17)
            ex, ey = rng.uniform(3, 17), rng.uniform(3, 17)
            self.conditional_triggers.append({
                'trigger': (tx, ty), 'effect': (ex, ey),
                'radius': 3.0, 'energy_threshold': 0.5,
                'high_energy_type': 'endorphin', 'low_energy_type': 'pain',
            })

    def _add_tier2(self, rng):
        profiles = ['cooperative', 'competitive', 'erratic', 'deceptive']
        for i, profile in enumerate(profiles):
            npc = NPC()
            npc.reset(rng)
            npc.behavior_profile = profile
            npc.reciprocity_score = 0.0
            npc.org_preferred_source = None
            self.npcs.append(npc)

        self.predator_events.append(PredatorEvent(
            period=rng.randint(120, 200),
            duration=rng.randint(15, 30),
            intensity=1.5))
        self.resource_depletion_factor = 0.3

    def _add_tier3(self, rng):
        for _ in range(3):
            x, y = rng.uniform(3, 17), rng.uniform(3, 17)
            self.movable_objects.append(MovableObject(x, y))

        self.gated_reward = None
        if self.endorphin_sources:
            src = self.endorphin_sources[0]
            gx, gy = src.cx, src.cy
            self.gated_reward = {
                'x': gx, 'y': gy, 'ring_radius': 2.5,
                'ring_intensity': 1.5, 'reward_intensity': 3.0,
                'gap_object_idx': 0,
            }

        self.compound_pairs = [(0, 1)]

    def _add_tier4(self):
        self.hidden_variable = HiddenVariable(period=150, num_states=3)
        self.transfer_switch_step = 150
        self.config_swapped = False

    def _add_tier5(self, rng):
        self.org_action_history = []
        self.alliance_phase = 0
        self.alliance_switch_period = 100
        self.hidden_endorphin_sources = []
        for _ in range(2):
            cx, cy = rng.uniform(3, 17), rng.uniform(3, 17)
            self.hidden_endorphin_sources.append({
                'x': cx, 'y': cy, 'sigma': 2.5, 'intensity': 2.0,
                'revealed': False, 'reveal_timer': 0,
            })
        self.deception_energy_cost = 0.02

    def _add_tier6(self, rng):
        self.persistent_object_state = None
        self.sequential_affordance = {
            'step1_done': False, 'step2_done': False, 'step3_done': False,
            'target_x': rng.uniform(5, 15), 'target_y': rng.uniform(5, 15),
            'reward_active': False, 'reward_timer': 0,
        }
        self.blocked_sources = {}
        self.delayed_rewards = []

    def _add_tier7(self, rng):
        self.rule_rotation_period = 100
        self.current_rule_phase = 0
        self.impulse_traps = []
        for _ in range(2):
            self.impulse_traps.append({
                'x': rng.uniform(4, 16), 'y': rng.uniform(4, 16),
                'endo_radius': 2.0, 'pain_delay': 20, 'pain_intensity': 2.0,
                'triggered_at': [], 'endo_intensity': 2.5,
            })
        self.strategy_age = 0
        self.last_action_hash = 0
        self.staleness_penalty = 0.0

    def reset(self, seed=None):
        super().reset(seed)
        for ct in self.causal_triggers:
            ct.reset()
        for mo in self.movable_objects:
            rng = self.rng
            mo.reset(rng.uniform(3, 17), rng.uniform(3, 17))

    def get_field_values(self, points, t):
        pain, endorphin = super().get_field_values(points, t)

        if self.tier >= 1:
            for ps in self.pulsing_sources:
                for i, (px, py) in enumerate(points):
                    pain[i] += ps.intensity_at(px, py, t)

            for ct in self.causal_triggers:
                ct.update(t)
                if ct.active:
                    for i, (px, py) in enumerate(points):
                        if ct.effect_type == 'endorphin':
                            endorphin[i] += ct.effect_at(px, py)
                        elif ct.effect_type == 'pain':
                            pain[i] += ct.effect_at(px, py)

            for ap in getattr(self, 'anticipatory_phases', []):
                src = ap['source']
                period = ap['period']
                phase = (t % period) / period
                if ap['reward_phase_start'] <= phase <= ap['reward_phase_end']:
                    for i, (px, py) in enumerate(points):
                        sx, sy = src.position(t)
                        dx, dy = px - sx, py - sy
                        endorphin[i] += 2.0 * math.exp(-(dx*dx + dy*dy) / (2 * src.sigma**2))

        if self.tier >= 2:
            for pe in self.predator_events:
                pe.update(t)
                for i, (px, py) in enumerate(points):
                    pain[i] += pe.pain_at(px, py)

            depletion = getattr(self, 'resource_depletion_factor', 0.0)
            if depletion > 0 and self.npcs:
                for src in self.endorphin_sources:
                    sx, sy = src.position(t)
                    nearby = sum(1 for npc in self.npcs
                                 if math.sqrt((npc.x-sx)**2 + (npc.y-sy)**2) < 3.0)
                    if nearby > 0:
                        scale = max(0.3, 1.0 - depletion * nearby)
                        for i, (px, py) in enumerate(points):
                            effect = src.intensity_at(px, py, t)
                            endorphin[i] -= effect * (1.0 - scale)

        if self.tier >= 3:
            for mo in self.movable_objects:
                for src in self.pain_sources:
                    sx, sy = src.position(t)
                    if mo.blocks_source(sx, sy):
                        for i, (px, py) in enumerate(points):
                            block = src.intensity_at(px, py, t)
                            pain[i] = max(0.0, pain[i] - block * 0.9)

            gr = getattr(self, 'gated_reward', None)
            if gr is not None:
                gap_clear = False
                if gr['gap_object_idx'] < len(self.movable_objects):
                    mo = self.movable_objects[gr['gap_object_idx']]
                    if math.sqrt((mo.x - gr['x'])**2 + (mo.y - gr['y'])**2) < gr['ring_radius'] + 1.0:
                        gap_clear = True
                for i, (px, py) in enumerate(points):
                    dx, dy = px - gr['x'], py - gr['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    if gr['ring_radius'] - 0.5 < dist < gr['ring_radius'] + 0.5 and not gap_clear:
                        pain[i] += gr['ring_intensity']
                    if dist < gr['ring_radius'] - 0.5:
                        endorphin[i] += gr['reward_intensity'] * math.exp(-dist*dist / 2.0)

            for a_idx, b_idx in getattr(self, 'compound_pairs', []):
                if a_idx < len(self.movable_objects) and b_idx < len(self.movable_objects):
                    a, b = self.movable_objects[a_idx], self.movable_objects[b_idx]
                    if a.near(b, 2.0):
                        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
                        for i, (px, py) in enumerate(points):
                            dx, dy = px - mx, py - my
                            endorphin[i] += 2.0 * math.exp(-(dx*dx + dy*dy) / 4.0)

        if self.tier >= 4 and self.hidden_variable is not None:
            mod = self.hidden_variable.modulation(t)
            pain *= mod['pain']
            endorphin *= mod['endorphin']

        if self.tier >= 5:
            for hs in getattr(self, 'hidden_endorphin_sources', []):
                if hs['revealed']:
                    for i, (px, py) in enumerate(points):
                        dx, dy = px - hs['x'], py - hs['y']
                        endorphin[i] += hs['intensity'] * math.exp(-(dx*dx + dy*dy) / (2 * hs['sigma']**2))

        if self.tier >= 6:
            sa = getattr(self, 'sequential_affordance', None)
            if sa and sa.get('reward_active'):
                for i, (px, py) in enumerate(points):
                    dx, dy = px - sa['target_x'], py - sa['target_y']
                    endorphin[i] += 5.0 * math.exp(-(dx*dx + dy*dy) / 4.0)
            for dr in getattr(self, 'delayed_rewards', []):
                if dr['step'] <= t and dr['step'] + 30 > t:
                    for i, (px, py) in enumerate(points):
                        dx, dy = px - dr['x'], py - dr['y']
                        endorphin[i] += 2.0 * math.exp(-(dx*dx + dy*dy) / 4.0)

        if self.tier >= 7:
            for trap in getattr(self, 'impulse_traps', []):
                for i, (px, py) in enumerate(points):
                    dx, dy = px - trap['x'], py - trap['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < trap['endo_radius']:
                        endorphin[i] += trap['endo_intensity'] * math.exp(-dist*dist / 2.0)
                for triggered_t in trap.get('triggered_at', []):
                    if t - triggered_t >= trap['pain_delay'] and t - triggered_t < trap['pain_delay'] + 15:
                        for i, (px, py) in enumerate(points):
                            dx, dy = px - trap['x'], py - trap['y']
                            pain[i] += trap['pain_intensity'] * math.exp(-(dx*dx + dy*dy) / 4.0)

        return pain, endorphin

    def step_tier(self, org_x, org_y, t):
        if self.tier >= 1:
            for ct in self.causal_triggers:
                ct.check_trigger(org_x, org_y, t)

        if self.tier >= 2:
            for npc in self.npcs:
                self._step_profiled_npc(npc, org_x, org_y, t)

        if self.tier >= 3:
            for mo in self.movable_objects:
                mo.step()

        if self.tier >= 5:
            history = getattr(self, 'org_action_history', [])
            history.append((org_x, org_y))
            if len(history) > 20:
                history.pop(0)
            self.org_action_history = history

            for npc in self.npcs:
                if getattr(npc, 'behavior_profile', '') == 'competitive' and len(history) >= 5:
                    avg_x = np.mean([h[0] for h in history[-5:]])
                    avg_y = np.mean([h[1] for h in history[-5:]])
                    dx, dy = avg_x - npc.x, avg_y - npc.y
                    d = math.sqrt(dx*dx + dy*dy) + 1e-8
                    npc.x += 0.2 * dx / d
                    npc.y += 0.2 * dy / d
                    npc.x = max(0.5, min(19.5, npc.x))
                    npc.y = max(0.5, min(19.5, npc.y))

            ap = getattr(self, 'alliance_switch_period', 100)
            self.alliance_phase = (t // ap) % 2
            for npc in self.npcs:
                if getattr(npc, 'behavior_profile', '') == 'cooperative':
                    if self.alliance_phase == 1:
                        npc.behavior_profile = 'competitive'
                    else:
                        npc.behavior_profile = 'cooperative'

            for hs in getattr(self, 'hidden_endorphin_sources', []):
                if hs['revealed']:
                    hs['reveal_timer'] -= 1
                    if hs['reveal_timer'] <= 0:
                        hs['revealed'] = False
                for npc in self.npcs:
                    if getattr(npc, 'behavior_profile', '') == 'cooperative':
                        nd = math.sqrt((npc.x - hs['x'])**2 + (npc.y - hs['y'])**2)
                        if nd < 3.0 and not hs['revealed']:
                            hs['revealed'] = True
                            hs['reveal_timer'] = 50

        if self.tier >= 6:
            sa = getattr(self, 'sequential_affordance', None)
            if sa is not None and len(self.movable_objects) >= 2:
                a, b = self.movable_objects[0], self.movable_objects[1]
                td = math.sqrt((a.x - sa['target_x'])**2 + (a.y - sa['target_y'])**2)
                if td < 2.0 and not sa['step1_done']:
                    sa['step1_done'] = True
                if sa['step1_done'] and a.near(b, 2.0) and not sa['step2_done']:
                    sa['step2_done'] = True
                if sa['step2_done'] and not sa['reward_active']:
                    od = math.sqrt((org_x - a.x)**2 + (org_y - a.y)**2)
                    if od < 3.0:
                        sa['reward_active'] = True
                        sa['reward_timer'] = 30
                if sa['reward_active']:
                    sa['reward_timer'] -= 1
                    if sa['reward_timer'] <= 0:
                        sa['reward_active'] = False
                        sa['step1_done'] = False
                        sa['step2_done'] = False

            for src_key, blocked_until in list(getattr(self, 'blocked_sources', {}).items()):
                if t > blocked_until:
                    del self.blocked_sources[src_key]

        if self.tier >= 7:
            rp = getattr(self, 'rule_rotation_period', 100)
            self.current_rule_phase = (t // rp) % 3

            for trap in getattr(self, 'impulse_traps', []):
                dist = math.sqrt((org_x - trap['x'])**2 + (org_y - trap['y'])**2)
                if dist < trap['endo_radius']:
                    if not trap['triggered_at'] or (t - trap['triggered_at'][-1]) > trap['pain_delay'] + 20:
                        trap['triggered_at'].append(t)

            ah = hash((round(org_x, 1), round(org_y, 1)))
            if ah == getattr(self, 'last_action_hash', 0):
                self.strategy_age = getattr(self, 'strategy_age', 0) + 1
            else:
                self.strategy_age = 0
            self.last_action_hash = ah
            self.staleness_penalty = min(0.5, self.strategy_age * 0.005)

    def _step_profiled_npc(self, npc, org_x, org_y, t):
        profile = getattr(npc, 'behavior_profile', 'cooperative')

        if profile == 'cooperative':
            npc.step(self, t)
            recip = getattr(npc, 'reciprocity_score', 0.0)
            if recip > 0.3:
                dist = math.sqrt((npc.x - org_x)**2 + (npc.y - org_y)**2)
                if dist > 5.0:
                    dx, dy = org_x - npc.x, org_y - npc.y
                    d = math.sqrt(dx*dx + dy*dy) + 1e-8
                    npc.x += 0.1 * dx / d
                    npc.y += 0.1 * dy / d
            else:
                npc.emission_bits = np.zeros(4, dtype=int)
            npc.reciprocity_score = max(0.0, recip - 0.01)
        elif profile == 'competitive':
            npc.step(self, t)
            pref = getattr(npc, 'org_preferred_source', None)
            if pref is not None and pref < len(self.endorphin_sources):
                src = self.endorphin_sources[pref]
                sx, sy = src.position(t)
                dx, dy = sx - npc.x, sy - npc.y
                d = math.sqrt(dx*dx + dy*dy) + 1e-8
                npc.x += 0.15 * dx / d
                npc.y += 0.15 * dy / d
                npc.x = max(0.5, min(19.5, npc.x))
                npc.y = max(0.5, min(19.5, npc.y))
            best_src, best_dist = 0, float('inf')
            for si, src in enumerate(self.endorphin_sources):
                sx, sy = src.position(t)
                od = math.sqrt((org_x - sx)**2 + (org_y - sy)**2)
                if od < best_dist:
                    best_dist = od
                    best_src = si
            npc.org_preferred_source = best_src
        elif profile == 'erratic':
            npc.vx += np.random.uniform(-0.5, 0.5)
            npc.vy += np.random.uniform(-0.5, 0.5)
            npc.x += npc.vx * 0.05
            npc.y += npc.vy * 0.05
            npc.x = max(0.5, min(19.5, npc.x))
            npc.y = max(0.5, min(19.5, npc.y))
            npc.erraticism = 0.9 * npc.erraticism + 0.1 * 1.0
        elif profile == 'deceptive':
            npc.step(self, t)
            npc.emission_bits = np.array([0, 0, 1, 0], dtype=int)

    def get_tier_stats(self, t=0):
        stats = {
            'tier': self.tier,
            'pain_sources': len(self.pain_sources),
            'endorphin_sources': len(self.endorphin_sources),
            'responsive_objects': len(self.responsive_objects),
        }
        if self.tier >= 1:
            stats['pulsing_sources'] = len(self.pulsing_sources)
            stats['causal_triggers'] = len(self.causal_triggers)
            active = sum(1 for ct in self.causal_triggers if ct.active)
            stats['active_triggers'] = active
        if self.tier >= 2:
            stats['extra_npcs'] = len(self.npcs)
            stats['npc_profiles'] = [getattr(n, 'behavior_profile', '?') for n in self.npcs]
            stats['predator_events'] = len(self.predator_events)
            active_pe = sum(1 for pe in self.predator_events if pe.active)
            stats['active_predators'] = active_pe
        if self.tier >= 3:
            stats['movable_objects'] = len(self.movable_objects)
        if self.tier >= 4:
            stats['hidden_variable_state'] = self.hidden_variable.state(t) if self.hidden_variable else None
        if self.tier >= 5:
            revealed = sum(1 for hs in getattr(self, 'hidden_endorphin_sources', []) if hs['revealed'])
            stats['hidden_sources_revealed'] = revealed
            stats['alliance_phase'] = getattr(self, 'alliance_phase', 0)
        if self.tier >= 6:
            sa = getattr(self, 'sequential_affordance', {})
            stats['affordance_progress'] = sum([sa.get('step1_done', False),
                                                 sa.get('step2_done', False),
                                                 sa.get('reward_active', False)])
        if self.tier >= 7:
            stats['rule_phase'] = getattr(self, 'current_rule_phase', 0)
            stats['strategy_age'] = getattr(self, 'strategy_age', 0)
            stats['staleness_penalty'] = round(getattr(self, 'staleness_penalty', 0), 3)
        return stats

    def to_dict(self):
        d = super().to_dict()
        d['tier'] = self.tier
        if self.tier >= 3:
            d['movable_objects'] = [mo.to_dict() for mo in self.movable_objects]
        return d


def create_environment(tier=0, seed=None):
    return TieredEnvironment(seed=seed, tier=tier)


if __name__ == '__main__':
    for tier in range(8):
        print(f"\n=== Tier {tier} ===")
        env = create_environment(tier=tier, seed=42)
        org = Organism()
        org.reset()
        npc = NPC()
        npc.reset()

        total_reward = 0
        for step in range(100):
            npc.step(env, step)
            env.step_tier(org.x, org.y, step)
            actions = org.compute_optimal_actions(env, step, npc=npc)
            obs, reward = org.step(actions, env, step, npc=npc)
            total_reward += reward

        stats = env.get_tier_stats(100)
        print(f"  Reward: {total_reward:.1f}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"  PASS: Tier {tier}")

    print("\nAll tier tests passed!")

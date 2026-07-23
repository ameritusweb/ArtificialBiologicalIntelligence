import numpy as np
from collections import deque
import math


class FieldSource:
    def __init__(self, cx, cy, ax, ay, omega_x, omega_y, phi_x, phi_y, sigma, intensity):
        self.cx = cx
        self.cy = cy
        self.ax = ax
        self.ay = ay
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.phi_x = phi_x
        self.phi_y = phi_y
        self.sigma = sigma
        self.intensity = intensity

    def position(self, t):
        x = np.clip(self.cx + self.ax * np.sin(self.omega_x * t + self.phi_x), 1.0, 19.0)
        y = np.clip(self.cy + self.ay * np.cos(self.omega_y * t + self.phi_y), 1.0, 19.0)
        return x, y

    def intensity_at(self, px, py, t, pz=0.0):
        sx, sy = self.position(t)
        dx, dy = px - sx, py - sy
        dz = pz - getattr(self, 'cz', 0.0)
        return self.intensity * np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * self.sigma ** 2))

    def gradient_at(self, px, py, t):
        sx, sy = self.position(t)
        dx, dy = px - sx, py - sy
        val = self.intensity * np.exp(-(dx * dx + dy * dy) / (2.0 * self.sigma ** 2))
        factor = -val / (self.sigma ** 2)
        return factor * dx, factor * dy

    def to_dict(self):
        return {
            'cx': round(self.cx, 3), 'cy': round(self.cy, 3),
            'ax': round(self.ax, 3), 'ay': round(self.ay, 3),
            'omega_x': round(self.omega_x, 4), 'omega_y': round(self.omega_y, 4),
            'phi_x': round(self.phi_x, 3), 'phi_y': round(self.phi_y, 3),
            'sigma': round(self.sigma, 3), 'intensity': round(self.intensity, 3),
        }


class ResponsiveObject:
    def __init__(self, cx, cy, ax, ay, omega_x, omega_y, phi_x, phi_y,
                 trigger_signal, response_type, trigger_range=5.0, response_duration=10):
        self.cx, self.cy = cx, cy
        self.ax, self.ay = ax, ay
        self.omega_x, self.omega_y = omega_x, omega_y
        self.phi_x, self.phi_y = phi_x, phi_y
        self.trigger_signal = tuple(trigger_signal)
        self.response_type = response_type
        self.trigger_range = trigger_range
        self.response_duration = response_duration
        self.responding = False
        self.response_timer = 0
        self.effect_sigma = 2.0

    def position(self, t):
        x = np.clip(self.cx + self.ax * np.sin(self.omega_x * t + self.phi_x), 1.0, 19.0)
        y = np.clip(self.cy + self.ay * np.cos(self.omega_y * t + self.phi_y), 1.0, 19.0)
        return x, y

    def distance_to(self, px, py, t):
        ox, oy = self.position(t)
        return math.sqrt((px - ox) ** 2 + (py - oy) ** 2)

    def check_trigger(self, org_x, org_y, emission_bits, t):
        dist = self.distance_to(org_x, org_y, t)
        if dist > self.trigger_range:
            return False
        return tuple(int(b) for b in emission_bits) == self.trigger_signal

    def update(self, org_x, org_y, emission_bits, t):
        if not self.responding and self.check_trigger(org_x, org_y, emission_bits, t):
            self.responding = True
            self.response_timer = self.response_duration
        if self.responding:
            self.response_timer -= 1
            if self.response_type == 'approach':
                ox, oy = self.position(t)
                dx, dy = org_x - ox, org_y - oy
                dist = math.sqrt(dx * dx + dy * dy) + 1e-8
                self.cx += 0.3 * dx / dist
                self.cy += 0.3 * dy / dist
            if self.response_timer <= 0:
                self.responding = False

    def effect_at(self, px, py, t):
        if not self.responding:
            return 0.0
        ox, oy = self.position(t)
        dx, dy = px - ox, py - oy
        return np.exp(-(dx * dx + dy * dy) / (2.0 * self.effect_sigma ** 2))

    def reset(self):
        self.responding = False
        self.response_timer = 0

    def to_dict(self, t=0):
        ox, oy = self.position(t)
        return {
            'x': round(ox, 3), 'y': round(oy, 3),
            'type': self.response_type,
            'trigger': list(self.trigger_signal),
            'responding': self.responding,
            'timer': self.response_timer,
            'range': self.trigger_range,
        }


class NPC:
    NUM_LIMBS = 6
    BASE_ANGLES = [i * math.pi / 3 for i in range(6)]
    THRUST_FORCE = 0.5
    DRAG = 0.4
    ANG_DRAG = 0.6
    MASS = 1.0
    MOI = 0.5
    DT = 0.05
    WALL_DIST = 1.5
    WALL_FORCE = 2.0
    MAX_SPEED = 3.0
    MAX_ANG_SPEED = 2.0
    WIDTH = 20.0
    HEIGHT = 20.0
    SIGNAL_RANGE = 8.0
    REPEL_SIGNAL = (1, 1, 0, 0)
    CALM_SIGNAL = (1, 0, 1, 0)
    SIGNAL_REPEL_DURATION = 15
    SIGNAL_REPEL_STRENGTH = 2.0
    SIGNAL_CALM_DURATION = 10
    SIGNAL_CALM_RATE = 0.3

    def __init__(self):
        self.x = 5.0
        self.y = 5.0
        self.heading = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.prev_speed = 0.0
        self.erraticism = 0.0
        self.signal_active = False
        self.signal_type = None
        self.signal_timer = 0
        self.signal_source_x = 0.0
        self.signal_source_y = 0.0
        self.emission_bits = np.zeros(4, dtype=int)

    def reset(self, rng=None):
        if rng is not None:
            while True:
                self.x = rng.uniform(2.0, 18.0)
                self.y = rng.uniform(2.0, 18.0)
                if math.sqrt((self.x - 10.0)**2 + (self.y - 10.0)**2) > 5.0:
                    break
            self.heading = rng.uniform(0, 2 * math.pi)
        else:
            self.x, self.y = 5.0, 5.0
            self.heading = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.prev_speed = 0.0
        self.erraticism = 0.0
        self.signal_active = False
        self.signal_type = None
        self.signal_timer = 0
        self.signal_source_x = 0.0
        self.signal_source_y = 0.0
        self.emission_bits = np.zeros(4, dtype=int)

    def receive_signal(self, emission_bits, org_x, org_y):
        dist = math.sqrt((self.x - org_x)**2 + (self.y - org_y)**2)
        if dist > self.SIGNAL_RANGE:
            return
        signal = tuple(int(b) for b in emission_bits)
        if signal == self.REPEL_SIGNAL:
            self.signal_active = True
            self.signal_type = 'repel'
            self.signal_timer = self.SIGNAL_REPEL_DURATION
            self.signal_source_x = org_x
            self.signal_source_y = org_y
        elif signal == self.CALM_SIGNAL:
            self.signal_active = True
            self.signal_type = 'calm'
            self.signal_timer = self.SIGNAL_CALM_DURATION

    def step(self, environment, time_step):
        gx, gy = environment.get_combined_gradient(self.x, self.y, time_step)

        if self.signal_active:
            if self.signal_type == 'repel':
                dx = self.x - self.signal_source_x
                dy = self.y - self.signal_source_y
                d = math.sqrt(dx * dx + dy * dy) + 1e-8
                gx += self.SIGNAL_REPEL_STRENGTH * dx / d
                gy += self.SIGNAL_REPEL_STRENGTH * dy / d
            elif self.signal_type == 'calm':
                self.erraticism *= (1.0 - self.SIGNAL_CALM_RATE)
            self.signal_timer -= 1
            if self.signal_timer <= 0:
                self.signal_active = False
                self.signal_type = None

        total_fx, total_fy, total_torque = 0.0, 0.0, 0.0
        grad_mag = math.sqrt(gx * gx + gy * gy)
        desired_dir = math.atan2(gy, gx) if grad_mag > 1e-6 else self.heading

        for i in range(self.NUM_LIMBS):
            world_angle = self.heading + self.BASE_ANGLES[i]
            dot = math.cos(desired_dir - (world_angle + math.pi))
            if dot > -0.3 and grad_mag > 1e-6:
                total_fx += -self.THRUST_FORCE * math.cos(world_angle)
                total_fy += -self.THRUST_FORCE * math.sin(world_angle)

        angle_diff = desired_dir - self.heading
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        if abs(angle_diff) > 0.15:
            total_torque += 0.4 * (1.0 if angle_diff > 0 else -1.0)

        total_fx -= self.DRAG * self.vx
        total_fy -= self.DRAG * self.vy
        total_torque -= self.ANG_DRAG * self.omega

        for wall_pos, axis, sign in [
            (0, 'x', 1), (self.WIDTH, 'x', -1),
            (0, 'y', 1), (self.HEIGHT, 'y', -1),
        ]:
            coord = self.x if axis == 'x' else self.y
            dist = abs(coord - wall_pos)
            if dist < self.WALL_DIST:
                force = self.WALL_FORCE * (1 - dist / self.WALL_DIST) ** 2
                if axis == 'x':
                    total_fx += force * sign
                else:
                    total_fy += force * sign

        self.vx += (total_fx / self.MASS) * self.DT
        self.vy += (total_fy / self.MASS) * self.DT
        self.omega += (total_torque / self.MOI) * self.DT

        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > self.MAX_SPEED:
            self.vx = self.vx / speed * self.MAX_SPEED
            self.vy = self.vy / speed * self.MAX_SPEED
        self.omega = max(-self.MAX_ANG_SPEED, min(self.MAX_ANG_SPEED, self.omega))

        self.x += self.vx * self.DT
        self.y += self.vy * self.DT
        self.heading += self.omega * self.DT
        self.x = max(0.5, min(19.5, self.x))
        self.y = max(0.5, min(19.5, self.y))

        cur_speed = self.speed_val()
        self.acceleration = (cur_speed - self.prev_speed) / self.DT
        self.prev_speed = cur_speed
        self.erraticism = 0.9 * self.erraticism + 0.1 * abs(self.omega)

        self.emission_bits = np.zeros(4, dtype=int)
        max_pain, max_endo = 0.0, 0.0
        for src in environment.pain_sources:
            max_pain = max(max_pain, src.intensity_at(self.x, self.y, time_step))
        for src in environment.endorphin_sources:
            max_endo = max(max_endo, src.intensity_at(self.x, self.y, time_step))
        if max_pain > 0.3:
            self.emission_bits = np.array([1, 1, 0, 0], dtype=int)
        elif max_endo > 0.3:
            self.emission_bits = np.array([0, 0, 1, 0], dtype=int)

    def speed_val(self):
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    def distance_to(self, px, py):
        return math.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)

    def to_dict(self):
        return {
            'x': round(self.x, 3), 'y': round(self.y, 3),
            'heading': round(self.heading, 3),
            'speed': round(self.speed_val(), 3),
            'accel': round(getattr(self, 'acceleration', 0.0), 3),
            'omega': round(self.omega, 3),
            'erratic': round(self.erraticism, 3),
            'signal': self.signal_type if self.signal_active else None,
            'signal_timer': self.signal_timer if self.signal_active else 0,
            'emission': [int(b) for b in self.emission_bits],
        }


class Environment:
    WIDTH = 20.0
    HEIGHT = 20.0
    TEMP_BASE = 0.5
    TEMP_COMFORTABLE_LOW = 0.3
    TEMP_COMFORTABLE_HIGH = 0.7
    PRESSURE_SENSE_DIST = 2.0

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.pain_sources = []
        self.endorphin_sources = []
        self.heat_sources = []
        self.cold_sources = []
        self.chemical_sources = []
        self.responsive_objects = []
        self._generate_sources()

    def _make_source(self, sigma, intensity):
        cx = self.rng.uniform(3, 17)
        cy = self.rng.uniform(3, 17)
        return FieldSource(
            cx=cx, cy=cy,
            ax=self.rng.uniform(2.0, 4.0), ay=self.rng.uniform(2.0, 4.0),
            omega_x=self.rng.uniform(0.01, 0.04), omega_y=self.rng.uniform(0.01, 0.04),
            phi_x=self.rng.uniform(0, 2 * np.pi), phi_y=self.rng.uniform(0, 2 * np.pi),
            sigma=sigma, intensity=intensity,
        )

    def _generate_sources(self):
        centers = [(5, 5), (15, 5), (10, 15), (5, 15), (15, 15)]
        self.rng.shuffle(centers)

        self.pain_sources = []
        for i in range(3):
            cx, cy = centers[i]
            cx += self.rng.uniform(-2, 2)
            cy += self.rng.uniform(-2, 2)
            self.pain_sources.append(FieldSource(
                cx=np.clip(cx, 2, 18), cy=np.clip(cy, 2, 18),
                ax=self.rng.uniform(2.0, 4.0), ay=self.rng.uniform(2.0, 4.0),
                omega_x=self.rng.uniform(0.01, 0.04), omega_y=self.rng.uniform(0.01, 0.04),
                phi_x=self.rng.uniform(0, 2 * np.pi), phi_y=self.rng.uniform(0, 2 * np.pi),
                sigma=2.5, intensity=1.0,
            ))

        self.endorphin_sources = []
        for i in range(3, 5):
            cx, cy = centers[i]
            cx += self.rng.uniform(-2, 2)
            cy += self.rng.uniform(-2, 2)
            self.endorphin_sources.append(FieldSource(
                cx=np.clip(cx, 2, 18), cy=np.clip(cy, 2, 18),
                ax=self.rng.uniform(2.0, 4.0), ay=self.rng.uniform(2.0, 4.0),
                omega_x=self.rng.uniform(0.01, 0.04), omega_y=self.rng.uniform(0.01, 0.04),
                phi_x=self.rng.uniform(0, 2 * np.pi), phi_y=self.rng.uniform(0, 2 * np.pi),
                sigma=3.0, intensity=1.5,
            ))

        self.heat_sources = [self._make_source(3.0, 0.5) for _ in range(2)]
        self.cold_sources = [self._make_source(3.0, 0.5)]
        self.chemical_sources = [self._make_source(3.0, 1.0) for _ in range(2)]

        self.responsive_objects = []
        obj_configs = [
            ((0, 0, 1, 0), 'endorphin'),
            ((0, 1, 0, 1), 'approach'),
            ((1, 0, 0, 1), 'cool'),
        ]
        for trigger, rtype in obj_configs:
            cx, cy = self.rng.uniform(3, 17), self.rng.uniform(3, 17)
            self.responsive_objects.append(ResponsiveObject(
                cx=cx, cy=cy,
                ax=self.rng.uniform(1.0, 3.0), ay=self.rng.uniform(1.0, 3.0),
                omega_x=self.rng.uniform(0.005, 0.02), omega_y=self.rng.uniform(0.005, 0.02),
                phi_x=self.rng.uniform(0, 2 * np.pi), phi_y=self.rng.uniform(0, 2 * np.pi),
                trigger_signal=trigger, response_type=rtype,
            ))

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._generate_sources()

    def get_field_values(self, points, t):
        pain = np.zeros(len(points))
        endorphin = np.zeros(len(points))
        for src in self.pain_sources:
            for i, (px, py) in enumerate(points):
                pain[i] += src.intensity_at(px, py, t)
        for src in self.endorphin_sources:
            for i, (px, py) in enumerate(points):
                endorphin[i] += src.intensity_at(px, py, t)
        return pain, endorphin

    def get_temperature_values(self, points, t):
        temps = np.full(len(points), self.TEMP_BASE)
        for src in self.heat_sources:
            for i, (px, py) in enumerate(points):
                temps[i] += src.intensity_at(px, py, t)
        for src in self.cold_sources:
            for i, (px, py) in enumerate(points):
                temps[i] -= src.intensity_at(px, py, t)
        return np.clip(temps, 0.0, 1.0)

    def get_chemical_values(self, points, t):
        chem = np.zeros(len(points))
        for src in self.chemical_sources:
            for i, (px, py) in enumerate(points):
                chem[i] += src.intensity_at(px, py, t)
        return chem

    def get_pressure_values(self, points):
        pressure = np.zeros(len(points))
        for i, (px, py) in enumerate(points):
            dist = min(px, self.WIDTH - px, py, self.HEIGHT - py)
            pressure[i] = max(0.0, 1.0 - dist / self.PRESSURE_SENSE_DIST)
        return pressure

    def get_combined_gradient(self, px, py, t):
        """Returns desired movement direction: away from pain, toward endorphin,
        temperature homeostasis, toward chemicals, away from walls."""
        gx, gy = 0.0, 0.0

        for src in self.pain_sources:
            dx, dy = src.gradient_at(px, py, t)
            gx -= dx
            gy -= dy

        for src in self.endorphin_sources:
            dx, dy = src.gradient_at(px, py, t)
            gx += dx
            gy += dy

        for src in self.chemical_sources:
            dx, dy = src.gradient_at(px, py, t)
            gx += 0.7 * dx
            gy += 0.7 * dy

        temp = self.TEMP_BASE
        for src in self.heat_sources:
            temp += src.intensity_at(px, py, t)
        for src in self.cold_sources:
            temp -= src.intensity_at(px, py, t)
        temp = max(0.0, min(1.0, temp))

        if temp > self.TEMP_COMFORTABLE_HIGH:
            for src in self.heat_sources:
                dx, dy = src.gradient_at(px, py, t)
                gx -= 0.8 * dx
                gy -= 0.8 * dy
        elif temp < self.TEMP_COMFORTABLE_LOW:
            for src in self.heat_sources:
                dx, dy = src.gradient_at(px, py, t)
                gx += 0.8 * dx
                gy += 0.8 * dy

        ws = 1.5
        if px < self.PRESSURE_SENSE_DIST:
            gx += ws * (1.0 - px / self.PRESSURE_SENSE_DIST)
        if px > self.WIDTH - self.PRESSURE_SENSE_DIST:
            gx -= ws * (1.0 - (self.WIDTH - px) / self.PRESSURE_SENSE_DIST)
        if py < self.PRESSURE_SENSE_DIST:
            gy += ws * (1.0 - py / self.PRESSURE_SENSE_DIST)
        if py > self.HEIGHT - self.PRESSURE_SENSE_DIST:
            gy -= ws * (1.0 - (self.HEIGHT - py) / self.PRESSURE_SENSE_DIST)

        return gx, gy

    def get_source_positions(self, t):
        return (
            [list(s.position(t)) for s in self.pain_sources],
            [list(s.position(t)) for s in self.endorphin_sources],
            [list(s.position(t)) for s in self.heat_sources],
            [list(s.position(t)) for s in self.cold_sources],
            [list(s.position(t)) for s in self.chemical_sources],
        )

    def to_dict(self):
        return {
            'width': self.WIDTH, 'height': self.HEIGHT,
            'pain_sources': [s.to_dict() for s in self.pain_sources],
            'endorphin_sources': [s.to_dict() for s in self.endorphin_sources],
            'heat_sources': [s.to_dict() for s in self.heat_sources],
            'cold_sources': [s.to_dict() for s in self.cold_sources],
            'chemical_sources': [s.to_dict() for s in self.chemical_sources],
            'pain_sigma': self.pain_sources[0].sigma if self.pain_sources else 2.5,
            'endorphin_sigma': self.endorphin_sources[0].sigma if self.endorphin_sources else 3.0,
            'heat_sigma': self.heat_sources[0].sigma if self.heat_sources else 3.0,
            'cold_sigma': self.cold_sources[0].sigma if self.cold_sources else 3.0,
            'chemical_sigma': self.chemical_sources[0].sigma if self.chemical_sources else 3.0,
            'responsive_objects': [o.to_dict() for o in self.responsive_objects],
        }


class Organism:
    THRUST_FORCE = 0.5
    FLEX_DELTA = math.pi / 12
    FLEX_SPRING = 0.08
    FLEX_MAX_DEV = math.pi / 6
    LIMB_LEN_RETRACTED = 0.8
    LIMB_LEN_EXTENDED = 1.5
    DRAG = 0.4
    ANG_DRAG = 0.6
    MASS = 1.0
    MOI = 0.5
    DT = 0.05
    WALL_DIST = 1.5
    WALL_FORCE = 2.0
    MAX_SPEED = 3.0
    MAX_ANG_SPEED = 2.0
    BODY_RADIUS = 0.4
    WIDTH = 20.0
    HEIGHT = 20.0

    SEQ_LEN = 32
    FATIGUE_RATE = 0.01
    RECOVERY_RATE = 0.005
    ENERGY_COST = 0.002
    ENERGY_RECOVERY = 0.003
    MEMORY_GRID_SIZE = 5
    MEMORY_CELL_SIZE = 4.0
    MEMORY_DECAY = 0.95
    MEMORY_PHANTOM_THRESHOLD = 0.3
    MEMORY_PHANTOM_STRENGTH = 0.5
    SLOW_PATHWAY_COST = 0.008
    TEMPORAL_AVERSION_ALPHA = 0.15
    TEMPORAL_AVERSION_DECAY = 0.97
    SENSITIZATION_RATE = 0.005
    HABITUATION_RATE = 0.003
    GAIN_RECOVERY_RATE = 0.002
    GAIN_MIN = 0.3
    GAIN_MAX = 2.0
    NUM_DISTANCE_RAYS = 8
    SENSE_DISTANCE = 3.0
    RAY_ANGLES = [i * math.pi / 4 for i in range(8)]
    DIST_GRADIENT_WEIGHT = 0.4
    EMISSION_BITS = 4
    EMISSION_COST = 0.001
    OBJ_SENSE_RANGE = 10.0
    EMPATHY_WEIGHT = 0.4
    EMPATHY_RANGE = 6.0
    NPC_SENSE_RANGE = 10.0
    NPC_COLLISION_RADIUS = 1.5
    NPC_COLLISION_PAIN = 0.8
    NPC_COMPETITION_RANGE = 3.0

    SEGMENT_DIST = 1.2
    JOINT_FLEX_DELTA = math.pi / 10
    JOINT_SPRING = 0.1
    JOINT_MAX_DEV = math.pi / 4
    DEPTH = 10.0

    def __init__(self, num_limbs=6, num_segments=1, dims=2):
        self.limbs_per_segment = num_limbs
        self.num_segments = num_segments
        self.dims = dims
        self.NUM_LIMBS = num_segments * num_limbs
        self.SEGMENT_BASE_ANGLES = [i * 2 * math.pi / num_limbs for i in range(num_limbs)]
        self.BASE_ANGLES = self.SEGMENT_BASE_ANGLES * num_segments
        num_joints = num_segments - 1
        self.NUM_ACTIONS = self.NUM_LIMBS * 3 + num_joints * 2 + self.EMISSION_BITS
        extra_dims = dims - 2
        self.OBS_DIM = 13 * self.NUM_LIMBS + 4 * num_joints + 82 + extra_dims * 2 + self.NUM_LIMBS + 3
        self.ENERGY_OBS_INDEX = 6 * self.NUM_LIMBS
        self.CORE_OBS_DIM = 9 * self.NUM_LIMBS + 42
        self.x = 10.0
        self.y = 10.0
        self.z = 5.0 if dims == 3 else 0.0
        self.heading = 0.0
        self.pitch = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.omega = 0.0
        self.joint_angles = [0.0] * num_joints
        self.limb_angles = list(self.BASE_ANGLES)
        self.limb_extended = [False] * self.NUM_LIMBS
        self.history = deque(maxlen=self.SEQ_LEN)
        self.last_actions = np.zeros(self.NUM_ACTIONS, dtype=int)
        self.fatigue = [0.0] * self.NUM_LIMBS
        self.energy = 1.0
        self.pain_memory = np.zeros((self.MEMORY_GRID_SIZE, self.MEMORY_GRID_SIZE))
        self.temporal_aversion = [0.0] * self.NUM_LIMBS
        self.receptor_gain = np.ones(self.NUM_LIMBS)
        self.prev_pain = np.zeros(self.NUM_LIMBS)
        self.mm_familiarity = 0.0
        self.mm_context_quality = 0.0
        self.mm_certainty = 0.0
        self.learning_progress = 0.0
        self.pattern_available = 0.0
        self.pattern_certainty = 0.0
        self.controllability = 0.0
        self.external_change = 0.0
        self.planning_value = 0.0
        self.empathic_aversion = 0.0
        self.optimism = 0.0
        self.goal_persistence = 0.0
        self.receptor_conflict = 0.0
        self.predicted_conflict = 0.0
        self.conflict_trend = 0.0
        self.prev_conflict = 0.0
        self.rupture_threshold = 0.5
        self.misalignment_memory = 0.0
        self.rupture_count = 0
        self.concept_match = 0.0
        self.concept_quality = 0.0
        self.physics_mode = False
        self.grip_state = [0] * self.NUM_LIMBS
        self.carried_mass = 0.0
        self.contact_count = 0.0
        self.contact_force = 0.0
        self.experience_log = []

    def reset(self, rng=None):
        self.x = 10.0
        self.y = 10.0
        self.z = 5.0 if self.dims == 3 else 0.0
        self.heading = (rng.uniform(0, 2 * np.pi) if rng else np.random.uniform(0, 2 * np.pi))
        self.pitch = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.omega = 0.0
        self.joint_angles = [0.0] * (self.num_segments - 1)
        self.limb_angles = list(self.BASE_ANGLES)
        self.limb_extended = [False] * self.NUM_LIMBS
        self.history.clear()
        self.last_actions = np.zeros(self.NUM_ACTIONS, dtype=int)
        self.fatigue = [0.0] * self.NUM_LIMBS
        self.energy = 1.0
        self.pain_memory = np.zeros((self.MEMORY_GRID_SIZE, self.MEMORY_GRID_SIZE))
        self.temporal_aversion = [0.0] * self.NUM_LIMBS
        self.receptor_gain = np.ones(self.NUM_LIMBS)
        self.prev_pain = np.zeros(self.NUM_LIMBS)
        self.mm_familiarity = 0.0
        self.mm_context_quality = 0.0
        self.mm_certainty = 0.0
        self.learning_progress = 0.0
        self.pattern_available = 0.0
        self.pattern_certainty = 0.0
        self.controllability = 0.0
        self.external_change = 0.0
        self.planning_value = 0.0
        self.empathic_aversion = 0.0
        self.optimism = 0.0
        self.goal_persistence = 0.0
        self.receptor_conflict = 0.0
        self.predicted_conflict = 0.0
        self.conflict_trend = 0.0
        self.prev_conflict = 0.0
        self.rupture_threshold = 0.5
        self.misalignment_memory = 0.0
        self.rupture_count = 0
        self.concept_match = 0.0
        self.concept_quality = 0.0
        self.grip_state = [0] * self.NUM_LIMBS
        self.carried_mass = 0.0
        self.contact_count = 0.0
        self.contact_force = 0.0
        self.experience_log = []

    def _update_pain_memory(self, tips, pain_values):
        self.pain_memory *= self.MEMORY_DECAY
        for i, (tx, ty) in enumerate(tips):
            gx = int(np.clip(tx / self.MEMORY_CELL_SIZE, 0, self.MEMORY_GRID_SIZE - 1))
            gy = int(np.clip(ty / self.MEMORY_CELL_SIZE, 0, self.MEMORY_GRID_SIZE - 1))
            deposit = 0.3 * pain_values[i] + 0.7 * self.temporal_aversion[i]
            self.pain_memory[gy, gx] += deposit

    def get_segment_pos(self, seg):
        if seg == 0:
            return self.x, self.y, self.heading
        cum_heading = self.heading
        sx, sy = self.x, self.y
        for j in range(seg):
            cum_heading += math.pi + self.joint_angles[j]
            sx += self.SEGMENT_DIST * math.cos(cum_heading)
            sy += self.SEGMENT_DIST * math.sin(cum_heading)
        return sx, sy, cum_heading + math.pi

    def _apply_collapse_response(self):
        """Post-rupture response — four strategies (Epacog taxonomy).

        The organism selects a response based on accumulated misalignment:
          Low misalignment: decay (gentle certainty reduction)
          Medium: adopt (shift toward recent observations)
          High: reset (large certainty drop on conflicted channels)
          Very high: noise (random exploration burst)
        """
        m = self.misalignment_memory
        if m < 0.1:
            for i in range(self.NUM_LIMBS):
                self.receptor_gain[i] *= 0.98
        elif m < 0.3:
            self.goal_persistence = max(0.0, self.goal_persistence - 0.1)
        elif m < 0.6:
            self.receptor_conflict *= 0.5
            self.predicted_conflict *= 0.5
        else:
            for i in range(self.NUM_LIMBS):
                self.receptor_gain[i] = self.GAIN_MIN + (
                    self.GAIN_MAX - self.GAIN_MIN) * np.random.random()

    def get_limb_tips(self):
        tips = []
        for i in range(self.NUM_LIMBS):
            seg = i // self.limbs_per_segment
            sx, sy, sh = self.get_segment_pos(seg)
            local_idx = i % self.limbs_per_segment
            world_angle = sh + self.SEGMENT_BASE_ANGLES[local_idx] + \
                (self.limb_angles[i] - self.BASE_ANGLES[i])
            length = self.LIMB_LEN_EXTENDED if self.limb_extended[i] else self.LIMB_LEN_RETRACTED
            tx = sx + length * math.cos(world_angle)
            ty = sy + length * math.sin(world_angle)
            tips.append((tx, ty))
        return tips

    def get_distance_sample_points(self):
        points = []
        for i in range(self.NUM_DISTANCE_RAYS):
            world_angle = self.heading + self.RAY_ANGLES[i]
            sx = max(0.0, min(self.WIDTH, self.x + self.SENSE_DISTANCE * math.cos(world_angle)))
            sy = max(0.0, min(self.HEIGHT, self.y + self.SENSE_DISTANCE * math.sin(world_angle)))
            points.append((sx, sy))
        return points

    def _get_npc_obs(self, npc):
        if npc is None:
            return np.zeros(12)
        dist = math.sqrt((npc.x - self.x)**2 + (npc.y - self.y)**2)
        npc_dist = max(0.0, 1.0 - dist / self.NPC_SENSE_RANGE)
        dx, dy = npc.x - self.x, npc.y - self.y
        rel_bearing = math.atan2(dy, dx) - self.heading
        npc_accel = np.clip(getattr(npc, 'acceleration', 0.0) / 10.0, -1.0, 1.0)
        npc_omega = np.clip(npc.omega / NPC.MAX_ANG_SPEED, -1.0, 1.0)
        npc_erratic = min(getattr(npc, 'erraticism', 0.0) / 1.0, 1.0)
        npc_em = getattr(npc, 'emission_bits', np.zeros(4))
        return np.array([npc_dist, math.sin(rel_bearing), math.cos(rel_bearing),
                         min(npc.speed_val() / NPC.MAX_SPEED, 1.0),
                         npc_accel, npc_omega, npc_erratic,
                         self.empathic_aversion,
                         float(npc_em[0]), float(npc_em[1]),
                         float(npc_em[2]), float(npc_em[3])])

    def step(self, actions, environment, time_step, slow_pathway_active=False, gate_value=0.0,
             predicted_pain=None, mm_features=None, pattern_features=None, agency_features=None,
             npc=None):
        obs_before = self.history[-1].copy() if len(self.history) > 0 else np.zeros(self.OBS_DIM)
        muscle_end = self.NUM_LIMBS * 3
        num_joints = self.num_segments - 1
        joint_end = muscle_end + num_joints * 2
        muscle_actions = actions[:muscle_end]
        joint_actions = actions[muscle_end:joint_end] if num_joints > 0 else np.array([])
        emission_bits = actions[joint_end:]
        acts = muscle_actions.reshape(self.NUM_LIMBS, 3)
        self.last_actions = actions.copy()

        for j in range(num_joints):
            jl, jr = int(joint_actions[j*2]), int(joint_actions[j*2+1])
            if jl and not jr:
                self.joint_angles[j] += self.JOINT_FLEX_DELTA
            elif jr and not jl:
                self.joint_angles[j] -= self.JOINT_FLEX_DELTA
            self.joint_angles[j] += self.JOINT_SPRING * (0.0 - self.joint_angles[j])
            self.joint_angles[j] = max(-self.JOINT_MAX_DEV,
                                       min(self.JOINT_MAX_DEV, self.joint_angles[j]))

        active_count = 0
        for i in range(self.NUM_LIMBS):
            extend, flex_l, flex_r = int(acts[i, 0]), int(acts[i, 1]), int(acts[i, 2])
            any_active = extend or flex_l or flex_r
            active_count += extend + flex_l + flex_r

            if any_active:
                self.fatigue[i] = min(1.0, self.fatigue[i] + self.FATIGUE_RATE)
            else:
                self.fatigue[i] = max(0.0, self.fatigue[i] - self.RECOVERY_RATE)

            if flex_l and not flex_r:
                self.limb_angles[i] += self.FLEX_DELTA
            elif flex_r and not flex_l:
                self.limb_angles[i] -= self.FLEX_DELTA

            self.limb_angles[i] += self.FLEX_SPRING * (self.BASE_ANGLES[i] - self.limb_angles[i])
            dev = self.limb_angles[i] - self.BASE_ANGLES[i]
            dev = max(-self.FLEX_MAX_DEV, min(self.FLEX_MAX_DEV, dev))
            self.limb_angles[i] = self.BASE_ANGLES[i] + dev
            self.limb_extended[i] = bool(extend)

        if not self.physics_mode:
            total_fx, total_fy, total_torque = 0.0, 0.0, 0.0
            for i in range(self.NUM_LIMBS):
                seg = i // self.limbs_per_segment
                _, _, seg_heading = self.get_segment_pos(seg)
                local_idx = i % self.limbs_per_segment
                world_angle = seg_heading + self.SEGMENT_BASE_ANGLES[local_idx] + \
                    (self.limb_angles[i] - self.BASE_ANGLES[i])

                if self.limb_extended[i]:
                    fatigue_factor = 1.0 - self.fatigue[i] * 0.8
                    energy_factor = 0.3 + 0.7 * self.energy
                    thrust = self.THRUST_FORCE * fatigue_factor * energy_factor
                    total_fx += -thrust * math.cos(world_angle)
                    total_fy += -thrust * math.sin(world_angle)

                if int(acts[i, 1]):
                    total_torque += 0.2
                if int(acts[i, 2]):
                    total_torque -= 0.2

            total_fx -= self.DRAG * self.vx
            total_fy -= self.DRAG * self.vy
            total_torque -= self.ANG_DRAG * self.omega

            for wall_pos, axis, sign in [
                (0, 'x', 1), (self.WIDTH, 'x', -1),
                (0, 'y', 1), (self.HEIGHT, 'y', -1),
            ]:
                coord = self.x if axis == 'x' else self.y
                dist = abs(coord - wall_pos)
                if dist < self.WALL_DIST:
                    force = self.WALL_FORCE * (1 - dist / self.WALL_DIST) ** 2
                    if axis == 'x':
                        total_fx += force * sign
                    else:
                        total_fy += force * sign

            self.vx += (total_fx / self.MASS) * self.DT
            self.vy += (total_fy / self.MASS) * self.DT
            self.omega += (total_torque / self.MOI) * self.DT

            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            if speed > self.MAX_SPEED:
                self.vx = self.vx / speed * self.MAX_SPEED
                self.vy = self.vy / speed * self.MAX_SPEED
            self.omega = max(-self.MAX_ANG_SPEED, min(self.MAX_ANG_SPEED, self.omega))

            self.x += self.vx * self.DT
            self.y += self.vy * self.DT
            self.heading += self.omega * self.DT

            self.x = max(0.5, min(19.5, self.x))
            self.y = max(0.5, min(19.5, self.y))

        self.energy = max(0.0, self.energy - self.ENERGY_COST * active_count)
        self.energy = max(0.0, self.energy - self.SLOW_PATHWAY_COST * gate_value)
        self.energy = max(0.0, self.energy - self.EMISSION_COST * emission_bits.sum())

        for obj in environment.responsive_objects:
            obj.update(self.x, self.y, emission_bits, time_step)
        recovery = 0.0
        for src in environment.endorphin_sources:
            recovery += src.intensity_at(self.x, self.y, time_step)
        for src in environment.chemical_sources:
            recovery += src.intensity_at(self.x, self.y, time_step)
        self.energy = min(1.0, self.energy + recovery * self.ENERGY_RECOVERY)

        tips = self.get_limb_tips()
        pain_raw, endorphin = environment.get_field_values(tips, time_step)
        temperature = environment.get_temperature_values(tips, time_step)
        chemical = environment.get_chemical_values(tips, time_step)
        pressure = environment.get_pressure_values(tips)

        if npc is not None:
            for src in environment.endorphin_sources:
                sx, sy = src.position(time_step)
                nd = math.sqrt((npc.x - sx)**2 + (npc.y - sy)**2)
                if nd < self.NPC_COMPETITION_RANGE:
                    comp = max(0.0, 1.0 - nd / self.NPC_COMPETITION_RANGE)
                    endorphin *= (1.0 - 0.5 * comp)
            for src in environment.chemical_sources:
                sx, sy = src.position(time_step)
                nd = math.sqrt((npc.x - sx)**2 + (npc.y - sy)**2)
                if nd < self.NPC_COMPETITION_RANGE:
                    comp = max(0.0, 1.0 - nd / self.NPC_COMPETITION_RANGE)
                    chemical *= (1.0 - 0.5 * comp)

        for obj in environment.responsive_objects:
            if obj.responding:
                for i, (tx, ty) in enumerate(tips):
                    effect = obj.effect_at(tx, ty, time_step)
                    if obj.response_type == 'endorphin':
                        endorphin[i] += effect * 1.5
                    elif obj.response_type == 'cool':
                        temperature[i] = max(0.0, temperature[i] - effect * 0.3)

        object_proximity = np.zeros(3)
        object_responding = np.zeros(3)
        for j, obj in enumerate(environment.responsive_objects):
            dist = obj.distance_to(self.x, self.y, time_step)
            object_proximity[j] = max(0.0, 1.0 - dist / self.OBJ_SENSE_RANGE)
            object_responding[j] = 1.0 if obj.responding else 0.0

        for i in range(self.NUM_LIMBS):
            if pain_raw[i] > 0.2:
                self.receptor_gain[i] = min(
                    self.GAIN_MAX,
                    self.receptor_gain[i] + self.SENSITIZATION_RATE)
            elif endorphin[i] + chemical[i] > 0.3 and pain_raw[i] < 0.1:
                self.receptor_gain[i] = max(
                    self.GAIN_MIN,
                    self.receptor_gain[i] - self.HABITUATION_RATE)
            else:
                self.receptor_gain[i] += self.GAIN_RECOVERY_RATE * (1.0 - self.receptor_gain[i])

        pain = pain_raw * self.receptor_gain

        if npc is not None:
            for i, (tx, ty) in enumerate(tips):
                td = math.sqrt((tx - npc.x)**2 + (ty - npc.y)**2)
                if td < self.NPC_COLLISION_RADIUS:
                    pain[i] += self.NPC_COLLISION_PAIN * math.exp(-(td**2) / 0.5)

        if npc is not None:
            npc_dist = npc.distance_to(self.x, self.y)
            npc_erratic = getattr(npc, 'erraticism', 0.0)
            proximity_factor = max(0.0, 1.0 - npc_dist / self.EMPATHY_RANGE)
            self.empathic_aversion = float(np.clip(
                self.EMPATHY_WEIGHT * proximity_factor * npc_erratic, 0.0, 1.0))
            for i in range(self.NUM_LIMBS):
                pain[i] += self.empathic_aversion * 0.2

        self._update_pain_memory(tips, pain)

        for i in range(self.NUM_LIMBS):
            self.temporal_aversion[i] = (
                self.TEMPORAL_AVERSION_DECAY * self.temporal_aversion[i]
                + self.TEMPORAL_AVERSION_ALPHA * pain[i]
            )

        dist_points = self.get_distance_sample_points()
        distant_pain, distant_endorphin = environment.get_field_values(dist_points, time_step)

        best_distant_endo = float(distant_endorphin.max()) if len(distant_endorphin) > 0 else 0.0
        avg_current_pain = float(pain.mean())
        self.optimism = float(np.clip(best_distant_endo - avg_current_pain, 0.0, 1.0))
        self.goal_persistence = 0.95 * self.goal_persistence + 0.05 * self.optimism

        pain_pressure = float(pain.mean())
        reward_pull = float(endorphin.mean() + 0.5 * chemical.mean())
        conflict_pain_reward = min(pain_pressure, reward_pull)
        conflict_optimism = self.goal_persistence * pain_pressure
        conflict_empathy = self.empathic_aversion * (1.0 - pain_pressure)
        self.receptor_conflict = float(np.clip(
            conflict_pain_reward + 0.5 * conflict_optimism + 0.3 * conflict_empathy, 0.0, 1.0))
        self.predicted_conflict = float(np.clip(
            self.receptor_conflict * (1.0 + 0.5 * (1.0 - self.mm_certainty)), 0.0, 1.5))
        self.conflict_trend = float(np.clip(
            0.9 * self.conflict_trend + 0.1 * (self.receptor_conflict - self.prev_conflict), -1.0, 1.0))
        self.prev_conflict = self.receptor_conflict

        if predicted_pain is not None:
            pe_mag = float(np.mean(np.abs(pain - np.array(predicted_pain))))
        else:
            pe_mag = 0.0
        self.misalignment_memory = 0.95 * self.misalignment_memory + 0.05 * pe_mag

        if self.receptor_conflict > self.rupture_threshold:
            self.rupture_count += 1
            self.rupture_threshold = min(1.0, self.rupture_threshold + 0.02)
            self._apply_collapse_response()
        else:
            self.rupture_threshold = max(0.2, self.rupture_threshold - 0.005)

        if predicted_pain is not None:
            prediction_error = np.abs(pain - np.array(predicted_pain))
        else:
            prediction_error = np.abs(pain - self.prev_pain)
        self.prev_pain = pain.copy()

        if mm_features is not None:
            self.mm_familiarity = float(mm_features[0])
            self.mm_context_quality = float(mm_features[1])
            if len(mm_features) > 2:
                self.mm_certainty = float(mm_features[2])
            if len(mm_features) > 3:
                self.learning_progress = float(mm_features[3])

        if pattern_features is not None:
            self.pattern_available = float(pattern_features[0])
            self.pattern_certainty = float(pattern_features[1])

        if agency_features is not None:
            self.controllability = float(agency_features[0])
            self.external_change = float(agency_features[1])
            self.planning_value = float(agency_features[2])

        obs = np.concatenate([
            pain, endorphin, temperature, chemical, pressure,
            np.array(self.fatigue, dtype=np.float64),
            np.array([self.energy]),
            np.clip(self.temporal_aversion, 0.0, 1.0),
            self.receptor_gain,
            self.pain_memory.flatten(),
            distant_pain,
            distant_endorphin,
            prediction_error,
            np.array([self.mm_familiarity, self.mm_context_quality,
                      self.mm_certainty, self.learning_progress]),
            np.array([self.pattern_available, self.pattern_certainty]),
            np.array([math.sqrt(self.vx**2 + self.vy**2 + self.vz**2) / self.MAX_SPEED,
                      self.omega / self.MAX_ANG_SPEED]
                     + ([self.pitch / (math.pi/4)] if self.dims == 3 else [])),
            np.array([(self.limb_angles[i] - self.BASE_ANGLES[i]) / self.FLEX_MAX_DEV
                      for i in range(self.NUM_LIMBS)], dtype=np.float64),
            self.last_actions.astype(np.float64),
            np.array([self.controllability, self.external_change, self.planning_value]),
            object_proximity,
            object_responding,
            self._get_npc_obs(npc),
            np.array(sum([[self.joint_angles[j] / self.JOINT_MAX_DEV,
                           0.0] for j in range(self.num_segments - 1)], [])) if self.num_segments > 1 else np.array([]),
            np.array([self.optimism, self.goal_persistence]),
            np.array([self.z / self.DEPTH] if self.dims == 3 else []),
            np.array([self.receptor_conflict, self.predicted_conflict, self.conflict_trend]),
            np.array([self.concept_match, self.concept_quality]),
            np.array(self.grip_state, dtype=np.float64),
            np.array([self.carried_mass, self.contact_count, self.contact_force]),
        ])
        self.history.append(obs)

        temp_discomfort = np.sum(
            np.maximum(0, temperature - 0.7) + np.maximum(0, 0.3 - temperature)
        )
        reward = (
            -pain.sum() + endorphin.sum()
            + 0.5 * chemical.sum()
            - 0.5 * temp_discomfort
            - 0.3 * pressure.sum()
            + 0.1 * self.energy
        )
        self.experience_log.append({
            'time_step': time_step,
            'action': actions.copy(),
            'obs_before': obs_before,
            'obs_after': obs.copy(),
            'reward': float(reward),
        })
        return obs, reward

    def get_observation_window(self):
        window = np.zeros((self.SEQ_LEN, self.OBS_DIM))
        hist = list(self.history)
        offset = self.SEQ_LEN - len(hist)
        for i, obs in enumerate(hist):
            window[offset + i] = obs
        return window

    def get_frame_data(self, environment, time_step, npc=None):
        pain_pos, endo_pos, heat_pos, cold_pos, chem_pos = environment.get_source_positions(time_step)
        tips = self.get_limb_tips()
        pain_raw, endorphin = environment.get_field_values(tips, time_step)
        pain = pain_raw * self.receptor_gain
        temperature = environment.get_temperature_values(tips, time_step)
        chemical = environment.get_chemical_values(tips, time_step)
        pressure = environment.get_pressure_values(tips)
        dist_points = self.get_distance_sample_points()
        dist_pain, dist_endo = environment.get_field_values(dist_points, time_step)

        temp_discomfort = np.sum(
            np.maximum(0, temperature - 0.7) + np.maximum(0, 0.3 - temperature)
        )
        reward = (
            -pain.sum() + endorphin.sum()
            + 0.5 * chemical.sum()
            - 0.5 * temp_discomfort
            - 0.3 * pressure.sum()
            + 0.1 * self.energy
        )
        return {
            't': time_step,
            'bx': round(self.x, 3), 'by': round(self.y, 3), 'bh': round(self.heading, 3),
            'la': [round(a, 3) for a in self.limb_angles],
            'le': [int(e) for e in self.limb_extended],
            'ma': self.last_actions.tolist(),
            'pr': [round(float(p), 3) for p in pain],
            'er': [round(float(e), 3) for e in endorphin],
            'tr': [round(float(v), 3) for v in temperature],
            'cr': [round(float(v), 3) for v in chemical],
            'wr': [round(float(v), 3) for v in pressure],
            'fa': [round(f, 3) for f in self.fatigue],
            'en': round(self.energy, 3),
            'ta': [round(min(v, 1.0), 3) for v in self.temporal_aversion],
            'rg': [round(float(g), 3) for g in self.receptor_gain],
            'dp': [round(float(v), 3) for v in dist_pain],
            'de': [round(float(v), 3) for v in dist_endo],
            'ds': [[round(p[0], 3), round(p[1], 3)] for p in dist_points],
            'pm': [round(float(v), 3) for v in self.pain_memory.flatten()],
            'sp': [[round(p[0], 3), round(p[1], 3)] for p in pain_pos],
            'se': [[round(e[0], 3), round(e[1], 3)] for e in endo_pos],
            'sh': [[round(h[0], 3), round(h[1], 3)] for h in heat_pos],
            'sk': [[round(c[0], 3), round(c[1], 3)] for c in cold_pos],
            'sc': [[round(c[0], 3), round(c[1], 3)] for c in chem_pos],
            'pe': [round(float(abs(pain[i] - self.prev_pain[i])), 3) for i in range(self.NUM_LIMBS)],
            'mf': round(self.mm_familiarity, 3),
            'mq': round(self.mm_context_quality, 3),
            'mc': round(self.mm_certainty, 3),
            'lp': round(self.learning_progress, 3),
            'pa': round(self.pattern_available, 3),
            'pc': round(self.pattern_certainty, 3),
            'ct': round(self.controllability, 3),
            'ec': round(self.external_change, 3),
            'pv': round(self.planning_value, 3),
            'spd': round(math.sqrt(self.vx**2 + self.vy**2) / self.MAX_SPEED, 3),
            'av': round(self.omega / self.MAX_ANG_SPEED, 3),
            'ld': [round((self.limb_angles[i] - self.BASE_ANGLES[i]) / self.FLEX_MAX_DEV, 3)
                   for i in range(self.NUM_LIMBS)],
            'em': [int(b) for b in self.last_actions[self.NUM_LIMBS * 3:]],
            'ro': [obj.to_dict(time_step) for obj in environment.responsive_objects],
            'ea': round(self.empathic_aversion, 3),
            'opt': round(self.optimism, 3),
            'gp': round(self.goal_persistence, 3),
            'rc': round(self.receptor_conflict, 3),
            'pc_conf': round(self.predicted_conflict, 3),
            'ct_trend': round(self.conflict_trend, 3),
            'cm': round(self.concept_match, 3),
            'cq': round(self.concept_quality, 3),
            'npc': npc.to_dict() if npc is not None else None,
            'r': round(float(reward), 3),
            'ls': len(self.experience_log),
        }

    def compute_optimal_actions(self, environment, time_step, npc=None,
                                 zone_targets=None):
        tips = self.get_limb_tips()
        actions = np.zeros(self.NUM_ACTIONS, dtype=int)
        pain_raw, _ = environment.get_field_values(tips, time_step)

        zone_gx, zone_gy = 0.0, 0.0
        if zone_targets:
            best_dist = float('inf')
            best_zx, best_zy = 0.0, 0.0
            for zx, zy in zone_targets:
                d = math.sqrt((zx - self.x)**2 + (zy - self.y)**2)
                if d < best_dist:
                    best_dist = d
                    best_zx, best_zy = zx, zy
            if best_dist > 0.5:
                strength = 3.0 + 2.0 / (best_dist + 0.1)
                zone_gx = (best_zx - self.x) / best_dist * strength
                zone_gy = (best_zy - self.y) / best_dist * strength

        dist_points = self.get_distance_sample_points()
        dist_pain, dist_endo = environment.get_field_values(dist_points, time_step)
        dist_gx, dist_gy = 0.0, 0.0
        for k in range(self.NUM_DISTANCE_RAYS):
            ray_angle = self.heading + self.RAY_ANGLES[k]
            signal = dist_endo[k] - dist_pain[k]
            dist_gx += signal * math.cos(ray_angle)
            dist_gy += signal * math.sin(ray_angle)
        dist_gx *= self.DIST_GRADIENT_WEIGHT
        dist_gy *= self.DIST_GRADIENT_WEIGHT

        npc_em = tuple(int(b) for b in getattr(npc, 'emission_bits', [0,0,0,0])) if npc is not None else (0,0,0,0)

        for i in range(self.NUM_LIMBS):
            tx, ty = tips[i]
            gx, gy = environment.get_combined_gradient(tx, ty, time_step)
            gx += dist_gx + zone_gx
            gy += dist_gy + zone_gy

            if self.goal_persistence > 0.2:
                gx *= (1.0 + 0.3 * self.goal_persistence)
                gy *= (1.0 + 0.3 * self.goal_persistence)

            for my in range(self.MEMORY_GRID_SIZE):
                for mx in range(self.MEMORY_GRID_SIZE):
                    mem_val = self.pain_memory[my, mx]
                    if mem_val > self.MEMORY_PHANTOM_THRESHOLD:
                        cell_cx = (mx + 0.5) * self.MEMORY_CELL_SIZE
                        cell_cy = (my + 0.5) * self.MEMORY_CELL_SIZE
                        dx = tx - cell_cx
                        dy = ty - cell_cy
                        dist_sq = dx * dx + dy * dy
                        if dist_sq > 0.1:
                            dist = math.sqrt(dist_sq)
                            strength = self.MEMORY_PHANTOM_STRENGTH * mem_val / (dist_sq + 1.0)
                            gx += strength * dx / dist
                            gy += strength * dy / dist

            if self.temporal_aversion[i] > 0.1:
                temporal_boost = 0.3 * self.temporal_aversion[i] * (1.0 - 0.3 * self.goal_persistence)
                gx *= (1.0 + temporal_boost)
                gy *= (1.0 + temporal_boost)

            pe = abs(float(pain_raw[i] * self.receptor_gain[i] - self.prev_pain[i]))
            if pe > 0.1:
                gx *= (1.0 + 0.5 * pe)
                gy *= (1.0 + 0.5 * pe)

            if self.mm_certainty < 0.3:
                caution = 0.4 * (1.0 - self.mm_certainty)
                gx *= (1.0 + caution)
                gy *= (1.0 + caution)

            if self.learning_progress > 0.2:
                dampen = 0.2 * self.learning_progress
                gx *= (1.0 - dampen)
                gy *= (1.0 - dampen)

            if self.pattern_available > 0.5 and self.pattern_certainty > 0.5:
                pat_dampen = 0.15 * self.pattern_available * self.pattern_certainty
                gx *= (1.0 - pat_dampen)
                gy *= (1.0 - pat_dampen)

            if self.controllability < 0.3:
                reactive = 0.3 * (1.0 - self.controllability)
                gx *= (1.0 + reactive)
                gy *= (1.0 + reactive)

            if self.controllability > 0.6 and self.planning_value > 0.3:
                plan_dampen = 0.15 * self.controllability * self.planning_value
                gx *= (1.0 - plan_dampen)
                gy *= (1.0 - plan_dampen)

            if npc is not None:
                dx_npc = tx - npc.x
                dy_npc = ty - npc.y
                npc_d2 = dx_npc * dx_npc + dy_npc * dy_npc
                if npc_d2 > 0.01:
                    npc_d = math.sqrt(npc_d2)
                    repulsion = 1.0 * math.exp(-npc_d2 / 8.0)
                    gx += repulsion * dx_npc / npc_d
                    gy += repulsion * dy_npc / npc_d

                    if npc_d < NPC.SIGNAL_RANGE:
                        if npc_em == NPC.REPEL_SIGNAL:
                            gx += 0.5 * dx_npc / npc_d
                            gy += 0.5 * dy_npc / npc_d
                        elif npc_em == (0, 0, 1, 0):
                            gx -= 0.3 * dx_npc / npc_d
                            gy -= 0.3 * dy_npc / npc_d

            if self.receptor_conflict > 0.3:
                conflict_dampen = 0.15 * self.receptor_conflict
                gx *= (1.0 - conflict_dampen)
                gy *= (1.0 - conflict_dampen)

            if self.predicted_conflict > 0.5 and self.conflict_trend > 0:
                gx *= (1.0 - 0.1 * self.predicted_conflict)
                gy *= (1.0 - 0.1 * self.predicted_conflict)
            elif self.conflict_trend < -0.1:
                gx *= (1.0 + 0.1 * abs(self.conflict_trend))
                gy *= (1.0 + 0.1 * abs(self.conflict_trend))

            if self.concept_match > 0.5 and self.concept_quality > 0.3:
                concept_conf = 0.1 * self.concept_match * self.concept_quality
                gx *= (1.0 - concept_conf)
                gy *= (1.0 - concept_conf)

            grad_mag = math.sqrt(gx * gx + gy * gy)
            if grad_mag < 1e-6:
                continue

            desired_dir = math.atan2(gy, gx)
            seg = i // self.limbs_per_segment
            _, _, seg_h = self.get_segment_pos(seg)
            local_idx = i % self.limbs_per_segment
            world_angle = seg_h + self.SEGMENT_BASE_ANGLES[local_idx] + \
                (self.limb_angles[i] - self.BASE_ANGLES[i])
            limb_dir = math.atan2(math.sin(world_angle), math.cos(world_angle))

            dot = math.cos(desired_dir - (limb_dir + math.pi))
            fatigue_penalty = self.fatigue[i] * 0.5
            energy_penalty = max(0, (0.3 - self.energy) * 1.0)
            if self.fatigue[i] < 0.9 and dot > (-0.3 + fatigue_penalty + energy_penalty):
                actions[i * 3] = 1

            angle_diff = desired_dir - (limb_dir + math.pi)
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

            if abs(angle_diff) > 0.15:
                rel_angle = desired_dir - seg_h
                rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
                limb_base = self.SEGMENT_BASE_ANGLES[local_idx]
                limb_dev = self.limb_angles[i] - self.BASE_ANGLES[i]
                limb_rel = limb_base + limb_dev
                diff_to_desired = rel_angle - (limb_rel + math.pi)
                diff_to_desired = math.atan2(math.sin(diff_to_desired), math.cos(diff_to_desired))
                if diff_to_desired > 0.1:
                    actions[i * 3 + 1] = 1
                elif diff_to_desired < -0.1:
                    actions[i * 3 + 2] = 1

        best_obj = None
        best_benefit = 0.0
        temperature = environment.get_temperature_values(tips, time_step)
        for obj in environment.responsive_objects:
            if obj.responding:
                continue
            dist = obj.distance_to(self.x, self.y, time_step)
            if dist > obj.trigger_range:
                continue
            if obj.response_type == 'endorphin':
                benefit = 2.0
            elif obj.response_type == 'cool' and np.mean(temperature) > 0.6:
                benefit = 1.5
            elif obj.response_type == 'approach':
                benefit = 1.0
            else:
                benefit = 0.0
            if benefit > best_benefit:
                best_benefit = benefit
                best_obj = obj

        em_start = self.NUM_LIMBS * 3 + (self.num_segments - 1) * 2
        if best_obj is not None:
            for k, bit in enumerate(best_obj.trigger_signal):
                actions[em_start + k] = bit
        elif npc is not None:
            npc_dist = npc.distance_to(self.x, self.y)
            if npc_dist < NPC.SIGNAL_RANGE:
                if npc_dist < self.NPC_COLLISION_RADIUS * 2.5:
                    for k, bit in enumerate(NPC.REPEL_SIGNAL):
                        actions[em_start + k] = bit
                elif npc.erraticism > 0.3 and npc_dist < self.EMPATHY_RANGE:
                    for k, bit in enumerate(NPC.CALM_SIGNAL):
                        actions[em_start + k] = bit

        return actions

    def compute_optimal_actions_continuous(self, environment, time_step, npc=None,
                                            zone_targets=None):
        """Return continuous force/torque per limb + binary grip + emission.

        Action layout: [force_0, torque_0, ..., force_N, torque_N,
                        grip_0, ..., grip_N, joint_actions..., emission_bits...]
        Continuous dims: NUM_LIMBS * 2
        Binary dims: NUM_LIMBS + (num_segments-1)*2 + 4
        """
        binary = self.compute_optimal_actions(environment, time_step, npc=npc,
                                               zone_targets=zone_targets)
        tips = self.get_limb_tips()
        num_continuous = self.NUM_LIMBS * 2
        num_binary = self.NUM_LIMBS + (self.num_segments - 1) * 2 + 4
        actions = np.zeros(num_continuous + num_binary, dtype=np.float64)

        dist_points = self.get_distance_sample_points()
        dist_pain, dist_endo = environment.get_field_values(dist_points, time_step)
        dist_gx, dist_gy = 0.0, 0.0
        for k in range(self.NUM_DISTANCE_RAYS):
            ray_angle = self.heading + self.RAY_ANGLES[k]
            signal = dist_endo[k] - dist_pain[k]
            dist_gx += signal * math.cos(ray_angle)
            dist_gy += signal * math.sin(ray_angle)
        dist_gx *= self.DIST_GRADIENT_WEIGHT
        dist_gy *= self.DIST_GRADIENT_WEIGHT

        zone_gx, zone_gy = 0.0, 0.0
        if zone_targets:
            best_dist = float('inf')
            best_zx, best_zy = 0.0, 0.0
            for zx, zy in zone_targets:
                d = math.sqrt((zx - self.x)**2 + (zy - self.y)**2)
                if d < best_dist:
                    best_dist = d
                    best_zx, best_zy = zx, zy
            if best_dist > 0.5:
                strength = 3.0 + 2.0 / (best_dist + 0.1)
                zone_gx = (best_zx - self.x) / best_dist * strength
                zone_gy = (best_zy - self.y) / best_dist * strength

        for i in range(self.NUM_LIMBS):
            tx, ty = tips[i]
            gx, gy = environment.get_combined_gradient(tx, ty, time_step)
            gx += dist_gx + zone_gx
            gy += dist_gy + zone_gy

            grad_mag = math.sqrt(gx * gx + gy * gy)
            if grad_mag < 1e-6:
                continue

            desired_dir = math.atan2(gy, gx)
            seg = i // self.limbs_per_segment
            _, _, seg_h = self.get_segment_pos(seg)
            local_idx = i % self.limbs_per_segment
            world_angle = seg_h + self.SEGMENT_BASE_ANGLES[local_idx] + \
                (self.limb_angles[i] - self.BASE_ANGLES[i])

            dot = math.cos(desired_dir - (world_angle + math.pi))
            fatigue_factor = 1.0 - self.fatigue[i] * 0.8
            energy_factor = 0.3 + 0.7 * self.energy
            force = float(np.clip(grad_mag * dot * fatigue_factor * energy_factor, -1.0, 1.0))
            actions[i * 2] = force

            angle_diff = desired_dir - (world_angle + math.pi)
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            torque = float(np.clip(angle_diff * 2.0, -1.0, 1.0))
            actions[i * 2 + 1] = torque

        grip_start = num_continuous
        for i in range(self.NUM_LIMBS):
            actions[grip_start + i] = binary[i * 3]

        em_binary_start = self.NUM_LIMBS * 3 + (self.num_segments - 1) * 2
        em_out_start = grip_start + self.NUM_LIMBS + (self.num_segments - 1) * 2
        for k in range(4):
            actions[em_out_start + k] = binary[em_binary_start + k]

        return actions

    def to_dict(self):
        return {
            'num_limbs': self.NUM_LIMBS,
            'limb_length_retracted': self.LIMB_LEN_RETRACTED,
            'limb_length_extended': self.LIMB_LEN_EXTENDED,
            'base_angles_deg': [round(math.degrees(a)) for a in self.BASE_ANGLES],
            'body_radius': self.BODY_RADIUS,
        }


def _run_test(num_limbs=6, steps=100):
    expected_obs = 13 * num_limbs + 82 + num_limbs + 3
    expected_actions = 3 * num_limbs + 4
    print(f"\n=== Testing {num_limbs}-limbed organism (obs={expected_obs}, actions={expected_actions}) ===")
    env = Environment(seed=42)
    org = Organism(num_limbs=num_limbs)
    org.reset()
    npc = NPC()
    npc.reset()
    total_reward = 0
    repel_count, calm_count = 0, 0
    for step in range(steps):
        npc.step(env, step)
        actions = org.compute_optimal_actions(env, step, npc=npc)
        obs, reward = org.step(actions, env, step, npc=npc)
        emission = actions[num_limbs * 3:]
        npc.receive_signal(emission, org.x, org.y)
        em_t = tuple(int(b) for b in emission)
        if em_t == NPC.REPEL_SIGNAL: repel_count += 1
        if em_t == NPC.CALM_SIGNAL: calm_count += 1
        total_reward += reward
    window = org.get_observation_window()
    print(f"  Reward: {total_reward:.1f}  Energy: {org.energy:.3f}")
    print(f"  Obs: {obs.shape}  Window: {window.shape}")
    assert obs.shape == (expected_obs,), f"Expected obs_dim={expected_obs}, got {obs.shape}"
    assert window.shape == (32, expected_obs), f"Expected (32,{expected_obs}), got {window.shape}"
    assert len(actions) == expected_actions, f"Expected {expected_actions} actions, got {len(actions)}"
    assert org.ENERGY_OBS_INDEX == 6 * num_limbs
    assert org.CORE_OBS_DIM == 9 * num_limbs + 42
    print(f"  ENERGY_OBS_INDEX={org.ENERGY_OBS_INDEX}  CORE_OBS_DIM={org.CORE_OBS_DIM}")
    conflict_idx = expected_obs - 1
    print(f"  Optimism: {obs[conflict_idx-2]:.3f}  Persistence: {obs[conflict_idx-1]:.3f}  Conflict: {obs[conflict_idx]:.3f}")
    print(f"  Signals: repel={repel_count} calm={calm_count}")
    print(f"  PASS: {num_limbs}-limbed organism")
    return total_reward


if __name__ == '__main__':
    _run_test(num_limbs=6)
    _run_test(num_limbs=4)
    _run_test(num_limbs=8)
    _run_test(num_limbs=12)

    print(f"\n=== Testing 2-segment organism ===")
    org2 = Organism(num_limbs=6, num_segments=2)
    org2.reset()
    npc2 = NPC()
    npc2.reset()
    env2 = Environment(seed=42)
    for step in range(50):
        npc2.step(env2, step)
        actions = org2.compute_optimal_actions(env2, step, npc=npc2)
        obs, reward = org2.step(actions, env2, step, npc=npc2)
    print(f"  2-seg obs: {obs.shape} (expected {org2.OBS_DIM})")
    print(f"  2-seg actions: {len(actions)} (expected {org2.NUM_ACTIONS})")
    assert obs.shape == (org2.OBS_DIM,), f"Expected {org2.OBS_DIM}, got {obs.shape}"
    print(f"  PASS: 2-segment organism")

    print(f"\n=== Testing 3D organism ===")
    org3 = Organism(num_limbs=6, dims=3)
    org3.reset()
    npc3 = NPC()
    npc3.reset()
    env3 = Environment(seed=42)
    for step in range(50):
        npc3.step(env3, step)
        actions = org3.compute_optimal_actions(env3, step, npc=npc3)
        obs, reward = org3.step(actions, env3, step, npc=npc3)
    print(f"  3D obs: {obs.shape} (expected {org3.OBS_DIM})")
    assert obs.shape == (org3.OBS_DIM,), f"Expected {org3.OBS_DIM}, got {obs.shape}"
    print(f"  PASS: 3D organism")

    print("\nAll scaling tests passed!")

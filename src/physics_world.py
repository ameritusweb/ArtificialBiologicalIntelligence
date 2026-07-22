import math
import numpy as np
import pymunk
from environment import Environment, Organism, NPC


COLLISION_TYPE_WALL = 1
COLLISION_TYPE_ORGANISM = 2
COLLISION_TYPE_LIMB = 3
COLLISION_TYPE_OBJECT = 4
COLLISION_TYPE_NPC = 5


COLLISION_TYPE_COMPOUND = 6


class CompoundObject:
    """Two rigid bodies connected by a joint — levers, gates, spring-loaded barriers."""

    def __init__(self, space, x, y, obj_type='lever', seed=None):
        rng = np.random.RandomState(seed)
        self.obj_type = obj_type
        self.bodies = []
        self.shapes = []
        self.joints = []

        if obj_type == 'lever':
            arm_len = 2.0
            mass_a, mass_b = 1.0, 1.0
            mom_a = pymunk.moment_for_segment(mass_a, (-arm_len/2, 0), (0, 0), 0.1)
            mom_b = pymunk.moment_for_segment(mass_b, (0, 0), (arm_len/2, 0), 0.1)

            body_a = pymunk.Body(mass_a, mom_a)
            body_a.position = (x - arm_len/4, y)
            seg_a = pymunk.Segment(body_a, (-arm_len/4, 0), (arm_len/4, 0), 0.15)
            seg_a.friction = 0.7
            seg_a.collision_type = COLLISION_TYPE_COMPOUND

            body_b = pymunk.Body(mass_b, mom_b)
            body_b.position = (x + arm_len/4, y)
            seg_b = pymunk.Segment(body_b, (-arm_len/4, 0), (arm_len/4, 0), 0.15)
            seg_b.friction = 0.7
            seg_b.collision_type = COLLISION_TYPE_COMPOUND

            pivot = pymunk.PivotJoint(body_a, body_b, (x, y))
            rot_spring = pymunk.DampedRotarySpring(body_a, body_b, 0, 50.0, 5.0)

            space.add(body_a, seg_a, body_b, seg_b, pivot, rot_spring)
            self.bodies = [body_a, body_b]
            self.shapes = [seg_a, seg_b]
            self.joints = [pivot, rot_spring]

        elif obj_type == 'spring_gate':
            mass_gate = 3.0
            moment = pymunk.moment_for_box(mass_gate, (1.5, 0.3))
            gate_body = pymunk.Body(mass_gate, moment)
            gate_body.position = (x, y)
            gate_shape = pymunk.Poly.create_box(gate_body, (1.5, 0.3))
            gate_shape.friction = 0.6
            gate_shape.collision_type = COLLISION_TYPE_COMPOUND

            anchor = space.static_body
            spring = pymunk.DampedSpring(anchor, gate_body, (x, y), (0, 0),
                                          rest_length=0, stiffness=80.0, damping=8.0)
            slide = pymunk.GrooveJoint(anchor, gate_body, (x - 2, y), (x + 2, y), (0, 0))

            space.add(gate_body, gate_shape, spring, slide)
            self.bodies = [gate_body]
            self.shapes = [gate_shape]
            self.joints = [spring, slide]

        elif obj_type == 'hinged_barrier':
            mass_bar = 2.0
            bar_len = 2.5
            moment = pymunk.moment_for_segment(mass_bar, (-bar_len/2, 0), (bar_len/2, 0), 0.15)
            bar_body = pymunk.Body(mass_bar, moment)
            bar_body.position = (x, y)
            bar_shape = pymunk.Segment(bar_body, (-bar_len/2, 0), (bar_len/2, 0), 0.15)
            bar_shape.friction = 0.7
            bar_shape.collision_type = COLLISION_TYPE_COMPOUND

            hinge_x = x - bar_len/2
            pivot = pymunk.PivotJoint(space.static_body, bar_body, (hinge_x, y))
            rot_spring = pymunk.DampedRotarySpring(space.static_body, bar_body, 0, 30.0, 4.0)

            space.add(bar_body, bar_shape, pivot, rot_spring)
            self.bodies = [bar_body]
            self.shapes = [bar_shape]
            self.joints = [pivot, rot_spring]

    @property
    def x(self):
        return float(self.bodies[0].position.x)

    @property
    def y(self):
        return float(self.bodies[0].position.y)

    @property
    def angle(self):
        return float(self.bodies[0].angle)

    def to_dict(self):
        return {
            'type': self.obj_type,
            'bodies': [{'x': round(float(b.position.x), 3),
                         'y': round(float(b.position.y), 3),
                         'angle': round(float(b.angle), 3)}
                        for b in self.bodies],
        }


class RigidObject:
    def __init__(self, space, x, y, mass=2.0, radius=0.6, friction=0.5):
        self.mass = mass
        self.radius = radius
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x, y)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = friction
        self.shape.elasticity = 0.3
        self.shape.collision_type = COLLISION_TYPE_OBJECT
        space.add(self.body, self.shape)

    @property
    def x(self):
        return float(self.body.position.x)

    @property
    def y(self):
        return float(self.body.position.y)

    def distance_to(self, px, py):
        return math.sqrt((self.x - px)**2 + (self.y - py)**2)

    def to_dict(self):
        return {
            'x': round(self.x, 3), 'y': round(self.y, 3),
            'vx': round(float(self.body.velocity.x), 3),
            'vy': round(float(self.body.velocity.y), 3),
            'mass': self.mass,
        }


class PhysicsWorld:
    WIDTH = 20.0
    HEIGHT = 20.0
    DT = 0.05
    LIMB_STIFFNESS = 200.0
    LIMB_DAMPING = 10.0
    THRUST_SCALE = 15.0

    GRIP_CONTACT_DIST = 1.0
    GRIP_ENERGY_COST = 0.003

    def __init__(self, environment, organism, num_objects=3, seed=None):
        self.env = environment
        self.org = organism
        self.org.physics_mode = True
        self.rng = np.random.RandomState(seed)

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.6

        self.persistent = False

        self._create_walls()
        self._create_organism_body()
        self._create_objects(num_objects)
        self.compound_objects = []

        self.contact_count = 0
        self.contact_force = 0.0
        self.grip_joints = {}
        self._setup_collision_handlers()

    def add_compound_objects(self, types=None):
        if types is None:
            types = ['lever', 'spring_gate', 'hinged_barrier']
        for i, obj_type in enumerate(types):
            x = self.rng.uniform(4, 16)
            y = self.rng.uniform(4, 16)
            co = CompoundObject(self.space, x, y, obj_type, seed=self.rng.randint(0, 100000))
            self.compound_objects.append(co)

    def enable_persistence(self):
        self.persistent = True
        self._saved_state = None

    def save_world_state(self):
        state = {
            'objects': [(float(o.body.position.x), float(o.body.position.y),
                         float(o.body.angle), float(o.body.velocity.x),
                         float(o.body.velocity.y)) for o in self.rigid_objects],
            'compounds': [{'bodies': [(float(b.position.x), float(b.position.y),
                                        float(b.angle)) for b in co.bodies]}
                           for co in self.compound_objects],
        }
        self._saved_state = state
        return state

    def restore_world_state(self, state=None):
        if state is None:
            state = self._saved_state
        if state is None:
            return
        for i, obj in enumerate(self.rigid_objects):
            if i < len(state['objects']):
                x, y, angle, vx, vy = state['objects'][i]
                obj.body.position = (x, y)
                obj.body.angle = angle
                obj.body.velocity = (vx, vy)
        for i, co in enumerate(self.compound_objects):
            if i < len(state.get('compounds', [])):
                for j, body in enumerate(co.bodies):
                    if j < len(state['compounds'][i]['bodies']):
                        x, y, angle = state['compounds'][i]['bodies'][j]
                        body.position = (x, y)
                        body.angle = angle
                        body.velocity = (0, 0)

    def _create_walls(self):
        walls = [
            ((0, 0), (self.WIDTH, 0)),
            ((self.WIDTH, 0), (self.WIDTH, self.HEIGHT)),
            ((self.WIDTH, self.HEIGHT), (0, self.HEIGHT)),
            ((0, self.HEIGHT), (0, 0)),
        ]
        for a, b in walls:
            seg = pymunk.Segment(self.space.static_body, a, b, 0.5)
            seg.friction = 0.8
            seg.elasticity = 0.3
            seg.collision_type = COLLISION_TYPE_WALL
            self.space.add(seg)

    def _create_organism_body(self):
        mass = self.org.MASS
        radius = self.org.BODY_RADIUS
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.org_body = pymunk.Body(mass, moment)
        self.org_body.position = (self.org.x, self.org.y)
        self.org_body.angle = self.org.heading
        self.org_shape = pymunk.Circle(self.org_body, radius)
        self.org_shape.friction = 0.5
        self.org_shape.elasticity = 0.2
        self.org_shape.collision_type = COLLISION_TYPE_ORGANISM
        self.space.add(self.org_body, self.org_shape)

        self.limb_bodies = []
        self.limb_shapes = []
        self.limb_joints = []

        for i in range(self.org.NUM_LIMBS):
            limb_mass = 0.1
            limb_radius = 0.1
            moment = pymunk.moment_for_circle(limb_mass, 0, limb_radius)
            lb = pymunk.Body(limb_mass, moment)

            angle = self.org.heading + self.org.BASE_ANGLES[i % len(self.org.SEGMENT_BASE_ANGLES)]
            dist = self.org.LIMB_LEN_RETRACTED
            lb.position = (
                self.org.x + dist * math.cos(angle),
                self.org.y + dist * math.sin(angle),
            )

            ls = pymunk.Circle(lb, limb_radius)
            ls.friction = 0.6
            ls.elasticity = 0.1
            ls.collision_type = COLLISION_TYPE_LIMB
            ls.limb_index = i

            spring = pymunk.DampedSpring(
                self.org_body, lb,
                (0, 0), (0, 0),
                rest_length=dist,
                stiffness=self.LIMB_STIFFNESS,
                damping=self.LIMB_DAMPING,
            )

            self.space.add(lb, ls, spring)
            self.limb_bodies.append(lb)
            self.limb_shapes.append(ls)
            self.limb_joints.append(spring)

    def _create_objects(self, num_objects):
        self.rigid_objects = []
        for _ in range(num_objects):
            x = self.rng.uniform(3, 17)
            y = self.rng.uniform(3, 17)
            mass = self.rng.uniform(0.5, 5.0)
            radius = self.rng.uniform(0.4, 0.8)
            friction = self.rng.uniform(0.3, 0.8)
            obj = RigidObject(self.space, x, y, mass, radius, friction)
            self.rigid_objects.append(obj)

    def _setup_collision_handlers(self):
        self._contact_data = {'count': 0, 'force': 0.0}

        def limb_object_begin(arbiter, space, data):
            self._contact_data['count'] += 1

        def limb_object_separate(arbiter, space, data):
            self._contact_data['count'] = max(0, self._contact_data['count'] - 1)

        def limb_object_post(arbiter, space, data):
            force = arbiter.total_impulse.length / self.DT
            self._contact_data['force'] = max(self._contact_data['force'], force)

        self.space.on_collision(
            collision_type_a=COLLISION_TYPE_LIMB,
            collision_type_b=COLLISION_TYPE_OBJECT,
            begin=limb_object_begin,
            separate=limb_object_separate,
            post_solve=limb_object_post,
        )

    def apply_organism_forces(self, actions):
        muscle_actions = actions[:self.org.NUM_LIMBS * 3]
        acts = muscle_actions.reshape(self.org.NUM_LIMBS, 3)

        for i in range(self.org.NUM_LIMBS):
            extend, flex_l, flex_r = int(acts[i, 0]), int(acts[i, 1]), int(acts[i, 2])

            seg_idx = i // self.org.limbs_per_segment
            local_idx = i % self.org.limbs_per_segment
            base_angle = self.org.SEGMENT_BASE_ANGLES[local_idx]
            dev = self.org.limb_angles[i] - self.org.BASE_ANGLES[i]
            world_angle = self.org_body.angle + base_angle + dev

            if extend:
                self.limb_joints[i].rest_length = self.org.LIMB_LEN_EXTENDED
                fatigue_factor = 1.0 - self.org.fatigue[i] * 0.8
                energy_factor = 0.3 + 0.7 * self.org.energy
                force_mag = self.THRUST_SCALE * fatigue_factor * energy_factor
                fx = -force_mag * math.cos(world_angle)
                fy = -force_mag * math.sin(world_angle)
                self.org_body.apply_force_at_local_point((fx, fy), (0, 0))
            else:
                self.limb_joints[i].rest_length = self.org.LIMB_LEN_RETRACTED

            if flex_l:
                self.org_body.torque += 4.0
            if flex_r:
                self.org_body.torque -= 4.0

    def check_grips(self, actions):
        muscle_actions = actions[:self.org.NUM_LIMBS * 3]
        acts = muscle_actions.reshape(self.org.NUM_LIMBS, 3)

        for i in range(self.org.NUM_LIMBS):
            extend = int(acts[i, 0])
            lb = self.limb_bodies[i]
            lx, ly = float(lb.position.x), float(lb.position.y)

            if extend and i not in self.grip_joints:
                for j, obj in enumerate(self.rigid_objects):
                    dist = math.sqrt((lx - obj.x)**2 + (ly - obj.y)**2)
                    if dist < self.GRIP_CONTACT_DIST:
                        joint = pymunk.PinJoint(lb, obj.body, (0, 0), (0, 0))
                        self.space.add(joint)
                        self.grip_joints[i] = (joint, j)
                        break

            elif not extend and i in self.grip_joints:
                joint, _ = self.grip_joints[i]
                self.space.remove(joint)
                del self.grip_joints[i]

        grip_count = len(self.grip_joints)
        self.org.energy = max(0.0, self.org.energy - self.GRIP_ENERGY_COST * grip_count)

    def get_grip_state(self):
        state = [0] * self.org.NUM_LIMBS
        total_mass = 0.0
        for limb_idx, (_, obj_idx) in self.grip_joints.items():
            state[limb_idx] = 1
            total_mass += self.rigid_objects[obj_idx].mass
        carried_mass = min(1.0, total_mass / 10.0)
        return state, carried_mass

    def apply_developmental_changes(self, step):
        """Step 52: body changes over lifetime — growth, maturation, sensitivity shift."""
        if not hasattr(self, '_base_limb_len'):
            self._base_limb_len = self.org.LIMB_LEN_EXTENDED
            self._base_gains = [g for g in self.org.receptor_gain]

        growth_phase = min(1.0, step / 100.0)
        maturation_phase = min(1.0, max(0.0, (step - 100) / 100.0))

        growth_factor = 0.7 + 0.3 * growth_phase
        self.org.LIMB_LEN_EXTENDED = self._base_limb_len * growth_factor

        pain_sensitivity = 1.5 - 0.5 * maturation_phase
        for i in range(self.org.NUM_LIMBS):
            self.org.receptor_gain[i] = max(
                self.org.GAIN_MIN,
                min(self.org.GAIN_MAX, self._base_gains[i] * pain_sensitivity)
            )

    def step(self):
        self._contact_data['force'] = 0.0
        self.space.step(self.DT)
        self._sync_to_organism()

    def _sync_to_organism(self):
        pos = self.org_body.position
        self.org.x = float(pos.x)
        self.org.y = float(pos.y)
        self.org.heading = float(self.org_body.angle)

        vel = self.org_body.velocity
        self.org.vx = float(vel.x)
        self.org.vy = float(vel.y)
        self.org.omega = float(self.org_body.angular_velocity)

        self.contact_count = min(self.org.NUM_LIMBS,
                                  self._contact_data['count'])
        self.contact_force = min(1.0,
                                  self._contact_data['force'] / 50.0)

        grip_state, carried_mass = self.get_grip_state()
        self.org.grip_state = grip_state
        self.org.carried_mass = carried_mass
        self.org.contact_count = self.contact_count / self.org.NUM_LIMBS
        self.org.contact_force = self.contact_force

    def get_limb_tip_positions(self):
        tips = []
        for lb in self.limb_bodies:
            tips.append((float(lb.position.x), float(lb.position.y)))
        return tips

    def get_object_positions(self):
        return [(obj.x, obj.y) for obj in self.rigid_objects]

    def reset(self, rng=None):
        if rng is None:
            rng = self.rng
        self.org_body.position = (self.org.x, self.org.y)
        self.org_body.angle = self.org.heading
        self.org_body.velocity = (0, 0)
        self.org_body.angular_velocity = 0

        for i, lb in enumerate(self.limb_bodies):
            angle = self.org.heading + self.org.BASE_ANGLES[i % len(self.org.SEGMENT_BASE_ANGLES)]
            dist = self.org.LIMB_LEN_RETRACTED
            lb.position = (
                self.org.x + dist * math.cos(angle),
                self.org.y + dist * math.sin(angle),
            )
            lb.velocity = (0, 0)
            self.limb_joints[i].rest_length = dist

        for obj in self.rigid_objects:
            obj.body.position = (rng.uniform(3, 17), rng.uniform(3, 17))
            obj.body.velocity = (0, 0)
            obj.body.angular_velocity = 0

        self._contact_data = {'count': 0, 'force': 0.0}
        self.contact_count = 0
        self.contact_force = 0.0

    def to_dict(self):
        return {
            'organism': {
                'x': round(self.org.x, 3),
                'y': round(self.org.y, 3),
                'heading': round(self.org.heading, 3),
            },
            'limb_tips': [(round(float(lb.position.x), 3), round(float(lb.position.y), 3))
                          for lb in self.limb_bodies],
            'objects': [obj.to_dict() for obj in self.rigid_objects],
            'compounds': [co.to_dict() for co in self.compound_objects],
            'contact_count': self.contact_count,
            'contact_force': round(self.contact_force, 3),
            'persistent': self.persistent,
        }


if __name__ == '__main__':
    print("=== Step 48: Physics World Test ===\n")

    env = Environment(seed=42)
    org = Organism()
    org.reset()
    npc = NPC()
    npc.reset()

    pw = PhysicsWorld(env, org, num_objects=4, seed=42)
    print(f"  Space bodies: {len(pw.space.bodies)}")
    print(f"  Rigid objects: {len(pw.rigid_objects)}")
    print(f"  Limb bodies: {len(pw.limb_bodies)}")

    total_reward = 0
    contacts_detected = 0

    for step in range(200):
        npc.step(env, step)
        actions = org.compute_optimal_actions(env, step, npc=npc)
        pw.apply_organism_forces(actions)
        pw.check_grips(actions)
        pw.step()
        obs, reward = org.step(actions, env, step, npc=npc)
        total_reward += reward

        if pw.contact_count > 0:
            contacts_detected += 1

        if step % 50 == 0:
            grip_state, carried = pw.get_grip_state()
            grips = sum(grip_state)
            print(f"  Step {step:3d}: org=({org.x:.2f},{org.y:.2f}) "
                  f"contacts={pw.contact_count} grips={grips} carried={carried:.2f} "
                  f"reward={reward:+.2f}")
            for i, obj in enumerate(pw.rigid_objects):
                print(f"    Object {i}: ({obj.x:.2f},{obj.y:.2f}) mass={obj.mass:.1f}")

    print(f"\n  Total reward: {total_reward:.1f}")
    print(f"  Contact steps: {contacts_detected}/200")
    print(f"  Final organism: ({org.x:.2f},{org.y:.2f})")
    print(f"  Final objects: {[(round(o.x,1), round(o.y,1)) for o in pw.rigid_objects]}")

    org2 = Organism()
    org2.reset()
    env2 = Environment(seed=99)
    npc2 = NPC()
    npc2.reset()
    reward_no_physics = 0
    for step in range(200):
        npc2.step(env2, step)
        actions = org2.compute_optimal_actions(env2, step, npc=npc2)
        obs, reward = org2.step(actions, env2, step, npc=npc2)
        reward_no_physics += reward
    print(f"\n  Reward without physics: {reward_no_physics:.1f}")
    print(f"  Reward with physics:    {total_reward:.1f}")

    assert len(pw.rigid_objects) == 4
    assert len(pw.limb_bodies) == org.NUM_LIMBS
    print("\n  Step 48 PASSED: Physics world operational")

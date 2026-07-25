"""Single-receptor elicitation: necessity_detection.

Design an environment where survival depends on detecting that a
relationship MUST hold — where alternatives have been eliminated
and the remaining option is the only possibility.

The environment: 4 zones, 3 are traps (delayed pain), 1 is safe (endorphin).
Which zone is safe rotates, but the organism can determine it by
elimination — visiting 3 traps identifies the 4th as necessarily safe.
The organism that detects necessity (visits 3, knows the 4th must be safe
without visiting) outperforms the organism that must visit all 4.

This is exhaustive_search + counterfactual_reasoning → necessity_detection.
"""

import os
import json
import numpy as np
from environment import Environment, Organism, NPC
from mental_model import build_mental_model
from receptor_discovery import build_tests, calibrate_null_thresholds, discover
from model import compute_obs_indices
from train import train_model, EXPLORE_RATE, PROBE_RATE_FLOOR


class NecessityEnvironment:
    """4 zones, 3 traps + 1 safe. Which is safe rotates every episode."""

    def __init__(self, base_env, safe_zone, seed=None):
        self.env = base_env
        self.rng = np.random.RandomState(seed)
        self.safe_zone = safe_zone
        self.zones = []
        for i in range(4):
            angle = i * np.pi / 2
            x = 10 + 6 * np.cos(angle)
            y = 10 + 6 * np.sin(angle)
            self.zones.append({
                'x': x, 'y': y, 'radius': 2.5,
                'is_safe': (i == safe_zone),
                'visited': False,
                'visit_step': -1,
            })
        self.total_reward = 0

    def step(self, org, step):
        reward = 0
        for i, zone in enumerate(self.zones):
            dx = org.x - zone['x']
            dy = org.y - zone['y']
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < zone['radius'] and not zone['visited']:
                zone['visited'] = True
                zone['visit_step'] = step
                if zone['is_safe']:
                    reward += 5.0
                else:
                    reward -= 2.0

        # Bonus: if organism goes directly to safe zone after visiting
        # exactly 3 traps (detected necessity)
        visited_traps = sum(1 for z in self.zones if z['visited'] and not z['is_safe'])
        visited_safe = any(z['visited'] and z['is_safe'] for z in self.zones)
        if visited_traps == 3 and visited_safe:
            safe_step = next(z['visit_step'] for z in self.zones if z['is_safe'])
            last_trap_step = max(z['visit_step'] for z in self.zones
                                 if z['visited'] and not z['is_safe'])
            if safe_step > last_trap_step:
                # Organism visited safe zone AFTER visiting 3 traps
                # Bonus scales inversely with gap — faster = more necessity-like
                gap = safe_step - last_trap_step
                if gap < 20:
                    reward += 3.0 * (1 - gap / 20)

        self.total_reward += reward
        return reward

    def modify_field(self, points, t):
        """Add pain at trap zones, endorphin at safe zone."""
        pain_add = np.zeros(len(points))
        endo_add = np.zeros(len(points))
        for zone in self.zones:
            for i, (px, py) in enumerate(points):
                dx, dy = px - zone['x'], py - zone['y']
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < zone['radius']:
                    intensity = np.exp(-dist*dist / (2 * 1.5**2))
                    if zone['is_safe']:
                        endo_add[i] += 3.0 * intensity
                    else:
                        if zone['visited']:
                            pain_add[i] += 2.0 * intensity
        return pain_add, endo_add


def run_elicitation(num_episodes=200, steps_per_episode=200, seed=42):
    print("=" * 60)
    print("SINGLE-RECEPTOR ELICITATION: necessity_detection")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    all_log = []

    for ep in range(num_episodes):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        safe_zone = ep % 4
        nec_env = NecessityEnvironment(env, safe_zone, seed=rng.randint(0, 100000))

        for step in range(steps_per_episode):
            npc.step(env, step)
            actions = org.compute_optimal_actions(env, step, npc=npc)

            r = rng.random()
            if r < PROBE_RATE_FLOOR:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < EXPLORE_RATE:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = actions

            obs, reward = org.step(executed, env, step, npc=npc)
            nec_reward = nec_env.step(org, step)

        all_log.extend(org.experience_log)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{num_episodes}")

    print(f"  Log: {len(all_log)} entries")
    print("  Building mental model...")
    engine = build_mental_model(all_log)

    print("  Calibrating null thresholds...")
    null_thresh = calibrate_null_thresholds(all_log, engine, num_shuffles=5)

    print("  Running discovery...")
    results = discover(all_log, engine, threshold_overrides=null_thresh,
                       log_provenance='oracle')

    target_receptors = ['necessity_detection', 'exhaustive_search',
                        'counterfactual_reasoning', 'quantity_detection',
                        'structural_invariance_math', 'ratio_detection']

    print(f"\n  Target receptors:")
    for r in target_receptors:
        score = results['scores'].get(r, '?')
        thresh = null_thresh.get(r, '?')
        found = r in results['discovered']
        print(f"    {r}: score={score}, thresh={thresh}, found={found}")

    total_discovered = len(results['discovered'])
    print(f"\n  Total discovered: {total_discovered}")
    print(f"  necessity_detection: {'FOUND' if 'necessity_detection' in results['discovered'] else 'NOT FOUND'}")

    # Save
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'elicitation_necessity.json'), 'w') as f:
        json.dump({
            'target': 'necessity_detection',
            'episodes': num_episodes,
            'discovered': results['discovered'],
            'scores': {k: float(v) if isinstance(v, (int, float)) else str(v)
                       for k, v in results['scores'].items()},
            'target_found': 'necessity_detection' in results['discovered'],
        }, f, indent=2)
    print(f"  Saved to data/elicitation_necessity.json")

    return results


if __name__ == '__main__':
    run_elicitation(num_episodes=200, steps_per_episode=200, seed=42)

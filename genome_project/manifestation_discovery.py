"""
Manifestation-Targeted Receptor Discovery

Runs the full deep time pipeline (TieredEnvironment + PhysicsWorld +
CombinedT7T8Environment + mental model + thinking substrate + evolutionary
selection) with a manifestation environment plugged in.

The environment factory returns a TieredEnvironment at the receptor's
assigned tier. run_generation_rich wraps it with PhysicsWorld and T7/T8.
The organism evolves through generations, building its mental model from
experience, with MCTS evaluating actions.

Usage:
    python -m genome_project.manifestation_discovery tool_use --gens 10
    python -m genome_project.manifestation_discovery --tier 3 --gens 5
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genome_project.manifester import ManifestationRegistry, RECEPTOR_TIER_MAP
from deep_time_overnight import run_overnight

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def run_manifestation_deep_time(receptor_id, num_generations=10,
                                 population_size=4, num_episodes=5,
                                 steps_per_episode=200, seed=42):
    """Run deep time evolution in a manifestation environment.

    Uses run_overnight from deep_time_overnight.py — the real pipeline with
    PhysicsWorld, CombinedT7T8Environment, mental model, thinking substrate,
    and evolutionary selection.
    """
    registry = ManifestationRegistry()
    envs = registry.get_environments(receptor_id)
    if not envs:
        print(f"No recipes for {receptor_id}")
        return None

    cfg = envs[0]
    tier = cfg.tier

    def env_factory(seed=None):
        return cfg.build(seed=seed)

    print(f"Receptor: {receptor_id}")
    print(f"Environment: TieredEnvironment(tier={tier}) + PhysicsWorld + T7/T8")
    print(f"Generations: {num_generations}, Population: {population_size}")
    print(f"Episodes/gen: {num_episodes}, Steps/episode: {steps_per_episode}")
    print()

    t0 = time.time()
    run_overnight(
        num_generations=num_generations,
        population_size=population_size,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        bootstrap_episodes=max(5, num_episodes * 2),
        epochs_per_gen=5,
        tier=tier,
        seed=seed,
        resume=False,
        env_factory=env_factory,
    )
    elapsed = time.time() - t0

    print(f"\nElapsed: {elapsed / 60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Manifestation deep time discovery')
    parser.add_argument('receptors', nargs='*', help='Receptor IDs to test')
    parser.add_argument('--tier', type=int, help='Run all receptors at this tier')
    parser.add_argument('--gens', type=int, default=10)
    parser.add_argument('--pop', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.receptors:
        targets = args.receptors
    elif args.tier is not None:
        targets = [r for r, t in RECEPTOR_TIER_MAP.items() if t == args.tier]
    else:
        parser.print_help()
        return

    for receptor_id in targets:
        run_manifestation_deep_time(
            receptor_id,
            num_generations=args.gens,
            population_size=args.pop,
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()

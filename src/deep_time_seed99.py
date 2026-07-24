"""Second seed deep time run (seed 99) for replication.

Tests whether the depth_reached prerequisite pattern replicates:
  - Do metacognition and conflation emerge before depth_reached?
  - Does depth_reached activate at all?
  - At what generation?

Uses the same parameters as the seed 42 run but different seed.
"""

from deep_time_overnight import run_overnight

if __name__ == '__main__':
    history = run_overnight(
        num_generations=40,
        population_size=4,
        num_episodes=5,
        steps_per_episode=200,
        bootstrap_episodes=20,
        epochs_per_gen=8,
        tier=4,
        seed=99,
        resume=True,
    )

    # Check for depth_reached and prerequisites
    print("\n" + "=" * 60)
    print("REPLICATION CHECK: depth_reached prerequisites")
    print("=" * 60)

    meta_gen = None
    conflation_gen = None
    depth_gen = None

    for rec in history:
        discovered = set(rec.get('discovered', []))
        if 'metacognition' in discovered and meta_gen is None:
            meta_gen = rec['generation']
        if 'conflation' in discovered and conflation_gen is None:
            conflation_gen = rec['generation']
        th = rec.get('thinking', {})
        if th.get('channels', {}).get('depth_reached', 0) > 0.01 and depth_gen is None:
            depth_gen = rec['generation']

    print(f"  metacognition first appeared: gen {meta_gen}")
    print(f"  conflation first appeared:    gen {conflation_gen}")
    print(f"  depth_reached activated:      gen {depth_gen}")

    if meta_gen and conflation_gen and depth_gen:
        if depth_gen > max(meta_gen, conflation_gen):
            print(f"\n  REPLICATES: depth ({depth_gen}) after prerequisites "
                  f"({max(meta_gen, conflation_gen)})")
        else:
            print(f"\n  DOES NOT REPLICATE: depth ({depth_gen}) before or same as "
                  f"prerequisites")
    elif depth_gen is None:
        print(f"\n  depth_reached did not activate in {len(history)} generations")
        if meta_gen and conflation_gen:
            print(f"  Prerequisites present (meta={meta_gen}, conflation={conflation_gen})")
            print(f"  May need more generations or different environmental pressure")
        else:
            missing = []
            if meta_gen is None:
                missing.append("metacognition")
            if conflation_gen is None:
                missing.append("conflation")
            print(f"  Prerequisites missing: {missing}")

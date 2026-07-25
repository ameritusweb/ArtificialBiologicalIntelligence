"""Motor store anxiety loop experiment.

Paired-seed comparison: 10 generations with vs without motor store,
1000 steps per episode. Measures whether shortcuts break the
pain<->conflict anxiety loop.

Optimizations applied:
- Parallel conditions via multiprocessing
- deque(maxlen=60000) instead of unbounded lists (9 GB -> 1.4 GB)
- Warm-start model between generations
- CPU inference during step loop
- Embedding cache in MCTS (ThinkingNode._embedding)
- Gate extract_sequences to once per episode
- Re-embed motor store after encoder rebuild
- Eviction staleness scaled to run length
"""

import os
import sys
import json
import numpy as np
import torch
from collections import deque
from multiprocessing import Process, Queue
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash
from model import compute_obs_indices, HierarchicalPolicy
from train import train_model, generate_training_data, EXPLORE_RATE, PROBE_RATE_FLOOR
from thinking_substrate import ThinkingTree
from cognitive_state import CognitiveStateDetector
from deep_time import EvolvingOrganism, select_and_reproduce
from receptor_coactivation import CoactivationLogger
from procedural_memory import (PeakExperienceIndex, ReplayEngine,
                               MotorSequenceStore, ShortcutExecutor)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _warm_train(model, X, Y, Z, epochs, steps_per_episode, idx):
    """Warm-start training: reuse existing weights, fewer epochs."""
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    nc = idx.get('num_continuous', 0)
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    bce = nn.BCEWithLogitsLoss()

    N = X.shape[0]
    batch_size = 256
    train_idx = np.arange(N)

    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        model.train()
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            bi = train_idx[start:start + batch_size]
            xb = torch.FloatTensor(X[bi]).to(device)
            yb = torch.FloatTensor(Y[bi]).to(device)
            zb = torch.FloatTensor(Z[bi]).to(device)

            result = model(xb)
            if nc > 0:
                blend_loss = (F.mse_loss(result['blended'][:, :nc], yb[:, :nc])
                              + bce(result['blended'][:, nc:], yb[:, nc:]))
            else:
                blend_loss = bce(result['blended'], yb)
            pred_loss = F.mse_loss(result['predicted_next_pain'], zb)
            loss = blend_loss + 0.2 * pred_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

    model.eval()
    return model


def run_condition(condition_name, use_motor_store, num_generations=10,
                  population_size=4, num_episodes=5, steps_per_episode=1000,
                  seed=42, result_queue=None):
    print(f"\n{'=' * 60}")
    print(f"CONDITION: {condition_name} (motor_store={use_motor_store})")
    print(f"  {num_generations} gens, {steps_per_episode} steps/ep, seed={seed}")
    print(f"{'=' * 60}")

    rng = np.random.RandomState(seed)
    idx = compute_obs_indices()
    num_actions = idx['num_actions']
    core_obs_dim = idx['core_obs_dim']
    obs_dim = idx['obs_dim']
    num_thinking_channels = idx.get('num_thinking_channels', 6)
    shortcut_threshold = 0.5

    organisms = [EvolvingOrganism(f"gen0_{i}") for i in range(population_size)]
    cumulative_log = []
    history = []

    # Bounded buffers instead of unbounded lists
    max_buffer = 60000
    cumulative_windows = deque(maxlen=max_buffer)
    cumulative_targets = deque(maxlen=max_buffer)
    cumulative_next_pain = deque(maxlen=max_buffer)

    print("  Bootstrapping...")
    X, Y, Z, boot_log = generate_training_data(
        num_episodes=20, steps_per_episode=steps_per_episode, seed=seed)
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
    peak_index = PeakExperienceIndex(max_size=200)
    replay_engine = ReplayEngine(peak_index)

    motor_store = MotorSequenceStore(num_continuous=0) if use_motor_store else None
    shortcut_executor = ShortcutExecutor() if use_motor_store else None

    for gen in range(num_generations):
        print(f"\n--- [{condition_name}] Generation {gen} ---")
        gen_logger = CoactivationLogger(idx)
        total_shortcuts = 0

        # Move model to CPU for batch-1 inference
        model.to('cpu')
        model.eval()

        for evo_org in organisms:
            org = evo_org.create_organism(rng)

            budget = int(evo_org.body_params.get('thinking_budget', 24))
            tree.max_simulations = budget
            v_keys = {k: v for k, v in evo_org.body_params.items()
                      if k.startswith('v_')}
            tree.v_weights = v_keys if v_keys else None
            thinking_cost = float(evo_org.body_params.get('thinking_cost', 0.001))

            npc = NPC()
            npc.reset(rng)
            env = Environment(seed=rng.randint(0, 100000))
            prev_action_hash = 0

            for ep in range(num_episodes):
                ep_did_extract = False

                for step in range(steps_per_episode):
                    npc.step(env, step)
                    obs_before = org.history[-1].copy() if org.history else np.zeros(obs_dim)

                    # Cognitive state
                    tt, ca = cog_detector.update(obs_before, engine, prev_action_hash)
                    org.thought_type_id = tt
                    org.concept_id = ca

                    shortcut_fired = False

                    # Motor store shortcut check
                    if use_motor_store and shortcut_executor is not None:
                        if shortcut_executor.is_active():
                            if shortcut_executor.should_abort(obs_before):
                                shortcut_executor.finish(motor_store, gen)
                            elif shortcut_executor.is_complete():
                                shortcut_executor.finish(motor_store, gen)
                            else:
                                executed = shortcut_executor.get_action()
                                if executed is not None:
                                    org.thinking_channels = np.zeros(num_thinking_channels)
                                    obs, reward = org.step(executed, env, step, npc=npc)
                                    shortcut_executor.add_reward(reward)
                                    evo_org.fitness += reward
                                    gen_logger.record(obs)
                                    prev_action_hash = action_to_hash(executed)
                                    total_shortcuts += 1
                                    shortcut_fired = True

                        if not shortcut_fired and not shortcut_executor.is_active():
                            embedding = engine.encoder.embed(obs_before[:core_obs_dim])
                            match, score = motor_store.query(tt, embedding, min_support=3)
                            if match is not None and score > shortcut_threshold:
                                shortcut_executor.start(match, obs_before)
                                executed = shortcut_executor.get_action()
                                if executed is not None:
                                    org.thinking_channels = np.zeros(num_thinking_channels)
                                    obs, reward = org.step(executed, env, step, npc=npc)
                                    shortcut_executor.add_reward(reward)
                                    evo_org.fitness += reward
                                    gen_logger.record(obs)
                                    prev_action_hash = action_to_hash(executed)
                                    total_shortcuts += 1
                                    shortcut_fired = True

                    if shortcut_fired:
                        continue

                    # MCTS thinking
                    if engine is not None:
                        org.thinking_channels = tree.think(obs_before, engine)
                        org.energy = max(0.0, org.energy - thinking_cost * budget)

                    # Action selection
                    if gen == 0:
                        actions = org.compute_optimal_actions(env, step, npc=npc)
                        executed = actions
                    else:
                        window = org.get_observation_window()
                        policy_action, _ = model.predict(window)
                        optimal = org.compute_optimal_actions(env, step, npc=npc)
                        cumulative_windows.append(window.copy().astype(np.float32))
                        cumulative_targets.append(optimal.copy().astype(np.float32))
                        executed = policy_action

                    r = rng.random()
                    if r < PROBE_RATE_FLOOR:
                        executed = np.zeros(num_actions, dtype=np.int32)
                    elif r < EXPLORE_RATE:
                        executed = rng.randint(0, 2, size=num_actions).astype(np.int32)

                    obs, reward = org.step(executed, env, step, npc=npc)
                    evo_org.fitness += reward
                    gen_logger.record(obs)
                    prev_action_hash = action_to_hash(executed)

                    # Peak indexing
                    reward_delta = reward - (float(np.mean(obs_before[:6]))
                                             if len(obs_before) >= 6 else 0)
                    if reward_delta > 0.5:
                        peak_index.add(reward_delta,
                                       len(cumulative_log) + len(org.experience_log),
                                       executed, obs_before)

                    # Replay (extract sequences gated to once per episode)
                    if replay_engine.should_replay(obs, step):
                        replay_engine.replay(engine, step)
                        if motor_store is not None and not ep_did_extract:
                            motor_store.extract_sequences(
                                org.experience_log, peak_index,
                                engine.encoder, cog_detector, core_obs_dim)
                            ep_did_extract = True

                # End of episode: next-pain targets
                ep_pain = [e['obs_after'][0:6].copy()
                           for e in org.experience_log[-steps_per_episode:]]
                for i in range(len(ep_pain)):
                    next_p = ep_pain[i + 1] if i + 1 < len(ep_pain) else ep_pain[-1]
                    cumulative_next_pain.append(next_p.astype(np.float32))

            cumulative_log.extend(org.experience_log)

        # Move model back to GPU for training
        model.to(DEVICE)

        # Warm-start retrain (3 epochs instead of 8 from scratch)
        if gen > 0 and len(cumulative_windows) >= 100:
            X = np.array(list(cumulative_windows), dtype=np.float32)
            Y = np.array(list(cumulative_targets), dtype=np.float32)
            Z = np.array(list(cumulative_next_pain), dtype=np.float32)
            model = _warm_train(model, X, Y, Z, epochs=3,
                                steps_per_episode=steps_per_episode, idx=idx)

        # Rebuild mental model
        log_slice = cumulative_log[-max_buffer:]
        engine = build_mental_model(log_slice)

        # Re-embed motor store entries with new encoder
        if motor_store is not None:
            motor_store.re_embed_all(engine.encoder, core_obs_dim)
            motor_store.evict_stale(gen, max_staleness=5)

        # Co-activation stats
        stats = gen_logger.get_stats()
        if stats is None:
            continue

        pain_conflict_lift = 0
        pc_forward = 0
        cp_forward = 0
        for p in stats.get('top_coactivations', []):
            if (p['a'] == 'pain' and p['b'] == 'conflict') or \
               (p['a'] == 'conflict' and p['b'] == 'pain'):
                pain_conflict_lift = p['lift']
        for s in stats.get('top_sequences', []):
            if s['from'] == 'pain' and s['to'] == 'conflict':
                pc_forward = s['lift']
            if s['from'] == 'conflict' and s['to'] == 'pain':
                cp_forward = s['lift']
        anxiety_loop = pc_forward > 1.5 and cp_forward > 1.5

        ms_stats = motor_store.get_stats() if motor_store else {}
        avg_fitness = np.mean([o.fitness for o in organisms])

        rec = {
            'generation': gen,
            'condition': condition_name,
            'total_steps': stats['total_steps'],
            'unique_patterns': stats['n_unique_patterns'],
            'pain_conflict_lift': round(pain_conflict_lift, 3),
            'anxiety_loop': anxiety_loop,
            'pain_to_conflict_lift': round(pc_forward, 3),
            'conflict_to_pain_lift': round(cp_forward, 3),
            'shortcuts_fired': total_shortcuts,
            'motor_entries': ms_stats.get('total_entries', 0),
            'motor_types': ms_stats.get('num_types', 0),
            'motor_success_rate': round(ms_stats.get('success_rate', 0), 3),
            'avg_fitness': round(float(avg_fitness), 2),
        }
        history.append(rec)

        ms_str = ""
        if motor_store:
            ms_str = (f"  Motor: {ms_stats['total_entries']} seqs, "
                      f"{total_shortcuts} shortcuts, "
                      f"success={ms_stats['success_rate']:.2f}")

        print(f"  Steps: {stats['total_steps']}  Patterns: {stats['n_unique_patterns']}  "
              f"Fitness: {avg_fitness:.1f}")
        print(f"  Pain<->Conflict: coact={pain_conflict_lift:.2f}  "
              f"P->C={pc_forward:.2f}  C->P={cp_forward:.2f}  "
              f"loop={'YES' if anxiety_loop else 'no'}")
        if ms_str:
            print(ms_str)

        # Reproduce
        if gen < num_generations - 1:
            organisms = select_and_reproduce(organisms, population_size, rng)
            for i, org_evo in enumerate(organisms):
                org_evo.organism_id = f"gen{gen+1}_{i}"

    if result_queue is not None:
        result_queue.put((condition_name, history))
    return history


def _run_condition_worker(condition_name, use_motor_store, seed, result_queue):
    """Wrapper for multiprocessing."""
    try:
        run_condition(condition_name, use_motor_store, seed=seed,
                      result_queue=result_queue)
    except Exception as e:
        print(f"ERROR in {condition_name}: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((condition_name, []))


def run_experiment(seed=42, parallel=True):
    print("MOTOR STORE ANXIETY LOOP EXPERIMENT")
    print(f"Paired seed={seed}, 1000 steps/episode, 10 generations")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    if parallel:
        q = Queue()
        p1 = Process(target=_run_condition_worker,
                      args=("control", False, seed, q))
        p2 = Process(target=_run_condition_worker,
                      args=("motor_store", True, seed, q))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        results_map = {}
        while not q.empty():
            name, hist = q.get()
            results_map[name] = hist
        control = results_map.get('control', [])
        treatment = results_map.get('motor_store', [])
    else:
        control = run_condition("control", use_motor_store=False, seed=seed)
        treatment = run_condition("motor_store", use_motor_store=True, seed=seed)

    print("\n" + "=" * 70)
    print("COMPARISON: Control vs Motor Store")
    print("=" * 70)
    print(f"\n{'Gen':>4} {'Ctrl Loop':>10} {'Ctrl P->C':>10} {'Ctrl C->P':>10}"
          f" | {'MS Loop':>8} {'MS P->C':>8} {'MS C->P':>8} {'Shortcuts':>10}")
    print("-" * 80)

    ctrl_loops = 0
    ms_loops = 0
    for c, t in zip(control, treatment):
        g = c['generation']
        c_loop = 'YES' if c['anxiety_loop'] else 'no'
        t_loop = 'YES' if t['anxiety_loop'] else 'no'
        if c['anxiety_loop']:
            ctrl_loops += 1
        if t['anxiety_loop']:
            ms_loops += 1
        print(f"{g:>4} {c_loop:>10} {c['pain_to_conflict_lift']:>10.2f} "
              f"{c['conflict_to_pain_lift']:>10.2f}"
              f" | {t_loop:>8} {t['pain_to_conflict_lift']:>8.2f} "
              f"{t['conflict_to_pain_lift']:>8.2f} {t['shortcuts_fired']:>10}")

    n = max(len(control), len(treatment))
    print(f"\nAnxiety loop frequency: control={ctrl_loops}/{n}, "
          f"motor_store={ms_loops}/{n}")
    if ms_loops < ctrl_loops:
        print(f"  Motor store REDUCED anxiety loop frequency by "
              f"{ctrl_loops - ms_loops} generations")
    elif ms_loops == ctrl_loops:
        print("  No difference in anxiety loop frequency")
    else:
        print("  Motor store INCREASED anxiety loop frequency (unexpected)")

    results = {'seed': seed, 'control': control, 'motor_store': treatment}
    with open(os.path.join(DATA_DIR, 'motor_store_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/motor_store_experiment.json")

    return results


if __name__ == '__main__':
    parallel = '--sequential' not in sys.argv
    run_experiment(seed=42, parallel=parallel)

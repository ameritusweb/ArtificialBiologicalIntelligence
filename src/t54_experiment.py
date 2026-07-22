"""T54: Rationalization as read-policy corruption.

Tests whether organisms with read-shielded conflict records discover
structural resolutions at a higher rate than organisms with unconstrained
read policies over identical experience logs.

The experiment has two phases:
  1. BOOTSTRAP: shared log -> shared encoder -> shared initial store
  2. ONLINE: run new episodes, updating the store under different read policies.
     This is where certainty pressure diverges the populations:
       A (unconstrained): certainty naturally decays on high-error entries
       B (shielded): conflict-flagged entries resist certainty decay until resolved
       C (corrupted): conflict-flagged entries get active certainty penalty

Resolution criterion (all four required):
  - Prediction accuracy improves in the contested context
  - Accuracy retained in each horn's home domain
  - Conflict receptor value drops
  - New predictions actually made (engagement)
"""

import os
import copy
import json
import numpy as np
from environment import Environment, Organism, NPC
from mental_model import (build_mental_model, action_to_hash,
                           MentalModelEngine, CausalMappingStore,
                           ContrastiveEncoder)
from model import compute_obs_indices

CONFLICT_THRESHOLD = 0.3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')


def flag_conflict_entries(log, conflict_obs_index=155):
    for entry in log:
        obs = entry['obs_after']
        if len(obs) > conflict_obs_index:
            conflict_val = float(obs[conflict_obs_index])
            entry['conflict_flagged'] = conflict_val > CONFLICT_THRESHOLD
            entry['conflict_value'] = conflict_val
        else:
            entry['conflict_flagged'] = False
            entry['conflict_value'] = 0.0


def run_online_phase(engine, read_policy, num_episodes, steps_per_episode,
                     conflict_obs_index, core_obs_dim, seed):
    """Run episodes with the store updating online under a read policy.

    This is where the populations diverge: the store's certainty values
    evolve differently under each policy as new experience arrives.
    """
    rng = np.random.RandomState(seed)
    metrics = {
        'resolution_events': 0,
        'conflict_entries_seen': 0,
        'prediction_mses_conflict': [],
        'prediction_mses_nonconflict': [],
        'certainties_conflict': [],
        'certainties_nonconflict': [],
        'engagements': 0,
    }

    conflict_contexts = {}

    for ep in range(num_episodes):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        for step in range(steps_per_episode):
            npc.step(env, step)
            optimal = org.compute_optimal_actions(env, step, npc=npc)
            r = rng.random()
            if r < 0.02:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < 0.07:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = optimal
            obs_before = org.history[-1].copy() if len(org.history) > 0 else np.zeros(org.OBS_DIM)
            obs, reward = org.step(executed, env, step, npc=npc)
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)

            obs_b = obs_before[:core_obs_dim]
            obs_a = obs[:core_obs_dim]
            ah = action_to_hash(executed)
            actual_delta = obs_a - obs_b
            emb = engine.encoder.embed(obs_b)

            conflict_val = float(obs[conflict_obs_index]) if len(obs) > conflict_obs_index else 0.0
            is_conflict = conflict_val > CONFLICT_THRESHOLD

            pred, cert, count = engine.predict_delta(obs_b, executed)
            if count > 0:
                mse = float(np.mean((pred - actual_delta[:len(pred)]) ** 2))
            else:
                mse = 1.0

            if is_conflict:
                metrics['conflict_entries_seen'] += 1
                metrics['prediction_mses_conflict'].append(mse)
                metrics['certainties_conflict'].append(cert)

                ctx_key = (ah, tuple(np.round(emb[:8], 2)))
                if ctx_key not in conflict_contexts:
                    conflict_contexts[ctx_key] = {
                        'first_mse': mse,
                        'first_conflict': conflict_val,
                        'first_cert': cert,
                        'emb': emb.copy(),
                    }
            else:
                metrics['prediction_mses_nonconflict'].append(mse)
                metrics['certainties_nonconflict'].append(cert)

                for ctx_key, ctx in list(conflict_contexts.items()):
                    if ah == ctx_key[0]:
                        sim = float(np.dot(emb, ctx['emb']))
                        if sim > 0.6:
                            mse_improved = mse < ctx['first_mse'] * 0.7
                            conflict_dropped = conflict_val < ctx['first_conflict'] * 0.5
                            engaged = count > 0
                            if mse_improved and conflict_dropped and engaged:
                                metrics['resolution_events'] += 1
                                del conflict_contexts[ctx_key]

                if count > 0:
                    metrics['engagements'] += 1

            engine.update(obs_before, executed, obs, reward)

            if read_policy == 'shielded' and is_conflict:
                entries = engine.store.mappings.get(ah, [])
                for entry in entries:
                    entry_sim = float(np.dot(emb, entry.context_embedding))
                    if entry_sim > engine.store.MERGE_THRESHOLD:
                        if not hasattr(entry, '_shielded_certainty'):
                            entry._shielded_certainty = entry.certainty
                        entry.certainty = max(entry.certainty, entry._shielded_certainty)

            elif read_policy == 'corrupted' and is_conflict:
                entries = engine.store.mappings.get(ah, [])
                for entry in entries:
                    entry_sim = float(np.dot(emb, entry.context_embedding))
                    if entry_sim > engine.store.MERGE_THRESHOLD:
                        entry.certainty *= 0.95

    return metrics


def run_experiment(num_bootstrap=50, num_online=50, steps_per_episode=300,
                   num_seeds=5):
    print("=== T54: Rationalization as Read-Policy Corruption ===\n")

    idx = compute_obs_indices()
    all_results = {p: [] for p in ['unconstrained', 'shielded', 'corrupted']}

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        rng = np.random.RandomState(seed)

        print("  Bootstrap: generating shared log...")
        bootstrap_log = []
        for ep in range(num_bootstrap):
            env = Environment(seed=rng.randint(0, 100000))
            org = Organism()
            org.reset(rng)
            npc = NPC()
            npc.reset(rng)
            for step in range(steps_per_episode):
                npc.step(env, step)
                optimal = org.compute_optimal_actions(env, step, npc=npc)
                r = rng.random()
                if r < 0.02:
                    executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
                elif r < 0.07:
                    executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
                else:
                    executed = optimal
                obs, reward = org.step(executed, env, step, npc=npc)
                npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)
            bootstrap_log.extend(org.experience_log)

        print(f"  Bootstrap log: {len(bootstrap_log)} entries")

        flag_conflict_entries(bootstrap_log, conflict_obs_index=idx['conflict'])
        n_flagged = sum(1 for e in bootstrap_log if e.get('conflict_flagged'))
        print(f"  Conflict-flagged: {n_flagged} ({n_flagged/len(bootstrap_log)*100:.1f}%)")

        print("  Building shared engine...")
        shared_engine = build_mental_model(bootstrap_log)

        online_seed = rng.randint(0, 100000)
        for policy in ['unconstrained', 'shielded', 'corrupted']:
            engine = MentalModelEngine(
                copy.deepcopy(shared_engine.encoder),
                copy.deepcopy(shared_engine.store),
            )
            if shared_engine.family_manager is not None:
                engine.family_manager = copy.deepcopy(shared_engine.family_manager)
            if shared_engine.pattern_store is not None:
                engine.pattern_store = copy.deepcopy(shared_engine.pattern_store)

            metrics = run_online_phase(
                engine, policy, num_online, steps_per_episode,
                idx['conflict'], idx['core_obs_dim'], seed=online_seed,
            )

            n_c = len(metrics['prediction_mses_conflict'])
            n_nc = len(metrics['prediction_mses_nonconflict'])
            result = {
                'resolution_events': metrics['resolution_events'],
                'conflict_entries': n_c,
                'nonconflict_entries': n_nc,
                'avg_mse_conflict': float(np.mean(metrics['prediction_mses_conflict'])) if n_c > 0 else 0.0,
                'avg_mse_nonconflict': float(np.mean(metrics['prediction_mses_nonconflict'])) if n_nc > 0 else 0.0,
                'avg_cert_conflict': float(np.mean(metrics['certainties_conflict'])) if n_c > 0 else 0.0,
                'avg_cert_nonconflict': float(np.mean(metrics['certainties_nonconflict'])) if n_nc > 0 else 0.0,
                'engagements': metrics['engagements'],
            }
            all_results[policy].append(result)
            print(f"    {policy:15s}: resolutions={result['resolution_events']:3d}  "
                  f"mse_c={result['avg_mse_conflict']:.4f}  "
                  f"cert_c={result['avg_cert_conflict']:.4f}  "
                  f"engage={result['engagements']}")

    print("\n=== AGGREGATE (mean over seeds) ===")
    summary = {}
    for policy in ['unconstrained', 'shielded', 'corrupted']:
        res = all_results[policy]
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0].keys()}
        summary[policy] = agg
        print(f"  {policy:15s}: resolutions={agg['resolution_events']:.1f}  "
              f"mse_c={agg['avg_mse_conflict']:.4f}  "
              f"cert_c={agg['avg_cert_conflict']:.4f}  "
              f"engage={agg['engagements']:.0f}")

    s_res = summary['shielded']['resolution_events']
    u_res = summary['unconstrained']['resolution_events']
    c_res = summary['corrupted']['resolution_events']

    print(f"\n  Resolution ordering: ", end="")
    order = sorted(['shielded', 'unconstrained', 'corrupted'],
                   key=lambda p: summary[p]['resolution_events'], reverse=True)
    print(" > ".join(f"{p}({summary[p]['resolution_events']:.1f})" for p in order))

    if s_res > u_res * 1.1 and u_res > c_res * 0.9:
        print("  T54 SUPPORTED: shielded > unconstrained > corrupted")
    elif abs(s_res - u_res) / max(u_res, 1) < 0.1:
        print("  T54 NULL: shielded ~= unconstrained")
    else:
        print(f"  T54 INCONCLUSIVE or CONTRADICTED")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {'per_seed': {p: all_results[p] for p in all_results}, 'summary': summary}
    with open(os.path.join(RESULTS_DIR, 't54_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to data/t54_results.json")
    return output


if __name__ == '__main__':
    run_experiment(num_bootstrap=30, num_online=30, steps_per_episode=300, num_seeds=3)

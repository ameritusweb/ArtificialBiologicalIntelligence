"""T54 v2: Closed-loop rationalization experiment.

Organisms act, the mental model updates online, certainty feeds the
observation vector (and thus the router/receptors), and the read policy
intervenes during update_certainty — not as a post-hoc re-scoring.

The gauge symmetry from v1 is broken: certainty is no longer a label,
it's a cause. Changing certainty changes what the organism sees, which
changes what it does, which changes what the mental model learns.

Pre-registered predictions:
  T55: shielded > unconstrained > corrupted on correspondence-verified resolutions
  T57: corrupted >= shielded (annealing — certainty release frees exploration)

Resolution criterion (correspondence):
  1. Prediction accuracy improves in the contested context
  2. Accuracy retained in horn A's home domain
  3. Accuracy retained in horn B's home domain
  4. New predictions actually made (engagement)

Shared encoder, multi-seed, paired by seed.
"""

import os
import copy
import json
import numpy as np
from environment import Environment, Organism, NPC
from mental_model import (build_mental_model, action_to_hash,
                           MentalModelEngine)
from model import compute_obs_indices

CONFLICT_THRESHOLD = 0.3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')


class ConflictTracker:
    """Track conflict contexts and detect correspondence-verified resolutions."""

    def __init__(self, core_obs_dim):
        self.core_obs_dim = core_obs_dim
        self.conflict_records = {}
        self.resolutions = 0
        self.total_conflict_steps = 0
        self.total_nonconflict_steps = 0
        self.conflict_mses = []
        self.nonconflict_mses = []
        self.certainties = []

    def observe(self, obs_before, action, obs_after, conflict_val,
                engine, embedding):
        ah = action_to_hash(action)
        obs_b = obs_before[:self.core_obs_dim]
        obs_a = obs_after[:self.core_obs_dim]
        actual_delta = obs_a - obs_b

        pred, cert, count = engine.predict_delta(obs_b, action)
        if count > 0:
            mse = float(np.mean((pred - actual_delta[:len(pred)]) ** 2))
        else:
            mse = 1.0

        self.certainties.append(cert)
        is_conflict = conflict_val > CONFLICT_THRESHOLD

        if is_conflict:
            self.total_conflict_steps += 1
            self.conflict_mses.append(mse)

            ctx_key = self._context_key(ah, embedding)
            if ctx_key not in self.conflict_records:
                self.conflict_records[ctx_key] = {
                    'first_mse': mse,
                    'first_conflict': conflict_val,
                    'first_cert': cert,
                    'emb': embedding.copy(),
                    'ah': ah,
                    'horn_mses': [mse],
                }
            else:
                self.conflict_records[ctx_key]['horn_mses'].append(mse)
        else:
            self.total_nonconflict_steps += 1
            self.nonconflict_mses.append(mse)

            for ctx_key, rec in list(self.conflict_records.items()):
                if rec['ah'] != ah:
                    continue
                sim = float(np.dot(embedding, rec['emb']))
                if sim < 0.5:
                    continue

                accuracy_improved = mse < rec['first_mse'] * 0.7
                conflict_dropped = conflict_val < rec['first_conflict'] * 0.5

                horn_baseline = np.mean(rec['horn_mses'][-5:]) if rec['horn_mses'] else rec['first_mse']
                horn_retained = mse < horn_baseline * 1.5

                engaged = count > 0

                if accuracy_improved and conflict_dropped and horn_retained and engaged:
                    self.resolutions += 1
                    del self.conflict_records[ctx_key]

        return mse, cert, is_conflict

    def _context_key(self, ah, emb):
        return (ah, tuple(np.round(emb[:6], 2)))

    def get_metrics(self):
        return {
            'resolutions': self.resolutions,
            'conflict_steps': self.total_conflict_steps,
            'nonconflict_steps': self.total_nonconflict_steps,
            'avg_mse_conflict': float(np.mean(self.conflict_mses)) if self.conflict_mses else 0.0,
            'avg_mse_nonconflict': float(np.mean(self.nonconflict_mses)) if self.nonconflict_mses else 0.0,
            'avg_certainty': float(np.mean(self.certainties)) if self.certainties else 0.0,
            'open_conflicts': len(self.conflict_records),
        }


def run_closed_loop_episodes(engine, read_policy, num_episodes,
                              steps_per_episode, idx, seed):
    """Run episodes where certainty has causal consequences.

    The mental model updates online. The read policy intervenes during
    update_certainty. Certainty feeds the observation vector.
    """
    rng = np.random.RandomState(seed)
    core_obs_dim = idx['core_obs_dim']
    conflict_idx = idx['conflict']
    tracker = ConflictTracker(core_obs_dim)

    for ep in range(num_episodes):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)

        prev_mm_certainty = 0.0
        prev_learning_progress = 0.0
        prev_action_hash = 0
        prev_controllability = 0.0
        prev_external_change = 0.0
        prev_planning_value = 0.0
        prev_predicted_pain = None

        for step in range(steps_per_episode):
            npc.step(env, step)
            obs_before = org.history[-1].copy() if len(org.history) > 0 else np.zeros(org.OBS_DIM)

            mm_fam, mm_qual = engine.get_context_features(obs_before)
            mm_features = (mm_fam, mm_qual, prev_mm_certainty, prev_learning_progress)
            pattern_features = None
            if engine.pattern_store is not None:
                pa, pc = engine.query_pattern(prev_action_hash, obs_before)
                pattern_features = (pa, pc)
            cm, cq = engine.query_concept(prev_action_hash, obs_before)
            org.concept_match = cm
            org.concept_quality = cq

            optimal = org.compute_optimal_actions(env, step, npc=npc)
            r = rng.random()
            if r < 0.02:
                executed = np.zeros(org.NUM_ACTIONS, dtype=np.int32)
            elif r < 0.07:
                executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32)
            else:
                executed = optimal

            obs, reward = org.step(
                executed, env, step,
                predicted_pain=prev_predicted_pain,
                mm_features=mm_features,
                pattern_features=pattern_features,
                agency_features=(prev_controllability, prev_external_change, prev_planning_value),
                npc=npc,
            )
            npc.receive_signal(executed[org.NUM_LIMBS * 3:], org.x, org.y)

            conflict_val = float(obs[conflict_idx]) if len(obs) > conflict_idx else 0.0
            emb = engine.encoder.embed(obs_before[:core_obs_dim])
            ah = action_to_hash(executed)

            mse, cert, is_conflict = tracker.observe(
                obs_before, executed, obs, conflict_val, engine, emb)

            pre_update_certainties = {}
            if read_policy in ('shielded', 'corrupted') and is_conflict:
                entries = engine.store.mappings.get(ah, [])
                for i, entry in enumerate(entries):
                    entry_sim = float(np.dot(emb, entry.context_embedding))
                    if entry_sim > engine.store.MERGE_THRESHOLD:
                        pre_update_certainties[(ah, i)] = entry.certainty

            pred_before, cert_before, _ = engine.predict_delta(obs_before, executed)
            obs_b_core = obs_before[:core_obs_dim]
            obs_a_core = obs[:core_obs_dim]
            actual_delta = obs_a_core - obs_b_core
            mse_before = float(np.mean((pred_before - actual_delta[:len(pred_before)])**2)) if len(pred_before) > 0 else 1.0

            engine.update(obs_before, executed, obs, reward)

            if read_policy == 'shielded' and is_conflict:
                entries = engine.store.mappings.get(ah, [])
                for i, entry in enumerate(entries):
                    key = (ah, i)
                    if key in pre_update_certainties:
                        entry.certainty = max(entry.certainty, pre_update_certainties[key])

            elif read_policy == 'corrupted' and is_conflict:
                entries = engine.store.mappings.get(ah, [])
                for i, entry in enumerate(entries):
                    entry_sim = float(np.dot(emb, entry.context_embedding))
                    if entry_sim > engine.store.MERGE_THRESHOLD:
                        entry.certainty *= 0.9

            pred_after, cert_after, _ = engine.predict_delta(obs_before, executed)
            mse_after = float(np.mean((pred_after - actual_delta[:len(pred_after)])**2)) if len(pred_after) > 0 else 1.0
            raw_lp = mse_before - mse_after
            prev_learning_progress = float(np.clip(raw_lp / (abs(raw_lp) + 0.01), 0.0, 1.0))
            prev_mm_certainty = cert_after

            ctrl, ext_ch, plan_v = engine.compute_agency_features(obs_before, executed, obs)
            prev_controllability = ctrl
            prev_external_change = ext_ch
            prev_planning_value = plan_v

            prev_predicted_pain = obs_b_core[:6] + pred_after[:6] if len(pred_after) >= 6 else obs_b_core[:6].copy()
            prev_action_hash = ah

    return tracker.get_metrics()


def run_experiment(num_bootstrap=30, num_online=50, steps_per_episode=300,
                   num_seeds=5):
    print("=== T54 v2: Closed-Loop Rationalization Experiment ===")
    print("  Certainty has causal consequences. Gauge symmetry broken.\n")

    idx = compute_obs_indices()
    all_results = {p: [] for p in ['unconstrained', 'shielded', 'corrupted']}

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        rng = np.random.RandomState(seed)

        print("  Bootstrap...")
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

        print(f"  Bootstrap: {len(bootstrap_log)} entries")
        print("  Building shared engine...")
        shared_engine = build_mental_model(bootstrap_log)
        print(f"  Store: {shared_engine.store.total_count} mappings")

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

            metrics = run_closed_loop_episodes(
                engine, policy, num_online, steps_per_episode,
                idx, seed=online_seed,
            )
            all_results[policy].append(metrics)
            print(f"    {policy:15s}: res={metrics['resolutions']:3d}  "
                  f"mse_c={metrics['avg_mse_conflict']:.4f}  "
                  f"cert={metrics['avg_certainty']:.4f}  "
                  f"open={metrics['open_conflicts']}")

    print("\n=== AGGREGATE (mean over seeds) ===")
    summary = {}
    for policy in ['unconstrained', 'shielded', 'corrupted']:
        res = all_results[policy]
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0].keys()}
        summary[policy] = agg
        print(f"  {policy:15s}: res={agg['resolutions']:.1f}  "
              f"mse_c={agg['avg_mse_conflict']:.4f}  "
              f"cert={agg['avg_certainty']:.4f}  "
              f"open={agg['open_conflicts']:.0f}")

    s = summary['shielded']['resolutions']
    u = summary['unconstrained']['resolutions']
    c = summary['corrupted']['resolutions']

    print(f"\n  Resolution ordering: ", end="")
    order = sorted(['shielded', 'unconstrained', 'corrupted'],
                   key=lambda p: summary[p]['resolutions'], reverse=True)
    print(" > ".join(f"{p}({summary[p]['resolutions']:.1f})" for p in order))

    print(f"\n  T55 prediction: shielded > unconstrained > corrupted")
    print(f"  T57 prediction: corrupted >= shielded")

    if s > u * 1.1:
        print(f"  --> T55 SUPPORTED (shielded wins)")
    elif c > s * 1.1:
        print(f"  --> T57 SUPPORTED (annealing wins)")
    elif abs(s - u) / max(u, 1) < 0.1 and abs(s - c) / max(c, 1) < 0.1:
        print(f"  --> NULL (no policy difference)")
    else:
        print(f"  --> INCONCLUSIVE")

    per_seed = []
    for i in range(len(all_results['unconstrained'])):
        per_seed.append({
            p: all_results[p][i] for p in all_results
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        'per_seed': per_seed,
        'summary': summary,
        'design': {
            'num_bootstrap': num_bootstrap,
            'num_online': num_online,
            'steps_per_episode': steps_per_episode,
            'num_seeds': len(all_results['unconstrained']),
            'resolution_criterion': 'correspondence (accuracy + horn retention + engagement)',
            'pre_registered_rival': 'T57 annealing (corrupted >= shielded)',
        },
    }
    with open(os.path.join(RESULTS_DIR, 't54_v2_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to data/t54_v2_results.json")
    return output


if __name__ == '__main__':
    run_experiment(num_bootstrap=20, num_online=30, steps_per_episode=300,
                   num_seeds=3)

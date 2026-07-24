import os
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash, CORE_OBS_DIM
from model import compute_obs_indices
from train import generate_training_data, PROBE_RATE_FLOOR
from itertools import combinations, islice

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@dataclass
class ReceptorTest:
    receptor_id: str
    family: str
    description: str
    threshold: float
    test_fn: Callable = None
    needs_closed_loop: bool = False  # True if the test measures organism *actions*
    # (behavior), which is only meaningful on a genuine closed-loop policy rollout,
    # not on shuffled/open-loop logs.


def build_tests():
    tests = []

    def test_static_repetition(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        high_count = [e for e in all_entries if e.count >= 5
                      and hasattr(e, 'm2') and e.m2 is not None
                      and np.mean(e.m2 / max(e.count, 1)) < 0.1]
        low_count = [e for e in all_entries if e.count <= 2]
        if len(high_count) < 10 or len(low_count) < 10:
            return 0.0
        avg_high = np.mean([e.certainty for e in high_count])
        avg_low = np.mean([e.certainty for e in low_count])
        return avg_high / (avg_low + 1e-8)

    tests.append(ReceptorTest('static_repetition', 'repetition',
        'Certainty higher for repeated low-variance vs novel contexts', 1.2, test_static_repetition))

    def test_rhythm(log, engine, **kw):
        if len(log) < 200:
            return 0.0
        pain_series = np.array([e['obs_after'][0] for e in log[:2000]])
        if pain_series.std() < 0.01:
            return 0.0
        normalized = (pain_series - pain_series.mean()) / (pain_series.std() + 1e-8)
        best_corr = 0.0
        for lag in [50, 75, 100, 125, 150, 200]:
            if lag >= len(normalized):
                continue
            corr = np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]
            if not np.isnan(corr):
                best_corr = max(best_corr, abs(corr))
        return best_corr

    tests.append(ReceptorTest('rhythm', 'repetition',
        'Autocorrelation of pain at rhythmic lags', 0.3, test_rhythm))

    def test_coincidence(log, engine, **kw):
        N = min(10000, len(log))
        pain_high = np.array([e['obs_after'][0] > 0.3 for e in log[:N]])
        endo_high = np.array([e['obs_after'][6] > 0.3 for e in log[:N]])
        p_pain = pain_high.mean()
        p_endo = endo_high.mean()
        p_both = (pain_high & endo_high).mean()
        p_independent = p_pain * p_endo
        if p_independent < 1e-6:
            return 0.0
        return p_both / (p_independent + 1e-8)

    tests.append(ReceptorTest('coincidence_detection', 'causality',
        'Co-occurrence of pain+endorphin above chance', 0.1, test_coincidence))

    def test_precedence(log, engine, **kw):
        idx = compute_obs_indices()
        L = idx['total_limbs']
        ta_start = idx['ta_start']
        N = min(5000, len(log))
        correct = 0
        total = 0
        for i in range(3, N):
            pain_pre = np.mean(_ch(log[i-3]['obs_after'], _pain))
            pain_onset = np.mean(_ch(log[i-2]['obs_after'], _pain))
            if pain_pre < 0.1 and pain_onset > 0.3:
                total += 1
                ta_pre = np.mean(log[i-3]['obs_after'][ta_start:ta_start+L])
                ta_post = np.mean(log[i]['obs_after'][ta_start:ta_start+L])
                expected_inc = 0.3 * pain_onset * 0.3
                if ta_post > ta_pre + expected_inc:
                    correct += 1
        return correct / (total + 1e-8)

    tests.append(ReceptorTest('precedence_detection', 'causality',
        'Pain onset precedes temporal aversion increase', 0.3, test_precedence))

    def test_causal_association(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        consistent = 0
        total = 0
        action_outcomes = defaultdict(list)
        for e in log[:N]:
            ah = action_to_hash(e['action'])
            delta = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            action_outcomes[ah].append(delta)
        for ah, deltas in action_outcomes.items():
            if len(deltas) >= 3:
                total += 1
                mean_delta = np.mean(deltas, axis=0)
                variance = np.mean([np.mean((d - mean_delta)**2) for d in deltas])
                if variance < 0.1:
                    consistent += 1
        return consistent / (total + 1e-8)

    tests.append(ReceptorTest('causal_association', 'causality',
        'Actions produce consistent outcomes across repetitions', 0.2, test_causal_association))

    def test_spatial_association(log, engine, **kw):
        N = min(5000, len(log))
        memories = np.array([e['obs_after'][49:74] for e in log[:N]])
        avg_memory = memories.mean(axis=0)
        active_cells = (avg_memory > 0.5).sum()
        return float(active_cells)

    tests.append(ReceptorTest('spatial_association', 'association',
        'Pain memory grid cells with high values', 3.0, test_spatial_association))

    def test_temporal_association(log, engine, **kw):
        cdim = engine.core_obs_dim
        L = min(cdim, 6)
        N = min(5000, len(log))
        ta = np.array([e['obs_after'][6*L+1:7*L+1] for e in log[:N]])
        pain = np.array([_ch(e['obs_after'], _pain) for e in log[:N]])
        if ta.std() < 0.01 or pain.std() < 0.01:
            return 0.0
        corr = np.corrcoef(ta.mean(axis=1), pain.mean(axis=1))[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    tests.append(ReceptorTest('temporal_association', 'association',
        'Temporal aversion correlates with pain history', 0.5, test_temporal_association))

    def test_perceptual_similarity(log, engine, **kw):
        all_entries = [e for entries in engine.store.mappings.values() for e in entries]
        if len(all_entries) < 100:
            return 0.0
        sample = np.random.choice(len(all_entries), min(500, len(all_entries)), replace=False)
        embeddings = np.array([all_entries[i].context_embedding for i in sample])
        certs = np.array([all_entries[i].certainty for i in sample])
        high = embeddings[certs > 0.6]
        low = embeddings[certs < 0.4]
        if len(high) < 10 or len(low) < 10:
            return 0.0
        high_spread = np.mean(np.std(high, axis=0))
        low_spread = np.mean(np.std(low, axis=0))
        return low_spread / (high_spread + 1e-8)

    tests.append(ReceptorTest('perceptual_similarity', 'similarity',
        'Encoder clusters high-certainty contexts tighter', 0.3, test_perceptual_similarity))

    def test_pattern_recognition(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        stats = engine.pattern_store.get_concept_stats()
        return float(stats['num_stable_concepts'])

    tests.append(ReceptorTest('pattern_recognition', 'compression',
        'Stable concepts count', 50.0, test_pattern_recognition))

    def test_compression_gain(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        stats = engine.pattern_store.get_stats()
        return stats['avg_compression_gain']

    tests.append(ReceptorTest('compression_gain', 'compression',
        'Average compression gain of patterns', 0.3, test_compression_gain))

    def test_other_detection(log, engine, **kw):
        cdim = engine.core_obs_dim
        N = min(5000, len(log))
        npc_present = [e for e in log[:N]
                       if len(e['obs_after']) > cdim + 40
                       and e['obs_after'][cdim + 40] > 0.1]
        npc_absent = [e for e in log[:N]
                      if len(e['obs_after']) > cdim + 40
                      and e['obs_after'][cdim + 40] < 0.1]
        if len(npc_present) < 50 or len(npc_absent) < 50:
            return 0.0
        reward_near = np.mean([e['reward'] for e in npc_present])
        reward_far = np.mean([e['reward'] for e in npc_absent])
        return abs(reward_near - reward_far) / (abs(reward_far) + 1e-8)

    tests.append(ReceptorTest('other_detection', 'social',
        'NPC proximity affects reward differently', 0.1, test_other_detection))

    def test_empathic_concern(log, engine, **kw):
        N = min(10000, len(log))
        cdim = engine.core_obs_dim
        empathic_count = 0
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > cdim + 47 and obs[cdim + 47] > 0.05:
                empathic_count += 1
        return empathic_count / N

    tests.append(ReceptorTest('empathic_concern', 'social',
        'Empathic aversion fires when NPC distressed', 0.2, test_empathic_concern))

    def test_controllability(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        ctrls = []
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > cdim + 32:
                ctrls.append(obs[cdim + 32])
        if not ctrls:
            return 0.0
        return float(np.max(ctrls) - np.min(ctrls))

    tests.append(ReceptorTest('controllability', 'agency',
        'Controllability signal range', 0.5, test_controllability))

    def test_planning(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        plans = []
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > cdim + 34:
                plans.append(obs[cdim + 34])
        if not plans:
            return 0.0
        return float(np.max(plans) - np.min(plans))

    tests.append(ReceptorTest('planning', 'agency',
        'Planning value range', 0.3, test_planning))

    def test_curiosity(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        pe_start = 8 * L + 42
        high_pe_regions = defaultdict(int)
        all_regions = defaultdict(int)
        for e in log[:N]:
            obs = e['obs_after']
            gx = int(np.clip(obs[0] * 2, 0, 9)) if len(obs) > 0 else 0
            gy = int(np.clip(obs[1] * 2, 0, 9)) if len(obs) > 1 else 0
            region = (gx, gy)
            all_regions[region] += 1
            if len(obs) > pe_start + L:
                pe = np.mean(obs[pe_start:pe_start + L])
                if pe > 0.15:
                    high_pe_regions[region] += 1
        if not high_pe_regions or not all_regions:
            return 0.0
        revisit_rate_high_pe = np.mean([all_regions[r] for r in high_pe_regions])
        revisit_rate_low_pe = np.mean([all_regions[r] for r in all_regions
                                       if r not in high_pe_regions])
        if revisit_rate_low_pe < 1:
            return 0.0
        return revisit_rate_high_pe / revisit_rate_low_pe

    tests.append(ReceptorTest('curiosity', 'meta_motivational',
        'Regions with high prediction error get revisited more', 1.1, test_curiosity))

    def test_optimism(log, engine, **kw):
        idx = compute_obs_indices()
        opt_idx = idx['opt_start']
        N = min(10000, len(log))
        opt_count = 0
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > opt_idx and obs[opt_idx] > 0.3:
                opt_count += 1
        return opt_count / N

    tests.append(ReceptorTest('optimism', 'meta_motivational',
        'Optimism signal > 0.3 frequency', 0.2, test_optimism))

    def test_conflict(log, engine, **kw):
        idx = compute_obs_indices()
        conflict_idx = idx['conflict']
        N = min(10000, len(log))
        conflict_count = 0
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > conflict_idx and obs[conflict_idx] > 0.2:
                conflict_count += 1
        return conflict_count / N

    tests.append(ReceptorTest('conflict_detection', 'meta_motivational',
        'Receptor conflict > 0.2 frequency', 0.1, test_conflict))

    def test_stress(log, engine, **kw):
        N = min(10000, len(log))
        L = min(engine.core_obs_dim // 9, 6)
        energy_idx = 6 * L
        stress_count = 0
        for e in log[:N]:
            obs = e['obs_after']
            pain_avg = np.mean(_ch(obs, _pain))
            energy = obs[energy_idx] if len(obs) > energy_idx else 1.0
            if pain_avg > 0.3 and energy < 0.3:
                stress_count += 1
        return stress_count / N

    tests.append(ReceptorTest('stress_detection', 'regulatory',
        'High pain + low energy co-occurrence', 0.05, test_stress))

    def test_change_detection(log, engine, **kw):
        idx = compute_obs_indices()
        ext_idx = idx['agency_start'] + 1
        N = min(5000, len(log))
        change_responses = []
        stable_responses = []
        for i in range(2, N):
            obs = log[i]['obs_after']
            if len(obs) <= ext_idx:
                continue
            external = obs[ext_idx]
            action_change = np.sum(np.abs(
                np.array(log[i]['action']) - np.array(log[i-1]['action'])))
            if external > 0.3:
                change_responses.append(action_change)
            elif external < 0.05:
                stable_responses.append(action_change)
        if len(change_responses) < 20 or len(stable_responses) < 20:
            return 0.0
        return np.mean(change_responses) / (np.mean(stable_responses) + 1e-8)

    tests.append(ReceptorTest('change_detection', 'observation',
        'Action change ratio: external changes vs stable periods', 1.3, test_change_detection))

    # === STEP 35: 21 additional receptor tests ===

    def test_dynamic_repetition(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        if len(all_entries) < 50:
            return 0.0
        high = [e for e in all_entries if e.count >= 5]
        low = [e for e in all_entries if 1 <= e.count <= 2]
        if not high or not low:
            return 0.0
        high_embs = np.array([e.context_embedding for e in high[:200]])
        high_spread = np.mean(np.std(high_embs, axis=0))
        low_embs = np.array([e.context_embedding for e in low[:200]])
        low_spread = np.mean(np.std(low_embs, axis=0))
        if high_spread < 1e-6:
            return 0.0
        return high_spread / (low_spread + 1e-8)

    tests.append(ReceptorTest('dynamic_repetition', 'repetition',
        'High-count entries span diverse contexts (dynamic, not static)', 0.8, test_dynamic_repetition))

    def test_cross_modal_association(log, engine, **kw):
        N = min(5000, len(log))
        L = min(engine.core_obs_dim // 9, 6)
        with_temp = [e for e in log[:N] if np.mean(e['obs_after'][2*L:3*L]) > 0.3]
        without_temp = [e for e in log[:N] if np.mean(e['obs_after'][2*L:3*L]) < 0.1]
        if len(with_temp) < 50 or len(without_temp) < 50:
            return 0.0
        endo_with = np.mean([np.mean(_ch(e['obs_after'], _endo)) for e in with_temp])
        endo_without = np.mean([np.mean(_ch(e['obs_after'], _endo)) for e in without_temp])
        return abs(endo_with - endo_without) / (endo_without + 1e-8)

    tests.append(ReceptorTest('cross_modal_association', 'association',
        'Cross-channel prediction boost (temp present vs absent)', 0.2, test_cross_modal_association))

    def test_functional_similarity(log, engine, **kw):
        cdim = engine.core_obs_dim
        N = min(5000, len(log))
        action_deltas = defaultdict(list)
        null_hash = 0
        for e in log[:N]:
            ah = action_to_hash(e['action'])
            delta = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            action_deltas[ah].append(delta)
        null_drift = np.zeros(cdim)
        if null_hash in action_deltas and len(action_deltas[null_hash]) >= 3:
            null_drift = np.mean(action_deltas[null_hash], axis=0)
        groups = {}
        for ah, deltas in action_deltas.items():
            if len(deltas) >= 3 and ah != null_hash:
                residual = np.mean(deltas, axis=0) - null_drift
                if np.linalg.norm(residual) > 1e-6:
                    groups[ah] = residual
        if len(groups) < 5:
            return 0.0
        means = list(groups.values())
        sims = []
        for a, b in list(combinations(range(len(means)), 2))[:500]:
            cos = np.dot(means[a], means[b]) / (np.linalg.norm(means[a]) * np.linalg.norm(means[b]) + 1e-8)
            sims.append(cos)
        if not sims:
            return 0.0
        high_sim = sum(1 for s in sims if s > 0.7)
        return high_sim / (len(sims) + 1e-8)

    tests.append(ReceptorTest('functional_similarity', 'similarity',
        'Different actions produce similar effects after drift subtraction', 0.03, test_functional_similarity))

    def test_categorical_perception(log, engine, **kw):
        all_entries = [e for entries in engine.store.mappings.values() for e in entries]
        if len(all_entries) < 100:
            return 0.0
        sample = np.random.choice(len(all_entries), min(500, len(all_entries)), replace=False)
        embs = np.array([all_entries[i].context_embedding for i in sample])
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(5, len(embs)//20+1), random_state=0, n_init=3).fit(embs)
        labels = km.labels_
        within, between = [], []
        for i in range(len(embs)):
            for j in range(i+1, min(i+20, len(embs))):
                d = np.linalg.norm(embs[i] - embs[j])
                if labels[i] == labels[j]:
                    within.append(d)
                else:
                    between.append(d)
        if not within or not between:
            return 0.0
        return np.mean(between) / (np.mean(within) + 1e-8)

    tests.append(ReceptorTest('categorical_perception', 'similarity',
        'Between-cluster vs within-cluster distance ratio', 1.5, test_categorical_perception))

    def test_causal_inference(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        intentional = 0
        total = 0
        for e in log[:N]:
            delta = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            pred, cert, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if cert > 0.5:
                total += 1
                actual_mag = np.linalg.norm(delta)
                pred_mag = np.linalg.norm(pred)
                if actual_mag > 0.01 and pred_mag > 0.01:
                    cos_sim = np.dot(delta, pred) / (actual_mag * pred_mag + 1e-8)
                    if cos_sim > 0.5:
                        intentional += 1
        return intentional / (total + 1e-8)

    tests.append(ReceptorTest('causal_inference', 'causality',
        'Fraction of high-certainty actions producing predicted outcomes', 0.4, test_causal_inference))

    def test_probabilistic_causation(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        if len(all_entries) < 100:
            return 0.0
        sample = all_entries[:min(500, len(all_entries))]
        calibrated = 0
        total = 0
        for entry in sample:
            if entry.count >= 3:
                total += 1
                if 0.3 < entry.certainty < 0.9:
                    calibrated += 1
        return calibrated / (total + 1e-8)

    tests.append(ReceptorTest('probabilistic_causation', 'causality',
        'Certainty values are distributed (not all 0 or 1) — calibrated uncertainty', 0.3, test_probabilistic_causation))

    def test_agency_salience(log, engine, **kw):
        idx = compute_obs_indices()
        ctrl_idx = idx['agency_start']
        N = min(5000, len(log))
        window = 10
        pairs = []
        for i in range(window, N):
            obs = log[i]['obs_after']
            if len(obs) <= ctrl_idx:
                continue
            ctrl = obs[ctrl_idx]
            actions_window = [action_to_hash(log[j]['action']) for j in range(i-window, i)]
            diversity = len(set(actions_window)) / window
            pairs.append((ctrl, diversity))
        if len(pairs) < 100:
            return 0.0
        ctrls = np.array([p[0] for p in pairs])
        divs = np.array([p[1] for p in pairs])
        corr = np.corrcoef(ctrls, divs)[0, 1]
        if np.isnan(corr):
            return 0.0
        return max(0.0, corr)

    tests.append(ReceptorTest('agency_salience', 'agency',
        'Higher controllability correlates with more action diversity', 0.15, test_agency_salience))

    def test_impulse_override(log, engine, **kw):
        idx = compute_obs_indices()
        L = idx['total_limbs']
        persist_idx = idx['opt_start'] + 1
        N = min(10000, len(log))
        persist_count = 0
        total_pain = 0
        for e in log[:N]:
            obs = e['obs_after']
            pain_avg = np.mean(_ch(obs, _pain))
            if pain_avg > 0.2:
                total_pain += 1
                if len(obs) > persist_idx and obs[persist_idx] > 0.15:
                    persist_count += 1
        return persist_count / (total_pain + 1e-8)

    tests.append(ReceptorTest('impulse_override', 'meta_motivational',
        'Goal persistence above floor during pain', 0.1, test_impulse_override))

    def test_arousal_regulation(log, engine, **kw):
        N = min(5000, len(log))
        L = min(engine.core_obs_dim // 9, 6)
        energy_idx = 6 * L
        dangers, action_rates = [], []
        for i in range(1, N):
            obs = log[i]['obs_after']
            pain_avg = np.mean(_ch(obs, _pain))
            dangers.append(pain_avg)
            prev_act = log[i-1]['action']
            action_rates.append(np.sum(prev_act[:L*3]) / (L*3))
        if len(dangers) < 100:
            return 0.0
        corr = np.corrcoef(dangers, action_rates)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    tests.append(ReceptorTest('arousal_regulation', 'regulatory',
        'Action rate correlates with danger level', 0.15, test_arousal_regulation))

    def test_pattern_resolution(log, engine, **kw):
        N = min(5000, len(log))
        L = min(engine.core_obs_dim // 9, 6)
        repeats, non_repeats = [], []
        for i in range(2, N):
            h0 = action_to_hash(log[i-2]['action'])
            h1 = action_to_hash(log[i-1]['action'])
            h2 = action_to_hash(log[i]['action'])
            pain_after = np.mean(_ch(log[i]['obs_after'], _pain))
            if h0 == h2:
                repeats.append(pain_after)
            else:
                non_repeats.append(pain_after)
        if len(repeats) < 50 or len(non_repeats) < 50:
            return 0.0
        return (np.mean(non_repeats) - np.mean(repeats)) / (np.mean(non_repeats) + 1e-8)

    tests.append(ReceptorTest('pattern_based_resolution', 'regulatory',
        'Repeated action sequences reduce subsequent pain', 0.05, test_pattern_resolution))

    def test_behavioral_prediction(log, engine, **kw):
        idx = compute_obs_indices()
        npc_start = idx['npc_start']
        L = idx['total_limbs']
        N = min(5000, len(log))
        anticipatory = 0
        reactive = 0
        for i in range(5, N - 5):
            obs = log[i]['obs_after']
            if len(obs) <= npc_start:
                continue
            npc_dists = []
            for j in range(i-3, i+4):
                if j < N and len(log[j]['obs_after']) > npc_start:
                    npc_dists.append(log[j]['obs_after'][npc_start])
            if len(npc_dists) < 7:
                continue
            approaching = npc_dists[3] < npc_dists[0]
            still_far = npc_dists[3] > 0.3
            if approaching and still_far:
                act_now = np.sum(log[i]['action'][:L*3])
                act_prev = np.sum(log[i-3]['action'][:L*3])
                if act_now > act_prev + 0.5:
                    anticipatory += 1
                else:
                    reactive += 1
        total = anticipatory + reactive
        if total < 20:
            return 0.0
        return anticipatory / total

    tests.append(ReceptorTest('behavioral_prediction', 'social',
        'Anticipatory avoidance when NPC approaches', 0.3, test_behavioral_prediction))

    def test_concept_formation(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        concepts = engine.pattern_store.extract_concepts(top_k=20)
        if len(concepts) < 5:
            return 0.0
        stabilities = [pat.certainty for _, _, pat in concepts]
        return float(np.mean(stabilities))

    tests.append(ReceptorTest('concept_formation', 'compression',
        'Average certainty of top concepts', 0.6, test_concept_formation))

    def test_categorical_compression(log, engine, **kw):
        if engine.family_manager is None:
            return 0.0
        fam_certs = defaultdict(list)
        for entries in engine.store.mappings.values():
            for e in entries:
                ah = int([k for k, v in engine.store.mappings.items() if e in v][0]) if False else 0
                fam_certs[0].append(e.certainty)
        all_entries = [e for entries in engine.store.mappings.values() for e in entries]
        if len(all_entries) < 100:
            return 0.0
        high_count = [e.certainty for e in all_entries if e.count >= 5]
        return np.mean(high_count) if high_count else 0.0

    tests.append(ReceptorTest('categorical_compression', 'compression',
        'High-count mappings have elevated certainty (category transfer)', 0.5, test_categorical_compression))

    def test_chunking(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        N = min(5000, len(log))
        matched, unmatched = [], []
        cdim = engine.core_obs_dim
        for i in range(1, N):
            prev_hash = action_to_hash(log[i-1]['action'])
            obs = log[i]['obs_before'][:cdim]
            emb = engine.encoder.embed(obs)
            results = engine.pattern_store.query(prev_hash, emb)
            reward = log[i]['reward']
            if results and results[0][1] > 0.5:
                matched.append(reward)
            else:
                unmatched.append(reward)
        if len(matched) < 50 or len(unmatched) < 50:
            return 0.0
        return (np.mean(matched) - np.mean(unmatched)) / (abs(np.mean(unmatched)) + 1e-8)

    tests.append(ReceptorTest('chunking', 'compression',
        'Pattern-matched steps produce higher reward', 0.05, test_chunking))

    def test_bias_as_compression(log, engine, **kw):
        all_entries = [e for entries in engine.store.mappings.values() for e in entries]
        if len(all_entries) < 100:
            return 0.0
        high_count = [e for e in all_entries if e.count >= 10]
        low_count = [e for e in all_entries if 1 <= e.count <= 2]
        if len(high_count) < 20 or len(low_count) < 20:
            return 0.0
        high_spread = np.mean([np.std(e.delta) for e in high_count if hasattr(e.delta, '__len__')])
        low_spread = np.mean([np.std(e.delta) for e in low_count if hasattr(e.delta, '__len__')])
        return (low_spread - high_spread) / (low_spread + 1e-8)

    tests.append(ReceptorTest('bias_as_compression', 'compression',
        'High-count entries have lower delta variance (information discarded)', 0.1, test_bias_as_compression))

    def test_compression_seeking(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        stats = engine.pattern_store.get_concept_stats()
        total = stats['total_patterns']
        stable = stats['num_stable_concepts']
        return stable / (total + 1e-8)

    tests.append(ReceptorTest('compression_receptor', 'compression',
        'Fraction of patterns that are stable concepts', 0.15, test_compression_seeking))

    def test_absence_observation(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        drops = 0
        total_high = 0
        for i in range(1, N):
            _, cert_before, _ = engine.predict_delta(log[i-1]['obs_before'][:cdim], log[i-1]['action'])
            _, cert_after, _ = engine.predict_delta(log[i]['obs_before'][:cdim], log[i]['action'])
            if cert_before > 0.6:
                total_high += 1
                if cert_after < cert_before - 0.1:
                    drops += 1
        return drops / (total_high + 1e-8)

    tests.append(ReceptorTest('absence_observation', 'observation',
        'Certainty drops after high-certainty prediction failures', 0.05, test_absence_observation))

    def test_exception_detection(log, engine, **kw):
        cdim = engine.core_obs_dim
        N = min(5000, len(log))
        high_cert_data = []
        for i in range(N - 3):
            e = log[i]
            pred, cert, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if cert < 0.6 or n == 0:
                continue
            actual = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            error = np.mean((pred - actual[:len(pred)])**2)
            future_activity = np.mean([np.sum(log[j]['action'])
                                       for j in range(i+1, min(i+4, N))])
            high_cert_data.append((error, future_activity))
        if len(high_cert_data) < 50:
            return 0.0
        errors = np.array([d[0] for d in high_cert_data])
        activities = np.array([d[1] for d in high_cert_data])
        exception_thresh = np.percentile(errors, 90)
        normal_thresh = np.percentile(errors, 50)
        exceptions = activities[errors >= exception_thresh]
        normals = activities[errors < normal_thresh]
        if len(exceptions) < 5 or len(normals) < 5:
            return 0.0
        ratio = normals.mean() / (exceptions.mean() + 1e-8)
        return max(0.0, ratio - 1.0)

    tests.append(ReceptorTest('exception_detection', 'formalization',
        'Organism more cautious after top-decile prediction failures', 0.1, test_exception_detection))

    def test_rule_extraction(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        if len(all_entries) < 100:
            return 0.0
        rules = [e for e in all_entries if e.certainty > 0.6 and e.count >= 5]
        return len(rules) / (len(all_entries) + 1e-8)

    tests.append(ReceptorTest('rule_extraction', 'formalization',
        'Fraction of mappings that qualify as reliable rules', 0.03, test_rule_extraction))

    def test_optimization(log, engine, **kw):
        """Organism replaces working solutions with better ones.
        Measured by: for action hashes seen in both early and late log,
        does late certainty exceed early certainty (solution improved)?"""
        cdim = engine.core_obs_dim; N = len(log)
        if N < 200:
            return 0.0
        quarter = N // 4
        early_certs = {}
        late_certs = {}
        for e in log[:quarter]:
            ah = action_to_hash(e['action'])
            obs = e['obs_before'][:cdim]
            _, cert, n = engine.predict_delta(obs, e['action'])
            if n > 0:
                early_certs.setdefault(ah, []).append(cert)
        for e in log[-quarter:]:
            ah = action_to_hash(e['action'])
            obs = e['obs_before'][:cdim]
            _, cert, n = engine.predict_delta(obs, e['action'])
            if n > 0:
                late_certs.setdefault(ah, []).append(cert)
        shared = set(early_certs.keys()) & set(late_certs.keys())
        if len(shared) < 3:
            return 0.0
        improvements = 0
        for ah in shared:
            if np.mean(late_certs[ah]) > np.mean(early_certs[ah]) + 0.05:
                improvements += 1
        return improvements / len(shared)

    tests.append(ReceptorTest('optimization', 'formalization',
        'Late-log solutions have higher certainty than early-log for same actions', 0.3,
        test_optimization))

    def test_boundary_detection(log, engine, **kw):
        idx = compute_obs_indices()
        dp_start = idx['distant_pain_start']
        N = min(5000, len(log))
        boundary_actions = []
        smooth_actions = []
        for i in range(2, N):
            obs_now = log[i]['obs_after']
            obs_prev = log[i-1]['obs_after']
            if len(obs_now) < dp_start + 8:
                continue
            dp_now = obs_now[dp_start:dp_start+8]
            dp_prev = obs_prev[dp_start:dp_start+8]
            gradient = np.linalg.norm(np.array(dp_now) - np.array(dp_prev))
            action_change = np.sum(np.abs(
                np.array(log[i]['action']) - np.array(log[i-1]['action'])))
            if gradient > 0.3:
                boundary_actions.append(action_change)
            elif gradient < 0.05:
                smooth_actions.append(action_change)
        if len(boundary_actions) < 10 or len(smooth_actions) < 10:
            return 0.0
        return np.mean(boundary_actions) / (np.mean(smooth_actions) + 1e-8)

    tests.append(ReceptorTest('boundary_detection', 'formalization',
        'More action change at spatial discontinuities than smooth regions', 1.3, test_boundary_detection))

    # === COMPLETION + MATHEMATICS + ORGANIZATION families ===

    def test_completion(log, engine, **kw):
        idx = compute_obs_indices()
        conflict_idx = idx['conflict']
        L = idx['total_limbs']
        N = min(5000, len(log))
        sharp_drops = 0
        total_high_conflict = 0
        for i in range(2, N):
            obs_prev = log[i-1]['obs_after']
            obs_now = log[i]['obs_after']
            if len(obs_prev) <= conflict_idx or len(obs_now) <= conflict_idx:
                continue
            conflict_prev = obs_prev[conflict_idx]
            conflict_now = obs_now[conflict_idx]
            endo_now = np.mean(_ch(obs_now, _endo))
            if conflict_prev > 0.2:
                total_high_conflict += 1
                drop = conflict_prev - conflict_now
                if drop > 0.1 and endo_now > 0.2:
                    sharp_drops += 1
        return sharp_drops / (total_high_conflict + 1e-8)

    tests.append(ReceptorTest('completion', 'compression',
        'Sharp conflict drop when reward is reached (gestalt completes)', 0.05, test_completion))

    def test_quantity_detection(log, engine, **kw):
        idx = compute_obs_indices()
        speed_idx = idx['proprio_start']
        de_start = idx['distant_endo_start']
        N = min(5000, len(log))
        high_speeds = []
        low_speeds = []
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) <= speed_idx or len(obs) < de_start + 8:
                continue
            speed = obs[speed_idx]
            max_endo = np.max(obs[de_start:de_start+8])
            if max_endo > 0.5:
                high_speeds.append(speed)
            elif max_endo < 0.1:
                low_speeds.append(speed)
        if len(high_speeds) < 20 or len(low_speeds) < 20:
            return 0.0
        return np.mean(high_speeds) / (np.mean(low_speeds) + 1e-8)

    tests.append(ReceptorTest('quantity_detection', 'mathematics',
        'Organism moves faster toward high-endorphin vs low-endorphin regions', 1.2, test_quantity_detection))

    def test_ratio_detection(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        pain_endo_ratios = []
        rewards = []
        for e in log[:N]:
            pain = np.mean(_ch(e['obs_after'], _pain))
            endo = np.mean(_ch(e['obs_after'], _endo))
            if pain + endo > 0.1:
                ratio = endo / (pain + endo + 1e-8)
                pain_endo_ratios.append(ratio)
                rewards.append(e['reward'])
        if len(pain_endo_ratios) < 100:
            return 0.0
        corr = np.corrcoef(pain_endo_ratios, rewards)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    tests.append(ReceptorTest('ratio_detection', 'mathematics',
        'Endorphin/pain ratio predicts reward (proportional reasoning)', 0.3, test_ratio_detection))

    def test_structural_invariance_math(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        if len(all_entries) < 200:
            return 0.0
        embs = np.array([e.context_embedding for e in all_entries[:500]])
        deltas = np.array([np.mean(np.abs(e.delta)) for e in all_entries[:500]])
        high_delta = embs[deltas > np.median(deltas)]
        low_delta = embs[deltas <= np.median(deltas)]
        if len(high_delta) < 20 or len(low_delta) < 20:
            return 0.0
        high_mean = np.mean(high_delta, axis=0)
        low_mean = np.mean(low_delta, axis=0)
        separation = np.linalg.norm(high_mean - low_mean)
        return float(separation)

    tests.append(ReceptorTest('structural_invariance_math', 'mathematics',
        'Encoder separates high-impact from low-impact contexts (invariant structure)', 0.1, test_structural_invariance_math))

    def test_org_boundary(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        action_changes = 0
        gradient_crossings = 0
        for i in range(1, N):
            pain_prev = np.mean(_ch(log[i-1]['obs_after'], _pain))
            pain_now = np.mean(_ch(log[i]['obs_after'], _pain))
            gradient_change = abs(pain_now - pain_prev)
            act_prev = action_to_hash(log[i-1]['action'])
            act_now = action_to_hash(log[i]['action'])
            action_changed = act_prev != act_now
            if gradient_change > 0.1:
                gradient_crossings += 1
                if action_changed:
                    action_changes += 1
        return action_changes / (gradient_crossings + 1e-8)

    tests.append(ReceptorTest('org_boundary_detection', 'organization',
        'Action changes correlate with field gradient crossings', 0.3, test_org_boundary))

    def test_part_whole(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        if L < 4:
            return 0.0
        body_directions = []
        limb_agreements = []
        for i in range(1, N):
            obs = log[i]['obs_after']
            if len(obs) > cdim + 3:
                speed = obs[cdim + 2] if cdim + 2 < len(obs) else 0
                omega = obs[cdim + 3] if cdim + 3 < len(obs) else 0
                action = log[i]['action']
                extends = [action[j*3] for j in range(L)]
                if sum(extends) > 0:
                    agreement = max(sum(extends), L - sum(extends)) / L
                    limb_agreements.append(agreement)
        if len(limb_agreements) < 100:
            return 0.0
        return float(np.mean(limb_agreements))

    tests.append(ReceptorTest('part_whole_detection', 'organization',
        'Limb activations are coordinated (part serves whole)', 0.6, test_part_whole))

    def test_functional_org(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        action_groups = defaultdict(list)
        for e in log[:N]:
            ah = action_to_hash(e['action'])
            action_groups[ah].append(e['reward'])
        if len(action_groups) < 5:
            return 0.0
        means = {ah: np.mean(r) for ah, r in action_groups.items() if len(r) >= 3}
        if len(means) < 5:
            return 0.0
        reward_spread = max(means.values()) - min(means.values())
        return float(reward_spread)

    tests.append(ReceptorTest('functional_organization', 'organization',
        'Different action arrangements produce different reward levels', 0.5, test_functional_org))

    # --- Physics-specific receptors (Interaction family) ---

    def test_grip_affordance(log, engine):
        """Detect grip affordance: organism grips objects and carries them."""
        grip_events = 0
        carry_steps = 0
        for entry in log:
            obs = entry['obs_after']
            if len(obs) > 165:
                grip_state = obs[160:166]
                carried_mass = obs[166]
                if sum(grip_state) > 0:
                    grip_events += 1
                if carried_mass > 0.01:
                    carry_steps += 1
        if len(log) < 10:
            return 0.0
        grip_rate = grip_events / len(log)
        carry_rate = carry_steps / len(log)
        return float(min(1.0, (grip_rate + carry_rate) * 5))

    tests.append(ReceptorTest('grip_affordance', 'interaction',
        'Organism grips objects and carries them', 0.3, test_grip_affordance))

    def test_lever_affordance(log, engine):
        """Detect lever affordance: compound objects change angle after contact."""
        contact_steps = 0
        angle_changes = 0
        for i in range(1, len(log)):
            obs = log[i]['obs_after']
            prev_obs = log[i-1]['obs_after']
            if len(obs) > 168:
                contact = obs[167]
                if contact > 0:
                    contact_steps += 1
                force_now = obs[168]
                force_prev = prev_obs[168] if len(prev_obs) > 168 else 0
                if abs(force_now - force_prev) > 0.05 and contact > 0:
                    angle_changes += 1
        if contact_steps < 5:
            return 0.0
        return float(min(1.0, angle_changes / max(1, contact_steps)))

    tests.append(ReceptorTest('lever_affordance', 'interaction',
        'Compound objects change state after organism contact', 0.2, test_lever_affordance))

    def test_contact_response(log, engine):
        """Detect contact response: actions change after contact events."""
        pre_contact_actions = []
        post_contact_actions = []
        in_contact = False
        for i in range(len(log)):
            obs = log[i]['obs_after']
            contact = obs[167] if len(obs) > 167 else 0
            action = log[i]['action']
            if contact > 0 and not in_contact:
                in_contact = True
                if i > 0:
                    pre_contact_actions.append(log[i-1]['action'])
                post_contact_actions.append(action)
            elif contact == 0:
                in_contact = False
        if len(pre_contact_actions) < 3 or len(post_contact_actions) < 3:
            return 0.0
        pre_mean = np.mean([np.sum(a) for a in pre_contact_actions])
        post_mean = np.mean([np.sum(a) for a in post_contact_actions])
        return float(min(1.0, abs(post_mean - pre_mean) / max(1, pre_mean + 0.01)))

    tests.append(ReceptorTest('contact_response', 'interaction',
        'Action pattern changes after contact events', 0.15, test_contact_response))

    def test_capability_change(log, engine):
        """Detect capability change: action patterns recalibrate after receptor gain shifts."""
        idx = compute_obs_indices()
        gain_start = idx['gain_start']
        L = idx['total_limbs']
        gain_end = gain_start + L
        N = min(5000, len(log))
        recalibrations = 0
        capability_shifts = 0
        for i in range(10, N):
            obs = log[i]['obs_after']
            prev = log[i-5]['obs_after']
            if len(obs) <= gain_end or len(prev) <= gain_end:
                continue
            gain_now = obs[gain_start:gain_end]
            gain_prev = prev[gain_start:gain_end]
            gain_shift = np.linalg.norm(np.array(gain_now) - np.array(gain_prev))
            if gain_shift > 0.1:
                capability_shifts += 1
                action_before = np.mean([np.sum(log[j]['action']) for j in range(i-5, i)])
                action_after = np.mean([np.sum(log[min(j, N-1)]['action']) for j in range(i, i+5)])
                if abs(action_after - action_before) > 0.5:
                    recalibrations += 1
        if capability_shifts < 5:
            return 0.0
        return recalibrations / capability_shifts

    tests.append(ReceptorTest('capability_change_detection', 'self_augmentation',
        'Action recalibration after receptor gain shifts', 0.2, test_capability_change))

    def test_environmental_modification(log, engine):
        """Detect environmental modification: object positions change due to organism actions."""
        if len(log) < 50:
            return 0.0
        obj_start = 135
        early_positions = []
        late_positions = []
        for entry in log[:len(log)//4]:
            obs = entry['obs_after']
            if len(obs) > obj_start + 5:
                early_positions.append(obs[obj_start:obj_start+6])
        for entry in log[-len(log)//4:]:
            obs = entry['obs_after']
            if len(obs) > obj_start + 5:
                late_positions.append(obs[obj_start:obj_start+6])
        if not early_positions or not late_positions:
            return 0.0
        early_mean = np.mean(early_positions, axis=0)
        late_mean = np.mean(late_positions, axis=0)
        drift = float(np.linalg.norm(early_mean - late_mean))
        return float(min(1.0, drift * 2))

    tests.append(ReceptorTest('environmental_modification', 'environmental_augmentation',
        'Object positions drift over time due to organism actions', 0.15, test_environmental_modification))

    def test_push_affordance(log, engine):
        """Detect push affordance: objects move when organism is nearby and extending."""
        push_events = 0
        for i in range(2, len(log)):
            obs = log[i]['obs_after']
            prev = log[i-1]['obs_after']
            action = log[i]['action']
            if len(obs) > 140 and len(prev) > 140:
                prox_now = obs[135:138]
                prox_prev = prev[135:138]
                extending = any(action[j*3] == 1 for j in range(min(6, len(action)//3)))
                for k in range(3):
                    if prox_prev[k] > 0.3 and abs(prox_now[k] - prox_prev[k]) > 0.05 and extending:
                        push_events += 1
        if len(log) < 10:
            return 0.0
        return float(min(1.0, push_events / (len(log) * 0.02 + 1)))

    tests.append(ReceptorTest('push_affordance', 'interaction',
        'Objects move when organism extends near them', 0.2, test_push_affordance))

    # ============================================================
    # 95 additional receptor tests.
    # All channel access via idx[name] tuples or named offsets.
    # All delta comparisons use effect_delta (drift-subtracted).
    # ============================================================
    idx = compute_obs_indices()
    L = idx['total_limbs']
    emission_start = L * 3 + idx['num_joints'] * 2
    _pain = idx['pain']
    _endo = idx['endorphin']
    _temp = idx['temperature']
    _chem = idx['chemical']
    _pres = idx['pressure']
    _fat = idx['fatigue']

    def _ch(obs, channel_tuple):
        return obs[channel_tuple[0]:channel_tuple[1]]

    def _turn(action):
        return float(np.sum(action[1:L*3:3]) - np.sum(action[2:L*3:3]))

    def _extends(action):
        return np.asarray(action[0:L*3:3], dtype=float)

    def _emission(action):
        return tuple(int(b) for b in action[emission_start:emission_start+4])

    def _safe_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if len(a) < 5 or len(b) < 5 or a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    def _estimate_drift(log, cdim):
        null_deltas = [e['obs_after'][:cdim] - e['obs_before'][:cdim]
                       for e in log if action_to_hash(e['action']) == 0]
        if len(null_deltas) < 10:
            return np.zeros(cdim)
        return np.mean(null_deltas, axis=0)

    def _effect_delta(entry, drift, cdim):
        return (entry['obs_after'][:cdim] - entry['obs_before'][:cdim]) - drift

    # ------------------------------------------------------------
    # GROUP 1 -- Regulatory / Foundation
    # ------------------------------------------------------------

    def test_pain(log, engine, **kw):
        N = min(5000, len(log))
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        if pain.std() < 1e-6:
            return 0.0
        return float(np.mean(pain > 0.1))

    tests.append(ReceptorTest('pain', 'regulatory',
        'Pain channels rise above baseline near pain sources', 0.05, test_pain))

    def test_fatigue(log, engine, **kw):
        N = min(5000, len(log))
        fatigue = np.array([np.mean(_ch(e['obs_after'], _fat)) for e in log[:N]])
        act_mag = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        return _safe_corr(act_mag, fatigue)

    tests.append(ReceptorTest('fatigue', 'regulatory',
        'Fatigue accumulates with extension activity', 0.15, test_fatigue,
        needs_closed_loop=True))

    def test_receptor_propagation(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        proximity = np.array([e['obs_after'][npc_i] for e in log[:N]])
        erratic = np.array([e['obs_after'][npc_i+6] for e in log[:N]])
        empathic = np.array([e['obs_after'][npc_i+7] for e in log[:N]])
        return _safe_corr(proximity * erratic, empathic)

    tests.append(ReceptorTest('receptor_propagation', 'regulatory',
        'Empathic-pain signal tracks NPC proximity x erraticism', 0.2,
        test_receptor_propagation))

    def test_theory_of_mind(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log)); lag = 5
        if N <= lag + 20:
            return 0.0
        turn = np.array([_turn(log[i]['action']) for i in range(N - lag)])
        bearing_now = np.array([log[i]['obs_after'][npc_i+1] for i in range(N - lag)])
        bearing_future = np.array([log[i+lag]['obs_after'][npc_i+1] for i in range(N - lag)])
        return _safe_corr(turn, bearing_future) - _safe_corr(turn, bearing_now)

    tests.append(ReceptorTest('theory_of_mind', 'social',
        "Own turning anticipates future NPC bearing better than current bearing", 0.03,
        test_theory_of_mind, needs_closed_loop=True))

    def test_rhythm_entrainment(log, engine, **kw):
        N = min(3000, len(log))
        if N < 300:
            return 0.0
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        act_mag = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        if pain.std() < 1e-3:
            return 0.0
        best_lag, best_pain_corr = 0, 0.0
        for lag in (20, 40, 60, 80, 100):
            c = _safe_corr(pain[:-lag], pain[lag:])
            if abs(c) > abs(best_pain_corr):
                best_pain_corr, best_lag = c, lag
        if best_lag == 0:
            return 0.0
        return _safe_corr(act_mag[:-best_lag], act_mag[best_lag:])

    tests.append(ReceptorTest('rhythm_entrainment', 'regulatory',
        'Action rhythm autocorrelates at the same lag as the pain-field rhythm', 0.1,
        test_rhythm_entrainment, needs_closed_loop=True))

    def test_self_soothing(log, engine, **kw):
        energy_i = idx['energy']; N = min(5000, len(log))
        stressed, calm = [], []
        for i in range(N - 5):
            obs = log[i]['obs_after']
            pain_avg = np.mean(_ch(obs, _pain)); energy = obs[energy_i]
            future_activity = np.mean([np.sum(_extends(log[j]['action'])) for j in range(i+1, i+6)])
            if pain_avg > 0.3 and energy < 0.3:
                stressed.append(future_activity)
            elif pain_avg < 0.1 and energy > 0.6:
                calm.append(future_activity)
        if len(stressed) < 20 or len(calm) < 20:
            return 0.0
        return float((np.mean(calm) - np.mean(stressed)) / (np.mean(calm) + 1e-8))

    tests.append(ReceptorTest('self_soothing', 'regulatory',
        'Low-activity recovery follows high-stress states more than calm states', 0.1,
        test_self_soothing, needs_closed_loop=True))

    def test_social_coregulation(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        calm_next, other_next = [], []
        for i in range(N - 5):
            is_calm = _emission(log[i]['action']) == NPC.CALM_SIGNAL
            erratic_now = log[i]['obs_after'][npc_i+6]
            erratic_future = np.mean([log[j]['obs_after'][npc_i+6] for j in range(i+1, i+6)])
            (calm_next if is_calm else other_next).append(erratic_now - erratic_future)
        if len(calm_next) < 10 or len(other_next) < 10:
            return 0.0
        return float(np.mean(calm_next) - np.mean(other_next))

    tests.append(ReceptorTest('social_coregulation', 'regulatory',
        'Calm-signal emission reduces NPC erraticism more than other emissions', 0.05,
        test_social_coregulation, needs_closed_loop=True))

    def test_emotional_intelligence(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        is_calm = np.array([1.0 if _emission(log[i]['action']) == NPC.CALM_SIGNAL else 0.0
                            for i in range(N)])
        distress = np.array([log[i]['obs_after'][npc_i+6] for i in range(N)])
        return _safe_corr(is_calm, distress)

    tests.append(ReceptorTest('emotional_intelligence', 'regulatory',
        'Calm-signal emission correlates with NPC distress level', 0.1,
        test_emotional_intelligence, needs_closed_loop=True))

    def test_ritual_formation(log, engine, **kw):
        N = min(5000, len(log)); cdim = engine.core_obs_dim
        triples = defaultdict(list)
        for i in range(2, N):
            h = (action_to_hash(log[i-2]['action']), action_to_hash(log[i-1]['action']),
                 action_to_hash(log[i]['action']))
            triples[h].append(log[i-2]['obs_before'][:cdim])
        if len(triples) < 5:
            return 0.0
        rituals = sum(1 for v in triples.values()
                      if len(v) >= 5 and np.mean(np.std(np.array(v), axis=0)) < 0.25)
        return rituals / len(triples)

    tests.append(ReceptorTest('ritual_formation', 'regulatory',
        'Fraction of repeated 3-action sequences occurring in tight contexts', 0.05,
        test_ritual_formation, needs_closed_loop=True))

    # ------------------------------------------------------------
    # GROUP 2 -- Agency / Causality
    # ------------------------------------------------------------

    def test_tool_use(log, engine, **kw):
        obj_i = idx['obj_start']; N = min(5000, len(log))
        triggers = [i for i in range(1, N)
                    if log[i]['obs_after'][obj_i+3:obj_i+6].sum() >
                       log[i-1]['obs_after'][obj_i+3:obj_i+6].sum()]
        if len(triggers) < 5:
            return 0.0
        post = np.mean([np.mean([log[j]['reward'] for j in range(i, min(i+10, N))])
                        for i in triggers])
        baseline = np.mean([e['reward'] for e in log[:N]])
        return float((post - baseline) / (abs(baseline) + 1e-8))

    tests.append(ReceptorTest('tool_use', 'agency',
        'Reward rises after the organism triggers an object response', 0.1, test_tool_use,
        needs_closed_loop=True))

    def test_environmental_manipulation(log, engine, **kw):
        obj_i = idx['obj_start']; N = min(5000, len(log))
        prox = np.array([e['obs_after'][obj_i:obj_i+3] for e in log[:N]])
        reward = np.array([e['reward'] for e in log[:N]])
        prox_change = np.linalg.norm(np.diff(prox, axis=0), axis=1)
        return _safe_corr(prox_change, reward[1:])

    tests.append(ReceptorTest('environmental_manipulation', 'agency',
        'Object repositioning correlates with reward change', 0.1,
        test_environmental_manipulation, needs_closed_loop=True))

    def test_distributed_agency(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        signaled, unsignaled = [], []
        for i in range(N - 5):
            if log[i]['obs_after'][npc_i] <= 0.3:
                continue
            future_reward = np.mean([log[j]['reward'] for j in range(i, i+5)])
            is_signaling = sum(_emission(log[i]['action'])) > 0
            (signaled if is_signaling else unsignaled).append(future_reward)
        if len(signaled) < 20 or len(unsignaled) < 20:
            return 0.0
        return float(np.mean(signaled) - np.mean(unsignaled))

    tests.append(ReceptorTest('distributed_agency', 'agency',
        'Signaling near the NPC yields higher subsequent reward than silence', 0.05,
        test_distributed_agency, needs_closed_loop=True))

    def test_niche_construction(log, engine, **kw):
        obj_i = idx['obj_start']; N = len(log)
        if N < 200:
            return 0.0
        early, late = log[:N//10], log[-N//10:]
        drift = np.linalg.norm(
            np.mean([e['obs_after'][obj_i:obj_i+3] for e in late], axis=0) -
            np.mean([e['obs_after'][obj_i:obj_i+3] for e in early], axis=0))
        if drift < 0.05:
            return 0.0
        early_r = np.mean([e['reward'] for e in early])
        late_r = np.mean([e['reward'] for e in late])
        return float((late_r - early_r) / (abs(early_r) + 1e-8))

    tests.append(ReceptorTest('niche_construction', 'agency',
        'Persistent object-position drift accompanies later reward gain', 0.05,
        test_niche_construction, needs_closed_loop=True))

    def test_long_range_causation(log, engine, **kw):
        N = min(5000, len(log)); lag = 15
        if N <= lag + 20:
            return 0.0
        act_mag = np.array([np.sum(_extends(log[i]['action'])) for i in range(N - lag)])
        future_reward = np.array([log[i+lag]['reward'] for i in range(N - lag)])
        immediate_reward = np.array([log[i]['reward'] for i in range(N - lag)])
        return _safe_corr(act_mag, future_reward) - 0.5 * _safe_corr(act_mag, immediate_reward)

    tests.append(ReceptorTest('long_range_causation', 'agency',
        'Action intensity predicts reward >10 steps later beyond immediate reward', 0.05,
        test_long_range_causation, needs_closed_loop=True))

    def test_causal_graph_reasoning(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        sims = []
        for i in range(N - 2):
            obs0 = log[i]['obs_before'][:cdim]
            delta_chain, cert = engine.chain([log[i]['action'], log[i+1]['action']], obs0)
            if cert < 0.3:
                continue
            actual = log[i+1]['obs_after'][:cdim] - obs0
            an, dn = np.linalg.norm(actual), np.linalg.norm(delta_chain)
            if an > 0.01 and dn > 0.01:
                sims.append(np.dot(actual, delta_chain) / (an * dn))
        if len(sims) < 20:
            return 0.0
        return float(np.mean(sims))

    tests.append(ReceptorTest('causal_graph_reasoning', 'causality',
        'Two-step chained predictions match actual two-step outcomes', 0.3,
        test_causal_graph_reasoning))

    def test_common_cause_detection(log, engine, **kw):
        N = min(5000, len(log))
        chem = np.array([np.mean(_ch(e['obs_after'], _chem)) for e in log[:N]])
        press = np.array([np.mean(_ch(e['obs_after'], _pres)) for e in log[:N]])
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        joint = _safe_corr(chem, press)
        if joint < 0.2:
            return 0.0
        c1, c2 = _safe_corr(chem, pain), _safe_corr(press, pain)
        return float(joint * (1.0 - abs(c1 - c2)))

    tests.append(ReceptorTest('common_cause_detection', 'causality',
        'Two co-varying channels predict a third similarly (shared cause)', 0.1,
        test_common_cause_detection))

    def test_hidden_confounder_detection(log, engine, **kw):
        ctrl_i = idx['agency_start']; N = min(5000, len(log))
        chem = np.array([np.mean(_ch(e['obs_after'], _chem)) for e in log[:N]])
        reward = np.array([e['reward'] for e in log[:N]])
        ctrl = np.array([e['obs_after'][ctrl_i] for e in log[:N]])
        corr_chem_reward = _safe_corr(chem, reward)
        corr_ctrl_reward = _safe_corr(ctrl, reward)
        if corr_chem_reward < 0.1:
            return 0.0
        return float(max(0.0, corr_chem_reward - corr_ctrl_reward))

    tests.append(ReceptorTest('hidden_confounder_detection', 'causality',
        'Correlated-but-uncontrollable outcomes are flagged (low controllability)', 0.1,
        test_hidden_confounder_detection))

    def test_causal_chains(log, engine, **kw):
        N = min(5000, len(log)); cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        pair_d, single_d = defaultdict(list), defaultdict(list)
        for i in range(N - 1):
            h0, h1 = action_to_hash(log[i]['action']), action_to_hash(log[i+1]['action'])
            if h0 == h1:
                pair_d[h0].append((log[i+1]['obs_after'][:cdim] - log[i]['obs_before'][:cdim]) - 2 * drift)
                single_d[h0].append(_effect_delta(log[i], drift, cdim))
        sims = []
        for h, deltas in pair_d.items():
            if len(deltas) >= 5:
                pred2 = 2 * np.mean(single_d[h], axis=0)
                actual2 = np.mean(deltas, axis=0)
                pn, an = np.linalg.norm(pred2), np.linalg.norm(actual2)
                if pn > 0.01 and an > 0.01:
                    sims.append(np.dot(pred2, actual2) / (pn * an))
        if len(sims) < 3:
            return 0.0
        return float(np.mean(sims))

    tests.append(ReceptorTest('causal_chains', 'causality',
        'Repeated action pairs produce roughly double the single-step delta', 0.3,
        test_causal_chains))

    def test_counterfactual_reasoning(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        diffs = []
        for e in log[:N]:
            obs = e['obs_before'][:cdim]
            pred_taken, ct, nt = engine.predict_delta(obs, e['action'])
            pred_null, cn, nn = engine.predict_delta(obs, np.zeros_like(e['action']))
            if nt > 0 and nn > 0:
                diffs.append(np.linalg.norm(pred_taken - pred_null))
        if len(diffs) < 20:
            return 0.0
        return float(np.mean(diffs))

    tests.append(ReceptorTest('counterfactual_reasoning', 'causality',
        'Predictions differ between the action taken and the null action', 0.05,
        test_counterfactual_reasoning))

    # ------------------------------------------------------------
    # GROUP 3 -- Meta-Motivational / Epistemic
    # ------------------------------------------------------------

    def test_conflict(log, engine, **kw):
        conflict_i = idx['conflict']; N = min(5000, len(log))
        both_high = np.array([1.0 if (np.mean(_ch(e['obs_after'], _pain)) > 0.2 and
                                       np.mean(_ch(e['obs_after'], _endo)) > 0.2) else 0.0
                              for e in log[:N]])
        conflict_sig = np.array([e['obs_after'][conflict_i] for e in log[:N]])
        return _safe_corr(both_high, conflict_sig)

    tests.append(ReceptorTest('conflict', 'meta_motivational',
        'Simultaneous pain+endorphin elevation drives the conflict signal', 0.15, test_conflict))

    def test_internal_conflict_model(log, engine, **kw):
        conflict_i = idx['conflict']; N = min(5000, len(log)); lag = 5
        if N <= lag + 20:
            return 0.0
        trend = np.array([log[i]['obs_after'][conflict_i+2] for i in range(N - lag)])
        now = np.array([log[i]['obs_after'][conflict_i] for i in range(N - lag)])
        future = np.array([log[i+lag]['obs_after'][conflict_i] for i in range(N - lag)])
        return _safe_corr(trend, future - now)

    tests.append(ReceptorTest('internal_conflict_model', 'meta_motivational',
        'Conflict-trend channel predicts actual future conflict change', 0.1,
        test_internal_conflict_model))

    def test_context_conditioned_arbitration(log, engine, **kw):
        if engine.family_manager is None:
            return 0.0
        spreads = engine.family_manager.get_stats().get('weight_spreads', {})
        if len(spreads) < 2:
            return 0.0
        return float(np.mean([s['std'] for s in spreads.values()]))

    tests.append(ReceptorTest('context_conditioned_arbitration', 'meta_motivational',
        'Distinct action-family weight vectors indicate context-dependent arbitration', 0.05,
        test_context_conditioned_arbitration))

    def test_attention_control(log, engine, **kw):
        N = min(5000, len(log))
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        temp = np.array([np.mean(_ch(e['obs_after'], _temp)) for e in log[:N]])
        act = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        return abs(_safe_corr(pain, act)) - abs(_safe_corr(temp, act))

    tests.append(ReceptorTest('attention_control', 'meta_motivational',
        'Action tracks pain more strongly than an irrelevant channel', 0.1,
        test_attention_control, needs_closed_loop=True))

    def test_self_regulation(log, engine, **kw):
        energy_i = idx['energy']; N = min(5000, len(log))
        energy = np.array([e['obs_after'][energy_i] for e in log[:N]])
        act_rate = np.array([np.sum(_extends(log[i]['action'])) / L for i in range(N)])
        return _safe_corr(energy, act_rate)

    tests.append(ReceptorTest('self_regulation', 'meta_motivational',
        'Activity level drops when energy is low (self-modulated arousal)', 0.1,
        test_self_regulation, needs_closed_loop=True))

    def test_regret(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim; opt_i = idx['opt_start']
        missed = []
        for i in range(N - 3):
            obs = log[i]['obs_before'][:cdim]
            pred_taken, ct, nt = engine.predict_delta(obs, log[i]['action'])
            pred_null, cn, nn = engine.predict_delta(obs, np.zeros_like(log[i]['action']))
            if nt == 0 or nn == 0:
                continue
            taken_val = pred_taken[6:12].sum() - pred_taken[0:6].sum()
            null_val = pred_null[6:12].sum() - pred_null[0:6].sum()
            if null_val > taken_val + 0.1:
                missed.append(log[i+3]['obs_after'][opt_i] - log[i]['obs_after'][opt_i])
        if len(missed) < 15:
            return 0.0
        return float(-np.mean(missed))

    tests.append(ReceptorTest('regret', 'meta_motivational',
        'Optimism drops after missing a better available null-action outcome', 0.02,
        test_regret, needs_closed_loop=True))

    def test_long_term_planning(log, engine, **kw):
        N = min(5000, len(log)); lag = 10
        if N <= lag + 20:
            return 0.0
        pain_now = np.array([np.mean(_ch(log[i]['obs_after'], _pain)) for i in range(N - lag)])
        endo_future = np.array([np.mean(_ch(log[i+lag]['obs_after'], _endo)) for i in range(N - lag)])
        return _safe_corr(pain_now, endo_future)

    tests.append(ReceptorTest('long_term_planning', 'meta_motivational',
        'Accepting pain now correlates with endorphin gain later', 0.05,
        test_long_term_planning, needs_closed_loop=True))

    def test_meta_planning(log, engine, **kw):
        plan_i = idx['agency_start'] + 2; conflict_i = idx['conflict']; N = min(5000, len(log))
        conflict = np.array([e['obs_after'][conflict_i] for e in log[:N]])
        plan = np.array([e['obs_after'][plan_i] for e in log[:N]])
        return _safe_corr(conflict, plan)

    tests.append(ReceptorTest('meta_planning', 'meta_motivational',
        'Planning-value signal rises specifically when conflict is high', 0.1,
        test_meta_planning))

    def test_multiple_receptor_types(log, engine, **kw):
        N = min(5000, len(log))
        groups = [(k*L, (k+1)*L) for k in range(6)]
        active_counts = [sum(1 for a, b in groups if np.mean(e['obs_after'][a:b]) > 0.15)
                         for e in log[:N]]
        return float(np.mean([1.0 if c >= 3 else 0.0 for c in active_counts]))

    tests.append(ReceptorTest('multiple_receptor_types', 'meta_motivational',
        'Three or more receptor groups are simultaneously active', 0.1,
        test_multiple_receptor_types))

    def test_belief_detection(log, engine, **kw):
        cert_i = idx['mm_start'] + 2; N = min(1500, len(log)); cdim = engine.core_obs_dim
        certs, errors = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            errors.append(np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2))
            certs.append(e['obs_after'][cert_i])
        if len(certs) < 30:
            return 0.0
        return -_safe_corr(certs, errors)

    tests.append(ReceptorTest('belief_detection', 'epistemic',
        'Self-reported certainty is negatively correlated with prediction error', 0.1,
        test_belief_detection))

    def test_doubt_detection(log, engine, **kw):
        persist_i = idx['opt_start'] + 1; N = min(1500, len(log)); cdim = engine.core_obs_dim
        mismatch, total = 0, 0
        for e in log[:N]:
            if e['obs_after'][persist_i] < 0.3:
                continue
            total += 1
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            err = np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2)
            if err > 0.05:
                mismatch += 1
        return mismatch / (total + 1e-8)

    tests.append(ReceptorTest('doubt_detection', 'epistemic',
        'Goal persistence continues despite high prediction error (commitment mismatch)', 0.05,
        test_doubt_detection, needs_closed_loop=True))

    def test_counterfactual_salience(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim; opt_i = idx['opt_start']
        foregone, optimism = [], []
        for e in log[:N]:
            pred_null, cn, nn = engine.predict_delta(e['obs_before'][:cdim], np.zeros_like(e['action']))
            if nn == 0:
                continue
            foregone.append(pred_null[6:12].sum() - pred_null[0:6].sum())
            optimism.append(e['obs_after'][opt_i])
        if len(foregone) < 30:
            return 0.0
        return _safe_corr(foregone, optimism)

    tests.append(ReceptorTest('counterfactual_salience', 'epistemic',
        'Optimism tracks the value of the unchosen null action', 0.05,
        test_counterfactual_salience))

    def test_epistemic_strategy(log, engine, **kw):
        cert_i = idx['mm_start'] + 2; N = min(5000, len(log)); w = 5
        if N <= w + 20:
            return 0.0
        certs = np.array([log[i]['obs_after'][cert_i] for i in range(N - w)])
        diversity = np.array([len({action_to_hash(log[j]['action']) for j in range(i, i+w)}) / w
                              for i in range(N - w)])
        return -_safe_corr(certs, diversity)

    tests.append(ReceptorTest('epistemic_strategy', 'epistemic',
        'Low certainty precedes more diverse (exploratory) action choices', 0.05,
        test_epistemic_strategy, needs_closed_loop=True))

    # ------------------------------------------------------------
    # GROUP 4 -- Compression / Sequential
    # ------------------------------------------------------------

    def test_analogy(log, engine, **kw):
        entries = [e for entries in engine.store.mappings.values() for e in entries][:250]
        if len(entries) < 30:
            return 0.0
        hits, total = 0, 0
        for a, b in islice(combinations(entries, 2), 3000):
            if np.linalg.norm(a.context_embedding - b.context_embedding) < 1.0:
                continue
            total += 1
            dn_a, dn_b = np.linalg.norm(a.delta), np.linalg.norm(b.delta)
            if dn_a > 0.01 and dn_b > 0.01 and np.dot(a.delta, b.delta)/(dn_a*dn_b) > 0.7:
                hits += 1
        return hits / (total + 1e-8)

    tests.append(ReceptorTest('analogy', 'compression',
        'Distant contexts (different embeddings) still produce similar outcome deltas', 0.03,
        test_analogy))

    def test_concept_grounding(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        concepts = engine.pattern_store.extract_concepts(top_k=20)
        if len(concepts) < 5:
            return 0.0
        ratios = [np.linalg.norm(pat.cumulative_delta[0:2*L]) /
                  (np.linalg.norm(pat.cumulative_delta) + 1e-8) for _, _, pat in concepts]
        return float(np.mean(ratios))

    tests.append(ReceptorTest('concept_grounding', 'compression',
        'Stable concepts project onto core pain/endorphin channels', 0.3,
        test_concept_grounding))

    def test_language_grounding(log, engine, **kw):
        N = min(5000, len(log))
        pain_high = np.array([1.0 if np.mean(_ch(e['obs_after'], _pain)) > 0.3 else 0.0 for e in log[:N]])
        repel = np.array([1.0 if _emission(log[i]['action']) == NPC.REPEL_SIGNAL else 0.0
                          for i in range(N)])
        return _safe_corr(pain_high, repel)

    tests.append(ReceptorTest('language_grounding', 'compression',
        "Repel-signal emission tracks the organism's own pain state", 0.05,
        test_language_grounding, needs_closed_loop=True))

    def test_mental_model(log, engine, **kw):
        N = min(3000, len(log)); cdim = engine.core_obs_dim
        used = 0
        for e in log[:N]:
            _, cert, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n > 0 and cert > 0.3:
                used += 1
        return used / N

    tests.append(ReceptorTest('mental_model', 'compression',
        'Fraction of steps with a confident retrieved mapping', 0.3, test_mental_model))

    def test_simplified_shared_signals(log, engine, **kw):
        N = min(5000, len(log))
        active = [_emission(log[i]['action']) for i in range(N) if any(_emission(log[i]['action']))]
        if len(active) < 20:
            return 0.0
        counts = defaultdict(int)
        for e in active:
            counts[e] += 1
        return float(max(counts.values()) / len(active))

    tests.append(ReceptorTest('simplified_shared_signals', 'compression',
        'Emission usage concentrates on a small number of signal patterns', 0.5,
        test_simplified_shared_signals, needs_closed_loop=True))

    def test_stage_prediction(log, engine, **kw):
        pe_i = idx['pe_start']; N = min(5000, len(log))
        pe = np.array([np.mean(e['obs_after'][pe_i:pe_i+L]) for e in log[:N]])
        act = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        return _safe_corr(pe, act)

    tests.append(ReceptorTest('stage_prediction', 'sequential_processing',
        'Action responds to same-step prediction error (partial-information acting)', 0.1,
        test_stage_prediction, needs_closed_loop=True))

    def test_pipeline_detection(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        N = min(1500, len(log)); cdim = engine.core_obs_dim; fam_i = idx['mm_start']
        novelty, depth = [], []
        for i in range(1, N):
            emb = engine.encoder.embed(log[i]['obs_before'][:cdim])
            results = engine.pattern_store.query(action_to_hash(log[i-1]['action']), emb, top_k=10)
            novelty.append(1.0 - log[i]['obs_after'][fam_i])
            depth.append(len(results))
        return _safe_corr(novelty, depth)

    tests.append(ReceptorTest('pipeline_detection', 'sequential_processing',
        'Processing depth (pattern matches searched) grows with novelty', 0.1,
        test_pipeline_detection))

    def test_cross_pipeline_prediction(log, engine, **kw):
        N = min(5000, len(log)); lag = 3
        if N <= lag + 20:
            return 0.0
        pain_delta = np.array([np.mean(_ch(log[i]['obs_after'], _pain)) - np.mean(_ch(log[i]['obs_before'], _pain))
                               for i in range(N - lag)])
        fatigue_future = np.array([np.mean(_ch(log[i+lag]['obs_after'], _fat)) for i in range(N - lag)])
        return _safe_corr(pain_delta, fatigue_future)

    tests.append(ReceptorTest('cross_pipeline_prediction', 'sequential_processing',
        'Pain-channel change predicts fatigue-channel state a few steps later', 0.1,
        test_cross_pipeline_prediction))

    def test_pipeline_optimization(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        seen = defaultdict(int); first, later = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            err = np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2)
            h = action_to_hash(e['action']); seen[h] += 1
            (first if seen[h] <= 2 else later).append(err)
        if len(first) < 20 or len(later) < 20:
            return 0.0
        return float((np.mean(first) - np.mean(later)) / (np.mean(first) + 1e-8))

    tests.append(ReceptorTest('pipeline_optimization', 'sequential_processing',
        'Prediction error for a repeated action shrinks with repetition', 0.05,
        test_pipeline_optimization))

    def test_prediction_architecture_awareness(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        concepts = engine.pattern_store.extract_concepts(top_k=30)
        if len(concepts) < 5:
            return 0.0
        return sum(1 for _, motif, _ in concepts if motif[0] != motif[1]) / len(concepts)

    tests.append(ReceptorTest('prediction_architecture_awareness', 'sequential_processing',
        'Stable concepts include novel (non-repeated-action) motifs', 0.3,
        test_prediction_architecture_awareness))

    def test_analogy_receptor(log, engine, **kw):
        reps = [(h, es[0]) for h, es in engine.store.mappings.items() if es][:200]
        if len(reps) < 10:
            return 0.0
        sims_emb, sims_delta = [], []
        for (h1, e1), (h2, e2) in islice(combinations(reps, 2), 4000):
            if h1 == h2:
                continue
            dn1, dn2 = np.linalg.norm(e1.delta), np.linalg.norm(e2.delta)
            if dn1 < 0.01 or dn2 < 0.01:
                continue
            sims_emb.append(np.dot(e1.context_embedding, e2.context_embedding))
            sims_delta.append(np.dot(e1.delta, e2.delta) / (dn1 * dn2))
        return _safe_corr(sims_emb, sims_delta)

    tests.append(ReceptorTest('analogy_receptor', 'compression',
        'Context-embedding similarity across actions predicts outcome similarity', 0.1,
        test_analogy_receptor))

    def test_prediction_error(log, engine, **kw):
        pe_i = idx['pe_start']; lp_i = idx['mm_start'] + 3; N = min(5000, len(log))
        pe = np.array([np.mean(e['obs_after'][pe_i:pe_i+L]) for e in log[:N]])
        lp = np.array([e['obs_after'][lp_i] for e in log[:N]])
        return _safe_corr(pe, lp)

    tests.append(ReceptorTest('prediction_error', 'sequential_processing',
        'Prediction-error channel correlates with subsequent learning progress', 0.1,
        test_prediction_error))

    def test_prediction(log, engine, **kw):
        N = min(3000, len(log)); cdim = engine.core_obs_dim
        good, total = 0, 0
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            total += 1
            actual = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            pn, an = np.linalg.norm(pred), np.linalg.norm(actual)
            if pn > 0.01 and an > 0.01 and np.dot(pred, actual)/(pn*an) > 0.3:
                good += 1
        return good / (total + 1e-8)

    tests.append(ReceptorTest('prediction', 'compression',
        'Generated predictions are directionally accurate above chance', 0.3, test_prediction))

    def test_metacognition(log, engine, **kw):
        cert_i = idx['mm_start'] + 2; N = min(1500, len(log)); cdim = engine.core_obs_dim
        certs, ns = [], []
        for e in log[:N]:
            _, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            certs.append(e['obs_after'][cert_i]); ns.append(n)
        return _safe_corr(certs, ns)

    tests.append(ReceptorTest('metacognition', 'meta_motivational',
        'Self-reported certainty tracks the amount of retrieved evidence', 0.1,
        test_metacognition))

    def test_organizational_mirror(log, engine, **kw):
        dev_i = idx['limb_dev_start']; N = min(5000, len(log))
        dev_var = np.array([np.std(e['obs_after'][dev_i:dev_i+L]) for e in log[:N]])
        act_var = np.array([np.std(_extends(log[i]['action'])) for i in range(N)])
        return _safe_corr(dev_var, act_var)

    tests.append(ReceptorTest('organizational_mirror', 'organization',
        'Perceived limb-deviation spread matches actual action-coordination spread', 0.15,
        test_organizational_mirror, needs_closed_loop=True))

    # ------------------------------------------------------------
    # GROUP 5 -- Similarity / Association / Repetition
    # ------------------------------------------------------------

    def test_abstract_association(log, engine, **kw):
        entries = [e for entries in engine.store.mappings.values() for e in entries][:120]
        if len(entries) < 20:
            return 0.0
        rel_e, rel_d = [], []
        for a, b in islice(combinations(entries, 2), 2000):
            rel_e.append(a.context_embedding - b.context_embedding)
            rel_d.append(a.delta - b.delta)
        sims = []
        for i, j in islice(combinations(range(len(rel_e)), 2), 3000):
            en1, en2 = np.linalg.norm(rel_e[i]), np.linalg.norm(rel_e[j])
            dn1, dn2 = np.linalg.norm(rel_d[i]), np.linalg.norm(rel_d[j])
            if en1 < 0.1 or en2 < 0.1 or dn1 < 0.01 or dn2 < 0.01:
                continue
            if np.dot(rel_e[i], rel_e[j])/(en1*en2) > 0.7:
                sims.append(np.dot(rel_d[i], rel_d[j])/(dn1*dn2))
        if len(sims) < 10:
            return 0.0
        return float(np.mean(sims))

    tests.append(ReceptorTest('abstract_association', 'association',
        'Similar embedding relations imply similar outcome relations', 0.2,
        test_abstract_association))

    def test_analogical_similarity(log, engine, **kw):
        reps = [(h, np.mean([e.delta for e in es], axis=0))
                for h, es in engine.store.mappings.items() if len(es) >= 3]
        if len(reps) < 4:
            return 0.0
        diffs = [d1 - d2 for (h1, d1), (h2, d2) in combinations(reps, 2)]
        hits, total = 0, 0
        for dv1, dv2 in islice(combinations(diffs, 2), 5000):
            n1, n2 = np.linalg.norm(dv1), np.linalg.norm(dv2)
            if n1 < 0.01 or n2 < 0.01:
                continue
            total += 1
            if np.dot(dv1, dv2)/(n1*n2) > 0.6:
                hits += 1
        return hits / (total + 1e-8)

    tests.append(ReceptorTest('analogical_similarity', 'similarity',
        'A:B::C:D style delta-difference pairs align across action pairs', 0.05,
        test_analogical_similarity))

    def test_structural_similarity(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        entries = [e for entries in engine.store.mappings.values() for e in entries][:250]
        if len(entries) < 30:
            return 0.0
        corrected = [np.sign(e.delta - drift[:len(e.delta)]) for e in entries]
        matches = [np.mean(corrected[a] == corrected[b])
                   for a, b in islice(combinations(range(len(corrected)), 2), 4000)]
        return float(np.mean(matches)) if matches else 0.0

    tests.append(ReceptorTest('structural_similarity', 'similarity',
        'Delta sign-patterns agree across contexts more than expected by chance', 0.6,
        test_structural_similarity))

    def test_structural_invariance(log, engine, **kw):
        consistent, total = 0, 0
        for h, es in engine.store.mappings.items():
            if len(es) < 4:
                continue
            total += 1
            signs = np.array([np.sign(e.delta) for e in es])
            if np.mean(np.abs(signs.mean(axis=0))) > 0.7:
                consistent += 1
        return consistent / (total + 1e-8)

    tests.append(ReceptorTest('structural_invariance', 'similarity',
        'Same action produces sign-consistent deltas across varied contexts', 0.3,
        test_structural_invariance))

    def test_prototype_formation(log, engine, **kw):
        dists_high, dists_low = [], []
        for h, es in engine.store.mappings.items():
            if len(es) < 5:
                continue
            centroid = np.mean([e.context_embedding for e in es], axis=0)
            for e in es:
                d = np.linalg.norm(e.context_embedding - centroid)
                (dists_high if e.count >= 5 else dists_low).append(d)
        if len(dists_high) < 20 or len(dists_low) < 20:
            return 0.0
        return float((np.mean(dists_low) - np.mean(dists_high)) / (np.mean(dists_low) + 1e-8))

    tests.append(ReceptorTest('prototype_formation', 'similarity',
        'Frequently-confirmed entries cluster tighter around their group centroid', 0.1,
        test_prototype_formation))

    def test_basic_sensorimotor_loop(log, engine, **kw):
        N = min(5000, len(log)); cdim = engine.core_obs_dim
        act_mag = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        obs_delta = np.array([np.linalg.norm(log[i]['obs_after'][:cdim] - log[i]['obs_before'][:cdim])
                              for i in range(N)])
        return _safe_corr(act_mag, obs_delta)

    tests.append(ReceptorTest('basic_sensorimotor_loop', 'association',
        'Action magnitude correlates with observation change magnitude', 0.15,
        test_basic_sensorimotor_loop, needs_closed_loop=True))

    def test_relational_analogy(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        pain_rel = np.array([np.mean(_ch(log[i]['obs_after'], _pain)) - np.mean(_ch(log[i]['obs_before'], _pain))
                             for i in range(N)])
        npc_rel = np.array([log[i]['obs_after'][npc_i] - log[i]['obs_before'][npc_i] for i in range(N)])
        return _safe_corr(pain_rel, npc_rel)

    tests.append(ReceptorTest('relational_analogy', 'association',
        'Change-relations transfer between the pain domain and the NPC domain', 0.05,
        test_relational_analogy))

    def test_multiple_sensor_modalities(log, engine, **kw):
        N = min(5000, len(log))
        temp = np.array([np.mean(_ch(e['obs_after'], _temp)) for e in log[:N]])
        chem = np.array([np.mean(_ch(e['obs_after'], _chem)) for e in log[:N]])
        reward = np.array([e['reward'] for e in log[:N]])
        c_temp, c_chem = _safe_corr(temp, reward), _safe_corr(chem, reward)
        combo = _safe_corr(temp + chem, reward)
        return float(abs(combo) - max(abs(c_temp), abs(c_chem)))

    tests.append(ReceptorTest('multiple_sensor_modalities', 'association',
        'Combined channels predict reward better than either channel alone', 0.02,
        test_multiple_sensor_modalities))

    def test_causal_rhythm(log, engine, **kw):
        N = min(3000, len(log))
        if N < 300:
            return 0.0
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        reward = np.array([e['reward'] for e in log[:N]])
        best_lag, best_c = 0, 0.0
        for lag in (20, 40, 60, 80, 100):
            c = _safe_corr(pain[:-lag], pain[lag:])
            if abs(c) > abs(best_c):
                best_c, best_lag = c, lag
        if best_lag == 0:
            return 0.0
        return _safe_corr(reward[:-best_lag], reward[best_lag:])

    tests.append(ReceptorTest('causal_rhythm', 'repetition',
        'Reward is periodic at the same lag as the pain-field rhythm', 0.1, test_causal_rhythm))

    def test_rhythmic_pattern(log, engine, **kw):
        N = min(3000, len(log))
        if N < 400:
            return 0.0
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        strong = sum(1 for lag in (10, 20, 30, 40, 50, 60, 70, 80)
                    if abs(_safe_corr(pain[:-lag], pain[lag:])) > 0.2)
        return strong / 8.0

    tests.append(ReceptorTest('rhythmic_pattern', 'repetition',
        'Multiple distinct lags show significant pain autocorrelation', 0.25,
        test_rhythmic_pattern))

    def test_nested_rhythm(log, engine, **kw):
        N = min(4000, len(log)); w = 20
        if N < w * 10:
            return 0.0
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        envelope = np.array([np.std(pain[i:i+w]) for i in range(0, N - w, w)])
        if len(envelope) < 20:
            return 0.0
        return _safe_corr(envelope[:-3], envelope[3:])

    tests.append(ReceptorTest('nested_rhythm', 'repetition',
        'The envelope of a fast rhythm is itself autocorrelated (rhythm of rhythms)', 0.15,
        test_nested_rhythm))

    def test_hierarchical_abstraction(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        pats = [p for pats in engine.pattern_store.patterns.values() for p in pats]
        if len(pats) < 10:
            return 0.0
        counts = [p.count for p in pats]; gains = [p.compression_gain for p in pats]
        return _safe_corr(counts, gains)

    tests.append(ReceptorTest('hierarchical_abstraction', 'compression',
        'More-repeated motifs achieve higher compression gain', 0.15,
        test_hierarchical_abstraction))

    # ------------------------------------------------------------
    # GROUP 6 -- Organization / Mathematics / Perception /
    #            Self-Augmentation / Interaction / Environmental
    # ------------------------------------------------------------

    def test_hierarchical_structure_detection(log, engine, **kw):
        speed_i = idx['proprio_start']; N = min(5000, len(log))
        speed = np.array([e['obs_after'][speed_i] for e in log[:N]])
        extends = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        return _safe_corr(extends, speed)

    tests.append(ReceptorTest('hierarchical_structure_detection', 'organization',
        'Coordinated limb extension (part) drives whole-body speed (whole)', 0.2,
        test_hierarchical_structure_detection, needs_closed_loop=True))

    def test_relational_structure_detection(log, engine, **kw):
        dev_i = idx['limb_dev_start']; N = min(5000, len(log))
        dev_spread = np.array([np.std(e['obs_after'][dev_i:dev_i+L]) for e in log[:N]])
        act_spread = np.array([np.std(_extends(log[i]['action'])) for i in range(N)])
        return -_safe_corr(dev_spread, act_spread)

    tests.append(ReceptorTest('relational_structure_detection', 'organization',
        'Low limb-deviation spread accompanies coordinated (low-spread) actions', 0.15,
        test_relational_structure_detection, needs_closed_loop=True))

    def test_system_detection(log, engine, **kw):
        N = min(5000, len(log))
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        fatigue = np.array([np.mean(_ch(e['obs_after'], _fat)) for e in log[:N]])
        reward = np.array([e['reward'] for e in log[:N]])
        c1, c2 = _safe_corr(pain, reward), _safe_corr(fatigue, reward)
        combo = _safe_corr(pain * fatigue, reward)
        return float(abs(combo) - max(abs(c1), abs(c2)))

    tests.append(ReceptorTest('system_detection', 'organization',
        'Pain and fatigue jointly explain reward better than either alone', 0.02,
        test_system_detection))

    def test_exhaustive_search(log, engine, **kw):
        pm_i = idx['pain_memory_start']; N = min(5000, len(log))
        if N < 500:
            return 0.0
        visited = np.zeros(25)
        for e in log[:N]:
            visited += (np.abs(e['obs_after'][pm_i:pm_i+25]) > 1e-6)
        return float(np.mean(visited > 0))

    tests.append(ReceptorTest('exhaustive_search', 'mathematics',
        'Fraction of the spatial memory grid visited (search coverage)', 0.4,
        test_exhaustive_search, needs_closed_loop=True))

    def test_proof_structure(log, engine, **kw):
        N = min(1000, len(log)); cdim = engine.core_obs_dim
        high_chain, total = 0, 0
        for i in range(N - 2):
            obs = log[i]['obs_before'][:cdim]
            _, c1, n1 = engine.predict_delta(obs, log[i]['action'])
            _, c2, n2 = engine.predict_delta(log[i]['obs_after'][:cdim], log[i+1]['action'])
            if n1 == 0 or n2 == 0 or c1 < 0.7 or c2 < 0.7:
                continue
            total += 1
            _, chain_c = engine.chain([log[i]['action'], log[i+1]['action']], obs)
            if chain_c > 0.5:
                high_chain += 1
        return high_chain / (total + 1e-8)

    tests.append(ReceptorTest('proof_structure', 'mathematics',
        'Chaining two high-certainty steps preserves high certainty (necessity propagates)', 0.3,
        test_proof_structure))

    def test_necessity_detection(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        pats = [p for pats in engine.pattern_store.patterns.values() for p in pats]
        necessary = [p for p in pats if p.certainty > 0.85]
        contingent = [p for p in pats if p.certainty < 0.5]
        if len(necessary) < 3 or len(contingent) < 3:
            return 0.0
        return len(necessary) / (len(necessary) + len(contingent))

    tests.append(ReceptorTest('necessity_detection', 'mathematics',
        'Some patterns are near-certain (necessary) vs. clearly contingent ones', 0.3,
        test_necessity_detection))

    def test_formal_composition(log, engine, **kw):
        N = min(1000, len(log)); cdim = engine.core_obs_dim
        diffs = []
        for i in range(N - 1):
            obs = log[i]['obs_before'][:cdim]
            d1, c1, n1 = engine.predict_delta(obs, log[i]['action'])
            if n1 == 0:
                continue
            chain_d, _ = engine.chain([log[i]['action'], log[i+1]['action']], obs)
            diffs.append(np.linalg.norm(chain_d - d1))
        if len(diffs) < 20:
            return 0.0
        return float(np.mean(diffs))

    tests.append(ReceptorTest('formal_composition', 'mathematics',
        'Composing a second action changes the predicted outcome non-trivially', 0.1,
        test_formal_composition))

    def test_multiple_hypotheses(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        multi, total = 0, 0
        for i in range(1, N):
            emb = engine.encoder.embed(log[i]['obs_before'][:cdim])
            results = engine.pattern_store.query(action_to_hash(log[i-1]['action']), emb, top_k=5)
            if not results:
                continue
            total += 1
            if sum(1 for _, s in results if s > 0.3) >= 2:
                multi += 1
        return multi / (total + 1e-8)

    tests.append(ReceptorTest('multiple_hypotheses', 'causality',
        'More than one plausible pattern match is retrieved at once', 0.05,
        test_multiple_hypotheses))

    def test_adaptive_depth(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        N = min(1500, len(log)); cdim = engine.core_obs_dim; pe_i = idx['pe_start']
        complexity, depth = [], []
        for i in range(1, N):
            emb = engine.encoder.embed(log[i]['obs_before'][:cdim])
            results = engine.pattern_store.query(action_to_hash(log[i-1]['action']), emb, top_k=10)
            complexity.append(np.mean(log[i]['obs_after'][pe_i:pe_i+L]))
            depth.append(len(results))
        return _safe_corr(complexity, depth)

    tests.append(ReceptorTest('adaptive_depth', 'perception',
        'Search depth (matches retrieved) grows with input complexity', 0.1,
        test_adaptive_depth))

    def test_prediction_branching(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        errors, n_results = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            errors.append(np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2))
            n_results.append(n)
        return _safe_corr(errors, n_results)

    tests.append(ReceptorTest('prediction_branching', 'perception',
        'Higher prediction error is met with a broader evidence search', 0.1,
        test_prediction_branching))

    def test_processing_speed_receptor(log, engine, **kw):
        fam_i = idx['mm_start']; N = min(1500, len(log)); cdim = engine.core_obs_dim
        fam, err = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            err.append(np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2))
            fam.append(e['obs_after'][fam_i])
        return -_safe_corr(fam, err)

    tests.append(ReceptorTest('processing_speed_receptor', 'perception',
        'Familiar contexts (cheap to process) show lower prediction error', 0.1,
        test_processing_speed_receptor))

    def test_response_loop_detection(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        errs = []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            errs.append(np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2)
                       if n > 0 else np.nan)
        errs = np.array(errs)
        valid = errs[~np.isnan(errs)]
        if len(valid) < 100:
            return 0.0
        c1 = _safe_corr(valid[:-1], valid[1:])
        c5 = _safe_corr(valid[:-5], valid[5:])
        return float(c1 - c5)

    tests.append(ReceptorTest('response_loop_detection', 'perception',
        'Prediction error decorrelates faster than a slow-drift baseline (correction)', 0.02,
        test_response_loop_detection))

    def test_staged_processing(log, engine, **kw):
        fam_i = idx['mm_start']; pat_i = idx['pattern_start']
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        both, either = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            err = np.mean((pred - (e['obs_after'][:cdim] - e['obs_before'][:cdim]))**2)
            fam_hi, pat_hi = e['obs_after'][fam_i] > 0.5, e['obs_after'][pat_i] > 0.5
            if fam_hi and pat_hi:
                both.append(err)
            elif fam_hi != pat_hi:
                either.append(err)
        if len(both) < 10 or len(either) < 10:
            return 0.0
        return float((np.mean(either) - np.mean(both)) / (np.mean(either) + 1e-8))

    tests.append(ReceptorTest('staged_processing', 'perception',
        'Error is lowest only when both retrieval stages agree (staged pipeline)', 0.05,
        test_staged_processing))

    def test_growth_tracking(log, engine, **kw):
        gain_i = idx['gain_start']; N = min(5000, len(log)); lag = 10
        if N <= lag + 20:
            return 0.0
        gain_now = np.array([np.mean(log[i]['obs_after'][gain_i:gain_i+L]) for i in range(N - lag)])
        gain_future = np.array([np.mean(log[i+lag]['obs_after'][gain_i:gain_i+L]) for i in range(N - lag)])
        reward_future = np.array([np.mean([log[j]['reward'] for j in range(i, i+lag)])
                                  for i in range(N - lag)])
        return _safe_corr(gain_future - gain_now, reward_future)

    tests.append(ReceptorTest('growth_tracking', 'self_augmentation',
        'Gains in receptor sensitivity accompany gains in subsequent reward', 0.05,
        test_growth_tracking))

    def test_developmental_trajectory(log, engine, **kw):
        gain_i = idx['gain_start']; N = min(5000, len(log)); w = 10
        if N < w * 3:
            return 0.0
        gain = np.array([np.mean(e['obs_after'][gain_i:gain_i+L]) for e in log[:N]])
        past_slope = gain[w:N-w] - gain[0:N-2*w]
        future_delta = gain[2*w:N] - gain[w:N-w]
        return _safe_corr(past_slope, future_delta)

    tests.append(ReceptorTest('developmental_trajectory', 'self_augmentation',
        'Past capability trend predicts future capability change', 0.1,
        test_developmental_trajectory))

    def test_identity_continuity(log, engine, **kw):
        cdim = engine.core_obs_dim; N = len(log)
        if N < 500:
            return 0.0
        quarter = N // 4
        action_profiles = []
        for start in [0, quarter, 2*quarter, 3*quarter]:
            chunk = log[start:start+quarter]
            profile = defaultdict(int)
            for e in chunk:
                profile[action_to_hash(e['action'])] += 1
            action_profiles.append(profile)
        all_hashes = set()
        for p in action_profiles:
            all_hashes.update(p.keys())
        vectors = []
        for p in action_profiles:
            v = np.array([p.get(h, 0) for h in sorted(all_hashes)], dtype=float)
            v = v / (v.sum() + 1e-8)
            vectors.append(v)
        sims = [np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8)
                for i in range(4) for j in range(i+1, 4)]
        return float(np.mean(sims))

    tests.append(ReceptorTest('identity_continuity', 'self_augmentation',
        'Action distribution stays consistent across quarters of the log', 0.5,
        test_identity_continuity))

    def test_metamorphic_planning(log, engine, **kw):
        plan_i = idx['agency_start'] + 2; gain_i = idx['gain_start']
        N = min(5000, len(log)); lag = 10
        if N <= lag + 20:
            return 0.0
        plan = np.array([log[i]['obs_after'][plan_i] for i in range(N - lag)])
        gain_change = np.array([np.mean(log[i+lag]['obs_after'][gain_i:gain_i+L]) -
                                np.mean(log[i]['obs_after'][gain_i:gain_i+L]) for i in range(N - lag)])
        return _safe_corr(plan, gain_change)

    tests.append(ReceptorTest('metamorphic_planning', 'self_augmentation',
        "Planning-value signal tracks the organism's own capability trajectory", 0.05,
        test_metamorphic_planning))

    def test_intervention_planning(log, engine, **kw):
        ctrl_i = idx['agency_start']; N = min(1500, len(log)); cdim = engine.core_obs_dim
        ctrl, pain_reduction = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            ctrl.append(e['obs_after'][ctrl_i])
            pain_reduction.append(-pred[0:6].sum())
        return _safe_corr(ctrl, pain_reduction)

    tests.append(ReceptorTest('intervention_planning', 'causality',
        'High controllability accompanies actions predicted to reduce pain', 0.1,
        test_intervention_planning, needs_closed_loop=True))

    def test_affordance_transfer(log, engine, **kw):
        obj_i = idx['obj_start']; N = min(5000, len(log))
        effects = [[], [], []]
        for i in range(1, N):
            if not any(_extends(log[i]['action'])):
                continue
            for k in range(3):
                prev = log[i-1]['obs_after'][obj_i+k]
                if prev > 0.2:
                    effects[k].append(abs(log[i]['obs_after'][obj_i+k] - prev))
        valid = [np.mean(e) for e in effects if len(e) >= 10]
        if len(valid) < 2:
            return 0.0
        return float(min(valid) / (max(valid) + 1e-8))

    tests.append(ReceptorTest('affordance_transfer', 'interaction',
        'Push-like effects generalize across different sensed objects', 0.3,
        test_affordance_transfer, needs_closed_loop=True))

    def test_composite_affordance(log, engine, **kw):
        grip_i = idx['grip_start']; obj_i = idx['obj_start']; N = min(5000, len(log))
        grip_active = np.array([1.0 if e['obs_after'][grip_i:grip_i+L].sum() > 0 else 0.0
                                for e in log[:N]])
        prox_change = np.array([abs(log[i]['obs_after'][obj_i] - log[i-1]['obs_after'][obj_i])
                                if i > 0 else 0.0 for i in range(N)])
        return _safe_corr(grip_active, prox_change)

    tests.append(ReceptorTest('composite_affordance', 'interaction',
        'Grip and push affordances co-occur more than chance', 0.1,
        test_composite_affordance, needs_closed_loop=True))

    def test_contact_response_detection(log, engine, **kw):
        force_i = idx['grip_start'] + L + 2; N = min(5000, len(log))
        if len(log[0]['obs_after']) <= force_i:
            return 0.0
        force = np.array([e['obs_after'][force_i] for e in log[:N]])
        act_change = np.array([np.sum(np.abs(np.array(log[i]['action']) - np.array(log[i-1]['action'])))
                               if i > 0 else 0.0 for i in range(N)])
        return _safe_corr(force, act_change)

    tests.append(ReceptorTest('contact_response_detection', 'interaction',
        'Contact-force magnitude correlates with the size of the action change', 0.15,
        test_contact_response_detection, needs_closed_loop=True))

    def test_proprioception(log, engine, **kw):
        speed_i = idx['proprio_start']; N = min(5000, len(log))
        speed = np.array([e['obs_after'][speed_i] for e in log[:N]])
        extends = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        return _safe_corr(speed, extends)

    tests.append(ReceptorTest('proprioception', 'interaction',
        'Self-motion speed signal reflects actual motor output', 0.2,
        test_proprioception, needs_closed_loop=True))

    def test_spatial_memory(log, engine, **kw):
        pm_i = idx['pain_memory_start']; N = len(log)
        if N < 600:
            return 0.0
        early = log[N//2 - 100]['obs_after'][pm_i:pm_i+25]
        late = log[N//2 + 100]['obs_after'][pm_i:pm_i+25]
        active = np.abs(early) > 0.1
        if active.sum() < 3:
            return 0.0
        return float(np.sum(active & (np.abs(late) > 0.1)) / active.sum())

    tests.append(ReceptorTest('spatial_memory', 'environmental_augmentation',
        'Pain-memory grid cells stay marked across a long temporal gap', 0.3,
        test_spatial_memory))

    def test_environmental_change_detection(log, engine, **kw):
        obj_i = idx['obj_start']; N = min(5000, len(log))
        resp = np.array([np.sum(np.abs(np.array(log[i]['obs_after'][obj_i+3:obj_i+6]) -
                                       np.array(log[i-1]['obs_after'][obj_i+3:obj_i+6])))
                        if i > 0 else 0.0 for i in range(N)])
        act = np.array([np.sum(np.abs(np.array(log[i]['action']) - np.array(log[i-1]['action'])))
                       if i > 0 else 0.0 for i in range(N)])
        return _safe_corr(resp, act)

    tests.append(ReceptorTest('environmental_change_detection', 'environmental_augmentation',
        'Action changes track object-responding state changes', 0.1,
        test_environmental_change_detection, needs_closed_loop=True))

    def test_environmental_trend_detection(log, engine, **kw):
        obj_i = idx['obj_start']; N = min(5000, len(log)); w = 5
        if N <= 2 * w + 20:
            return 0.0
        prox = np.array([e['obs_after'][obj_i] for e in log[:N]])
        trend = prox[w:N-w] - prox[0:N-2*w]
        future = prox[2*w:N] - prox[w:N-w]
        return _safe_corr(trend, future)

    tests.append(ReceptorTest('environmental_trend_detection', 'environmental_augmentation',
        'Recent object-proximity trend predicts its continuation', 0.15,
        test_environmental_trend_detection))

    def test_modification_attribution(log, engine, **kw):
        ext_i = idx['agency_start'] + 1; N = min(5000, len(log))
        act_mag = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        ext = np.array([e['obs_after'][ext_i] for e in log[:N]])
        low = ext[act_mag < 1]; high = ext[act_mag >= 3]
        if len(low) < 20 or len(high) < 20:
            return 0.0
        return float(np.mean(low) - np.mean(high))

    tests.append(ReceptorTest('modification_attribution', 'environmental_augmentation',
        'External-change signal is higher when the organism itself is inactive', 0.05,
        test_modification_attribution, needs_closed_loop=True))

    def test_deliberate_complexification(log, engine, **kw):
        obj_i = idx['obj_start']; N = len(log)
        if N < 600:
            return 0.0
        early_var = np.std([e['obs_after'][obj_i:obj_i+3] for e in log[:N//3]])
        late_var = np.std([e['obs_after'][obj_i:obj_i+3] for e in log[-N//3:]])
        return float(late_var - early_var)

    tests.append(ReceptorTest('deliberate_complexification', 'environmental_augmentation',
        'Object-configuration variance increases over the course of the log', 0.02,
        test_deliberate_complexification, needs_closed_loop=True))

    def test_developmental_environment_engineering(log, engine, **kw):
        obj_i = idx['obj_start']; N = len(log)
        if N < 800:
            return 0.0
        q1 = np.mean([e['obs_after'][obj_i:obj_i+3] for e in log[N//4:N//2]], axis=0)
        q3 = np.mean([e['obs_after'][obj_i:obj_i+3] for e in log[N//2:3*N//4]], axis=0)
        q4 = np.mean([e['obs_after'][obj_i:obj_i+3] for e in log[3*N//4:]], axis=0)
        drift_early = np.linalg.norm(q3 - q1)
        if drift_early < 0.02:
            return 0.0
        return float(1.0 - np.linalg.norm(q4 - q3) / (drift_early + 1e-8))

    tests.append(ReceptorTest('developmental_environment_engineering', 'environmental_augmentation',
        'Object-configuration drift stabilizes into a lasting arrangement', 0.1,
        test_developmental_environment_engineering, needs_closed_loop=True))

    def test_self_model(log, engine, **kw):
        N = min(1500, len(log)); cdim = engine.core_obs_dim
        pred_mags, actual_mags = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            pred_mags.append(np.linalg.norm(pred))
            actual_mags.append(np.linalg.norm(e['obs_after'][:cdim] - e['obs_before'][:cdim]))
        return _safe_corr(pred_mags, actual_mags)

    tests.append(ReceptorTest('self_model', 'agency',
        'Predicted self-caused change magnitude tracks actual change magnitude', 0.2,
        test_self_model))

    def test_self_model_applied_to_others(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        correct, total = 0, 0
        for i in range(3, N):
            bearing_change = log[i]['obs_after'][npc_i+1] - log[i-3]['obs_after'][npc_i+1]
            if abs(bearing_change) < 0.05:
                continue
            own_turn = _turn(log[i]['action'])
            if own_turn == 0:
                continue
            total += 1
            if np.sign(bearing_change) == np.sign(own_turn):
                correct += 1
        return correct / (total + 1e-8)

    tests.append(ReceptorTest('self_model_applied_to_others', 'social',
        "Own turning direction matches the NPC's recent bearing change more than chance",
        0.55, test_self_model_applied_to_others, needs_closed_loop=True))

    def test_perspective_taking(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        approach = np.array([log[i]['obs_after'][npc_i] - log[i-1]['obs_after'][npc_i]
                             if i > 0 else 0.0 for i in range(N)])
        distress = np.array([e['obs_after'][npc_i+6] for e in log[:N]])
        return -_safe_corr(approach, distress)

    tests.append(ReceptorTest('perspective_taking', 'social',
        'Organism does not close distance on the NPC when it is distressed', 0.05,
        test_perspective_taking, needs_closed_loop=True))

    def test_intention_recognition(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log)); lag = 3
        if N <= lag + 20:
            return 0.0
        npc_omega_future = np.array([log[i+lag]['obs_after'][npc_i+5] for i in range(N - lag)])
        own_turn = np.array([_turn(log[i]['action']) for i in range(N - lag)])
        return _safe_corr(own_turn, npc_omega_future)

    tests.append(ReceptorTest('intention_recognition', 'social',
        "Own turning anticipates the NPC's future angular velocity", 0.05,
        test_intention_recognition, needs_closed_loop=True))

    def test_predictability_mismatch(log, engine, **kw):
        npc_i = idx['npc_start']; fam_i = idx['mm_start']; N = min(5000, len(log))
        erratic = np.array([e['obs_after'][npc_i+6] for e in log[:N]])
        fam = np.array([e['obs_after'][fam_i] for e in log[:N]])
        return -_safe_corr(erratic, fam)

    tests.append(ReceptorTest('predictability_mismatch', 'social',
        'Overall context familiarity drops when the NPC becomes erratic', 0.1,
        test_predictability_mismatch))

    def test_prediction_accuracy(log, engine, **kw):
        cert_i = idx['mm_start'] + 2; N = min(1500, len(log)); cdim = engine.core_obs_dim
        certs, cosines = [], []
        for e in log[:N]:
            pred, c, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            actual = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            pn, an = np.linalg.norm(pred), np.linalg.norm(actual)
            if pn < 0.01 or an < 0.01:
                continue
            cosines.append(np.dot(pred, actual) / (pn * an))
            certs.append(e['obs_after'][cert_i])
        return _safe_corr(certs, cosines)

    tests.append(ReceptorTest('prediction_accuracy', 'meta_motivational',
        'Certainty is well-calibrated to actual prediction cosine-accuracy', 0.15,
        test_prediction_accuracy))

    def test_spatial_reasoning(log, engine, **kw):
        npc_i = idx['npc_start']; N = min(5000, len(log))
        mask = np.array([log[i]['obs_after'][npc_i] > 0.2 for i in range(N)])
        if mask.sum() < 50:
            return 0.0
        bearing = np.array([log[i]['obs_after'][npc_i+1] for i in range(N)])[mask]
        turn = np.array([_turn(log[i]['action']) for i in range(N)])[mask]
        return _safe_corr(bearing, turn)

    tests.append(ReceptorTest('spatial_reasoning', 'social',
        "Turning direction matches the NPC's relative bearing when it is in range", 0.1,
        test_spatial_reasoning, needs_closed_loop=True))

    def test_value_hierarchy(log, engine, **kw):
        N = min(5000, len(log))
        pain = np.array([np.mean(_ch(e['obs_after'], _pain)) for e in log[:N]])
        fatigue = np.array([np.mean(_ch(e['obs_after'], _fat)) for e in log[:N]])
        act = np.array([np.sum(_extends(log[i]['action'])) for i in range(N)])
        return float(abs(_safe_corr(pain, act)) - abs(_safe_corr(fatigue, act)))

    tests.append(ReceptorTest('value_hierarchy', 'meta_motivational',
        'Action suppression tracks pain more strongly than fatigue (priority order)', 0.05,
        test_value_hierarchy, needs_closed_loop=True))

    # ============================================================
    # 17 reviewed receptor tests (Logic, Language, Bridging,
    # Compression additions, Interaction addition)
    # Three review rounds. All indices from idx. Drift-subtracted.
    # ============================================================

    def test_response_recognition(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        action_errors, null_errors = [], []
        for e in log[:min(5000, len(log))]:
            pred, cert, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            actual = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            error = float(np.mean((pred - actual[:len(pred)])**2))
            if action_to_hash(e['action']) == 0:
                null_errors.append(error)
            else:
                action_errors.append(error)
        if len(null_errors) < 5 or len(action_errors) < 5:
            return 0.0
        rng = np.random.RandomState(0)
        matched = rng.choice(action_errors, size=min(len(action_errors), len(null_errors) * 3), replace=False)
        return max(0.0, float(np.mean(null_errors) - np.mean(matched)))

    tests.append(ReceptorTest('response_recognition', 'interaction',
        'Lower prediction error on actions than null-action probes (matched counts)', 0.01,
        test_response_recognition))

    def test_constraint_shape(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        targeting_scores = []
        for i in range(len(log) - 5):
            pred, cert, n = engine.predict_delta(log[i]['obs_before'][:cdim], log[i]['action'])
            if cert < 0.5 or n == 0:
                continue
            actual = log[i]['obs_after'][:cdim] - log[i]['obs_before'][:cdim]
            per_dim_error = (pred - actual[:len(pred)])**2
            per_dim_var = np.abs(actual[:len(pred)])
            good_dims = per_dim_error < 0.05
            bad_dims = per_dim_error > 0.2
            if good_dims.sum() < 3 or bad_dims.sum() < 2:
                continue
            bad_vol = per_dim_var[bad_dims].mean()
            matched_mask = (~bad_dims) & (np.abs(per_dim_var - bad_vol) < bad_vol * 0.5)
            if matched_mask.sum() < 2:
                continue
            future_deltas = [_effect_delta(log[j], drift, cdim)
                             for j in range(i + 1, min(i + 5, len(log)))]
            if not future_deltas:
                continue
            future_mean = np.mean(future_deltas, axis=0)[:len(pred)]
            bad_activity = float(np.mean(np.abs(future_mean[bad_dims])))
            matched_activity = float(np.mean(np.abs(future_mean[matched_mask])))
            targeting_scores.append(bad_activity - matched_activity)
        if len(targeting_scores) < 10:
            return 0.0
        return float(np.mean(targeting_scores))

    tests.append(ReceptorTest('constraint_shape', 'compression',
        'Future actions target failed prediction dims more than matched-volatility dims', 0.01,
        test_constraint_shape, needs_closed_loop=True))

    def test_shaped_absence(log, engine, **kw):
        cdim = engine.core_obs_dim
        gap_revisit = 0
        gap_total = 0
        nongap_revisit = 0
        nongap_total = 0
        for i in range(len(log) - 20):
            pred, cert, n = engine.predict_delta(log[i]['obs_before'][:cdim], log[i]['action'])
            if n == 0 or cert < 0.3:
                continue
            actual = log[i]['obs_after'][:cdim] - log[i]['obs_before'][:cdim]
            error = float(np.mean((pred - actual[:len(pred)])**2))
            base_emb = engine.encoder.embed(log[i]['obs_before'][:cdim])
            revisited = False
            for j in range(i + 5, min(i + 15, len(log))):
                future_emb = engine.encoder.embed(log[j]['obs_before'][:cdim])
                if float(np.dot(base_emb, future_emb)) > 0.7:
                    revisited = True
                    break
            if error > 0.15:
                gap_total += 1
                if revisited:
                    gap_revisit += 1
            elif error < 0.05:
                nongap_total += 1
                if revisited:
                    nongap_revisit += 1
        if gap_total < 10 or nongap_total < 10:
            return 0.0
        return gap_revisit / gap_total - nongap_revisit / nongap_total

    tests.append(ReceptorTest('shaped_absence', 'compression',
        'Gap contexts revisited more than non-gap contexts (skip 5 steps for continuity)', 0.02,
        test_shaped_absence, needs_closed_loop=True))

    def test_missing_piece_located(log, engine, **kw):
        cdim = engine.core_obs_dim
        gap_improvements = []
        normal_improvements = []
        for i in range(len(log) - 10):
            pred, cert_before, n = engine.predict_delta(log[i]['obs_before'][:cdim], log[i]['action'])
            if n == 0:
                continue
            actual = log[i]['obs_after'][:cdim] - log[i]['obs_before'][:cdim]
            error = float(np.mean((pred - actual[:len(pred)])**2))
            ah = action_to_hash(log[i]['action'])
            for j in range(i + 5, min(i + 10, len(log))):
                if action_to_hash(log[j]['action']) == ah:
                    _, cert_after, n2 = engine.predict_delta(log[j]['obs_before'][:cdim], log[j]['action'])
                    if n2 == 0:
                        continue
                    improvement = cert_after - cert_before
                    if 0.3 < cert_before < 0.6:
                        if error > 0.15:
                            gap_improvements.append(improvement)
                        elif error < 0.05:
                            normal_improvements.append(improvement)
                    break
        if len(gap_improvements) < 10 or len(normal_improvements) < 10:
            return 0.0
        return float(np.mean(gap_improvements) - np.mean(normal_improvements))

    tests.append(ReceptorTest('missing_piece_located', 'compression',
        'Certainty improves faster for gap contexts than non-gap (matched starting certainty)', 0.01,
        test_missing_piece_located))

    def test_semantic_relation(log, engine, **kw):
        if not hasattr(engine, 'entity_store') or len(engine.entity_store.entities) == 0:
            return None
        stats = engine.entity_store.get_stats()
        if stats['num_relations'] < 5:
            return 0.0
        all_types = set()
        for eid in stats['entity_ids']:
            for rel in engine.entity_store.get_relations(eid):
                all_types.add(rel['type'])
        return float(len(all_types))

    tests.append(ReceptorTest('semantic_relation', 'logic',
        'Distinct relation types in entity-relation store', 2.0,
        test_semantic_relation))

    def test_transitivity(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        sims = []
        for i in range(min(1500, len(log)) - 2):
            obs0 = log[i]['obs_before'][:cdim]
            delta_chain, cert = engine.chain([log[i]['action'], log[i + 1]['action']], obs0)
            if cert < 0.2:
                continue
            actual = (log[i + 1]['obs_after'][:cdim] - obs0) - 2 * drift
            chain_corrected = delta_chain - 2 * drift
            an = np.linalg.norm(actual)
            dn = np.linalg.norm(chain_corrected)
            if an > 0.01 and dn > 0.01:
                sims.append(float(np.dot(actual, chain_corrected) / (an * dn)))
        if len(sims) < 20:
            return 0.0
        return float(np.mean(sims))

    tests.append(ReceptorTest('transitivity', 'logic',
        'Drift-subtracted 2-step chain predictions match actual outcomes', 0.3,
        test_transitivity))

    def test_conjunction(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        a_deltas, b_deltas, both_deltas = [], [], []
        for e in log[:min(3000, len(log))]:
            delta = _effect_delta(e, drift, cdim)
            pain_high = np.mean(_ch(e['obs_before'], _pain)) > 0.3
            endo_high = np.mean(_ch(e['obs_before'], _endo)) > 0.3
            if pain_high and endo_high:
                both_deltas.append(delta)
            elif pain_high:
                a_deltas.append(delta)
            elif endo_high:
                b_deltas.append(delta)
        if len(both_deltas) < 10 or len(a_deltas) < 10 or len(b_deltas) < 10:
            return 0.0
        additive = np.mean(a_deltas, axis=0) + np.mean(b_deltas, axis=0)
        actual_both = np.mean(both_deltas, axis=0)
        non_additivity = float(np.linalg.norm(actual_both - additive))
        baseline = float(np.linalg.norm(actual_both))
        if baseline < 0.01:
            return 0.0
        return non_additivity / baseline

    tests.append(ReceptorTest('conjunction', 'logic',
        'Both-state delta differs from sum of single-state deltas (non-additivity)', 0.1,
        test_conjunction))

    def test_quantifier(log, engine, **kw):
        cdim = engine.core_obs_dim
        split = int(len(log) * 0.8)
        test_log = log[split:]
        if len(test_log) < 100:
            return 0.0
        small_bucket, large_bucket = [], []
        for e in test_log:
            ah = action_to_hash(e['action'])
            entries = engine.store.mappings.get(ah, [])
            if not entries:
                continue
            pred, cert, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if n == 0:
                continue
            actual = e['obs_after'][:cdim] - e['obs_before'][:cdim]
            error = float(np.mean((pred - actual[:len(pred)])**2))
            if len(entries) >= 10:
                large_bucket.append((cert, error))
            elif len(entries) <= 3:
                small_bucket.append((cert, error))
        if len(small_bucket) < 15 or len(large_bucket) < 15:
            return 0.0
        large_cert = float(np.mean([c for c, e in large_bucket]))
        small_cert = float(np.mean([c for c, e in small_bucket]))
        large_err = float(np.mean([e for c, e in large_bucket]))
        small_err = float(np.mean([e for c, e in small_bucket]))
        cert_residual = large_cert - small_cert
        error_advantage = small_err - large_err
        if error_advantage < 0:
            return 0.0
        return max(0.0, cert_residual - error_advantage)

    tests.append(ReceptorTest('quantifier', 'logic',
        'Large-bucket certainty exceeds accuracy improvement (excess confidence = universality)', 0.02,
        test_quantifier))

    def test_contradiction(log, engine, **kw):
        contradictions_resolved = 0
        contradictions_total = 0
        for ah, entries in engine.store.mappings.items():
            for i, a in enumerate(entries):
                for b in entries[i + 1:]:
                    sim = float(np.dot(a.context_embedding, b.context_embedding))
                    if sim < 0.7:
                        continue
                    sign_agree = float(np.mean(np.sign(a.delta) == np.sign(b.delta)))
                    if sign_agree > 0.4:
                        continue
                    contradictions_total += 1
                    if (a.count > 5 and a.certainty < 0.4) or (b.count > 5 and b.certainty < 0.4):
                        contradictions_resolved += 1
        if contradictions_total < 3:
            return 0.0
        return contradictions_resolved / contradictions_total

    tests.append(ReceptorTest('contradiction', 'logic',
        'High-count contradicting entries show certainty revision (high count + low cert)', 0.1,
        test_contradiction))

    def test_it_follows(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        chain_certs = []
        actual_accuracies = []
        for i in range(min(1000, len(log)) - 2):
            obs = log[i]['obs_before'][:cdim]
            delta_chain, chain_cert = engine.chain([log[i]['action'], log[i + 1]['action']], obs)
            actual = (log[i + 1]['obs_after'][:cdim] - obs) - 2 * drift
            chain_corrected = delta_chain - 2 * drift
            an = np.linalg.norm(actual)
            cn = np.linalg.norm(chain_corrected)
            if an < 0.01 or cn < 0.01:
                continue
            accuracy = float(np.dot(actual, chain_corrected) / (an * cn))
            chain_certs.append(chain_cert)
            actual_accuracies.append(accuracy)
        if len(chain_certs) < 30:
            return 0.0
        return _safe_corr(chain_certs, actual_accuracies)

    tests.append(ReceptorTest('it_follows', 'logic',
        'Chain certainty predicts actual 2-step accuracy (calibration, not arithmetic)', 0.1,
        test_it_follows))

    def test_naming(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        cdim = engine.core_obs_dim
        steps_per_ep = 300
        N = len(log)
        num_eps = max(1, N // steps_per_ep)
        if num_eps < 3:
            return 0.0
        motif_contexts = defaultdict(list)
        for ep in range(num_eps):
            s = ep * steps_per_ep
            for i in range(s, min(s + steps_per_ep - 1, N)):
                h = (action_to_hash(log[i]['action']),
                     action_to_hash(log[min(i + 1, N - 1)]['action']))
                emb = engine.encoder.embed(log[i]['obs_before'][:cdim])
                motif_contexts[h].append((ep, emb))
        named = 0
        for h, contexts in motif_contexts.items():
            eps = set(c[0] for c in contexts)
            if len(eps) < 2:
                continue
            cross_ep_embs = []
            for ep in list(eps)[:3]:
                ep_embs = [c[1] for c in contexts if c[0] == ep]
                if ep_embs:
                    cross_ep_embs.append(ep_embs[0])
            if len(cross_ep_embs) >= 2:
                sim = float(np.dot(cross_ep_embs[0], cross_ep_embs[1]))
                if sim > 0.5:
                    named += 1
        return named / (len(motif_contexts) + 1e-8)

    tests.append(ReceptorTest('naming', 'language',
        'Motifs recur across episodes in similar contexts (embedding-gated)', 0.02,
        test_naming))

    def test_self_talk(log, engine, **kw):
        conflict_idx = idx['conflict']
        N = min(5000, len(log))
        high_latency, low_latency = [], []
        high_reward, low_reward = [], []
        for i in range(1, N):
            obs = log[i]['obs_after']
            if len(obs) <= conflict_idx:
                continue
            conflict = obs[conflict_idx]
            action_change = float(np.sum(np.abs(
                np.array(log[i]['action']) - np.array(log[i - 1]['action']))))
            reward = log[i]['reward']
            if conflict > 0.3:
                high_latency.append(action_change)
                high_reward.append(reward)
            elif conflict < 0.1:
                low_latency.append(action_change)
                low_reward.append(reward)
        if len(high_latency) < 20 or len(low_latency) < 20:
            return 0.0
        all_lat = high_latency + low_latency
        all_rew = high_reward + low_reward
        lat_std = float(np.std(all_lat)) + 1e-8
        rew_std = float(np.std(all_rew)) + 1e-8
        lat_z = (np.mean(high_latency) - np.mean(low_latency)) / lat_std
        rew_z = (np.mean(high_reward) - np.mean(low_reward)) / rew_std
        if lat_z <= 0 or rew_z <= 0:
            return 0.0
        return float(min(lat_z, rew_z))

    tests.append(ReceptorTest('self_talk', 'language',
        'High conflict -> more action variation AND better reward (z-scored, both positive)', 0.1,
        test_self_talk, needs_closed_loop=True))

    def test_referential_grounding(log, engine, **kw):
        if engine.pattern_store is None:
            return 0.0
        affect_certs, nonaffect_certs = [], []
        all_pats = [p for pats in engine.pattern_store.patterns.values() for p in pats]
        for pat in all_pats:
            if pat.count < 3 or pat.count > 20:
                continue
            d = pat.cumulative_delta
            affect_mag = float(np.linalg.norm(d[_pain[0]:_pain[1]]) +
                               np.linalg.norm(d[_endo[0]:_endo[1]]))
            total_mag = float(np.linalg.norm(d)) + 1e-8
            frac = affect_mag / total_mag
            if frac > 0.3:
                affect_certs.append(pat.certainty)
            elif frac < 0.1:
                nonaffect_certs.append(pat.certainty)
        if len(affect_certs) < 5 or len(nonaffect_certs) < 5:
            return 0.0
        return float(np.mean(affect_certs) - np.mean(nonaffect_certs))

    tests.append(ReceptorTest('referential_grounding', 'language',
        'Affect-grounded patterns have higher certainty (count-controlled)', 0.05,
        test_referential_grounding))

    def test_mimicry(log, engine, **kw):
        npc_i = idx['npc_start']
        N = min(5000, len(log))
        lag = 3
        if N <= lag + 30:
            return 0.0
        npc_activity = []
        org_activity = []
        env_state = []
        for i in range(N - lag):
            obs = log[i]['obs_after']
            if len(obs) <= npc_i + 4:
                continue
            npc_activity.append(obs[npc_i + 3] + abs(obs[npc_i + 4]))
            org_activity.append(float(np.sum(_extends(log[i + lag]['action']))) / L)
            env_state.append(float(np.mean(_ch(log[i]['obs_after'], _pain))))
        if len(npc_activity) < 50:
            return 0.0
        raw_corr = _safe_corr(npc_activity, org_activity)
        env_npc = _safe_corr(env_state, npc_activity)
        env_org = _safe_corr(env_state, org_activity)
        denom = np.sqrt(max(0.0001, (1 - env_npc**2) * (1 - env_org**2)))
        partial = (raw_corr - env_npc * env_org) / denom
        return max(0.0, float(partial))

    tests.append(ReceptorTest('mimicry', 'bridging',
        'Organism activity tracks NPC activity (partial corr, controlling environment)', 0.05,
        test_mimicry, needs_closed_loop=True))

    def test_trust(log, engine, **kw):
        npc_i = idx['npc_start']
        steps_per_ep = 300
        N = len(log)
        num_eps = N // steps_per_ep
        if num_eps < 10:
            return 0.0
        ep_reliability = []
        ep_mimicry = []
        for ep in range(num_eps):
            s = ep * steps_per_ep
            e_end = min(s + steps_per_ep, N)
            npc_dists = [log[i]['obs_after'][npc_i] for i in range(s, e_end)
                         if len(log[i]['obs_after']) > npc_i]
            rewards = [log[i]['reward'] for i in range(s, e_end)]
            reliability = _safe_corr(npc_dists, rewards)
            npc_acts = [log[i]['obs_after'][npc_i + 3] for i in range(s, e_end)
                        if len(log[i]['obs_after']) > npc_i + 3]
            org_acts = [float(np.sum(_extends(log[i]['action']))) / L for i in range(s, e_end)]
            mimicry = _safe_corr(npc_acts[:len(org_acts)], org_acts[:len(npc_acts)])
            ep_reliability.append(reliability)
            ep_mimicry.append(mimicry)
        return _safe_corr(ep_reliability, ep_mimicry)

    tests.append(ReceptorTest('trust', 'bridging',
        'Per-episode mimicry correlates with per-episode NPC reliability', 0.1,
        test_trust, needs_closed_loop=True))

    def test_executability(log, engine, **kw):
        cdim = engine.core_obs_dim
        N = min(3000, len(log))
        bad_pairs = []
        for e in log[:N // 2]:
            if e['reward'] < -0.5:
                emb = engine.encoder.embed(e['obs_before'][:cdim])
                bad_pairs.append((emb, action_to_hash(e['action'])))
        if len(bad_pairs) < 5:
            return 0.0
        repeat_similar = 0
        total_similar = 0
        repeat_dissimilar = 0
        total_dissimilar = 0
        for i in range(N // 2, N):
            emb = engine.encoder.embed(log[i]['obs_before'][:cdim])
            ah = action_to_hash(log[i]['action'])
            for bad_emb, bad_ah in bad_pairs[:20]:
                sim = float(np.dot(emb, bad_emb))
                if sim > 0.6:
                    total_similar += 1
                    if ah == bad_ah:
                        repeat_similar += 1
                elif sim < 0.3:
                    total_dissimilar += 1
                    if ah == bad_ah:
                        repeat_dissimilar += 1
            if total_similar > 200:
                break
        if total_similar < 10 or total_dissimilar < 10:
            return 0.0
        similar_rate = repeat_similar / total_similar
        dissimilar_rate = repeat_dissimilar / total_dissimilar
        return max(0.0, float(dissimilar_rate - similar_rate))

    tests.append(ReceptorTest('executability', 'bridging',
        'Bad actions repeated less in similar context than dissimilar (context-specific avoidance)', 0.01,
        test_executability, needs_closed_loop=True))

    def test_translation(log, engine, **kw):
        cdim = engine.core_obs_dim
        drift = _estimate_drift(log, cdim)
        action_means = {}
        for ah, entries in engine.store.mappings.items():
            if len(entries) >= 5:
                action_means[ah] = np.mean([e.delta for e in entries], axis=0) - drift[:len(entries[0].delta)]
        equiv = defaultdict(set)
        for a1, d1 in action_means.items():
            for a2, d2 in action_means.items():
                if a1 >= a2:
                    continue
                n1 = np.linalg.norm(d1)
                n2 = np.linalg.norm(d2)
                if n1 > 0.01 and n2 > 0.01 and float(np.dot(d1, d2) / (n1 * n2)) > 0.8:
                    equiv[a1].add(a2)
                    equiv[a2].add(a1)
        num_hashes = len(action_means)
        if len(equiv) < 2 or num_hashes < 4:
            return 0.0
        switch_to_equiv = 0
        switch_total = 0
        for i in range(1, len(log)):
            ah_prev = action_to_hash(log[i - 1]['action'])
            if log[i - 1]['reward'] >= 0 or ah_prev not in equiv:
                continue
            ah_now = action_to_hash(log[i]['action'])
            if ah_now != ah_prev:
                switch_total += 1
                if ah_now in equiv[ah_prev]:
                    switch_to_equiv += 1
        if switch_total < 10:
            return 0.0
        observed_rate = switch_to_equiv / switch_total
        n_equiv = len(equiv.get(ah_prev, set()))
        base_rate = n_equiv / max(1, num_hashes - 1)
        return max(0.0, float(observed_rate - base_rate))

    tests.append(ReceptorTest('translation', 'bridging',
        'After failure, switches to functional equivalent above base rate', 0.01,
        test_translation, needs_closed_loop=True))

    # ============================================================
    # 14 NEW receptor tests (Observation, Social, Formalization)
    # Review-corrected: partial correlations, pain-stratified splits,
    # Granger causality, drift-subtracted deltas, proper store iteration.
    # ============================================================

    _pe_start = idx['pe_start']
    _mm_start = idx['mm_start']
    _npc_start = idx['npc_start']
    _energy_idx = idx['energy']
    _cdim = idx.get('core_obs_dim', CORE_OBS_DIM)

    def _lstsq_resid(X, y):
        X_aug = np.column_stack([X, np.ones(len(y))])
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        return y - X_aug @ beta

    def _hamming(a, b):
        return sum(x != y for x, y in zip(a, b))

    # --- OBSERVATION FAMILY ---

    def test_relational_observation(log, engine, **kw):
        if len(log) < 100:
            return 0.0
        pairs = [(_pain, _endo), (_temp, _chem), (_pain, _pres), (_endo, _temp)]
        advantages = []
        for ch_a, ch_b in pairs:
            vals_a, vals_b, signs, hams = [], [], [], []
            for i in range(1, min(len(log), 3000)):
                obs = log[i]['obs_before']
                va = float(np.mean(_ch(obs, ch_a)))
                vb = float(np.mean(_ch(obs, ch_b)))
                s = 1.0 if va > vb else -1.0
                h = _hamming(log[i]['action'], log[i - 1]['action'])
                vals_a.append(va)
                vals_b.append(vb)
                signs.append(s)
                hams.append(h)
            n = len(hams)
            if n < 30:
                continue
            X = np.column_stack([vals_a, vals_b])
            y = np.array(hams, dtype=float)
            s = np.array(signs)
            try:
                resid_y = _lstsq_resid(X, y)
                resid_s = _lstsq_resid(X, s)
            except Exception:
                continue
            pc = abs(_safe_corr(resid_s, resid_y))
            advantages.append(pc)
        if len(advantages) < 2:
            return 0.0
        return float(np.mean(advantages))

    tests.append(ReceptorTest('relational_observation', 'observation',
        'Action correlates with sign(a-b) after residualizing absolutes', 0.02,
        test_relational_observation))

    def test_selective_observation(log, engine, **kw):
        if len(log) < 200:
            return 0.0
        pe_vals = [float(np.mean(e['obs_before'][_pe_start:_pe_start + 6])) for e in log]
        pain_vals = [float(np.mean(_ch(e['obs_before'], _pain))) for e in log]
        pe_med = np.median(pe_vals)
        pain_med = np.median(pain_vals)
        cells = [[], [], [], []]
        for i, entry in enumerate(log):
            hp = int(pain_vals[i] > pain_med)
            hpe = int(pe_vals[i] > pe_med)
            cells[hp * 2 + hpe].append(np.array(entry['action'], dtype=float))
        if any(len(c) < 15 for c in cells):
            return 0.0
        def _div(a_list, b_list):
            return float(np.mean(np.abs(np.mean(a_list, axis=0) - np.mean(b_list, axis=0))))
        d_high_pain = _div(cells[3], cells[2])
        d_low_pain = _div(cells[1], cells[0])
        return (d_high_pain + d_low_pain) / 2.0

    tests.append(ReceptorTest('selective_observation', 'observation',
        'PE-driven action divergence within matched pain strata', 0.02,
        test_selective_observation))

    def test_comparative_observation(log, engine, **kw):
        if len(log) < 100:
            return 0.0
        WINDOW = 10
        trends, levels, changes = [], [], []
        for i in range(WINDOW, min(len(log) - 1, 3000)):
            energies = [log[j]['obs_before'][_energy_idx] for j in range(i - WINDOW, i)]
            trend = energies[-1] - energies[0]
            level = log[i]['obs_before'][_energy_idx]
            ham = _hamming(log[i]['action'], log[i + 1]['action'])
            trends.append(trend)
            levels.append(level)
            changes.append(ham)
        if len(trends) < 50:
            return 0.0
        y = np.array(changes, dtype=float)
        t = np.array(trends)
        X = np.array(levels).reshape(-1, 1)
        try:
            resid_y = _lstsq_resid(X, y)
            resid_t = _lstsq_resid(X, t)
        except Exception:
            return 0.0
        return float(abs(_safe_corr(resid_t, resid_y)))

    tests.append(ReceptorTest('comparative_observation', 'observation',
        'Trend predicts strategy change after controlling for level', 0.02,
        test_comparative_observation))

    def test_cross_modal_observation(log, engine, **kw):
        if not hasattr(engine, 'store'):
            return None
        groups = {'pain': _pain, 'endorphin': _endo, 'temperature': _temp,
                  'chemical': _chem, 'pressure': _pres}
        cross_certs, same_certs = [], []
        for entry_list in engine.store.mappings.values():
            entries = entry_list if isinstance(entry_list, list) else [entry_list]
            for m in entries:
                if m.certainty < 0.3 or m.count < 3:
                    continue
                if not hasattr(m, 'representative_obs') or m.representative_obs is None:
                    continue
                raw_obs = m.representative_obs
                delta_group, max_d = None, 0
                for name, (s, e) in groups.items():
                    d = float(np.mean(np.abs(m.delta[s:e])))
                    if d > max_d:
                        max_d = d
                        delta_group = name
                ctx_group, max_c = None, 0
                for name, (s, e) in groups.items():
                    c = float(np.mean(np.abs(raw_obs[s:e])))
                    if c > max_c:
                        max_c = c
                        ctx_group = name
                if delta_group is None or ctx_group is None:
                    continue
                if max_d < 0.01 or max_c < 0.01:
                    continue
                if delta_group != ctx_group:
                    cross_certs.append(m.certainty)
                else:
                    same_certs.append(m.certainty)
        if len(cross_certs) < 5 or len(same_certs) < 5:
            return 0.0
        return float(max(0.0, np.mean(cross_certs) - np.mean(same_certs)))

    tests.append(ReceptorTest('cross_modal_observation', 'observation',
        'Cross-modal entries have higher certainty than same-modal', 0.01,
        test_cross_modal_observation))

    def test_meta_observation(log, engine, **kw):
        if kw.get('log_provenance') == 'oracle':
            return 0.0
        if len(log) < 100:
            return 0.0
        LP_OFFSET = 1
        lp_vals, ext_feats, act_scalars = [], [], []
        for entry in log:
            obs = entry['obs_before']
            lp = obs[_mm_start + LP_OFFSET] if len(obs) > _mm_start + LP_OFFSET else 0.0
            pv = float(np.mean(_ch(obs, _pain)))
            ev = float(np.mean(_ch(obs, _endo)))
            lp_vals.append(lp)
            ext_feats.append([pv, ev])
            act_scalars.append(float(np.mean(entry['action'])))
        n = len(lp_vals)
        if n < 50:
            return 0.0
        X = np.array(ext_feats)
        y = np.array(act_scalars)
        lp = np.array(lp_vals)
        try:
            resid_y = _lstsq_resid(X, y)
            resid_lp = _lstsq_resid(X, lp)
        except Exception:
            return 0.0
        return float(abs(_safe_corr(resid_lp, resid_y)))

    tests.append(ReceptorTest('meta_observation', 'observation',
        'Learning progress predicts action after controlling for external state', 0.02,
        test_meta_observation, needs_closed_loop=True))

    # --- SOCIAL FAMILY ---

    def test_belief_attribution(log, engine, **kw):
        if kw.get('log_provenance') == 'oracle':
            return 0.0
        if not hasattr(engine, 'entity_store'):
            return None
        if len(log) < 300:
            return 0.0
        npc_present = [any(abs(v) > 0.1
                           for v in entry['obs_before'][_npc_start:_npc_start + 12])
                       for entry in log]
        reappearance_tests = []
        i = 0
        while i < len(log) - 1:
            if npc_present[i] and i + 1 < len(log) and not npc_present[i + 1]:
                last_seen_obs = log[i]['obs_before']
                for j in range(i + 5, min(i + 50, len(log))):
                    if npc_present[j]:
                        reappear_obs = log[j]['obs_before']
                        pain_then = float(np.mean(_ch(last_seen_obs, _pain)))
                        pain_now = float(np.mean(_ch(reappear_obs, _pain)))
                        if abs(pain_now - pain_then) > 0.2:
                            reappearance_tests.append({
                                'old': pain_then, 'new': pain_now,
                                'action': log[j]['action'],
                            })
                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1
        if len(reappearance_tests) < 15:
            return 0.0
        action_mags = [sum(t['action']) for t in reappearance_tests]
        r_old = abs(_safe_corr([t['old'] for t in reappearance_tests], action_mags))
        r_new = abs(_safe_corr([t['new'] for t in reappearance_tests], action_mags))
        return float(max(0.0, r_old - r_new))

    tests.append(ReceptorTest('belief_attribution', 'social',
        'Behavior at NPC reappearance tracks stale state over current', 0.02,
        test_belief_attribution, needs_closed_loop=True))

    def test_social_learning(log, engine, **kw):
        if kw.get('log_provenance') == 'oracle':
            return 0.0
        if len(log) < 200:
            return 0.0
        npc_present = [any(abs(v) > 0.1
                           for v in entry['obs_before'][_npc_start:_npc_start + 12])
                       for entry in log]
        transition = None
        run_start, run_len = None, 0
        for i, present in enumerate(npc_present):
            if present:
                if run_start is None:
                    run_start = i
                run_len += 1
                if run_len >= 10:
                    transition = run_start
                    break
            else:
                run_start, run_len = None, 0
        if transition is None or transition < 50:
            return 0.0
        pre_actions = set()
        for entry in log[:transition]:
            pre_actions.add(action_to_hash(entry['action']))
        if len(pre_actions) < 5:
            return 0.0
        npc_novel, non_npc_novel = set(), set()
        post_npc, post_no_npc = 0, 0
        for i in range(transition, len(log)):
            h = action_to_hash(log[i]['action'])
            if h in pre_actions:
                continue
            if npc_present[i]:
                npc_novel.add(h)
                post_npc += 1
            else:
                non_npc_novel.add(h)
                post_no_npc += 1
        if post_npc < 20 or post_no_npc < 20:
            return 0.0
        npc_rate = len(npc_novel) / post_npc
        non_npc_rate = len(non_npc_novel) / post_no_npc
        return float(max(0.0, npc_rate - non_npc_rate))

    tests.append(ReceptorTest('social_learning', 'social',
        'Repertoire expansion rate higher during NPC presence', 0.01,
        test_social_learning, needs_closed_loop=True))

    def test_cultural_transmission(log, engine, **kw):
        generation = kw.get('generation', None)
        parent_engine = kw.get('parent_engine', None)
        if generation is None or generation < 1:
            return None
        if parent_engine is None:
            return None
        if not hasattr(engine, 'store') or not hasattr(parent_engine, 'store'):
            return None
        parent_confident = set()
        parent_all = set()
        for ah, entry_list in parent_engine.store.mappings.items():
            entries = entry_list if isinstance(entry_list, list) else [entry_list]
            parent_all.add(ah)
            for m in entries:
                if m.certainty > 0.7 and m.count > 5:
                    parent_confident.add(ah)
        if len(parent_confident) < 5:
            return 0.0
        inherited_speeds, control_speeds = [], []
        for ah, entry_list in engine.store.mappings.items():
            entries = entry_list if isinstance(entry_list, list) else [entry_list]
            for m in entries:
                if m.count < 3:
                    continue
                speed = m.certainty / np.log(m.count + 1)
                if ah in parent_confident:
                    inherited_speeds.append(speed)
                elif ah not in parent_all:
                    control_speeds.append(speed)
        if len(inherited_speeds) < 5 or len(control_speeds) < 5:
            return 0.0
        return float(max(0.0, np.mean(inherited_speeds) - np.mean(control_speeds)))

    tests.append(ReceptorTest('cultural_transmission', 'social',
        'Offspring converge faster on parent-mastered patterns', 0.02,
        test_cultural_transmission))

    def test_deception_detection(log, engine, **kw):
        if kw.get('log_provenance') == 'oracle':
            return 0.0
        if len(log) < 200:
            return 0.0
        npc_steps = []
        for i, entry in enumerate(log):
            obs = entry['obs_before']
            npc_obs = obs[_npc_start:_npc_start + 12]
            if any(abs(v) > 0.1 for v in npc_obs):
                obs_after = entry['obs_after']
                pd = float(np.mean(_ch(obs_after, _pain)) - np.mean(_ch(obs, _pain)))
                npc_steps.append({'step': i, 'pain_delta': pd,
                                  'action': entry['action']})
        if len(npc_steps) < 30:
            return 0.0
        third = len(npc_steps) // 3
        early = npc_steps[:third]
        late = npc_steps[-third:]
        early_bad_rate = sum(1 for s in early if s['pain_delta'] > 0.1) / len(early)
        if early_bad_rate < 0.3:
            return 0.0
        non_npc_actions = []
        for entry in log:
            obs = entry['obs_before']
            if not any(abs(v) > 0.1 for v in obs[_npc_start:_npc_start + 12]):
                non_npc_actions.append(np.array(entry['action'], dtype=float))
        if len(non_npc_actions) < 20:
            return 0.0
        baseline = np.mean(non_npc_actions, axis=0)
        early_dev = np.mean([np.mean(np.abs(np.array(s['action'], dtype=float) - baseline))
                             for s in early])
        late_dev = np.mean([np.mean(np.abs(np.array(s['action'], dtype=float) - baseline))
                            for s in late])
        return float(max(0.0, late_dev - early_dev))

    tests.append(ReceptorTest('deception_detection', 'social',
        'Behavioral coupling with NPC decreases after bad outcomes', 0.02,
        test_deception_detection, needs_closed_loop=True))

    def test_moral_reasoning(log, engine, **kw):
        npc_count = kw.get('npc_count', 1)
        if npc_count < 2:
            return None
        return None

    tests.append(ReceptorTest('moral_reasoning', 'social',
        'Third-party punishment (needs multi-NPC)', 0.1,
        test_moral_reasoning, needs_closed_loop=True))

    def test_nested_theory_of_mind(log, engine, **kw):
        if kw.get('log_provenance') == 'oracle':
            return 0.0
        if not hasattr(engine, 'entity_store'):
            return None
        if len(log) < 300:
            return 0.0
        LAG = 5
        HISTORY = 10
        npc_series, org_actions = [], []
        for entry in log:
            obs = entry['obs_before']
            npc_obs = obs[_npc_start:_npc_start + 12]
            if any(abs(v) > 0.1 for v in npc_obs):
                npc_series.append(float(np.mean(np.abs(npc_obs))))
                org_actions.append(float(np.mean(entry['action'])))
        n = len(npc_series)
        if n < HISTORY + LAG + 50:
            return 0.0
        targets, X_base, X_aug = [], [], []
        for t in range(HISTORY, n - LAG):
            targets.append(npc_series[t + LAG])
            npc_hist = [npc_series[t - h] for h in range(HISTORY)]
            X_base.append(npc_hist + [1.0])
            X_aug.append(npc_hist + [org_actions[t]] + [1.0])
        if len(targets) < 30:
            return 0.0
        targets = np.array(targets)
        X_b = np.array(X_base)
        X_a = np.array(X_aug)
        try:
            beta_b = np.linalg.lstsq(X_b, targets, rcond=None)[0]
            var_base = float(np.var(targets - X_b @ beta_b))
            beta_a = np.linalg.lstsq(X_a, targets, rcond=None)[0]
            var_aug = float(np.var(targets - X_a @ beta_a))
        except Exception:
            return 0.0
        if var_base < 1e-8:
            return 0.0
        return float(max(0.0, (var_base - var_aug) / var_base))

    tests.append(ReceptorTest('nested_theory_of_mind', 'social',
        'Organism actions Granger-cause NPC future state', 0.01,
        test_nested_theory_of_mind, needs_closed_loop=True))

    # --- FORMALIZATION FAMILY ---

    def test_rule_generalization(log, engine, **kw):
        if not hasattr(engine, 'store'):
            return None
        if len(log) < 100:
            return 0.0
        drift = _estimate_drift(log, _cdim)
        mid = len(log) // 2
        early_obs = np.array([e['obs_before'][:_cdim] for e in log[:mid]])
        centroid = np.mean(early_obs, axis=0)
        early_dists = [np.linalg.norm(obs - centroid) for obs in early_obs]
        dist_threshold = np.percentile(early_dists, 75)
        familiar_accs, novel_accs = [], []
        for entry in log[mid:]:
            obs = entry['obs_before']
            action = entry['action']
            pred_delta, cert, n = engine.predict_delta(obs, action)
            if n == 0 or cert < 0.3:
                continue
            actual = _effect_delta(entry, drift, _cdim)
            error = float(np.mean(np.abs(pred_delta[:_cdim] - actual[:_cdim])))
            accurate = 1.0 if error < 0.3 else 0.0
            d = np.linalg.norm(obs[:_cdim] - centroid)
            if d <= dist_threshold:
                familiar_accs.append(accurate)
            else:
                novel_accs.append(accurate)
        if len(novel_accs) < 10:
            return 0.0
        novel_acc = float(np.mean(novel_accs))
        if len(familiar_accs) >= 10:
            fam_acc = float(np.mean(familiar_accs))
            if fam_acc > 0.1:
                return float(novel_acc * (novel_acc / fam_acc))
        return novel_acc

    tests.append(ReceptorTest('rule_generalization', 'formalization',
        'Prediction accuracy on out-of-distribution observations', 0.1,
        test_rule_generalization))

    def test_rule_composition(log, engine, **kw):
        if not hasattr(engine, 'store'):
            return None
        if not hasattr(engine, 'chain'):
            return None
        if len(log) < 100:
            return 0.0
        drift = _estimate_drift(log, _cdim)
        chain_errors, single_errors = [], []
        for i in range(1, min(len(log) - 1, 200)):
            a1 = log[i - 1]['action']
            a2 = log[i]['action']
            obs_start = log[i - 1]['obs_before']
            try:
                chain_delta, chain_cert = engine.chain([a1, a2], obs_start)
            except Exception:
                continue
            if chain_cert < 0.1:
                continue
            actual = (log[i]['obs_after'][:_cdim] - obs_start[:_cdim]) - drift[:_cdim] * 2
            chain_err = float(np.mean(np.abs(chain_delta[:_cdim] - actual)))
            single_delta, _, _ = engine.predict_delta(obs_start, a2)
            single_err = float(np.mean(np.abs(single_delta[:_cdim] - actual)))
            chain_errors.append(chain_err)
            single_errors.append(single_err)
        if len(chain_errors) < 15:
            return 0.0
        mc = float(np.mean(chain_errors))
        ms = float(np.mean(single_errors))
        if ms < 0.01:
            return 0.0
        improvement = max(0.0, (ms - mc) / ms)
        if kw.get('log_provenance') == 'oracle':
            return float(improvement)
        sp_rng = np.random.RandomState(42)
        chosen_certs, random_certs = [], []
        for i in range(1, min(len(log), 150)):
            a1 = log[i - 1]['action']
            a2 = log[i]['action']
            obs = log[i - 1]['obs_before']
            try:
                _, cc = engine.chain([a1, a2], obs)
                chosen_certs.append(cc)
            except Exception:
                continue
            ra1 = log[sp_rng.randint(len(log))]['action']
            ra2 = log[sp_rng.randint(len(log))]['action']
            try:
                _, rc = engine.chain([ra1, ra2], obs)
                random_certs.append(rc)
            except Exception:
                random_certs.append(0.0)
        if len(chosen_certs) < 10:
            return float(improvement)
        usage = float(np.mean(chosen_certs) - np.mean(random_certs))
        return float(improvement * 0.5 + max(0.0, usage) * 0.5)

    tests.append(ReceptorTest('rule_composition', 'formalization',
        'Chain predictions beat single-step; organism selects high-cert chains', 0.02,
        test_rule_composition))

    def test_rule_revision(log, engine, **kw):
        if not hasattr(engine, 'pattern_store'):
            return None
        if len(log) < 200:
            return 0.0
        patterns = getattr(engine.pattern_store, 'patterns', [])
        revision_signals = 0
        total_candidates = 0
        for pattern in patterns:
            if pattern.count < 15:
                continue
            total_candidates += 1
            if pattern.certainty <= 0.6:
                continue
            cum_delta = getattr(pattern, 'cum_delta', None)
            if cum_delta is None:
                continue
            delta_mag = float(np.mean(np.abs(cum_delta)))
            if delta_mag < 0.01:
                continue
            m2_val = pattern.m2
            if isinstance(m2_val, np.ndarray):
                m2_val = float(np.mean(m2_val))
            rel_var = m2_val / (delta_mag + 1e-8)
            if rel_var > 0.5:
                revision_signals += 1
        if total_candidates < 10:
            return 0.0
        return float(revision_signals / total_candidates)

    tests.append(ReceptorTest('rule_revision', 'formalization',
        'Patterns with high certainty AND high relative variance (revised)', 0.05,
        test_rule_revision))

    def test_theory_formation(log, engine, **kw):
        if not hasattr(engine, 'store'):
            return None
        entries = []
        for entry_list in engine.store.mappings.values():
            el = entry_list if isinstance(entry_list, list) else [entry_list]
            for m in el:
                if m.certainty < 0.5 or m.count < 5:
                    continue
                if m.context_embedding is None:
                    continue
                if not hasattr(m, 'delta'):
                    continue
                entries.append(m)
        if len(entries) < 30:
            return 0.0
        if len(entries) > 200:
            sample_rng = np.random.RandomState(42)
            entries = [entries[i] for i in sample_rng.choice(len(entries), 200, replace=False)]
        embeddings = np.array([m.context_embedding for m in entries])
        deltas = np.array([m.delta for m in entries])
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        emb_n = embeddings / emb_norms
        d_norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
        d_n = deltas / d_norms
        n = len(entries)
        K = min(5, n // 5)
        if K < 1:
            return 0.0
        sim_matrix = emb_n @ emb_n.T
        neighbor_sims, distant_sims = [], []
        for i in range(n):
            ctx_sims = sim_matrix[i].copy()
            ctx_sims[i] = -2
            sorted_idx = np.argsort(ctx_sims)
            for j in sorted_idx[-K:]:
                neighbor_sims.append(float(d_n[i] @ d_n[j]))
            for j in sorted_idx[:K]:
                distant_sims.append(float(d_n[i] @ d_n[j]))
        if len(neighbor_sims) < 20:
            return 0.0
        return float(max(0.0, np.mean(neighbor_sims) - np.mean(distant_sims)))

    tests.append(ReceptorTest('theory_formation', 'formalization',
        'Context-neighbors share delta direction (organized structure)', 0.02,
        test_theory_formation))

    # --- EPISTEMIC FAMILY (new: conflation, fundamental_distinction) ---

    def test_conflation(log, engine, **kw):
        """Detects conflation via near-neighbor divergent pairs: entries in the
        same action bucket with similar context embeddings (0.5-0.9 cosine sim)
        but opposite delta directions. "Similar situation, two systematically
        different outcomes" is the conflation signature."""
        if not hasattr(engine, 'store'):
            return None

        divergent_pairs = 0
        aligned_pairs = 0

        for ah, entry_list in engine.store.mappings.items():
            entries = entry_list if isinstance(entry_list, list) else [entry_list]
            qualified = [e for e in entries
                         if e.count >= 3 and e.context_embedding is not None
                         and hasattr(e, 'delta')]
            if len(qualified) < 2:
                continue

            for i in range(len(qualified)):
                ei = qualified[i]
                emb_i = ei.context_embedding
                norm_i = np.linalg.norm(emb_i) + 1e-8
                delta_i = ei.delta
                dnorm_i = np.linalg.norm(delta_i) + 1e-8

                for j in range(i + 1, len(qualified)):
                    ej = qualified[j]
                    emb_j = ej.context_embedding
                    norm_j = np.linalg.norm(emb_j) + 1e-8

                    ctx_sim = float(emb_i @ emb_j / (norm_i * norm_j))
                    if ctx_sim < 0.5 or ctx_sim > 0.95:
                        continue

                    delta_j = ej.delta
                    dnorm_j = np.linalg.norm(delta_j) + 1e-8
                    delta_sim = float(delta_i @ delta_j / (dnorm_i * dnorm_j))

                    if delta_sim < -0.1:
                        divergent_pairs += 1
                    elif delta_sim > 0.3:
                        aligned_pairs += 1

        total = divergent_pairs + aligned_pairs
        if total < 5:
            return 0.0
        return float(divergent_pairs / total)

    tests.append(ReceptorTest('conflation', 'epistemic',
        'Same-action entries with similar context but divergent deltas', None,
        test_conflation))

    def test_fundamental_distinction(log, engine, **kw):
        """After identifying conflated regions (divergent-pair signature),
        narrowing context improves prediction MORE in conflated regions than
        in clean regions. The contrast isolates the distinction signal from
        the k-NN bias-variance effect."""
        if not hasattr(engine, 'store'):
            return None
        if len(log) < 200:
            return 0.0

        # Flag conflated action hashes: those with divergent-pair signature
        conflated_hashes = set()
        clean_hashes = set()
        for ah, entry_list in engine.store.mappings.items():
            entries = entry_list if isinstance(entry_list, list) else [entry_list]
            qualified = [e for e in entries
                         if e.count >= 3 and e.context_embedding is not None
                         and hasattr(e, 'delta')]
            if len(qualified) < 2:
                clean_hashes.add(ah)
                continue
            has_divergent = False
            for i in range(len(qualified)):
                ei = qualified[i]
                ni = np.linalg.norm(ei.context_embedding) + 1e-8
                di = np.linalg.norm(ei.delta) + 1e-8
                for j in range(i + 1, len(qualified)):
                    ej = qualified[j]
                    nj = np.linalg.norm(ej.context_embedding) + 1e-8
                    ctx_sim = float(ei.context_embedding @ ej.context_embedding / (ni * nj))
                    if 0.5 < ctx_sim < 0.95:
                        dj = np.linalg.norm(ej.delta) + 1e-8
                        delta_sim = float(ei.delta @ ej.delta / (di * dj))
                        if delta_sim < -0.1:
                            has_divergent = True
                            break
                if has_divergent:
                    break
            if has_divergent:
                conflated_hashes.add(ah)
            else:
                clean_hashes.add(ah)

        if len(conflated_hashes) < 3 or len(clean_hashes) < 3:
            return 0.0

        # Compare narrow-vs-broad improvement in conflated vs clean regions
        def _improvement_for_hashes(target_hashes):
            broad_errs, narrow_errs = [], []
            for entry in log[len(log)//2:]:
                ah = action_to_hash(entry['action'])
                if ah not in target_hashes:
                    continue
                obs = entry['obs_before']
                action = entry['action']
                actual = entry['obs_after'][:_cdim] - obs[:_cdim]

                pred_b, cert_b, n_b = engine.predict_delta(obs, action, top_k=10)
                pred_n, cert_n, n_n = engine.predict_delta(obs, action, top_k=3)
                if n_b < 2 or n_n < 1:
                    continue
                broad_errs.append(float(np.mean(np.abs(pred_b[:_cdim] - actual))))
                narrow_errs.append(float(np.mean(np.abs(pred_n[:_cdim] - actual))))
            if len(broad_errs) < 10:
                return None
            mb = float(np.mean(broad_errs))
            mn = float(np.mean(narrow_errs))
            if mb < 0.01:
                return 0.0
            return (mb - mn) / mb

        imp_conflated = _improvement_for_hashes(conflated_hashes)
        imp_clean = _improvement_for_hashes(clean_hashes)

        if imp_conflated is None or imp_clean is None:
            return 0.0

        # Score: narrowing helps MORE in conflated regions than clean regions
        return float(max(0.0, imp_conflated - imp_clean))

    tests.append(ReceptorTest('fundamental_distinction', 'epistemic',
        'Narrowing context improves prediction more in conflated than clean regions', None,
        test_fundamental_distinction))

    return tests


def discover(log, engine, threshold_overrides=None, log_provenance='oracle'):
    """Run receptor discovery battery.

    log_provenance: 'oracle' (data from compute_optimal_actions) or
                    'policy' (data from trained model). Tests marked
                    needs_closed_loop=True are skipped on oracle logs.
    """
    tests = build_tests()
    results = {
        'discovered': [],
        'not_found': [],
        'skipped': [],
        'scores': {},
        'thresholds': {},
        'details': [],
    }

    for test in tests:
        if getattr(test, 'needs_closed_loop', False) and log_provenance != 'policy':
            results['skipped'].append(test.receptor_id)
            results['details'].append({
                'receptor_id': test.receptor_id,
                'family': test.family,
                'description': test.description,
                'score': None,
                'threshold': test.threshold,
                'discovered': False,
                'skipped': True,
            })
            continue
        try:
            score = test.test_fn(log, engine)
        except Exception:
            score = 0.0
        if score is None:
            results['skipped'].append(test.receptor_id)
            results['details'].append({
                'receptor_id': test.receptor_id,
                'family': test.family,
                'description': test.description,
                'score': None,
                'threshold': test.threshold,
                'discovered': False,
                'skipped': True,
            })
            continue
        threshold = threshold_overrides.get(test.receptor_id, test.threshold) \
            if threshold_overrides else test.threshold
        passed = score > 0 and score >= threshold
        results['scores'][test.receptor_id] = round(float(score), 4)
        results['thresholds'][test.receptor_id] = test.threshold

        if passed:
            results['discovered'].append(test.receptor_id)
        else:
            results['not_found'].append(test.receptor_id)

        results['details'].append({
            'receptor_id': test.receptor_id,
            'family': test.family,
            'description': test.description,
            'score': round(float(score), 4),
            'threshold': test.threshold,
            'discovered': bool(passed),
        })

    results['topology_vector'] = [1 if d['discovered'] else 0 for d in results['details']]
    return results


def calibrate_null_thresholds(log, engine, num_shuffles=10, percentile=95):
    """Set thresholds at the 95th percentile of null-model scores.

    Two null types:
    - Shuffled: real log with temporal structure destroyed (action-obs pairings scrambled).
      Tests correlation-based claims.
    - Engine-rebuilt: engine built from the shuffled log, so engine-internal tests
      (predict_delta, chain, pattern_store queries) get a null engine, not the real one.

    'Discovered' means 'distinguishable from what noise produces.'
    """
    tests = build_tests()
    null_scores = {t.receptor_id: [] for t in tests}

    for shuffle_i in range(num_shuffles):
        rng = np.random.RandomState(shuffle_i)

        null_log = [dict(e) for e in log]
        action_perm = rng.permutation(len(null_log))
        for i in range(len(null_log)):
            null_log[i]['action'] = log[int(action_perm[i])]['action']
            null_log[i]['reward'] = log[int(action_perm[i])]['reward']

        try:
            null_engine = build_mental_model(null_log)
        except Exception:
            null_engine = engine

        for test in tests:
            try:
                score = test.test_fn(null_log, null_engine)
            except Exception:
                score = 0.0
            if score is not None:
                null_scores[test.receptor_id].append(float(score))

    thresholds = {}
    for receptor_id, scores in null_scores.items():
        if scores:
            thresholds[receptor_id] = float(np.percentile(scores, percentile))
        else:
            thresholds[receptor_id] = 0.0

    return thresholds


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Step 32: Receptor Discovery ===\n")
    print("--- Generating experience log ---")
    _, _, _, global_log = generate_training_data(num_episodes=500, steps_per_episode=300)
    print(f"  Log: {len(global_log):,} entries")

    print("\n--- Building mental model ---")
    engine = build_mental_model(global_log, core_obs_dim=96)
    stats = engine.get_stats()
    print(f"  Mappings: {stats['total_mappings']:,}")

    print("\n--- Running receptor discovery ---")
    results = discover(global_log, engine)

    print(f"\n  Discovered: {len(results['discovered'])} / {len(results['discovered']) + len(results['not_found'])}")
    print(f"\n  {'Receptor':<25} {'Family':<20} {'Score':>8} {'Thresh':>8} {'Status'}")
    print(f"  {'-'*25} {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    for d in results['details']:
        status = 'FOUND' if d['discovered'] else 'not found'
        marker = '+' if d['discovered'] else ' '
        print(f"  {marker} {d['receptor_id']:<23} {d['family']:<20} {d['score']:>8.3f} {d['threshold']:>8.2f} {status}")

    print(f"\n  Topology vector: {results['topology_vector']}")
    print(f"  Discovered receptors: {results['discovered']}")

    path = os.path.join(DATA_DIR, 'receptor_discovery.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")
    print("\nStep 32 complete.")

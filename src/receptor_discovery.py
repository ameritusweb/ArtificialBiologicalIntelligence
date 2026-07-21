import os
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any
from environment import Environment, Organism, NPC
from mental_model import build_mental_model, action_to_hash, CORE_OBS_DIM
from train import generate_training_data, PROBE_RATE_FLOOR

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@dataclass
class ReceptorTest:
    receptor_id: str
    family: str
    description: str
    threshold: float
    test_fn: Callable = None


def build_tests():
    tests = []

    def test_static_repetition(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        high_count = [e for e in all_entries if e.count >= 5]
        low_count = [e for e in all_entries if e.count <= 2]
        if not high_count or not low_count:
            return 0.0
        avg_high = np.mean([e.certainty for e in high_count])
        avg_low = np.mean([e.certainty for e in low_count])
        return avg_high / (avg_low + 1e-8)

    tests.append(ReceptorTest('static_repetition', 'repetition',
        'Certainty higher for repeated vs novel contexts', 1.2, test_static_repetition))

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
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        correct_order = 0
        total_pairs = 0
        for i in range(1, N):
            pain_before = np.mean(log[i-1]['obs_after'][0:L])
            pain_after = np.mean(log[i]['obs_after'][0:L])
            ta_before = np.mean(log[i-1]['obs_after'][6*L+1:7*L+1])
            ta_after = np.mean(log[i]['obs_after'][6*L+1:7*L+1])
            if pain_before > 0.3:
                total_pairs += 1
                if ta_after > ta_before:
                    correct_order += 1
        return correct_order / (total_pairs + 1e-8)

    tests.append(ReceptorTest('precedence_detection', 'causality',
        'Pain precedes temporal aversion increase (directional)', 0.5, test_precedence))

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
        pain = np.array([e['obs_after'][0:L] for e in log[:N]])
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
        N = min(10000, len(log))
        opt_count = 0
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > len(obs) - 5 and obs[-5] > 0.3:
                opt_count += 1
        return opt_count / N

    tests.append(ReceptorTest('optimism', 'meta_motivational',
        'Goal persistence > 0.3 frequency', 0.2, test_optimism))

    def test_conflict(log, engine, **kw):
        N = min(10000, len(log))
        conflict_count = 0
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > len(obs) - 3 and obs[-5] > 0.2:
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
            pain_avg = np.mean(obs[0:L])
            energy = obs[energy_idx] if len(obs) > energy_idx else 1.0
            if pain_avg > 0.3 and energy < 0.3:
                stress_count += 1
        return stress_count / N

    tests.append(ReceptorTest('stress_detection', 'regulatory',
        'High pain + low energy co-occurrence', 0.05, test_stress))

    def test_change_detection(log, engine, **kw):
        N = min(10000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        pe_start = 8 * L + 42
        spikes = 0
        for e in log[:N]:
            obs = e['obs_after']
            if len(obs) > pe_start + L:
                pe = np.mean(obs[pe_start:pe_start + L])
                if pe > 0.2:
                    spikes += 1
        return spikes / max(1, N // 300)

    tests.append(ReceptorTest('change_detection', 'observation',
        'Prediction error spikes per episode', 10.0, test_change_detection))

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
        endo_with = np.mean([np.mean(e['obs_after'][L:2*L]) for e in with_temp])
        endo_without = np.mean([np.mean(e['obs_after'][L:2*L]) for e in without_temp])
        return abs(endo_with - endo_without) / (endo_without + 1e-8)

    tests.append(ReceptorTest('cross_modal_association', 'association',
        'Cross-channel prediction boost (temp present vs absent)', 0.2, test_cross_modal_association))

    def test_functional_similarity(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        action_deltas = defaultdict(list)
        for e in log[:N]:
            ah = action_to_hash(e['action'])
            delta = e['obs_after'][:min(6, cdim)] - e['obs_before'][:min(6, cdim)]
            action_deltas[ah].append(delta)
        groups = {ah: np.mean(d, axis=0) for ah, d in action_deltas.items() if len(d) >= 3}
        if len(groups) < 5:
            return 0.0
        means = list(groups.values())
        from itertools import combinations
        sims = []
        for a, b in list(combinations(range(len(means)), 2))[:500]:
            cos = np.dot(means[a], means[b]) / (np.linalg.norm(means[a]) * np.linalg.norm(means[b]) + 1e-8)
            sims.append(cos)
        high_sim_pairs = sum(1 for s in sims if s > 0.8)
        return high_sim_pairs / (len(sims) + 1e-8)

    tests.append(ReceptorTest('functional_similarity', 'similarity',
        'Different actions produce similar outcomes (functional equivalence)', 0.05, test_functional_similarity))

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
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        ctrls, diversities = [], []
        window = 10
        for i in range(window, N):
            obs = log[i]['obs_after']
            if len(obs) > cdim + 32:
                ctrls.append(obs[cdim + 32])
            actions_window = [action_to_hash(log[j]['action']) for j in range(i-window, i)]
            diversities.append(len(set(actions_window)) / window)
        if len(ctrls) < 100:
            return 0.0
        min_len = min(len(ctrls), len(diversities))
        corr = np.corrcoef(ctrls[:min_len], diversities[:min_len])[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    tests.append(ReceptorTest('agency_salience', 'agency',
        'Action diversity correlates with controllability', 0.15, test_agency_salience))

    def test_impulse_override(log, engine, **kw):
        N = min(10000, len(log))
        L = min(engine.core_obs_dim // 9, 6)
        persist_count = 0
        total_pain = 0
        for e in log[:N]:
            obs = e['obs_after']
            pain_avg = np.mean(obs[0:L])
            if len(obs) > len(obs) - 5:
                persistence = obs[-5] if len(obs) > 5 else 0
                distant_endo_max = np.max(obs[8*L+34:8*L+42]) if len(obs) > 8*L+42 else 0
                if pain_avg > 0.2:
                    total_pain += 1
                    if persistence > 0.3 and distant_endo_max > 0.3:
                        persist_count += 1
        return persist_count / (total_pain + 1e-8)

    tests.append(ReceptorTest('impulse_override', 'meta_motivational',
        'Persists through pain when future reward predicted', 0.1, test_impulse_override))

    def test_arousal_regulation(log, engine, **kw):
        N = min(5000, len(log))
        L = min(engine.core_obs_dim // 9, 6)
        energy_idx = 6 * L
        dangers, action_rates = [], []
        for i in range(1, N):
            obs = log[i]['obs_after']
            pain_avg = np.mean(obs[0:L])
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
            pain_after = np.mean(log[i]['obs_after'][0:L])
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
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        npc_start = cdim + 40
        if len(log[0]['obs_after']) <= npc_start + 3:
            return 0.0
        npc_dists = [e['obs_after'][npc_start] for e in log[:N]]
        rewards = [e['reward'] for e in log[:N]]
        corr = np.corrcoef(npc_dists[:-1], rewards[1:])[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    tests.append(ReceptorTest('behavioral_prediction', 'social',
        'NPC distance predicts next-step reward', 0.1, test_behavioral_prediction))

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
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        exceptions = 0
        for e in log[:N]:
            pred, cert, n = engine.predict_delta(e['obs_before'][:cdim], e['action'])
            if cert > 0.6 and n > 0:
                actual = e['obs_after'][:cdim] - e['obs_before'][:cdim]
                error = np.mean((pred - actual)**2)
                if error > 0.1:
                    exceptions += 1
        return exceptions / N

    tests.append(ReceptorTest('exception_detection', 'formalization',
        'High-certainty predictions that fail (exceptions)', 0.02, test_exception_detection))

    def test_rule_extraction(log, engine, **kw):
        all_entries = [entry for entries in engine.store.mappings.values() for entry in entries]
        if len(all_entries) < 100:
            return 0.0
        rules = [e for e in all_entries if e.certainty > 0.6 and e.count >= 5]
        return len(rules) / (len(all_entries) + 1e-8)

    tests.append(ReceptorTest('rule_extraction', 'formalization',
        'Fraction of mappings that qualify as reliable rules', 0.03, test_rule_extraction))

    def test_boundary_detection(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        gradient_changes = 0
        total = 0
        for i in range(1, N):
            pain_now = np.mean(log[i]['obs_after'][0:L])
            pain_prev = np.mean(log[i-1]['obs_after'][0:L])
            delta = abs(pain_now - pain_prev)
            total += 1
            if delta > 0.15:
                gradient_changes += 1
        return gradient_changes / (total + 1e-8)

    tests.append(ReceptorTest('boundary_detection', 'formalization',
        'Sharp receptor changes between consecutive steps (boundaries crossed)', 0.05, test_boundary_detection))

    # === COMPLETION + MATHEMATICS + ORGANIZATION families ===

    def test_completion(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        sharp_drops = 0
        total_high_conflict = 0
        for i in range(2, N):
            obs_prev = log[i-1]['obs_after']
            obs_now = log[i]['obs_after']
            endo_now = np.mean(obs_now[L:2*L])
            if len(obs_prev) > cdim + 55 and len(obs_now) > cdim + 55:
                conflict_prev = obs_prev[cdim + 55] if cdim + 55 < len(obs_prev) else 0
                conflict_now = obs_now[cdim + 55] if cdim + 55 < len(obs_now) else 0
                if conflict_prev > 0.2:
                    total_high_conflict += 1
                    drop = conflict_prev - conflict_now
                    if drop > 0.1 and endo_now > 0.2:
                        sharp_drops += 1
        return sharp_drops / (total_high_conflict + 1e-8)

    tests.append(ReceptorTest('completion', 'compression',
        'Sharp conflict drop when reward is reached (gestalt completes)', 0.05, test_completion))

    def test_quantity_detection(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        high_endo_approach = 0
        low_endo_approach = 0
        for i in range(1, N):
            dist_endo = log[i]['obs_after'][8*L+34:8*L+42] if len(log[i]['obs_after']) > 8*L+42 else np.zeros(8)
            max_distant = np.max(dist_endo) if len(dist_endo) > 0 else 0
            speed = np.sqrt(log[i]['obs_after'][cdim+2]**2 if len(log[i]['obs_after']) > cdim+2 else 0)
            if max_distant > 0.5:
                high_endo_approach += speed
            elif max_distant < 0.1:
                low_endo_approach += speed
        if low_endo_approach < 1e-6:
            return 0.0
        return high_endo_approach / (low_endo_approach + 1e-8)

    tests.append(ReceptorTest('quantity_detection', 'mathematics',
        'Organism moves faster toward high-endorphin vs low-endorphin regions', 1.2, test_quantity_detection))

    def test_ratio_detection(log, engine, **kw):
        N = min(5000, len(log))
        cdim = engine.core_obs_dim
        L = min(cdim // 9, 6)
        pain_endo_ratios = []
        rewards = []
        for e in log[:N]:
            pain = np.mean(e['obs_after'][0:L])
            endo = np.mean(e['obs_after'][L:2*L])
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
            pain_prev = np.mean(log[i-1]['obs_after'][0:L])
            pain_now = np.mean(log[i]['obs_after'][0:L])
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

    return tests


def discover(log, engine, threshold_overrides=None):
    tests = build_tests()
    results = {
        'discovered': [],
        'not_found': [],
        'scores': {},
        'thresholds': {},
        'details': [],
    }

    for test in tests:
        try:
            score = test.test_fn(log, engine)
        except Exception:
            score = 0.0
        threshold = threshold_overrides.get(test.receptor_id, test.threshold) \
            if threshold_overrides else test.threshold
        passed = score >= threshold
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

"""Live receptor bank: 73 receptor signals computed per-step.

Every receptor outputs a float in [0, 1]. The bank maintains ring buffers
for buffer-computable receptors and computes live-ready ones from the
current obs + engine state.

These are the colored dots at the top of the architecture diagram.
They feed back into the observation vector every cycle.
"""

import math
import numpy as np
from collections import deque
from model import compute_obs_indices

NUM_LIVE_RECEPTORS = 73

RECEPTOR_NAMES = [
    # --- LIVE-READY (37) ---
    # Causality
    'causal_inference', 'counterfactual_reasoning', 'multiple_hypotheses',
    'intervention_planning',
    # Agency
    'self_model',
    # Meta-motivational
    'context_conditioned_arbitration', 'regret', 'multiple_receptor_types',
    'metacognition',
    # Regulatory
    'stress_detection', 'receptor_propagation', 'emotional_intelligence',
    # Compression
    'pattern_recognition', 'compression_gain', 'concept_formation',
    'concept_grounding', 'chunking', 'compression_receptor',
    'mental_model_confidence', 'prediction_accuracy',
    # Sequential processing
    'pipeline_detection', 'prediction_architecture_awareness',
    # Perception
    'staged_processing', 'prediction_branching', 'processing_speed',
    'adaptive_depth',
    # Epistemic
    'belief_detection', 'doubt_detection', 'counterfactual_salience',
    # Mathematics
    'ratio_detection', 'proof_structure', 'necessity_detection',
    'formal_composition',
    # Organization
    'part_whole_detection', 'organizational_mirror',
    # Interaction
    'grip_affordance_live',
    # Logic
    'semantic_relation',

    # --- BUFFER-COMPUTABLE (36) ---
    # Repetition
    'static_repetition', 'rhythm', 'rhythmic_pattern', 'nested_rhythm',
    'causal_rhythm',
    # Association
    'basic_sensorimotor_loop',
    # Causality
    'coincidence_detection', 'precedence_detection', 'probabilistic_causation',
    'causal_graph_reasoning',
    # Agency
    'agency_salience',
    # Meta-motivational
    'curiosity_live',
    # Regulatory
    'rhythm_entrainment', 'self_soothing', 'social_coregulation',
    # Social
    'self_model_applied_to_others',
    # Compression
    'categorical_compression', 'completion',
    # Observation
    'change_detection', 'absence_observation', 'comparative_observation',
    # Formalization
    'boundary_detection', 'exception_detection', 'rule_extraction',
    'rule_revision',
    # Mathematics
    'exhaustive_search',
    # Organization
    'org_boundary_detection',
    # Self-augmentation
    'capability_change_detection', 'developmental_trajectory',
    # Interaction
    'lever_affordance', 'contact_response', 'push_affordance',
    # Environmental augmentation
    'environmental_trend_detection',
    # Sequential processing
    'cross_pipeline_prediction',
    # Epistemic
    'epistemic_strategy',
    # Logic
    'transitivity',
]

assert len(RECEPTOR_NAMES) == NUM_LIVE_RECEPTORS


class LiveReceptorBank:
    """Computes 73 live receptor signals per step."""

    def __init__(self):
        self.idx = compute_obs_indices()
        L = self.idx['total_limbs']
        self._L = L

        # Ring buffers
        self.pain_buf = deque(maxlen=400)
        self.reward_buf = deque(maxlen=400)
        self.energy_buf = deque(maxlen=20)
        self.gain_buf = deque(maxlen=20)
        self.action_hash_buf = deque(maxlen=10)
        self.action_raw_buf = deque(maxlen=5)
        self.pe_buf = deque(maxlen=10)
        self.obs_buf = deque(maxlen=5)
        self.npc_erratic_buf = deque(maxlen=10)
        self.emission_buf = deque(maxlen=10)
        self.proximity_buf = deque(maxlen=10)

        # Running state
        self.prev_obs = None
        self.prev_action = None
        self.prev_action_hash = 0
        self.prev_certainty = 0.0
        self.stress_onset_step = -1
        self.visited_cells = set()
        self.co_occur_count = 0
        self.total_count = 0
        self.step_count = 0

    def compute(self, obs, action, engine, reward=0.0, active_mask=None):
        """Compute all 73 receptor values. Returns ndarray of shape [73].

        If active_mask is provided (bool array of length 73), only computes
        receptors where active_mask is True. Inactive receptors return 0.
        This skips expensive engine queries for receptors the activation
        manager has deactivated.
        """
        out = np.zeros(NUM_LIVE_RECEPTORS, dtype=np.float64)
        L = self._L
        idx = self.idx

        pain = obs[0:L]
        endo = obs[L:2*L]
        mean_pain = float(np.mean(pain))
        mean_endo = float(np.mean(endo))
        energy = float(obs[idx['energy']])
        conflict = float(obs[idx['conflict']]) if idx['conflict'] < len(obs) else 0.0
        pe_start = idx.get('pe_start', 90)
        pe = obs[pe_start:pe_start+L] if pe_start+L <= len(obs) else np.zeros(L)
        mean_pe = float(np.mean(np.abs(pe)))

        mm_start = idx['mm_start']
        mm_cert = float(obs[mm_start+2]) if mm_start+2 < len(obs) else 0.0
        mm_lp = float(obs[mm_start+3]) if mm_start+3 < len(obs) else 0.0

        opt_start = idx['opt_start']
        optimism = float(obs[opt_start]) if opt_start < len(obs) else 0.0
        persistence = float(obs[opt_start+1]) if opt_start+1 < len(obs) else 0.0

        npc_start = idx['npc_start']
        npc_obs = obs[npc_start:npc_start+12] if npc_start+12 <= len(obs) else np.zeros(12)
        npc_dist = float(npc_obs[0]) if len(npc_obs) > 0 else 0.0
        npc_erratic = float(npc_obs[4]) if len(npc_obs) > 4 else 0.0

        grip_start = idx['grip_start']
        grip_state = obs[grip_start:grip_start+L] if grip_start+L <= len(obs) else np.zeros(L)
        carried_mass = float(obs[grip_start+L]) if grip_start+L < len(obs) else 0.0
        contact_force = float(obs[grip_start+L+2]) if grip_start+L+2 < len(obs) else 0.0

        obj_start = idx['obj_start']
        obj_prox = obs[obj_start:obj_start+3] if obj_start+3 <= len(obs) else np.zeros(3)

        agency_start = idx['agency_start']
        ctrl = float(obs[agency_start]) if agency_start < len(obs) else 0.0

        efference_start = idx['efference_start']

        core_obs_dim = idx['core_obs_dim']
        action_hash = self._action_hash(action)

        # Update buffers (always, even if receptors skipped)
        self.pain_buf.append(mean_pain)
        self.reward_buf.append(reward)
        self.energy_buf.append(energy)
        gain_start = idx['gain_start']
        self.gain_buf.append(float(np.mean(obs[gain_start:gain_start+L])) if gain_start+L <= len(obs) else 1.0)
        self.action_hash_buf.append(action_hash)
        self.action_raw_buf.append(action.copy() if hasattr(action, 'copy') else np.array(action))
        self.pe_buf.append(mean_pe)
        self.obs_buf.append(obs.copy())
        self.npc_erratic_buf.append(npc_erratic)
        emission_bits = action[L*3:] if len(action) > L*3 else np.zeros(4)
        self.emission_buf.append(float(np.sum(emission_bits)))
        self.proximity_buf.append(float(np.mean(obj_prox)))
        self.step_count += 1

        def _active(i):
            return active_mask is None or active_mask[i]

        # ====== CACHED ENGINE QUERIES ======
        # Compute shared results once, reuse across receptors
        _pred_prev = _cert_prev = _n_prev = None
        _pred_curr = _cert_curr = _n_curr = None
        _pred_null = _cert_null = _n_null = None
        _emb_curr = None
        _chain_result = None
        _concept_stats = None
        _pattern_stats = None
        _store_stats = None

        def _get_pred_prev():
            nonlocal _pred_prev, _cert_prev, _n_prev
            if _pred_prev is None and engine is not None and self.prev_obs is not None and self.prev_action is not None:
                _pred_prev, _cert_prev, _n_prev = engine.predict_delta(self.prev_obs[:core_obs_dim], self.prev_action)
            return _pred_prev, _cert_prev, _n_prev if _pred_prev is not None else (np.zeros(core_obs_dim), 0.0, 0)

        def _get_pred_curr():
            nonlocal _pred_curr, _cert_curr, _n_curr
            if _pred_curr is None and engine is not None:
                _pred_curr, _cert_curr, _n_curr = engine.predict_delta(obs[:core_obs_dim], action)
            return _pred_curr, _cert_curr, _n_curr if _pred_curr is not None else (np.zeros(core_obs_dim), 0.0, 0)

        def _get_pred_null():
            nonlocal _pred_null, _cert_null, _n_null
            if _pred_null is None and engine is not None:
                null_act = np.zeros_like(action)
                _pred_null, _cert_null, _n_null = engine.predict_delta(obs[:core_obs_dim], null_act)
            return _pred_null, _cert_null, _n_null if _pred_null is not None else (np.zeros(core_obs_dim), 0.0, 0)

        def _get_emb():
            nonlocal _emb_curr
            if _emb_curr is None and engine is not None:
                _emb_curr = engine.encoder.embed(obs[:core_obs_dim])
            return _emb_curr

        def _get_chain():
            nonlocal _chain_result
            if _chain_result is None and engine is not None and len(self.action_raw_buf) >= 2:
                prev2 = self.action_raw_buf[-2]
                _chain_result = engine.chain([prev2, action], obs[:core_obs_dim])
            return _chain_result if _chain_result is not None else (np.zeros(core_obs_dim), 0.0)

        def _get_concept_stats():
            nonlocal _concept_stats
            if _concept_stats is None and engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
                _concept_stats = engine.pattern_store.get_concept_stats()
            return _concept_stats or {}

        def _get_pattern_stats():
            nonlocal _pattern_stats
            if _pattern_stats is None and engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
                _pattern_stats = engine.pattern_store.get_stats()
            return _pattern_stats or {}

        def _get_store_high_count():
            nonlocal _store_stats
            if _store_stats is None and engine is not None:
                high = 0
                reliable = 0
                calibrated = 0
                total = engine.store.total_count
                for entries in engine.store.mappings.values():
                    el = entries if isinstance(entries, list) else [entries]
                    for e in el:
                        if e.count >= 5 and e.certainty > 0.6:
                            high += 1
                        if e.certainty > 0.6 and e.count >= 5:
                            reliable += 1
                        if 0.3 < e.certainty < 0.9:
                            calibrated += 1
                _store_stats = {'high': high, 'reliable': reliable,
                                'calibrated': calibrated, 'total': total}
            return _store_stats or {'high': 0, 'reliable': 0, 'calibrated': 0, 'total': 0}

        # ===================== LIVE-READY (37) =====================

        # 0: causal_inference — cosine(predicted, actual)
        if _active(0) and engine is not None and self.prev_obs is not None and self.prev_action is not None:
            pred, cert, n = _get_pred_prev()
            if n > 0:
                actual = obs[:core_obs_dim] - self.prev_obs[:core_obs_dim]
                pn = np.linalg.norm(pred) + 1e-8
                an = np.linalg.norm(actual) + 1e-8
                out[0] = float(np.clip((np.dot(pred, actual[:len(pred)]) / (pn * an) + 1) / 2, 0, 1))

        # 1: counterfactual_reasoning — predict(chosen) vs predict(null)
        if _active(1) and engine is not None:
            pred_chosen, cc, nc = _get_pred_curr()
            pred_null, cn, nn = _get_pred_null()
            if nc > 0 and nn > 0:
                diff = float(np.mean(pred_chosen[:L]) - np.mean(pred_null[:L]))
                out[1] = float(np.clip(abs(diff), 0, 1))

        # 2: multiple_hypotheses
        if _active(2) and engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            emb = _get_emb()
            if emb is not None:
                results = engine.pattern_store.query(action_hash, emb, top_k=5)
                out[2] = float(np.clip(len(results) / 5.0, 0, 1))

        # 3: intervention_planning
        if _active(3) and engine is not None:
            pred, cert, n = _get_pred_curr()
            if n > 0:
                pain_reduction = -float(np.mean(pred[:L]))
                out[3] = float(np.clip(pain_reduction * ctrl, 0, 1))

        # 4: self_model
        if _active(4) and engine is not None and self.prev_obs is not None and self.prev_action is not None:
            pred, cert, n = _get_pred_prev()
            if n > 0:
                actual = obs[:core_obs_dim] - self.prev_obs[:core_obs_dim]
                pred_mag = float(np.linalg.norm(pred))
                act_mag = float(np.linalg.norm(actual[:len(pred)]))
                if act_mag > 1e-6:
                    out[4] = float(np.clip(1.0 - abs(pred_mag - act_mag) / (act_mag + 1e-6), 0, 1))

        # 5: context_conditioned_arbitration
        if _active(5) and engine is not None and hasattr(engine, 'family_manager') and engine.family_manager is not None:
            fs = engine.family_manager.get_stats()
            spreads = fs.get('weight_spreads', {})
            if spreads:
                max_spread = max(s['std'] for s in spreads.values())
                out[5] = float(np.clip(max_spread * 5, 0, 1))

        # 6: regret
        if _active(6) and engine is not None and self.prev_obs is not None and self.prev_action is not None:
            pred_taken, ct, nt = _get_pred_prev()
            pred_null_prev, cn, nn = engine.predict_delta(self.prev_obs[:core_obs_dim], np.zeros_like(self.prev_action))
            if nt > 0 and nn > 0:
                val_taken = -float(np.mean(pred_taken[:L])) + float(np.mean(pred_taken[L:2*L]))
                val_null = -float(np.mean(pred_null_prev[:L])) + float(np.mean(pred_null_prev[L:2*L]))
                out[6] = float(np.clip(max(0, val_null - val_taken), 0, 1))

        # 7: multiple_receptor_types
        groups = [(0, L), (L, 2*L), (2*L, 3*L), (3*L, 4*L), (4*L, 5*L), (5*L, 6*L)]
        active = sum(1 for s, e in groups if e <= len(obs) and float(np.mean(np.abs(obs[s:e]))) > 0.15)
        out[7] = float(active / max(len(groups), 1))

        # 8: metacognition
        out[8] = float(np.clip(mm_cert * (1.0 - mean_pe), 0, 1))

        # 9: stress_detection
        out[9] = 1.0 if mean_pain > 0.3 and energy < 0.3 else 0.0

        # 10: receptor_propagation
        out[10] = float(np.clip(npc_dist * npc_erratic, 0, 1))

        # 11: emotional_intelligence
        npc_distress = float(npc_obs[5]) if len(npc_obs) > 5 else 0.0
        out[11] = float(np.clip(npc_distress * float(np.sum(emission_bits)) / 4.0, 0, 1))

        # 12: pattern_recognition
        if _active(12):
            cs = _get_concept_stats()
            out[12] = float(np.clip(cs.get('num_stable_concepts', 0) / 50.0, 0, 1))

        # 13: compression_gain
        if _active(13):
            ps = _get_pattern_stats()
            out[13] = float(np.clip(ps.get('avg_compression_gain', 0), 0, 1))

        # 14: concept_formation
        if _active(14):
            cs = _get_concept_stats()
            out[14] = float(np.clip(cs.get('avg_concept_quality', 0), 0, 1))

        # 15: concept_grounding
        if _active(15) and engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            concepts = engine.pattern_store.extract_concepts(top_k=3) if hasattr(engine.pattern_store, 'extract_concepts') else []
            if concepts:
                best = concepts[0]
                cd = best.cumulative_delta if hasattr(best, 'cumulative_delta') else np.zeros(core_obs_dim)
                affect = abs(float(np.mean(cd[:L]))) + abs(float(np.mean(cd[L:2*L])))
                out[15] = float(np.clip(affect, 0, 1))

        # 16: chunking
        if _active(16) and engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            emb = _get_emb()
            if emb is not None:
                results = engine.pattern_store.query(action_hash, emb, top_k=1)
                if results:
                    out[16] = float(np.clip(results[0][1], 0, 1))

        # 17: compression_receptor
        if _active(17):
            cs = _get_concept_stats()
            total = cs.get('total_patterns', 1)
            stable = cs.get('num_stable_concepts', 0)
            out[17] = float(np.clip(stable / max(total, 1), 0, 1))

        # 18: mental_model_confidence
        if _active(18) and engine is not None:
            pred, cert, n = _get_pred_curr()
            out[18] = float(np.clip(cert, 0, 1)) if n > 0 else 0.0

        # 19: prediction_accuracy
        out[19] = out[0]  # same as causal_inference — cosine accuracy

        # 20: pipeline_detection
        if _active(20) and engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            emb = _get_emb()
            if emb is not None:
                results = engine.pattern_store.query(action_hash, emb, top_k=3)
                out[20] = float(np.clip(len(results) / 3.0, 0, 1))

        # 21: prediction_architecture_awareness
        if _active(21):
            cs = _get_concept_stats()
            out[21] = float(np.clip(cs.get('num_stable_concepts', 0) / 100.0, 0, 1))

        # 22: staged_processing
        mm_fam = float(obs[mm_start]) if mm_start < len(obs) else 0.0
        pat_avail = float(obs[idx['pattern_start']]) if idx['pattern_start'] < len(obs) else 0.0
        out[22] = float(np.clip(mm_fam * pat_avail * (1.0 - mean_pe), 0, 1))

        # 23: prediction_branching
        out[23] = float(np.clip(mean_pe * mm_cert, 0, 1))

        # 24: processing_speed
        out[24] = float(np.clip(mm_fam * (1.0 - mean_pe), 0, 1))

        # 25: adaptive_depth
        out[25] = out[20]  # pattern query depth

        # 26: belief_detection
        out[26] = mm_cert

        # 27: doubt_detection
        out[27] = float(np.clip(persistence * mean_pe, 0, 1))

        # 28: counterfactual_salience
        if _active(28) and engine is not None:
            pred_null, cn, nn = _get_pred_null()
            if nn > 0:
                null_pain = float(np.mean(pred_null[:L]))
                out[28] = float(np.clip(abs(null_pain) * optimism, 0, 1))

        # 29: ratio_detection
        denom = mean_pain + mean_endo + 1e-8
        out[29] = float(np.clip(mean_endo / denom, 0, 1))

        # 30: proof_structure
        if _active(30):
            chain_delta, chain_cert = _get_chain()
            out[30] = float(np.clip(chain_cert, 0, 1))

        # 31: necessity_detection
        if _active(31):
            ps = _get_pattern_stats()
            high_cert = ps.get('num_high_certainty', 0) if 'num_high_certainty' in ps else 0
            total = ps.get('total_patterns', 1)
            out[31] = float(np.clip(high_cert / max(total, 1), 0, 1))

        # 32: formal_composition
        if _active(32) and engine is not None:
            chain_delta, chain_cert = _get_chain()
            single_delta, sc, sn = _get_pred_curr()
            if sn > 0:
                diff = float(np.linalg.norm(chain_delta - single_delta[:len(chain_delta)]))
                out[32] = float(np.clip(diff * chain_cert, 0, 1))

        # 33: part_whole_detection
        muscle_acts = action[:L*3].reshape(L, 3) if len(action) >= L*3 else np.zeros((L, 3))
        ext_sum = float(np.sum(muscle_acts[:, 0]))
        agreement = max(ext_sum, L - ext_sum) / max(L, 1)
        out[33] = float(np.clip(agreement, 0, 1))

        # 34: organizational_mirror
        limb_dev_start = idx['limb_dev_start']
        if limb_dev_start + L <= len(obs):
            dev_std = float(np.std(obs[limb_dev_start:limb_dev_start+L]))
            act_std = float(np.std(muscle_acts[:, 0]))
            out[34] = float(np.clip(1.0 - abs(dev_std - act_std), 0, 1))

        # 35: grip_affordance_live
        out[35] = float(np.clip(float(np.max(grip_state)) + carried_mass, 0, 1))

        # 36: semantic_relation
        if engine is not None and hasattr(engine, 'entity_store'):
            es = engine.entity_store.get_stats()
            out[36] = float(np.clip(len(es.get('entity_ids', [])) / 10.0, 0, 1))

        # ===================== BUFFER-COMPUTABLE (36) =====================

        # 37: static_repetition
        if _active(37):
            ss = _get_store_high_count()
            out[37] = float(np.clip(ss['high'] / 50.0, 0, 1))

        # 38: rhythm — autocorrelation of pain
        if len(self.pain_buf) >= 100:
            buf = np.array(self.pain_buf)
            buf = buf - buf.mean()
            norm = float(np.dot(buf, buf))
            if norm > 1e-8:
                best_ac = 0.0
                for lag in [50, 75, 100]:
                    if lag < len(buf):
                        ac = float(np.dot(buf[lag:], buf[:-lag])) / norm
                        best_ac = max(best_ac, ac)
                out[38] = float(np.clip(best_ac, 0, 1))

        # 39: rhythmic_pattern — multiple lag autocorrelations
        if len(self.pain_buf) >= 80:
            buf = np.array(list(self.pain_buf)[-80:])
            buf = buf - buf.mean()
            norm = float(np.dot(buf, buf))
            if norm > 1e-8:
                count = 0
                for lag in [10, 20, 30, 40]:
                    ac = float(np.dot(buf[lag:], buf[:-lag])) / norm
                    if ac > 0.3:
                        count += 1
                out[39] = float(count / 4.0)

        # 40: nested_rhythm — std envelope autocorrelation
        if len(self.pain_buf) >= 200:
            buf = np.array(list(self.pain_buf)[-200:])
            win = 20
            envelope = np.array([float(np.std(buf[i:i+win])) for i in range(0, len(buf)-win, win)])
            if len(envelope) >= 5:
                envelope = envelope - envelope.mean()
                norm = float(np.dot(envelope, envelope))
                if norm > 1e-8:
                    ac = float(np.dot(envelope[1:], envelope[:-1])) / norm
                    out[40] = float(np.clip(ac, 0, 1))

        # 41: causal_rhythm — reward periodicity at pain lag
        if len(self.pain_buf) >= 100 and len(self.reward_buf) >= 100:
            pbuf = np.array(list(self.pain_buf)[-100:])
            rbuf = np.array(list(self.reward_buf)[-100:])
            pbuf = pbuf - pbuf.mean()
            rbuf = rbuf - rbuf.mean()
            pn = float(np.linalg.norm(pbuf)) + 1e-8
            rn = float(np.linalg.norm(rbuf)) + 1e-8
            best = 0.0
            for lag in [25, 50, 75]:
                if lag < len(pbuf):
                    cc = float(np.dot(pbuf[lag:], rbuf[:-lag])) / (pn * rn)
                    best = max(best, abs(cc))
            out[41] = float(np.clip(best, 0, 1))

        # 42: basic_sensorimotor_loop
        if self.prev_obs is not None and self.prev_action is not None:
            obs_delta = float(np.linalg.norm(obs[:core_obs_dim] - self.prev_obs[:core_obs_dim]))
            act_change = float(np.sum(np.abs(action - self.prev_action)))
            out[42] = float(np.clip(obs_delta * act_change / 10.0, 0, 1))

        # 43: coincidence_detection
        self.total_count += 1
        if mean_pain > 0.2 and mean_endo > 0.2:
            self.co_occur_count += 1
        base_rate = (self.co_occur_count / max(self.total_count, 1))
        out[43] = float(np.clip(base_rate * 5.0, 0, 1))

        # 44: precedence_detection
        if len(self.pain_buf) >= 3:
            pain_onset = self.pain_buf[-3] < 0.1 and self.pain_buf[-2] > 0.2
            ta_start = idx['ta_start']
            ta_now = float(np.mean(obs[ta_start:ta_start+L])) if ta_start+L <= len(obs) else 0.0
            ta_prev = float(np.mean(self.prev_obs[ta_start:ta_start+L])) if self.prev_obs is not None and ta_start+L <= len(self.prev_obs) else 0.0
            if pain_onset and ta_now > ta_prev:
                out[44] = 1.0

        # 45: probabilistic_causation
        if _active(45):
            ss = _get_store_high_count()
            if ss['total'] > 0:
                out[45] = float(np.clip(ss['calibrated'] / max(ss['total'], 1), 0, 1))

        # 46: causal_graph_reasoning
        if engine is not None and len(self.action_raw_buf) >= 2 and self.prev_obs is not None:
            prev2 = self.action_raw_buf[-2]
            chain_delta, chain_cert = engine.chain([prev2, action], self.prev_obs[:core_obs_dim])
            if chain_cert > 0.1:
                actual = obs[:core_obs_dim] - self.prev_obs[:core_obs_dim]
                pn = float(np.linalg.norm(chain_delta)) + 1e-8
                an = float(np.linalg.norm(actual[:len(chain_delta)])) + 1e-8
                cos = float(np.dot(chain_delta, actual[:len(chain_delta)])) / (pn * an)
                out[46] = float(np.clip((cos + 1) / 2, 0, 1))

        # 47: agency_salience
        if len(self.action_hash_buf) >= 5:
            unique = len(set(list(self.action_hash_buf)[-5:]))
            out[47] = float(np.clip(ctrl * unique / 5.0, 0, 1))

        # 48: curiosity_live — revisiting high-PE regions
        if mean_pe > 0.3:
            cell = (int(obs[0] * 10) if len(obs) > 0 else 0,
                    int(obs[1] * 10) if len(obs) > 1 else 0)
            was_new = cell not in self.visited_cells
            self.visited_cells.add(cell)
            out[48] = 1.0 if was_new else 0.3
        elif mean_pe > 0.1:
            out[48] = 0.1

        # 49: rhythm_entrainment
        if len(self.action_raw_buf) >= 5 and len(self.pain_buf) >= 50:
            act_series = [float(np.sum(a[:L*3])) for a in list(self.action_raw_buf)[-5:]]
            act_arr = np.array(act_series) - np.mean(act_series)
            an = float(np.dot(act_arr, act_arr))
            if an > 1e-8 and len(act_arr) >= 2:
                ac = float(np.dot(act_arr[1:], act_arr[:-1])) / an
                out[49] = float(np.clip(ac, 0, 1))

        # 50: self_soothing
        if self.prev_obs is not None:
            prev_pain = float(np.mean(self.prev_obs[:L]))
            if prev_pain > 0.4 and mean_pain < prev_pain:
                act_intensity = float(np.sum(action[:L*3])) / (L * 3)
                out[50] = float(np.clip(1.0 - act_intensity, 0, 1))

        # 51: social_coregulation (lagged: did past emission reduce NPC erraticism?)
        if len(self.npc_erratic_buf) >= 5 and len(self.emission_buf) >= 5:
            past_emit = float(self.emission_buf[-5])
            erratic_then = float(self.npc_erratic_buf[-5])
            erratic_now = npc_erratic
            if past_emit > 0 and erratic_then > 0.1:
                reduction = erratic_then - erratic_now
                out[51] = float(np.clip(reduction, 0, 1))

        # 52: self_model_applied_to_others
        if len(self.obs_buf) >= 3:
            prev3 = self.obs_buf[-3]
            own_turn = float(obs[idx['proprio_start']+1]) if idx['proprio_start']+1 < len(obs) else 0.0
            prev_npc_bearing = float(prev3[npc_start+2]) if npc_start+2 < len(prev3) else 0.0
            npc_bearing_now = float(npc_obs[2]) if len(npc_obs) > 2 else 0.0
            npc_change = abs(npc_bearing_now - prev_npc_bearing)
            out[52] = float(np.clip(abs(own_turn) * npc_change, 0, 1))

        # 53: categorical_compression
        if _active(53):
            ss = _get_store_high_count()
            out[53] = float(np.clip(ss['high'] / 100.0, 0, 1))

        # 54: completion
        if self.prev_obs is not None:
            prev_conflict = float(self.prev_obs[idx['conflict']]) if idx['conflict'] < len(self.prev_obs) else 0.0
            conflict_drop = prev_conflict - conflict
            endo_spike = mean_endo - float(np.mean(self.prev_obs[L:2*L]))
            if conflict_drop > 0.2 and endo_spike > 0.1:
                out[54] = float(np.clip(conflict_drop + endo_spike, 0, 1))

        # 55: change_detection
        if self.prev_obs is not None:
            ext_change = float(np.linalg.norm(obs[:core_obs_dim] - self.prev_obs[:core_obs_dim]))
            act_change = float(np.sum(np.abs(action - self.prev_action))) if self.prev_action is not None else 0
            out[55] = float(np.clip(ext_change / (act_change + 1.0), 0, 1))

        # 56: absence_observation
        if self.prev_certainty > 0.5 and mm_cert < self.prev_certainty - 0.1:
            out[56] = float(np.clip(self.prev_certainty - mm_cert, 0, 1))

        # 57: comparative_observation
        if len(self.energy_buf) >= 10:
            trend = np.polyfit(range(10), list(self.energy_buf)[-10:], 1)[0]
            out[57] = float(np.clip(abs(trend) * 20, 0, 1))

        # 58: boundary_detection
        if self.prev_obs is not None:
            dp_start = idx['distant_pain_start']
            if dp_start + 8 <= len(obs) and dp_start + 8 <= len(self.prev_obs):
                grad = float(np.max(np.abs(obs[dp_start:dp_start+8] - self.prev_obs[dp_start:dp_start+8])))
                out[58] = float(np.clip(grad * 3, 0, 1))

        # 59: exception_detection
        if len(self.pe_buf) >= 3:
            recent_pe = list(self.pe_buf)[-3:]
            if recent_pe[-1] > 2 * np.mean(recent_pe[:-1]) and recent_pe[-1] > 0.2:
                out[59] = float(np.clip(recent_pe[-1], 0, 1))

        # 60: rule_extraction
        if _active(60):
            ss = _get_store_high_count()
            out[60] = float(np.clip(ss['reliable'] / 30.0, 0, 1))

        # 61: rule_revision
        if engine is not None and hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            patterns = engine.pattern_store.patterns if hasattr(engine.pattern_store, 'patterns') else {}
            revised = 0
            for pats in (patterns.values() if isinstance(patterns, dict) else []):
                for p in (pats if isinstance(pats, list) else [pats]):
                    if hasattr(p, 'certainty') and hasattr(p, 'm2') and p.m2 is not None:
                        if p.certainty > 0.5 and float(np.mean(p.m2)) > 0.1:
                            revised += 1
            out[61] = float(np.clip(revised / 10.0, 0, 1))

        # 62: exhaustive_search
        pm_start = idx['pain_memory_start']
        if pm_start + 25 <= len(obs):
            grid = obs[pm_start:pm_start+25]
            visited = int(np.sum(grid > 0.01))
            self.visited_cells.update(range(visited))
            out[62] = float(np.clip(visited / 25.0, 0, 1))

        # 63: org_boundary_detection
        if self.prev_obs is not None and self.prev_action is not None:
            dp_start = idx['distant_pain_start']
            if dp_start + 8 <= len(obs):
                grad_cross = float(np.max(obs[dp_start:dp_start+8])) > 0.3
                hash_changed = action_hash != self.prev_action_hash
                out[63] = 1.0 if grad_cross and hash_changed else 0.0

        # 64: capability_change_detection
        if len(self.gain_buf) >= 5:
            recent = list(self.gain_buf)[-5:]
            shift = abs(recent[-1] - recent[0])
            out[64] = float(np.clip(shift * 5, 0, 1))

        # 65: developmental_trajectory
        if len(self.gain_buf) >= 20:
            first = np.mean(list(self.gain_buf)[:10])
            second = np.mean(list(self.gain_buf)[10:])
            slope = second - first
            out[65] = float(np.clip(abs(slope) * 10, 0, 1))

        # 66: lever_affordance
        if self.prev_obs is not None:
            prev_cf = float(self.prev_obs[grip_start+L+2]) if grip_start+L+2 < len(self.prev_obs) else 0.0
            force_change = abs(contact_force - prev_cf)
            out[66] = float(np.clip(force_change * 3, 0, 1))

        # 67: contact_response
        if self.prev_obs is not None:
            prev_contact = float(self.prev_obs[grip_start+L+1]) if grip_start+L+1 < len(self.prev_obs) else 0.0
            contact_now = float(obs[grip_start+L+1]) if grip_start+L+1 < len(obs) else 0.0
            if contact_now > 0 and prev_contact == 0:
                act_change = float(np.sum(np.abs(action - self.prev_action))) if self.prev_action is not None else 0
                out[67] = float(np.clip(act_change / 10.0, 0, 1))

        # 68: push_affordance
        if self.prev_obs is not None and len(self.proximity_buf) >= 2:
            prox_change = self.proximity_buf[-1] - self.proximity_buf[-2]
            extending = float(np.sum(muscle_acts[:, 0])) > 0
            if extending and prox_change != 0:
                out[68] = float(np.clip(abs(prox_change) * 3, 0, 1))

        # 69: environmental_trend_detection
        if len(self.proximity_buf) >= 10:
            trend = np.polyfit(range(10), list(self.proximity_buf)[-10:], 1)[0]
            out[69] = float(np.clip(abs(trend) * 20, 0, 1))

        # 70: cross_pipeline_prediction
        if len(self.pain_buf) >= 3:
            pain_delta = self.pain_buf[-1] - self.pain_buf[-3]
            fatigue_val = float(np.mean(obs[5*L:6*L])) if 6*L <= len(obs) else 0.0
            out[70] = float(np.clip(abs(pain_delta) * fatigue_val * 3, 0, 1))

        # 71: epistemic_strategy
        if len(self.action_hash_buf) >= 5:
            diversity = len(set(list(self.action_hash_buf)[-5:])) / 5.0
            out[71] = float(np.clip(diversity * mm_cert, 0, 1))

        # 72: transitivity
        if engine is not None and len(self.action_raw_buf) >= 2 and self.prev_obs is not None:
            prev2 = self.action_raw_buf[-2]
            chain_d, chain_c = engine.chain([prev2, action], self.prev_obs[:core_obs_dim])
            actual = obs[:core_obs_dim] - self.prev_obs[:core_obs_dim]
            if chain_c > 0.1 and float(np.linalg.norm(actual[:len(chain_d)])) > 1e-6:
                pn = float(np.linalg.norm(chain_d)) + 1e-8
                an = float(np.linalg.norm(actual[:len(chain_d)])) + 1e-8
                cos = float(np.dot(chain_d, actual[:len(chain_d)])) / (pn * an)
                out[72] = float(np.clip((cos + 1) / 2 * chain_c, 0, 1))

        # Update state for next step
        self.prev_obs = obs.copy()
        self.prev_action = action.copy() if hasattr(action, 'copy') else np.array(action)
        self.prev_action_hash = action_hash
        self.prev_certainty = mm_cert

        return np.clip(out, 0.0, 1.0)

    def _action_hash(self, action):
        h = 0
        for i in range(min(len(action), 22)):
            h |= (int(action[i]) & 1) << i
        return h

    def reset(self):
        """Reset all buffers for a new episode/organism."""
        self.pain_buf.clear()
        self.reward_buf.clear()
        self.energy_buf.clear()
        self.gain_buf.clear()
        self.action_hash_buf.clear()
        self.action_raw_buf.clear()
        self.pe_buf.clear()
        self.obs_buf.clear()
        self.npc_erratic_buf.clear()
        self.emission_buf.clear()
        self.proximity_buf.clear()
        self.prev_obs = None
        self.prev_action = None
        self.prev_action_hash = 0
        self.prev_certainty = 0.0
        self.stress_onset_step = -1
        self.visited_cells = set()
        self.co_occur_count = 0
        self.total_count = 0
        self.step_count = 0

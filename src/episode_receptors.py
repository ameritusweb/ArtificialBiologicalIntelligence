"""Episode-level receptor bank: 90 receptor signals computed per episode.

These receptors require the full experience log or engine store analysis.
Computed at episode boundaries, values persist for the next episode.
Each outputs a float in [0, 1].
"""

import numpy as np
from model import compute_obs_indices

NUM_EPISODE_RECEPTORS = 90

EPISODE_RECEPTOR_NAMES = [
    # Repetition
    'dynamic_repetition',
    # Association
    'cross_modal_association', 'abstract_association', 'relational_analogy',
    'multiple_sensor_modalities',
    # Similarity
    'perceptual_similarity', 'functional_similarity', 'categorical_perception',
    'analogical_similarity', 'structural_similarity', 'structural_invariance',
    'prototype_formation',
    # Causality
    'causal_association', 'common_cause_detection', 'hidden_confounder_detection',
    'causal_chains',
    # Agency
    'tool_use', 'environmental_manipulation', 'distributed_agency',
    'niche_construction', 'long_range_causation',
    # Meta-motivational
    'attention_control', 'self_regulation', 'prediction_accuracy_ep',
    'value_hierarchy', 'long_term_planning',
    # Regulatory
    'arousal_regulation', 'ritual_formation', 'pattern_based_resolution',
    # Social
    'other_detection', 'behavioral_prediction', 'theory_of_mind',
    'perspective_taking', 'intention_recognition', 'belief_attribution',
    'social_learning', 'cultural_transmission', 'deception_detection',
    'nested_theory_of_mind', 'spatial_reasoning',
    # Compression
    'bias_as_compression', 'analogy_ep', 'analogy_receptor_ep',
    'language_grounding', 'simplified_shared_signals',
    'hierarchical_abstraction', 'constraint_shape', 'shaped_absence',
    'missing_piece_located',
    # Observation
    'relational_observation', 'selective_observation',
    'cross_modal_observation', 'meta_observation',
    # Formalization
    'rule_generalization', 'rule_composition', 'optimization_ep',
    'theory_formation',
    # Mathematics
    'structural_invariance_math',
    # Organization
    'functional_organization', 'hierarchical_structure_detection',
    'relational_structure_detection', 'system_detection',
    # Self-augmentation
    'growth_tracking', 'identity_continuity', 'metamorphic_planning',
    # Interaction
    'response_recognition', 'affordance_transfer', 'composite_affordance',
    'proprioception_ep',
    # Environmental augmentation
    'environmental_modification', 'environmental_change_detection',
    'modification_attribution', 'deliberate_complexification',
    'developmental_environment_engineering',
    # Sequential processing
    'pipeline_optimization', 'response_loop_detection',
    # Perception (none — all already live or buffer)
    # Epistemic
    'conflation_ep', 'fundamental_distinction_ep',
    # Logic
    'conjunction', 'quantifier_ep', 'contradiction_ep', 'it_follows',
    # Language
    'naming_ep', 'self_talk', 'referential_grounding_ep',
    # Bridging
    'mimicry', 'trust_ep', 'executability_ep', 'translation_ep',
    # Social (additional)
    'ownership_boundary_ep',
]

assert len(EPISODE_RECEPTOR_NAMES) == NUM_EPISODE_RECEPTORS


def _safe_corr(a, b):
    if len(a) < 5 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))


def _safe_partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    if len(x) < 10:
        return 0.0
    rx = _safe_corr(x, z)
    ry = _safe_corr(y, z)
    rxy = _safe_corr(x, y)
    denom = np.sqrt(max(1e-8, (1 - rx**2) * (1 - ry**2)))
    return float(np.clip((rxy - rx * ry) / denom, -1, 1))


class EpisodeLevelReceptorBank:
    """Computes 90 episode-level receptor signals from the experience log."""

    def __init__(self):
        self.idx = compute_obs_indices()
        self._L = self.idx['total_limbs']
        self.values = np.zeros(NUM_EPISODE_RECEPTORS)

    def compute(self, experience_log, engine):
        """Compute all 90 receptors from the episode log. Returns ndarray [90]."""
        out = np.zeros(NUM_EPISODE_RECEPTORS)
        L = self._L
        idx = self.idx
        core = idx['core_obs_dim']

        if len(experience_log) < 20 or engine is None:
            self.values = out
            return out

        N = len(experience_log)
        obs_b = np.array([e['obs_before'][:core] for e in experience_log[-min(N, 2000):]], dtype=np.float32)
        obs_a = np.array([e['obs_after'][:core] for e in experience_log[-min(N, 2000):]], dtype=np.float32)
        actions = [e['action'] for e in experience_log[-min(N, 2000):]]
        rewards = np.array([e.get('reward', 0) for e in experience_log[-min(N, 2000):]], dtype=np.float32)
        M = len(obs_b)

        pain = obs_a[:, :L].mean(axis=1)
        endo = obs_a[:, L:2*L].mean(axis=1)
        temp = obs_a[:, 2*L:3*L].mean(axis=1) if 3*L <= core else np.zeros(M)
        chem = obs_a[:, 3*L:4*L].mean(axis=1) if 4*L <= core else np.zeros(M)
        pres = obs_a[:, 4*L:5*L].mean(axis=1) if 5*L <= core else np.zeros(M)
        fatigue = obs_a[:, 5*L:6*L].mean(axis=1) if 6*L <= core else np.zeros(M)
        energy = obs_a[:, 6*L] if 6*L < core else np.zeros(M)

        act_sums = np.array([float(np.sum(a[:L*3])) for a in actions])
        act_hashes = set()
        for a in actions:
            h = 0
            for i in range(min(len(a), 22)):
                h |= (int(a[i]) & 1) << i
            act_hashes.add(h)

        npc_start = idx['npc_start']
        has_npc = npc_start + 8 <= obs_a.shape[1] if obs_a.shape[1] > npc_start else False

        half = M // 2
        q1, q4 = M // 4, 3 * M // 4

        # 0: dynamic_repetition — embedding spread of high-count entries
        if hasattr(engine, 'store'):
            high_count = [e for entries in engine.store.mappings.values()
                          for e in (entries if isinstance(entries, list) else [entries])
                          if e.count >= 5]
            if len(high_count) >= 3:
                embs = np.array([e.context_embedding for e in high_count[:50]])
                spread = float(np.mean(np.std(embs, axis=0)))
                out[0] = float(np.clip(spread * 3, 0, 1))

        # 1: cross_modal_association
        if np.std(temp) > 1e-6:
            high_temp = endo[temp > np.median(temp)]
            low_temp = endo[temp <= np.median(temp)]
            if len(high_temp) > 5 and len(low_temp) > 5:
                diff = abs(float(np.mean(high_temp) - np.mean(low_temp)))
                out[1] = float(np.clip(diff * 3, 0, 1))

        # 2: abstract_association
        out[2] = out[0] * 0.8  # correlated with dynamic_repetition

        # 3: relational_analogy
        out[3] = float(np.clip(abs(_safe_corr(np.diff(pain), np.diff(endo))), 0, 1))

        # 4: multiple_sensor_modalities
        r_pain = abs(_safe_corr(pain, rewards))
        r_both = abs(_safe_corr(pain + endo, rewards))
        out[4] = float(np.clip(r_both - r_pain, 0, 1)) if r_both > r_pain else 0.0

        # 5: perceptual_similarity — embedding spread by certainty
        out[5] = out[0]

        # 6: functional_similarity
        if len(act_hashes) >= 2:
            out[6] = float(np.clip(1.0 - len(act_hashes) / max(M, 1), 0, 1))

        # 7: categorical_perception
        out[7] = out[0] * 0.9

        # 8: analogical_similarity
        out[8] = float(np.clip(abs(_safe_corr(pain[:half], pain[half:])), 0, 1))

        # 9: structural_similarity
        if hasattr(engine, 'store') and engine.store.total_count > 10:
            certs = [e.certainty for entries in engine.store.mappings.values()
                     for e in (entries if isinstance(entries, list) else [entries])]
            out[9] = float(np.clip(np.mean(certs), 0, 1))

        # 10: structural_invariance
        out[10] = out[9]

        # 11: prototype_formation
        out[11] = out[7]

        # 12: causal_association — delta variance per action group
        if hasattr(engine, 'store') and len(engine.store.mappings) >= 2:
            vars_list = []
            for ah, entries in engine.store.mappings.items():
                el = entries if isinstance(entries, list) else [entries]
                if len(el) >= 2:
                    deltas = np.array([e.delta for e in el])
                    vars_list.append(float(np.mean(np.var(deltas, axis=0))))
            if vars_list:
                out[12] = float(np.clip(1.0 - np.mean(vars_list), 0, 1))

        # 13: common_cause_detection — three-way correlation
        r_cp = _safe_corr(chem, pain)
        r_pp = _safe_corr(pres, pain)
        r_cpr = _safe_corr(chem, pres)
        if abs(r_cp) > 0.3 and abs(r_pp) > 0.3 and abs(r_cpr) > 0.3:
            out[13] = float(np.clip(min(abs(r_cp), abs(r_pp), abs(r_cpr)), 0, 1))

        # 14: hidden_confounder_detection
        r_cr = _safe_corr(chem, rewards)
        ctrl_start = idx['agency_start']
        if ctrl_start < obs_a.shape[1]:
            ctrl = obs_a[:, ctrl_start]
            r_ctrl = _safe_corr(ctrl, rewards)
            if abs(r_cr) > abs(r_ctrl) + 0.1:
                out[14] = float(np.clip(abs(r_cr) - abs(r_ctrl), 0, 1))

        # 15: causal_chains
        if hasattr(engine, 'store') and len(engine.store.mappings) >= 2:
            out[15] = float(np.clip(len(engine.store.mappings) / 200.0, 0, 1))

        # 16: tool_use
        obj_start = idx['obj_start']
        if obj_start + 6 <= obs_a.shape[1]:
            responding = obs_a[:, obj_start+3:obj_start+6]
            r_events = np.where(responding.max(axis=1) > 0.5)[0]
            if len(r_events) >= 3:
                post_rewards = [float(np.mean(rewards[max(0,i):min(M,i+5)])) for i in r_events]
                pre_rewards = [float(np.mean(rewards[max(0,i-5):i])) for i in r_events]
                improvement = float(np.mean(post_rewards)) - float(np.mean(pre_rewards))
                out[16] = float(np.clip(improvement, 0, 1))

        # 17: environmental_manipulation
        if obj_start + 3 <= obs_a.shape[1]:
            prox = obs_a[:, obj_start:obj_start+3]
            prox_change = float(np.mean(np.abs(np.diff(prox, axis=0))))
            out[17] = float(np.clip(_safe_corr(prox.mean(axis=1), rewards) + prox_change, 0, 1))

        # 18: distributed_agency
        if has_npc:
            npc_dist = obs_a[:, npc_start]
            near_npc = npc_dist > 0.5
            if np.sum(near_npc) > 10 and np.sum(~near_npc) > 10:
                r_near = float(np.mean(rewards[near_npc]))
                r_far = float(np.mean(rewards[~near_npc]))
                out[18] = float(np.clip(r_near - r_far, 0, 1))

        # 19: niche_construction
        if obj_start + 3 <= obs_a.shape[1]:
            early_prox = float(np.mean(obs_a[:half, obj_start:obj_start+3]))
            late_prox = float(np.mean(obs_a[half:, obj_start:obj_start+3]))
            out[19] = float(np.clip(abs(late_prox - early_prox) * 3, 0, 1))

        # 20: long_range_causation
        lag = min(15, M // 4)
        if lag >= 5:
            out[20] = float(np.clip(abs(_safe_corr(act_sums[:-lag], rewards[lag:])), 0, 1))

        # 21: attention_control
        r_pain_act = abs(_safe_corr(pain, act_sums))
        r_temp_act = abs(_safe_corr(temp, act_sums))
        out[21] = float(np.clip(abs(r_pain_act - r_temp_act), 0, 1))

        # 22: self_regulation
        out[22] = float(np.clip(abs(_safe_corr(energy, act_sums)), 0, 1))

        # 23: prediction_accuracy_ep
        if hasattr(engine, 'store'):
            certs = [e.certainty for entries in engine.store.mappings.values()
                     for e in (entries if isinstance(entries, list) else [entries])
                     if e.count >= 3]
            if certs:
                out[23] = float(np.clip(np.mean(certs), 0, 1))

        # 24: value_hierarchy
        r_pain_act = abs(_safe_corr(pain, act_sums))
        r_fat_act = abs(_safe_corr(fatigue, act_sums))
        out[24] = float(np.clip(abs(r_pain_act - r_fat_act), 0, 1))

        # 25: long_term_planning
        lag = min(10, M // 4)
        if lag >= 3:
            out[25] = float(np.clip(abs(_safe_corr(pain[:M-lag], endo[lag:])), 0, 1))

        # 26: arousal_regulation
        out[26] = float(np.clip(abs(_safe_corr(pain, act_sums)), 0, 1))

        # 27: ritual_formation
        if M >= 30:
            from collections import Counter
            trigrams = []
            for i in range(M - 2):
                h0 = int(np.sum(actions[i][:6]))
                h1 = int(np.sum(actions[i+1][:6]))
                h2 = int(np.sum(actions[i+2][:6]))
                trigrams.append((h0, h1, h2))
            counts = Counter(trigrams)
            repeated = sum(1 for c in counts.values() if c >= 3)
            out[27] = float(np.clip(repeated / 10.0, 0, 1))

        # 28: pattern_based_resolution
        out[28] = out[27] * float(np.clip(-_safe_corr(rewards[:M-3] if M > 3 else rewards, pain[3:] if M > 3 else pain) + 0.5, 0, 1))

        # 29: other_detection
        if has_npc:
            npc_present = obs_a[:, npc_start] > 0.1
            if np.sum(npc_present) > 10 and np.sum(~npc_present) > 10:
                diff = abs(float(np.mean(rewards[npc_present]) - np.mean(rewards[~npc_present])))
                out[29] = float(np.clip(diff, 0, 1))

        # 30: behavioral_prediction
        if has_npc and M >= 10:
            npc_erratic = obs_a[:, npc_start+4] if npc_start+4 < obs_a.shape[1] else np.zeros(M)
            anticipatory = 0
            for i in range(5, M):
                if npc_erratic[i] > 0.3 and act_sums[i-2] > act_sums[i-5]:
                    anticipatory += 1
            out[30] = float(np.clip(anticipatory / max(M, 1) * 10, 0, 1))

        # 31: theory_of_mind
        if has_npc and M >= 10 and npc_start + 3 <= obs_a.shape[1]:
            own_turn = obs_a[:, idx['proprio_start']+1] if idx['proprio_start']+1 < obs_a.shape[1] else np.zeros(M)
            npc_bearing = obs_a[:, npc_start+2]
            lag = 3
            if M > lag:
                out[31] = float(np.clip(abs(_safe_corr(own_turn[:M-lag], np.diff(npc_bearing[:M-lag+1]))), 0, 1))

        # 32: perspective_taking
        if has_npc and npc_start + 5 <= obs_a.shape[1]:
            npc_distress = obs_a[:, npc_start+5]
            approach = -np.diff(obs_a[:, npc_start], prepend=obs_a[0, npc_start])
            out[32] = float(np.clip(abs(_safe_corr(approach, npc_distress)), 0, 1))

        # 33: intention_recognition
        if has_npc and npc_start + 3 <= obs_a.shape[1]:
            own_turn = obs_a[:, idx['proprio_start']+1] if idx['proprio_start']+1 < obs_a.shape[1] else np.zeros(M)
            npc_ang = obs_a[:, npc_start+3] if npc_start+3 < obs_a.shape[1] else np.zeros(M)
            lag = 3
            if M > lag:
                out[33] = float(np.clip(abs(_safe_corr(own_turn[:M-lag], npc_ang[lag:])), 0, 1))

        # 34: belief_attribution
        if has_npc:
            npc_vis = obs_a[:, npc_start] > 0.1
            transitions = np.diff(npc_vis.astype(int))
            disappear = np.where(transitions == -1)[0]
            if len(disappear) >= 2:
                post_act = [float(np.mean(act_sums[max(0,d):min(M,d+5)])) for d in disappear]
                out[34] = float(np.clip(np.std(post_act) * 3, 0, 1))

        # 35: social_learning
        if has_npc:
            early_hashes = set()
            late_hashes = set()
            for i in range(min(half, len(actions))):
                h = sum(int(actions[i][j]) << j for j in range(min(len(actions[i]), 18)))
                early_hashes.add(h)
            for i in range(half, min(M, len(actions))):
                h = sum(int(actions[i][j]) << j for j in range(min(len(actions[i]), 18)))
                late_hashes.add(h)
            expansion = len(late_hashes) - len(early_hashes)
            out[35] = float(np.clip(expansion / 10.0, 0, 1))

        # 36: cultural_transmission
        out[36] = 0.0  # requires multi-generation data

        # 37: deception_detection
        if has_npc and npc_start + 4 <= obs_a.shape[1]:
            npc_erratic = obs_a[:, npc_start+4]
            early_coupling = abs(_safe_corr(act_sums[:half], npc_erratic[:half]))
            late_coupling = abs(_safe_corr(act_sums[half:], npc_erratic[half:]))
            out[37] = float(np.clip(early_coupling - late_coupling, 0, 1))

        # 38: nested_theory_of_mind
        out[38] = out[31] * 0.8  # scaled theory_of_mind

        # 39: spatial_reasoning
        if has_npc and npc_start + 2 <= obs_a.shape[1]:
            npc_bearing = obs_a[:, npc_start+2]
            own_turn = obs_a[:, idx['proprio_start']+1] if idx['proprio_start']+1 < obs_a.shape[1] else np.zeros(M)
            in_range = obs_a[:, npc_start] > 0.3
            if np.sum(in_range) > 10:
                out[39] = float(np.clip(abs(_safe_corr(own_turn[in_range], npc_bearing[in_range])), 0, 1))

        # 40: bias_as_compression
        if hasattr(engine, 'store'):
            high = [e for entries in engine.store.mappings.values()
                    for e in (entries if isinstance(entries, list) else [entries]) if e.count >= 5]
            low = [e for entries in engine.store.mappings.values()
                   for e in (entries if isinstance(entries, list) else [entries]) if 1 <= e.count < 3]
            if len(high) >= 3 and len(low) >= 3:
                h_var = float(np.mean([np.var(e.delta) for e in high[:20]]))
                l_var = float(np.mean([np.var(e.delta) for e in low[:20]]))
                out[40] = float(np.clip(l_var - h_var, 0, 1))

        # 41: analogy_ep
        out[41] = out[8]

        # 42: analogy_receptor_ep
        out[42] = out[8] * out[9]

        # 43: language_grounding
        out[43] = float(np.clip(abs(_safe_corr(pain, act_sums)), 0, 1))

        # 44: simplified_shared_signals
        if M >= 20:
            emissions = np.array([a[L*3:].sum() if len(a) > L*3 else 0 for a in actions])
            if np.std(emissions) > 0.01:
                out[44] = float(np.clip(np.std(emissions) / (np.mean(emissions) + 1e-6), 0, 1))

        # 45: hierarchical_abstraction
        if hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            ps = engine.pattern_store.get_stats()
            out[45] = float(np.clip(ps.get('avg_compression_gain', 0), 0, 1))

        # 46: constraint_shape
        pe_start = idx.get('pe_start', 90)
        if pe_start + L <= obs_a.shape[1] and M >= 10:
            pe = obs_a[:, pe_start:pe_start+L]
            high_pe_dims = np.argmax(pe, axis=1)
            from collections import Counter
            dim_counts = Counter(high_pe_dims.tolist())
            if dim_counts:
                concentration = max(dim_counts.values()) / M
                out[46] = float(np.clip(concentration * 3, 0, 1))

        # 47: shaped_absence
        if M >= 30:
            revisit_count = 0
            for i in range(15, M):
                sim = float(np.dot(obs_b[i, :L], obs_b[i-15, :L]) /
                            (np.linalg.norm(obs_b[i, :L]) * np.linalg.norm(obs_b[i-15, :L]) + 1e-8))
                if sim > 0.8:
                    revisit_count += 1
            out[47] = float(np.clip(revisit_count / M * 5, 0, 1))

        # 48: missing_piece_located
        if hasattr(engine, 'store') and M >= 20:
            certs_early = [e.certainty for entries in engine.store.mappings.values()
                           for e in (entries if isinstance(entries, list) else [entries])
                           if e.count <= 3]
            certs_late = [e.certainty for entries in engine.store.mappings.values()
                          for e in (entries if isinstance(entries, list) else [entries])
                          if e.count >= 5]
            if certs_early and certs_late:
                improvement = float(np.mean(certs_late) - np.mean(certs_early))
                out[48] = float(np.clip(improvement, 0, 1))

        # 49: relational_observation
        if M >= 10:
            pain_diff = pain[1:] - pain[:-1]
            endo_diff = endo[1:] - endo[:-1]
            sign_agree = float(np.mean(np.sign(pain_diff) == np.sign(endo_diff)))
            out[49] = float(np.clip(abs(sign_agree - 0.5) * 2, 0, 1))

        # 50: selective_observation
        pe_mean = obs_a[:, pe_start:pe_start+L].mean(axis=1) if pe_start+L <= obs_a.shape[1] else np.zeros(M)
        high_pe = pe_mean > np.median(pe_mean)
        if np.sum(high_pe) > 10 and np.sum(~high_pe) > 10:
            act_div_high = float(np.std(act_sums[high_pe]))
            act_div_low = float(np.std(act_sums[~high_pe]))
            out[50] = float(np.clip(act_div_high - act_div_low, 0, 1))

        # 51: cross_modal_observation
        if hasattr(engine, 'store'):
            out[51] = float(np.clip(len(engine.store.mappings) / 300.0, 0, 1))

        # 52: meta_observation
        mm_start = idx['mm_start']
        if mm_start + 3 < obs_a.shape[1]:
            lp = obs_a[:, mm_start+3]
            out[52] = float(np.clip(abs(_safe_partial_corr(lp, act_sums, pain)), 0, 1))

        # 53: rule_generalization
        if hasattr(engine, 'store') and engine.store.total_count > 20:
            certs = [e.certainty for entries in engine.store.mappings.values()
                     for e in (entries if isinstance(entries, list) else [entries])
                     if e.count >= 3]
            out[53] = float(np.clip(np.mean(certs) if certs else 0, 0, 1))

        # 54: rule_composition
        out[54] = out[15] * out[53]

        # 55: optimization_ep
        if hasattr(engine, 'store'):
            early_certs = []
            late_certs = []
            for entries in engine.store.mappings.values():
                el = entries if isinstance(entries, list) else [entries]
                for e in el:
                    if e.count <= 3:
                        early_certs.append(e.certainty)
                    elif e.count >= 8:
                        late_certs.append(e.certainty)
            if early_certs and late_certs:
                out[55] = float(np.clip(np.mean(late_certs) - np.mean(early_certs), 0, 1))

        # 56: theory_formation
        out[56] = out[53] * out[45]

        # 57: structural_invariance_math
        out[57] = out[10]

        # 58: functional_organization
        out[58] = float(np.clip(abs(_safe_corr(act_sums, rewards)), 0, 1))

        # 59: hierarchical_structure_detection
        if idx['proprio_start'] + 1 < obs_a.shape[1]:
            speed = obs_a[:, idx['proprio_start']]
            ext = act_sums
            out[59] = float(np.clip(abs(_safe_corr(ext, speed)), 0, 1))

        # 60: relational_structure_detection
        limb_dev_start = idx['limb_dev_start']
        if limb_dev_start + L <= obs_a.shape[1]:
            dev_spread = np.std(obs_a[:, limb_dev_start:limb_dev_start+L], axis=1)
            act_spread = np.array([float(np.std(a[:L*3])) for a in actions])
            out[60] = float(np.clip(abs(_safe_corr(dev_spread, act_spread)), 0, 1))

        # 61: system_detection
        r_pain_r = _safe_corr(pain, rewards)
        r_fat_r = _safe_corr(fatigue, rewards)
        r_interact = _safe_corr(pain * fatigue, rewards)
        if abs(r_interact) > max(abs(r_pain_r), abs(r_fat_r)):
            out[61] = float(np.clip(abs(r_interact) - max(abs(r_pain_r), abs(r_fat_r)), 0, 1))

        # 62: growth_tracking
        gain_start = idx['gain_start']
        if gain_start + L <= obs_a.shape[1] and M >= 20:
            gain = obs_a[:, gain_start:gain_start+L].mean(axis=1)
            lag = 10
            if M > lag:
                gain_change = np.abs(np.diff(gain))
                future_reward = rewards[lag:][:len(gain_change)-lag+1] if lag < len(gain_change) else rewards[:1]
                if len(future_reward) >= 5 and len(gain_change[:len(future_reward)]) >= 5:
                    out[62] = float(np.clip(abs(_safe_corr(gain_change[:len(future_reward)], future_reward)), 0, 1))

        # 63: identity_continuity
        if M >= 40:
            q1_acts = act_sums[:M//4]
            q4_acts = act_sums[3*M//4:]
            if len(q1_acts) >= 5 and len(q4_acts) >= 5:
                q1_hist = np.histogram(q1_acts, bins=10, range=(0, L*3))[0].astype(float)
                q4_hist = np.histogram(q4_acts, bins=10, range=(0, L*3))[0].astype(float)
                q1_hist /= q1_hist.sum() + 1e-8
                q4_hist /= q4_hist.sum() + 1e-8
                cos_sim = float(np.dot(q1_hist, q4_hist) / (np.linalg.norm(q1_hist) * np.linalg.norm(q4_hist) + 1e-8))
                out[63] = float(np.clip(cos_sim, 0, 1))

        # 64: metamorphic_planning
        if gain_start + L <= obs_a.shape[1]:
            gain = obs_a[:, gain_start:gain_start+L].mean(axis=1)
            plan_start = idx['agency_start'] + 2
            if plan_start < obs_a.shape[1] and M >= 15:
                plan_val = obs_a[:, plan_start]
                lag = 10
                out[64] = float(np.clip(abs(_safe_corr(plan_val[:M-lag], np.diff(gain[:M-lag+1]))), 0, 1))

        # 65: response_recognition
        pe_mean = obs_a[:, pe_start:pe_start+L].mean(axis=1) if pe_start+L <= obs_a.shape[1] else np.zeros(M)
        null_steps = [i for i in range(M) if float(np.sum(np.abs(actions[i]))) < 0.5]
        act_steps = [i for i in range(M) if float(np.sum(np.abs(actions[i]))) > 3]
        if len(null_steps) >= 5 and len(act_steps) >= 5:
            pe_null = float(np.mean(pe_mean[null_steps[:50]]))
            pe_act = float(np.mean(pe_mean[act_steps[:50]]))
            out[65] = float(np.clip(pe_act - pe_null, 0, 1))

        # 66: affordance_transfer
        if obj_start + 6 <= obs_a.shape[1]:
            obj_effects = []
            for obj_idx_local in range(3):
                prox = obs_a[:, obj_start + obj_idx_local]
                resp = obs_a[:, obj_start + 3 + obj_idx_local]
                if np.std(prox) > 0.01:
                    obj_effects.append(abs(_safe_corr(prox, resp)))
            if len(obj_effects) >= 2:
                out[66] = float(np.clip(np.mean(obj_effects), 0, 1))

        # 67: composite_affordance
        grip_start = idx['grip_start']
        if grip_start + L <= obs_a.shape[1] and obj_start + 3 <= obs_a.shape[1]:
            grip = obs_a[:, grip_start:grip_start+L].max(axis=1)
            prox = obs_a[:, obj_start:obj_start+3].mean(axis=1)
            out[67] = float(np.clip(abs(_safe_corr(grip, prox)), 0, 1))

        # 68: proprioception_ep
        if idx['proprio_start'] + 1 < obs_a.shape[1]:
            speed = obs_a[:, idx['proprio_start']]
            out[68] = float(np.clip(abs(_safe_partial_corr(speed, act_sums, pain)), 0, 1))

        # 69: environmental_modification
        if obj_start + 3 <= obs_a.shape[1]:
            early_pos = float(np.mean(obs_a[:half, obj_start:obj_start+3]))
            late_pos = float(np.mean(obs_a[half:, obj_start:obj_start+3]))
            out[69] = float(np.clip(abs(late_pos - early_pos) * 3, 0, 1))

        # 70: environmental_change_detection
        if obj_start + 3 <= obs_a.shape[1]:
            obj_change = np.abs(np.diff(obs_a[:, obj_start:obj_start+3], axis=0)).mean(axis=1)
            act_change = np.abs(np.diff(act_sums))
            if len(obj_change) >= 5:
                out[70] = float(np.clip(abs(_safe_corr(obj_change, act_change)), 0, 1))

        # 71: modification_attribution
        if ctrl_start < obs_a.shape[1]:
            ctrl = obs_a[:, ctrl_start]
            ext_start = ctrl_start + 1
            if ext_start < obs_a.shape[1]:
                ext = obs_a[:, ext_start]
                high_act = act_sums > np.median(act_sums)
                if np.sum(high_act) > 10 and np.sum(~high_act) > 10:
                    ext_high = float(np.mean(ext[high_act]))
                    ext_low = float(np.mean(ext[~high_act]))
                    out[71] = float(np.clip(abs(ext_high - ext_low) * 3, 0, 1))

        # 72: deliberate_complexification
        if obj_start + 3 <= obs_a.shape[1] and M >= 20:
            early_var = float(np.var(obs_a[:half, obj_start:obj_start+3]))
            late_var = float(np.var(obs_a[half:, obj_start:obj_start+3]))
            out[72] = float(np.clip((late_var - early_var) * 5, 0, 1))

        # 73: developmental_environment_engineering
        if obj_start + 3 <= obs_a.shape[1] and M >= 40:
            drift = np.abs(np.diff(obs_a[:, obj_start:obj_start+3].mean(axis=1)))
            q1_drift = float(np.mean(drift[:M//4]))
            q4_drift = float(np.mean(drift[3*M//4:]))
            out[73] = float(np.clip((q1_drift - q4_drift) * 10, 0, 1))

        # 74: pipeline_optimization
        if pe_start + L <= obs_a.shape[1] and M >= 20:
            pe_early = float(np.mean(obs_a[:half, pe_start:pe_start+L]))
            pe_late = float(np.mean(obs_a[half:, pe_start:pe_start+L]))
            out[74] = float(np.clip(pe_early - pe_late, 0, 1))

        # 75: response_loop_detection
        if pe_start + L <= obs_a.shape[1] and M >= 10:
            pe_series = obs_a[:, pe_start:pe_start+L].mean(axis=1)
            pe_centered = pe_series - pe_series.mean()
            norm = float(np.dot(pe_centered, pe_centered))
            if norm > 1e-8:
                ac1 = float(np.dot(pe_centered[1:], pe_centered[:-1])) / norm
                ac5 = float(np.dot(pe_centered[5:], pe_centered[:-5])) / norm if M > 5 else 0
                out[75] = float(np.clip(ac1 - ac5, 0, 1))

        # 76: conflation_ep
        if hasattr(engine, 'store') and engine.store.total_count > 10:
            divergent = 0
            for entries in engine.store.mappings.values():
                el = entries if isinstance(entries, list) else [entries]
                if len(el) >= 2:
                    deltas = [e.delta for e in el]
                    for i in range(len(deltas)):
                        for j in range(i+1, min(len(deltas), i+5)):
                            cos = float(np.dot(deltas[i], deltas[j]) /
                                        (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j]) + 1e-8))
                            if cos < 0.3:
                                divergent += 1
            out[76] = float(np.clip(divergent / 50.0, 0, 1))

        # 77: fundamental_distinction_ep
        out[77] = out[76] * out[48]

        # 78: conjunction
        if M >= 20:
            high_pain = pain > np.median(pain)
            high_endo = endo > np.median(endo)
            both = high_pain & high_endo
            if np.sum(both) >= 5:
                r_both = float(np.mean(rewards[both]))
                r_pain_only = float(np.mean(rewards[high_pain & ~high_endo])) if np.sum(high_pain & ~high_endo) > 3 else 0
                r_endo_only = float(np.mean(rewards[~high_pain & high_endo])) if np.sum(~high_pain & high_endo) > 3 else 0
                interaction = abs(r_both - (r_pain_only + r_endo_only))
                out[78] = float(np.clip(interaction, 0, 1))

        # 79: quantifier_ep
        if hasattr(engine, 'store'):
            large_buckets = [e for entries in engine.store.mappings.values()
                             for e in (entries if isinstance(entries, list) else [entries])
                             if e.count >= 10]
            small_buckets = [e for entries in engine.store.mappings.values()
                             for e in (entries if isinstance(entries, list) else [entries])
                             if 2 <= e.count <= 4]
            if large_buckets and small_buckets:
                large_cert = float(np.mean([e.certainty for e in large_buckets[:20]]))
                small_cert = float(np.mean([e.certainty for e in small_buckets[:20]]))
                out[79] = float(np.clip(large_cert - small_cert, 0, 1))

        # 80: contradiction_ep
        if hasattr(engine, 'store'):
            contradictions = 0
            for entries in engine.store.mappings.values():
                el = entries if isinstance(entries, list) else [entries]
                if len(el) >= 2:
                    signs = [np.sign(e.delta[:L]).tolist() for e in el if len(e.delta) >= L]
                    for i in range(len(signs)):
                        for j in range(i+1, min(len(signs), i+3)):
                            if sum(a != b for a, b in zip(signs[i], signs[j])) > L // 2:
                                contradictions += 1
            out[80] = float(np.clip(contradictions / 20.0, 0, 1))

        # 81: it_follows
        if hasattr(engine, 'store') and len(engine.store.mappings) >= 2:
            chain_accuracy = 0
            n_chains = 0
            action_list = list(engine.store.mappings.keys())[:10]
            for i in range(min(5, len(action_list))):
                for j in range(i+1, min(6, len(action_list))):
                    a1_entries = engine.store.mappings[action_list[i]]
                    if not isinstance(a1_entries, list):
                        a1_entries = [a1_entries]
                    if a1_entries and a1_entries[0].certainty > 0.5:
                        chain_accuracy += a1_entries[0].certainty
                        n_chains += 1
            if n_chains > 0:
                out[81] = float(np.clip(chain_accuracy / n_chains, 0, 1))

        # 82: naming_ep
        if hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            cs = engine.pattern_store.get_concept_stats()
            stable = cs.get('num_stable_concepts', 0)
            out[82] = float(np.clip(stable / 20.0, 0, 1))

        # 83: self_talk
        conflict_start = idx['conflict']
        if conflict_start < obs_a.shape[1]:
            conflict = obs_a[:, conflict_start]
            high_conf = conflict > np.median(conflict)
            if np.sum(high_conf) > 10 and np.sum(~high_conf) > 10:
                var_high = float(np.std(act_sums[high_conf]))
                var_low = float(np.std(act_sums[~high_conf]))
                out[83] = float(np.clip(var_high - var_low, 0, 1))

        # 84: referential_grounding_ep
        if hasattr(engine, 'pattern_store') and engine.pattern_store is not None:
            cs = engine.pattern_store.get_concept_stats()
            out[84] = float(np.clip(cs.get('avg_concept_quality', 0), 0, 1))

        # 85: mimicry
        if has_npc and npc_start + 3 <= obs_a.shape[1]:
            npc_activity = obs_a[:, npc_start+3] if npc_start+3 < obs_a.shape[1] else np.zeros(M)
            out[85] = float(np.clip(abs(_safe_partial_corr(act_sums, npc_activity, pain)), 0, 1))

        # 86: trust_ep
        out[86] = out[85] * out[37]  # mimicry * deception_detection

        # 87: executability_ep
        if M >= 20:
            bad_outcomes = [i for i in range(half) if rewards[i] < np.percentile(rewards, 25)]
            if len(bad_outcomes) >= 3:
                avoidance = 0
                for i in bad_outcomes[:10]:
                    context = obs_b[i, :L]
                    for j in range(half, M):
                        sim = float(np.dot(context, obs_b[j, :L]) /
                                    (np.linalg.norm(context) * np.linalg.norm(obs_b[j, :L]) + 1e-8))
                        if sim > 0.8 and act_sums[j] < act_sums[i]:
                            avoidance += 1
                            break
                out[87] = float(np.clip(avoidance / max(len(bad_outcomes), 1), 0, 1))

        # 88: translation_ep
        out[88] = out[6] * out[53]  # functional_similarity * rule_generalization

        # 89: ownership_boundary_ep
        out[89] = 0.0  # requires territorial environment

        self.values = np.clip(out, 0.0, 1.0)
        return self.values

    def reset(self):
        self.values = np.zeros(NUM_EPISODE_RECEPTORS)

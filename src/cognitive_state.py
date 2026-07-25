"""Cognitive State Detector.

Maintains a codebook of observed co-activation patterns (thought types)
and tracks which stored concept was last retrieved. Feeds both back
as observation channels.

Thought type: which combination of receptors is active (pattern-level).
Concept activation: which specific stored causal chain was retrieved.
"""

import numpy as np
from model import compute_obs_indices


class CognitiveStateDetector:
    """Detects thought types and concept activations at runtime."""

    def __init__(self, max_codebook_size=256, activation_threshold=0.3):
        self.idx = compute_obs_indices()
        self.threshold = activation_threshold
        self.max_codebook_size = max_codebook_size

        # Codebook of known thought types
        # Each entry: binary activation vector
        self.codebook = []
        self.codebook_counts = []

        # Channel groups to monitor for thought types
        self._build_channel_groups()

        # Last detected states
        self.current_thought_type = 0.0
        self.current_concept_id = 0.0

    def _build_channel_groups(self):
        """Define which observation ranges to monitor."""
        idx = self.idx
        self.groups = []
        self.group_names = []

        for name in ['pain', 'endorphin', 'temperature', 'chemical',
                     'pressure', 'fatigue']:
            if name in idx and isinstance(idx[name], tuple):
                self.groups.append(idx[name])
                self.group_names.append(name)

        # Scalar channels
        for name, key in [('energy', 'energy'),
                          ('conflict', 'conflict')]:
            if key in idx:
                v = idx[key]
                if isinstance(v, int):
                    self.groups.append((v, v + 1))
                    self.group_names.append(name)

        # MM features
        mm_s = idx.get('mm_start')
        if mm_s is not None:
            for i, name in enumerate(['mm_certainty', 'mm_lp',
                                       'mm_ctrl', 'mm_plan']):
                self.groups.append((mm_s + i, mm_s + i + 1))
                self.group_names.append(name)

        # Thinking channels
        think_s = idx.get('thinking_start')
        if think_s is not None:
            for i, name in enumerate(['think_val', 'think_ent',
                                       'think_conv', 'think_div',
                                       'think_unex', 'think_depth']):
                self.groups.append((think_s + i, think_s + i + 1))
                self.group_names.append(name)

        self.n_groups = len(self.groups)

    def _get_activation_key(self, obs):
        """Extract binary activation signature from observation."""
        key = np.zeros(self.n_groups, dtype=np.int8)
        for i, (start, end) in enumerate(self.groups):
            if end <= len(obs):
                val = float(np.mean(np.abs(obs[start:end])))
                if val > self.threshold:
                    key[i] = 1
        return key

    def detect_thought_type(self, obs):
        """Match current observation against codebook. Returns normalized type ID."""
        key = self._get_activation_key(obs)

        # Search codebook for match
        best_match = -1
        for i, entry in enumerate(self.codebook):
            if np.array_equal(key, entry):
                best_match = i
                self.codebook_counts[i] += 1
                break

        if best_match < 0:
            # New thought type
            if len(self.codebook) < self.max_codebook_size:
                best_match = len(self.codebook)
                self.codebook.append(key.copy())
                self.codebook_counts.append(1)
            else:
                # Replace least-used entry
                min_idx = int(np.argmin(self.codebook_counts))
                best_match = min_idx
                self.codebook[min_idx] = key.copy()
                self.codebook_counts[min_idx] = 1

        # Normalize to [0, 1]
        self.current_thought_type = best_match / max(1, len(self.codebook) - 1)
        return self.current_thought_type

    def detect_concept_activation(self, engine, obs, prev_action_hash):
        """Identify which concept was retrieved. Returns normalized concept ID."""
        if engine is None or engine.pattern_store is None:
            self.current_concept_id = 0.0
            return 0.0

        pa, pc = engine.query_pattern(prev_action_hash, obs)
        if pa < 0.1:
            self.current_concept_id = 0.0
            return 0.0

        # Get the pattern's hash as an ID
        core_obs = obs[:engine.core_obs_dim] if len(obs) > engine.core_obs_dim else obs
        emb = engine.encoder.embed(core_obs)
        results = engine.pattern_store.query(prev_action_hash, emb, top_k=1)

        if not results:
            self.current_concept_id = 0.0
            return 0.0

        # Use the motif hash as a normalized ID
        pattern, score = results[0]
        motif_hash = hash(pattern.motif) % 1000
        self.current_concept_id = motif_hash / 1000.0
        return self.current_concept_id

    def update(self, obs, engine=None, prev_action_hash=0):
        """Detect both thought type and concept activation. Returns (type_id, concept_id)."""
        tt = self.detect_thought_type(obs)
        ca = self.detect_concept_activation(engine, obs, prev_action_hash)
        return tt, ca

    def get_stats(self):
        return {
            'codebook_size': len(self.codebook),
            'total_observations': sum(self.codebook_counts) if self.codebook_counts else 0,
            'most_common_type': int(np.argmax(self.codebook_counts)) if self.codebook_counts else -1,
            'most_common_count': int(max(self.codebook_counts)) if self.codebook_counts else 0,
        }

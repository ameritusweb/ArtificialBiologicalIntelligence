"""Procedural Memory: replay, peak experience, motor sequences, shortcuts.

Replay: re-experience successful sequences during low-demand periods.
Peak experience: index of highest-reward-delta entries.
Motor sequence store: named action sequences keyed on thought_type_id.
Shortcut executor: state machine for executing stored motor sequences,
bypassing MCTS when a confident match exists.
"""

import numpy as np
from collections import defaultdict, deque


class PeakExperienceIndex:
    """Maintains a sorted index of highest-reward-delta entries in the log.

    At write time, each entry is tagged with its reward delta. The index
    keeps the top-K entries sorted by delta magnitude.
    """

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.entries = []  # (delta, log_index, action, obs_context)

    def add(self, reward_delta, log_index, action, obs_context):
        """Add an entry to the index."""
        self.entries.append((
            float(reward_delta),
            log_index,
            action.copy() if hasattr(action, 'copy') else action,
            obs_context[:96].copy() if len(obs_context) > 96 else obs_context.copy(),
        ))
        if len(self.entries) > self.max_size * 2:
            self.entries.sort(key=lambda x: -x[0])
            self.entries = self.entries[:self.max_size]

    def get_top(self, n=5):
        """Return top-n highest reward delta entries."""
        self.entries.sort(key=lambda x: -x[0])
        return self.entries[:n]

    def get_best_action_for_context(self, obs_context, top_k=10):
        """Find the best historical action for a similar context."""
        if not self.entries:
            return None, 0.0

        self.entries.sort(key=lambda x: -x[0])
        candidates = self.entries[:top_k]

        if len(obs_context) > 96:
            query = obs_context[:96]
        else:
            query = obs_context

        query_norm = np.linalg.norm(query) + 1e-8

        best_action = None
        best_score = -float('inf')

        for delta, idx, action, ctx in candidates:
            ctx_norm = np.linalg.norm(ctx) + 1e-8
            sim = float(np.dot(query, ctx[:len(query)]) / (query_norm * ctx_norm))
            score = delta * max(0.0, sim)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score

    def get_stats(self):
        if not self.entries:
            return {'size': 0, 'max_delta': 0, 'min_delta': 0, 'mean_delta': 0}
        deltas = [e[0] for e in self.entries]
        return {
            'size': len(self.entries),
            'max_delta': round(float(max(deltas)), 3),
            'min_delta': round(float(min(deltas)), 3),
            'mean_delta': round(float(np.mean(deltas)), 3),
        }


class ReplayEngine:
    """Replays successful experience sequences during low-demand periods.

    Triggers during low-threat states (low pain, low conflict, adequate
    energy). Replays entries from the peak index through the mental
    model to reinforce successful mappings.
    """

    def __init__(self, peak_index, replay_threshold=0.3):
        self.peak_index = peak_index
        self.replay_threshold = replay_threshold
        self.replay_count = 0
        self.last_replay_step = -100

    def should_replay(self, obs, step, min_gap=20):
        """Determine if conditions are right for replay."""
        if step - self.last_replay_step < min_gap:
            return False
        if self.peak_index is None or len(self.peak_index.entries) < 5:
            return False

        L = min(6, len(obs) // 9)
        if L < 1:
            return False

        pain = float(np.mean(obs[0:L]))
        energy = float(obs[6*L]) if len(obs) > 6*L else 0.5

        # Replay during low-demand periods
        return pain < self.replay_threshold and energy > 0.3

    def replay(self, engine, step):
        """Replay top entries through the mental model to reinforce them."""
        if engine is None:
            return 0

        top = self.peak_index.get_top(3)
        reinforced = 0

        for delta, log_idx, action, ctx in top:
            # Re-query the mental model with this context to reinforce
            pred, cert, n = engine.predict_delta(ctx, action)
            if n > 0 and cert > 0.3:
                reinforced += 1

        self.replay_count += reinforced
        self.last_replay_step = step
        return reinforced

    def get_stats(self):
        return {
            'total_replays': self.replay_count,
            'peak_index': self.peak_index.get_stats(),
        }


def build_peak_index(experience_log, max_size=100):
    """Build a peak experience index from an existing log."""
    index = PeakExperienceIndex(max_size)

    for i, entry in enumerate(experience_log):
        reward = entry.get('reward', 0.0)
        if reward > 0:
            index.add(reward, i, entry['action'], entry['obs_before'])

    return index


class SequenceEntry:
    __slots__ = ('thought_type_id', 'context_embedding', 'action_sequence',
                 'total_reward', 'times_observed', 'times_fired',
                 'times_succeeded', 'last_fired_gen', '_raw_obs')

    def __init__(self, thought_type_id, context_embedding, action_sequence,
                 total_reward, times_observed=1, times_fired=0,
                 times_succeeded=0, last_fired_gen=-1, raw_obs=None):
        self.thought_type_id = thought_type_id
        self.context_embedding = context_embedding
        self.action_sequence = action_sequence
        self.total_reward = total_reward
        self.times_observed = times_observed
        self.times_fired = times_fired
        self.times_succeeded = times_succeeded
        self.last_fired_gen = last_fired_gen
        self._raw_obs = raw_obs


def _sequence_match_ratio(seq_a, seq_b, num_continuous=0):
    """Positional match ratio for binary action dims."""
    if len(seq_a) != len(seq_b):
        return 0.0
    matches = 0
    for a, b in zip(seq_a, seq_b):
        binary_a = a[num_continuous:]
        binary_b = b[num_continuous:]
        diffs = sum(1 for x, y in zip(binary_a, binary_b) if x != y)
        if diffs < 3:
            matches += 1
    return matches / len(seq_a)


class MotorSequenceStore:
    """Stores action sequences keyed on thought_type_id.

    Separate from the policy and mental model. These are action recipes —
    specific sequences that worked in specific cognitive states.
    """

    def __init__(self, num_continuous=0):
        self.entries = defaultdict(list)
        self.num_continuous = num_continuous

    def store_sequence(self, thought_type_id, context_embedding, action_sequence,
                       total_reward, raw_obs=None):
        bucket = self.entries[thought_type_id]

        for entry in bucket:
            if _sequence_match_ratio(entry.action_sequence, action_sequence,
                                     self.num_continuous) > 0.7:
                entry.total_reward = 0.9 * entry.total_reward + 0.1 * total_reward
                entry.times_observed += 1
                return

        bucket.append(SequenceEntry(
            thought_type_id=thought_type_id,
            context_embedding=context_embedding.copy(),
            action_sequence=[a.copy() for a in action_sequence],
            total_reward=total_reward,
            raw_obs=raw_obs.copy() if raw_obs is not None else None,
        ))

    def query(self, thought_type_id, context_embedding, min_support=3):
        """Find the best stored sequence for this thought type + context.
        Returns (SequenceEntry, score) or (None, 0)."""
        bucket = self.entries.get(thought_type_id, [])
        if not bucket:
            return None, 0.0

        best = None
        best_score = 0.0

        for entry in bucket:
            if entry.times_observed < min_support:
                continue

            confidence = (entry.times_succeeded + 1) / (entry.times_fired + 2)

            ctx_norm = np.linalg.norm(context_embedding) + 1e-8
            ent_norm = np.linalg.norm(entry.context_embedding) + 1e-8
            sim = float(np.dot(context_embedding, entry.context_embedding)
                        / (ctx_norm * ent_norm))
            sim = max(0.0, sim)

            score = sim * confidence * entry.total_reward
            if score > best_score:
                best_score = score
                best = entry

        return best, best_score

    def update_outcome(self, entry, reward, current_gen):
        entry.times_fired += 1
        entry.last_fired_gen = current_gen
        if reward > 0:
            entry.times_succeeded += 1

    def re_embed_all(self, encoder, core_obs_dim):
        """Re-embed all stored context vectors with a new encoder.
        Call after build_mental_model trains a new ContrastiveEncoder,
        otherwise stored and query embeddings are in different spaces."""
        for type_id, bucket in self.entries.items():
            for entry in bucket:
                if hasattr(entry, '_raw_obs') and entry._raw_obs is not None:
                    entry.context_embedding = encoder.embed(
                        entry._raw_obs[:core_obs_dim])

    def evict_stale(self, current_gen, max_staleness=20):
        for type_id in list(self.entries.keys()):
            self.entries[type_id] = [
                e for e in self.entries[type_id]
                if current_gen - e.last_fired_gen < max_staleness
                or e.last_fired_gen < 0
            ]
            if not self.entries[type_id]:
                del self.entries[type_id]

    def extract_sequences(self, experience_log, peak_index, encoder,
                          cog_detector, core_obs_dim, min_length=3,
                          max_length=8):
        """Extract high-reward contiguous runs from experience log."""
        top_entries = peak_index.get_top(20)

        for delta, log_idx, action, ctx in top_entries:
            if log_idx >= len(experience_log):
                continue

            start = log_idx
            while start > 0:
                prev = experience_log[start - 1]
                curr = experience_log[start]
                if prev.get('reward', 0) <= 0:
                    break
                if curr.get('time_step', start) <= prev.get('time_step', start - 1):
                    break
                start -= 1
                if log_idx - start >= max_length - 1:
                    break

            end = log_idx + 1
            while end < len(experience_log):
                curr = experience_log[end]
                prev = experience_log[end - 1]
                if curr.get('reward', 0) <= 0:
                    break
                if curr.get('time_step', end) <= prev.get('time_step', end - 1):
                    break
                end += 1
                if end - start >= max_length:
                    break

            length = end - start
            if length < min_length or length > max_length:
                continue

            sequence = [experience_log[i]['action'] for i in range(start, end)]
            context_obs = experience_log[start].get('obs_before',
                                                     np.zeros(core_obs_dim))
            total_reward = sum(experience_log[i].get('reward', 0)
                               for i in range(start, end))

            embedding = encoder.embed(context_obs[:core_obs_dim])
            thought_type = cog_detector.detect_thought_type(context_obs)

            self.store_sequence(thought_type, embedding, sequence, total_reward,
                                raw_obs=context_obs)

    def get_stats(self):
        total_entries = sum(len(v) for v in self.entries.values())
        total_fired = sum(e.times_fired for v in self.entries.values() for e in v)
        total_succeeded = sum(e.times_succeeded for v in self.entries.values() for e in v)
        return {
            'num_types': len(self.entries),
            'total_entries': total_entries,
            'total_fired': total_fired,
            'total_succeeded': total_succeeded,
            'success_rate': total_succeeded / max(1, total_fired),
        }


class ShortcutExecutor:
    """Manages execution of multi-step motor sequences.
    Tracks position, abortable on condition change."""

    def __init__(self):
        self.current_sequence = None
        self.current_step = 0
        self.sequence_entry = None
        self.start_obs = None
        self.cumulative_reward = 0.0

    def start(self, entry, obs):
        self.current_sequence = entry.action_sequence
        self.current_step = 0
        self.sequence_entry = entry
        self.start_obs = obs.copy()
        self.cumulative_reward = 0.0

    def get_action(self):
        if self.current_sequence is None:
            return None
        if self.current_step >= len(self.current_sequence):
            return None
        action = self.current_sequence[self.current_step]
        self.current_step += 1
        return action

    def is_active(self):
        return self.current_sequence is not None

    def is_complete(self):
        if self.current_sequence is None:
            return False
        return self.current_step >= len(self.current_sequence)

    def should_abort(self, obs):
        """Abort if pain increased substantially since sequence started."""
        if self.start_obs is None:
            return False
        L = min(6, len(obs) // 9)
        if L < 1:
            return False
        pain_now = float(np.mean(obs[0:L]))
        pain_start = float(np.mean(self.start_obs[0:L]))
        return pain_now - pain_start > 0.3

    def add_reward(self, reward):
        self.cumulative_reward += reward

    def finish(self, motor_store, current_gen):
        """End sequence and update store with actual outcome."""
        if self.sequence_entry is not None and motor_store is not None:
            motor_store.update_outcome(
                self.sequence_entry,
                self.cumulative_reward,
                current_gen)
        self.current_sequence = None
        self.current_step = 0
        self.sequence_entry = None
        self.start_obs = None
        self.cumulative_reward = 0.0

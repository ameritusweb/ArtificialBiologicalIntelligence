"""Thinking Substrate: MCTS as cognitive architecture.

The tree records which thinking paths were taken, how often, and with
what outcomes. Its metadata becomes input to receptors: visit patterns
trigger shaped_absence, UCB scores trigger curiosity, value convergence
triggers completion, path divergence triggers exception_detection.

The mental model's predict_delta() is the simulation function.
The receptor topology makes the evaluation function intrinsic.
"""

import math
import numpy as np
from mental_model import action_to_hash


class ThinkingNode:
    __slots__ = ('obs', 'action', 'parent', 'children',
                 'visit_count', 'value_sum', 'prior', '_embedding')

    def __init__(self, obs, action=None, parent=None, prior=0.0):
        self.obs = obs
        self.action = action
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self._embedding = None

    def get_embedding(self, engine):
        if self._embedding is None:
            self._embedding = engine.encoder.embed(
                engine._core_obs(self.obs))
        return self._embedding

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb(self, c=1.4):
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else 1
        exploit = self.value()
        explore = c * math.sqrt(math.log(parent_visits + 1) / self.visit_count)
        return exploit + explore


NUM_THINKING_CHANNELS = 6


class ThinkingTree:
    """MCTS-based thinking substrate.

    Before acting, the organism runs internal simulations using the mental
    model's predict_delta(). The tree structure is analyzed and its metadata
    written into observation channels that receptors can fire on.

    Channels (6 total):
      0: best_value        - highest rollout value found (normalized)
      1: visit_entropy      - how evenly distributed visits are (exploration breadth)
      2: value_convergence  - how stable the value estimates are (completion signal)
      3: path_divergence    - max value difference between sibling branches
      4: underexplored      - fraction of children with < 2 visits (shaped absence)
      5: depth_reached      - deepest path explored (normalized by max_depth)
    """

    def __init__(self, num_actions, max_simulations=32, max_depth=4,
                 explore_constant=1.4, use_ema=False, ema_alpha=0.05,
                 v_weights=None):
        self.num_actions = num_actions
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.explore_constant = explore_constant
        self.v_weights = v_weights
        self.root = None
        self._last_analysis = np.zeros(NUM_THINKING_CHANNELS)
        self._cached_candidates = None
        self._cache_store_count = -1
        self._use_ema = use_ema
        self._ema_mean = np.zeros(NUM_THINKING_CHANNELS)
        self._ema_var = np.ones(NUM_THINKING_CHANNELS)
        self._ema_alpha = ema_alpha
        self._ema_count = 0
        self._channel_log = []
        self._logging = False

    def think(self, obs, engine, candidate_actions=None):
        """Run MCTS from current observation. Returns analysis channels."""
        if engine is None:
            self._last_analysis = np.zeros(NUM_THINKING_CHANNELS)
            return self._last_analysis

        self.root = ThinkingNode(obs)
        self._rollout_max_depth = 0

        if candidate_actions is None:
            store_count = engine.store.total_count
            if self._cached_candidates is None or store_count != self._cache_store_count:
                self._cached_candidates = self._generate_candidates_from_store(engine)
                self._cache_store_count = store_count
            candidate_actions = self._cached_candidates

        for sim in range(self.max_simulations):
            node = self._select(self.root)
            if node.visit_count > 0 and len(node.children) == 0:
                self._expand_batch(node, engine, candidate_actions)
                if node.children:
                    node = node.children[0]
            value = self._rollout(node, engine, candidate_actions)
            self._backpropagate(node, value)

        raw = self._analyze()
        if self._logging:
            self._channel_log.append(raw.copy())
        if self._use_ema:
            self._last_analysis = self._ema_normalize(raw)
        else:
            self._last_analysis = raw
        return self._last_analysis

    def get_best_action(self):
        """Return the action of the most-visited root child."""
        if self.root is None or not self.root.children:
            return None
        best = max(self.root.children, key=lambda c: c.visit_count)
        return best.action

    def get_analysis(self):
        return self._last_analysis.copy()

    def _generate_candidates_from_store(self, engine):
        """Build candidate actions from the mental model's known action hashes.
        These are actions the organism has actually experienced, so
        predict_delta will have entries for them."""
        candidates = []
        seen_hashes = set()
        for ah, entries in engine.store.mappings.items():
            if ah in seen_hashes:
                continue
            seen_hashes.add(ah)
            el = entries if isinstance(entries, list) else [entries]
            best = max(el, key=lambda e: e.certainty)
            if best.certainty < 0.3:
                continue
            if hasattr(best, 'action') and best.action is not None:
                candidates.append(np.array(best.action, dtype=np.int32))
            else:
                a = self._hash_to_action(ah)
                candidates.append(a)
            if len(candidates) >= 16:
                break
        if not candidates:
            candidates.append(np.zeros(self.num_actions, dtype=np.int32))
        return candidates

    def _hash_to_action(self, ah):
        """Reconstruct an action from its hash. Approximate — uses the hash
        bits directly since action_to_hash packs bits into an int."""
        a = np.zeros(self.num_actions, dtype=np.int32)
        for i in range(self.num_actions):
            a[i] = (ah >> i) & 1
        return a

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda c: c.ucb(self.explore_constant))
        return node

    def _expand_batch(self, node, engine, candidate_actions):
        """Expand all candidate actions at once using batched predict_delta."""
        if hasattr(engine, 'predict_delta_batch'):
            batch_results = engine.predict_delta_batch(node.obs, candidate_actions)
            for action, (pred_delta, cert, n) in zip(candidate_actions, batch_results):
                if n == 0:
                    continue
                child_obs = node.obs.copy()
                cdim = min(len(pred_delta), len(child_obs))
                child_obs[:cdim] += pred_delta[:cdim]
                child = ThinkingNode(child_obs, action=action.copy(),
                                     parent=node, prior=cert)
                node.children.append(child)
        else:
            for action in candidate_actions:
                pred_delta, cert, n = engine.predict_delta(node.obs, action)
                if n == 0:
                    continue
                child_obs = node.obs.copy()
                cdim = min(len(pred_delta), len(child_obs))
                child_obs[:cdim] += pred_delta[:cdim]
                child = ThinkingNode(child_obs, action=action.copy(),
                                     parent=node, prior=cert)
                node.children.append(child)

    def _rollout(self, node, engine, candidate_actions, depth=0):
        if depth >= self.max_depth:
            self._rollout_max_depth = max(getattr(self, '_rollout_max_depth', 0), depth)
            return self._evaluate(node.obs)

        if not candidate_actions:
            self._rollout_max_depth = max(getattr(self, '_rollout_max_depth', 0), depth)
            return self._evaluate(node.obs)

        action = candidate_actions[np.random.randint(len(candidate_actions))]
        emb = node.get_embedding(engine)
        pred_delta, cert, n = engine.predict_delta_from_embedding(emb, action)
        if n == 0 or cert < 0.15:
            self._rollout_max_depth = max(getattr(self, '_rollout_max_depth', 0), depth)
            return self._evaluate(node.obs)

        next_obs = node.obs.copy()
        cdim = min(len(pred_delta), len(next_obs))
        next_obs[:cdim] += pred_delta[:cdim]

        if np.random.random() > cert:
            self._rollout_max_depth = max(getattr(self, '_rollout_max_depth', 0), depth + 1)
            return self._evaluate(next_obs)

        rollout_node = ThinkingNode(next_obs, action=action, parent=node)
        return self._evaluate(next_obs) * 0.5 + \
               self._rollout(rollout_node, engine, candidate_actions, depth + 1) * 0.5

    def _evaluate(self, obs):
        """Intrinsic evaluation: receptor-based.

        Uses heritable v_weights when available (evolved valence).
        Falls back to defaults otherwise.
        """
        L = min(6, len(obs) // 9)
        if L < 1:
            return 0.0
        pain = float(np.sum(obs[0:L]))
        endorphin = float(np.sum(obs[L:2*L]))
        temperature = float(np.sum(obs[2*L:3*L])) if len(obs) > 3*L else 0.5
        chemical = float(np.sum(obs[3*L:4*L])) if len(obs) > 4*L else 0.0
        pressure = float(np.sum(obs[4*L:5*L])) if len(obs) > 5*L else 0.0
        energy = float(obs[6*L]) if len(obs) > 6*L else 0.5

        w = self.v_weights
        if w is not None:
            value = (w.get('v_pain', -1.0) * pain +
                     w.get('v_endorphin', 1.0) * endorphin +
                     w.get('v_energy', 0.5) * energy +
                     w.get('v_chemical', 0.3) * chemical +
                     w.get('v_temperature', -0.3) * temperature +
                     w.get('v_pressure', -0.2) * pressure)
        else:
            temp_discomfort = max(0, temperature - 0.7 * L) + max(0, 0.3 * L - temperature)
            value = -pain + endorphin + 0.5 * energy + 0.3 * chemical - 0.3 * temp_discomfort - 0.2 * pressure
        return float(np.clip(value / 5.0, -1.0, 1.0))

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _analyze(self):
        """Extract receptor-relevant metadata from the tree."""
        channels = np.zeros(NUM_THINKING_CHANNELS)
        if self.root is None or not self.root.children:
            return channels

        children = self.root.children
        visits = np.array([c.visit_count for c in children], dtype=float)
        values = np.array([c.value() for c in children])

        # 0: best_value — highest value found
        channels[0] = float(np.max(values)) if len(values) > 0 else 0.0

        # 1: visit_entropy — exploration breadth
        total_visits = visits.sum()
        if total_visits > 0:
            probs = visits / total_visits
            probs = probs[probs > 0]
            entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
            max_entropy = math.log(max(len(children), 1) + 1e-10)
            channels[1] = entropy / max_entropy if max_entropy > 0 else 0.0

        # 2: value_convergence — how stable are the estimates
        if len(values) >= 2:
            channels[2] = max(0.0, 1.0 - float(np.std(values)))

        # 3: path_divergence — max value gap between siblings
        if len(values) >= 2:
            channels[3] = float(np.max(values) - np.min(values))

        # 4: underexplored — fraction with < 2 visits (shaped absence)
        if len(visits) > 0:
            channels[4] = float(np.mean(visits < 2))

        # 5: depth_reached — deepest rollout path (normalized)
        # Uses _rollout_max_depth which tracks the actual deepest rollout
        # across all simulations in this think() call, not just the tree structure
        channels[5] = min(1.0, self._rollout_max_depth / max(self.max_depth, 1))

        return np.clip(channels, -1.0, 1.0)

    def start_logging(self):
        self._logging = True
        self._channel_log = []

    def stop_logging(self):
        self._logging = False

    def get_channel_stats(self):
        """Return per-channel mean, std, min, max from logged data."""
        if not self._channel_log:
            return None
        arr = np.array(self._channel_log)
        names = ['best_value', 'visit_entropy', 'value_convergence',
                 'path_divergence', 'underexplored', 'depth_reached']
        stats = {}
        for i, name in enumerate(names):
            col = arr[:, i]
            stats[name] = {
                'mean': round(float(np.mean(col)), 4),
                'std': round(float(np.std(col)), 4),
                'min': round(float(np.min(col)), 4),
                'max': round(float(np.max(col)), 4),
                'nonzero_frac': round(float(np.mean(col > 0.01)), 4),
            }
        stats['n_samples'] = len(self._channel_log)
        return stats

    def _ema_normalize(self, raw):
        """Normalize raw analysis channels against running EMA statistics.

        Stabilizes the distribution across mental model rebuilds:
        the policy sees z-scored channels whose scale doesn't shift
        when the mental model changes, only when the organism's
        thinking genuinely changes.
        """
        a = self._ema_alpha
        self._ema_count += 1

        if self._ema_count == 1:
            self._ema_mean = raw.copy()
            self._ema_var = np.ones(NUM_THINKING_CHANNELS) * 0.1
            return np.zeros(NUM_THINKING_CHANNELS)

        self._ema_mean = (1 - a) * self._ema_mean + a * raw
        diff = raw - self._ema_mean
        self._ema_var = (1 - a) * self._ema_var + a * (diff * diff)

        std = np.sqrt(self._ema_var + 1e-8)
        normalized = diff / std

        return np.clip(normalized, -2.0, 2.0)

    def _tree_depth(self, node, current=0):
        if not node.children:
            return current
        return max(self._tree_depth(c, current + 1) for c in node.children)


def integrate_thinking(obs, thinking_channels):
    """Append thinking channels to an observation vector."""
    return np.concatenate([obs, thinking_channels])

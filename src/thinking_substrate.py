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
                 'visit_count', 'value_sum', 'prior')

    def __init__(self, obs, action=None, parent=None, prior=0.0):
        self.obs = obs
        self.action = action
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

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
                 explore_constant=1.4):
        self.num_actions = num_actions
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.explore_constant = explore_constant
        self.root = None
        self._last_analysis = np.zeros(NUM_THINKING_CHANNELS)

    def think(self, obs, engine, candidate_actions=None):
        """Run MCTS from current observation. Returns analysis channels."""
        if engine is None:
            self._last_analysis = np.zeros(NUM_THINKING_CHANNELS)
            return self._last_analysis

        self.root = ThinkingNode(obs)

        if candidate_actions is None:
            candidate_actions = self._generate_candidates_from_store(engine)

        for _ in range(self.max_simulations):
            node = self._select(self.root)
            if node.visit_count > 0 and len(node.children) == 0:
                self._expand(node, engine, candidate_actions)
                if node.children:
                    node = node.children[0]
            value = self._rollout(node, engine, candidate_actions)
            self._backpropagate(node, value)

        self._last_analysis = self._analyze()
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

    def _expand(self, node, engine, candidate_actions):
        for action in candidate_actions:
            pred_delta, cert, n = engine.predict_delta(node.obs, action)
            if n == 0:
                continue
            child_obs = node.obs.copy()
            cdim = min(len(pred_delta), len(child_obs))
            child_obs[:cdim] += pred_delta[:cdim]
            prior = cert
            child = ThinkingNode(child_obs, action=action.copy(),
                                 parent=node, prior=prior)
            node.children.append(child)

    def _rollout(self, node, engine, candidate_actions, depth=0):
        if depth >= self.max_depth:
            return self._evaluate(node.obs)

        if not candidate_actions:
            return self._evaluate(node.obs)

        action = candidate_actions[np.random.randint(len(candidate_actions))]
        pred_delta, cert, n = engine.predict_delta(node.obs, action)
        if n == 0 or cert < 0.1:
            return self._evaluate(node.obs)

        next_obs = node.obs.copy()
        cdim = min(len(pred_delta), len(next_obs))
        next_obs[:cdim] += pred_delta[:cdim]

        rollout_node = ThinkingNode(next_obs, action=action, parent=node)
        return self._evaluate(next_obs) * 0.5 + \
               self._rollout(rollout_node, engine, candidate_actions, depth + 1) * 0.5

    def _evaluate(self, obs):
        """Intrinsic evaluation: receptor-based, not designer-specified.

        Value = endorphin - pain + certainty bonus - conflict penalty.
        All terms come from observation channels the organism already has.
        """
        L = min(6, len(obs) // 9)
        if L < 1:
            return 0.0
        pain = float(np.sum(obs[0:L]))
        endorphin = float(np.sum(obs[L:2*L]))
        energy = float(obs[6*L]) if len(obs) > 6*L else 0.5

        value = -pain + endorphin + 0.5 * energy
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

        # 5: depth_reached — deepest path (normalized)
        max_depth = self._tree_depth(self.root)
        channels[5] = min(1.0, max_depth / max(self.max_depth, 1))

        return np.clip(channels, -1.0, 1.0)

    def _tree_depth(self, node, current=0):
        if not node.children:
            return current
        return max(self._tree_depth(c, current + 1) for c in node.children)


def integrate_thinking(obs, thinking_channels):
    """Append thinking channels to an observation vector."""
    return np.concatenate([obs, thinking_channels])

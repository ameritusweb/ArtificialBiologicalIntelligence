"""Receptor Co-activation Logger.

Tracks which observation channels are active together, in sequence,
and in exclusion — making the organism's thought patterns observable.

A "thought" is a cascade of receptor firings. This logger records the
participating set at every timestep, builds co-activation matrices,
detects recurring patterns, and identifies temporal sequences.

Usage:
    logger = CoactivationLogger(idx)
    for step in episode:
        logger.record(obs, step)
    stats = logger.get_stats()
"""

import numpy as np
from collections import defaultdict, Counter
from model import compute_obs_indices


class CoactivationLogger:
    """Records receptor activation patterns across timesteps."""

    def __init__(self, idx=None, activation_threshold=0.3):
        self.idx = idx or compute_obs_indices()
        self.threshold = activation_threshold

        # Named channel groups to track
        self.channels = self._build_channel_map()
        self.n_channels = len(self.channels)
        self.channel_names = list(self.channels.keys())

        # Per-step activation history
        self.activation_history = []

        # Co-activation matrix: how often pairs fire together
        self.coactivation_counts = np.zeros((self.n_channels, self.n_channels), dtype=int)

        # Sequence counts: how often A at t is followed by B at t+1
        self.sequence_counts = np.zeros((self.n_channels, self.n_channels), dtype=int)

        # Per-channel activation counts
        self.activation_counts = np.zeros(self.n_channels, dtype=int)

        self.total_steps = 0

    def _build_channel_map(self):
        """Map named receptor channels to observation indices."""
        idx = self.idx
        channels = {}

        # Sensory
        for name in ['pain', 'endorphin', 'temperature', 'chemical', 'pressure', 'fatigue']:
            if name in idx and isinstance(idx[name], tuple):
                channels[name] = idx[name]

        # Internal
        channels['energy'] = (idx['energy'], idx['energy'] + 1)

        pe_s = idx.get('pe_start')
        if pe_s is not None:
            channels['prediction_error'] = (pe_s, pe_s + 6)

        mm_s = idx.get('mm_start')
        if mm_s is not None:
            channels['mm_certainty'] = (mm_s, mm_s + 1)
            channels['mm_learning_progress'] = (mm_s + 1, mm_s + 2)
            channels['mm_controllability'] = (mm_s + 2, mm_s + 3)
            channels['mm_planning'] = (mm_s + 3, mm_s + 4)

        pat_s = idx.get('pattern_start')
        if pat_s is not None:
            channels['pattern_available'] = (pat_s, pat_s + 1)
            channels['pattern_certainty'] = (pat_s + 1, pat_s + 2)

        # Social
        npc_s = idx.get('npc_start')
        if npc_s is not None:
            channels['npc_proximity'] = (npc_s, npc_s + 1)
            channels['npc_speed'] = (npc_s + 3, npc_s + 4)
            channels['npc_erraticism'] = (npc_s + 6, npc_s + 7)
            channels['npc_emission'] = (npc_s + 8, npc_s + 12)

        # Higher cognition
        opt_s = idx.get('opt_start')
        if opt_s is not None:
            channels['optimism'] = (opt_s, opt_s + 1)
            channels['goal_persistence'] = (opt_s + 1, opt_s + 2)

        conf_s = idx.get('conflict')
        if conf_s is not None:
            channels['conflict'] = (conf_s, conf_s + 1)

        # Thinking
        think_s = idx.get('thinking_start')
        if think_s is not None:
            channels['think_best_value'] = (think_s, think_s + 1)
            channels['think_visit_entropy'] = (think_s + 1, think_s + 2)
            channels['think_value_convergence'] = (think_s + 2, think_s + 3)
            channels['think_path_divergence'] = (think_s + 3, think_s + 4)
            channels['think_underexplored'] = (think_s + 4, think_s + 5)
            channels['think_depth_reached'] = (think_s + 5, think_s + 6)

        return channels

    def _get_activation_vector(self, obs):
        """Extract binary activation vector: which channels are above threshold."""
        active = np.zeros(self.n_channels, dtype=int)
        for i, (name, (start, end)) in enumerate(self.channels.items()):
            if end <= len(obs):
                val = float(np.mean(np.abs(obs[start:end])))
                if val > self.threshold:
                    active[i] = 1
        return active

    def record(self, obs, step=None):
        """Record one timestep's activation pattern."""
        active = self._get_activation_vector(obs)
        self.activation_history.append(active)
        self.total_steps += 1

        # Update activation counts
        self.activation_counts += active

        # Update co-activation matrix
        active_indices = np.where(active)[0]
        for i in active_indices:
            for j in active_indices:
                self.coactivation_counts[i, j] += 1

        # Update sequence counts (from previous step)
        if len(self.activation_history) >= 2:
            prev = self.activation_history[-2]
            prev_indices = np.where(prev)[0]
            for i in prev_indices:
                for j in active_indices:
                    self.sequence_counts[i, j] += 1

    def get_stats(self):
        """Compute co-activation statistics."""
        if self.total_steps < 10:
            return None

        # Activation rates
        rates = self.activation_counts / self.total_steps

        # Co-activation matrix normalized by total steps
        coact_rate = self.coactivation_counts / self.total_steps

        # Conditional probability: P(B|A) = P(A&B) / P(A)
        conditional = np.zeros_like(coact_rate, dtype=float)
        for i in range(self.n_channels):
            if self.activation_counts[i] > 0:
                conditional[i, :] = self.coactivation_counts[i, :] / self.activation_counts[i]

        # Lift: P(A&B) / (P(A) * P(B)) — >1 means co-activate more than chance
        lift = np.zeros_like(coact_rate, dtype=float)
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                expected = rates[i] * rates[j]
                if expected > 0.001:
                    lift[i, j] = coact_rate[i, j] / expected

        # Top co-activating pairs (by lift, excluding self-pairs)
        pairs = []
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                if lift[i, j] > 1.2 and coact_rate[i, j] > 0.05:
                    pairs.append({
                        'a': self.channel_names[i],
                        'b': self.channel_names[j],
                        'lift': round(float(lift[i, j]), 3),
                        'coact_rate': round(float(coact_rate[i, j]), 3),
                        'conditional_ab': round(float(conditional[i, j]), 3),
                        'conditional_ba': round(float(conditional[j, i]), 3),
                    })
        pairs.sort(key=lambda p: -p['lift'])

        # Top exclusion pairs (lift < 0.5, both active >10% of time)
        exclusions = []
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                if lift[i, j] < 0.5 and rates[i] > 0.1 and rates[j] > 0.1:
                    exclusions.append({
                        'a': self.channel_names[i],
                        'b': self.channel_names[j],
                        'lift': round(float(lift[i, j]), 3),
                    })
        exclusions.sort(key=lambda p: p['lift'])

        # Top temporal sequences: P(B at t+1 | A at t)
        seq_conditional = np.zeros_like(self.sequence_counts, dtype=float)
        for i in range(self.n_channels):
            if self.activation_counts[i] > 0:
                seq_conditional[i, :] = self.sequence_counts[i, :] / self.activation_counts[i]

        sequences = []
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if i == j:
                    continue
                sc = float(seq_conditional[i, j])
                base = float(rates[j])
                if sc > base * 1.3 and sc > 0.1:
                    sequences.append({
                        'from': self.channel_names[i],
                        'to': self.channel_names[j],
                        'p_given': round(sc, 3),
                        'p_base': round(base, 3),
                        'lift': round(sc / max(base, 0.001), 3),
                    })
        sequences.sort(key=lambda s: -s['lift'])

        # Recurring patterns (most common activation sets)
        pattern_counts = Counter()
        for active in self.activation_history:
            key = tuple(np.where(active)[0])
            if len(key) >= 2:
                pattern_counts[key] += 1

        top_patterns = []
        for pattern, count in pattern_counts.most_common(10):
            if count >= self.total_steps * 0.02:
                names = [self.channel_names[i] for i in pattern]
                top_patterns.append({
                    'channels': names,
                    'count': count,
                    'frequency': round(count / self.total_steps, 3),
                })

        return {
            'total_steps': self.total_steps,
            'activation_rates': {self.channel_names[i]: round(float(rates[i]), 3)
                                  for i in range(self.n_channels) if rates[i] > 0.01},
            'top_coactivations': pairs[:15],
            'top_exclusions': exclusions[:10],
            'top_sequences': sequences[:15],
            'recurring_patterns': top_patterns,
            'n_unique_patterns': len(pattern_counts),
        }


def report_coactivation(log, idx=None, threshold=0.3):
    """Run co-activation analysis on a log."""
    logger = CoactivationLogger(idx, activation_threshold=threshold)

    for entry in log:
        logger.record(entry['obs_after'])

    stats = logger.get_stats()
    if stats is None:
        print("  Insufficient data for co-activation analysis.")
        return stats

    print(f"\n{'='*60}")
    print(f"RECEPTOR CO-ACTIVATION ANALYSIS")
    print(f"{'='*60}")
    print(f"  Steps: {stats['total_steps']}")
    print(f"  Unique activation patterns: {stats['n_unique_patterns']}")

    print(f"\n  Activation rates:")
    for name, rate in sorted(stats['activation_rates'].items(), key=lambda x: -x[1]):
        bar = '#' * int(rate * 40)
        print(f"    {name:<28} {rate:.3f} {bar}")

    if stats['top_coactivations']:
        print(f"\n  Top co-activations (lift > 1.2):")
        for p in stats['top_coactivations'][:10]:
            print(f"    {p['a']} + {p['b']}: lift={p['lift']:.2f} "
                  f"(P={p['coact_rate']:.3f})")

    if stats['top_exclusions']:
        print(f"\n  Top exclusions (lift < 0.5, both >10%):")
        for p in stats['top_exclusions'][:5]:
            print(f"    {p['a']} vs {p['b']}: lift={p['lift']:.2f}")

    if stats['top_sequences']:
        print(f"\n  Top temporal sequences (A at t -> B at t+1):")
        for s in stats['top_sequences'][:10]:
            print(f"    {s['from']} -> {s['to']}: "
                  f"P(B|A)={s['p_given']:.3f} vs base={s['p_base']:.3f} "
                  f"(lift={s['lift']:.2f})")

    if stats['recurring_patterns']:
        print(f"\n  Recurring thought patterns:")
        for p in stats['recurring_patterns']:
            print(f"    [{', '.join(p['channels'])}] "
                  f"x{p['count']} ({p['frequency']:.1%})")

    return stats


if __name__ == '__main__':
    import numpy as np
    from environment import Environment, Organism, NPC
    from mental_model import build_mental_model

    print("Building organism (6000 steps)...")
    rng = np.random.RandomState(42)
    all_log = []
    for ep in range(10):
        env = Environment(seed=rng.randint(0, 100000))
        org = Organism()
        org.reset(rng)
        npc = NPC()
        npc.reset(rng)
        for step in range(600):
            npc.step(env, step)
            a = org.compute_optimal_actions(env, step, npc=npc)
            r = rng.random()
            executed = rng.randint(0, 2, size=org.NUM_ACTIONS).astype(np.int32) if r < 0.07 else a
            obs, reward = org.step(executed, env, step, npc=npc)
        all_log.extend(org.experience_log)

    print(f"Log: {len(all_log)} entries")
    stats = report_coactivation(all_log)

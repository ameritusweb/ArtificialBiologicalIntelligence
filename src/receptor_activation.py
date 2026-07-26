"""Receptor Activation Manager.

Tracks which of the 163 receptor channels (73 live + 90 episode) are
active. Inactive channels are masked to zero — the organism doesn't
pay metabolic cost for receptors it isn't using.

The activation mask evolves: receptors that fire consistently become
active, receptors that stop firing become inactive. The organism's
effective obs_dim is the base 177 + number of active receptor channels.

This connects Territory II (metabolic budget) to the architecture:
the organism evolves its own Umwelt resolution.
"""

import numpy as np

NUM_LIVE = 73
NUM_EPISODE = 90
NUM_TOTAL_RECEPTORS = NUM_LIVE + NUM_EPISODE


class ReceptorActivationManager:
    """Manages which receptor channels are active in the observation vector."""

    def __init__(self, activation_threshold=0.05, activation_window=100,
                 deactivation_window=500, cost_per_receptor=0.0005):
        self.activation_threshold = activation_threshold
        self.activation_window = activation_window
        self.deactivation_window = deactivation_window
        self.cost_per_receptor = cost_per_receptor

        self.active_mask = np.zeros(NUM_TOTAL_RECEPTORS, dtype=bool)
        self.running_mean = np.zeros(NUM_TOTAL_RECEPTORS)
        self.steps_active = np.zeros(NUM_TOTAL_RECEPTORS, dtype=int)
        self.steps_silent = np.zeros(NUM_TOTAL_RECEPTORS, dtype=int)
        self.total_steps = 0

    def update_live(self, live_values):
        """Update activation stats from per-step live receptor values."""
        alpha = 2.0 / (self.activation_window + 1)
        self.running_mean[:NUM_LIVE] = (
            (1 - alpha) * self.running_mean[:NUM_LIVE] + alpha * np.abs(live_values))
        self._update_masks(slice(0, NUM_LIVE))
        self.total_steps += 1

    def update_episode(self, episode_values):
        """Update activation stats from episode-level receptor values.
        Episode receptors activate after 3 episodes (not 100 steps)."""
        self.running_mean[NUM_LIVE:] = np.abs(episode_values)
        above = self.running_mean[NUM_LIVE:] > self.activation_threshold
        s = slice(NUM_LIVE, NUM_TOTAL_RECEPTORS)
        self.steps_active[s] = np.where(above, self.steps_active[s] + 30, 0)
        self.steps_silent[s] = np.where(~above, self.steps_silent[s] + 30, 0)
        newly_active = (self.steps_active[s] >= self.activation_window) & ~self.active_mask[s]
        newly_inactive = (self.steps_silent[s] >= self.deactivation_window) & self.active_mask[s]
        self.active_mask[s] = self.active_mask[s] | newly_active
        self.active_mask[s] = self.active_mask[s] & ~newly_inactive

    def _update_masks(self, s):
        above = self.running_mean[s] > self.activation_threshold
        self.steps_active[s] = np.where(above, self.steps_active[s] + 1, 0)
        self.steps_silent[s] = np.where(~above, self.steps_silent[s] + 1, 0)

        newly_active = (self.steps_active[s] >= self.activation_window) & ~self.active_mask[s]
        newly_inactive = (self.steps_silent[s] >= self.deactivation_window) & self.active_mask[s]

        self.active_mask[s] = self.active_mask[s] | newly_active
        self.active_mask[s] = self.active_mask[s] & ~newly_inactive

    def apply_mask(self, live_values, episode_values):
        """Zero inactive channels. Returns masked copies."""
        masked_live = live_values.copy()
        masked_live[~self.active_mask[:NUM_LIVE]] = 0.0
        masked_episode = episode_values.copy()
        masked_episode[~self.active_mask[NUM_LIVE:]] = 0.0
        return masked_live, masked_episode

    def get_metabolic_cost(self):
        """Cost proportional to number of active receptors."""
        return float(np.sum(self.active_mask)) * self.cost_per_receptor

    def get_active_count(self):
        return int(np.sum(self.active_mask))

    def get_active_live_count(self):
        return int(np.sum(self.active_mask[:NUM_LIVE]))

    def get_active_episode_count(self):
        return int(np.sum(self.active_mask[NUM_LIVE:]))

    def get_active_names(self, live_names, episode_names):
        """Return names of active receptors."""
        active = []
        for i in range(NUM_LIVE):
            if self.active_mask[i]:
                active.append(live_names[i])
        for i in range(NUM_EPISODE):
            if self.active_mask[NUM_LIVE + i]:
                active.append(episode_names[i])
        return active

    def get_stats(self):
        return {
            'active_total': self.get_active_count(),
            'active_live': self.get_active_live_count(),
            'active_episode': self.get_active_episode_count(),
            'metabolic_cost': round(self.get_metabolic_cost(), 4),
            'total_steps': self.total_steps,
        }

    def force_activate_all(self):
        """Activate all receptors (for testing or baseline comparison)."""
        self.active_mask[:] = True

    def reset(self):
        self.active_mask[:] = False
        self.running_mean[:] = 0.0
        self.steps_active[:] = 0
        self.steps_silent[:] = 0
        self.total_steps = 0

"""
Rhythm Receptor Validation Tool

Tests whether the rhythm receptor has emerged in an organism's mental model
by analyzing temporal delay distributions in learned mappings.

Receptor: rhythm (Repetition family)
Signature: Mental model learns temporal delay distributions that peak at
           recurrence interval T, not uniform distribution.

Environment: Sinusoidal field sources (already present in current sim)
Prediction: Should emerge by generation 50-90
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats


class RhythmDetector:
    """Detects emergence of rhythm receptor from mental model state."""

    def __init__(self, mental_model_engine, environment):
        """
        Args:
            mental_model_engine: MentalModelEngine instance with trained mappings
            environment: Environment instance with field sources
        """
        self.mm = mental_model_engine
        self.env = environment

    def extract_temporal_delays(self):
        """
        Extract learned temporal delay distributions from mental model.

        Returns:
            dict: {action_hash: [delays]} for actions with sufficient observations
        """
        delays_by_action = defaultdict(list)

        # Iterate through all mappings in the store
        for action_hash, entries in self.mm.store.mappings.items():
            for entry in entries:
                # We don't have explicit delay in current schema
                # This is a placeholder - need to add delay tracking to mental model
                # For now, extract from experience log if available
                pass

        # TODO: Mental model needs to track time delays explicitly
        # For now, we'll analyze experience log directly
        if hasattr(self.mm, 'experience_log'):
            return self._extract_from_experience_log()

        return delays_by_action

    def _extract_from_experience_log(self):
        """Extract delays from raw experience log (fallback method)."""
        # Group experiences by action
        action_sequences = defaultdict(list)

        for i, entry in enumerate(self.mm.experience_log):
            action_hash = hash(tuple(entry['action']))
            action_sequences[action_hash].append((i, entry['time_step']))

        # Calculate time delays between same action
        delays_by_action = defaultdict(list)
        for action_hash, sequence in action_sequences.items():
            for i in range(1, len(sequence)):
                delay = sequence[i][1] - sequence[i-1][1]
                delays_by_action[action_hash].append(delay)

        return delays_by_action

    def detect_periodicity(self, delays, expected_period=None):
        """
        Test whether delay distribution shows periodic structure.

        Args:
            delays: List of temporal delays
            expected_period: Known period from environment (for validation)

        Returns:
            dict with:
                - has_rhythm: bool
                - peak_period: float
                - clustering_score: float (0-1)
                - p_value: float
        """
        if len(delays) < 10:
            return {
                'has_rhythm': False,
                'reason': 'insufficient_data',
                'n_samples': len(delays)
            }

        delays = np.array(delays)

        # Test 1: Histogram clustering
        # Strong rhythm should show peak at period T
        hist, bins = np.histogram(delays, bins=30)
        peak_idx = np.argmax(hist)
        peak_period = (bins[peak_idx] + bins[peak_idx + 1]) / 2

        # Calculate what fraction of mass is near peak
        peak_window = peak_period * 0.2  # ±20% window
        in_window = np.sum((delays >= peak_period - peak_window) &
                          (delays <= peak_period + peak_window))
        clustering_score = in_window / len(delays)

        # Test 2: Statistical test vs uniform
        # Chi-square test: is distribution different from uniform?
        expected_counts = np.ones_like(hist) * len(delays) / len(hist)
        chi2, p_value = stats.chisquare(hist, expected_counts)

        # Test 3: If expected period known, check alignment
        period_match_score = None
        if expected_period is not None:
            period_error = abs(peak_period - expected_period) / expected_period
            period_match_score = max(0, 1.0 - period_error)

        # Emergence criteria from receptor spec:
        # "70-90% of mass within ±20% of true interval"
        has_rhythm = (clustering_score >= 0.7 and p_value < 0.01)

        return {
            'has_rhythm': has_rhythm,
            'peak_period': float(peak_period),
            'clustering_score': float(clustering_score),
            'p_value': float(p_value),
            'period_match_score': period_match_score,
            'n_samples': len(delays),
            'threshold_met': clustering_score >= 0.7,
            'significant': p_value < 0.01
        }

    def get_environment_periods(self):
        """Extract true periods from environment field sources."""
        periods = []
        for source in self.env.pain_sources + self.env.endorphin_sources:
            # Sinusoidal sources have period = 2π/ω
            if hasattr(source, 'omega_x') and source.omega_x > 0:
                period_x = 2 * np.pi / source.omega_x
                periods.append(period_x)
            if hasattr(source, 'omega_y') and source.omega_y > 0:
                period_y = 2 * np.pi / source.omega_y
                periods.append(period_y)
        return periods

    def validate_rhythm_receptor(self, min_actions=5):
        """
        Full validation of rhythm receptor emergence.

        Args:
            min_actions: Minimum number of action types to test

        Returns:
            dict with validation results
        """
        print("=== Rhythm Receptor Validation ===\n")

        # Get environment periods
        env_periods = self.get_environment_periods()
        if env_periods:
            print(f"Environment periods: {[f'{p:.1f}' for p in env_periods]}")
            avg_period = np.mean(env_periods)
        else:
            print("Warning: No periodic structure detected in environment")
            avg_period = None

        # Extract delays
        delays_by_action = self.extract_temporal_delays()

        if not delays_by_action:
            return {
                'receptor_present': False,
                'reason': 'no_temporal_data',
                'recommendation': 'Add time delay tracking to mental model'
            }

        # Test each action type
        results = []
        for action_hash, delays in delays_by_action.items():
            if len(delays) < 10:
                continue

            result = self.detect_periodicity(delays, expected_period=avg_period)
            result['action_hash'] = action_hash
            results.append(result)

        if len(results) < min_actions:
            return {
                'receptor_present': False,
                'reason': 'insufficient_action_coverage',
                'n_actions_tested': len(results),
                'required': min_actions
            }

        # Summary statistics
        rhythm_detected = [r['has_rhythm'] for r in results]
        clustering_scores = [r['clustering_score'] for r in results]

        proportion_rhythmic = np.mean(rhythm_detected)
        avg_clustering = np.mean(clustering_scores)

        # Receptor emergence criteria:
        # At least 50% of action types show rhythmic structure
        receptor_present = proportion_rhythmic >= 0.5

        print(f"\nActions tested: {len(results)}")
        print(f"Rhythmic actions: {sum(rhythm_detected)} ({proportion_rhythmic*100:.1f}%)")
        print(f"Average clustering: {avg_clustering:.3f}")
        print(f"\nReceptor present: {receptor_present}")

        return {
            'receptor_present': receptor_present,
            'proportion_rhythmic': float(proportion_rhythmic),
            'avg_clustering_score': float(avg_clustering),
            'n_actions_tested': len(results),
            'per_action_results': results,
            'environment_periods': env_periods
        }

    def plot_delay_distribution(self, action_hash=None, save_path=None):
        """
        Visualize temporal delay distribution for rhythm detection.

        Args:
            action_hash: Specific action to plot, or None for aggregate
            save_path: Where to save plot, or None to display
        """
        delays_by_action = self.extract_temporal_delays()

        if action_hash is not None:
            if action_hash not in delays_by_action:
                print(f"Action {action_hash} not found")
                return
            delays_to_plot = {action_hash: delays_by_action[action_hash]}
        else:
            # Plot top 4 most common actions
            sorted_actions = sorted(delays_by_action.items(),
                                  key=lambda x: len(x[1]),
                                  reverse=True)[:4]
            delays_to_plot = dict(sorted_actions)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        env_periods = self.get_environment_periods()
        avg_period = np.mean(env_periods) if env_periods else None

        for idx, (ah, delays) in enumerate(delays_to_plot.items()):
            if idx >= 4:
                break

            ax = axes[idx]
            delays_arr = np.array(delays)

            # Histogram
            ax.hist(delays_arr, bins=30, alpha=0.7, edgecolor='black')

            # Mark expected period if known
            if avg_period is not None:
                ax.axvline(avg_period, color='red', linestyle='--',
                          label=f'Env period: {avg_period:.1f}')

            # Detection result
            result = self.detect_periodicity(delays, expected_period=avg_period)

            ax.set_xlabel('Time delay (steps)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Action {ah}\n'
                        f'Rhythm: {result["has_rhythm"]}, '
                        f'Clustering: {result["clustering_score"]:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(delays_to_plot), 4):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()


def main():
    """Example usage."""
    # This would be called with actual mental model and environment instances
    # from a training run

    print("""
    Rhythm Receptor Validator

    Usage:
        from mental_model import build_mental_model
        from environment import Environment
        from validation.rhythm_detector import RhythmDetector

        # After training
        env = Environment()
        mm_engine = build_mental_model(global_log)

        detector = RhythmDetector(mm_engine, env)
        results = detector.validate_rhythm_receptor()

        if results['receptor_present']:
            print("Rhythm receptor has emerged!")
            detector.plot_delay_distribution()
        else:
            print(f"Not yet emerged: {results['reason']}")

    Emergence Criteria (from receptor spec):
        - 70-90% of temporal delay mass within ±20% of true interval
        - Chi-square p < 0.01 vs uniform distribution
        - At least 50% of action types show rhythmic structure

    Expected Timeline:
        - Should emerge by generation 50-90 in environments with
          sinusoidal field sources (current sim has this)
    """)


if __name__ == '__main__':
    main()

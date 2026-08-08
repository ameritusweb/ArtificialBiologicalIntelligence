"""
Manifestation Environment Generator

Each receptor in the genome project needs a specific environment tier to create
the selection pressure for it to emerge. The tier IS the manifestation —
TieredEnvironment at the correct tier provides the real environmental features
(NPCs, movable objects, hidden variables, causal triggers, etc.) that the
organism must navigate to survive.

The runner (deep_time_overnight.run_generation_rich) adds PhysicsWorld and
CombinedT7T8Environment on top of the TieredEnvironment, giving the full
environment stack: fields + rigid body physics + abstract causal problems.

Usage:
    from genome_project.manifester import ManifestationRegistry

    registry = ManifestationRegistry()
    envs = registry.get_environments('tool_use')
    for env_config in envs:
        env = env_config.build(seed=42)
        # env is TieredEnvironment(tier=3) with movable objects, gated rewards, etc.
"""

import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import Environment, FieldSource, ResponsiveObject, NPC
from environment_tiers import TieredEnvironment

MANIFESTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manifestations')


@dataclass
class ManifestationEnvConfig:
    """Configuration for a manifestation environment.

    tier: which TieredEnvironment tier (0-7). The tier provides the real
    environmental features the receptor needs to emerge.
    """
    receptor_id: str
    manifestation_id: str
    description: str = ''
    is_anti: bool = False
    tier: int = 0
    steps_per_episode: int = 500
    num_episodes: int = 5
    env_modifiers: list = field(default_factory=list)

    def build(self, seed=None):
        env = TieredEnvironment(seed=seed, tier=self.tier)
        for modifier in self.env_modifiers:
            modifier(env)
        return env


RECIPES = {}


def recipe(receptor_id, manifestation_id, is_anti=False):
    def decorator(fn):
        RECIPES[(receptor_id, manifestation_id, is_anti)] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Receptor → Tier mapping
#
# Each receptor is assigned to the minimum tier that provides the
# environmental features it needs to emerge under evolutionary selection.
#
# Tier 0: Base fields (pain/endorphin/temperature/chemical/pressure),
#          responsive objects, single NPC
# Tier 1: + pulsing sources, causal triggers with delays, anticipatory rewards
# Tier 2: + 4 profiled NPCs (cooperative/competitive/erratic/deceptive),
#          predator sweeps, resource competition
# Tier 3: + 3 movable objects, gated rewards (pain ring around endorphin),
#          compound object pairs
# Tier 4: + deterministic + stochastic hidden state modulating 4 modalities,
#          cross-modal sources
# Tier 5: + alliance switching, competitive NPC tracking, hidden endorphin
#          revealed by cooperative NPCs, deception energy cost
# Tier 6: + 3-step object puzzle, delayed rewards, persistent object state
# Tier 7: + rule rotation, impulse traps (immediate reward → delayed pain),
#          strategy staleness penalty
# ---------------------------------------------------------------------------

RECEPTOR_TIER_MAP = {
    **{r: 0 for r in [
        'basic_sensorimotor_loop', 'change_detection', 'spatial_association',
        'temporal_association', 'coincidence_detection', 'stress_detection',
        'arousal_regulation', 'staged_processing', 'stage_prediction',
        'self_model', 'controllability', 'movement_onset', 'velocity', 'effort',
        'coordination', 'body_boundary', 'postural_state', 'postural_change',
        'joint_limit', 'movement_anticipation', 'static_repetition',
        'perceptual_similarity', 'categorical_perception', 'quantity_detection',
        'boundary_detection', 'compression_receptor', 'categorical_compression',
        'pattern_recognition', 'compression_gain', 'comparative_observation',
        'relational_observation', 'absence_observation', 'functional_similarity',
        'attention_control', 'selective_observation', 'concept_formation',
        'concept_grounding', 'prediction_accuracy', 'prediction_branching',
        'pipeline_detection', 'cross_pipeline_prediction', 'multiple_receptor_types',
        'pattern_based_resolution', 'optimism', 'rule_extraction',
        'bias_as_compression', 'structural_similarity', 'structural_invariance',
        'structural_invariance_math', 'prototype_formation', 'functional_organization',
        'hierarchical_structure_detection', 'completion', 'constraint_shape',
        'shaped_absence', 'missing_piece_located', 'part_whole_detection',
        'ratio_detection', 'value_hierarchy', 'analogy_receptor',
        'analogy_receptor_ep', 'analogy_ep', 'analogical_similarity',
        'hierarchical_abstraction', 'satisfaction', 'processing_speed',
        'adaptive_depth', 'contradiction', 'contradiction_ep', 'it_follows',
        'formal_composition', 'proof_structure', 'necessity_detection',
        'rule_generalization', 'abstract_association', 'concept_activation',
        'self_soothing', 'relational_analogy',
    ]},
    **{r: 1 for r in [
        'rhythm', 'rhythm_entrainment', 'causal_inference', 'causal_association',
        'causal_chains', 'precedence_detection', 'dynamic_repetition',
        'rhythmic_pattern', 'causal_rhythm', 'nested_rhythm', 'long_range_causation',
        'transitivity', 'chunking', 'ritual_formation', 'impulse_override',
        'environmental_change_detection', 'environmental_trend_detection',
        'curiosity', 'curiosity_live', 'prediction_accuracy_ep',
        'exception_detection', 'probabilistic_causation', 'causal_graph_reasoning',
        'counterfactual_reasoning', 'counterfactual_salience', 'regret',
        'rule_composition', 'rule_revision', 'optimization', 'optimization_ep',
        'long_term_planning', 'meta_planning', 'theory_formation',
        'exhaustive_search',
    ]},
    **{r: 2 for r in [
        'other_detection', 'behavioral_prediction', 'empathy',
        'self_model_applied_to_others', 'mimicry', 'perspective_taking',
        'social_learning', 'social_coregulation', 'cultural_transmission',
        'instruction_detection', 'moral_reasoning', 'receptor_propagation',
        'spatial_reasoning', 'rarity_detection', 'statistical_anomaly',
        'significance_detection', 'growth_tracking',
    ]},
    **{r: 3 for r in [
        'contact', 'contact_response', 'push_affordance', 'grip_affordance',
        'grip_affordance_live', 'grip_state_proprio', 'tool_use',
        'lever_affordance', 'composite_affordance', 'affordance_transfer',
        'resistance', 'haptic_recognition', 'environmental_manipulation',
        'environmental_modification', 'modification_attribution',
        'niche_construction', 'capability_change_detection',
        'deliberate_complexification', 'developmental_environment_engineering',
        'structural_preservation', 'proprioception_ep', 'muscle_memory',
        'replay', 'greatest_positive_increase', 'shortcut_activation',
    ]},
    **{r: 4 for r in [
        'common_cause_detection', 'hidden_confounder_detection',
        'cross_modal_association', 'cross_modal_observation',
        'context_conditioned_arbitration', 'multiple_hypotheses',
        'conflation', 'conflation_ep', 'fundamental_distinction',
        'fundamental_distinction_ep', 'relative_truth', 'absolute_truth',
        'doubt_detection', 'belief_detection',
    ]},
    **{r: 5 for r in [
        'deception_detection', 'trust', 'trust_ep', 'ownership_boundary',
        'ownership_boundary_ep', 'theory_of_mind', 'nested_theory_of_mind',
        'belief_attribution', 'intention_recognition', 'compliance_detection',
        'instruction_source_discrimination', 'contextual_signal_interpretation',
        'mental_model_confidence', 'emotional_intelligence',
    ]},
    **{r: 6 for r in [
        'intervention_planning', 'distributed_agency', 'naming', 'naming_ep',
        'self_talk', 'referential_grounding', 'referential_grounding_ep',
        'language_grounding', 'simplified_shared_signals', 'semantic_relation',
        'conjunction', 'quantifier', 'quantifier_ep', 'translation',
        'translation_ep', 'executability', 'executability_ep',
        'identity_continuity', 'identity_preservation', 'knowledge_preservation',
        'response_recognition', 'developmental_trajectory',
    ]},
    **{r: 7 for r in [
        'metacognition', 'meta_observation', 'thought_type_detection',
        'topology_awareness', 'epistemic_strategy', 'self_regulation',
        'frustration', 'futility', 'pipeline_optimization',
        'prediction_architecture_awareness', 'response_loop_detection',
        'organizational_mirror', 'metamorphic_planning', 'forgetting',
        'remembering', 'system_detection', 'relational_structure_detection',
        'org_boundary_detection', 'conflict', 'agency_salience',
    ]},
}


# ---------------------------------------------------------------------------
# Register recipes from the tier map
# ---------------------------------------------------------------------------

def _register_recipes():
    for receptor_id, tier in RECEPTOR_TIER_MAP.items():
        mid = f'tier{tier}'

        def _make(rid=receptor_id, t=tier):
            def _fn():
                return ManifestationEnvConfig(
                    receptor_id=rid, manifestation_id=f'tier{t}', tier=t,
                    steps_per_episode=500)
            return _fn

        RECIPES[(receptor_id, mid, False)] = _make()

    # Anti-manifestations: tier 0 for receptors that need higher tiers
    for receptor_id, tier in RECEPTOR_TIER_MAP.items():
        if tier >= 2:
            def _make_anti(rid=receptor_id):
                def _fn():
                    return ManifestationEnvConfig(
                        receptor_id=rid, manifestation_id='tier0_baseline',
                        tier=0, is_anti=True, steps_per_episode=500)
                return _fn
            RECIPES[(receptor_id, 'tier0_baseline', True)] = _make_anti()


_register_recipes()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ManifestationRegistry:

    def __init__(self, manifestation_dir=None):
        self.manifestation_dir = manifestation_dir or MANIFESTATION_DIR
        self._cache = {}

    def load_yaml(self, receptor_id):
        if receptor_id in self._cache:
            return self._cache[receptor_id]
        path = os.path.join(self.manifestation_dir, f'{receptor_id}.yaml')
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError:
            self._cache[receptor_id] = None
            return None
        self._cache[receptor_id] = data
        return data

    def get_environments(self, receptor_id):
        configs = []
        for (rid, mid, is_anti), recipe_fn in RECIPES.items():
            if rid == receptor_id and not is_anti:
                configs.append(recipe_fn())
        return configs

    def get_anti_environments(self, receptor_id):
        configs = []
        for (rid, mid, is_anti), recipe_fn in RECIPES.items():
            if rid == receptor_id and is_anti:
                configs.append(recipe_fn())
        return configs

    def get_all_environments(self, receptor_id):
        return self.get_environments(receptor_id) + self.get_anti_environments(receptor_id)

    def list_receptors_with_recipes(self):
        return sorted(set(rid for rid, mid, is_anti in RECIPES))

    def list_all_manifestation_ids(self):
        if not os.path.exists(self.manifestation_dir):
            return []
        return sorted(f.replace('.yaml', '') for f in os.listdir(self.manifestation_dir)
                      if f.endswith('.yaml'))

    def coverage_report(self):
        all_yamls = set(self.list_all_manifestation_ids())
        with_recipes = set(self.list_receptors_with_recipes())
        covered = all_yamls & with_recipes
        uncovered = all_yamls - with_recipes
        return {
            'total_yamls': len(all_yamls),
            'with_recipes': len(covered),
            'without_recipes': len(uncovered),
            'coverage_pct': round(100 * len(covered) / max(1, len(all_yamls)), 1),
            'covered': sorted(covered),
            'uncovered': sorted(uncovered),
        }

    def get_yaml_info(self, receptor_id):
        data = self.load_yaml(receptor_id)
        if data is None:
            return None
        return {
            'receptor_id': data.get('receptor_id', receptor_id),
            'family': data.get('family', ''),
            'tier': data.get('tier', ''),
            'stage': data.get('stage', 0),
            'conceptual_space': data.get('conceptual_space', ''),
            'manifestations': [
                {'id': m['id'], 'description': m.get('description', '')}
                for m in data.get('manifestations', [])
            ],
            'anti_manifestations': [
                {'id': m['id'], 'description': m.get('description', '')}
                for m in data.get('anti_manifestations', [])
            ],
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_environment(receptor_id, seed=42):
    registry = ManifestationRegistry()
    envs = registry.get_environments(receptor_id)
    if not envs:
        raise ValueError(f'No recipes found for receptor: {receptor_id}')
    return envs[0].build(seed=seed)


if __name__ == '__main__':
    registry = ManifestationRegistry()
    report = registry.coverage_report()
    print(f"Manifestation Environment Coverage")
    print(f"===================================")
    print(f"Total manifestation YAMLs: {report['total_yamls']}")
    print(f"Receptors with recipes:    {report['with_recipes']}")
    print(f"Coverage:                  {report['coverage_pct']}%")
    print()

    from collections import Counter
    tier_counts = Counter()
    for r in report['covered']:
        tier_counts[RECEPTOR_TIER_MAP.get(r, -1)] += 1
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]} receptors")

    print()
    for t in range(8):
        rid = next((r for r in report['covered'] if RECEPTOR_TIER_MAP.get(r) == t), None)
        if rid:
            env = build_environment(rid, seed=42)
            print(f"  Tier {t} ({rid}): {type(env).__name__}, "
                  f"npcs={len(getattr(env, 'npcs', []))}, "
                  f"objects={len(getattr(env, 'movable_objects', []))}, "
                  f"hidden_var={getattr(env, 'hidden_variable', None) is not None}")

    if report['uncovered']:
        print(f"\nUncovered ({len(report['uncovered'])}):")
        for r in report['uncovered']:
            print(f"  {r}")

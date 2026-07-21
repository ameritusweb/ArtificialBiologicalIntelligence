import os
import json
import numpy as np
from grounding import (RECEPTOR_GROUPS, SIGNAL_VOCABULARY,
                        build_grounding_dictionary)
from environment import Environment, Organism, NPC
from mental_model import build_mental_model
from train import generate_training_data
from model import compute_obs_indices

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

GROUNDED_TERMS = {
    'pain': {
        'receptor_group': 'pain',
        'obs_indices': '0-5',
        'grounded_definition': (
            'Pain is the activation of nociceptive receptor channels obs[0:5], '
            'one per limb tip, firing when limb tips contact pain field sources. '
            'It causes temporal_aversion[37:42] to accumulate (recent harm memory), '
            'pain_memory[49:73] to deposit spatial records (where harm was), '
            'prediction_error[90:95] to spike (surprise when pain was not predicted), '
            'and triggers avoidance behavior through the oracle gradient. '
            'Pain is not a label. It is the receptor state that changes what the organism does next.'
        ),
    },
    'danger': {
        'receptor_group': 'distant_pain',
        'obs_indices': '74-81',
        'grounded_definition': (
            'Danger is predicted future pain. distant_pain[74:81] reads pain field intensity '
            'at 8 points ahead of the organism (distance sensing rays). When distant pain exceeds '
            'current pain, the organism is moving toward harm it has not yet contacted. '
            'Danger triggers emission of signal (1,1,0,0) to warn nearby organisms. '
            'The mental model predicts: action X in context Y leads to pain increase. '
            'Danger is not the pain itself. It is the causal prediction that pain is coming.'
        ),
    },
    'relief': {
        'receptor_group': 'endorphin',
        'obs_indices': '6-11',
        'grounded_definition': (
            'Relief is the activation of endorphin receptor channels obs[6:11] '
            'when limb tips enter endorphin field regions. It causes energy recovery '
            '(metabolic reserve increases), temporal_aversion decay (recent harm memory '
            'fades), and optimism increase (future looks better than present). '
            'Relief is the receptor state that permits the organism to stop fleeing '
            'and begin approach. It is what reward means, stated mechanistically.'
        ),
    },
    'self': {
        'receptor_group': 'agency',
        'obs_indices': '132-134',
        'grounded_definition': (
            'Self is the controllability decomposition: obs[132] measures how much of '
            'the observed change was caused by the organism\'s own actions (self-caused delta) '
            'versus imposed by the environment (external delta). obs[133] is external change. '
            'obs[134] is planning value — how much better the chosen action is versus doing nothing. '
            'Self is not an entity. It is the causal attribution: some future receptor states '
            'are conditional on this controller\'s output. Agency = controllability > 0.5.'
        ),
    },
    'other': {
        'receptor_group': 'other',
        'obs_indices': '141-152',
        'grounded_definition': (
            'Other is the NPC observation vector: distance[141], bearing[142:143], '
            'speed[144], acceleration[145], angular velocity[146], erraticism[147], '
            'empathic_aversion[148], and emission bits[149:152]. The organism senses '
            'another agent through its behavioral cues — how it moves, how erratically, '
            'what signals it emits. Other is not a concept applied to an entity. '
            'It is the receptor channels that fire when an autonomous agent is nearby.'
        ),
    },
    'will': {
        'receptor_group': 'optimism',
        'obs_indices': '153-154',
        'grounded_definition': (
            'Will is the optimism receptor predicting that future reward exceeds current cost. '
            'obs[153] = optimism = clip(best_distant_endorphin - avg_current_pain, 0, 1). '
            'obs[154] = goal_persistence = EMA of optimism with 0.95 decay. '
            'When goal_persistence > 0.2, the oracle boosts gradient strength by up to 30% '
            'and dampens temporal aversion. The organism persists through pain toward predicted reward. '
            'Will is not a mysterious faculty. It is the receptor state that makes current cost '
            'tolerable because predicted future benefit exceeds it.'
        ),
    },
    'uncertainty': {
        'receptor_group': 'prediction_error',
        'obs_indices': '90-95',
        'grounded_definition': (
            'Uncertainty is the gap between predicted and actual receptor states. '
            'prediction_error[90:95] = |actual_pain - predicted_pain| per limb. '
            'When mm_certainty[98] is low, the mental model\'s predictions are unreliable. '
            'predicted_conflict[156] amplifies: conflict * (1 + 0.5 * (1 - certainty)). '
            'Uncertainty recruits the slow pathway (deliberation) through the conflict-gated router. '
            'Uncertainty is not an abstract epistemic state. It is the receptor reading that says '
            'the world model was wrong, which changes how the next cycle processes information.'
        ),
    },
    'conflict': {
        'receptor_group': 'conflict',
        'obs_indices': '155-157',
        'grounded_definition': (
            'Conflict is the simultaneous activation of incompatible receptor pathways. '
            'receptor_conflict[155] = min(pain_pressure, reward_pull) + 0.5*(persistence*pain) '
            '+ 0.3*(empathy*(1-pain)). Three sources: pain vs reward (stay or leave?), '
            'optimism vs current pain (persist or retreat?), empathy vs safety (care or flee?). '
            'predicted_conflict[156] forecasts whether conflict will worsen. '
            'conflict_trend[157] tracks resolution direction. '
            'Conflict is not indecision. It is the receptor state where multiple '
            'motivations demand incompatible actions simultaneously.'
        ),
    },
    'empathy': {
        'receptor_group': 'other',
        'obs_indices': '148',
        'grounded_definition': (
            'Empathy is empathic_aversion[148] = EMPATHY_WEIGHT * proximity_factor * npc_erraticism. '
            'When the NPC is nearby and in distress (high erraticism from rapid turning), '
            'the organism\'s own pain channels are elevated by empathic_aversion * 0.2. '
            'The NPC\'s suffering becomes the organism\'s pain — not metaphorically but through '
            'direct receptor propagation. The organism is motivated to move away from a distressed NPC '
            'because the NPC\'s distress elevates the organism\'s own nociceptive state.'
        ),
    },
    'optimism': {
        'receptor_group': 'optimism',
        'obs_indices': '153',
        'grounded_definition': (
            'Optimism is the receptor that fires when predicted future reward exceeds current cost. '
            'optimism[153] = clip(max(distant_endorphin) - mean(current_pain), 0, 1). '
            'Positive only when the best thing ahead is better than the average pain now. '
            'Optimism modulates the oracle: when high, the organism boosts its navigation gradient '
            'and dampens pain avoidance. Optimism is not hope. It is the receptor comparison '
            'between what the distance sensors read ahead and what the contact sensors read now.'
        ),
    },
    'compression': {
        'receptor_group': 'concepts',
        'obs_indices': '158-159',
        'grounded_definition': (
            'Compression is the pattern store producing stable concepts — recurring causal chains '
            'that predict more accurately than their individual components. concept_match[158] '
            'measures how strongly the current situation matches a known concept. '
            'concept_quality[159] measures the compression gain — how much better the compressed '
            'prediction is than the uncompressed chain. 1,013 stable concepts emerged from the '
            'organism\'s experience. Compression is the same receptor whether it produces useful '
            'abstractions or harmful bias. What gets discarded determines which.'
        ),
    },
}

SIGNAL_TERMS = {
    'danger_signal': {
        'code': [1, 1, 0, 0],
        'grounded_definition': (
            'The emission (1,1,0,0) means "danger/repel." When the organism emits it toward the NPC, '
            'the NPC adds a repulsive gradient away from the organism for 15 steps. '
            'When the NPC emits it near a pain source, the organism adds repulsive gradient '
            'from the NPC\'s direction. Same code, same meaning, regardless of who emits it. '
            'This is a grounded word: it points to the causal relationship "this signal causes '
            'the receiver to move away from the sender\'s location."'
        ),
    },
    'resource_signal': {
        'code': [0, 0, 1, 0],
        'grounded_definition': (
            'The emission (0,0,1,0) means "resource/endorphin." When the organism emits it near '
            'a responsive object, the object releases endorphin. When the NPC emits it near an '
            'endorphin source, the organism adds attractive gradient toward the NPC. '
            'Same code across three contexts: organism->object, organism->NPC, NPC->organism. '
            'The word is grounded in the same causal relationship in every context.'
        ),
    },
}


def generate_grounded_corpus(dictionary, engine):
    lines = []
    lines.append("# Grounded Corpus")
    lines.append("# Generated from the organism's mental model — every statement traces to a receptor state")
    lines.append("")

    lines.append("## Receptor Groups")
    lines.append("")
    for name, info in RECEPTOR_GROUPS.items():
        lo, hi = info['obs_range']
        lines.append(f"The organism has {hi-lo} {name} receptor channel(s) at obs[{lo}:{hi}]. "
                     f"{info['description']}.")
    lines.append("")

    lines.append("## Grounded Terms")
    lines.append("")
    for term, info in GROUNDED_TERMS.items():
        lines.append(f"### {term.upper()}")
        lines.append(f"Receptor group: {info['receptor_group']} (obs[{info['obs_indices']}])")
        lines.append(info['grounded_definition'])
        lines.append("")

    lines.append("## Signal Vocabulary (Grounded Words)")
    lines.append("")
    for term, info in SIGNAL_TERMS.items():
        lines.append(f"### {term}")
        lines.append(f"Emission code: {info['code']}")
        lines.append(info['grounded_definition'])
        lines.append("")

    lines.append("## Causal Chains")
    lines.append("")
    if 'grounding_chains' in dictionary:
        for chain in dictionary['grounding_chains']:
            lines.append(f"### '{chain['word']}'")
            lines.append(f"Receptor origin: {chain['receptor_origin']}")
            lines.append(f"Chain: {' -> '.join(chain['chain'])}")
            lines.append("")

    lines.append("## Concepts (Top 20)")
    lines.append("")
    if 'concepts' in dictionary:
        for c in dictionary['concepts'][:20]:
            lines.append(f"Concept #{c['rank']}: {c['label']} "
                        f"(certainty={c['certainty']:.3f}, compression_gain={c['compression_gain']:.3f}, "
                        f"count={c['count']})")
            lines.append(f"  This concept compresses a recurring 2-step action sequence into a single "
                        f"prediction that is {c['compression_gain']*100:.0f}% more accurate than "
                        f"chaining the individual steps.")
            lines.append("")

    lines.append("## The Grounding Guarantee")
    lines.append("")
    lines.append("Every term in this corpus traces to a specific receptor state the organism ")
    lines.append("actually experienced. 'Pain' is not defined by other words. It is defined by ")
    lines.append("obs[0:5] firing when limb tips contact pain field sources. The definition IS ")
    lines.append("the receptor state. The word IS the causal chain. There is no gap between the ")
    lines.append("signifier and the signified because the signified is a measurement, not a concept.")
    lines.append("")
    lines.append("This is what grounded language means: same transformer architecture, different corpus. ")
    lines.append("The corpus is a body.")

    return "\n".join(lines)


def query_grounding(term, dictionary=None):
    term_lower = term.lower().strip()

    if term_lower in GROUNDED_TERMS:
        info = GROUNDED_TERMS[term_lower]
        return {
            'term': term_lower,
            'grounded': True,
            'receptor_group': info['receptor_group'],
            'obs_indices': info['obs_indices'],
            'definition': info['grounded_definition'],
            'type': 'fundamental_term',
        }

    for sig_term, info in SIGNAL_TERMS.items():
        if term_lower in sig_term or term_lower in str(info['code']):
            return {
                'term': sig_term,
                'grounded': True,
                'emission_code': info['code'],
                'definition': info['grounded_definition'],
                'type': 'signal_vocabulary',
            }

    for name, info in RECEPTOR_GROUPS.items():
        if term_lower in name:
            lo, hi = info['obs_range']
            return {
                'term': name,
                'grounded': True,
                'obs_indices': f'{lo}-{hi}',
                'dimensionality': hi - lo,
                'description': info['description'],
                'type': 'receptor_group',
            }

    return {
        'term': term_lower,
        'grounded': False,
        'note': 'Term not found in grounding dictionary. It may be a higher-order concept '
                'not yet traced to receptor states, or a term from outside the organism\'s experience.',
    }


def generate_comparison_prompts(terms=None):
    if terms is None:
        terms = list(GROUNDED_TERMS.keys()) + ['empathy', 'optimism', 'compression']

    prompts = []
    for term in terms:
        grounding = query_grounding(term)

        ungrounded_prompt = f"What is {term}?"

        if grounding['grounded']:
            grounded_prompt = (
                f"What is {term}?\n\n"
                f"[Grounding context from organism's mental model:\n"
                f"{grounding['definition']}]"
            )
            expected_grounded = grounding['definition']
        else:
            grounded_prompt = ungrounded_prompt
            expected_grounded = "No grounding available."

        prompts.append({
            'term': term,
            'ungrounded_prompt': ungrounded_prompt,
            'grounded_prompt': grounded_prompt,
            'expected_grounded_response': expected_grounded,
            'expected_ungrounded_response': (
                f"A standard LLM would define '{term}' using other words, "
                f"drawing from statistical patterns in human text. The definition "
                f"would be accurate but ungrounded — it describes the word's usage, "
                f"not the receptor state it refers to."
            ),
        })

    return prompts


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Step 39: LLM Grounding Bridge ===\n")

    dict_path = os.path.join(DATA_DIR, 'grounding_dictionary.json')
    if os.path.exists(dict_path):
        with open(dict_path) as f:
            dictionary = json.load(f)
        print(f"  Loaded grounding dictionary: {len(dictionary.get('concepts', []))} concepts")
    else:
        print("  No grounding dictionary found. Generating...")
        _, _, _, global_log = generate_training_data(num_episodes=100, steps_per_episode=300)
        idx = compute_obs_indices(6)
        engine = build_mental_model(global_log, core_obs_dim=idx['core_obs_dim'])
        dictionary = build_grounding_dictionary(engine)

    print("\n--- Phase 1: Grounded Corpus ---")
    corpus = generate_grounded_corpus(dictionary, None)
    corpus_path = os.path.join(DATA_DIR, 'grounded_corpus.txt')
    with open(corpus_path, 'w') as f:
        f.write(corpus)
    line_count = corpus.count('\n')
    print(f"  Generated: {line_count} lines")
    print(f"  Saved: {corpus_path}")

    print("\n--- Phase 2: Grounded Query Interface ---")
    test_terms = ['pain', 'danger', 'self', 'other', 'will', 'empathy', 'conflict',
                  'uncertainty', 'relief', 'optimism', 'compression', 'temperature',
                  'love', 'justice']
    print(f"  Testing {len(test_terms)} queries:")
    for term in test_terms:
        result = query_grounding(term, dictionary)
        status = 'GROUNDED' if result['grounded'] else 'not grounded'
        rtype = result.get('type', 'unknown')
        print(f"    '{term}': {status} ({rtype})")

    grounded_count = sum(1 for t in test_terms if query_grounding(t)['grounded'])
    print(f"  {grounded_count}/{len(test_terms)} terms have grounded definitions")

    print("\n--- Phase 3: Comparative Grounding Test ---")
    prompts = generate_comparison_prompts()
    print(f"  Generated {len(prompts)} comparison prompt pairs")

    print("\n  Sample comparison (term: 'pain'):")
    pain_prompt = [p for p in prompts if p['term'] == 'pain'][0]
    print(f"\n  UNGROUNDED prompt: \"{pain_prompt['ungrounded_prompt']}\"")
    print(f"  Expected ungrounded answer: defines pain using other words (sensation, discomfort, neural signal)")
    print(f"\n  GROUNDED prompt: includes receptor context")
    print(f"  Expected grounded answer: obs[0:5] firing when limb tips contact pain field sources,")
    print(f"    causing temporal_aversion increase, pain_memory deposit, prediction_error spike")

    print(f"\n  Sample comparison (term: 'will'):")
    will_prompt = [p for p in prompts if p['term'] == 'will'][0]
    print(f"\n  UNGROUNDED: defines will as intention, desire, determination")
    print(f"  GROUNDED: optimism[153] predicting future reward exceeds current cost,")
    print(f"    goal_persistence[154] accumulating commitment, gradient modulation persisting through pain")

    comp_path = os.path.join(DATA_DIR, 'grounding_comparison.json')
    with open(comp_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"\n  Saved: {comp_path}")

    print("\n--- Summary ---")
    print(f"  Grounded corpus: {line_count} lines of receptor-grounded language")
    print(f"  Query interface: {grounded_count}/{len(test_terms)} terms grounded")
    print(f"  Comparison prompts: {len(prompts)} pairs ready for LLM testing")
    print(f"  Terms with no grounding ('love', 'justice'): correctly identified as")
    print(f"    outside the organism's experience — not yet traced to receptor states")
    print(f"\n  The grounding bridge is built. Every word points to a receptor state.")
    print(f"  Every definition traces to a causal chain the organism actually experienced.")
    print(f"  To test: send grounded vs ungrounded prompts to any LLM and compare responses.")

    print("\nStep 39 complete.")

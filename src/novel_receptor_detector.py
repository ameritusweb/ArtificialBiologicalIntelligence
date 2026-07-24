"""Novel Receptor Detector.

Scans the mental model for behavioral capacities that don't map to any
of the genome's receptor entries. Looks for structure the organism
carved that the designer didn't anticipate.

Method:
1. Extract all high-quality mental model entries (high certainty, high count)
2. Cluster them by context embedding similarity
3. For each cluster, characterize its "signature" — what observation
   channels it responds to, what deltas it produces, what actions it uses
4. Compare each cluster signature against all known receptor test signatures
5. Clusters with no matching receptor test are novel receptor candidates

A novel receptor is: a coherent, high-certainty cluster of mental model
entries that represents a behavioral distinction the organism learned
to make, but that doesn't correspond to any receptor in the genome.
"""

import numpy as np
from collections import defaultdict
from model import compute_obs_indices
from mental_model import CORE_OBS_DIM


def extract_qualified_entries(engine, min_certainty=0.5, min_count=5):
    """Get all high-quality entries from the store."""
    entries = []
    for ah, entry_list in engine.store.mappings.items():
        el = entry_list if isinstance(entry_list, list) else [entry_list]
        for e in el:
            if e.certainty >= min_certainty and e.count >= min_count:
                if e.context_embedding is not None:
                    entries.append({
                        'embedding': e.context_embedding,
                        'delta': e.delta,
                        'certainty': e.certainty,
                        'count': e.count,
                        'reward': e.reward,
                        'action_hash': ah,
                    })
    return entries


def cluster_entries(entries, sim_threshold=0.7, max_clusters=50):
    """Simple greedy clustering by context embedding similarity."""
    if not entries:
        return []

    embeddings = np.array([e['embedding'] for e in entries])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms

    assigned = np.full(len(entries), -1, dtype=int)
    centroids = []
    cluster_members = []

    for i in range(len(entries)):
        if assigned[i] >= 0:
            continue

        best_cluster = -1
        best_sim = sim_threshold
        for c, centroid in enumerate(centroids):
            sim = float(normed[i] @ centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c

        if best_cluster >= 0:
            assigned[i] = best_cluster
            cluster_members[best_cluster].append(i)
            members = cluster_members[best_cluster]
            centroids[best_cluster] = np.mean(normed[members], axis=0)
            centroids[best_cluster] /= np.linalg.norm(centroids[best_cluster]) + 1e-8
        else:
            if len(centroids) >= max_clusters:
                continue
            new_id = len(centroids)
            assigned[i] = new_id
            centroids.append(normed[i].copy())
            cluster_members.append([i])

    clusters = []
    for c in range(len(centroids)):
        members = cluster_members[c]
        if len(members) < 3:
            continue
        cluster_entries_list = [entries[m] for m in members]
        clusters.append({
            'id': c,
            'centroid': centroids[c],
            'members': cluster_entries_list,
            'size': len(members),
        })

    return sorted(clusters, key=lambda c: -c['size'])


def characterize_cluster(cluster):
    """Compute a cluster's behavioral signature."""
    members = cluster['members']
    idx = compute_obs_indices()
    cdim = idx.get('core_obs_dim', CORE_OBS_DIM)

    deltas = np.array([m['delta'][:cdim] for m in members])
    embeddings = np.array([m['embedding'] for m in members])
    certs = np.array([m['certainty'] for m in members])
    rewards = np.array([m['reward'] for m in members])
    action_hashes = [m['action_hash'] for m in members]

    mean_delta = np.mean(deltas, axis=0)
    delta_consistency = 1.0 - float(np.mean(np.std(deltas, axis=0)))

    groups = {
        'pain': idx['pain'],
        'endorphin': idx['endorphin'],
        'temperature': idx['temperature'],
        'chemical': idx['chemical'],
        'pressure': idx['pressure'],
        'fatigue': idx['fatigue'],
    }

    dominant_input = None
    max_input = 0
    mean_emb = np.mean(embeddings, axis=0)
    for name, (s, e) in groups.items():
        if e <= len(mean_emb):
            val = float(np.mean(np.abs(mean_emb[s:e])))
            if val > max_input:
                max_input = val
                dominant_input = name

    dominant_output = None
    max_output = 0
    for name, (s, e) in groups.items():
        if e <= len(mean_delta):
            val = float(np.mean(np.abs(mean_delta[s:e])))
            if val > max_output:
                max_output = val
                dominant_output = name

    unique_actions = len(set(action_hashes))
    action_specificity = 1.0 - min(1.0, unique_actions / max(len(members), 1))

    return {
        'dominant_input': dominant_input,
        'dominant_output': dominant_output,
        'input_strength': round(max_input, 4),
        'output_strength': round(max_output, 4),
        'delta_consistency': round(float(np.clip(delta_consistency, 0, 1)), 4),
        'mean_certainty': round(float(np.mean(certs)), 4),
        'mean_reward': round(float(np.mean(rewards)), 4),
        'reward_std': round(float(np.std(rewards)), 4),
        'action_specificity': round(action_specificity, 4),
        'unique_actions': unique_actions,
        'mean_delta_magnitude': round(float(np.mean(np.abs(mean_delta))), 4),
        'cross_modal': dominant_input != dominant_output and dominant_input is not None and dominant_output is not None,
    }


KNOWN_SIGNATURES = {
    'pain_avoidance': {'input': 'pain', 'output': 'pain', 'direction': 'decrease'},
    'endorphin_seeking': {'input': 'endorphin', 'output': 'endorphin', 'direction': 'increase'},
    'temperature_regulation': {'input': 'temperature', 'output': 'temperature', 'direction': 'neutral'},
    'chemical_seeking': {'input': 'chemical', 'output': 'chemical', 'direction': 'increase'},
    'fatigue_management': {'input': 'fatigue', 'output': 'fatigue', 'direction': 'decrease'},
    'pressure_avoidance': {'input': 'pressure', 'output': 'pressure', 'direction': 'decrease'},
    'pain_endorphin_tradeoff': {'input': 'pain', 'output': 'endorphin', 'direction': 'increase'},
    'endorphin_pain_tradeoff': {'input': 'endorphin', 'output': 'pain', 'direction': 'increase'},
}


def matches_known_signature(sig):
    """Check if a cluster signature matches any known receptor pattern."""
    for name, known in KNOWN_SIGNATURES.items():
        if sig['dominant_input'] == known['input'] and sig['dominant_output'] == known['output']:
            return name
    return None


def detect_novel_receptors(engine, discovered_receptor_ids=None,
                           min_certainty=0.5, min_count=5,
                           sim_threshold=0.7):
    """Main entry point: find behavioral clusters that don't match known receptors.

    Returns a list of novel receptor candidates, each with:
      - cluster signature (what it responds to, what it produces)
      - novelty score (how different it is from known patterns)
      - size (how many mental model entries support it)
      - example entries
    """
    if discovered_receptor_ids is None:
        discovered_receptor_ids = set()

    entries = extract_qualified_entries(engine, min_certainty, min_count)
    if len(entries) < 10:
        return []

    clusters = cluster_entries(entries, sim_threshold)
    if not clusters:
        return []

    novel_candidates = []

    for cluster in clusters:
        sig = characterize_cluster(cluster)

        known_match = matches_known_signature(sig)
        if known_match:
            continue

        if sig['delta_consistency'] < 0.3:
            continue

        if sig['mean_certainty'] < 0.5:
            continue

        novelty_score = _compute_novelty(sig, clusters)

        candidate = {
            'cluster_id': cluster['id'],
            'size': cluster['size'],
            'signature': sig,
            'known_match': known_match,
            'novelty_score': round(novelty_score, 4),
            'description': _generate_description(sig),
        }
        novel_candidates.append(candidate)

    novel_candidates.sort(key=lambda c: -c['novelty_score'])
    return novel_candidates


def _compute_novelty(sig, all_clusters):
    """How novel is this signature relative to the known patterns?"""
    score = 0.0

    if sig['cross_modal']:
        score += 0.3

    score += sig['delta_consistency'] * 0.2

    score += sig['action_specificity'] * 0.2

    if sig['reward_std'] > 0.5:
        score += 0.15

    if sig['mean_delta_magnitude'] > 0.1:
        score += 0.15

    return min(1.0, score)


def _generate_description(sig):
    """Human-readable description of what the cluster does."""
    parts = []

    if sig['dominant_input'] and sig['dominant_output']:
        if sig['cross_modal']:
            parts.append(f"Cross-modal: responds to {sig['dominant_input']}, "
                        f"produces {sig['dominant_output']} changes")
        elif sig['dominant_input'] == sig['dominant_output']:
            parts.append(f"Within-modal {sig['dominant_input']} processing")
        else:
            parts.append(f"{sig['dominant_input']} → {sig['dominant_output']}")

    if sig['action_specificity'] > 0.7:
        parts.append("highly action-specific")
    elif sig['action_specificity'] < 0.3:
        parts.append("action-general (fires across many actions)")

    if sig['reward_std'] > 0.5:
        parts.append("reward-volatile (sometimes good, sometimes bad)")

    if sig['delta_consistency'] > 0.8:
        parts.append("highly consistent effect")

    return "; ".join(parts) if parts else "uncharacterized cluster"


def report_novel_receptors(engine, discovered_ids=None, verbose=True):
    """Run detection and print a report."""
    candidates = detect_novel_receptors(engine, discovered_ids)

    if verbose:
        print(f"\n{'='*60}")
        print(f"NOVEL RECEPTOR DETECTION")
        print(f"{'='*60}")

        entries = extract_qualified_entries(engine)
        clusters = cluster_entries(entries)
        print(f"  Qualified entries: {len(entries)}")
        print(f"  Clusters found: {len(clusters)}")
        print(f"  Novel candidates: {len(candidates)}")

        if not candidates:
            print("  No novel receptors detected.")
            print("  (All clusters match known patterns or lack coherence)")
        else:
            for i, c in enumerate(candidates):
                print(f"\n  Candidate {i+1}: (novelty={c['novelty_score']:.3f}, "
                      f"size={c['size']}, cert={c['signature']['mean_certainty']:.3f})")
                print(f"    {c['description']}")
                sig = c['signature']
                print(f"    Input: {sig['dominant_input']} ({sig['input_strength']:.3f})")
                print(f"    Output: {sig['dominant_output']} ({sig['output_strength']:.3f})")
                print(f"    Consistency: {sig['delta_consistency']:.3f}")
                print(f"    Reward: {sig['mean_reward']:.3f} ± {sig['reward_std']:.3f}")
                print(f"    Actions: {sig['unique_actions']} unique "
                      f"(specificity={sig['action_specificity']:.3f})")

    return candidates


if __name__ == '__main__':
    from environment import Environment, Organism, NPC
    from mental_model import build_mental_model

    print("Building test organism (500 steps)...")
    env = Environment(seed=42)
    org = Organism()
    org.reset()
    npc = NPC()
    npc.reset()

    for step in range(500):
        npc.step(env, step)
        actions = org.compute_optimal_actions(env, step, npc=npc)
        obs, reward = org.step(actions, env, step, npc=npc)

    engine = build_mental_model(org.experience_log)

    candidates = report_novel_receptors(engine)
    print(f"\nTotal novel candidates: {len(candidates)}")

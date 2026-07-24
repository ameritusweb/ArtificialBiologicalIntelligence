import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict


CORE_OBS_DIM = 96
NUM_MUSCLE_BITS = 18
EMBED_DIM = 32


@dataclass
class MappingEntry:
    context_embedding: np.ndarray
    delta: np.ndarray
    certainty: float
    count: int
    reward: float


@dataclass
class PatternEntry:
    motif: tuple
    context_embedding: np.ndarray
    cumulative_delta: np.ndarray
    certainty: float
    count: int
    compression_gain: float
    m2: np.ndarray = None
    representative_obs: np.ndarray = None


class ContrastiveEncoder(nn.Module):
    def __init__(self, obs_dim=96, embed_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

    def embed(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            emb = self.forward(obs)
        return emb.squeeze(0).numpy()

    def embed_batch(self, obs_array):
        t = torch.FloatTensor(obs_array)
        with torch.no_grad():
            emb = self.forward(t)
        return emb.numpy()


class ActionFamilyManager:
    def __init__(self, embed_dim=EMBED_DIM):
        self.embed_dim = embed_dim
        self.families = {}
        self.action_to_family = {}
        self.family_weights = {}

    def discover_families(self, global_log):
        for entry in global_log:
            ah = action_to_hash(entry['action'])
            emission = entry['action'][NUM_MUSCLE_BITS:]
            has_emission = any(int(b) for b in emission)
            self.action_to_family[ah] = 1 if has_emission else 0

        family_counts = defaultdict(int)
        for fam in self.action_to_family.values():
            family_counts[fam] += 1
        self.families = dict(family_counts)

        for fam_id in self.families:
            self.family_weights[fam_id] = np.ones(self.embed_dim, dtype=np.float32)

    def learn_weights(self, global_log, encoder, store, seed=0):
        cdim = getattr(encoder, 'obs_dim', CORE_OBS_DIM)
        family_data = defaultdict(lambda: {'embs': [], 'errors': []})
        rng = np.random.RandomState(seed)

        sample_size = min(20000, len(global_log))
        indices = rng.choice(len(global_log), sample_size, replace=False)

        for idx in indices:
            entry = global_log[idx]
            ah = action_to_hash(entry['action'])
            fam = self.action_to_family.get(ah, 0)

            obs_b = entry['obs_before'][:cdim]
            obs_a = entry['obs_after'][:cdim]
            actual_delta = obs_a - obs_b
            emb = encoder.embed(obs_b)

            results = store.query(ah, emb, top_k=5)
            if not results:
                continue

            scores = np.maximum(np.array([s for _, s in results]), 0.0)
            total = scores.sum()
            if total < 1e-8:
                continue
            weights = scores / total
            predicted = sum(w * e.delta for (e, _), w in zip(results, weights))
            error = float(np.mean((predicted - actual_delta) ** 2))

            family_data[fam]['embs'].append(emb)
            family_data[fam]['errors'].append(error)

        for fam_id, data in family_data.items():
            if len(data['embs']) < 50:
                continue
            embs = np.array(data['embs'])
            errors = np.array(data['errors'])
            median_err = np.median(errors)
            good = errors < median_err
            bad = errors >= median_err

            if good.sum() < 10 or bad.sum() < 10:
                continue

            good_mean = embs[good].mean(axis=0)
            bad_mean = embs[bad].mean(axis=0)
            diff = np.abs(good_mean - bad_mean)
            diff_norm = diff / (diff.max() + 1e-8)
            self.family_weights[fam_id] = (0.5 + 0.5 * diff_norm).astype(np.float32)

    def get_family(self, action_hash):
        return self.action_to_family.get(action_hash, 0)

    def get_weights(self, action_hash):
        fam = self.get_family(action_hash)
        return self.family_weights.get(fam, np.ones(self.embed_dim, dtype=np.float32))

    def weighted_similarity(self, emb_a, emb_b, action_hash):
        w = self.get_weights(action_hash)
        wa = emb_a * w
        wb = emb_b * w
        na = np.linalg.norm(wa) + 1e-8
        nb = np.linalg.norm(wb) + 1e-8
        return float(np.dot(wa, wb) / (na * nb))

    def get_stats(self):
        weight_spreads = {}
        for fam_id, w in self.family_weights.items():
            weight_spreads[fam_id] = {'min': float(w.min()), 'max': float(w.max()),
                                       'std': float(w.std())}
        return {
            'num_families': len(self.families),
            'family_sizes': dict(self.families),
            'weight_spreads': weight_spreads,
        }


def action_to_hash(action):
    return int(sum(int(action[i]) << i for i in range(len(action))))


def train_contrastive_encoder(global_log, obs_dim=96, embed_dim=32,
                              epochs=10, batch_size=256, lr=1e-3,
                              temperature=0.1, max_samples=30000, seed=0):
    rng = np.random.RandomState(seed)
    encoder = ContrastiveEncoder(obs_dim, embed_dim)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    sample_indices = rng.choice(len(global_log), min(max_samples, len(global_log)), replace=False)
    sampled = [global_log[i] for i in sample_indices]

    action_groups = defaultdict(list)
    all_obs = np.zeros((len(sampled), obs_dim), dtype=np.float32)
    all_deltas = np.zeros((len(sampled), obs_dim), dtype=np.float32)

    for idx, entry in enumerate(sampled):
        ah = action_to_hash(entry['action'])
        action_groups[ah].append(idx)
        all_obs[idx] = entry['obs_before'][:obs_dim]
        all_deltas[idx] = entry['obs_after'][:obs_dim] - entry['obs_before'][:obs_dim]

    valid_groups = {ah: np.array(indices) for ah, indices in action_groups.items() if len(indices) >= 4}
    if not valid_groups:
        print("  Warning: no valid action groups, skipping encoder training")
        return encoder

    group_keys = list(valid_groups.keys())
    all_obs_t = torch.FloatTensor(all_obs)

    for group_key in valid_groups:
        g_idx = valid_groups[group_key]
        g_deltas = all_deltas[g_idx]
        norms = np.linalg.norm(g_deltas, axis=1, keepdims=True) + 1e-8
        valid_groups[group_key] = (g_idx, g_deltas / norms)

    batches_per_epoch = max(1, len(sampled) // batch_size // 2)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for _ in range(batches_per_epoch):
            anchor_indices = []
            pos_indices = []
            neg_indices = []

            for _ in range(batch_size):
                gk = group_keys[rng.randint(len(group_keys))]
                g_idx, g_norm_deltas = valid_groups[gk]
                if len(g_idx) < 4:
                    continue

                a_pos = rng.randint(len(g_idx))
                a_delta = g_norm_deltas[a_pos]
                sims = g_norm_deltas @ a_delta
                sorted_pos = np.argsort(-sims)
                sorted_pos = sorted_pos[sorted_pos != a_pos]

                if len(sorted_pos) < 2:
                    continue

                p_pos = sorted_pos[rng.randint(min(3, len(sorted_pos)))]
                n_pos = sorted_pos[-(rng.randint(min(3, len(sorted_pos))) + 1)]

                anchor_indices.append(g_idx[a_pos])
                pos_indices.append(g_idx[p_pos])
                neg_indices.append(g_idx[n_pos])

            if len(anchor_indices) < 8:
                continue

            a_obs = all_obs_t[anchor_indices]
            p_obs = all_obs_t[pos_indices]
            n_obs = all_obs_t[neg_indices]

            encoder.train()
            z_a = encoder(a_obs)
            z_p = encoder(p_obs)
            z_n = encoder(n_obs)

            pos_sim = (z_a * z_p).sum(dim=-1) / temperature
            neg_sim = (z_a * z_n).sum(dim=-1) / temperature
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if n_batches > 0 and (epoch + 1) % 5 == 0:
            print(f"    Encoder epoch {epoch+1}/{epochs}  loss={total_loss/n_batches:.4f}")

    encoder.eval()
    return encoder


class CausalMappingStore:
    MERGE_THRESHOLD = 0.90
    CERTAINTY_WINDOW = 20

    def __init__(self):
        self.mappings = defaultdict(list)
        self.total_count = 0
        self._emb_cache = None
        self._reward_cache = None
        self._cache_dirty = True

    def add_mapping(self, action_hash, context_embedding, delta, reward):
        entries = self.mappings[action_hash]
        for entry in entries:
            sim = float(np.dot(context_embedding, entry.context_embedding))
            if sim > self.MERGE_THRESHOLD:
                n = entry.count
                delta_diff = np.linalg.norm(entry.delta - delta)
                delta_norm = np.linalg.norm(delta) + 1e-8
                outcome_match = max(0.0, 1.0 - min(1.0, delta_diff / delta_norm))
                entry.delta = (entry.delta * n + delta) / (n + 1)
                entry.reward = (entry.reward * n + reward) / (n + 1)
                alpha = max(1.0 / self.CERTAINTY_WINDOW, 1.0 / (n + 1))
                entry.certainty = entry.certainty * (1 - alpha) + outcome_match * alpha
                entry.certainty = float(np.clip(entry.certainty, 0.05, 0.99))
                entry.count += 1
                return

        self.mappings[action_hash].append(MappingEntry(
            context_embedding=context_embedding.copy(),
            delta=delta.copy(),
            certainty=0.5,
            count=1,
            reward=float(reward),
        ))
        self.total_count += 1
        self._cache_dirty = True

    def get_cached_embeddings(self):
        if self._cache_dirty or self._emb_cache is None:
            all_embs = []
            all_rewards = []
            for entries in self.mappings.values():
                for entry in entries:
                    all_embs.append(entry.context_embedding)
                    all_rewards.append(entry.reward)
            if all_embs:
                self._emb_cache = np.array(all_embs, dtype=np.float32)
                self._reward_cache = np.array(all_rewards, dtype=np.float32)
            else:
                self._emb_cache = np.zeros((0, 32), dtype=np.float32)
                self._reward_cache = np.zeros(0, dtype=np.float32)
            self._cache_dirty = False
        return self._emb_cache, self._reward_cache

    def query(self, action_hash, context_embedding, top_k=10, family_manager=None):
        entries = self.mappings.get(action_hash, [])
        if not entries:
            return []

        if family_manager is not None:
            w = family_manager.get_weights(action_hash)
            q_weighted = context_embedding * w
            q_norm = np.linalg.norm(q_weighted) + 1e-8
            scored = []
            for e in entries:
                e_weighted = e.context_embedding * w
                e_norm = np.linalg.norm(e_weighted) + 1e-8
                sim = float(np.dot(q_weighted, e_weighted) / (q_norm * e_norm))
                scored.append((e, sim * e.certainty))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        if len(entries) <= top_k:
            return [(e, float(np.dot(context_embedding, e.context_embedding)) * e.certainty)
                    for e in entries]

        embs = np.array([e.context_embedding for e in entries])
        sims = embs @ context_embedding
        certs = np.array([e.certainty for e in entries])
        scores = sims * certs
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(entries[i], float(scores[i])) for i in top_idx]

    def update_certainty(self, action_hash, context_embedding, predicted_delta, observed_delta):
        entries = self.mappings.get(action_hash, [])
        if not entries:
            return

        embs = np.array([e.context_embedding for e in entries])
        sims = embs @ context_embedding
        best_i = int(np.argmax(sims))

        if sims[best_i] > 0.5:
            entry = entries[best_i]
            obs_norm = np.linalg.norm(observed_delta) + 1e-8
            outcome_match = max(0.0, 1.0 - min(1.0, np.linalg.norm(predicted_delta - observed_delta) / obs_norm))
            alpha = max(1.0 / self.CERTAINTY_WINDOW, 1.0 / (entry.count + 1))
            entry.certainty = entry.certainty * (1 - alpha) + outcome_match * alpha
            entry.certainty = float(np.clip(entry.certainty, 0.05, 0.99))

    def build_from_log(self, global_log, encoder, batch_size=1024):
        n = len(global_log)
        cdim = getattr(encoder, 'obs_dim', CORE_OBS_DIM)
        all_obs = np.array([e['obs_before'][:cdim] for e in global_log], dtype=np.float32)
        all_deltas = np.array([e['obs_after'][:cdim] - e['obs_before'][:cdim] for e in global_log], dtype=np.float32)
        all_rewards = np.array([e['reward'] for e in global_log], dtype=np.float32)
        all_hashes = np.array([action_to_hash(e['action']) for e in global_log])

        all_embeddings = np.zeros((n, 32), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            all_embeddings[start:end] = encoder.embed_batch(all_obs[start:end])

        for i in range(n):
            self.add_mapping(int(all_hashes[i]), all_embeddings[i], all_deltas[i], float(all_rewards[i]))

            if (i + 1) % 50000 == 0:
                print(f"    Processed {i+1:,}/{n:,} log entries, "
                      f"store size: {self.total_count:,}")

    def get_stats(self):
        all_entries = [e for entries in self.mappings.values() for e in entries]
        if not all_entries:
            return {'total_mappings': 0, 'num_action_patterns': 0,
                    'avg_certainty': 0.0, 'avg_count': 0.0}
        return {
            'total_mappings': len(all_entries),
            'num_action_patterns': len(self.mappings),
            'avg_certainty': float(np.mean([e.certainty for e in all_entries])),
            'avg_count': float(np.mean([e.count for e in all_entries])),
        }


class PatternStore:
    MERGE_THRESHOLD = 0.85
    MIN_COUNT = 5

    def __init__(self):
        self.patterns = defaultdict(list)
        self._first_action_index = defaultdict(list)
        self.total_count = 0

    def build_from_log(self, global_log, encoder, causal_store, steps_per_episode=300):
        n = len(global_log)
        cdim = getattr(encoder, 'obs_dim', CORE_OBS_DIM)
        all_obs = np.array([e['obs_before'][:cdim] for e in global_log], dtype=np.float32)
        all_obs_after = np.array([e['obs_after'][:cdim] for e in global_log], dtype=np.float32)
        all_hashes = [action_to_hash(e['action']) for e in global_log]

        all_embeddings = encoder.embed_batch(all_obs)
        num_episodes = n // steps_per_episode

        motif_data = defaultdict(list)
        for ep in range(num_episodes):
            ep_start = ep * steps_per_episode
            for s in range(steps_per_episode - 1):
                idx = ep_start + s
                motif = (all_hashes[idx], all_hashes[idx + 1])
                cum_delta = all_obs_after[idx + 1] - all_obs[idx]
                motif_data[motif].append((all_embeddings[idx], cum_delta, all_obs[idx]))

        for motif, entries in motif_data.items():
            if len(entries) < self.MIN_COUNT:
                continue
            for emb, delta, obs in entries:
                self._add_pattern_obs(motif, emb, delta, obs)

        self._prune_and_index(causal_store, encoder)

        if self.total_count > 0 and (num_episodes % 100 == 0 or num_episodes == n // steps_per_episode):
            pass

    def _add_pattern_obs(self, motif, embedding, delta, obs=None):
        for pat in self.patterns[motif]:
            sim = float(np.dot(embedding, pat.context_embedding))
            if sim > self.MERGE_THRESHOLD:
                n = pat.count
                old_mean = pat.cumulative_delta.copy()
                pat.cumulative_delta = (old_mean * n + delta) / (n + 1)
                new_diff = delta - pat.cumulative_delta
                old_diff = delta - old_mean
                if pat.m2 is None:
                    pat.m2 = np.zeros_like(delta)
                pat.m2 += old_diff * new_diff
                pat.context_embedding = (pat.context_embedding * n + embedding) / (n + 1)
                norm = np.linalg.norm(pat.context_embedding)
                if norm > 1e-8:
                    pat.context_embedding /= norm
                if obs is not None:
                    pat.representative_obs = obs.copy()
                pat.count += 1
                return

        self.patterns[motif].append(PatternEntry(
            motif=motif,
            context_embedding=embedding.copy(),
            cumulative_delta=delta.copy(),
            certainty=0.5,
            count=1,
            compression_gain=0.0,
            m2=np.zeros_like(delta),
            representative_obs=obs.copy() if obs is not None else None,
        ))

    def _prune_and_index(self, causal_store, encoder):
        pruned = defaultdict(list)
        engine_temp = MentalModelEngine(encoder, causal_store)

        for motif, entries in self.patterns.items():
            for pat in entries:
                if pat.count < self.MIN_COUNT:
                    continue
                if pat.m2 is not None and pat.count > 1:
                    variance = pat.m2 / pat.count
                    std = float(np.sqrt(np.mean(variance)))
                    pat.certainty = float(np.clip(1.0 - std, 0.05, 0.99))
                else:
                    pat.certainty = 0.5

                num_actions = getattr(encoder, 'num_actions', 22)
                a1_action = np.zeros(num_actions, dtype=int)
                for bit in range(num_actions):
                    if motif[0] & (1 << bit):
                        a1_action[bit] = 1
                a2_action = np.zeros(num_actions, dtype=int)
                for bit in range(num_actions):
                    if motif[1] & (1 << bit):
                        a2_action[bit] = 1

                if pat.representative_obs is not None:
                    chain_obs = pat.representative_obs
                else:
                    chain_obs = np.zeros(getattr(encoder, 'obs_dim', CORE_OBS_DIM))
                _, chain_cert = engine_temp.chain([a1_action, a2_action], chain_obs)
                pat.compression_gain = pat.certainty - chain_cert

                if pat.compression_gain > 0:
                    pruned[motif].append(pat)

        self.patterns = pruned
        self.total_count = sum(len(v) for v in self.patterns.values())

        self._first_action_index.clear()
        for motif in self.patterns:
            self._first_action_index[motif[0]].append(motif)

    def query(self, prev_action_hash, context_embedding, top_k=5):
        motifs = self._first_action_index.get(prev_action_hash, [])
        if not motifs:
            return []

        results = []
        for motif in motifs:
            for pat in self.patterns[motif]:
                sim = float(np.dot(context_embedding, pat.context_embedding))
                score = max(0.0, sim) * pat.certainty
                results.append((pat, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self):
        all_pats = [p for pats in self.patterns.values() for p in pats]
        if not all_pats:
            return {'total_patterns': 0, 'num_motif_types': 0,
                    'avg_certainty': 0.0, 'avg_count': 0.0, 'avg_compression_gain': 0.0}
        return {
            'total_patterns': len(all_pats),
            'num_motif_types': len(self.patterns),
            'avg_certainty': float(np.mean([p.certainty for p in all_pats])),
            'avg_count': float(np.mean([p.count for p in all_pats])),
            'avg_compression_gain': float(np.mean([p.compression_gain for p in all_pats])),
        }

    def extract_concepts(self, top_k=10):
        scored = []
        for motif, patterns in self.patterns.items():
            for pat in patterns:
                score = pat.certainty * pat.count * max(0.0, pat.compression_gain)
                scored.append((score, motif, pat))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def get_concept_stats(self):
        all_pats = [p for pats in self.patterns.values() for p in pats]
        stable = [p for p in all_pats
                  if p.certainty > 0.7 and p.count > 10 and p.compression_gain > 0.1]
        return {
            'num_stable_concepts': len(stable),
            'total_patterns': len(all_pats),
            'avg_concept_quality': float(np.mean([p.compression_gain for p in stable])) if stable else 0.0,
            'avg_concept_certainty': float(np.mean([p.certainty for p in stable])) if stable else 0.0,
        }


class EntityRelationStore:
    """Named entities with typed relations — for social cognition canopy.

    Inspired by BitGenesis knowledge graph. Stores entities (identified by
    embedding similarity) and relations between them. Enables reasoning
    about specific agents rather than anonymous observation-vector regions.
    """

    def __init__(self):
        self.entities = {}
        self.relations = defaultdict(list)

    def observe_entity(self, entity_id, embedding, properties=None):
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                'embedding': embedding.copy(),
                'properties': properties or {},
                'observation_count': 1,
                'last_seen': 0,
            }
        else:
            e = self.entities[entity_id]
            n = e['observation_count']
            e['embedding'] = (e['embedding'] * n + embedding) / (n + 1)
            norm = np.linalg.norm(e['embedding'])
            if norm > 1e-8:
                e['embedding'] /= norm
            if properties:
                e['properties'].update(properties)
            e['observation_count'] += 1

    def add_relation(self, subject_id, relation_type, object_id, certainty=0.5):
        for rel in self.relations[subject_id]:
            if rel['type'] == relation_type and rel['object'] == object_id:
                rel['certainty'] = 0.8 * rel['certainty'] + 0.2 * certainty
                rel['count'] += 1
                return
        self.relations[subject_id].append({
            'type': relation_type,
            'object': object_id,
            'certainty': certainty,
            'count': 1,
        })

    def get_entity(self, entity_id):
        return self.entities.get(entity_id)

    def get_relations(self, entity_id, relation_type=None):
        rels = self.relations.get(entity_id, [])
        if relation_type:
            return [r for r in rels if r['type'] == relation_type]
        return rels

    def find_similar_entity(self, embedding, threshold=0.7):
        best_id, best_sim = None, -1
        for eid, e in self.entities.items():
            sim = float(np.dot(embedding, e['embedding']))
            if sim > best_sim:
                best_sim = sim
                best_id = eid
        if best_sim >= threshold:
            return best_id, best_sim
        return None, 0.0

    def infer_transitive(self, entity_id, relation_type, max_depth=3):
        visited = set()
        results = []
        queue = [(entity_id, 0, 1.0)]
        while queue:
            eid, depth, cert = queue.pop(0)
            if eid in visited or depth >= max_depth:
                continue
            visited.add(eid)
            for rel in self.get_relations(eid, relation_type):
                chain_cert = cert * rel['certainty']
                results.append((rel['object'], chain_cert, depth + 1))
                queue.append((rel['object'], depth + 1, chain_cert))
        return results

    def get_stats(self):
        return {
            'num_entities': len(self.entities),
            'num_relations': sum(len(v) for v in self.relations.values()),
            'entity_ids': list(self.entities.keys()),
        }


class MentalModelEngine:
    def __init__(self, encoder, store):
        self.encoder = encoder
        self.store = store
        self.pattern_store = None
        self.family_manager = None
        self.entity_store = EntityRelationStore()
        self.core_obs_dim = getattr(encoder, 'obs_dim', CORE_OBS_DIM)

    def _core_obs(self, obs):
        if isinstance(obs, np.ndarray) and obs.shape[-1] > self.core_obs_dim:
            return obs[..., :self.core_obs_dim]
        return obs

    def predict_delta(self, obs_before, action, top_k=10):
        obs_before = self._core_obs(obs_before)
        embedding = self.encoder.embed(obs_before)
        ah = action_to_hash(action)
        results = self.store.query(ah, embedding, top_k, family_manager=self.family_manager)

        if not results:
            return np.zeros_like(obs_before), 0.0, 0

        scores = np.maximum(np.array([s for _, s in results]), 0.0)
        total = scores.sum()
        if total < 1e-8:
            return np.zeros_like(obs_before), 0.0, 0
        weights = scores / total
        predicted = sum(w * e.delta for (e, _), w in zip(results, weights))
        avg_cert = float(np.mean([e.certainty for e, _ in results]))
        return predicted, avg_cert, len(results)

    def predict_delta_batch(self, obs_before, actions, top_k=10):
        """Predict deltas for multiple actions from the same observation.

        One embedding call, grouped store queries. Returns list of
        (predicted_delta, certainty, n_entries) tuples, one per action.
        """
        obs_before = self._core_obs(obs_before)
        embedding = self.encoder.embed(obs_before)
        cdim = len(obs_before)

        results = []
        hashes = [action_to_hash(a) for a in actions]

        # Group by hash to avoid redundant queries
        hash_to_indices = defaultdict(list)
        for i, ah in enumerate(hashes):
            hash_to_indices[ah].append(i)

        per_action = [None] * len(actions)

        for ah, indices in hash_to_indices.items():
            query_results = self.store.query(
                ah, embedding, top_k, family_manager=self.family_manager)

            if not query_results:
                for i in indices:
                    per_action[i] = (np.zeros(cdim), 0.0, 0)
                continue

            scores = np.maximum(np.array([s for _, s in query_results]), 0.0)
            total = scores.sum()
            if total < 1e-8:
                for i in indices:
                    per_action[i] = (np.zeros(cdim), 0.0, 0)
                continue

            weights = scores / total
            predicted = sum(w * e.delta for (e, _), w in zip(query_results, weights))
            avg_cert = float(np.mean([e.certainty for e, _ in query_results]))
            result = (predicted, avg_cert, len(query_results))

            for i in indices:
                per_action[i] = result

        return per_action

    def chain(self, action_sequence, obs_before, top_k=10):
        state = obs_before.copy()
        cumulative_delta = np.zeros_like(obs_before)
        chain_certainty = 1.0

        for action in action_sequence:
            delta, cert, count = self.predict_delta(state, action, top_k)
            cumulative_delta += delta
            chain_certainty *= max(cert, 0.01)
            state = state + delta

        return cumulative_delta, chain_certainty

    def observe_npc(self, obs, npc_start=141):
        """Extract NPC as a named entity from observation vector."""
        if len(obs) <= npc_start + 7:
            return
        npc_features = obs[npc_start:npc_start + 8]
        npc_dist = npc_features[0]
        if npc_dist < 0.01:
            return
        npc_emb = self.encoder.embed(obs[:self.core_obs_dim])
        entity_id, sim = self.entity_store.find_similar_entity(npc_emb, threshold=0.6)
        if entity_id is None:
            entity_id = f"npc_{len(self.entity_store.entities)}"

        props = {
            'distance': float(npc_dist),
            'speed': float(npc_features[3]) if len(npc_features) > 3 else 0,
            'erraticism': float(npc_features[6]) if len(npc_features) > 6 else 0,
        }
        self.entity_store.observe_entity(entity_id, npc_emb, props)

        if npc_features[6] > 0.3:
            self.entity_store.add_relation(entity_id, 'exhibits', 'erratic_behavior', 0.7)
        if npc_dist > 0.5:
            self.entity_store.add_relation('self', 'near', entity_id, float(npc_dist))

    def update(self, obs_before, action, obs_after, reward):
        obs_before = self._core_obs(obs_before)
        obs_after = self._core_obs(obs_after)
        embedding = self.encoder.embed(obs_before)
        ah = action_to_hash(action)
        delta = obs_after - obs_before

        predicted, _, _ = self.predict_delta(obs_before, action)
        self.store.update_certainty(ah, embedding, predicted, delta)
        self.store.add_mapping(ah, embedding, delta, reward)

    def compute_learning_progress(self, obs_before, action, obs_after, reward):
        obs_b = self._core_obs(obs_before)
        obs_a = self._core_obs(obs_after)
        actual_delta = obs_a - obs_b

        predicted_before, cert_before, n_before = self.predict_delta(obs_b, action)
        mse_before = float(np.mean((predicted_before - actual_delta) ** 2)) if n_before > 0 else 1.0

        self.update(obs_before, action, obs_after, reward)

        predicted_after, cert_after, n_after = self.predict_delta(obs_b, action)
        mse_after = float(np.mean((predicted_after - actual_delta) ** 2)) if n_after > 0 else 1.0

        raw_progress = mse_before - mse_after
        learning_progress = float(np.clip(raw_progress / (abs(raw_progress) + 0.01), 0.0, 1.0))
        return learning_progress, cert_after

    def get_prediction_quality(self, obs_before, action, obs_after):
        obs_before = self._core_obs(obs_before)
        obs_after = self._core_obs(obs_after)
        predicted, avg_cert, num_retrieved = self.predict_delta(obs_before, action)
        actual_delta = obs_after - obs_before

        if num_retrieved == 0:
            return {'mse': 0.0, 'pain_mse': 0.0, 'cosine': 0.0,
                    'num_retrieved': 0, 'avg_certainty': 0.0}

        mse = float(np.mean((predicted - actual_delta) ** 2))
        pain_mse = float(np.mean((predicted[:6] - actual_delta[:6]) ** 2))

        pred_norm = np.linalg.norm(predicted)
        actual_norm = np.linalg.norm(actual_delta)
        cosine = float(np.dot(predicted, actual_delta) / (pred_norm * actual_norm + 1e-8)) if pred_norm > 1e-8 and actual_norm > 1e-8 else 0.0

        return {'mse': mse, 'pain_mse': pain_mse, 'cosine': cosine,
                'num_retrieved': num_retrieved, 'avg_certainty': avg_cert}

    def compute_agency_features(self, obs_before, action, obs_after):
        obs_b = self._core_obs(obs_before)
        obs_a = self._core_obs(obs_after)
        actual_delta = obs_a - obs_b

        predicted_self, cert, count = self.predict_delta(obs_b, action)

        actual_norm = np.linalg.norm(actual_delta)
        predicted_norm = np.linalg.norm(predicted_self)
        external_delta = actual_delta - predicted_self
        external_norm = np.linalg.norm(external_delta)

        if count == 0 or actual_norm < 1e-6:
            controllability = 0.0
        else:
            controllability = float(np.clip(1.0 - external_norm / (actual_norm + 1e-6), 0.0, 1.0))

        external_change = float(np.clip(external_norm / (external_norm + 1.0), 0.0, 1.0))

        null_action = np.zeros_like(action)
        pred_null, _, n_null = self.predict_delta(obs_b, null_action)

        action_benefit = -predicted_self[0:6].sum() + predicted_self[6:12].sum()
        null_benefit = -pred_null[0:6].sum() + pred_null[6:12].sum()
        raw_improvement = action_benefit - null_benefit
        planning_value = float(np.clip(raw_improvement / (abs(raw_improvement) + 0.1), 0.0, 1.0))

        return controllability, external_change, planning_value

    def get_context_features_batch(self, observations):
        observations = self._core_obs(observations)
        embeddings = self.encoder.embed_batch(observations)

        stored_embs, stored_rewards = self.store.get_cached_embeddings()

        N = len(observations)
        if len(stored_embs) == 0:
            return np.zeros(N, dtype=np.float32), np.full(N, 0.5, dtype=np.float32)

        sims = embeddings @ stored_embs.T

        mm_familiarity = np.clip(np.max(sims, axis=1), 0.0, 1.0)

        top_k = min(10, sims.shape[1])
        top_indices = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        top_sims = np.take_along_axis(sims, top_indices, axis=1)
        top_rewards = stored_rewards[top_indices]
        weights = np.maximum(top_sims, 0.0)
        weight_sums = weights.sum(axis=1, keepdims=True) + 1e-8
        raw_quality = (weights * top_rewards).sum(axis=1) / weight_sums.squeeze()
        mm_context_quality = np.clip((raw_quality + 5.0) / 10.0, 0.0, 1.0)

        return mm_familiarity.astype(np.float32), mm_context_quality.astype(np.float32)

    def get_context_features(self, observation):
        obs = observation.reshape(1, -1) if observation.ndim == 1 else observation
        fam, quality = self.get_context_features_batch(obs)
        return float(fam[0]), float(quality[0])

    def query_pattern(self, prev_action_hash, obs_before):
        if self.pattern_store is None:
            return 0.0, 0.0
        obs_core = self._core_obs(obs_before)
        embedding = self.encoder.embed(obs_core)
        results = self.pattern_store.query(prev_action_hash, embedding)
        if not results:
            return 0.0, 0.0
        best_pat, best_score = results[0]
        return min(1.0, best_score), best_pat.certainty

    def query_concept(self, prev_action_hash, obs_before):
        if self.pattern_store is None:
            return 0.0, 0.0
        obs_core = self._core_obs(obs_before)
        embedding = self.encoder.embed(obs_core)
        results = self.pattern_store.query(prev_action_hash, embedding)
        if not results:
            return 0.0, 0.0
        best_pat, best_score = results[0]
        return min(1.0, best_score), max(0.0, best_pat.compression_gain)

    def get_stats(self):
        stats = self.store.get_stats()
        stats['encoder_params'] = sum(p.numel() for p in self.encoder.parameters())
        if self.pattern_store:
            stats.update({f'pat_{k}': v for k, v in self.pattern_store.get_stats().items()})
        return stats


def build_mental_model(global_log, obs_dim=96, core_obs_dim=None):
    if core_obs_dim is not None:
        obs_dim = core_obs_dim
    print("  Training contrastive encoder...")
    encoder = train_contrastive_encoder(global_log, obs_dim=obs_dim, epochs=10)
    print("  Building mapping store...")
    store = CausalMappingStore()
    store.build_from_log(global_log, encoder)
    engine = MentalModelEngine(encoder, store)
    print("  Discovering action families...")
    fm = ActionFamilyManager()
    fm.discover_families(global_log)
    fm.learn_weights(global_log, encoder, store)
    engine.family_manager = fm
    fs = fm.get_stats()
    print(f"    Families: {fs['num_families']}, sizes: {fs['family_sizes']}")
    for fam_id, ws in fs['weight_spreads'].items():
        print(f"    Family {fam_id}: weight range [{ws['min']:.3f}, {ws['max']:.3f}], std={ws['std']:.3f}")
    print("  Building pattern store...")
    pstore = PatternStore()
    pstore.build_from_log(global_log, encoder, store)
    engine.pattern_store = pstore
    ps = pstore.get_stats()
    print(f"    Patterns: {ps['total_patterns']}, Motif types: {ps['num_motif_types']}, "
          f"Avg cert: {ps['avg_certainty']:.3f}, Avg gain: {ps['avg_compression_gain']:.3f}")
    return engine


if __name__ == '__main__':
    from environment import Environment, Organism

    print("=== Building test experience log ===")
    env = Environment(seed=42)
    org = Organism()
    org.reset()
    for step in range(300):
        actions = org.compute_optimal_actions(env, step)
        org.step(actions, env, step)

    log = org.experience_log
    print(f"  Log entries: {len(log)}")

    print("\n=== Building mental model ===")
    engine = build_mental_model(log)
    stats = engine.get_stats()
    print(f"  Mappings: {stats['total_mappings']}, "
          f"Actions: {stats['num_action_patterns']}, "
          f"Avg certainty: {stats['avg_certainty']:.3f}, "
          f"Avg count: {stats['avg_count']:.1f}")

    print("\n=== Testing prediction ===")
    test_obs = log[150]['obs_before']
    test_action = log[150]['action']
    test_obs_after = log[150]['obs_after']
    quality = engine.get_prediction_quality(test_obs, test_action, test_obs_after)
    print(f"  MSE={quality['mse']:.4f}, Pain MSE={quality['pain_mse']:.4f}, "
          f"Cosine={quality['cosine']:.3f}, Retrieved={quality['num_retrieved']}")

    print("\nMental model tests passed!")

# The Causal Receptor Mental Model
## A Framework for Legible, Grounded Intelligence

---

## The Problem with Current AI Mental Models

In current transformer-based AI, the mental model is implicit. It lives inside the weights — unreadable, monolithic, and inseparable from the processing that uses it. You cannot inspect what the system believes about the world. You cannot edit a specific belief. You cannot transfer knowledge between systems without retraining. And you cannot tell the difference between "the system knows this" and "the system has learned to pattern-match text that sounds like knowing this."

This is not a minor engineering inconvenience. It is the fundamental reason current AI lacks grounded understanding. The mental model and the inference engine are tangled together, and the tangle is what prevents grounding.

This framework separates them.

---

## The Core Definition

A mental model is a collection of cause-effect mappings of the form:

**action → receptor state change, with a time delay and a certainty.**

Where:
- **Action** can be a physical movement, an imagined movement (forward simulation), or an internal cognitive operation — shifting attention, updating a hypothesis, retrieving a memory
- **Effect** is always ultimately a receptor firing — a change in the organism's internal state
- **Time delay** is a distribution, not a fixed value — effects arrive with varying timing
- **Certainty** is a probability derived from observation count and outcome history

This definition keeps the mental model grounded by construction. Every chain of reasoning, however long or abstract, must eventually cash out in a receptor state. There is no free-floating cognition. The tether is always there.

---

## Why Receptors as the Effect

Every other approach to mental models treats them as separate from the motivational system. The world model lives here. The drives and receptors live there. A third thing connects them. That third thing is always where the framework gets vague.

This framework avoids the connection problem by eliminating the separation.

By making receptor states the output of every cause-effect mapping, the mental model and the receptor economy are the same structure. The organism doesn't consult its world model and then check it against its motivations. The world model *is* the motivational system, expressed as learned predictions about what causes what to fire.

This also means the mental model is inherently evaluative. The organism doesn't just represent what will happen — it represents what will happen to *it*, in terms of states it is already motivated to seek or avoid. Neutral facts don't exist in this framework. Every mapping points toward or away from something that matters.

---

## The Three Levels

The mental model operates at three levels, each built from the one below:

### Level 1: Primitive Mappings

Single cause-effect pairs. One action, one receptor, one delta, one delay, one certainty.

*Extend limb 3 → pain receptor 4 decreases by 0.3, in approximately 200ms, with 72% certainty.*

These are the atoms. They update constantly as the organism observes outcomes. They are the fastest to acquire and the fastest to revise.

### Level 2: Chains

Sequences of primitive mappings where the predicted receptor state from one mapping becomes the context for the next query.

*Extend limb 3, then rotate left, then retract limb 2 → net receptor state after 800ms, with compounded certainty.*

Chains are how the organism plans. They are assembled on the fly from primitives by the Python engine — not stored as fixed sequences, but composed dynamically from the primitive table. Certainty multiplies across steps. Delays convolve. Receptor deltas accumulate.

### Level 3: Patterns

Recurring subsequences identified across multiple chains. When the organism repeatedly traverses similar causal paths to reach similar receptor outcomes, those paths get recognized as reusable patterns.

This is where concepts emerge. A concept is a pattern — a compressed causal chain that the organism has found reliably useful across different contexts. The compression receptor fires when a pattern is identified. The pattern gets stored as a shortcut.

Analogy is pattern matching across chains. Two situations are analogous not because they look the same on the surface but because their cause-effect chains share causal topology — they move through similar receptor states in a similar order, even when the specific actions differ.

---

## The Architecture

The mental model lives outside the transformer as an explicit, legible database. The transformer reads from it but is not the same thing as it.

```
ENVIRONMENT
     │
     ▼
RECEPTORS ──────────────────────────────────────────┐
     │                                               │
     ▼                                               │
TRANSFORMER                                         │
(inference engine)                                  │
     │                                              │
     │ queries                                      │ updates
     ▼                                               │
MENTAL MODEL DATABASE ◄─────────────────────────────┘
(causal_mappings + causal_contexts + raw_experience_log)
     │
     ▼
MUSCLE ACTIVATIONS
     │
     ▼
ENVIRONMENT
```

The transformer's job is narrowly defined: take the current receptor state, query the mental model for relevant cause-effect mappings, evaluate candidate action chains, and output muscle activations. Fast, parallel, trained on this specific task.

The mental model's job is to accumulate the organism's learned understanding of what causes what, expressed entirely in terms of receptor outcomes. It updates after every action-observation cycle. It grows throughout the organism's lifetime.

---

## The Database Schema

### Primitive Mappings Table

```sql
CREATE TABLE causal_mappings (
    mapping_id INT PRIMARY KEY,
    initiating_action VARCHAR(255),     -- physical, imagined, or cognitive action
    resulting_receptor VARCHAR(255),    -- target receptor (e.g., 'PAIN_L4', 'COMPRESSION')
    delta_magnitude FLOAT,              -- expected direction and amount of change (+/-)
    initial_receptor_value FLOAT,       -- receptor value at time of observation
    time_delay_ms INT,                  -- expected temporal distance
    time_delay_std_ms INT,              -- uncertainty in timing
    certainty FLOAT,                    -- confidence score (0.0 to 1.0)
    use_count INT DEFAULT 0,            -- how often this mapping has been traversed
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_action_receptor ON causal_mappings(initiating_action, resulting_receptor);
```

One row per receptor affected. An action that changes six receptors simultaneously creates six rows. The Python engine groups by action when assembling the full receptor delta picture.

### Context Table

```sql
CREATE TABLE causal_contexts (
    context_id INT PRIMARY KEY,
    mapping_id INT REFERENCES causal_mappings(mapping_id),
    energy_level FLOAT,
    position_x FLOAT,
    position_y FLOAT,
    heading FLOAT,
    active_receptors JSONB,         -- snapshot of all receptor values at observation time
    environment_state JSONB,        -- field intensities nearby
    context_vector vector(64),      -- embedding for fuzzy similarity search
    action_family VARCHAR(255)      -- learned action family assignment
);

CREATE INDEX idx_context_mapping ON causal_contexts(mapping_id);
CREATE INDEX idx_context_vector ON causal_contexts
    USING ivfflat (context_vector vector_cosine_ops);
CREATE INDEX idx_action_family ON causal_contexts(action_family);
```

Context lives in a separate table to keep the mappings table fast and flat. The vector index enables fuzzy context matching — finding mappings observed in situations similar to the current one, even when the match isn't exact. The `action_family` field assigns each observation to a cluster of actions that share outcome-response profiles; this drives action-conditional similarity search (see The Contrastive Encoder below).

### The Raw Experience Log

```sql
CREATE TABLE experience_log (
    log_id BIGINT PRIMARY KEY,
    logged_at TIMESTAMP DEFAULT NOW(),
    initiating_action VARCHAR(255),
    context_snapshot JSONB,         -- full receptor and environment state at time of action
    observed_delta FLOAT,           -- actual receptor change observed
    resulting_receptor VARCHAR(255),
    delay_ms INT
);
```

**This table is append-only and never overwritten by any learned process.**

Every other component in the architecture — certainty scores, context embeddings, action family clusters, encoder weights — is a *view* derived from this log. The log is ground truth. No failure mode in any derived component can corrupt it.

This is what makes the architecture recoverable. "Resetting" any derived component means recomputing it from the log, not losing accumulated experience. The only cost of any reset is compute, not the organism's history of having acted in the world.

---

## The Contrastive Encoder

The context table uses cosine similarity over a 64-dimensional embedding to retrieve mappings observed in situations similar to the current one. This is nonparametric regression — a kernel estimator of `E[Δreceptor | action, context]` where the kernel is cosine similarity in the embedding space.

The estimator works exactly to the degree that distance in embedding space tracks difference in outcome distribution. A fixed embedding — one whose geometry is not shaped by outcomes — creates a metric mismatch problem: two contexts can be cosine-close because they agree on dimensions that don't matter for a given action while differing on the one that does. Retrieval returns spuriously similar contexts; certainty scores converge to noise.

There is a subtler version of the same problem: relevance is action-conditional. The context dimensions that determine the outcome of action A are generally not the ones that determine the outcome of action B. A single global metric imposes one partitioning of context space on every mapping, and is therefore wrong for most actions most of the time.

The fix is a small learned component: a **contrastive encoder** trained on `(context, action, Δreceptor)` triples from the experience log, with a per-action-family Mahalanobis reweighting that lets each action family attend to its own relevant context dimensions.

The training objective: pull together context pairs where the same action produced similar outcome distributions; push apart pairs where it did not. The supervision is the experience log itself — no external labels required.

**This does not compromise the separation between inference and knowledge.** The transformer still does not retrain. The knowledge base remains readable. A tiny encoder sits upstream of retrieval, learns which context dimensions matter per action family, and is the only component that trains on a gradient. Everything else — certainty updates, knowledge base rows — remains transparent and directly editable.

### Action Family Discovery

Action families are not predefined. Two actions belong to the same family when their outcome profiles covary with context in the same way — when they are sensitive to the same context dimensions. Families are discovered by clustering actions according to the similarity of their context → Δreceptor response functions in the raw log, not by anything about the action strings themselves.

The degenerate starting point is coherent: one family containing all actions is just the shared global metric. The system starts there and splits families as evidence accumulates that subsets of actions disagree about which context dimensions matter. Splitting is triggered by within-family variance in learned reweightings or persistent prediction error concentrated on a subset of actions.

### Bootstrapping and Inheritance

Early in an organism's lifetime, before the experience log is populated, the encoder has no signal. Three mitigations:

1. **Near-isotropic initialization.** Raw receptor-space similarity is a bad metric but an unbiased one — better than a random projection's confident noise.
2. **Scheduled update gating.** Early on, similarity thresholds are wide — most contexts count for updating, accepting regime-blur in exchange for accumulating any statistics at all. The threshold tightens as contrastive signal accumulates. This is the bias-variance dial scheduled rather than constant.
3. **Exploration alignment.** The intrinsic exploration signal already generates exactly what contrastive learning needs: the same action tried across deliberately varied contexts. The exploration drive and the encoder's training objective are naturally aligned. An intrinsic bonus for revisiting an action in contexts the encoder currently considers dissimilar directly buys useful contrastive pairs at no architectural cost.

Both encoder weights and action family assignments are inheritable across generations. "Which context dimensions tend to matter" and "which actions behave alike" are more stable across generations than any specific cause-effect row — they are closer to facts about the organism's embodiment and environment structure than to accumulated knowledge. Each generation inherits the previous generation's endpoint as its starting point and refines rather than relearns.

Canalization risk: inherited encoders can entrench — an offspring whose encoder is confidently wrong about a shifted environment retrieves badly with conviction. Mutation noise on inherited encoder weights, or an inherited-vs-relearned interpolation weight that decays with prediction error, handles this.

---

## The Python Engine

The engine handles everything the database cannot: traversal, branching, uncertainty propagation, and plan selection.

### Querying Relevant Mappings

```python
def query_relevant_mappings(current_state, action, db, top_k=20):
    action_family = get_action_family(action, db)
    current_vector = encoder.embed(current_state, action_family)  # action-conditional embedding
    
    # Find similar contexts via action-conditional similarity
    similar_contexts = db.query("""
        SELECT mapping_id, mahalanobis_similarity(context_vector, %s, %s) AS similarity
        FROM causal_contexts
        WHERE action_family = %s
        ORDER BY similarity DESC
        LIMIT %s
    """, [current_vector, action_family, action_family, top_k])
    
    # Pull mappings ranked by certainty * context similarity
    mappings = db.query("""
        SELECT cm.*, cc.similarity
        FROM causal_mappings cm
        JOIN causal_contexts cc ON cm.mapping_id = cc.mapping_id
        WHERE cm.mapping_id = ANY(%s)
        AND cm.initiating_action = %s
        ORDER BY cm.certainty * cc.similarity DESC
    """, [similar_context_ids, action])
    
    return mappings
```

The ordering `certainty * similarity` is the key: a high-certainty mapping observed in a very different context ranks below a moderate-certainty mapping observed in a nearly identical one. The action-conditional embedding ensures "similar" means similar in the dimensions that actually matter for this action.

### Chaining

```python
def chain(action_sequence, current_state, db):
    state = current_state.copy()
    chain_certainty = 1.0
    chain_delay_ms = 0
    receptor_deltas = defaultdict(float)
    
    for action in action_sequence:
        mappings = query_relevant_mappings(state, action, db)
        
        for mapping in mappings:
            # Accumulate receptor effects
            receptor_deltas[mapping.resulting_receptor] += (
                mapping.delta_magnitude * mapping.certainty
            )
            # Multiply certainty across steps
            chain_certainty *= mapping.certainty
            # Convolve delays
            chain_delay_ms += mapping.time_delay_ms
            # Update state for next query
            state = apply_delta(state, mapping)
    
    return receptor_deltas, chain_certainty, chain_delay_ms
```

### Branching — Forward Simulation

```python
def select_best_action(candidate_actions, current_state, db):
    results = []
    
    for action_sequence in candidate_actions:
        deltas, certainty, delay = chain(action_sequence, current_state, db)
        
        # Score by predicted receptor outcome
        score = (
            deltas.get('ENDORPHIN', 0) - 
            deltas.get('PAIN', 0)
        ) * certainty
        
        results.append((action_sequence, score, deltas, certainty, delay))
    
    return max(results, key=lambda x: x[1])
```

This is forward simulation expressed as database queries. The organism evaluates branches in parallel and selects the one with the best predicted receptor outcome. No special planning module — just the Python engine running CHAIN queries against the mental model.

### Updating After Observation

```python
def update_mapping(action, receptor, predicted_delta, observed_delta,
                   context, delay_ms, db):
    outcome_match = 1.0 - abs(predicted_delta - observed_delta)
    
    # Append to the immutable experience log first
    db.execute("""
        INSERT INTO experience_log 
            (initiating_action, context_snapshot, observed_delta, resulting_receptor, delay_ms)
        VALUES (%s, %s, %s, %s, %s)
    """, [action, context, observed_delta, receptor, delay_ms])
    
    # Then update the derived certainty score
    db.execute("""
        UPDATE causal_mappings
        SET certainty = bayesian_update(certainty, use_count, %s),
            use_count = use_count + 1,
            last_updated = NOW()
        WHERE initiating_action = %s
        AND resulting_receptor = %s
    """, [outcome_match, action, receptor])
```

Every observation is written to the experience log before updating any derived state. The log write is the canonical record. The certainty update is a derived view.

---

## What This Enables

### Legibility

The mental model can be inspected directly at any time. You can read what the organism believes — which actions it thinks cause which receptor states, with what certainty, over what timeframe. This is the transparency the field has been seeking from neural networks for years. It isn't available there because the weights are opaque. It is available here because the knowledge is explicit.

### Targeted Updating

A new observation updates specific rows. The transformer doesn't retrain. Learning and inference are fully separated. The organism can update its belief about one specific action-receptor relationship without disturbing everything else it knows.

### Recoverability

Because every derived component is a view over the append-only experience log, any component can be reset and recomputed without losing the organism's accumulated experience. The log is what persists. Everything else is derivable from it. See the companion document *Observability and Recovery* for the full diagnostic and intervention protocol.

### Knowledge Transfer

One organism's mental model can seed another's directly — not through language, not through training on text, but through copying rows from one database to another. Cultural transmission expressed as database replication. The receiving organism inherits actual cause-effect mappings, not compressed statistical patterns. Encoder weights and action family assignments are inheritable by the same mechanism, seeding the offspring's retrieval geometry with the parent's learned structure.

### Direct Comparison

Two organisms' mental models can be compared row by row. Where do their certainties diverge? Which mappings does one have that the other lacks? What does one organism know about navigating pain fields that the other hasn't learned yet? These questions become answerable with a JOIN.

### Variable Structure Across Organisms

Different organisms, shaped by different environments and selection pressures, will accumulate different mental models. One organism's model might be sparse and tree-like — a few deep chains through a stable environment. Another's might be dense and graph-like — many interconnected mappings through a volatile one. The structure emerges from experience rather than being imposed by architecture. The database schema is the same. The data that fills it is not.

---

## The Relationship to the Transformer

The transformer is the inference engine. The mental model is the knowledge base. They are distinct components with distinct roles.

The transformer is trained to do one thing well: given the current receptor state and a set of relevant mappings retrieved from the database, produce the muscle activations that best match the predicted optimal action. It doesn't need to store knowledge. It needs to be fast and accurate at using knowledge that's already been retrieved.

The contrastive encoder sits upstream of retrieval — not inside the transformer, not inside the knowledge base, but in the retrieval layer that connects them. It is the one small component that trains on a gradient. It shapes the geometry of context space so that similarity search returns mappings that are genuinely relevant. The transformer still receives already-retrieved knowledge; it still does not carry the weight of everything the organism has ever learned.

This is a division of labor that makes all components better. The transformer stays small and fast. The knowledge base stays legible and editable. The encoder stays small and targeted. The organism becomes inspectable at every layer.

---

## The Grounding Guarantee

Every chain in the mental model, however long, terminates in a receptor state. A chain can be arbitrarily abstract — imagined movements, internal cognitive operations, hypothetical future states — but it always cashes out in something that fires. That is the tether that current AI lacks.

Current language models predict text about pain, borrowed from beings who had it. This architecture predicts receptor states from a history of actually having them. Same transformer architecture. Grounded corpus.

The mental model is where the grounding lives. Outside the weights, explicit, legible, and tethered to the organism's own experience of navigating a world that pushes back.

The experience log is where that history lives — immutable, append-only, the one record that no learned process can overwrite. Everything the organism knows is a view over what it has done and what happened as a result.

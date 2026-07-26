# Conflation as the Engine of Vocabulary Growth

## The Prediction (T108)

You cannot coin a term for a distinction until you detect you've been collapsing it. Conflation fires on insufficient resolution in the organism's own topology — "I've been treating two different things as one." This makes conflation the driver of productive language: the capacity to generate new terms rather than just deploy inherited ones.

Metacognition + conflation should be prerequisites to productive language.

## The Evidence So Far

### Conflation is a prerequisite to depth

- Knockout experiment: epistemic_strategy 0/15 without conflation (T107 confirmed)
- Convergence result: conflation predicted theoretically, evolved independently in both seeds (T94)
- Conflation appears at gen 17 (seed 99), gen 27-28 (seed 42) — always before deep canopy receptors

### Conflation is now a live channel

The episode-level receptor bank includes `conflation_ep` — detecting divergent pairs in the engine store (entries under the same action hash with cosine < 0.3 between their deltas). The organism can sense in real time that it's conflating.

## The Untested Prediction

**Conflation -> vocabulary growth:** After conflation activates, the organism should show:
1. Increased naming behavior (more distinct emission patterns)
2. New token types (tokens it didn't use before conflation)
3. Tokens that correlate with the conflated dimension (the new term points to the newly-distinguished feature)

**Without conflation -> no new terms:** An organism that deploys inherited vocabulary but has conflation knocked out should NOT coin new terms. It uses the tokens it was taught but doesn't generate novel ones.

## Test Design

### Test A: Conflation onset and emission diversity

Track two time series across generations:
1. `conflation_ep` activation level
2. Emission diversity (number of distinct token patterns used)

The prediction: emission diversity increases AFTER conflation activates, not before. The correlation should be lagged — conflation leads, vocabulary follows.

### Test B: Knockout

Run two conditions:
1. Full topology — conflation available
2. Conflation knocked out — all other receptors available

Measure emission diversity at generation 20 in both conditions. The prediction: condition 1 has higher diversity.

### Test C: Token-distinction mapping

When the organism coins a new emission pattern after conflation activates, check whether the new pattern correlates with the dimension that conflation detected. If the organism was conflating temperature and chemical (treating both as "environmental stressor"), the new token should differentiate between high-temperature and high-chemical contexts.

This requires the multi-agent architecture from item 28 (language emergence design). In the current single-organism setup, emission_bits go to the NPC but the NPC doesn't evolve. The test needs co-evolving organisms that can receive and respond to novel tokens.

## Connection to Human Data

### Lexical differentiation follows distinction-detection

In child language development, vocabulary growth accelerates precisely when the child begins making distinctions that adults make. "Dog" covers all four-legged animals until the child notices dogs and cats are different — then "cat" appears. The conflation receptor fires first (these are different), the term follows.

### Expert vocabulary grows with domain expertise

Domain experts have larger vocabularies than novices not because they learned more words, but because they make more distinctions. A sommelier has 50 words for wine flavors because they detect 50 distinctions that a novice conflates into "tastes good / tastes bad." Each conflation-detection event produces a new term.

### The Pirahã case again

The Pirahã have no exact count words. The framework says this is because the counting receptor requires naming as a prerequisite (number column, item 27). But it also predicts: the Pirahã should have vocabulary for every distinction their topology DOES make. They don't lack vocabulary for numbers because they lack language — they lack vocabulary for numbers because they lack the number-distinction detector. Their Umwelt has approximate magnitude but not exact enumeration. Their vocabulary matches their topology exactly.

## Current Status

- Conflation is a live channel (episode_receptors.py, index 76)
- Emission diversity is measurable from the action vector (emission_bits)
- The single-organism correlation test (A) is runnable now
- The knockout test (B) requires conflation in the topology bias (already supported)
- The token-distinction mapping test (C) requires multi-agent architecture

# Manifestation Environment Enhancements

## What This Is

The genome project specifies 224 receptors across 24 families, each with a manifestation YAML describing concrete survival scenarios where the receptor should fire. A gap analysis found 18 categories of environmental features described in these manifestations but missing from the environment code. This work enhanced the real environment files to provide those features.

## What Changed

### environment_tiers.py — 9 new classes, features across tiers 2-5

**Tier 2 (Social Cognition):**

- `InternalBarrier` — line segments that block line-of-sight and attenuate field propagation. 2-4 barriers generated per environment. Creates perceptual asymmetries between organism and NPCs. `has_line_of_sight(ax, ay, bx, by)` is the shared primitive.

- `NPCBeliefState` — tracks what each NPC has observed and when. NPCs steer toward *believed* source positions instead of true gradient. When sources move behind barriers, NPC beliefs become stale. `knowledge_staleness` and `belief_divergence` are synced to the NPC object each step for the organism to sense via `crossmodal_channels[0:2]`.

- `NPCActionRecorder` — records last 20 (x, y, heading) tuples per NPC. `trajectory_similarity(org_trajectory)` returns [0,1] score comparing NPC and organism movement patterns, exposed via `audio_channels[9]`.

- `NPCTrustTracker` — per-NPC signal-outcome reliability. Records NPC emission signals, then checks whether outcomes matched. Trust scores update from outcomes recorded in `Organism.step()`. Exposed via `crossmodal_channels[2]`.

- NPC vocabulary system — each NPC at tier 5+ maintains an emission-to-source mapping. Cooperative NPCs emit consistent patterns near sources. Deceptive NPCs emit wrong patterns. Vocabulary applies after profiled stepping so it overrides default behavior only when the NPC is near a mapped source.

**Tier 3 (Object Manipulation):**

- `DepletableSource` — wraps a `FieldSource` with `spoilage_rate` (passive decay) and `depletion_rate` (when organism is nearby harvesting). Sources go to zero when depleted. 1-2 endorphin sources wrapped at tier 3. State exposed via `audio_channels[6:9]`.

- `Container(MovableObject)` — holds other objects inside it. `contents`, `intake_radius`, `capacity`, `is_open`. Objects only inserted when the organism is near both the object and the open container. State exposed via `audio_channels[3:6]`.

**Tier 4 (Hidden Variables):**

- `TerrainZone` — circular region with `elevation`, `substrate` (ground/water/mud/rock), `shelter` (blocks predator pain), `movement_cost` (multiplies fatigue rate). 3 zones generated at tier 4. Water zones reduce temperature and disrupt scent trails. Shelter zones attenuate predator sweep pain. Movement cost exposed via `audio_channels[10]`.

**Tier 5 (Strategic Social):**

- `SeasonalCycle` — long-period (1500-2500 step) modulator affecting endorphin, pain, and temperature. Creates the temporal structure that forces planning across episodes. Phase exposed via `audio_channels[11]`.

- `ScentTrail` — decaying chemical deposits left by moving NPCs. Each deposit has position and intensity that decays per step. Contributes to `get_chemical_values()` (not endorphin). Water terrain zones clear trails on contact.

### environment.py — Enhanced empathy + expanded signals + observation wiring

**Empathic resonance (Organism.step):**
- Multi-channel: NPC erraticism + distress emission pattern (1,1,x,x)
- Calm contagion: low erraticism reduces organism stress
- Safety-gated: empathic component scaled by `(1.0 - mean_pain)` to separate resonance from shared threat
- NPC frustration: stale beliefs amplify empathic aversion

**Expanded NPC signals:**
- 4 new signal types: approach, alert, food, danger (plus generic handler for any non-zero emission)
- NPCs respond to approach signals by moving toward sender, alert signals by increasing erraticism

**Observation channel population:**
- `crossmodal_channels[0:1]`: Peacock block — rich structured outcome-irrelevant signal (P4/P8 permanent negative control)
- `crossmodal_channels[2]`: Echo — organism emission reflected from barriers at rigid 2d/c lag (P16 like-me composition test)
- `audio_channels[0:3]`: tool reach, force multiplier, tool contact count (set by PhysicsWorld via run_generation_rich)
- `audio_channels[3:6]`: nearest container proximity, contents ratio, lid state
- `audio_channels[6:9]`: nearest depletable source charge, quality, live count
- `audio_channels[9]`: Boom — pain events at distance produce delayed audio at distance/c (P19 ranging)
- `audio_channels[10]`: terrain movement cost
- `audio_channels[11]`: seasonal cycle phase
- `visual_channels[0:12]`: barrier proximity and direction vectors (4 barriers x 3 values)

**Terrain cost wiring:**
- `_terrain_movement_cost` read from environment, multiplies `FATIGUE_RATE` per limb

**Trust outcome recording:**
- `trust_tracker.record_outcome()` called each step with reward delta, so trust scores actually update

**Position history:**
- `_position_history` (deque, 20 entries) tracks organism (x, y, heading) for mimicry comparison

### physics_world.py — Tool mediation + shape variety + persistence

**Shape variety:**
- `RigidObject` now accepts `shape_type` ('circle', 'segment', 'box') and `length`. Default is 'circle' (backward compatible). One segment-shaped "stick" tool created per environment alongside circular objects.

**Tool mediation:**
- `get_tool_reach()` — max reach extension from gripped segment objects (normalized 0-1)
- `get_tool_force_multiplier()` — force amplification from gripped mass, capped at 3x
- `get_tool_extended_tips()` — limb tip positions extended by gripped object geometry
- `check_tool_interactions()` — detects gripped objects interacting with field sources, returns reward delta
- `detect_constructed_structures()` — detects when 2+ objects are arranged in a line near a pain source (functional wall)

**Barrier integration:**
- Environment barriers added as pymunk static segments with COLLISION_TYPE_WALL — physically blocks organism movement

**State persistence:**
- `save_world_state()` includes depletable source intensities and container states
- `restore_world_state()` recovers depletable source and container states across episodes

**Integration:**
- `run_generation_rich()` in deep_time_overnight.py calls `check_tool_interactions()`, sets `_tool_reach`, `_tool_force_mult`, `_tool_contact_count` on organism each step, adds `tool_reward * 0.2` to episode reward

### abstract_env.py — Expanded causal templates

5 new templates added to `CAUSAL_TEMPLATES`:
- `diamond`: A→B, A→C, B→D, C→D (4 nodes, convergent at D)
- `long_chain`: A→B→C→D→E (5 nodes, sequential)
- `multi_confounder`: H1→A, H1→B, H2→B, H2→C (5 nodes, 2 hidden confounders)
- `intervention_test`: A→B→C + A→C with intervention target at B
- `probabilistic_chain`: A→B (70%), B→C (50%) — stochastic edge activation

**Dynamic zone creation:**
- `_create_zones()` now derives node set from template edges instead of hardcoding A/B/C. Supports any number of nodes with adaptive spacing.

**Probabilistic activation:**
- Templates can include a `probabilistic` dict mapping edge keys to activation probabilities. When a zone is consumed and triggers a downstream zone, the activation fires with the specified probability instead of always.

## Observation Budget

All 27 previously-empty observation slots now carry environment state:

| Channels | Count | Content |
|---|---|---|
| audio[0:3] | 3 | Tool state (reach, force, contacts) |
| audio[3:6] | 3 | Container state (proximity, contents, lid) |
| audio[6:9] | 3 | Depletable resource state (charge, quality, live count) |
| audio[9] | 1 | Mimicry trajectory similarity |
| audio[10] | 1 | Terrain movement cost |
| audio[11] | 1 | Seasonal cycle phase |
| visual[0:12] | 12 | Barrier proximity and direction (4 barriers x 3) |
| crossmodal[0:3] | 3 | NPC staleness, belief divergence, trust score |
| **Total** | **27** | **All slots used** |

## Backward Compatibility

- Tier 0 and Tier 1: identical behavior. All new features begin at tier 2+.
- New NPC attributes (`knowledge_staleness`, `belief_divergence`, `belief_navigation`, `_belief_state_ref`) default to 0/False/None. `getattr` guards throughout.
- Previously-zero observation channels (audio, visual, crossmodal) now carry values at tier 2+. Models trained on old data saw constant zeros in these positions and learned to ignore them.
- OBS_DIM remains 400. `compute_obs_indices()` unchanged.
- `RigidObject` default `shape_type='circle'` — existing code unchanged.
- Base `Environment.__init__` unchanged except `_position_history` deque added to Organism.

## Files Modified

| File | Changes |
|---|---|
| `environment_tiers.py` | 9 new classes, _add_tier2-5 enhanced, get_field_values/get_temperature_values/get_chemical_values updated, step_tier updated, has_line_of_sight added |
| `environment.py` | Empathic resonance enhanced, 6 NPC signal types, audio/visual/crossmodal channels populated, terrain cost applied to fatigue, trust outcome recorded, position history tracked |
| `physics_world.py` | RigidObject shape variety, 5 tool mediation methods, barrier pymunk integration, state persistence for depletable sources and containers |
| `abstract_env.py` | 5 new causal templates (4-5 nodes), dynamic zone creation, probabilistic edge activation |
| `deep_time_overnight.py` | env_factory parameter on run_overnight and run_generation_rich, tool interaction calls and organism state sync, tool_reward added to episode reward |
| `train.py` | env_factory parameter on generate_training_data |

## What This Unlocks

The 18 gaps addressed cover the environmental infrastructure needed for all 224 manifestation scenarios across stages 1-12:

| Tier | Features | Receptor Families Unlocked |
|---|---|---|
| 2 | Barriers, NPC beliefs, mimicry recording, trust tracking | Social (perspective_taking, belief_attribution, theory_of_mind, deception_detection, trust, mimicry) |
| 3 | Containers, depletable sources, tool shapes | Interaction (tool_use, grip_affordance, composite_affordance), Observation (absence_observation) |
| 4 | Terrain zones | Agency (spatial_reasoning), Proprioception (resistance, body_boundary) |
| 5 | Seasonal cycles, scent trails, NPC vocabulary | Formalization (theory_formation), Language (naming, referential_grounding), Bridging (translation) |
| 6 | State persistence, functional structures | Self-Augmentation (niche_construction), Environmental Augmentation (developmental_environment_engineering) |
| 7 | Expanded causal templates, probabilistic causation | Causality (causal_graph_reasoning, probabilistic_causation), Epistemic (counterfactual_reasoning) |
| Base | Enhanced empathy, expanded signals | Regulatory (empathy, social_coregulation), Social (emotional_intelligence) |

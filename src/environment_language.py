"""EL-0 — the environment-language interpreter (T153, phase 0).

The environment-organism's version-0 cognitive layer: a deterministic
seed grammar (simplified English: SVO, modifiers, spatial/temporal
phrases, conditionals) whose sentences DESCRIBE and RECONSTRUCT the
descriptive stratum of a TieredEnvironment world (tiers 0-4, the rich
stack). For this organism, saying is doing: interpret(corpus) builds the
world the corpus describes, using the real environment classes — the
decree stratum (physics: how fields propagate, how triggers fire, how
NPCs move) stays in code and is NOT expressible or refinable in the
language. The language addresses configuration — what exists, where,
with which parameters — which is exactly the layer every hand-written
tier design and manifestation recipe has been editing by hand.

Constitutional properties (environment_organism_requirements.md):
- Decree/description split: sentences never encode physics, only the
  descriptive configuration physics acts on.
- Deterministic, auditable, no LLMs, no imported corpora: the grammar is
  a fixed template set; floats round-trip at full precision.
- The etymology ledger is seeded: every word carries its origin receipts
  (the tier feature it names, and the manifestation-registry families
  whose worlds need it). After EL-0, lexical moves are emergent only.

Acceptance (el0_acceptance.py): round-trip fixed point
(describe(interpret(describe(env))) == describe(env)) at tiers 0-4, and
zero behavioral regression — a real Organism run on the described-then-
reinterpreted world produces bit-identical observations and rewards to
the directly built world under the paired runtime protocol.
"""

import json
import os
import re
import glob

import numpy as np

from environment import Environment, FieldSource, ResponsiveObject, NPC
from environment_tiers import (
    TieredEnvironment, PulsingSource, CausalTrigger, PredatorEvent,
    MovableObject, HiddenVariable, StochasticHiddenVariable,
    CrossModalSource, InternalBarrier, NPCBeliefState, NPCActionRecorder,
    NPCTrustTracker, DepletableSource, Container, TerrainZone,
)

MANIFESTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'genome_project', 'manifestations')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results')


def _f(x):
    """Full-precision float word: repr round-trips exactly."""
    return repr(float(x))


NUM = r'(-?\d+(?:\.\d+)?(?:e-?\d+)?)'


# ---------------------------------------------------------------------------
# The seed lexicon and its etymology
# ---------------------------------------------------------------------------

SEED_LEXICON = {
    # nouns (feature words)
    'source':    ('noun', 'field structure'),
    'object':    ('noun', 'responsive/movable structure'),
    'npc':       ('noun', 'inhabitant-adjacent agent'),
    'barrier':   ('noun', 'spatial occlusion'),
    'predator':  ('noun', 'periodic hazard'),
    'ring':      ('noun', 'gated reward structure'),
    'container': ('noun', 'holding structure'),
    'zone':      ('noun', 'terrain region'),
    'state':     ('noun', 'hidden variable'),
    'pair':      ('noun', 'compound coupling'),
    'world':     ('noun', 'the whole'),
    # modality adjectives
    'pain': ('adj', 'modality'), 'endorphin': ('adj', 'modality'),
    'heat': ('adj', 'modality'), 'cold': ('adj', 'modality'),
    'chemical': ('adj', 'modality'),
    # verbs (what structures do)
    'drifts':    ('verb', 'source motion'),
    'pulses':    ('verb', 'tier1 pulsing'),
    'causes':    ('verb', 'tier1 causal trigger'),
    'yields':    ('verb', 'tier1 conditional trigger'),
    'roams':     ('verb', 'tier2 npc'),
    'blocks':    ('verb', 'tier2 barrier'),
    'sweeps':    ('verb', 'tier2 predator'),
    'deplete':   ('verb', 'tier2 resource pressure'),
    'rests':     ('verb', 'tier3 movable'),
    'guards':    ('verb', 'tier3 gated reward'),
    'holds':     ('verb', 'tier3 container'),
    'spoils':    ('verb', 'tier3 depletable'),
    'couple':    ('verb', 'tier3 compound pair'),
    'cycles':    ('verb', 'tier4 hidden variable'),
    'wanders':   ('verb', 'tier4 stochastic hidden'),
    'modulates': ('verb', 'tier4 cross-modal'),
    'covers':    ('verb', 'tier4 terrain'),
    'responds':  ('verb', 'base responsive object'),
    'swaps':     ('verb', 'tier4 transfer switch'),
}


def load_registry_families():
    """The seed lexicon's ancestry: the manifestation registry's family
    inventory (the '18 feature categories')."""
    fams = {}
    for path in sorted(glob.glob(os.path.join(MANIFESTATION_DIR, '*.yaml'))):
        try:
            with open(path, encoding='utf-8') as fh:
                head = fh.read(400)
            m = re.search(r'^family:\s*(\S+)', head, re.M)
            if m:
                fams.setdefault(m.group(1), 0)
                fams[m.group(1)] += 1
        except OSError:
            continue
    return fams


def write_etymology_ledger(path=None):
    """Seed the etymology ledger: one append-only record per seed word,
    citing its tier-feature receipt and the registry ancestry. EL-0 seeds
    it; EL-2's lexical moves will append to it."""
    path = path or os.path.join(RESULTS_DIR, 'el0_etymology.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fams = load_registry_families()
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({
            'event': 'seed_ancestry',
            'registry_families': fams,
            'registry_yaml_count': int(sum(fams.values())),
        }) + '\n')
        for word, (part, receipt) in sorted(SEED_LEXICON.items()):
            fh.write(json.dumps({
                'event': 'word_seeded', 'word': word, 'part': part,
                'origin': 'seed', 'receipts': [receipt],
            }) + '\n')
    return path


# ---------------------------------------------------------------------------
# describe: world -> corpus (the environment reading itself aloud)
# ---------------------------------------------------------------------------

def _say_field_source(kind, s):
    return (f"a {kind} source at ({_f(s.cx)}, {_f(s.cy)}) drifts by "
            f"({_f(s.ax)}, {_f(s.ay)}) with frequency ({_f(s.omega_x)}, "
            f"{_f(s.omega_y)}) phase ({_f(s.phi_x)}, {_f(s.phi_y)}) spread "
            f"{_f(s.sigma)} intensity {_f(s.intensity)}")


def describe(env):
    """Emit the seed-grammar corpus for a tier<=4 world, in canonical
    order. Every sentence names one descriptive fact; together they are
    the world's configuration, spoken."""
    if getattr(env, 'seasonal_cycle', None) is not None:
        raise NotImplementedError('EL-0 covers tiers 0-4')
    tier = getattr(env, 'tier', 0)
    if tier > 4:
        raise NotImplementedError('EL-0 covers tiers 0-4')

    c = [f"this world holds tier {tier} structure"]

    for kind, lst in (('pain', env.pain_sources),
                      ('endorphin', env.endorphin_sources),
                      ('heat', env.heat_sources),
                      ('cold', env.cold_sources),
                      ('chemical', env.chemical_sources)):
        for s in lst:
            c.append(_say_field_source(kind, s))

    for o in env.responsive_objects:
        bits = ''.join(str(int(b)) for b in o.trigger_signal)
        c.append(
            f"an object at ({_f(o.cx)}, {_f(o.cy)}) drifts by ({_f(o.ax)}, "
            f"{_f(o.ay)}) with frequency ({_f(o.omega_x)}, {_f(o.omega_y)}) "
            f"phase ({_f(o.phi_x)}, {_f(o.phi_y)}) and responds to signal "
            f"{bits} with {o.response_type} for {int(o.response_duration)} "
            f"steps within {_f(o.trigger_range)}")

    if tier >= 1:
        for i, p in enumerate(env.pulsing_sources):
            c.append(f"the pain source {i} pulses with period "
                     f"{int(p.pulse_period)} and amplitude "
                     f"{_f(p.pulse_amplitude)}")
        for t in env.causal_triggers:
            c.append(
                f"touching ({_f(t.trigger_x)}, {_f(t.trigger_y)}) within "
                f"{_f(t.trigger_radius)} causes {t.effect_type} at "
                f"({_f(t.effect_x)}, {_f(t.effect_y)}) after {int(t.delay)} "
                f"steps with intensity {_f(t.effect_intensity)} for "
                f"{int(t.effect_duration)} steps")
        for ct in getattr(env, 'conditional_triggers', []):
            c.append(
                f"touching ({_f(ct['trigger'][0])}, {_f(ct['trigger'][1])}) "
                f"within {_f(ct['radius'])} yields "
                f"{ct['high_energy_type']} at ({_f(ct['effect'][0])}, "
                f"{_f(ct['effect'][1])}) when energy exceeds "
                f"{_f(ct['energy_threshold'])} else {ct['low_energy_type']}")

    if tier >= 2:
        for npc in env.npcs:
            c.append(
                f"a {npc.behavior_profile} npc roams from ({_f(npc.x)}, "
                f"{_f(npc.y)}) heading {_f(npc.heading)}")
        for b in env.barriers:
            caps = []
            if b.blocks_sight:
                caps.append('sight')
            if b.blocks_field:
                caps.append('field')
            if b.blocks_movement:
                caps.append('movement')
            what = ' and '.join(caps) if caps else 'nothing'
            c.append(
                f"a barrier from ({_f(b.x1)}, {_f(b.y1)}) to ({_f(b.x2)}, "
                f"{_f(b.y2)}) blocks {what} attenuating {_f(b.attenuation)}")
        for p in env.predator_events:
            c.append(f"a predator sweeps every {int(p.period)} steps for "
                     f"{int(p.duration)} steps at intensity "
                     f"{_f(p.intensity)} speed {_f(p.speed)}")
        c.append(f"resources deplete by "
                 f"{_f(env.resource_depletion_factor)} under crowding")

    if tier >= 3:
        for m in env.movable_objects:
            if isinstance(m, Container):
                continue
            c.append(f"a movable object of mass {_f(m.mass)} friction "
                     f"{_f(m.friction)} radius {_f(m.radius)} rests at "
                     f"({_f(m.x)}, {_f(m.y)})")
        if getattr(env, 'gated_reward', None):
            g = env.gated_reward
            c.append(
                f"a pain ring of radius {_f(g['ring_radius'])} intensity "
                f"{_f(g['ring_intensity'])} guards the reward "
                f"{_f(g['reward_intensity'])} at ({_f(g['x'])}, "
                f"{_f(g['y'])}) opened by object {int(g['gap_object_idx'])}")
        for a, b in getattr(env, 'compound_pairs', []):
            c.append(f"objects {int(a)} and {int(b)} couple as a pair")
        for i, d in enumerate(env.depletable_sources):
            idx = env.endorphin_sources.index(d.source)
            c.append(
                f"the endorphin source {idx} spoils at "
                f"{_f(d.spoilage_rate)} depletes at {_f(d.depletion_rate)} "
                f"season {int(getattr(d, 'seasonal_period', 0) or 0)} "
                f"swing {_f(getattr(d, 'seasonal_amplitude', 0.0) or 0.0)}")
        for k in env.containers:
            c.append(f"a container of mass {_f(k.mass)} friction "
                     f"{_f(k.friction)} at ({_f(k.x)}, {_f(k.y)}) holds "
                     f"{int(k.capacity)} within {_f(k.intake_radius)}")

    if tier >= 4:
        hv = env.hidden_variable
        c.append(f"a hidden state cycles through {int(hv.num_states)} "
                 f"states with period {int(hv.period)}")
        c.append(f"a second hidden state wanders through "
                 f"{int(env.stochastic_hidden.num_states)} states")
        c.append(f"the hidden configuration swaps at step "
                 f"{int(env.transfer_switch_step)}")
        for x in env.cross_modal_sources:
            c.append(
                f"a cross-modal source at ({_f(x.cx)}, {_f(x.cy)}) spread "
                f"{_f(x.sigma)} modulates pain "
                f"{_f(x.modality_intensities['pain'])} temperature "
                f"{_f(x.modality_intensities['temperature'])} chemical "
                f"{_f(x.modality_intensities['chemical'])}")
        for z in env.terrain_zones:
            shelter = 'sheltered' if z.shelter else 'open'
            c.append(
                f"a {shelter} {z.substrate} zone of radius {_f(z.radius)} "
                f"covers ({_f(z.cx)}, {_f(z.cy)}) at elevation "
                f"{_f(z.elevation)} costing {_f(z.movement_cost)} to cross")

    return c


# ---------------------------------------------------------------------------
# interpret: corpus -> world (saying is doing)
# ---------------------------------------------------------------------------

_P = {
    'tier': re.compile(r'^this world holds tier (\d+) structure$'),
    'source': re.compile(
        rf'^a (pain|endorphin|heat|cold|chemical) source at \({NUM}, {NUM}\)'
        rf' drifts by \({NUM}, {NUM}\) with frequency \({NUM}, {NUM}\)'
        rf' phase \({NUM}, {NUM}\) spread {NUM} intensity {NUM}$'),
    'object': re.compile(
        rf'^an object at \({NUM}, {NUM}\) drifts by \({NUM}, {NUM}\) with'
        rf' frequency \({NUM}, {NUM}\) phase \({NUM}, {NUM}\) and responds'
        rf' to signal (\d{{4}}) with (\w+) for (\d+) steps within {NUM}$'),
    'pulse': re.compile(
        rf'^the pain source (\d+) pulses with period (\d+) and amplitude'
        rf' {NUM}$'),
    'causal': re.compile(
        rf'^touching \({NUM}, {NUM}\) within {NUM} causes (\w+) at'
        rf' \({NUM}, {NUM}\) after (\d+) steps with intensity {NUM} for'
        rf' (\d+) steps$'),
    'conditional': re.compile(
        rf'^touching \({NUM}, {NUM}\) within {NUM} yields (\w+) at'
        rf' \({NUM}, {NUM}\) when energy exceeds {NUM} else (\w+)$'),
    'npc': re.compile(
        rf'^a (\w+) npc roams from \({NUM}, {NUM}\) heading {NUM}$'),
    'barrier': re.compile(
        rf'^a barrier from \({NUM}, {NUM}\) to \({NUM}, {NUM}\) blocks'
        rf' ([\w ]+) attenuating {NUM}$'),
    'predator': re.compile(
        rf'^a predator sweeps every (\d+) steps for (\d+) steps at'
        rf' intensity {NUM} speed {NUM}$'),
    'depletion': re.compile(
        rf'^resources deplete by {NUM} under crowding$'),
    'movable': re.compile(
        rf'^a movable object of mass {NUM} friction {NUM} radius {NUM}'
        rf' rests at \({NUM}, {NUM}\)$'),
    'gate': re.compile(
        rf'^a pain ring of radius {NUM} intensity {NUM} guards the reward'
        rf' {NUM} at \({NUM}, {NUM}\) opened by object (\d+)$'),
    'pair': re.compile(r'^objects (\d+) and (\d+) couple as a pair$'),
    'depletable': re.compile(
        rf'^the endorphin source (\d+) spoils at {NUM} depletes at {NUM}'
        rf' season (\d+) swing {NUM}$'),
    'container': re.compile(
        rf'^a container of mass {NUM} friction {NUM} at \({NUM}, {NUM}\)'
        rf' holds (\d+) within {NUM}$'),
    'hidden': re.compile(
        r'^a hidden state cycles through (\d+) states with period (\d+)$'),
    'stochastic': re.compile(
        r'^a second hidden state wanders through (\d+) states$'),
    'swap': re.compile(r'^the hidden configuration swaps at step (\d+)$'),
    'crossmodal': re.compile(
        rf'^a cross-modal source at \({NUM}, {NUM}\) spread {NUM} modulates'
        rf' pain {NUM} temperature {NUM} chemical {NUM}$'),
    'zone': re.compile(
        rf'^a (sheltered|open) (\w+) zone of radius {NUM} covers'
        rf' \({NUM}, {NUM}\) at elevation {NUM} costing {NUM} to cross$'),
}

_OBJ_TRIGGERS = {'endorphin': (0, 0, 1, 0), 'approach': (0, 1, 0, 1),
                 'cool': (1, 0, 0, 1)}


def interpret(corpus):
    """Build the world the corpus describes, from the real environment
    classes. Deterministic: identical corpus -> identical world."""
    env = TieredEnvironment(seed=0, tier=0)
    # Empty the shell: the corpus is the sole authority on configuration.
    env.pain_sources = []
    env.endorphin_sources = []
    env.heat_sources = []
    env.cold_sources = []
    env.chemical_sources = []
    env.responsive_objects = []
    env.cross_modal_sources = []      # tier-4 attrs the shell lacks
    env.hidden_variable = None
    env.stochastic_hidden = None
    pulse_specs = []
    depletable_specs = []

    for line in corpus:
        line = line.strip()
        if not line:
            continue
        m = _P['tier'].match(line)
        if m:
            env.tier = int(m.group(1))
            continue
        m = _P['source'].match(line)
        if m:
            kind = m.group(1)
            vals = [float(x) for x in m.groups()[1:]]
            src = FieldSource(*vals)
            getattr(env, f'{kind}_sources').append(src)
            continue
        m = _P['object'].match(line)
        if m:
            g = m.groups()
            bits = tuple(int(ch) for ch in g[8])
            env.responsive_objects.append(ResponsiveObject(
                cx=float(g[0]), cy=float(g[1]), ax=float(g[2]),
                ay=float(g[3]), omega_x=float(g[4]), omega_y=float(g[5]),
                phi_x=float(g[6]), phi_y=float(g[7]),
                trigger_signal=bits, response_type=g[9],
                trigger_range=float(g[11]),
                response_duration=int(g[10])))
            continue
        m = _P['pulse'].match(line)
        if m:
            pulse_specs.append((int(m.group(1)), int(m.group(2)),
                                float(m.group(3))))
            continue
        m = _P['causal'].match(line)
        if m:
            g = m.groups()
            env.causal_triggers.append(CausalTrigger(
                (float(g[0]), float(g[1])), (float(g[4]), float(g[5])),
                int(g[6]), effect_type=g[3],
                effect_intensity=float(g[7]), effect_duration=int(g[8]),
                trigger_radius=float(g[2])))
            continue
        m = _P['conditional'].match(line)
        if m:
            g = m.groups()
            if not hasattr(env, 'conditional_triggers') or \
                    env.conditional_triggers is None:
                env.conditional_triggers = []
            env.conditional_triggers.append({
                'trigger': (float(g[0]), float(g[1])),
                'effect': (float(g[4]), float(g[5])),
                'radius': float(g[2]),
                'energy_threshold': float(g[6]),
                'high_energy_type': g[3], 'low_energy_type': g[7]})
            continue
        m = _P['npc'].match(line)
        if m:
            npc = NPC()
            npc.reset(None)
            npc.x, npc.y = float(m.group(2)), float(m.group(3))
            npc.heading = float(m.group(4))
            npc.behavior_profile = m.group(1)
            npc.reciprocity_score = 0.0
            npc.org_preferred_source = None
            npc.belief_navigation = True
            npc.knowledge_staleness = 0.0
            npc.belief_divergence = 0.0
            env.npcs.append(npc)
            continue
        m = _P['barrier'].match(line)
        if m:
            g = m.groups()
            caps = g[4]
            env.barriers.append(InternalBarrier(
                float(g[0]), float(g[1]), float(g[2]), float(g[3]),
                blocks_sight=('sight' in caps),
                blocks_field=('field' in caps),
                blocks_movement=('movement' in caps),
                attenuation=float(g[5])))
            continue
        m = _P['predator'].match(line)
        if m:
            env.predator_events.append(PredatorEvent(
                period=int(m.group(1)), duration=int(m.group(2)),
                intensity=float(m.group(3)), speed=float(m.group(4))))
            continue
        m = _P['depletion'].match(line)
        if m:
            env.resource_depletion_factor = float(m.group(1))
            continue
        m = _P['movable'].match(line)
        if m:
            g = m.groups()
            env.movable_objects.append(MovableObject(
                float(g[3]), float(g[4]), mass=float(g[0]),
                friction=float(g[1]), radius=float(g[2])))
            continue
        m = _P['gate'].match(line)
        if m:
            g = m.groups()
            env.gated_reward = {
                'x': float(g[3]), 'y': float(g[4]),
                'ring_radius': float(g[0]), 'ring_intensity': float(g[1]),
                'reward_intensity': float(g[2]),
                'gap_object_idx': int(g[5])}
            continue
        m = _P['pair'].match(line)
        if m:
            if not hasattr(env, 'compound_pairs') or \
                    env.compound_pairs is None:
                env.compound_pairs = []
            env.compound_pairs.append((int(m.group(1)), int(m.group(2))))
            continue
        m = _P['depletable'].match(line)
        if m:
            depletable_specs.append((int(m.group(1)), float(m.group(2)),
                                     float(m.group(3)), int(m.group(4)),
                                     float(m.group(5))))
            continue
        m = _P['container'].match(line)
        if m:
            g = m.groups()
            env.containers.append(Container(
                float(g[2]), float(g[3]), mass=float(g[0]),
                friction=float(g[1]), capacity=int(g[4]),
                intake_radius=float(g[5])))
            continue
        m = _P['hidden'].match(line)
        if m:
            env.hidden_variable = HiddenVariable(
                period=int(m.group(2)), num_states=int(m.group(1)))
            continue
        m = _P['stochastic'].match(line)
        if m:
            env.stochastic_hidden = StochasticHiddenVariable(
                num_states=int(m.group(1)), rng=env.rng)
            continue
        m = _P['swap'].match(line)
        if m:
            env.transfer_switch_step = int(m.group(1))
            env.config_swapped = False
            continue
        m = _P['crossmodal'].match(line)
        if m:
            g = m.groups()
            env.cross_modal_sources.append(CrossModalSource(
                float(g[0]), float(g[1]), float(g[2]),
                {'pain': float(g[3]), 'temperature': float(g[4]),
                 'chemical': float(g[5])}))
            continue
        m = _P['zone'].match(line)
        if m:
            g = m.groups()
            env.terrain_zones.append(TerrainZone(
                float(g[3]), float(g[4]), float(g[2]),
                elevation=float(g[5]), substrate=g[1],
                shelter=(g[0] == 'sheltered'),
                movement_cost=float(g[6])))
            continue
        raise ValueError(f'EL-0: unparseable sentence: {line!r}')

    # Reconstruct derived runtime structure the sentences imply
    if env.tier >= 1:
        env.pulsing_sources = []
        env.anticipatory_phases = []
        for idx, period, amp in pulse_specs:
            src = env.pain_sources[idx]
            env.pulsing_sources.append(PulsingSource(
                src.cx, src.cy, src.ax, src.ay, src.omega_x, src.omega_y,
                src.phi_x, src.phi_y, src.sigma, src.intensity,
                pulse_period=period, pulse_amplitude=amp))
            env.anticipatory_phases.append({
                'source': src, 'period': period,
                'reward_phase_start': 0.7, 'reward_phase_end': 0.9})
    if env.tier >= 2:
        for i in range(len(env.npcs)):
            npc_id = f'npc_{i}'
            belief = NPCBeliefState()
            env.npc_beliefs[npc_id] = belief
            env.npcs[i]._belief_state_ref = belief
            env.npc_recorders[npc_id] = NPCActionRecorder()
        env.trust_tracker = NPCTrustTracker()
    if env.tier >= 3:
        for idx, spoil, deplete, period, swing in depletable_specs:
            kwargs = {'spoilage_rate': spoil, 'depletion_rate': deplete}
            if period:
                kwargs['seasonal_period'] = period
            if swing:
                kwargs['seasonal_amplitude'] = swing
            env.depletable_sources.append(DepletableSource(
                env.endorphin_sources[idx], **kwargs))
        # containers also participate in the movable list downstream
    return env

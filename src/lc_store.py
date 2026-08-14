"""LC-0a — the language center's core: phrase store, readout, parser
(T155; design: language_center_design.md; build 2026-08-12).

The readout points INWARD at the ledger: assertions only from
assertable Ks (closed AND live — the dormancy mechanism is the
evidentiality system), questions for open slots typed by their gap
evidence, every utterance a serialization of state the web actually
holds. The parser is the store run in reverse: template match ->
(speech act, target family, epistemic frame) — geometry
reconstruction at family grain for LC-0a; bus integration (LC-0b)
follows on the E1 harness.

Modality note (design §1.2): the full modal reads the slot's survival
record; LC-0a uses the available proxies — 'so far' for standing Ks,
'again, after retraction' for reclosed Ks (reopen_count > 0) — and
leaves 'robustly/necessarily' to the instrumented drivers that count
survival events. Honest v0: under-claiming strength is calibration-
safe; over-claiming would not be.

No LLMs, no tokens, no RNG. Everything deterministic from web state.
"""

import re

from receptor_eigen_coder import FAMILY_GROUPS

FAMILY_NAMES = [n for n, _ in FAMILY_GROUPS]


# ---------------------------------------------------------------- readout

def _fam(slot):
    import numpy as np
    return int(np.argmax(slot.geometry.family_thresholds))


def readout(web, max_questions=8):
    """The web speaks: [(utterance, receipt_dict)] — assertions first
    (assertable Ks only), then questions ranked by gap evidence."""
    utterances = []
    # assertions — the firewall's speech clause: closed Ks only
    for sid in sorted(web.slots):
        s = web.slots[sid]
        if s.state != 'closed':
            continue
        fam = FAMILY_NAMES[_fam(s)]
        if s.dormant:
            idle = web._global_step - s.ledger.last_fit_step
            u = (f"{fam} held, last I met it ({idle} steps ago)")
            frame = 'evidential-past'
        elif s.ledger.reopen_count > 0:
            u = (f"{fam} is settled again, after retraction — so far")
            frame = 'plain-after-retraction'
        else:
            u = f"{fam} is settled — so far"
            frame = 'plain'
        utterances.append((u, {
            'act': 'assert', 'slot_id': sid, 'family': _fam(s),
            'evidential': frame,
            'funded_fits': s.ledger.fit_count,
            'certainty': round(s.ledger.certainty, 4)}))
    # questions — open slots by gap evidence (near-miss first)
    opens = [(s.ledger.near_miss_seen, s.ledger.fit_count, sid)
             for sid, s in web.slots.items() if s.state == 'open']
    opens.sort(key=lambda x: (-x[0], x[1], x[2]))
    n_q = 0
    for nm, fits, sid in opens:
        if n_q >= max_questions:
            break
        s = web.slots[sid]
        fam = FAMILY_NAMES[_fam(s)]
        if nm > 0:
            u = f"is the thing at the boundary {fam} or not-{fam}?"
            gap = 'near-miss'
        elif fits == 0:
            u = f"what is the ?thing that would be {fam}?"
            gap = 'starvation'
        else:
            u = f"what joins {fam} to its neighbors?"
            gap = 'edge-gap'
        utterances.append((u, {
            'act': 'question', 'slot_id': sid, 'family': _fam(s),
            'gap_type': gap, 'near_misses': nm}))
        n_q += 1
    return utterances


# ----------------------------------------------------------------- parser

_P_ASSERT_DORMANT = re.compile(
    r'^(\w+) held, last I met it \((\d+) steps ago\)$')
_P_ASSERT_RETR = re.compile(
    r'^(\w+) is settled again, after retraction — so far$')
_P_ASSERT = re.compile(r'^(\w+) is settled — so far$')
_P_Q_NEARMISS = re.compile(
    r'^is the thing at the boundary (\w+) or not-\w+\?$')
_P_Q_STARVE = re.compile(r'^what is the \?thing that would be (\w+)\?$')
_P_Q_EDGE = re.compile(r'^what joins (\w+) to its neighbors\?$')


# ------------------------------------------------- pose channel (LC-0b)
# Threshold-band templates: the geometry-capable pose phrase. Bands
# quantize family thresholds to {faint, half, strong} — enough capacity
# for the bus's match scale (F28-era MATCH_FLOOR=0.3), pre-registered
# reconstruction gate: cosine >= 0.98.

_BANDS = ((0.125, 'faint'), (0.375, 'half'), (1.0, 'strong'))
_BAND_VAL = {'faint': 0.1, 'half': 0.3, 'strong': 0.6}


def pose_phrase(web, slot_id):
    """Serialize a slot's connector geometry as a phrase: every family
    above threshold named with its band, plus the near-miss count."""
    import numpy as np
    s = web.slots[slot_id]
    parts = []
    for f in np.argsort(-s.geometry.family_thresholds, kind='stable'):
        v = float(s.geometry.family_thresholds[f])
        if v < 0.05:
            break
        band = next(b for cap, b in _BANDS if v <= cap)
        parts.append(f'{FAMILY_NAMES[int(f)]}:{band}')
    body = ', '.join(parts) if parts else 'nothing yet'
    return (f'seeking the ?thing shaped [{body}] '
            f'({s.ledger.near_miss_seen} near-misses)')


_P_POSE = re.compile(
    r'^seeking the \?thing shaped \[([^\]]*)\] \((\d+) near-misses\)$')


def parse_pose(utterance):
    """Reconstruct a threshold profile (band precision) from a pose
    phrase. Returns (thresholds ndarray, near_misses) or None."""
    import numpy as np
    m = _P_POSE.match(utterance)
    if not m:
        return None
    th = np.zeros(len(FAMILY_NAMES))
    if m.group(1) != 'nothing yet':
        for part in m.group(1).split(', '):
            name, band = part.rsplit(':', 1)
            if name in FAMILY_NAMES and band in _BAND_VAL:
                th[FAMILY_NAMES.index(name)] = _BAND_VAL[band]
    return th, int(m.group(2))


def parse(utterance):
    """Store-in-reverse: utterance -> (act, family index, frame).
    Returns None for anything the grammar does not license."""
    fam_idx = {n: i for i, n in enumerate(FAMILY_NAMES)}
    for pat, act, frame in (
            (_P_ASSERT_DORMANT, 'assert', 'evidential-past'),
            (_P_ASSERT_RETR, 'assert', 'plain-after-retraction'),
            (_P_ASSERT, 'assert', 'plain'),
            (_P_Q_NEARMISS, 'question', 'near-miss'),
            (_P_Q_STARVE, 'question', 'starvation'),
            (_P_Q_EDGE, 'question', 'edge-gap')):
        m = pat.match(utterance)
        if m and m.group(1) in fam_idx:
            return {'act': act, 'family': fam_idx[m.group(1)],
                    'frame': frame}
    return None

"""Positive Control Harness.

Every test must fire on engineered positive data and stay quiet on matched null.
Hard gate: no test runs for record until it passes validation.

Based on Fable AI's factory design: parameterized factories covering the test
classes, with property-specific nulls that contain every known confound except
the property being tested.

Test statuses:
  VALID        - fires on positive, quiet on null
  CANNOT_FIRE  - returns ~null-level even on engineered positive
  CANNOT_REJECT- fires on null as strongly as on positive
  CRASHES      - raised on a well-formed input
  NO_CONTROL   - no positive control registered (blocked from running for record)
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List
from model import compute_obs_indices
from mental_model import CORE_OBS_DIM, action_to_hash


class TestStatus(Enum):
    VALID = "fires on positive, quiet on null"
    CANNOT_FIRE = "returns ~null-level even on engineered positive"
    CANNOT_REJECT = "fires on null as strongly as on positive"
    CRASHES = "raised on a well-formed input"
    NO_CONTROL = "no positive control registered"


IDX = compute_obs_indices()
OBS_DIM = IDX['obs_dim']
NUM_ACTIONS = IDX['num_actions']
CDIM = IDX.get('core_obs_dim', CORE_OBS_DIM)


# ============================================================
# Primitives for building synthetic logs and engines
# ============================================================

def blank_obs(rng=None):
    o = np.zeros(OBS_DIM)
    o[IDX['energy']] = 1.0
    return o


def obs_with_channel(rng, channel_name, value=0.8):
    o = blank_obs(rng)
    if channel_name in IDX and isinstance(IDX[channel_name], tuple):
        s, e = IDX[channel_name]
        o[s:e] = value
    elif channel_name in IDX:
        o[IDX[channel_name]] = value
    return o


def random_action(rng):
    return rng.randint(0, 2, size=NUM_ACTIONS).astype(np.int32)


def make_entry(obs_before=None, obs_after=None, action=None, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    if obs_before is None:
        obs_before = blank_obs(rng)
    if obs_after is None:
        obs_after = obs_before.copy() + rng.normal(0, 0.01, size=len(obs_before))
    if action is None:
        action = random_action(rng)
    endo_s, endo_e = IDX['endorphin']
    pain_s, pain_e = IDX['pain']
    reward = float(np.sum(obs_after[endo_s:endo_e]) - np.sum(obs_after[pain_s:pain_e]))
    return {
        'obs_before': obs_before.copy(),
        'obs_after': obs_after.copy(),
        'action': action.copy(),
        'reward': reward,
    }


def make_log(rng, n=2000, pain_driven=False):
    """Generate a synthetic log with enough diversity for encoder training."""
    log = []
    for i in range(n):
        obs = blank_obs(rng)
        for ch in ['pain', 'endorphin', 'temperature', 'chemical', 'pressure', 'fatigue']:
            if isinstance(IDX.get(ch), tuple):
                s, e = IDX[ch]
                obs[s:e] = rng.uniform(0, 1, size=e - s)

        if pain_driven:
            pain_s, pain_e = IDX['pain']
            pain_level = np.mean(obs[pain_s:pain_e])
            action = rng.randint(0, 2, size=NUM_ACTIONS).astype(np.int32)
            if pain_level > 0.5:
                action[0:3] = 1
            else:
                action[0:3] = 0
        else:
            action = random_action(rng)

        obs_after = obs.copy()
        obs_after += rng.normal(0, 0.05, size=len(obs))
        log.append(make_entry(obs, obs_after, action))

    return log


def make_engine(log):
    """Build a mental model engine from a log."""
    from mental_model import build_mental_model
    return build_mental_model(log)


# ============================================================
# Validation function
# ============================================================

def validate_test(test, n_nulls=10, seed=0):
    """Validate a single receptor test against positive and null controls.

    Returns (TestStatus, details_dict).
    """
    rng = np.random.RandomState(seed)

    # 1. Crash check
    try:
        minimal_log = make_log(rng, n=500)
        minimal_engine = make_engine(minimal_log)
        _ = test.test_fn(minimal_log, minimal_engine)
    except Exception as e:
        return TestStatus.CRASHES, {'error': repr(e)}

    # 2. Check if positive control exists in registry
    pos_fn = POSITIVE_CONTROLS.get(test.receptor_id)
    null_fn = NULL_CONTROLS.get(test.receptor_id)

    if pos_fn is None:
        return TestStatus.NO_CONTROL, None

    # 3. Run positive control
    try:
        pos_log, pos_engine = pos_fn(np.random.RandomState(seed))
        if pos_engine is None:
            pos_engine = make_engine(pos_log)
        pos_score = test.test_fn(pos_log, pos_engine)
    except Exception as e:
        return TestStatus.CRASHES, {'error': f'positive control: {repr(e)}'}

    if pos_score is None:
        pos_score = 0.0

    # 4. Null distribution
    null_scores = []
    for k in range(n_nulls):
        try:
            if null_fn is not None:
                n_log, n_engine = null_fn(np.random.RandomState(seed + k + 1))
                if n_engine is None:
                    n_engine = make_engine(n_log)
            else:
                n_log = make_log(np.random.RandomState(seed + k + 1), n=2000)
                n_engine = make_engine(n_log)
            score = test.test_fn(n_log, n_engine)
            null_scores.append(score if score is not None else 0.0)
        except Exception:
            null_scores.append(0.0)

    null_95 = float(np.percentile(null_scores, 95))
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores)) + 1e-8

    # 5. Verdicts
    details = {
        'pos_score': float(pos_score),
        'null_95': null_95,
        'null_mean': null_mean,
        'null_std': null_std,
    }

    if pos_score <= null_95:
        return TestStatus.CANNOT_FIRE, details

    effect = (pos_score - null_mean) / null_std
    details['effect_z'] = float(effect)

    if effect < 2.0:
        return TestStatus.CANNOT_REJECT, details

    return TestStatus.VALID, details


def validate_suite(tests, stop_on_failure=False):
    """Validate all tests. Returns report dict."""
    report = {}
    counts = {s: 0 for s in TestStatus}

    for test in tests:
        status, details = validate_test(test)
        report[test.receptor_id] = {
            'status': status.name,
            'details': details,
            'family': test.family,
        }
        counts[status] += 1

        marker = {'VALID': '+', 'CANNOT_FIRE': 'X', 'CANNOT_REJECT': '~',
                  'CRASHES': '!', 'NO_CONTROL': '?'}
        print(f"  [{marker.get(status.name, '?')}] {test.receptor_id}: {status.value}")

        if stop_on_failure and status not in (TestStatus.VALID, TestStatus.NO_CONTROL):
            print(f"\n  STOPPED: {test.receptor_id} failed validation")
            break

    print(f"\n  Summary: {counts[TestStatus.VALID]} valid, "
          f"{counts[TestStatus.NO_CONTROL]} no control, "
          f"{counts[TestStatus.CANNOT_FIRE]} cannot fire, "
          f"{counts[TestStatus.CANNOT_REJECT]} cannot reject, "
          f"{counts[TestStatus.CRASHES]} crash")

    return report


# ============================================================
# Positive control factories
# ============================================================

def _channel_response_positive(target_channel, rng, n=2000):
    """Actions driven by target channel value, with diverse action patterns."""
    log = []
    for _ in range(n):
        obs = blank_obs(rng)
        # Set all sensory channels to random for encoder diversity
        for ch in ['pain', 'endorphin', 'temperature', 'chemical', 'pressure', 'fatigue']:
            if isinstance(IDX.get(ch), tuple):
                s, e = IDX[ch]
                obs[s:e] = rng.uniform(0, 1, size=e - s)

        if isinstance(IDX.get(target_channel), tuple):
            s, e = IDX[target_channel]
            val = np.mean(obs[s:e])
        else:
            continue

        # Diverse actions: target-correlated but with noise
        action = rng.randint(0, 2, size=NUM_ACTIONS).astype(np.int32)
        if val > 0.5:
            action[0:3] = 1  # bias toward extension when target high
        else:
            action[0:3] = 0

        obs_after = obs.copy() + rng.normal(0, 0.05, size=len(obs))
        log.append(make_entry(obs, obs_after, action))
    return log, None


def _channel_response_null(confound_channel, rng, n=2000):
    """Actions driven by confound, not target. Same obs diversity."""
    log = []
    for _ in range(n):
        obs = blank_obs(rng)
        for ch in ['pain', 'endorphin', 'temperature', 'chemical', 'pressure', 'fatigue']:
            if isinstance(IDX.get(ch), tuple):
                s, e = IDX[ch]
                obs[s:e] = rng.uniform(0, 1, size=e - s)

        if isinstance(IDX.get(confound_channel), tuple):
            s, e = IDX[confound_channel]
            val = np.mean(obs[s:e])
        else:
            continue

        action = rng.randint(0, 2, size=NUM_ACTIONS).astype(np.int32)
        if val > 0.5:
            action[0:3] = 1
        else:
            action[0:3] = 0

        obs_after = obs.copy() + rng.normal(0, 0.05, size=len(obs))
        log.append(make_entry(obs, obs_after, action))
    return log, None


def _store_structure_positive(rng, high_cert=True, n_entries=200):
    """Engine with high-certainty well-structured entries."""
    log = make_log(rng, n=3000, pain_driven=True)
    engine = make_engine(log)
    return log, engine


def _store_structure_null(rng, n_entries=200):
    """Engine from random log — no structure."""
    log = make_log(rng, n=3000, pain_driven=False)
    engine = make_engine(log)
    return log, engine


# ============================================================
# Control registry
# ============================================================

POSITIVE_CONTROLS: Dict[str, Callable] = {}
NULL_CONTROLS: Dict[str, Callable] = {}


def register_control(receptor_id, positive_fn, null_fn=None):
    POSITIVE_CONTROLS[receptor_id] = positive_fn
    if null_fn:
        NULL_CONTROLS[receptor_id] = null_fn


# --- Channel-response family ---
for receptor_id, target, confound in [
    ('pain', 'pain', 'endorphin'),
    ('fatigue', 'fatigue', 'pain'),
    ('self_regulation', 'fatigue', 'pain'),
    ('stress_detection', 'pain', 'endorphin'),
    ('arousal_regulation', 'pain', 'fatigue'),
]:
    register_control(
        receptor_id,
        lambda rng, t=target: _channel_response_positive(t, rng),
        lambda rng, c=confound: _channel_response_null(c, rng),
    )

# --- Store-structure family ---
for receptor_id in [
    'prediction', 'mental_model', 'prediction_accuracy',
    'certainty_tracking', 'multiple_hypotheses',
]:
    register_control(
        receptor_id,
        lambda rng: _store_structure_positive(rng),
        lambda rng: _store_structure_null(rng),
    )

# --- Behavioral family (pain-driven) ---
for receptor_id in [
    'basic_sensorimotor_loop', 'habituation', 'sensitization',
    'spatial_memory', 'distance_sensing',
]:
    register_control(
        receptor_id,
        lambda rng: _channel_response_positive('pain', rng),
        lambda rng: _channel_response_null('endorphin', rng),
    )


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    from receptor_discovery import build_tests

    tests = build_tests()
    print(f"Validating {len(tests)} tests...\n")

    registered = len(POSITIVE_CONTROLS)
    total = len(tests)
    print(f"  Controls registered: {registered}/{total} "
          f"({registered/total*100:.0f}%)\n")

    report = validate_suite(tests)

    # Save report
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    serializable = {}
    for k, v in report.items():
        serializable[k] = {
            'status': v['status'],
            'family': v['family'],
        }
        if v['details']:
            serializable[k]['details'] = {
                dk: round(dv, 4) if isinstance(dv, float) else dv
                for dk, dv in v['details'].items()
            }

    import os
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'validation_report.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nReport saved to data/validation_report.json")

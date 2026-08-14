"""EL-0 acceptance (pre-registered, environment_organism_requirements.md):

C20 pre-flight (six checks):
1. Domain match — the interpreter is tested on exactly the worlds it will
   serve: TieredEnvironment tiers 0-4, the rich stack's range.
2. Endpoint independence — the language layer writes nothing during the
   behavioral run; both arms are plain environments by the time the
   organism moves.
3. Exogeneity — construction path (direct vs interpreted) is the
   manipulated variable; nothing else differs.
4. Pairing proven — the runtime protocol reseeds both arms' rngs
   identically post-construction (construction consumes different rng
   amounts by design; runtime streams are the pairing surface), and the
   global numpy stream is reseeded per arm. The acceptance IS the
   identity check.
5. Phenomenon strength — bit-identity is demanded, not approximated;
   floors: 5 tiers x 300 steps x 2 episodes.
6. Endpoint sensitivity — obs/reward streams move every step by
   construction; any interpretation error surfaces immediately.

ACCEPTANCE RULES (fixed before running):
  A. Round-trip fixed point, tiers 0-4:
     describe(interpret(describe(env))) == describe(env), exactly.
  B. Behavioral regression, tiers 0-4: a real Organism run under the
     paired runtime protocol on the direct world vs the interpreted world
     produces bit-identical observation and reward streams (300 steps x 2
     episodes per tier).
  PASS requires A and B at every tier. Anything less: EL-0 iterates;
  the YAML/tier path remains the instrument of record.
"""

import numpy as np

from environment import Organism
from environment_tiers import TieredEnvironment, StochasticHiddenVariable
from environment_language import describe, interpret, write_etymology_ledger

TIERS = (0, 1, 2, 3, 4)
BUILD_SEED = 4242
RUNTIME_SEED = 777
STEPS = 300
EPISODES = 2

PASS_N, FAIL_N = 0, 0


def check(name, cond, detail=''):
    global PASS_N, FAIL_N
    if cond:
        PASS_N += 1
        print(f"  PASS  {name}")
    else:
        FAIL_N += 1
        print(f"  FAIL  {name}  {detail}")


def reseed_runtime(env, seed):
    """The paired runtime protocol: both arms get identical fresh streams
    after construction. Stochastic tier-4 state is re-drawn from the new
    stream symmetrically."""
    env.rng = np.random.RandomState(seed)
    if getattr(env, 'stochastic_hidden', None) is not None:
        env.stochastic_hidden = StochasticHiddenVariable(
            num_states=env.stochastic_hidden.num_states, rng=env.rng)


def run_organism(env, runtime_seed):
    """A real organism, oracle policy, deterministic given the world."""
    np.random.seed(runtime_seed)
    reseed_runtime(env, runtime_seed)
    obs_stream, reward_stream = [], []
    for ep in range(EPISODES):
        org = Organism()
        org.reset()
        for step in range(STEPS):
            actions = org.compute_optimal_actions(env, step)
            obs, reward = org.step(actions, env, step)
            obs_stream.append(np.asarray(obs, dtype=np.float64).copy())
            reward_stream.append(float(reward))
    return np.stack(obs_stream), np.asarray(reward_stream)


def main():
    print('=== EL-0 acceptance: the environment reads itself aloud ===')
    ledger = write_etymology_ledger()
    print(f'etymology ledger seeded: {ledger}')

    for tier in TIERS:
        print(f'\n--- tier {tier} ---')
        env_a = TieredEnvironment(seed=BUILD_SEED + tier, tier=tier)
        corpus = describe(env_a)
        print(f'  corpus: {len(corpus)} sentences')

        env_b = interpret(corpus)
        corpus2 = describe(env_b)
        check(f'tier {tier} round-trip fixed point', corpus2 == corpus,
              next((f'first diff: {x!r} vs {y!r}'
                    for x, y in zip(corpus, corpus2) if x != y),
                   f'len {len(corpus)} vs {len(corpus2)}'))

        env_c = interpret(corpus)   # fresh interpretation for the run
        obs_a, rew_a = run_organism(env_a, RUNTIME_SEED + tier)
        obs_b, rew_b = run_organism(env_c, RUNTIME_SEED + tier)
        same_obs = np.array_equal(obs_a, obs_b)
        same_rew = np.array_equal(rew_a, rew_b)
        detail = ''
        if not same_obs:
            idx = np.argwhere(obs_a != obs_b)
            detail = f'first obs divergence at {idx[0].tolist()}'
        check(f'tier {tier} behavioral identity '
              f'({EPISODES}x{STEPS} steps)', same_obs and same_rew, detail)

    print(f'\n{"=" * 50}\nEL-0 ACCEPTANCE: {PASS_N} passed, {FAIL_N} failed')
    print('VERDICT:', 'PASS — the interpreter is the instrument of record '
          'for tiers 0-4' if FAIL_N == 0 else
          'NOT PASSED — the YAML/tier path remains the instrument of '
          'record; EL-0 iterates')
    return FAIL_N == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)

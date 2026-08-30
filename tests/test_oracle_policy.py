"""Sanity check: is RobotReachEnv itself solvable at all?

This does not touch PPO. A hand-coded controller reads the target-robot
displacement straight out of the observation (obs[4:6]) and always moves at
full speed directly toward the target -- no learning involved. With the
target sampled from [-1, 1] x [-1, 1], the robot starting at the origin, a
top speed of 2.0 m/s, and 500 steps available, this should resolve almost
every episode (worst case ~1.41 m away needs roughly 170 steps at top
speed).

If this test fails, the problem is in the environment/reward/termination
contract itself, not in anything PPO does -- see docs/design.md for why this
was added (Andrea's review flagged that this diagnostic was missing).
"""

import numpy as np

from src.environment.reaching_env import RobotReachEnv


def oracle_action(obs: np.ndarray) -> np.ndarray:
    """Move at full speed directly toward the target."""
    delta = obs[4:6]
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return (delta / norm).astype(np.float32)


def test_oracle_controller_reaches_the_target_reliably():
    env = RobotReachEnv(render_mode=None)
    n_seeds = 100
    successes = 0
    max_steps_used = 0

    try:
        for seed in range(n_seeds):
            obs, _ = env.reset(seed=seed)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                obs, _, terminated, truncated, _ = env.step(oracle_action(obs))
                steps += 1
            successes += int(terminated)
            max_steps_used = max(max_steps_used, steps)
    finally:
        env.close()

    # The target is fixed-base: a correct-direction oracle must solve every
    # sampled target in this bounded workspace.
    assert successes == n_seeds, (
        f"only {successes}/{n_seeds} resolved -- the environment itself may "
        f"not be reliably solvable, independent of PPO"
    )
    assert max_steps_used <= 250, (
        f"slowest resolved episode took {max_steps_used} steps against a "
        f"500-step budget and a ~170-step worst-case estimate -- distance/"
        f"velocity accounting may not match the docstring's assumptions"
    )

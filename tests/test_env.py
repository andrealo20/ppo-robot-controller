"""RobotReachEnv API tests.

Distance-to-target is monkeypatched rather than driven through real physics:
what these tests need to pin down is the terminated/truncated contract, which
should hold regardless of exactly how PyBullet moves the robot on a given
run.
"""

import numpy as np
import pytest

from src.environment.reaching_env import (
    MAX_STEPS,
    SUCCESS_DISTANCE,
    WORKSPACE_LIMIT,
    RobotReachEnv,
)


@pytest.fixture
def env():
    e = RobotReachEnv(render_mode=None)
    yield e
    e.close()


def test_reset_returns_correctly_shaped_observation(env):
    obs, info = env.reset()
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_gymnasium_five_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reaching_the_target_terminates_not_truncates(env, monkeypatch):
    env.reset()
    monkeypatch.setattr(env, "_distance_to_target", lambda: SUCCESS_DISTANCE / 2)

    _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward > 0  # success bonus should dominate the small negative distance term


def test_step_budget_truncates_not_terminates(env, monkeypatch):
    env.reset()
    monkeypatch.setattr(env, "_distance_to_target", lambda: 5.0)  # never within reach

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
        assert steps <= MAX_STEPS + 1  # guard against an infinite loop on failure

    assert steps == MAX_STEPS
    assert terminated is False
    assert truncated is True


def test_robot_cannot_drift_outside_the_workspace(env, monkeypatch):
    """The M1.1 fix (see docs/design.md, 'M1.1: bounding the workspace'):
    without this, sustained maximal action lets the robot travel roughly
    0.0083 m/step * 500 steps ~= 4.2 m from the origin -- well past
    WORKSPACE_LIMIT. Running the maximal action for a full episode must
    never push the observed robot position past the bound on either axis.
    The target is monkeypatched out of reach so the episode always runs the
    full MAX_STEPS budget instead of ending early on a lucky success.
    """
    env.reset()
    monkeypatch.setattr(env, "_distance_to_target", lambda: 5.0)
    max_action = np.array([1.0, 1.0], dtype=np.float32)

    terminated = truncated = False
    obs = None
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step(max_action)
        robot_xy = obs[:2]
        assert np.all(np.abs(robot_xy) <= WORKSPACE_LIMIT + 1e-6)

    assert obs is not None


def test_action_is_not_mutated_by_step(env):
    """resetBaseVelocity is given `action * 2.0`, computed via np.asarray --
    verifies the caller's array isn't written back into."""
    env.reset()
    action = np.array([0.3, -0.7], dtype=np.float32)
    action_copy = action.copy()

    env.step(action)

    assert np.array_equal(action, action_copy)

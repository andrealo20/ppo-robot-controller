"""RobotReachEnv API tests.

Distance-to-target is monkeypatched rather than driven through real physics:
what these tests need to pin down is the terminated/truncated contract, which
should hold regardless of exactly how PyBullet moves the robot on a given
run.
"""

import numpy as np
import pytest

from src.environment.reaching_env import MAX_STEPS, SUCCESS_DISTANCE, RobotReachEnv


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


def test_action_is_not_mutated_by_step(env):
    """resetBaseVelocity is given `action * 2.0`, computed via np.asarray --
    verifies the caller's array isn't written back into."""
    env.reset()
    action = np.array([0.3, -0.7], dtype=np.float32)
    action_copy = action.copy()

    env.step(action)

    assert np.array_equal(action, action_copy)

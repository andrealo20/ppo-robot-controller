"""RunningMeanStd / NormalizeObservation tests."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.running_normalizer import NormalizeObservation, RunningMeanStd


def test_running_mean_std_matches_numpy_on_a_single_batch():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(500, 4))

    rms = RunningMeanStd(shape=(4,))
    rms.update(data)

    # RunningMeanStd starts with a small epsilon "phantom" sample count, so
    # it won't match numpy exactly -- but with 500 real samples against an
    # epsilon of 1e-4 the difference is negligible.
    assert np.allclose(rms.mean, data.mean(axis=0), atol=1e-2)
    assert np.allclose(rms.var, data.var(axis=0), atol=1e-1)


def test_running_mean_std_incremental_matches_batch():
    """Feeding data in two chunks must give the same result (within floating
    point tolerance) as feeding it all at once -- this is the whole point of
    Chan's parallel-merge formula over naively averaging per-batch stats.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(loc=-1.0, scale=0.5, size=(400, 3))

    rms_batched = RunningMeanStd(shape=(3,))
    rms_batched.update(data)

    rms_incremental = RunningMeanStd(shape=(3,))
    rms_incremental.update(data[:150])
    rms_incremental.update(data[150:])

    assert np.allclose(rms_batched.mean, rms_incremental.mean, atol=1e-8)
    assert np.allclose(rms_batched.var, rms_incremental.var, atol=1e-8)


class _ConstantObsEnv(gym.Env):
    """Minimal env returning a fixed observation, for wrapper tests."""

    def __init__(self, value):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._value = value

    def reset(self, seed=None, options=None):
        return np.array(self._value, dtype=np.float32), {}

    def step(self, action):
        return np.array(self._value, dtype=np.float32), 0.0, False, False, {}


def test_normalized_constant_observation_trends_toward_zero():
    """Feeding the wrapper the same observation repeatedly should drive its
    running mean toward that value, so normalized observations trend toward
    zero -- a cheap end-to-end check that update()/observation() are wired
    together correctly.
    """
    env = NormalizeObservation(_ConstantObsEnv(value=[5.0, -5.0]))
    env.reset()

    last_obs = None
    for _ in range(200):
        last_obs, _, _, _, _ = env.step(np.zeros(1, dtype=np.float32))

    assert np.all(np.abs(last_obs) < 0.5)


def test_frozen_stats_do_not_update():
    env = NormalizeObservation(_ConstantObsEnv(value=[1.0, 1.0]), update_stats=False)
    env.reset()

    count_before = env.rms.count
    for _ in range(10):
        env.step(np.zeros(1, dtype=np.float32))

    assert env.rms.count == count_before

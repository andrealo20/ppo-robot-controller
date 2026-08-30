"""Running observation normalization.

PPO on continuous-control tasks is usually trained against observations
normalized to roughly zero mean and unit variance -- without it, a reward or
gradient scale that happens to suit one observation component can be wrong by
orders of magnitude for another, and the shared network trunk here has no way
to correct for that on its own. `RobotReachEnv`'s observation mixes raw
position and displacement components with no shared scale, so this matters
concretely, not just in principle.

The mean and variance can't be known in advance -- they depend on the policy
being trained, which changes what states get visited -- so they're estimated
online from the same rollouts used for training, with Welford's algorithm
(the parallel/batch form, so a whole batch of observations can be folded in
with one call instead of one at a time).
"""

import gymnasium as gym
import numpy as np


class RunningMeanStd:
    """Tracks a running mean and variance over batches of vectors.

    Uses Chan et al.'s parallel variant of Welford's algorithm: merging two
    summaries (running stats so far, and a new batch) from their counts,
    means and variances alone, without revisiting old data.
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class NormalizeObservation(gym.ObservationWrapper):
    """Gymnasium wrapper that normalizes observations with a running mean/std.

    Statistics are updated on every observation seen through reset()/step()
    -- including during evaluation, matching common practice (e.g. Stable-
    Baselines3's VecNormalize) of letting statistics keep adapting rather than
    freezing them at an arbitrary point. Pass `update_stats=False` to freeze
    them, e.g. when replaying a fixed evaluation seed.
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        update_stats: bool = True,
    ):
        super().__init__(env)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)
        self.epsilon = epsilon
        self.clip = clip
        self.update_stats = update_stats

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.update_stats:
            self.rms.update(observation[None, :])
        normalized = (observation - self.rms.mean) / np.sqrt(
            self.rms.var + self.epsilon
        )
        normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized.astype(np.float32)

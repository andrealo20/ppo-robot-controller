"""Running observation normalization with checkpointable statistics."""

import gymnasium as gym
import numpy as np


class RunningMeanStd:
    """Track running mean/variance with Chan/Welford batch updates."""

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

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
        self.count = float(total_count)

    def state_dict(self) -> dict:
        """Return a copy safe to serialize in a model checkpoint."""
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore statistics, validating their shape and numerical sanity."""
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError(
                f"normalizer shape mismatch: checkpoint {mean.shape}, expected {self.mean.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
            raise ValueError("normalizer checkpoint contains non-finite statistics")
        if np.any(var < 0.0) or not np.isfinite(count) or count <= 0.0:
            raise ValueError("normalizer checkpoint contains invalid variance/count")
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = count


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations using running statistics.

    Training should use ``update_stats=True``. Evaluation should restore the
    training statistics from the checkpoint and then set ``update_stats=False``.
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
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

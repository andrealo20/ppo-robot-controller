"""Neural network architectures for policy and value functions."""

import math

import torch
from torch import nn

# Conservative bounds for the state-independent exploration parameter. PPO's
# policy is squashed with tanh before reaching the environment, so the action
# bounds are now handled probabilistically rather than by post-hoc clipping.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
INITIAL_LOG_STD = -0.5


class PolicyValueNetwork(nn.Module):
    """Shared-trunk actor-critic network for continuous-control PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.log_std = nn.Parameter(torch.full((action_dim,), INITIAL_LOG_STD))
        self._init_weights()

    @staticmethod
    def _init_linear(layer: nn.Linear, gain: float) -> None:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)

    def _init_weights(self) -> None:
        """PPO-style orthogonal initialization with head-specific gains."""
        self._init_linear(self.shared[0], math.sqrt(2.0))
        self._init_linear(self.shared[2], math.sqrt(2.0))
        self._init_linear(self.policy_head[0], math.sqrt(2.0))
        self._init_linear(self.policy_head[2], 0.01)
        self._init_linear(self.value_head[0], math.sqrt(2.0))
        self._init_linear(self.value_head[2], 1.0)

    def forward(self, obs: torch.Tensor):
        """Return unsquashed Gaussian mean/std and scalar value estimate."""
        features = self.shared(obs)
        mean = self.policy_head(features)

        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        value = self.value_head(features)
        return mean, std, value

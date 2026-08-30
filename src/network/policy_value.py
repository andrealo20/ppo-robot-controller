"""Neural network architectures for policy and value functions."""

import torch
from torch import nn

# Bounds on the learned log standard deviation. Without a clamp this parameter
# is free to run away during training: it can grow until sampled actions are
# almost pure noise, or collapse until the policy is numerically deterministic
# and log-probability gradients blow up. -20/2 in log-space means std lands in
# [exp(-20), exp(2)] ~= [2e-9, 7.4].
#
# M1 training plateaued with log_std pinned at this 2.0 ceiling (std ~= 7.4,
# far above the [-1, 1] action range) -- see docs/design.md, "M1: a reward
# plateau, a runaway hypothesis, and a disproved fix". Both a smaller ceiling
# (0.0, std <= 1.0) and a smaller entropy_coef were tried, expecting either to
# help; both made training measurably *less* stable (faster, deeper
# collapses into runaway trajectories), so LOG_STD_MAX is kept at its
# original value here rather than "fixed" on a hypothesis that real training
# runs disproved. The actual mechanism behind the plateau is now understood
# to sit in the environment's reward/workspace design, not this clamp --
# also in docs/design.md.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class PolicyValueNetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Log standard deviation (learned, state-independent, clamped in forward())
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor):
        """Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            mean, std, value
        """
        features = self.shared(obs)

        # Policy output (mean)
        mean = self.policy_head(features)

        # Standard deviation, clamped in log-space before exponentiating so a
        # single bad gradient step can't send it to 0 or inf.
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        std = std.unsqueeze(0).expand_as(mean)

        # Value output
        value = self.value_head(features)

        return mean, std, value

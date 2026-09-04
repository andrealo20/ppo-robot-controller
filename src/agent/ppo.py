"""Proximal Policy Optimization agent for bounded continuous actions."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

from src.network.policy_value import LOG_STD_MAX, LOG_STD_MIN, PolicyValueNetwork

# Below torch 2.6, torch.load defaults to unrestricted unpickling, so reading a
# checkpoint of unknown provenance executes whatever code it carries. Every
# checkpoint save() writes holds only tensors, dicts, lists and scalars, all of
# which restricted unpickling handles, so weights_only=True costs nothing here.
# The argument itself only exists from torch 1.13 on; older versions fall back
# to the plain call rather than failing outright.
# inspect.signature raises for callables that carry no signature metadata, which
# some builds expose for C-implemented functions. That would turn a capability
# probe into an import-time crash, so an unreadable signature is treated as
# "argument not available" rather than propagating.
try:
    _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = (
        "weights_only" in inspect.signature(torch.load).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - build-dependent
    _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = False


class PPOAgent:
    """PPO with a tanh-squashed diagonal Gaussian policy."""

    _ACTION_EPS = 1e-6
    CHECKPOINT_VERSION = 2

    def __init__(self, observation_space, action_space, config=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = dict(config or {})

        self.lr = self.config.get("lr", 1e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.eps_clip = self.config.get("eps_clip", 0.2)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyValueNetwork(
            obs_dim=observation_space.shape[0], action_dim=action_space.shape[0]
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("PPOAgent requires finite continuous action bounds")
        if np.any(high <= low):
            raise ValueError("each action-space high bound must exceed its low bound")
        self._action_bias = torch.as_tensor((high + low) / 2.0, device=self.device)
        self._action_scale = torch.as_tensor((high - low) / 2.0, device=self.device)

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminated = []
        self.episode_end = []
        self.boundary_value = []

    def _distribution(self, states: torch.Tensor):
        means, stds, values = self.network(states)
        return torch.distributions.Normal(means, stds), values

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self._action_bias + self._action_scale * torch.tanh(raw_action)

    def _unsquash(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unit = (action - self._action_bias) / self._action_scale
        unit = torch.clamp(unit, -1.0 + self._ACTION_EPS, 1.0 - self._ACTION_EPS)
        raw = torch.atanh(unit)
        return raw, unit

    def _log_prob_from_dist(
        self, dist: torch.distributions.Normal, action: torch.Tensor
    ) -> torch.Tensor:
        """Log probability of the bounded action under the transformed policy."""
        raw, unit = self._unsquash(action)
        base_log_prob = dist.log_prob(raw)
        # y = bias + scale * tanh(x), so |dy/dx| = scale * (1 - tanh(x)^2).
        log_abs_det = torch.log(self._action_scale) + torch.log(
            torch.clamp(1.0 - unit.square(), min=self._ACTION_EPS)
        )
        return (base_log_prob - log_abs_det).sum(-1)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate stored bounded actions under the current policy."""
        dist, values = self._distribution(states)
        log_probs = self._log_prob_from_dist(dist, actions)
        # Base-Gaussian entropy is a stable exploration proxy for the squashed
        # policy; the exact transformed entropy has no simple closed form.
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values

    def select_action_and_value(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """Return a bounded action, its log-probability, and V(state).

        The network already produces the value estimate alongside the policy
        distribution, so a rollout step that needs both should ask for both
        here rather than calling select_action() and get_value() in sequence,
        which runs the same forward pass twice on the same observation.
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dist, value = self._distribution(state_tensor)
            raw_action = dist.mean if deterministic else dist.sample()
            action_tensor = self._squash(raw_action)

            # Round-trip through numpy first, then score exactly the action that
            # the caller receives/stores. This preserves the PPO ratio identity
            # even for saturated float32 tanh outputs near +/-1.
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            returned_action = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            log_prob = self._log_prob_from_dist(dist, returned_action).item()
        return action, float(log_prob), float(value.item())

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float]:
        """Return a bounded environment action and its matching log-probability."""
        action, log_prob, _ = self.select_action_and_value(state, deterministic)
        return action, log_prob

    def store_transition(
        self,
        state,
        action,
        reward,
        value,
        log_prob,
        terminated,
        truncated=False,
        boundary_value=0.0,
    ):
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.terminated.append(terminated)
        self.episode_end.append(terminated or truncated)
        self.boundary_value.append(boundary_value if truncated else 0.0)

    def compute_returns_and_advantages(self, next_value):
        returns = []
        advantages = []
        gae = 0.0
        n = len(self.rewards)

        for t in reversed(range(n)):
            if self.episode_end[t]:
                bootstrap = 0.0 if self.terminated[t] else self.boundary_value[t]
                next_non_terminal = 0.0 if self.terminated[t] else 1.0
                gae_continues = 0.0
            elif t == n - 1:
                bootstrap = next_value
                next_non_terminal = 1.0
                gae_continues = 1.0
            else:
                bootstrap = self.values[t + 1]
                next_non_terminal = 1.0
                gae_continues = 1.0

            delta = (
                self.rewards[t]
                + self.gamma * next_non_terminal * bootstrap
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * gae_continues * gae
            advantages.append(gae)
            returns.append(gae + self.values[t])

        # The loop walks the buffer backwards, so both lists come out reversed;
        # appending and reversing once is linear, where insert(0, ...) per
        # transition is quadratic in the rollout length.
        advantages.reverse()
        returns.reverse()

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(
            advantages, dtype=torch.float32
        )

    def update(self, next_value, epochs=3, minibatch_size=None) -> dict[str, float]:
        """Run PPO epochs and return aggregate diagnostics for logging."""
        if not self.states:
            return {}

        returns, advantages = self.compute_returns_and_advantages(next_value)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        states = torch.as_tensor(
            np.asarray(self.states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.asarray(self.actions), dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            self.log_probs, dtype=torch.float32, device=self.device
        )
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        n = states.shape[0]
        batch_size = n if minibatch_size is None else min(minibatch_size, n)
        metric_rows: list[dict[str, float]] = []

        for _ in range(epochs):
            indices = (
                np.random.permutation(n) if minibatch_size is not None else np.arange(n)
            )
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                metric_rows.append(
                    self._gradient_step(
                        states[idx],
                        actions[idx],
                        old_log_probs[idx],
                        returns[idx],
                        advantages[idx],
                    )
                )

        self.clear_buffer()
        if not metric_rows:
            return {}
        return {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }

    def _gradient_step(self, states, actions, old_log_probs, returns, advantages):
        new_log_probs, entropy_per_sample, values = self.evaluate_actions(
            states, actions
        )
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((values.squeeze(-1) - returns) ** 2).mean()
        entropy = entropy_per_sample.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite PPO loss")

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite PPO gradient norm")
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.eps_clip).float().mean()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
            "grad_norm": float(grad_norm.item()),
            "policy_std": float(
                torch.exp(torch.clamp(self.network.log_std, LOG_STD_MIN, LOG_STD_MAX))
                .mean()
                .item()
            ),
        }

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.terminated.clear()
        self.episode_end.clear()
        self.boundary_value.clear()

    def get_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, _, value = self.network(state_tensor)
        return float(value.item())

    def save(
        self,
        path,
        observation_rms=None,
        config: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> None:
        """Save a complete, evaluation-safe training checkpoint."""
        checkpoint = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": dict(self.config if config is None else config),
            "training_state": dict(training_state or {}),
            "normalizer_state": (
                observation_rms.state_dict() if observation_rms is not None else None
            ),
        }
        torch.save(checkpoint, Path(path))

    def load(self, path, observation_rms=None, load_optimizer: bool = False) -> dict:
        """Load a checkpoint; a bare network ``state_dict`` file also works."""
        load_kwargs = (
            {"weights_only": True} if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY else {}
        )
        payload = torch.load(Path(path), map_location=self.device, **load_kwargs)
        if isinstance(payload, dict) and "network_state_dict" in payload:
            self.network.load_state_dict(payload["network_state_dict"])
            if load_optimizer and payload.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            restored = False
            if (
                observation_rms is not None
                and payload.get("normalizer_state") is not None
            ):
                observation_rms.load_state_dict(payload["normalizer_state"])
                restored = True
            return {
                "checkpoint_version": payload.get("checkpoint_version", 1),
                "config": payload.get("config", {}),
                "training_state": payload.get("training_state", {}),
                "normalizer_restored": restored,
                "legacy": False,
            }

        # Bare state_dict, with no normalizer/optimizer/config alongside it.
        self.network.load_state_dict(payload)
        return {
            "checkpoint_version": 0,
            "config": {},
            "training_state": {},
            "normalizer_restored": False,
            "legacy": True,
        }

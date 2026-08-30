"""PPO Agent implementation."""

import numpy as np
import torch
from torch import optim

from src.network.policy_value import PolicyValueNetwork


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, observation_space, action_space, config=None):
        """Initialize PPO agent.

        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            config: Configuration dict
        """
        self.observation_space = observation_space
        self.action_space = action_space

        # Hyperparameters
        self.lr = config.get("lr", 1e-4) if config else 1e-4
        self.gamma = config.get("gamma", 0.99) if config else 0.99
        self.gae_lambda = config.get("gae_lambda", 0.95) if config else 0.95
        self.eps_clip = config.get("eps_clip", 0.2) if config else 0.2
        self.value_coef = config.get("value_coef", 0.5) if config else 0.5
        self.entropy_coef = config.get("entropy_coef", 0.01) if config else 0.01

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.network = PolicyValueNetwork(
            obs_dim=observation_space.shape[0], action_dim=action_space.shape[0]
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Rollout buffer. May span multiple episodes (see docs/design.md,
        # "Multi-episode rollouts"): `episode_end[t]` is True whenever
        # transition t is the LAST transition of its episode, for any reason
        # (task resolved OR the environment's step budget ran out).
        # `terminated[t]` narrows that further to "resolved" specifically --
        # only meaningful when episode_end[t] is True. `boundary_value[t]` is
        # the bootstrap value to use at a truncation boundary (V(next_state),
        # computed by the caller at collection time); it is unused --
        # ignored -- everywhere episode_end[t] is False or terminated[t] is
        # True.
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminated = []
        self.episode_end = []
        self.boundary_value = []

    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Select action from current state.

        Args:
            state: Current observation

        Returns:
            action, log_prob
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, std, _value = self.network(state_tensor)

            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().item()

        return np.clip(action, -1.0, 1.0), log_prob

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
        """Store one environment transition in the rollout buffer.

        Args:
            terminated: True only if this transition ended its episode by
                resolving the task. A transition that ends its episode purely
                because the step limit was reached (Gymnasium's `truncated`)
                must be passed here as terminated=False -- exactly as
                env.step()'s (terminated, truncated) pair keeps them apart.
            truncated: True only if this transition ended its episode by
                running out of the step budget. `terminated` and `truncated`
                must never both be True (Gymnasium's own contract).
            boundary_value: Required (and only meaningful) when
                truncated=True: V(the real next state), i.e. the critic's own
                estimate of what happens after this transition, computed by
                the caller *before* the environment is reset for the next
                episode. Ignored when truncated=False.
        """
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
        """Compute returns and GAE advantages over the whole rollout buffer.

        The buffer may contain several episodes back to back (see
        docs/design.md). At every `episode_end[t]`, the GAE recursion is cut
        (the advantage of whatever episode comes after t must never leak
        backward into t's episode) and the TD bootstrap is either 0.0 (a true
        termination) or the precomputed `boundary_value[t]` (a truncation).
        Everywhere else -- including the very last buffer transition, if its
        episode is still ongoing -- behaves exactly like the original
        single-episode recursion, bootstrapping from `next_value`.

        Args:
            next_value: Bootstrap value for the state *after* the last stored
                transition, used only if that transition is not itself an
                episode_end (i.e. the rollout was cut off mid-episode purely
                because the buffer reached its step budget, not because the
                episode ended). Ignored otherwise.
        """
        returns = []
        advantages = []
        gae = 0.0
        n = len(self.rewards)

        for t in reversed(range(n)):
            if self.episode_end[t]:
                bootstrap = 0.0 if self.terminated[t] else self.boundary_value[t]
                next_non_terminal = 0.0 if self.terminated[t] else 1.0
                gae_continues = 0.0  # always cut the recursion at an episode end
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

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def update(self, next_value, epochs=3, minibatch_size=None):
        """Update policy and value network on the current rollout buffer.

        Args:
            next_value: See compute_returns_and_advantages.
            epochs: Number of passes over the rollout.
            minibatch_size: If given, each epoch shuffles the rollout and
                splits it into minibatches of this size (the last one may be
                smaller). If None (default), each epoch is a single full-batch
                gradient step over the whole rollout -- the original M0
                behaviour, kept as the default so existing single-episode
                callers (and their tests) are unaffected.
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # Normalize advantages once, over the whole rollout, before any
        # minibatching -- standard practice (splitting first would let small
        # minibatches skew each other's normalization). unbiased=False
        # deliberately: torch's default (Bessel-corrected) std of a
        # single-element tensor is 0/0 = NaN -- a one-transition rollout (a
        # legal, if unusual, edge case) would silently poison the whole
        # update. The biased std of one element is exactly 0, which + 1e-8
        # leaves that one advantage at 0 after normalization: inert, not NaN.
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        n = states.shape[0]
        batch_size = n if minibatch_size is None else min(minibatch_size, n)

        for _ in range(epochs):
            indices = (
                np.random.permutation(n) if minibatch_size is not None else np.arange(n)
            )
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                self._gradient_step(
                    states[idx],
                    actions[idx],
                    old_log_probs[idx],
                    returns[idx],
                    advantages[idx],
                )

        # Clear buffer
        self.clear_buffer()

    def _gradient_step(self, states, actions, old_log_probs, returns, advantages):
        """One PPO gradient step on a (mini)batch."""
        means, stds, values = self.network(states)

        dist = torch.distributions.Normal(means, stds)
        new_log_probs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = ((values.squeeze(-1) - returns) ** 2).mean()

        entropy = dist.entropy().sum(-1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

    def clear_buffer(self):
        """Clear rollout buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.terminated.clear()
        self.episode_end.clear()
        self.boundary_value.clear()

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value = self.network(state_tensor)
        return value.cpu().item()

    def save(self, path):
        """Save model checkpoint."""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """Load model checkpoint."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))

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

        # Rollout buffer. `self.terminated` records, per stored transition,
        # whether *that* transition ended the episode by resolving the task
        # (as opposed to running out of the step budget) -- see
        # compute_returns_and_advantages() for why the distinction matters.
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminated = []

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

    def store_transition(self, state, action, reward, value, log_prob, terminated):
        """Store one environment transition in the rollout buffer.

        Args:
            terminated: True only if this transition ended the episode by
                resolving the task. A transition that ends the episode purely
                because the step limit was reached (Gymnasium's `truncated`)
                must be stored as terminated=False -- the caller is
                responsible for keeping the two apart, exactly as
                env.step()'s (terminated, truncated) pair keeps them apart.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.terminated.append(terminated)

    def compute_returns_and_advantages(self, next_value):
        """Compute returns and GAE advantages.

        Args:
            next_value: Bootstrap value for the state *after* the last stored
                transition. Must be 0.0 if the episode actually terminated
                there, and V(next_state) (from the network) if it was only
                truncated -- see PPOAgent.update() and docs/design.md.
        """
        returns = []
        advantages = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                # Only the terminal flag of the *last* transition can ever be
                # True in this single-episode rollout: every earlier stored
                # transition is, by construction, mid-episode.
                next_non_terminal = 1.0 - float(self.terminated[t])
                next_value_t = next_value
            else:
                next_non_terminal = 1.0
                next_value_t = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value_t * next_non_terminal
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def update(self, next_value, epochs=3):
        """Update policy and value network.

        Args:
            next_value: See compute_returns_and_advantages -- pass 0.0 if the
                rollout ended in a true termination, or V(next_state) if it
                ended only in a truncation.
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # Normalize advantages. unbiased=False deliberately: torch's default
        # (Bessel-corrected) std of a single-element tensor is 0/0 = NaN, with
        # only a warning to say so -- a one-transition rollout (a legal, if
        # unusual, edge case) would silently poison the whole update with
        # NaNs. The biased std of one element is exactly 0, which + 1e-8
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

        # PPO update
        for _ in range(epochs):
            means, stds, values = self.network(states)

            # Policy loss
            dist = torch.distributions.Normal(means, stds)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = ((values.squeeze() - returns) ** 2).mean()

            # Entropy bonus
            entropy = dist.entropy().sum(-1).mean()

            # Total loss
            loss = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        # Clear buffer
        self.clear_buffer()

    def clear_buffer(self):
        """Clear rollout buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.terminated.clear()

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

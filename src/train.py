"""Training script for PPO agent.

Run as a module from the repository root so the `src` package resolves:

    python -m src.train --num-episodes 1000
"""

import argparse
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.agent.ppo import PPOAgent
from src.environment.reaching_env import RobotReachEnv
from src.utils.running_normalizer import NormalizeObservation


def train(config):
    """Train PPO agent."""

    # Setup
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir / "logs")

    # Environment
    env = NormalizeObservation(RobotReachEnv(render_mode=None))

    # Agent
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config,
    )

    # Training loop
    total_steps = 0
    best_reward = -np.inf

    for episode in range(config["num_episodes"]):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Select action
            action, log_prob = agent.select_action(state)

            # Get value estimate
            value = agent.get_value(state)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # `terminated` (task resolved) and `truncated` (step budget ran
            # out) are kept apart all the way into the buffer: PPOAgent needs
            # the difference to bootstrap correctly at the end of the
            # episode. See docs/design.md.
            agent.store_transition(state, action, reward, value, log_prob, terminated)

            episode_reward += reward
            state = next_state
            total_steps += 1

        # Bootstrap value for whatever comes after the last stored
        # transition: nothing (0.0) if the task actually resolved, otherwise
        # the network's own estimate of the state we stopped at.
        next_value = 0.0 if terminated else agent.get_value(next_state)
        agent.update(next_value, epochs=config["ppo_epochs"])

        # Logging
        if (episode + 1) % config["log_interval"] == 0:
            print(
                f"Episode {episode + 1}/{config['num_episodes']} | "
                f"Reward: {episode_reward:.2f} | Steps: {total_steps}"
            )
            writer.add_scalar("reward/episode", episode_reward, episode)

        # Checkpointing
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(output_dir / "best_model.pt")

        if (episode + 1) % config["checkpoint_interval"] == 0:
            agent.save(output_dir / f"model_ep{episode + 1}.pt")

    env.close()
    writer.close()
    print(f"Training completed! Best reward: {best_reward:.2f}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/checkpoints")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(vars(args))

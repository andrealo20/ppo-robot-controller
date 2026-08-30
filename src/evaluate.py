"""Evaluate a trained PPO checkpoint.

Run as a module from the repository root:

    python -m src.evaluate --model-path experiments/checkpoints/best_model.pt
"""

import argparse

import numpy as np

from src.agent.ppo import PPOAgent
from src.environment.reaching_env import RobotReachEnv
from src.utils.running_normalizer import NormalizeObservation


def evaluate(model_path: str, episodes: int, render: bool, seed: int | None = None):
    """Run a trained agent for a fixed number of episodes and report reward.

    Observation statistics are frozen (update_stats=False): evaluation should
    measure the trained policy, not keep adapting the normalizer to whatever
    it sees during evaluation itself.
    """
    env = NormalizeObservation(
        RobotReachEnv(render_mode="human" if render else None),
        update_stats=False,
    )

    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    agent.load(model_path)

    episode_rewards = []
    successes = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # select_action already runs under torch.no_grad() internally.
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        successes += int(terminated)
        print(
            f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.2f} | "
            f"Resolved: {terminated}"
        )

    env.close()

    rewards = np.array(episode_rewards)
    print(
        f"\nMean reward: {rewards.mean():.2f} +/- {rewards.std():.2f} "
        f"over {episodes} episodes"
    )
    print(f"Resolved (reached target before truncation): {successes}/{episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.model_path, args.episodes, args.render, args.seed)

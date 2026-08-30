"""Evaluate a trained PPO checkpoint with frozen training normalization."""

import argparse
import warnings

import numpy as np

from src.agent.ppo import PPOAgent
from src.environment.reaching_env import RobotReachEnv
from src.utils.running_normalizer import NormalizeObservation


def evaluate(
    model_path: str,
    episodes: int,
    render: bool,
    seed: int | None = None,
    deterministic: bool = True,
    allow_legacy_checkpoint: bool = False,
):
    """Evaluate without adapting observation statistics.

    New checkpoints restore the exact running mean/variance used during
    training. Legacy network-only checkpoints are rejected by default because
    their normalizer statistics were never saved and therefore cannot be
    evaluated under the original input transformation.
    """
    env = NormalizeObservation(
        RobotReachEnv(render_mode="human" if render else None),
        update_stats=False,
    )
    agent = PPOAgent(env.observation_space, env.action_space)
    metadata = agent.load(model_path, observation_rms=env.rms)

    if not metadata["normalizer_restored"]:
        message = (
            "checkpoint does not contain observation-normalizer statistics; "
            "a faithful evaluation is impossible for legacy M0-M1.3 models"
        )
        if not allow_legacy_checkpoint:
            env.close()
            raise ValueError(
                message + ". Pass --allow-legacy-checkpoint to run anyway."
            )
        warnings.warn(message + "; using identity-initialized frozen statistics")

    episode_rewards = []
    successes = 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = agent.select_action(state, deterministic=deterministic)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        successes += int(terminated)
        print(
            f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.2f} | "
            f"Resolved: {terminated}"
        )

    env.close()
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    mode = "deterministic" if deterministic else "stochastic"
    print(f"\nPolicy mode: {mode}")
    print(
        f"Mean reward: {rewards.mean():.2f} +/- {rewards.std():.2f} "
        f"over {episodes} episodes"
    )
    print(f"Resolved: {successes}/{episodes} ({100.0 * successes / episodes:.1f}%)")
    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "deterministic": deterministic,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the policy instead of evaluating its deterministic mean action.",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Evaluate old network-only checkpoints despite missing training normalizer stats.",
    )
    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.episodes,
        args.render,
        args.seed,
        deterministic=not args.stochastic,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )

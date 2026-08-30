"""Training script for PPO agent.

Run as a module from the repository root so the `src` package resolves:

    python -m src.train --num-episodes 1000

The agent collects a fixed-size rollout (`--rollout-steps`) that can span
several episodes back to back, then runs multiple epochs of minibatched PPO
updates on it.
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent.ppo import PPOAgent
from src.environment.reaching_env import RobotReachEnv
from src.utils.running_normalizer import NormalizeObservation


def collect_rollout(env, agent, state, num_steps, episode_reward=0.0):
    """Collect exactly `num_steps` transitions into `agent`'s buffer,
    resetting `env` internally whenever an episode ends along the way.

    Args:
        episode_reward: running total for whatever episode is already in
            progress at `state` (0.0 if none -- i.e. `state` is a fresh
            reset). This has to be threaded in from the caller rather than
            reset to 0.0 on every call: a rollout boundary very often lands
            mid-episode (whenever rollout_steps isn't a multiple of the
            episode length, which is the common case), and the reward
            accumulated so far for that episode would otherwise be silently
            dropped.

    Returns:
        next_state: the observation to resume from on the *next* call (the
            environment is never reset after this function returns, even if
            the last stored transition happened to end an episode -- the
            caller resets lazily, on the next transition, exactly like a
            normal env/agent loop would).
        next_value: bootstrap value for `next_state`, valid to pass straight
            into `agent.update()` -- 0.0 if the rollout happened to end
            exactly on an episode boundary (unused there, since
            compute_returns_and_advantages ignores it in that case) and
            V(next_state) otherwise.
        completed_episode_rewards: total reward of every episode that
            started and fully finished (terminated or truncated) inside this
            call, in order. An episode straddling two calls to
            collect_rollout is only reported once it actually finishes.
        episode_reward: running total for whatever episode is still in
            progress when this call returns (0.0 if the rollout ended
            exactly on an episode boundary) -- pass this back in as
            `episode_reward` on the next call.
    """
    completed_episode_rewards = []

    for _ in range(num_steps):
        action, log_prob = agent.select_action(state)
        value = agent.get_value(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        boundary_value = agent.get_value(next_state) if truncated else 0.0
        agent.store_transition(
            state,
            action,
            reward,
            value,
            log_prob,
            terminated,
            truncated,
            boundary_value,
        )

        if terminated or truncated:
            completed_episode_rewards.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    # `state` is now the observation to resume from. If the rollout ended
    # exactly on an episode boundary, `state` is a fresh env.reset() and the
    # bootstrap value is irrelevant (compute_returns_and_advantages never
    # looks at `next_value` in that case). Otherwise it's the real next
    # value estimate for the ongoing episode.
    next_value = agent.get_value(state)
    return state, next_value, completed_episode_rewards, episode_reward


def train(config):
    """Train PPO agent."""

    # Seed NumPy (also governs PPOAgent.update's minibatch shuffling) and
    # PyTorch (network initialization and action sampling) before anything
    # that consumes randomness is constructed, so a given seed reproduces a
    # given run end to end. Left unseeded (None) by default.
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Setup
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir / "logs")

    # Environment
    env = NormalizeObservation(
        RobotReachEnv(render_mode=None, fixed_target=config.get("fixed_target"))
    )

    # Agent
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config,
    )

    # Seeding the first reset is enough for full env determinism: Gymnasium's
    # Env.reset(seed=...) stores the seeded RNG on the instance, and every
    # later unseeded reset() call (inside collect_rollout) keeps drawing from
    # that same, already-seeded generator.
    state, _ = env.reset(seed=seed)
    episode_reward_carry = 0.0
    total_steps = 0
    total_episodes = 0
    num_updates = config["num_episodes"] * 500 // config["rollout_steps"] + 1

    # Model selection uses a rolling mean over the most recent
    # `best_model_window` completed episodes rather than any single episode's
    # reward: each episode targets a different random goal, so one lucky
    # (easy, nearby) target can otherwise look like the "best" policy purely
    # by chance rather than by skill.
    window = config.get("best_model_window", 20)
    recent_rewards = deque(maxlen=window)
    best_rolling_mean = -np.inf

    for update_idx in range(num_updates):
        state, next_value, episode_rewards, episode_reward_carry = collect_rollout(
            env, agent, state, config["rollout_steps"], episode_reward_carry
        )
        total_steps += config["rollout_steps"]

        # Record completed-episode rewards and save the policy that actually
        # produced them -- this has to happen *before* agent.update() below,
        # which changes the network weights in place. Saving after the
        # update would checkpoint a different policy than the one being
        # scored.
        for ep_reward in episode_rewards:
            total_episodes += 1
            recent_rewards.append(ep_reward)
            if len(recent_rewards) == window:
                rolling_mean = float(np.mean(recent_rewards))
                if rolling_mean > best_rolling_mean:
                    best_rolling_mean = rolling_mean
                    agent.save(
                        output_dir / "best_model.pt",
                        observation_rms=env.rms,
                        config=config,
                        training_state={
                            "total_steps": total_steps,
                            "total_episodes": total_episodes,
                            "update_idx": update_idx,
                            "best_rolling_mean_reward": best_rolling_mean,
                            "best_model_window": window,
                        },
                    )

            if total_episodes % config["log_interval"] == 0:
                print(
                    f"Episode {total_episodes} (update {update_idx + 1}/{num_updates}) | "
                    f"Reward: {ep_reward:.2f} | Steps: {total_steps}"
                )
            writer.add_scalar("reward/episode", ep_reward, total_episodes)

        metrics = agent.update(
            next_value,
            epochs=config["ppo_epochs"],
            minibatch_size=config["minibatch_size"],
        )
        for name, value in metrics.items():
            writer.add_scalar(f"ppo/{name}", value, update_idx + 1)

        if (update_idx + 1) % config["checkpoint_interval"] == 0:
            agent.save(
                output_dir / f"model_update{update_idx + 1}.pt",
                observation_rms=env.rms,
                config=config,
                training_state={
                    "total_steps": total_steps,
                    "total_episodes": total_episodes,
                    "update_idx": update_idx + 1,
                    "best_rolling_mean_reward": best_rolling_mean,
                },
            )

        if total_episodes >= config["num_episodes"]:
            break

    env.close()
    writer.close()
    print(
        f"Training completed! Episodes: {total_episodes} | "
        f"Updates: {update_idx + 1} | Best {window}-episode rolling mean "
        f"reward: {best_rolling_mean:.2f}"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="Transitions collected per PPO update; may span multiple episodes.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Minibatch size within each rollout update. 0 disables "
        "minibatching (one full-batch gradient step per epoch).",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="experiments/checkpoints")
    parser.add_argument(
        "--fixed-target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Use one fixed target for a cheap PPO overfit diagnostic before full training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed NumPy, PyTorch, and the environment for a reproducible run. "
        "Unseeded by default.",
    )
    parser.add_argument(
        "--best-model-window",
        type=int,
        default=20,
        help="Number of most recent episodes averaged when deciding whether "
        "to overwrite best_model.pt.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    config = vars(args)
    config["minibatch_size"] = config["minibatch_size"] or None
    train(config)

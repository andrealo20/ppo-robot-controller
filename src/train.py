"""Training script for PPO agent.

Run as a module from the repository root so the `src` package resolves:

    python -m src.train --num-episodes 1000

M1 update: the agent now collects a fixed-size rollout (`--rollout-steps`)
that can span several episodes back to back, instead of updating once per
single episode. See docs/design.md, "Multi-episode rollouts", for why: the
M0-era single-episode/full-batch update showed no clear improving trend over
900 real training episodes, and this is the fix that was already flagged in
the README's M1 scope as something to revisit *if* sample efficiency turned
out to need it.
"""

import argparse
from pathlib import Path

import numpy as np
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
            dropped -- caught by
            tests/test_collect_rollout.py::test_ongoing_episode_state_
            carries_over_between_calls, which fails on exactly that number
            (an in-progress episode's reward truncated to whatever was
            collected only in the *second* call) if this isn't threaded
            through.

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

    state, _ = env.reset()
    episode_reward_carry = 0.0
    total_steps = 0
    total_episodes = 0
    best_reward = -np.inf
    num_updates = config["num_episodes"] * 500 // config["rollout_steps"] + 1

    for update_idx in range(num_updates):
        state, next_value, episode_rewards, episode_reward_carry = collect_rollout(
            env, agent, state, config["rollout_steps"], episode_reward_carry
        )
        total_steps += config["rollout_steps"]

        agent.update(
            next_value,
            epochs=config["ppo_epochs"],
            minibatch_size=config["minibatch_size"],
        )

        for ep_reward in episode_rewards:
            total_episodes += 1
            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(output_dir / "best_model.pt")

            if total_episodes % config["log_interval"] == 0:
                print(
                    f"Episode {total_episodes} (update {update_idx + 1}/{num_updates}) | "
                    f"Reward: {ep_reward:.2f} | Steps: {total_steps}"
                )
            writer.add_scalar("reward/episode", ep_reward, total_episodes)

        if (update_idx + 1) % config["checkpoint_interval"] == 0:
            agent.save(output_dir / f"model_update{update_idx + 1}.pt")

        if total_episodes >= config["num_episodes"]:
            break

    env.close()
    writer.close()
    print(
        f"Training completed! Episodes: {total_episodes} | "
        f"Updates: {update_idx + 1} | Best reward: {best_reward:.2f}"
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
        "minibatching (one full-batch gradient step per epoch, the M0 "
        "behaviour).",
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
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    config = vars(args)
    config["minibatch_size"] = config["minibatch_size"] or None
    train(config)

"""GAE / bootstrap correctness tests.

These pin down the exact bug this repository had: every episode ended by
timing out (RobotReachEnv only sets `terminated` on success), but the
training loop always passed `next_value=0.0` into the update, and the buffer
stored a single `done` flag that conflated termination with truncation. Both
of those forced every truncated rollout to be treated as if the episode had
truly ended -- silently biasing the value target low on every update.

The values below are hand-derived from the GAE recursion, not copied from the
implementation, so a regression that reintroduces either half of the bug
changes a returned number here rather than just changing behaviour nobody is
watching. See docs/design.md for the full derivation and the numbers this
file's constants were built from.
"""

import math

import pytest
from gymnasium import spaces

from src.agent.ppo import PPOAgent

GAMMA = 0.99
GAE_LAMBDA = 0.95


def make_agent():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    return PPOAgent(
        obs_space, action_space, config={"gamma": GAMMA, "gae_lambda": GAE_LAMBDA}
    )


def load_two_step_rollout(agent, terminated_last):
    """A fixed 2-step rollout: rewards [1.0, 2.0], values [0.5, 0.5]."""
    agent.rewards = [1.0, 2.0]
    agent.values = [0.5, 0.5]
    agent.terminated = [False, terminated_last]
    # states/actions/log_probs are irrelevant to compute_returns_and_advantages
    agent.states = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    agent.actions = [[0.0], [0.0]]
    agent.log_probs = [0.0, 0.0]


def hand_computed_gae(rewards, values, terminated_last, next_value):
    """Reference implementation, written independently of PPOAgent's."""
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    values_ext = values + [next_value]

    for t in reversed(range(n)):
        if t == n - 1:
            next_non_terminal = 0.0 if terminated_last else 1.0
            next_value_t = next_value
        else:
            next_non_terminal = 1.0
            next_value_t = values[t + 1]

        delta = rewards[t] + GAMMA * next_value_t * next_non_terminal - values_ext[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values_ext[t]

    return returns, advantages


@pytest.mark.parametrize(
    "terminated_last, next_value",
    [
        (True, 0.0),  # true termination: must bootstrap with exactly 0.0
        (False, 0.7),  # truncation: must bootstrap with the network's own estimate
    ],
)
def test_matches_hand_derived_reference(terminated_last, next_value):
    agent = make_agent()
    load_two_step_rollout(agent, terminated_last)

    returns, advantages = agent.compute_returns_and_advantages(next_value)

    expected_returns, expected_advantages = hand_computed_gae(
        agent.rewards, agent.values, terminated_last, next_value
    )

    for i in range(2):
        assert math.isclose(returns[i].item(), expected_returns[i], abs_tol=1e-4)
        assert math.isclose(advantages[i].item(), expected_advantages[i], abs_tol=1e-4)


def test_truncation_and_termination_give_different_bootstrap():
    """The bug this pins down: a real implementation MUST treat these
    differently. A version that (like the original code) always bootstraps
    with 0.0 regardless of why the episode ended would make this fail, since
    the two rollouts are identical except for the final terminated flag and
    the bootstrap value passed in.
    """
    agent_terminated = make_agent()
    load_two_step_rollout(agent_terminated, terminated_last=True)
    returns_terminated, _ = agent_terminated.compute_returns_and_advantages(0.0)

    agent_truncated = make_agent()
    load_two_step_rollout(agent_truncated, terminated_last=False)
    returns_truncated, _ = agent_truncated.compute_returns_and_advantages(0.7)

    assert not math.isclose(
        returns_terminated[-1].item(), returns_truncated[-1].item(), abs_tol=1e-3
    )
    assert not math.isclose(
        returns_terminated[0].item(), returns_truncated[0].item(), abs_tol=1e-3
    )


def test_zero_reward_zero_value_trajectory_is_zero():
    """Degenerate case, useful as a sanity check on its own: an all-zero
    rollout that terminates truly must return all zeros -- there is nothing
    for GAE to propagate.
    """
    agent = make_agent()
    agent.rewards = [0.0, 0.0]
    agent.values = [0.0, 0.0]
    agent.terminated = [False, True]
    agent.states = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    agent.actions = [[0.0], [0.0]]
    agent.log_probs = [0.0, 0.0]

    returns, advantages = agent.compute_returns_and_advantages(0.0)

    for i in range(2):
        assert math.isclose(returns[i].item(), 0.0, abs_tol=1e-6)
        assert math.isclose(advantages[i].item(), 0.0, abs_tol=1e-6)

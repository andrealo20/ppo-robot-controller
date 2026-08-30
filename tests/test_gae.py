"""GAE / bootstrap correctness tests.

Two bugs are pinned down here, both found by comparing against a hand-derived
reference (written independently of PPOAgent's own recursion, not copied from
it) rather than by reading the code and agreeing with it.

1. The original repository bug (M0): every episode ended by timing out
   (RobotReachEnv only sets `terminated` on success), but the training loop
   always passed `next_value=0.0` into the update, and the buffer stored a
   single `done` flag that conflated termination with truncation. Both
   effects forced every truncated rollout to bootstrap as if the episode had
   truly ended, biasing the value target low on every single update.

2. The multi-episode rollout bug this design invites, if you are not careful
   (M1): once a single buffer can hold several episodes back to back (see
   docs/design.md, "Multi-episode rollouts"), the GAE recursion must be cut
   at *every* episode boundary inside the buffer, not just at the last
   transition -- otherwise one episode's advantage silently leaks backward
   into the previous episode's, through the `gae` accumulator.
   `test_second_episode_does_not_leak_into_first` pins that down with a
   4-transition, 2-episode buffer where the leak (if present) changes a
   number that has nothing to do with the second episode.

A consequence of (2) worth spelling out because it changes what the
`next_value` argument means relative to M0: `next_value` (passed to
`compute_returns_and_advantages`/`update`) is now used *only* when the last
buffer transition does not itself end an episode -- i.e. the rollout was cut
off purely because the buffer's step budget ran out, mid-episode. If the
buffer's last transition *is* an episode boundary (terminated or truncated),
the bootstrap comes from that transition's own `boundary_value` (computed by
the caller at collection time, exactly like every other episode boundary in
the buffer), and `next_value` is ignored. `test_next_value_is_ignored_at_an_
episode_boundary` pins that contract down explicitly.

See docs/design.md for the full derivation and the numbers these tests were
built from.
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


def load_rollout(agent, rewards, values, terminated, episode_end, boundary_value):
    """Directly populate an agent's buffer for a hand-checkable rollout.
    states/actions/log_probs are irrelevant to compute_returns_and_advantages.
    """
    n = len(rewards)
    agent.rewards = list(rewards)
    agent.values = list(values)
    agent.terminated = list(terminated)
    agent.episode_end = list(episode_end)
    agent.boundary_value = list(boundary_value)
    agent.states = [[0.0, 0.0, 0.0]] * n
    agent.actions = [[0.0]] * n
    agent.log_probs = [0.0] * n


def load_two_step_episode(agent, terminated_last, boundary_value=0.0):
    """A fixed 2-step, single-episode rollout filling the whole buffer:
    rewards [1.0, 2.0], values [0.5, 0.5], ending at t=1 either in a true
    termination or a truncation with the given boundary_value."""
    load_rollout(
        agent,
        rewards=[1.0, 2.0],
        values=[0.5, 0.5],
        terminated=[False, terminated_last],
        episode_end=[False, True],
        boundary_value=[0.0, 0.0 if terminated_last else boundary_value],
    )


def hand_computed_gae(rewards, values, terminated_last, bootstrap):
    """Reference implementation for a single-episode rollout that fills the
    whole buffer (so its last transition is always the episode's own
    boundary), written independently of PPOAgent's."""
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    values_ext = values + [bootstrap]

    for t in reversed(range(n)):
        if t == n - 1:
            next_non_terminal = 0.0 if terminated_last else 1.0
            next_value_t = bootstrap
        else:
            next_non_terminal = 1.0
            next_value_t = values[t + 1]

        delta = rewards[t] + GAMMA * next_value_t * next_non_terminal - values_ext[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values_ext[t]

    return returns, advantages


@pytest.mark.parametrize(
    "terminated_last, boundary_value",
    [
        (True, 0.0),  # true termination: must bootstrap with exactly 0.0
        (False, 0.7),  # truncation: must bootstrap with the precomputed boundary value
    ],
)
def test_matches_hand_derived_reference(terminated_last, boundary_value):
    agent = make_agent()
    load_two_step_episode(agent, terminated_last, boundary_value)

    # next_value is irrelevant here (the buffer's last transition is itself
    # the episode boundary) -- passed as an implausible sentinel to make that
    # explicit; see test_next_value_is_ignored_at_an_episode_boundary below
    # for a direct test of this contract.
    returns, advantages = agent.compute_returns_and_advantages(next_value=999.0)

    expected_returns, expected_advantages = hand_computed_gae(
        agent.rewards, agent.values, terminated_last, boundary_value
    )

    for i in range(2):
        assert math.isclose(returns[i].item(), expected_returns[i], abs_tol=1e-4)
        assert math.isclose(advantages[i].item(), expected_advantages[i], abs_tol=1e-4)


def test_truncation_and_termination_give_different_bootstrap():
    """The M0 bug this pins down: a real implementation MUST treat these
    differently. A version that (like the original code) always bootstraps
    with 0.0 regardless of why the episode ended would make this fail, since
    the two rollouts are identical except for the final terminated flag and
    the boundary value.
    """
    agent_terminated = make_agent()
    load_two_step_episode(agent_terminated, terminated_last=True)
    returns_terminated, _ = agent_terminated.compute_returns_and_advantages(0.0)

    agent_truncated = make_agent()
    load_two_step_episode(agent_truncated, terminated_last=False, boundary_value=0.7)
    returns_truncated, _ = agent_truncated.compute_returns_and_advantages(0.0)

    assert not math.isclose(
        returns_terminated[-1].item(), returns_truncated[-1].item(), abs_tol=1e-3
    )
    assert not math.isclose(
        returns_terminated[0].item(), returns_truncated[0].item(), abs_tol=1e-3
    )


def test_next_value_is_ignored_at_an_episode_boundary():
    """If the buffer's last transition is itself an episode boundary
    (terminated or truncated), the external `next_value` argument must be
    completely ignored -- the bootstrap comes from that transition's own
    `boundary_value` (or 0.0, if terminated) instead. Two calls that differ
    only in `next_value` must return identical results.
    """
    agent = make_agent()
    load_two_step_episode(agent, terminated_last=False, boundary_value=0.7)

    returns_a, advantages_a = agent.compute_returns_and_advantages(next_value=0.0)
    returns_b, advantages_b = agent.compute_returns_and_advantages(next_value=123.0)

    for i in range(2):
        assert math.isclose(returns_a[i].item(), returns_b[i].item(), abs_tol=1e-6)
        assert math.isclose(
            advantages_a[i].item(), advantages_b[i].item(), abs_tol=1e-6
        )


def test_next_value_bootstraps_an_ongoing_episode_cut_by_the_buffer():
    """The other half of the contract: when the buffer's last transition is
    NOT an episode boundary (the rollout was cut off purely because the step
    budget ran out, mid-episode -- the normal case with rollout_steps <
    episode length), `next_value` is exactly what gets bootstrapped, the same
    way `values[t + 1]` would for any other interior transition.
    """
    agent = make_agent()
    load_rollout(
        agent,
        rewards=[1.0, 2.0],
        values=[0.5, 0.5],
        terminated=[False, False],
        episode_end=[False, False],  # episode is still ongoing past the buffer
        boundary_value=[0.0, 0.0],
    )

    returns, advantages = agent.compute_returns_and_advantages(next_value=0.7)

    expected_returns, expected_advantages = hand_computed_gae(
        agent.rewards, agent.values, terminated_last=False, bootstrap=0.7
    )
    for i in range(2):
        assert math.isclose(returns[i].item(), expected_returns[i], abs_tol=1e-4)
        assert math.isclose(advantages[i].item(), expected_advantages[i], abs_tol=1e-4)


def test_zero_reward_zero_value_trajectory_is_zero():
    """Degenerate case, useful as a sanity check on its own: an all-zero
    rollout that terminates truly must return all zeros -- there is nothing
    for GAE to propagate.
    """
    agent = make_agent()
    load_two_step_episode(agent, terminated_last=True)
    agent.rewards = [0.0, 0.0]
    agent.values = [0.0, 0.0]

    returns, advantages = agent.compute_returns_and_advantages(0.0)

    for i in range(2):
        assert math.isclose(returns[i].item(), 0.0, abs_tol=1e-6)
        assert math.isclose(advantages[i].item(), 0.0, abs_tol=1e-6)


def test_second_episode_does_not_leak_into_first():
    """A 4-transition buffer holding two episodes back to back:

        t=0,1: episode A -- t=1 truncates (step budget), boundary_value=0.4
        t=2,3: episode B -- t=3 terminates (task resolved)

    Episode A's advantages must come out identical to a *standalone* 2-step
    truncated episode with the same rewards/values and the same boundary
    value -- whatever happens in episode B (t=2, t=3) must not change them.
    A GAE recursion that forgets to cut at t=1 (treating episode B as a
    continuation of episode A instead of a new trajectory) would leak episode
    B's advantage backward into t=1 and t=0, failing this test without
    touching test_matches_hand_derived_reference at all -- that test never
    has more than one episode in the buffer to leak from.
    """
    agent = make_agent()
    load_rollout(
        agent,
        rewards=[1.0, 2.0, -0.5, 3.0],
        values=[0.5, 0.5, 0.2, 0.2],
        terminated=[False, False, False, True],
        episode_end=[False, True, False, True],
        boundary_value=[0.0, 0.4, 0.0, 0.0],
    )

    returns, advantages = agent.compute_returns_and_advantages(next_value=999.0)

    ref_agent = make_agent()
    load_two_step_episode(ref_agent, terminated_last=False, boundary_value=0.4)
    ref_returns, ref_advantages = ref_agent.compute_returns_and_advantages(
        next_value=999.0
    )

    for i in range(2):
        assert math.isclose(returns[i].item(), ref_returns[i].item(), abs_tol=1e-4)
        assert math.isclose(
            advantages[i].item(), ref_advantages[i].item(), abs_tol=1e-4
        )

    # Sanity check on episode B in isolation (a standalone terminated 2-step
    # episode with the same rewards/values) -- confirms the buffer's second
    # half is computed correctly too, not just left untouched.
    ref_b = make_agent()
    load_two_step_episode(ref_b, terminated_last=True)
    ref_b.rewards = [-0.5, 3.0]
    ref_b.values = [0.2, 0.2]
    ref_b_returns, ref_b_advantages = ref_b.compute_returns_and_advantages(
        next_value=0.0
    )
    for i in range(2):
        assert math.isclose(
            returns[2 + i].item(), ref_b_returns[i].item(), abs_tol=1e-4
        )
        assert math.isclose(
            advantages[2 + i].item(), ref_b_advantages[i].item(), abs_tol=1e-4
        )

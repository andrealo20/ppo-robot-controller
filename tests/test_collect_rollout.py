"""Tests for src.train.collect_rollout: the bookkeeping that lets a single
rollout buffer span multiple episodes (see docs/design.md, "Multi-episode
rollouts"). Uses a tiny scripted fake environment instead of RobotReachEnv --
this is about the loop's bookkeeping (does it reset at the right time, does
it report the right episode rewards, does it call store_transition with the
right flags), not about PyBullet.
"""

import numpy as np
from gymnasium import spaces

from src.agent.ppo import PPOAgent
from src.train import collect_rollout


class ScriptedEnv:
    """A minimal Gymnasium-like env whose episodes end at a fixed length,
    alternating between termination and truncation so both boundary types
    get exercised. Reward is always 1.0 per step, so an episode's total
    reward is just its length -- easy to check by eye.
    """

    def __init__(self, episode_length=3):
        self.episode_length = episode_length
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self._step_in_episode = 0
        self._episode_count = 0

    def reset(self, seed=None, options=None):
        self._step_in_episode = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._step_in_episode += 1
        done = self._step_in_episode >= self.episode_length
        if done:
            self._episode_count += 1
            # Alternate: even-numbered episodes (0, 2, ...) terminate (task
            # resolved), odd-numbered ones truncate (step budget ran out).
            terminated = self._episode_count % 2 == 1
            truncated = not terminated
        else:
            terminated = truncated = False
        return np.zeros(3, dtype=np.float32), 1.0, terminated, truncated, {}


def make_agent(obs_dim=3, action_dim=1):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))
    return PPOAgent(obs_space, action_space, config={"lr": 1e-3})


def test_collect_rollout_stores_exactly_num_steps():
    env = ScriptedEnv(episode_length=3)
    agent = make_agent()
    state, _ = env.reset()

    next_state, next_value, _episode_rewards, _carry = collect_rollout(
        env, agent, state, num_steps=10
    )

    assert len(agent.rewards) == 10
    assert isinstance(next_state, np.ndarray)
    assert isinstance(next_value, float)


def test_collect_rollout_reports_completed_episodes_in_order():
    env = ScriptedEnv(episode_length=3)
    agent = make_agent()
    state, _ = env.reset()

    # 10 steps of a 3-step episode: 3 episodes complete (steps 3, 6, 9), one
    # is left one step short (started at step 9, one transition stored).
    _, _, episode_rewards, _carry = collect_rollout(env, agent, state, num_steps=10)

    assert episode_rewards == [3.0, 3.0, 3.0]


def test_episode_boundaries_are_flagged_with_the_right_terminated_truncated():
    env = ScriptedEnv(episode_length=2)
    agent = make_agent()
    state, _ = env.reset()

    # 4 steps = exactly 2 complete episodes: episode 1 terminates (odd
    # count), episode 2 truncates (even count) -- see ScriptedEnv.step.
    collect_rollout(env, agent, state, num_steps=4)

    assert agent.episode_end == [False, True, False, True]
    assert agent.terminated == [False, True, False, False]
    # The truncation boundary (t=3) must carry the network's own value
    # estimate for the real next state, not the placeholder 0.0 a
    # termination boundary gets. ScriptedEnv always returns an all-zero next
    # observation, so this is directly checkable against a fresh
    # get_value() call (the network hasn't been updated in between, so it's
    # deterministic).
    assert agent.boundary_value[1] == 0.0  # t=1 terminated, not truncated
    expected_boundary_value = agent.get_value(np.zeros(3, dtype=np.float32))
    assert abs(agent.boundary_value[3] - expected_boundary_value) < 1e-6


def test_ongoing_episode_state_carries_over_between_calls():
    """If the rollout is cut off mid-episode, the next call must resume from
    the returned state rather than resetting -- otherwise every rollout
    boundary would silently truncate an episode that hadn't actually ended.
    """
    env = ScriptedEnv(episode_length=5)
    agent = make_agent()
    state, _ = env.reset()

    # First call stops after 3 steps, mid-episode (episode_length=5).
    state, _, episode_rewards_1, carry = collect_rollout(env, agent, state, num_steps=3)
    assert episode_rewards_1 == []
    assert carry == 3.0  # 3 steps of reward 1.0 each, not yet reported
    assert env._step_in_episode == 3  # env was never reset

    # Second call finishes that same episode (2 more steps, using the
    # carried-over reward) then starts a new one for 3 more steps.
    agent.clear_buffer()
    _, _, episode_rewards_2, _carry2 = collect_rollout(
        env, agent, state, num_steps=5, episode_reward=carry
    )
    assert episode_rewards_2 == [5.0]  # the episode that was in progress

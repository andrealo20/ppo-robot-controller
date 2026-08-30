"""End-to-end smoke tests for PPOAgent.

test_agent_can_be_constructed alone would have caught the repository's
original bug: `from .network import PolicyValueNetwork` inside
src/agent/ppo.py resolved to the nonexistent module src.agent.network, so
PPOAgent.__init__ raised ModuleNotFoundError before doing anything else.
Nothing in the repository ever exercised that line.
"""

import numpy as np
from gymnasium import spaces

from src.agent.ppo import PPOAgent


def make_spaces():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return obs_space, action_space


def test_agent_can_be_constructed():
    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space)
    assert agent.network is not None


def test_select_action_respects_action_bounds():
    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space)
    state = np.zeros(6, dtype=np.float32)

    action, log_prob = agent.select_action(state)

    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert isinstance(log_prob, float)


def test_update_runs_end_to_end_and_clears_buffer():
    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space, config={"lr": 1e-3})

    state = np.zeros(6, dtype=np.float32)
    for terminated in (False, False, True):
        action, log_prob = agent.select_action(state)
        value = agent.get_value(state)
        agent.store_transition(
            state,
            action,
            reward=1.0,
            value=value,
            log_prob=log_prob,
            terminated=terminated,
        )

    agent.update(next_value=0.0, epochs=2)

    assert agent.states == []
    assert agent.rewards == []
    assert agent.terminated == []


def test_save_and_load_round_trip(tmp_path):
    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space)
    state = np.zeros(6, dtype=np.float32)

    value_before = agent.get_value(state)
    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)

    reloaded = PPOAgent(obs_space, action_space)
    reloaded.load(checkpoint)
    value_after = reloaded.get_value(state)

    assert value_before == value_after


def test_update_with_epochs_zero_still_clears_buffer():
    """epochs=0 is a degenerate but legal config; the buffer must still be
    cleared or the next episode's rollout would silently include stale data.
    """
    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space)
    state = np.zeros(6, dtype=np.float32)

    action, log_prob = agent.select_action(state)
    value = agent.get_value(state)
    agent.store_transition(
        state, action, reward=0.0, value=value, log_prob=log_prob, terminated=True
    )

    agent.update(next_value=0.0, epochs=0)

    assert agent.states == []

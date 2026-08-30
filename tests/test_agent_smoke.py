"""End-to-end smoke tests for PPOAgent."""

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


def test_deterministic_action_is_repeatable_and_bounded():
    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space)
    state = np.array([0.3, -0.2, 0.5, 0.1, 0.2, 0.3], dtype=np.float32)

    action_a, _ = agent.select_action(state, deterministic=True)
    action_b, _ = agent.select_action(state, deterministic=True)

    assert np.array_equal(action_a, action_b)
    assert np.all(action_a >= action_space.low)
    assert np.all(action_a <= action_space.high)


def test_checkpoint_restores_normalizer_and_optimizer(tmp_path):
    from src.utils.running_normalizer import RunningMeanStd

    obs_space, action_space = make_spaces()
    agent = PPOAgent(obs_space, action_space, config={"lr": 3e-4})
    rms = RunningMeanStd(shape=obs_space.shape)
    rms.update(np.arange(60, dtype=np.float64).reshape(10, 6))

    checkpoint = tmp_path / "complete.pt"
    agent.save(
        checkpoint,
        observation_rms=rms,
        config={"lr": 3e-4, "tag": "test"},
        training_state={"total_steps": 123},
    )

    reloaded = PPOAgent(obs_space, action_space, config={"lr": 3e-4})
    restored_rms = RunningMeanStd(shape=obs_space.shape)
    metadata = reloaded.load(
        checkpoint, observation_rms=restored_rms, load_optimizer=True
    )

    assert metadata["normalizer_restored"] is True
    assert metadata["training_state"]["total_steps"] == 123
    assert metadata["config"]["tag"] == "test"
    assert np.array_equal(restored_rms.mean, rms.mean)
    assert np.array_equal(restored_rms.var, rms.var)
    assert restored_rms.count == rms.count

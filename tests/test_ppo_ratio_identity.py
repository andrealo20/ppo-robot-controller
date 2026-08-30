"""PPO's importance ratio must be one before any policy update."""

import numpy as np
import torch
from gymnasium import spaces

from src.agent.ppo import PPOAgent


def make_agent():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return PPOAgent(obs_space, action_space)


def ratio_before_any_update(agent):
    states = torch.as_tensor(
        np.asarray(agent.states), dtype=torch.float32, device=agent.device
    )
    actions = torch.as_tensor(
        np.asarray(agent.actions), dtype=torch.float32, device=agent.device
    )
    old_log_probs = torch.as_tensor(
        agent.log_probs, dtype=torch.float32, device=agent.device
    )
    with torch.no_grad():
        new_log_probs, _, _ = agent.evaluate_actions(states, actions)
    return torch.exp(new_log_probs - old_log_probs).cpu()


def _collect(agent, log_std, n=200):
    with torch.no_grad():
        agent.network.log_std.fill_(log_std)
    state = np.zeros(6, dtype=np.float32)
    for _ in range(n):
        action, log_prob = agent.select_action(state)
        agent.store_transition(
            state,
            action,
            0.0,
            0.0,
            log_prob,
            terminated=False,
            truncated=True,
            boundary_value=0.0,
        )


def test_ratio_is_one_with_small_std():
    agent = make_agent()
    _collect(agent, -4.0)
    ratio = ratio_before_any_update(agent)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_ratio_is_one_with_large_std_and_saturated_tanh_actions():
    agent = make_agent()
    _collect(agent, 1.0, n=500)
    actions = np.asarray(agent.actions)
    # Ensure the stress case actually includes actions close to the bounds.
    assert np.mean(np.any(np.abs(actions) > 0.99, axis=1)) > 0.25
    ratio = ratio_before_any_update(agent)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4)


def test_sabotaged_old_log_prob_is_detected():
    agent = make_agent()
    _collect(agent, 0.0, n=32)
    agent.log_probs[0] += 0.5
    ratio = ratio_before_any_update(agent)
    assert not torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4)

"""PolicyValueNetwork tests: output shapes, and that log_std is actually
clamped rather than merely documented as such.
"""

import math

import torch

from src.network.policy_value import LOG_STD_MAX, LOG_STD_MIN, PolicyValueNetwork


def test_forward_output_shapes():
    net = PolicyValueNetwork(obs_dim=6, action_dim=2)
    obs = torch.zeros(4, 6)  # batch of 4

    mean, std, value = net(obs)

    assert mean.shape == (4, 2)
    assert std.shape == (4, 2)
    assert value.shape == (4, 1)


def test_log_std_is_clamped_when_pushed_out_of_range():
    net = PolicyValueNetwork(obs_dim=3, action_dim=2)

    with torch.no_grad():
        net.log_std.fill_(1000.0)  # a bad gradient step could do this
    _, std_high, _ = net(torch.zeros(1, 3))
    assert torch.allclose(
        std_high, torch.full_like(std_high, math.exp(LOG_STD_MAX)), atol=1e-3
    )

    with torch.no_grad():
        net.log_std.fill_(-1000.0)
    _, std_low, _ = net(torch.zeros(1, 3))
    assert torch.allclose(
        std_low, torch.full_like(std_low, math.exp(LOG_STD_MIN)), atol=1e-6
    )


def test_log_std_within_bounds_stays_unchanged():
    """The clamp must not distort values that are already inside the range."""
    net = PolicyValueNetwork(obs_dim=3, action_dim=2)

    with torch.no_grad():
        net.log_std.fill_(0.5)
    _, std, _ = net(torch.zeros(1, 3))
    assert torch.allclose(std, torch.full_like(std, math.exp(0.5)), atol=1e-6)

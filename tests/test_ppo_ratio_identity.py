"""PPO ratio identity: before any gradient step, the importance-sampling
ratio exp(log_pi_new(a) - log_pi_old(a)) must equal 1.0 for every stored
transition, because pi_new and pi_old are the same network evaluated at the
same sampled action `a`.

This is a diagnostic Andrea's independent review of the repository proposed
(see docs/design.md): if the ratio is not 1 here, `old_log_probs` and the
`a` PPO actually re-scores at update time disagree about what was sampled --
exactly what happens when select_action()'s np.clip() silently diverges from
the log_prob it reports (log_prob is computed on the raw sampled action;
select_action then returns the clipped action, which is what gets stored and
later re-scored).
"""

import numpy as np
import torch
from gymnasium import spaces

from src.agent.ppo import PPOAgent


def make_agent():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return PPOAgent(obs_space, action_space)


def ratio_before_any_update(agent):
    """Replicates exactly the ratio computation _gradient_step performs,
    without taking an optimizer step -- so this measures whether pi_old and
    pi_new (same weights) agree, not whether training is stable."""
    states = torch.FloatTensor(np.array(agent.states))
    actions = torch.FloatTensor(np.array(agent.actions))
    old_log_probs = torch.FloatTensor(agent.log_probs)

    means, stds, _ = agent.network(states)
    dist = torch.distributions.Normal(means, stds)
    new_log_probs = dist.log_prob(actions).sum(-1)
    return torch.exp(new_log_probs - old_log_probs)


def test_ratio_is_one_when_std_is_small_and_clipping_is_rare():
    """Control case: with a tiny std, sampled actions almost never land
    outside [-1, 1], so clipping almost never fires and the ratio should be
    ~1 regardless of whether the clip/log_prob bug exists. This isolates the
    failure to clipping specifically, rather than to select_action/
    store_transition/update in general.
    """
    agent = make_agent()
    with torch.no_grad():
        agent.network.log_std.fill_(-10.0)  # std ~= exp(-10), clipping ~never fires

    state = np.zeros(6, dtype=np.float32)
    for _ in range(200):
        action, log_prob = agent.select_action(state)
        agent.store_transition(
            state,
            action,
            reward=0.0,
            value=0.0,
            log_prob=log_prob,
            terminated=False,
            truncated=True,
            boundary_value=0.0,
        )

    ratio = ratio_before_any_update(agent)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)


def test_ratio_is_one_when_std_is_large_and_clipping_is_common():
    """The actual failure case: with std at the trained M1/M1.1/M1.2 ceiling
    (LOG_STD_MAX=2.0, std ~= 7.4), sampled raw actions land outside [-1, 1]
    on most draws, so clipping fires on most transitions. Same identity as
    the control case above -- pi_old and pi_new are still the same network,
    still evaluated at whatever `a` was actually stored -- so this must also
    hold if select_action/store_transition are internally consistent.
    """
    agent = make_agent()
    with torch.no_grad():
        agent.network.log_std.fill_(2.0)  # matches LOG_STD_MAX, the trained ceiling

    state = np.zeros(6, dtype=np.float32)
    clipped_count = 0
    n = 200
    for _ in range(n):
        action, log_prob = agent.select_action(state)
        if np.any(np.abs(action) >= 1.0 - 1e-6):
            clipped_count += 1
        agent.store_transition(
            state,
            action,
            reward=0.0,
            value=0.0,
            log_prob=log_prob,
            terminated=False,
            truncated=True,
            boundary_value=0.0,
        )

    # Sanity check on the setup itself: this test is only meaningful if
    # clipping actually fired on most of the 200 draws.
    assert clipped_count / n >= 0.5, (
        f"only {clipped_count}/{n} actions were clipped -- std=exp(2.0) "
        f"should clip on the large majority of draws against a [-1, 1] "
        f"action space; the test setup, not the hypothesis, would be wrong "
        f"if this fails"
    )

    ratio = ratio_before_any_update(agent)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

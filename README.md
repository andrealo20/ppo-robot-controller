# PPO Robot Controller

[![CI](https://github.com/andrealo20/ppo-robot-controller/actions/workflows/ci.yml/badge.svg)](https://github.com/andrealo20/ppo-robot-controller/actions/workflows/ci.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![language: Python 3.10+](https://img.shields.io/badge/language-Python%203.10%2B-blue.svg)](requirements.txt)
[![tests: 21](https://img.shields.io/badge/tests-21-green.svg)](tests/)
[![status: M0](https://img.shields.io/badge/status-M0-orange.svg)](docs/design.md)

**A from-scratch PPO implementation (PyTorch, no RL library) for a PyBullet
reaching task, with a hand-derived GAE bootstrap that is tested against an
independent reference and a mutation check.**

## Status

M0: the plumbing is correct and tested, not a trained agent yet. `python -m
src.train` and `python -m src.evaluate` both run end to end against the real
environment; no policy has been trained to convergence, so there is no
headline reward number here to report honestly. `docs/design.md` records
what M0 actually was: four real bugs found while getting from a scaffold to
something that runs and is tested, including one (a truncated-episode
bootstrap defect) that would not have shown up as a crash, only as a policy
that trained worse than it should have for no visible reason.

## Milestones

- **M0 — plumbing correct, tested, documented (this milestone).** Fixed a
  broken import that prevented the agent from being constructed at all, a
  module-invocation bug that broke the documented entry point independently
  of that, a bug that bootstrapped every truncated episode as if it had
  terminated (see `docs/design.md` for the derivation and the test that pins
  it down), and an observation-shape mismatch (7 values returned against a
  declared 6) that only the test suite caught. Added observation
  normalization, a clamp on the policy's log standard deviation, and 21
  tests covering the agent, the network, the environment's
  terminated/truncated contract, and the normalizer.
- **M1 — first trained policy, not yet started.** Train to convergence on
  the reaching task, report the actual reward curve and success rate
  (including if it plateaus somewhere unflattering), and revisit the
  single-episode full-batch update in `PPOAgent.update()` if sample
  efficiency turns out to need it.

## How it works

`RobotReachEnv` (PyBullet, Gymnasium API) rewards the negative planar
distance to a random target, with a bonus for reaching it. The "robot" is a
fixed-base `r2d2.urdf` mesh whose base position/velocity is set directly
rather than driven by joint torques or forces — see Limitations below for
what that does and doesn't mean for any results trained on it.

`PPOAgent` implements PPO from the ground up: a shared-trunk actor-critic
network (`PolicyValueNetwork`), clipped surrogate policy loss, GAE advantage
estimation, and entropy regularization. No RL library (Stable-Baselines3,
RLlib, ...) is used for the algorithm itself — only PyTorch, Gymnasium (for
the environment API) and PyBullet (for the physics).

<p align="center">
  <img src="assets/reaching_env.png" alt="RobotReachEnv rendered in PyBullet: the r2d2 stand-in robot and a red target marker on a checkered ground plane" width="600">
</p>

<p align="center"><sub>PyBullet render of a real reset of <code>RobotReachEnv</code>.
The target is drawn as an enlarged red marker for visibility — the actual
<code>sphere_small.urdf</code> collision/visual target is a few millimetres
across at this scale and would not read as a dot in a screenshot.</sub></p>

## Installation

```sh
git clone https://github.com/andrealo20/ppo-robot-controller.git
cd ppo-robot-controller
pip install -r requirements.txt
```

## Quick start

Run everything as a module, from the repository root — see `docs/design.md`
for why `python src/train.py` (the natural-looking command) does not work:

```sh
python -m src.train --num-episodes 1000
python -m src.evaluate --model-path experiments/checkpoints/best_model.pt --episodes 20
python -m pytest tests -q
```

## Repository layout

```
src/
├── environment/    reaching_env.py — the PyBullet/Gymnasium environment
├── agent/          ppo.py — the PPO agent (rollout buffer, GAE, clipped update)
├── network/        policy_value.py — shared-trunk actor-critic network
├── utils/          running_normalizer.py — running observation normalization
├── train.py        training entry point (python -m src.train)
└── evaluate.py      evaluation entry point (python -m src.evaluate)

tests/              21 tests: GAE reference + mutation check, network clamp,
                    agent smoke tests, environment terminated/truncated
                    contract, running normalizer
docs/design.md      what was found, what was fixed, and why — including the
                    truncated-episode bootstrap bug in full
conftest.py         empty; exists only so pytest puts the repo root on
                    sys.path (see docs/design.md)
experiments/        checkpoints and TensorBoard logs land here at runtime
```

## Limitations

Stated here rather than left to be discovered.

- **The reaching task is kinematic, not dynamic.** The robot's base
  position/velocity is set directly each step; gravity and rigid-body
  dynamics play no role. This is a 2D point-reaching problem with a robot
  mesh on top, not a locomotion or manipulation task — nothing here should
  be read as evidence about either.
- **No trained result yet.** M0 is the plumbing milestone. There is no
  reward curve, no success rate, and no comparison to a baseline controller
  in this README because none has been produced and verified yet.
- **Single-episode, full-batch updates.** `PPOAgent.update()` trains on
  exactly one episode's rollout per update, with no minibatching. This is a
  known, documented departure from typical PPO practice (see
  `docs/design.md`), left for M1 to revisit once there are real training
  curves to judge it against.
- **No parallel environments.** Rollouts are collected from a single
  environment instance, serially.

## References

- [PPO paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
- [PyBullet docs](https://pybullet.org/)
- [Gymnasium](https://gymnasium.farama.org/)

## Licence

MIT — see [LICENSE](LICENSE).

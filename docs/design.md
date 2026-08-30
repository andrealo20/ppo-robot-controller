# Design notes

Rationale behind the non-obvious design and implementation decisions in this
repository, for anyone extending it or reviewing it in depth.

## Running as a module, not a script

`python src/train.py` does not work. Running a script directly puts the
*script's own directory* (`src/`) on `sys.path`, not the repository root, so
there is no `src` package for `from src.x import y` to resolve against.
Running as a module instead —

```sh
python -m src.train
python -m src.evaluate --model-path ...
```

— puts the current directory (the repository root, when invoked from there)
on `sys.path`, which is what makes the package imports work. `pytest` needs
the same fix by a different mechanism: `conftest.py` at the repository root
has no content, but its existence is enough — pytest adds a conftest.py's
directory to `sys.path` when it discovers it, which is what lets
`tests/*.py` do `from src.x import y` regardless of the directory pytest was
invoked from.

## Environment

### Reward, termination, and truncation

`RobotReachEnv.step()` reports `terminated=True` only when the robot
actually reaches the target (planar distance below `SUCCESS_DISTANCE`), and
`truncated=True` only when the step budget runs out without that happening.
Gymnasium's `(terminated, truncated)` distinction exists specifically so a
value function can be bootstrapped correctly: a truncated rollout should
bootstrap from the network's own value estimate of what happens next
(something *does* happen next, just outside the recorded rollout), while a
terminated rollout should bootstrap from exactly 0.0. `PPOAgent` tracks
`terminated` separately from the episode-end flag for exactly this reason —
see `compute_returns_and_advantages` below.

### Workspace bound

Nothing in the reward function itself stops the robot from drifting
arbitrarily far from the target. `velocity = action * 2.0` m/s, integrated
at PyBullet's default 240 Hz with one physics step per env step, means the
robot can move up to `2.0 / 240 ~= 0.0083` m per step — up to
`500 * 0.0083 ~= 4.17` m over a full episode if the action stays near an
extreme the whole time, roughly 3x farther than the target's own
`[-1, 1] x [-1, 1]` sampling range ever requires. Left unbounded, a
consistently bad policy's reward keeps growing worse the longer it persists,
rather than settling at some fixed (if bad) value. `RobotReachEnv.step()`
clamps the robot's planar position back inside `WORKSPACE_LIMIT = 1.5` m — 
50% headroom over the target's own range, enough that the robot is never
blocked from approaching a target near the edge of its sampling range, while
still capping how far a bad policy can be punished for drifting.
`tests/test_env.py::test_robot_cannot_drift_outside_the_workspace` pins this
down under sustained maximal action for a full episode, verified by
sabotage.

### Static target

The target URDF is loaded with `useFixedBase=True`, the same as the robot.
Without it, the target is an ordinary dynamic rigid body that PyBullet's
physics can move on contact — and because the robot's velocity is
force-set every physics step via `resetBaseVelocity` regardless of contact
forces, a moving robot behaves like an unstoppable wall against anything it
touches, capable of shoving a lightweight target out of position rather than
stopping at it. A fixed base removes that failure mode entirely: the target
is a landmark the reward function's distance calculation can treat as
static, matching what the reward and termination logic already assume. A
hand-coded oracle controller (`tests/test_oracle_policy.py`: always move at
full speed directly toward the target, no learning involved) is a cheap way
to check this and the rest of the environment/reward/termination contract
independent of PPO — with the target sampled from `[-1, 1] x [-1, 1]`, the
robot starting at the origin, a top speed of 2.0 m/s, and 500 steps
available, a correct-direction oracle should resolve every seed in well
under the step budget (worst case ~1.41 m away needs roughly 170 steps at
top speed), and it does.

### Observation normalization

`RobotReachEnv`'s six observation components mix raw position and
displacement values on no common scale. `src/utils/running_normalizer.py`
implements a batched Welford/Chan running mean-and-variance tracker plus a
`gym.ObservationWrapper` that normalizes and clips observations against it.
Statistics update online during training and are frozen during evaluation
(`update_stats=False`), the standard approach (e.g. Stable-Baselines3's
`VecNormalize`) given the visited state distribution depends on the policy
being trained. Checkpoints save the normalizer's running statistics
alongside the network and optimizer state (see "Checkpoints" below) so
evaluation restores the exact transformation the policy was trained under,
rather than a fresh, differently-scaled one.

## Agent

### Bounded policy: tanh transform, not post-hoc clipping

The actor parameterizes an unconstrained diagonal Gaussian; sampled latent
actions are mapped through `tanh` and then affinely scaled into the
environment's finite action bounds. PPO evaluates the transformed density,
including the change-of-variables Jacobian
(`log|dy/dx| = log(scale) + log(1 - tanh(x)^2)`). `select_action()`
deliberately re-scores the exact float32 bounded action returned to the
caller, so the rollout's stored action and stored log-probability refer to
the same random variable — this matters because PPO's importance ratio
`exp(log_pi_new(a) - log_pi_old(a))` is only meaningful when both sides
score the same value, and it must equal exactly 1.0 for every stored
transition before any optimizer step (the same distribution, evaluated at
the same point, with unchanged weights).
`tests/test_ppo_ratio_identity.py` checks this directly for both low-noise
and near-saturated policies, and includes a sabotage case that perturbs one
stored log-probability and verifies the invariant is detected as broken.

### GAE across multi-episode rollouts

`PPOAgent` collects a configurable `--rollout-steps` (default 2048) worth of
transitions per update, spanning as many episodes as it takes to fill that
budget, then updates over shuffled minibatches for several epochs. Because a
single buffer can hold several episodes back to back,
`compute_returns_and_advantages`'s GAE recursion is cut at *every* episode
boundary inside the buffer (`episode_end[t]`), not only at the buffer's last
transition — otherwise one episode's advantage would leak backward into the
previous episode's through the `gae` accumulator.
`tests/test_gae.py::test_second_episode_does_not_leak_into_first` pins this
down with a 4-transition, 2-episode buffer, verified by sabotage. The
`next_value` argument passed to `compute_returns_and_advantages`/`update` is
used only when the buffer's last transition does *not* itself end an
episode (the rollout was cut off purely because the step budget ran out,
mid-episode); when the last transition is itself a boundary, the bootstrap
comes from that transition's own `boundary_value` instead, and `next_value`
is ignored — `tests/test_gae.py::test_next_value_is_ignored_at_an_episode_
boundary` pins that contract down explicitly.

`src/train.py`'s `collect_rollout` threads a running `episode_reward` total
through its return value and back in as an argument on the next call (the
same way `state` already has to be), because a rollout boundary very often
lands mid-episode: without carrying the running total across calls, the
reward accumulated so far for an in-progress episode would be dropped
whenever that episode straddles two rollout collections.

### Numerical edge cases

- **`log_std` is clamped** to `[-5, 1]` in log-space (`std` in roughly
  `[6.7e-3, 2.7]`), bracketing anything a policy over actions mapped through
  `tanh` should need, and preventing an unlucky gradient step from sending
  it to a degenerate extreme.
- **Advantage normalization uses `unbiased=False`** for `advantages.std()`.
  The Bessel-corrected default is `0/0 = NaN` for a single-transition
  buffer — a legal if unusual case, given any episode that terminates or
  truncates on its very first step. The unbiased-false value for a single
  element is exactly `0`, leaving that advantage inert after normalization
  instead of NaN-poisoning the whole update.
- **Non-finite loss or gradient norm raises immediately**
  (`FloatingPointError`) rather than silently poisoning later updates.

### Initialization

Hidden layers use orthogonal initialization with gain `sqrt(2)`; the actor's
output layer uses gain `0.01` (small initial action variance around zero);
the critic's output layer uses gain `1.0`. This is the standard head-specific
PPO initialization scheme rather than a single gain applied uniformly across
every layer.

### Checkpoints

A checkpoint saves network weights, optimizer state, configuration, training
progress counters, and the observation-normalizer's running statistics
together (`checkpoint_version`-tagged, so the format can evolve). Evaluation
restores and freezes those normalizer statistics before the first reset, so
it evaluates the policy under the exact input transformation it was trained
under rather than a fresh mean=0/variance=1 one. A bare network-only
`state_dict` file (no normalizer/optimizer/config alongside it) still loads,
but `src.evaluate` rejects it for evaluation by default — pass
`--allow-legacy-checkpoint` to opt into the known-invalid fallback
explicitly. Evaluation uses the squashed policy mean (deterministic) by
default; `--stochastic` opts into sampling instead.

`best_model.pt` is saved as soon as a new best is found, *before* the
rollout's PPO update runs — the network weights that produced a given
episode's reward have to be the ones written to disk, not the weights one
gradient step later. "Best" itself is a rolling mean over the most recent
`--best-model-window` (default 20) completed episodes rather than any
single episode's raw reward: with a different random target every episode,
one easy (nearby) target can otherwise look like the best policy purely by
chance. `model_updateN.pt` periodic snapshots are unambiguous either way —
they are simply "the network after N updates" — so they save post-update as
before.

### Training diagnostics

Each PPO update returns and TensorBoard-logs policy loss, value loss, base
Gaussian entropy, approximate KL, clip fraction, gradient norm, and mean
policy standard deviation (`ppo/*` scalars), alongside per-episode reward
(`reward/episode`).

## Testing strategy

Results are checked against hand-derived references and sabotage, not by
reading the implementation and agreeing with it. `tests/test_gae.py`
verifies `compute_returns_and_advantages` against a GAE reference written
independently of the agent's own recursion; `tests/test_ppo_ratio_identity.py`
verifies the pre-update PPO ratio is exactly 1.0 under both low-noise and
near-saturated policies; `tests/test_oracle_policy.py` verifies the
environment itself is solvable independent of PPO, via a hand-coded
always-move-toward-target controller; `tests/test_env.py` verifies the
terminated/truncated contract and the workspace bound. Several of these are
explicitly paired with a sabotage check: deliberately reintroducing the
failure mode a test is meant to catch (e.g. removing the workspace clamp, or
letting the GAE recursion continue across an episode boundary) fails exactly
that test and nothing else in the suite.

## CI

`.github/workflows/ci.yml` runs on `ubuntu-latest` with Python 3.10 and
3.11. PyPI ships prebuilt `pybullet` wheels only for `manylinux` and for
Python up to 3.11 — no macOS wheel exists for any Python version, and no
wheel exists for Python 3.12 on any platform — so narrowing the matrix to
this combination means CI installs a prebuilt binary rather than compiling
the full Bullet physics engine from source on every run.

## Figures

`assets/reaching_env.png` is a real render of a real `env.reset()` call
(seed 7) — camera position, robot pose, and target position all come
straight from PyBullet state. The one liberty taken: the target is drawn as
a separate, larger red sphere overlaid at the true target's position,
because the actual target body (`sphere_small.urdf`, `globalScaling=0.2`)
has an AABB about 1.2 cm across and is invisible at any camera distance that
also frames the robot.

`assets/training_reward_random_target.png` and
`assets/training_reward_fixed_target.png` are real per-episode and rolling-
mean reward data pulled from TensorBoard event logs of real training runs,
not mockups.

## Hyperparameters

The headline result (see the README) uses:

```sh
python -m src.train --num-episodes 3000 --lr 1e-4 --rollout-steps 2048 --minibatch-size 64
```

`gamma=0.99`, `gae_lambda=0.95`, `eps_clip=0.2`, `value_coef=0.5`,
`entropy_coef=0.01`, `max_grad_norm=0.5`, `ppo_epochs=4` are the
`PPOAgent`/`train.py` defaults and were not tuned beyond `lr`, which is
lower than the common PPO literature default of `3e-4` — this task's action
scale and reward magnitude made the lower rate the more stable choice.

## Reproducibility

`--seed` seeds NumPy (which also governs `PPOAgent.update`'s minibatch
shuffling, since it calls `np.random.permutation` on the module-level RNG),
PyTorch (network initialization and Gaussian action sampling both draw from
its default generator), and the environment's own RNG via the first
`env.reset(seed=...)` call — Gymnasium's `Env.reset` stores that seeded
generator on the instance, so every later unseeded `reset()` inside
`collect_rollout` keeps drawing from the same, already-seeded sequence.
Passing the same `--seed` reproduces a run's episode rewards exactly.

Run-to-run variance on the headline result was checked directly rather than
assumed, with a systematic sweep over `--seed 0` through `--seed 4`, 3000
episodes each, otherwise identical hyperparameters, each evaluated on the
same held-out protocol (deterministic policy, frozen normalizer statistics,
50 episodes, seed 1000):

| seed | held-out success | mean reward |
|---|---|---|
| 0 | 46/50 (92%) | -30.01 +/- 37.45 |
| 1 | 47/50 (94%) | -27.73 +/- 34.67 |
| 2 | 47/50 (94%) | -28.40 +/- 37.17 |
| 3 | 46/50 (92%) | -34.08 +/- 52.14 |
| 4 | 50/50 (100%) | -22.08 +/- 19.63 |

**Mean 94.4% +/- 3.3% held-out success** (population std 2.9%; pooled
236/250 episodes across all 5 seeds). All five training curves share the
same shape: an early-block (episodes 1-300) mean between -137 and -267,
settling to roughly -20 to -30 by the back half of the run, with a positive
linear trend (`r` in the 0.64-0.65 range across ten 300-episode chunks).
Raw per-seed results are in `results/seed_sweep_summary.csv`, produced by
`run_seed_sweep.sh` at the repository root, which trains and evaluates all
5 seeds back to back with fresh output directories per seed.

An earlier, smaller check (two runs, unseeded and `--seed 2026`, 84% and
86% held-out respectively) pointed the same direction before the full sweep
was run; the 5-seed sweep above supersedes it as the reported result.

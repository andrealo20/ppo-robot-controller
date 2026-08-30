# Design notes

## Where this started

The repository existed locally with a scaffolded README, a plausible
directory layout, and PPO/environment/network code that read as reasonable
PyTorch -- but `tests/`, `experiments/` and `src/utils/` were empty, and the
git history carried a branch (`andrealo20-redesigned-pancake`) and
`refs/copilot/checkpoints/` entries characteristic of GitHub Copilot's
autonomous coding agent rather than manual commits. Nothing here was run
before this pass: the very first import in the documented entry point fails
(see below), which is not something a single successful run could have left
in place.

That git history has been discarded. What follows is what was found and
fixed, kept as the honest record this repository is supposed to carry going
forward -- not because a prior AI-authored pass is something to hide (the
rest of this portfolio is built the same way, working turn by turn with an
assistant), but because *this* pass was never verified against anything, and
the discarded history was closer to a liability than a record worth keeping.

## Bug 1: the agent could not be constructed

`src/agent/ppo.py` contained:

```python
from .network import PolicyValueNetwork
```

A relative import from `src/agent/` resolves `.network` to `src.agent.network`,
which does not exist -- the network lives in `src/network/policy_value.py`, a
sibling package of `agent`, not a child of it. `PPOAgent.__init__` raised
`ModuleNotFoundError` on the first call, before any training code ran.

Fixed by importing the concrete module directly:
`from src.network.policy_value import PolicyValueNetwork`, matching the
absolute-import style `train.py` already used for the environment.

## Bug 2: the documented entry point could not run either

Even past Bug 1, the README's Quick Start (`python src/train.py ...`) fails on
its own first line, `from src.environment.reaching_env import RobotReachEnv`:
running a script directly puts the *script's directory* (`src/`) on
`sys.path`, not the repository root, so there is no `src` package visible to
import from. This is independent of Bug 1 and would have surfaced immediately
on the first real run.

Fixed by documenting and using module invocation instead, which puts the
current directory (the repository root, when run from there) on `sys.path`:

```sh
python -m src.train
python -m src.evaluate --model-path ...
```

For `pytest`, the same problem needed a different fix: a `conftest.py` at the
repository root, whose only job is to exist there -- pytest adds a
conftest.py's directory to `sys.path` when it imports it, which is what lets
`tests/*.py` do `from src.x import y` regardless of the directory pytest was
invoked from.

## Bug 3: truncated episodes were bootstrapped as if they had terminated

This is the one worth describing in full, because it wouldn't have shown up
as a crash -- it would have shown up as a policy that quietly trained worse
than it should have, with nothing pointing at why.

`RobotReachEnv.step()` originally set `terminated = step_count >= 500`: every
episode ends by running out of the step budget (reaching the target only adds
a reward bonus, it never used to end the episode), so *every* episode was
reported as `terminated`, never `truncated`. Separately, `train.py`'s loop
was:

```python
while not done:
    ...
    done = terminated or truncated
    ...
next_value = agent.get_value(next_state) if not done else 0.0
```

`done` is always `True` the instant the loop exits, by construction of the
loop condition -- so `next_value` was *always* `0.0`, regardless of why the
episode ended. And in `PPOAgent.compute_returns_and_advantages`, the buffer
stored one `done` flag per step (rather than distinguishing termination from
truncation), used as `next_non_terminal = 1.0 - done`, which zeroed out the
GAE bootstrap term a second time at exactly the same point.

Both effects push the same direction: at the last step of every episode --
which in this environment is *every* episode, since none of them were true
terminations -- the value target was forced toward what a true, resolved
ending would look like, even when the agent was cut off mid-task by the
clock. Gymnasium's `(terminated, truncated)` pair exists specifically to
prevent this: a truncated rollout should bootstrap from the network's own
estimate of what happens next, because something *does* happen next, just
outside the recorded rollout.

Fixed in three places that all had to agree:

1. `RobotReachEnv.step()` now sets `terminated` only on actually reaching the
   target (`distance < SUCCESS_DISTANCE`), and `truncated` only when the step
   budget runs out without that happening.
2. `PPOAgent.store_transition` and the rollout buffer now track `terminated`
   specifically (not the OR of terminated/truncated) -- true only on the last
   stored transition, and only if the episode actually resolved.
3. `train.py` computes the bootstrap as `0.0 if terminated else
   agent.get_value(next_state)`, using the real `terminated` flag from the
   final `env.step()` call rather than the loop's exit condition.

`tests/test_gae.py` pins this down against a hand-derived reference (written
independently of `compute_returns_and_advantages`, not copied from it) for
both a terminated and a truncated ending from the same starting rollout, and
includes a test that asserts the two must produce different bootstrapped
returns. Reverting the fix (verified by hand: forcing
`next_non_terminal = 0.0` unconditionally at the last step, mirroring the
original bug) fails exactly those two tests and nothing else -- the rest of
the suite doesn't touch this code path.

## Bug 4: the observation was 7-dimensional against a declared shape of 6

Found by `tests/test_env.py::test_reset_returns_correctly_shaped_observation`,
not by reading the code:

```python
obs = np.array([
    *robot_pos[:2],
    *target_pos[:2],
    *np.array(target_pos) - np.array(robot_pos)   # full 3D vector, not [:2]
], dtype=np.float32)
```

`observation_space` declared `shape=(6,)`, but the displacement term used the
full `(x, y, z)` difference instead of the planar one the rest of the
observation uses, making the actual array length 7. Gymnasium's `Box` space
does not itself validate what `step()`/`reset()` return, so nothing caught
this at runtime -- it would only ever have surfaced as a shape-mismatch
exception the first time a fixed-size network layer saw it, most likely
reported far from this function.

Fixed by computing the planar displacement explicitly: `target_xy - robot_xy`.

## Robustness fixes (not bugs the code had exercised, but real failure modes)

- **Unclamped `log_std`.** `PolicyValueNetwork.log_std` was a free parameter
  with no bound. A bad gradient step can send it arbitrarily large (near-pure
  noise) or arbitrarily negative (a numerically deterministic policy, with
  `Normal.log_prob` gradients that blow up as std -> 0). Clamped to
  `[-20, 2]` in log-space (`std` in roughly `[2e-9, 7.4]`), which brackets
  anything a policy over actions in `[-1, 1]` should need. Verified in
  `tests/test_network.py` by pushing `log_std` far outside the range and
  confirming the forward pass still clamps.
- **No observation normalization.** `RobotReachEnv`'s six observation
  components mix raw position and displacement values on no common scale.
  Added `src/utils/running_normalizer.py`: a batched Welford/Chan running
  mean-and-variance tracker plus a `gym.ObservationWrapper` that normalizes
  and clips observations against it. Statistics update online during
  training (frozen during evaluation via `update_stats=False`), the standard
  approach (e.g. Stable-Baselines3's `VecNormalize`) given the visited state
  distribution depends on the policy being trained.
- **Advantage-normalization NaN on a one-transition rollout.** `advantages.std()`
  defaults to Bessel-corrected (`unbiased=True`), which is `0/0 = NaN` for a
  single-element tensor -- a legal if unusual buffer size, given here by any
  episode that terminates or truncates on its very first step. Switched to
  `unbiased=False`, whose single-element value is exactly `0`, leaving that
  advantage inert after normalization instead of NaN-poisoning the whole
  update.

## Lint

`ruff check`/`ruff format` (the same tool `stackbound`'s CI uses) found the
import-order and unused-import issues expected from files that had never been
linted, plus two real findings worth naming: an unpacked-but-unused `value`
in `select_action` (harmless, but flagged and renamed `_value`), and
`RobotReachEnv.metadata` as a mutable class-level dict without a `ClassVar`
annotation (also harmless here since nothing mutates it post-definition, but
the annotation makes that an invariant instead of an accident). Both jobs are
now in `.github/workflows/ci.yml`.

## What this pass deliberately did not do

- **Rollout batching.** `PPOAgent.update()` still updates once per episode on
  a single full batch, with no minibatching and no multi-episode rollout
  collection before an update. That's a real departure from typical PPO
  practice (larger, shuffled batches from parallel or concatenated episodes)
  and will cost sample efficiency once training is actually attempted -- but
  it is a design choice to revisit with real training curves in hand, not a
  bug to silently redesign underneath untested numbers. Documented here as a
  known limitation, not fixed.
- **`RobotReachEnv`'s kinematic (not dynamic) physics.** The robot is a
  fixed-base `r2d2.urdf` moved by directly resetting its base
  position/velocity rather than applying joint torques or forces, so gravity
  and rigid-body dynamics play no role in the task -- it's a 2D
  point-reaching problem with a robot mesh on top, not a locomotion or
  manipulation task. Left as-is and stated plainly in the README rather than
  quietly implied to be more than it is.

## The README figure

`assets/reaching_env.png` is a real render of a real `env.reset()` call
(seed 7), not a mockup — camera position, robot pose and target position all
come straight from PyBullet state. The one liberty taken: the target is
drawn as a separate, larger red sphere overlaid at the true target's
position, because the actual target body (`sphere_small.urdf`,
`globalScaling=0.2`) has an AABB about 1.2 cm across and is invisible at any
camera distance that also frames the robot. No training curve or reward plot
is included yet, deliberately -- there is no trained policy behind one, and
adding a plausible-looking one before M1 would misrepresent the M0 status
above it.

## CI: macOS and Python 3.12 dropped from the test matrix

The first pushed version of `.github/workflows/ci.yml` matrixed
`{ubuntu-latest, macos-latest} x {3.10, 3.12}`, copied from `stackbound`'s CI
without checking whether it fit this repository's dependencies. It didn't:
both `macos-latest` jobs failed with `ERROR: Failed building wheel for
pybullet`. Checking PyPI directly (`pybullet`'s latest release is `3.2.7`)
shows why -- its published wheels are manylinux-only:

```
pybullet-3.2.7-cp310-cp310-manylinux_2_17_x86_64...
pybullet-3.2.7-cp311-cp311-manylinux_2_17_x86_64...
pybullet-3.2.7-cp36 .. cp39, pp37 .. pp39  (same manylinux_x86_64 pattern)
pybullet-3.2.7.tar.gz
```

No macOS wheel exists at all, for either architecture, and no wheel exists
for Python 3.12 on *any* platform -- `pip` falls back to compiling PyBullet
(the full Bullet physics engine, in C/C++) from `pybullet-3.2.7.tar.gz`.
That build happened to succeed on `ubuntu-latest` (a full C/C++ toolchain is
present by default) and failed on `macos-latest`.

Rather than debug a from-source Bullet build against a moving macOS Xcode
toolchain for a portfolio project's CI, the matrix was narrowed to
`ubuntu-latest x {3.10, 3.11}` -- both versions PyPI ships a real wheel for,
so CI installs a prebuilt binary instead of compiling anything. This is a
real, disclosed narrowing of test coverage relative to `stackbound`'s
(pure-Python) CI, not an attempt to hide the macOS failure by deleting the
job silently.

## Status

M0: the plumbing is correct and covered by 21 tests (`python -m pytest tests
-q`), `python -m src.train` and `python -m src.evaluate` both run end to end
against the real PyBullet environment. No policy has been trained to
convergence and no headline result exists yet -- that's the next milestone,
not this one.

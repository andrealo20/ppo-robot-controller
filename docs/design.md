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

## M1: multi-episode rollouts

M0 shipped with a documented, deliberate limitation: `PPOAgent.update()`
trained on exactly one episode's rollout per update, no minibatching, no
multi-episode collection. The M0 README said this would be revisited "if
sample efficiency turns out to need it" -- once M1 started, it did: 900
real episodes of M0-era training (single episode/full-batch, lr=1e-4) showed
no improving trend at all (chunked means across the run: -471, -445, -465,
-423, -340, -410, -349, -459, -383 -- noise, not a curve).

The fix: `PPOAgent` now collects a configurable `--rollout-steps` (default
2048) worth of transitions per update, spanning as many episodes as it takes
to fill that budget, then updates over shuffled minibatches for several
epochs. This meant `compute_returns_and_advantages` had to change from
"the buffer is exactly one episode" to "the buffer may hold several episodes
back to back" -- the GAE recursion now cuts at *every* episode boundary
inside the buffer (`episode_end[t]`), not only at the buffer's last
transition, or one episode's advantage would leak backward into the
previous episode's through the `gae` accumulator.
`tests/test_gae.py::test_second_episode_does_not_leak_into_first` pins this
down with a 4-transition, 2-episode buffer and was verified by sabotage:
forcing the recursion to never cut (`gae_continues = 1.0` unconditionally)
fails exactly that one test and nothing else.

A second, genuine bug surfaced the same way during development, in
`src/train.py`'s `collect_rollout`: the running total for whatever episode
is still in progress when a rollout ends mid-episode was being reset to
`0.0` on every call instead of carried over to the next one, so an episode
that started in one rollout and finished in the next would be reported with
only the second half of its true reward.
`tests/test_collect_rollout.py::test_ongoing_episode_state_carries_over_
between_calls` caught this immediately (expected `5.0`, got `2.0`) the first
time it was run against the real implementation -- not found by inspection,
found by the test failing on real numbers. Fixed by threading
`episode_reward` through `collect_rollout`'s return value and back in as an
argument on the next call, the same way `state` already had to be.

## M1: a reward plateau, a runaway hypothesis, and a disproved fix

With multi-episode rollouts and minibatching in place, four real training
runs were made (all logged in full via TensorBoard, nothing here is
extrapolated or estimated):

1. **lr=3e-4** (the common PPO literature default), entropy_coef=0.01,
   LOG_STD_MAX=2.0: unstable. Reward repeatedly collapsed to between -1000
   and -1900 for stretches of hundreds of episodes before partially
   recovering, then collapsing again -- worse than M0, not better.
2. **lr=1e-4** (M0's original, more conservative value), entropy_coef=0.01,
   LOG_STD_MAX=2.0, 3000 episodes: stable -- no catastrophic collapse this
   time -- but flat. Chunked means across all 3000 episodes: -403, -411,
   -419, -413, -419, -414, -401, -418, -406, -421. No improving trend
   anywhere in the run. Only 17 of 3002 episodes (0.6%) had positive reward
   (i.e. reached the target) during training, and that rate did not increase
   over time. `assets/training_reward_m1.png` is this run's real per-episode
   reward and 50-episode rolling mean -- flat at roughly -400 for the entire
   3000 episodes.
3. A post-hoc evaluation of that run's `best_model.pt` (frozen observation
   normalizer, 50 episodes, seed 1000, `python -m src.evaluate`) confirms
   the training curve rather than being an artifact of exploration noise:
   **1/50 episodes resolved (2%), mean reward -420.14 +/- 142.31.**

Loading that checkpoint and inspecting `log_std` directly showed it pinned
almost exactly at `LOG_STD_MAX` (2.0033, 2.0008 -- i.e. std ~= 7.4), far
above the [-1, 1] action range. The hypothesis: with std that large, sampled
actions clip to +-1 almost regardless of the network's mean output, so the
policy's actual behaviour is close to a coin flip between the two extremes
no matter what it has "learned" -- which would explain both why nothing the
value/policy network does moves the average reward, and why std would keep
growing (the entropy bonus rewards it, and once actions are dominated by
clipping there is little policy-gradient signal pushing back).

Two follow-up runs were made specifically to test that hypothesis, not to
guess at a fix and hope:

4. **entropy_coef=0.0** (remove the pressure toward higher std entirely),
   same lr=1e-4: worse, not better -- collapsed into the same -1000/-1900
   territory as the lr=3e-4 run, starting within a few hundred episodes.
5. **LOG_STD_MAX=0.0** (std capped at 1.0, matching the action range
   directly instead of relying on the entropy bonus to keep it sane),
   entropy_coef back at 0.01, lr=1e-4: also worse -- catastrophic collapse
   from episode ~100 onward, worst observed reward -1852.

Both changes, made independently to test the same "std is too large"
hypothesis, made training measurably *less* stable. The hypothesis was
wrong, or at least incomplete, and both changes were reverted rather than
shipped on the strength of a plausible story that real runs disproved --
`LOG_STD_MAX` is back at its original 2.0, `entropy_coef` back at its
original 0.01, matching run 2 above, the one stable configuration actually
found.

**What is actually going on, most likely:** the environment has no bound on
the robot's workspace, and the action-to-target-range ratio makes runaway
trajectories cheap to fall into and expensive to be in. `velocity = action *
2.0` m/s, integrated at PyBullet's default 240 Hz with one physics step per
env step, means the robot can move up to `2.0 / 240 ~= 0.0083` m per step --
up to `500 * 0.0083 ~= 4.17` m over a full episode if the action stays near
an extreme the whole time. The target is sampled from `[-1, 1] x [-1, 1]`,
so the robot can end up roughly 3x farther from the target than the target
region's own diagonal ever requires, with nothing in `RobotReachEnv`
penalizing or bounding that. A policy that is even moderately consistent in
one bad direction for a stretch of an episode racks up a strongly negative
reward for every one of the (up to 500) remaining steps: summing a linearly
growing distance term over 500 steps lands in the -1000 to -2000 range,
matching the collapsed runs' numbers almost exactly. M1's multi-episode
minibatched update is *more* effective at making the policy consistent
(that was the point of building it) -- including, transiently, consistently
bad -- which plausibly makes this failure mode easier to fall into than it
was under M0's weak, single-episode updates, not harder.

This was not fixed in this pass. Doing so blind (capping the action
multiplier, bounding the workspace, reshaping the reward to penalize large
distances more than linearly, shortening `MAX_STEPS` so a bad episode can't
compound as long) would be exactly the kind of guessed tolerance this
repository's own conventions rule out -- any of those needs to be tried and
measured against a real curve the same way the two disproved fixes above
were, not shipped on a plausible story alone.

## M1.1: bounding the workspace -- a real fix for the wrong hypothesis

The structural explanation above made a specific, testable prediction: if
unbounded drift is what drives the training plateau, then bounding the
robot's workspace should make training curves improve, not just stop
collapsing. That prediction was tested, not assumed.

`RobotReachEnv.step()` now clamps the robot's planar position back inside
`WORKSPACE_LIMIT = 1.5` m (50% headroom over the target's own `[-1, 1] x
[-1, 1]` sampling range) whenever a step would push it past that bound --
`tests/test_env.py::test_robot_cannot_drift_outside_the_workspace` pins this
down under sustained maximal action for a full episode, verified by
sabotage (removing the clamp fails exactly that test, at exactly the
predicted magnitude: the unclamped run reached 1.5083 m, just past the 1.5 m
bound).

A full 3000-episode run with this fix, otherwise identical to M1's stable
run (lr=1e-4, entropy_coef=0.01, `LOG_STD_MAX=2.0`), was compared directly
against it -- `assets/training_reward_m1_1_comparison.png` overlays both
50-episode rolling means. Result: **the two curves track each other almost
exactly for the first ~2700 episodes** (both flat around -400, exactly the
plateau this was meant to fix), and the bounded run's tail is *worse*, not
better -- a late-training regression down to a rolling mean of ~-900 in the
last few hundred episodes, driven by rewards as low as -1518. A held-out
evaluation (frozen normalizer, 50 episodes, seed 1000) of this run's
`best_model.pt`: 3/50 resolved (6%, versus M1's 2%) but mean reward -918.08
+/- 361.36 (versus M1's -420.14 +/- 142.31) -- a slightly higher success
rate bought with much worse average performance and much higher variance,
not a clean improvement.

**The hypothesis is disproved as an explanation for the plateau.** The
workspace bound is being kept anyway -- it is still a real, independently
justified fix (it removes the specific *unbounded, ever-growing* runaway
failure mode that runs 1 and 4 of the previous section fell into, and the
sabotage test proves it works as designed) -- but it does not make the
agent learn to reach the target, so it is not M1's actual blocker.

**The most likely honest explanation left, and the one this pass did not
rule out:** RL sample complexity. 3000 episodes is roughly 1.5M environment
steps -- a modest budget by the standards of published from-scratch PPO
results on continuous-control tasks, many of which use tens of millions of
steps. Nothing tested here (bootstrap correctness, minibatching, the
workspace bound) addresses that; it may simply be that this task, with this
network size and this action scale, needs an order of magnitude more
training than was run in this pass to show a clear improving trend at all.
That is a testable claim -- run substantially longer and check -- not a
guess to ship as if it were the answer, so it is left here as the next
thing to actually test, not as a conclusion.

## M1.2: the training-budget hypothesis, tested at 3x the budget -- disproved too

The previous section's hypothesis made a specific, testable prediction: if
this task simply needs more environment steps than the ~1.5M run so far,
training substantially longer should show a clear improving trend even if it
hasn't converged yet. That prediction was tested, not assumed -- a full
9000-episode run (~4.37M environment steps, 3x the M1/M1.1 budget), otherwise
identical to M1.1 (lr=1e-4, entropy_coef=0.01, `LOG_STD_MAX=2.0`, the
workspace bound in place, no other hyperparameter changed).

Result: **no improving trend across the full run.** Splitting the 9000
episodes into thirds:

| | episodes | mean reward | success rate |
|---|---|---|---|
| 1st third | 1-3000 | -855.02 | 0.57% (17/3000) |
| 2nd third | 3001-6000 | -894.21 | 0.37% (11/3000) |
| 3rd third | 6001-9000 | -898.21 | 0.37% (11/3000) |

A linear fit over eighteen 500-episode chunks gives a *negative* slope for
mean reward (r=-0.638) and an essentially flat, weakly negative slope for
success rate (r=-0.259) -- the opposite of the improving trend the
hypothesis predicted. `assets/training_reward_m1_2.png` shows this directly:
the 50-episode rolling mean drops from the run's noisy start to roughly -900
within the first ~1500 episodes and then stays there, flat, for the
remaining 7500.

A held-out evaluation (frozen normalizer, 50 episodes, seed 1000) of this
run's `best_model.pt`: 4/50 resolved (8%), mean reward -802.41 +/- 377.35.
Read on its own this looks like the best held-out number of the three runs
(M1: 2%, M1.1: 6%, M1.2: 8%) -- but with n=50 per evaluation and no seed
control on the training runs themselves, the standard error on a ~5% true
rate at n=50 is already about 3 percentage points, so 2/50 vs 4/50 is well
within run-to-run noise. It does not contradict the training-curve trend
analysis above, which is the more reliable signal here (9000 data points per
run, not 50) and shows no improvement. The honest reading is: the held-out
number moved around inside noise; the training curve, which is what the
hypothesis actually predicted would change, did not improve.

**The hypothesis is disproved at this scale.** 3x the environment steps,
with everything else held fixed, produced a flat curve, not a rising one --
not even the beginning of a rising one. This does **not** rule out that a
much larger budget (the tens-of-millions-of-steps scale common in published
from-scratch PPO continuous-control results, roughly another order of
magnitude beyond what M1.2 ran) would eventually show a trend; that remains
untested and would need to be run and measured the same way, not assumed
from this result either. What M1.2 does rule out is the more modest,
practically-runnable version of the hypothesis -- that a few more thousand
episodes was the missing ingredient.

Three structural hypotheses for the plateau have now been tested against
real curves and disproved: log_std/entropy saturation (M1), the unbounded
workspace (M1.1), and a 3x training budget (M1.2). None of the fixes tried
made things better; two of them (entropy tuning, LOG_STD_MAX) made things
measurably worse and were reverted. What's left unruled-out is either a much
larger budget than was practical to test here, or a limitation in the
task/network/reward setup itself that none of these three hypotheses
targeted -- reward shaping, network capacity, or the action scale relative
to the workspace are the more likely candidates for a future pass.

## M1 status: honest, not converged

- The M1/M1.1 *infrastructure* is correct and tested: multi-episode
  rollouts, minibatched updates, a hard workspace bound, 29 tests (8 more
  than M0's 21), training runs that complete end to end with no crashes, no
  NaNs, and no unbounded divergence.
- The M1 *agent* does not work: 2-8% success rate across three held-out
  evaluations, no improving trend across four separate training runs from
  900 to 9000 episodes -- including one run 3x longer than the others,
  launched specifically to test whether more training was the missing
  ingredient -- and three independently tested hypotheses (log_std/entropy, the workspace
  bound, then the training budget) that were each disproved by real curves
  rather than shipped anyway.
- What is left unruled-out is a much larger budget than was practical to
  test here (roughly another order of magnitude beyond M1.2's 4.37M steps),
  or a limitation this pass's three hypotheses didn't target at all --
  reward shaping, network capacity, or the action scale relative to the
  workspace.

This is the "including if it plateaus somewhere unflattering" case the M0
README explicitly said to report honestly rather than dress up. M1 is not
complete.

## Status

M0: the plumbing was correct and covered by 21 tests, `python -m src.train`
and `python -m src.evaluate` ran end to end against the real PyBullet
environment. No policy had been trained to convergence and no headline
result existed yet.

M1/M1.1/M1.2 (current): multi-episode rollouts, minibatched updates, and a
hard workspace bound are implemented and tested (29 tests total, `python -m
pytest tests -q`), training runs stably end to end, and real training curves
plus real held-out evaluations exist across four separate runs (900-9000
episodes) -- 2-8% success rate, no improving trend in any of them. The agent
does not yet solve the task. Three structural hypotheses (log_std/entropy
tuning, the workspace bound, then a 3x training budget) were tested against
real curves and disproved rather than shipped. What remains untested is a
much larger budget (tens of millions of steps) and non-budget explanations
(reward shaping, network capacity, action scale) -- see "M1.2: the
training-budget hypothesis..." above for the full reasoning.

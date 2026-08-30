"""Custom PyBullet environments for robotic control tasks."""

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

MAX_STEPS = 500
SUCCESS_DISTANCE = 0.1
SUCCESS_BONUS = 10.0


class RobotReachEnv(gym.Env):
    """2D reaching task: reward is the negative distance to a random target.

    The "robot" is a fixed-base r2d2.urdf loaded purely as a visual stand-in.
    Its planar position is driven by directly setting base velocity/position
    rather than by joint torques or forces, so gravity and rigid-body dynamics
    play no role in the task -- this is a kinematic point-reaching problem,
    not a dynamic locomotion or manipulation one. See the repository README's
    Limitations section for what that does and doesn't mean for the results.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        """Initialize reaching environment.

        Args:
            render_mode: "human" for visualization or None
        """
        self.render_mode = render_mode
        self.client = None
        self.robot_id = None
        self.target_id = None
        self.step_count = 0

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self._setup_physics()

    def _setup_physics(self):
        """Initialize PyBullet physics engine."""
        if self.client is None:
            mode = p.GUI if self.render_mode == "human" else p.DIRECT
            self.client = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # Load environment
            self.plane_id = p.loadURDF("plane.urdf")
            self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1], useFixedBase=True)
            self.target_id = p.loadURDF(
                "sphere_small.urdf", [1, 1, 1], globalScaling=0.2
            )

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        if self.client is None:
            self._setup_physics()

        # Random target position
        target_pos = self.np_random.uniform(-1, 1, 3)
        target_pos[2] = 1.0
        p.resetBasePositionAndOrientation(self.target_id, target_pos, [0, 0, 0, 1])

        # Robot reset
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 1], [0, 0, 0, 1])

        self.step_count = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """Execute one step."""
        # Apply action (simple velocity control)
        velocity = np.asarray(action) * 2.0
        p.resetBaseVelocity(self.robot_id, velocity.tolist() + [0])

        # Simulation step
        p.stepSimulation()
        self.step_count += 1

        # Get observation, reward and the single planar distance both of them
        # depend on -- computed once so reward and termination can't disagree.
        obs = self._get_observation()
        distance = self._distance_to_target()
        reward = self._compute_reward(distance)

        # `terminated` means the episode ended because the task resolved (the
        # target was reached); `truncated` means it ended only because the
        # step budget ran out. The distinction matters downstream: a PPO agent
        # should bootstrap the value of a truncated episode from the network,
        # but not from a terminated one -- see PPOAgent.update() and
        # docs/design.md for what conflating the two used to cause here.
        terminated = bool(distance < SUCCESS_DISTANCE)
        truncated = bool(self.step_count >= MAX_STEPS and not terminated)

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Get current observation: robot (x, y), target (x, y), planar
        displacement (x, y) -- six components, matching observation_space.

        The displacement used to be the full 3D vector (target_pos - robot_pos
        with no slicing), which made this method return 7 values against a
        6-element observation_space with nothing ever catching the mismatch
        at runtime -- Gymnasium's Box space does not itself validate the
        shape of what step()/reset() return. Caught by
        tests/test_env.py::test_reset_returns_correctly_shaped_observation.
        """
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        target_pos, _ = p.getBasePositionAndOrientation(self.target_id)

        robot_xy = np.array(robot_pos[:2])
        target_xy = np.array(target_pos[:2])

        obs = np.array(
            [
                *robot_xy,
                *target_xy,
                *(target_xy - robot_xy),
            ],
            dtype=np.float32,
        )
        return obs

    def _distance_to_target(self):
        """Planar (x, y) distance between the robot and the target."""
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        target_pos, _ = p.getBasePositionAndOrientation(self.target_id)
        return float(np.linalg.norm(np.array(robot_pos[:2]) - np.array(target_pos[:2])))

    def _compute_reward(self, distance):
        """Compute reward signal from a precomputed distance."""
        reward = -distance
        if distance < SUCCESS_DISTANCE:
            reward += SUCCESS_BONUS
        return reward

    def render(self):
        """Render environment."""

    def close(self):
        """Close environment."""
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None

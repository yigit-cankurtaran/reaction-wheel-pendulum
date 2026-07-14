"""Gymnasium environment for an internally actuated reaction-wheel balancer."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class ReactionWheelPendulumEnv(gym.Env[np.ndarray, np.ndarray]):
    """Recover and balance a pendulum using only an internal flywheel.

    Angle zero is the hanging-down configuration and pi is upright.  Observations
    avoid the angle discontinuity and are ``[cos(theta), sin(theta), body_rate,
    wheel_rate]`` (rates are scaled by 10 and 100 rad/s). The normalized action is
    the wheel motor torque.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        frame_skip: int = 5,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "rgb_array", "human"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        xml_path = Path(__file__).parent / "assets" / "reaction_wheel_pendulum.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -5.0, -10.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 5.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._torque_limit = float(self.model.actuator_ctrlrange[0, 1])

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    @property
    def viewer_is_running(self) -> bool:
        """Whether the human viewer remains open, when one has been launched."""
        return self._viewer is None or self._viewer.is_running()

    def _get_obs(self) -> np.ndarray:
        theta = float(self.data.qpos[0])
        obs = np.array(
            [np.cos(theta), np.sin(theta), self.data.qvel[0] / 10.0, self.data.qvel[1] / 100.0],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _metrics(self, action: float) -> tuple[float, bool, dict[str, float]]:
        theta = float(self.data.qpos[0])
        body_rate, wheel_rate = map(float, self.data.qvel)
        uprightness = 0.5 * (1.0 - np.cos(theta))
        angle_error = abs(float(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))))
        balanced = angle_error < np.deg2rad(12.0) and abs(body_rate) < 0.8

        # Dense swing-up signal plus a sharp bonus for settling at the top.
        reward = (
            3.0 * uprightness
            + 3.0 * float(balanced)
            - 0.01 * body_rate**2
            - 0.0001 * wheel_rate**2
            - 0.001 * action**2
        )
        info = {
            "angle_error_rad": angle_error,
            "body_rate": body_rate,
            "wheel_rate": wheel_rate,
            "is_balanced": float(balanced),
        }
        return float(reward), balanced, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        options = options or {}
        if options.get("upright", False):
            theta = np.pi + self.np_random.uniform(-0.08, 0.08)
        elif options.get("hanging", False):
            theta = self.np_random.uniform(-0.20, 0.20)
        else:
            # Attitude-recovery curriculum: realistic disturbances around upright,
            # with occasional global starts to keep the policy state-aware.
            theta = np.pi + self.np_random.uniform(-0.60, 0.60)
            if self.np_random.random() < 0.10:
                theta = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[:] = [theta, self.np_random.uniform(-np.pi, np.pi)]
        self.data.qvel[:] = self.np_random.normal(0.0, [0.35, 0.5])
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {"initial_angle_rad": float(theta)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        normalized_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.data.ctrl[0] = normalized_action * self._torque_limit
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        reward, _, info = self._metrics(normalized_action)
        terminated = not bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        truncated = self._step_count >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.0, 0.0, -0.38]
            camera.distance = 2.7
            camera.azimuth = 90
            camera.elevation = 0
            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

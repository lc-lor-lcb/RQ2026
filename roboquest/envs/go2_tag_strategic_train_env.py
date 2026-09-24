"""Training-only strategic flee environment for Tier2 experiments.

This environment keeps the same observation/action contract as
Go2TagHierarchicalEnv. It changes only the training reward and optional
curriculum parameters, so policies remain compatible with the standard
evaluation environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mujoco
import numpy as np

from roboquest.envs.go2_tag_env import (
    ARENA_HALF,
    CONTROL_HZ,
    ONI_QPOS_X,
    ONI_QPOS_Y,
    ONI_QVEL_X,
    ONI_QVEL_Y,
)
from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv, N_LOW_STEPS


@dataclass
class StrategicRewardConfig:
    distance_delta_weight: float = 8.0
    distance_closing_penalty: float = 10.0
    wall_margin_soft: float = 0.45
    wall_margin_hard: float = 0.25
    wall_penalty_weight: float = 1.5
    danger_wall_penalty_weight: float = 4.0
    danger_distance: float = 1.15
    escape_bonus: float = 100.0
    action_change_weight: float = 0.02
    lateral_action_weight: float = 0.04
    yaw_action_weight: float = 0.02
    stall_penalty_weight: float = 0.03
    min_forward_hint: float = 0.08
    extra_info: dict = field(default_factory=dict)


class Go2TagStrategicTrainEnv(Go2TagHierarchicalEnv):
    """Tier2 training env with stronger strategic rewards.

    The policy still sees the original 10-dimensional high-level observation
    and outputs the original 3-dimensional velocity command.
    """

    def __init__(
        self,
        *args,
        strategic_config: Optional[StrategicRewardConfig | dict] = None,
        initial_distance_range: Tuple[float, float] = (1.5, ARENA_HALF),
        max_episode_seconds: float = 60.0,
        **kwargs,
    ):
        n_low_steps = int(kwargs.get("n_low_steps", N_LOW_STEPS))
        kwargs["max_episode_steps"] = max(1, int(max_episode_seconds * CONTROL_HZ / n_low_steps))
        kwargs.setdefault("high_level_command_mode", "safe_forward")
        super().__init__(*args, **kwargs)
        if isinstance(strategic_config, dict):
            self.strategic_config = StrategicRewardConfig(**strategic_config)
        else:
            self.strategic_config = strategic_config or StrategicRewardConfig()
        self.initial_distance_range = initial_distance_range
        self._prev_oni_distance = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._wall_steps = 0
        self._min_distance = float("inf")
        self._min_wall_margin = float("inf")

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if options:
            if "initial_distance_range" in options:
                self.initial_distance_range = tuple(options["initial_distance_range"])
            if "oni_speed" in options:
                self.oni_speed = float(options["oni_speed"])

        lo, hi = self.initial_distance_range
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        dist = self.np_random.uniform(float(lo), float(hi))
        oni_x = np.clip(dist * np.cos(angle), -ARENA_HALF, ARENA_HALF)
        oni_y = np.clip(dist * np.sin(angle), -ARENA_HALF, ARENA_HALF)
        self.data.qpos[ONI_QPOS_X] = oni_x
        self.data.qpos[ONI_QPOS_Y] = oni_y
        self.data.qvel[ONI_QVEL_X] = 0.0
        self.data.qvel[ONI_QVEL_Y] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._prev_oni_distance = self._oni_distance()
        self._prev_action[:] = 0.0
        self._wall_steps = 0
        self._min_distance = self._prev_oni_distance
        self._min_wall_margin = self._wall_margin()
        return self._get_high_obs(), info

    def step(self, vel_cmd):
        vel_cmd = np.asarray(vel_cmd, dtype=np.float32)
        prev_dist = float(self._oni_distance())
        prev_action = self._prev_action.copy()

        obs, reward, terminated, truncated, info = super().step(vel_cmd)

        cfg = self.strategic_config
        dist = float(self._oni_distance())
        delta = dist - prev_dist
        wall_margin = self._wall_margin()
        self._min_distance = min(self._min_distance, dist)
        self._min_wall_margin = min(self._min_wall_margin, wall_margin)
        if wall_margin < cfg.wall_margin_soft:
            self._wall_steps += 1

        if delta >= 0.0:
            reward += cfg.distance_delta_weight * delta
        else:
            reward += cfg.distance_closing_penalty * delta

        soft_violation = max(0.0, cfg.wall_margin_soft - wall_margin)
        hard_violation = max(0.0, cfg.wall_margin_hard - wall_margin)
        reward -= cfg.wall_penalty_weight * soft_violation * soft_violation
        reward -= cfg.wall_penalty_weight * 4.0 * hard_violation * hard_violation
        if dist < cfg.danger_distance:
            reward -= cfg.danger_wall_penalty_weight * soft_violation

        action_change = float(np.sum((np.clip(vel_cmd, -1.0, 1.0) - prev_action) ** 2))
        reward -= cfg.action_change_weight * action_change
        reward -= cfg.lateral_action_weight * float(vel_cmd[1] ** 2)
        reward -= cfg.yaw_action_weight * float(vel_cmd[2] ** 2)
        if abs(float(vel_cmd[0])) < cfg.min_forward_hint and dist < cfg.danger_distance:
            reward -= cfg.stall_penalty_weight

        escaped = bool(truncated and not terminated)
        if escaped:
            reward += cfg.escape_bonus

        failure_reason = "running"
        if escaped:
            failure_reason = "escaped"
        elif terminated and bool(info.get("is_tagged", False)):
            failure_reason = "tagged"
        elif terminated:
            failure_reason = "fell"

        info.update(
            distance_delta=float(delta),
            wall_margin=float(wall_margin),
            wall_steps=int(self._wall_steps),
            min_oni_distance=float(self._min_distance),
            min_wall_margin=float(self._min_wall_margin),
            escaped=escaped,
            failure_reason=failure_reason,
        )
        self._prev_oni_distance = dist
        self._prev_action = np.clip(vel_cmd, -1.0, 1.0).astype(np.float32)
        return obs, reward, terminated, truncated, info

    def _wall_margin(self) -> float:
        xy = self._low_env.robot_xy
        return float(ARENA_HALF - max(abs(float(xy[0])), abs(float(xy[1]))))

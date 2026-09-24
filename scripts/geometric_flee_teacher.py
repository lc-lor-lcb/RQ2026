"""Geometric flee teacher for imitation-learning experiments.

This controller is intentionally simple and deterministic. It is meant to
produce teacher actions that can be distilled into a submission-compatible PPO
policy later.
"""
from __future__ import annotations

import numpy as np

from roboquest.envs.go2_tag_env import ARENA_HALF


def teacher_action_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return [vx, vy, omega] from the standard 10D high-level observation.

    obs layout: rel_dx, rel_dy, distance, ang_vel(3), projected_gravity(3), time_left.
    The low-level walk policy is best at forward motion, so this teacher mostly
    asks for forward movement and uses yaw/lateral commands sparingly.
    """
    rel = np.asarray(obs[:2], dtype=np.float64)
    dist = float(obs[2])
    if np.linalg.norm(rel) < 1e-6:
        away = np.array([1.0, 0.0])
    else:
        away = -rel / np.linalg.norm(rel)

    # Without global position in the compatible observation, use the relative
    # vector only. The train/eval scripts can provide wall-aware variants by
    # calling teacher_action_from_state instead.
    vx = 0.75 if dist < 1.5 else 0.45
    vy = float(np.clip(0.25 * away[1], -0.35, 0.35))
    omega = float(np.clip(-0.55 * away[1], -0.8, 0.8))
    return np.array([vx, vy, omega], dtype=np.float32)


def teacher_action_from_state(robot_xy: np.ndarray, oni_xy: np.ndarray) -> np.ndarray:
    rel = oni_xy - robot_xy
    dist = float(np.linalg.norm(rel))
    away = -rel / max(dist, 1e-6)
    margin_x = ARENA_HALF - abs(float(robot_xy[0]))
    margin_y = ARENA_HALF - abs(float(robot_xy[1]))
    wall_push = np.zeros(2)
    if margin_x < 0.45:
        wall_push[0] = -np.sign(robot_xy[0]) * (0.45 - margin_x)
    if margin_y < 0.45:
        wall_push[1] = -np.sign(robot_xy[1]) * (0.45 - margin_y)
    desired = away + 1.8 * wall_push
    norm = np.linalg.norm(desired)
    if norm > 1e-6:
        desired = desired / norm
    vx = 0.8 if dist < 1.4 else 0.55
    vy = float(np.clip(0.35 * desired[1], -0.45, 0.45))
    omega = float(np.clip(-0.9 * desired[1], -1.0, 1.0))
    return np.array([vx, vy, omega], dtype=np.float32)

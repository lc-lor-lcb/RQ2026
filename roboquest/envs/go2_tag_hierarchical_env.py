"""
階層学習環境: 歩行ポリシーを活用した鬼ごっこ逃げポリシーの学習

階層構造:
  高レベルポリシー（学習対象）
    - 行動: 速度コマンド [vx, vy, omega] ∈ [-1, 1]^3
    - 観測: 鬼への相対位置 + ロボット姿勢（簡略版）
    - 報酬: 生存ボーナス + 鬼との距離報酬 - タグペナルティ

  低レベルポリシー（事前学習済み、固定）
    - Go2WalkEnv で学習した速度コマンド追従ポリシー
    - 高レベルのコマンドを受け取り、関節制御を実行

使い方:
    env = Go2TagHierarchicalEnv(low_level_model_path="walk_model.zip")
    # あとは通常の Gymnasium 環境と同じ
"""
import os
import json
from pathlib import Path
import pickle

from typing import Optional, List

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from roboquest.envs.go2_tag_env import (
    Go2TagEnv, ARENA_XML, TAG_DISTANCE, MAX_EPISODE_STEPS,
    ARENA_HALF, ONI_QPOS_X, ONI_QPOS_Y, ONI_QVEL_X, ONI_QVEL_Y, CONTROL_HZ,
)
from roboquest.envs.go2_walk_env import Go2WalkEnv, STANDING_POS, ACTION_SCALE, KP, KD
from roboquest.utils.reward_utils import FleeRewardConfig

# 高レベル1ステップあたりの低レベルステップ数
N_LOW_STEPS = 10  # 高レベル 5Hz、低レベル 50Hz


class Go2TagHierarchicalEnv(gym.Env):
    """階層強化学習による鬼ごっこ逃げ環境。

    高レベルポリシーが速度コマンドを出力し、
    事前学習済みの低レベルポリシーが実際の関節制御を行う。

    高レベル観測 (10次元):
      [0:2]  鬼への相対位置 dx, dy
      [1]    鬼までの距離
      [3:6]  ロボットの角速度 xyz
      [6:9]  重力方向ベクトル（姿勢）
      [9]    残り時間（正規化: 1.0→0.0）
    合計: 10次元

    高レベル行動 (3次元):
      [vx, vy, omega] ∈ [-1, 1]^3
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        low_level_model_path: str,
        low_level_vecnorm_path: Optional[str] = None,
        flee_config: Optional[FleeRewardConfig] = None,
        oni_speed: float = 0.025,
        max_episode_steps: int = MAX_EPISODE_STEPS // N_LOW_STEPS,
        render_mode: Optional[str] = None,
        n_low_steps: int = N_LOW_STEPS,
        high_level_command_mode: str = "direct",
        high_level_command_ranges: Optional[dict] = None,
        high_level_observation_mode: str = "standard",
    ):
        super().__init__()
        self.flee_config = flee_config or FleeRewardConfig()
        self.oni_speed = oni_speed
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.n_low_steps = n_low_steps
        self.high_level_command_mode = high_level_command_mode
        self.high_level_observation_mode = high_level_observation_mode
        if high_level_command_ranges is None:
            high_level_command_ranges = {
                "vx": (0.25, 0.55),
                "vy": (-0.05, 0.05),
                "omega": (-0.20, 0.20),
            }
        self.high_level_command_ranges = high_level_command_ranges
        self._step_count = 0

        # Carry the trained action range into the hierarchical simulation.
        params_path = Path(low_level_model_path).parent / 'walk_params.json'
        walk_params = json.loads(params_path.read_text()) if params_path.exists() else {}
        walk_env_kwargs = walk_params.get('walk_env_kwargs', {})

        # 低レベル環境（物理シミュレーション本体）
        self._low_env = Go2WalkEnv(
            xml_path=ARENA_XML,
            randomize_cmd=False,
            **walk_env_kwargs,
        )
        self.model = self._low_env.model
        self.data = self._low_env.data

        # 鬼の初期位置
        self._oni_init = np.zeros(2, dtype=np.float64)

        # 低レベルポリシー読み込み
        from stable_baselines3 import PPO
        self._low_policy = PPO.load(low_level_model_path)
        self._low_obs_rms = None
        self._low_clip_obs = 10.0
        self._low_epsilon = 1e-8
        self._load_low_vecnorm(low_level_model_path, low_level_vecnorm_path)

        # 高レベル観測・行動空間
        # 観測: [rel_dx, rel_dy, dist, ang_vel(3), proj_grav(3), time_left]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        # 行動: 速度コマンド [vx, vy, omega]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # 俯瞰レンダラー
        if render_mode == "rgb_array":
            self._top_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._top_cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view"
            )
        else:
            self._top_renderer = None
            self._top_cam_id = -1

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._low_env.reset(seed=seed)

        # 鬼の初期配置
        angle = self.np_random.uniform(0, 2 * np.pi)
        dist = self.np_random.uniform(1.5, ARENA_HALF)
        oni_x = np.clip(dist * np.cos(angle), -ARENA_HALF, ARENA_HALF)
        oni_y = np.clip(dist * np.sin(angle), -ARENA_HALF, ARENA_HALF)
        self.data.qpos[ONI_QPOS_X] = oni_x
        self.data.qpos[ONI_QPOS_Y] = oni_y
        self.data.qvel[ONI_QVEL_X] = 0.0
        self.data.qvel[ONI_QVEL_Y] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_high_obs(), {}

    def step(self, vel_cmd: np.ndarray):
        """高レベルポリシーの1ステップ（内部で n_low_steps 回低レベル制御を実行）。"""
        raw_action = np.clip(np.asarray(vel_cmd, dtype=np.float32), -1.0, 1.0)
        vel_cmd = self._map_high_level_action(raw_action)
        self._low_env.set_vel_cmd(float(vel_cmd[0]), float(vel_cmd[1]), float(vel_cmd[2]))

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.n_low_steps):
            low_obs = self._low_env._get_obs().astype(np.float32)
            low_obs = self._normalize_low_obs(low_obs)
            low_action, _ = self._low_policy.predict(low_obs, deterministic=True)
            _, _, low_term, _, _ = self._low_env.step(low_action)

            # 鬼の移動
            self._move_oni()

            # 高レベル報酬
            oni_dist = self._oni_distance()
            total_reward += self.flee_config.survival_weight / self.n_low_steps
            total_reward += self.flee_config.distance_weight * oni_dist * 0.01

            if oni_dist < TAG_DISTANCE:
                terminated = True
                total_reward -= self.flee_config.tag_penalty
                break

            if low_term:  # 転倒
                terminated = True
                total_reward -= self.flee_config.fall_penalty
                break

        self._step_count += 1
        truncated = (not terminated) and (self._step_count >= self.max_episode_steps)

        info["oni_distance"] = float(self._oni_distance())
        info["is_tagged"] = bool(self._oni_distance() < TAG_DISTANCE)
        info["survived_seconds"] = (
            self._step_count * self.n_low_steps / CONTROL_HZ
        )
        info["raw_high_action"] = raw_action.astype(float).tolist()
        info["velocity_command"] = vel_cmd.astype(float).tolist()

        return self._get_high_obs(), total_reward, terminated, truncated, info

    def render(self):
        if self._top_renderer is None:
            return None
        opt = mujoco.MjvOption()
        self._top_renderer.update_scene(
            self.data, camera=self._top_cam_id, scene_option=opt
        )
        return self._top_renderer.render()

    def close(self):
        if self._top_renderer is not None:
            self._top_renderer.close()
        self._low_env.close()

    def record_episode(
        self,
        model=None,
        fps: int = 5,
    ) -> List[np.ndarray]:
        """俯瞰カメラで1エピソードを録画。

        Args:
            model: 高レベル学習済みモデル（None でランダム行動）
            fps: フレームレート（高レベルステップレートに合わせて低め）
        """
        frames = []
        obs, _ = self.reset()

        for _ in range(self.max_episode_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = self.action_space.sample()

            obs, _, terminated, truncated, _ = self.step(action)
            frame = self.render()
            if frame is not None:
                frames.append(frame.copy())

            if terminated or truncated:
                break

        return frames

    # ── 内部メソッド ─────────────────────────────────────────────────────

    def _load_low_vecnorm(
        self,
        low_level_model_path: str,
        low_level_vecnorm_path: Optional[str],
    ) -> None:
        """Load walk-policy VecNormalize stats when available."""
        candidates = []
        if low_level_vecnorm_path:
            candidates.append(low_level_vecnorm_path)
        candidates.append(low_level_model_path + "_vecnorm.pkl")

        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                vec_norm = pickle.load(f)
            self._low_obs_rms = vec_norm.obs_rms
            self._low_clip_obs = float(getattr(vec_norm, "clip_obs", 10.0))
            self._low_epsilon = float(getattr(vec_norm, "epsilon", 1e-8))
            return

    def _normalize_low_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._low_obs_rms is None:
            return obs
        obs = (obs - self._low_obs_rms.mean) / np.sqrt(
            self._low_obs_rms.var + self._low_epsilon
        )
        return np.clip(obs, -self._low_clip_obs, self._low_clip_obs).astype(np.float32)

    def _map_high_level_action(self, action: np.ndarray) -> np.ndarray:
        """Map policy action to a walk-policy-friendly velocity command.

        The bundled smooth walk policy is validated mainly for standing and
        forward motion. Passing an untrained high-level PPO action directly as
        [vx, vy, omega] often asks the low-level policy for backward/lateral
        commands it never learned, causing immediate falls and no useful
        learning signal. The default keeps the action space shape unchanged but
        maps it into a conservative forward-moving command range.
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.high_level_command_mode == "direct":
            return action.astype(np.float32)
        if self.high_level_command_mode == "auto_escape":
            return self._auto_escape_command()
        if self.high_level_command_mode != "safe_forward":
            raise ValueError(f"Unknown high_level_command_mode: {self.high_level_command_mode}")

        def scale(name: str, value: float) -> float:
            lo, hi = self.high_level_command_ranges[name]
            return float(lo + (float(value) + 1.0) * 0.5 * (hi - lo))

        return np.array([
            scale("vx", action[0]),
            scale("vy", action[1]),
            scale("omega", action[2]),
        ], dtype=np.float32)

    def _auto_escape_command(self) -> np.ndarray:
        """Rule-based escape command used when learning fails to bootstrap."""
        robot = self._low_env.robot_xy
        oni = self._oni_xy
        away = robot - oni
        dist = float(np.linalg.norm(away))
        if dist < 1e-6:
            away = np.array([1.0, 0.0])
        else:
            away = away / dist

        wall_push = np.zeros(2, dtype=np.float64)
        wall_margin = 0.65
        if ARENA_HALF - abs(float(robot[0])) < wall_margin:
            wall_push[0] = -np.sign(float(robot[0])) * 2.0
        if ARENA_HALF - abs(float(robot[1])) < wall_margin:
            wall_push[1] = -np.sign(float(robot[1])) * 2.0

        # Pure "run directly away" reaches the wall quickly in a 5m arena.
        # Add a tangent component and choose the tangent that points farther
        # from the nearest wall, creating an orbit-like escape path.
        tangent_a = np.array([-away[1], away[0]], dtype=np.float64)
        tangent_b = -tangent_a
        pos_a = robot + 0.6 * tangent_a
        pos_b = robot + 0.6 * tangent_b
        margin_a = ARENA_HALF - max(abs(float(pos_a[0])), abs(float(pos_a[1])))
        margin_b = ARENA_HALF - max(abs(float(pos_b[0])), abs(float(pos_b[1])))
        tangent = tangent_a if margin_a >= margin_b else tangent_b
        tangent_weight = 0.75 if dist > 1.15 else 0.35

        desired_world = away + tangent_weight * tangent + wall_push
        desired_world = desired_world / max(float(np.linalg.norm(desired_world)), 1e-6)
        desired_body = self._low_env._world_to_body(np.array([desired_world[0], desired_world[1], 0.0]))[:2]
        yaw_error = float(np.arctan2(desired_body[1], desired_body[0]))

        # Sprint when the escape path is roughly ahead. If it is behind, use
        # the walker's validated hard-turn gait instead of running toward oni.
        if abs(yaw_error) < 0.65:
            vx = 1.45
        elif abs(yaw_error) < 1.45:
            vx = 1.05
        else:
            vx = 0.35
        omega = float(np.clip(2.4 * yaw_error, -1.15, 1.15))
        return np.array([vx, 0.0, omega], dtype=np.float32)

    def _get_high_obs(self) -> np.ndarray:
        robot_xy = self._low_env.robot_xy
        oni_xy = self._oni_xy
        rel = oni_xy - robot_xy
        if self.high_level_observation_mode == "body_relative":
            rel = self._low_env._world_to_body(np.array([rel[0], rel[1], 0.0]))[:2]
        elif self.high_level_observation_mode != "standard":
            raise ValueError(f"Unknown high_level_observation_mode: {self.high_level_observation_mode}")
        dist = np.linalg.norm(rel)
        ang_vel = self.data.qvel[3:6].copy()
        proj_grav = self._low_env._projected_gravity()
        time_left = 1.0 - self._step_count / self.max_episode_steps
        return np.concatenate([
            rel, [dist], ang_vel, proj_grav, [time_left]
        ]).astype(np.float32)

    def _move_oni(self) -> None:
        oni_xy = self._oni_xy
        robot_xy = self._low_env.robot_xy
        direction = robot_xy - oni_xy
        d = np.linalg.norm(direction)
        if d > 1e-6:
            delta = self.oni_speed * direction / d
            new_pos = np.clip(oni_xy + delta, -ARENA_HALF, ARENA_HALF)
            self.data.qpos[ONI_QPOS_X] = new_pos[0]
            self.data.qpos[ONI_QPOS_Y] = new_pos[1]
        self.data.qvel[ONI_QVEL_X] = 0.0
        self.data.qvel[ONI_QVEL_Y] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _oni_distance(self) -> float:
        return float(np.linalg.norm(self._low_env.robot_xy - self._oni_xy))

    @property
    def _oni_xy(self) -> np.ndarray:
        return np.array([
            self.data.qpos[ONI_QPOS_X],
            self.data.qpos[ONI_QPOS_Y],
        ], dtype=np.float64)

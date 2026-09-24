"""
鬼ごっこ環境: 逃げる Go2 vs 追いかける鬼

フィールド:
  - 5m × 5m の壁付きアリーナ（arena.xml）
  - 俯瞰カメラ (top_view) で2体を録画

鬼:
  - 赤い球として arena.xml 内で定義
  - ルールベースで Go2 に直進するスクリプト制御
  - qpos[19]=x, qpos[20]=y で位置を直接管理

終了条件:
  - タグされる (鬼との距離 < TAG_DISTANCE) → terminated
  - 転倒 (胴体高さ < 0.15m) → terminated
  - 1分間逃げ切り (3000 ステップ @ 50Hz) → truncated（成功）

観測空間: 45 + 3 = 48 次元
"""
import os

from typing import Optional, List

import mujoco
import numpy as np
from gymnasium import spaces

from roboquest.envs.go2_walk_env import Go2WalkEnv, _MODEL_DIR
from roboquest.utils.reward_utils import FleeRewardConfig

# アリーナ（壁付き5m×5m + 俯瞰カメラ + 鬼）。
# 歩行モデルは go2_posctrl.xml の位置制御で学習するため、Tag環境も
# 同じ制御方式のアリーナを使う。motor版の arena.xml へ q_target を
# 渡すとトルク指令として解釈され、立つだけでも即転倒する。
ARENA_XML = os.path.join(_MODEL_DIR, "arena_posctrl.xml")

# タグ判定距離（鬼の球半径 0.28m + ロボット半径 ~0.3m で余裕をもたせ）
TAG_DISTANCE = 0.65

# 1分間 = 60s × 50Hz
EPISODE_SECONDS = 60
CONTROL_HZ = 50
MAX_EPISODE_STEPS = EPISODE_SECONDS * CONTROL_HZ  # = 3000

# アリーナ内部（壁の内側）
ARENA_HALF = 2.35  # 壁が ±2.55 → 内側は ±2.35m

# 鬼の qpos/qvel インデックス（arena.xml: go2 freejoint=7, joints=12 → oni 開始=19）
ONI_QPOS_X = 19
ONI_QPOS_Y = 20
ONI_QVEL_X = 18
ONI_QVEL_Y = 19


class Go2TagEnv(Go2WalkEnv):
    """鬼ごっこ環境（1分間逃げ切りを目標）。

    追加観測 (3次元):
      - 鬼への相対位置 dx, dy (2)
      - 鬼までの距離 d (1)
    合計観測: 45 + 3 = 48次元

    録画には俯瞰カメラ "top_view" を使用。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        flee_config: Optional[FleeRewardConfig] = None,
        oni_speed: float = 0.025,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            xml_path=ARENA_XML,
            randomize_cmd=False,   # タグ環境では速度コマンドは使わない
        )
        self.flee_config = flee_config or FleeRewardConfig()
        self.oni_speed = oni_speed

        # 観測空間を 45 → 48 に拡張
        obs_dim = self.observation_space.shape[0] + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 俯瞰カメラ用レンダラー（top_view）
        self._top_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._top_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view"
        )

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)

        # 鬼の初期位置: ロボットの反対側にランダム配置
        angle = self.np_random.uniform(0, 2 * np.pi)
        dist = self.np_random.uniform(1.5, ARENA_HALF)
        oni_x = np.clip(dist * np.cos(angle), -ARENA_HALF, ARENA_HALF)
        oni_y = np.clip(dist * np.sin(angle), -ARENA_HALF, ARENA_HALF)
        self.data.qpos[ONI_QPOS_X] = oni_x
        self.data.qpos[ONI_QPOS_Y] = oni_y
        self.data.qvel[ONI_QVEL_X] = 0.0
        self.data.qvel[ONI_QVEL_Y] = 0.0
        mujoco.mj_forward(self.model, self.data)

        return self._tag_obs(obs), info

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = super().step(action)

        # 鬼を移動（物理後に直接 qpos 更新 + mj_forward）
        self._move_oni()

        # 鬼ごっこ報酬
        dist = self._oni_distance()
        reward = self._compute_flee_reward(dist)

        # タグ判定
        if dist < TAG_DISTANCE:
            terminated = True
            reward -= self.flee_config.tag_penalty

        info["oni_distance"] = float(dist)
        info["is_tagged"] = bool(dist < TAG_DISTANCE)
        info["survived_steps"] = self._step_count
        info["survived_seconds"] = self._step_count / CONTROL_HZ

        return self._tag_obs(obs), reward, terminated, truncated, info

    def render(self):
        """俯瞰カメラ映像を返す（render_mode='rgb_array' の場合）。"""
        if self._top_renderer is None:
            return None
        opt = mujoco.MjvOption()
        self._top_renderer.update_scene(self.data, camera=self._top_cam_id, scene_option=opt)
        return self._top_renderer.render()

    def close(self):
        if self._top_renderer is not None:
            self._top_renderer.close()
        super().close()

    # ── 録画ユーティリティ ────────────────────────────────────────────────

    def record_episode(
        self,
        model=None,
        n_steps: Optional[int] = None,
        fps: int = 30,
    ) -> List[np.ndarray]:
        """俯瞰カメラで1エピソードを録画してフレームリストを返す。

        Args:
            model: 学習済み SB3 モデル（None でランダム行動）
            n_steps: 録画ステップ数（None で最大エピソード長）
            fps: 記録フレームレート

        Returns:
            RGB フレームのリスト
        """
        n_steps = n_steps or self.max_episode_steps
        frames = []

        obs, _ = self.reset()
        step_per_frame = max(1, CONTROL_HZ // fps)

        for i in range(n_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = self.action_space.sample()

            obs, _, terminated, truncated, _ = self.step(action)

            if i % step_per_frame == 0:
                frames.append(self.render().copy())

            if terminated or truncated:
                break

        return frames

    # ── 内部メソッド ─────────────────────────────────────────────────────

    def _move_oni(self) -> None:
        """鬼をルールベースで Go2 に向けて移動（kinematic）。"""
        oni_xy = self._oni_xy
        robot_xy = self.robot_xy
        direction = robot_xy - oni_xy
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            delta = self.oni_speed * direction / dist
            new_pos = np.clip(oni_xy + delta, -ARENA_HALF, ARENA_HALF)
            self.data.qpos[ONI_QPOS_X] = new_pos[0]
            self.data.qpos[ONI_QPOS_Y] = new_pos[1]
        self.data.qvel[ONI_QVEL_X] = 0.0
        self.data.qvel[ONI_QVEL_Y] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _tag_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """基本観測に鬼への相対位置と距離を付加する。"""
        rel = self._oni_xy - self.robot_xy  # ロボットから見た鬼の方向
        dist = np.linalg.norm(rel).reshape(1)
        return np.concatenate([base_obs, rel, dist]).astype(np.float32)

    def _compute_flee_reward(self, oni_dist: float) -> float:
        cfg = self.flee_config
        r_survive = cfg.survival_weight
        r_dist = cfg.distance_weight * oni_dist * 0.1
        return r_survive + r_dist

    def _oni_distance(self) -> float:
        return float(np.linalg.norm(self.robot_xy - self._oni_xy))

    @property
    def _oni_xy(self) -> np.ndarray:
        return np.array([
            self.data.qpos[ONI_QPOS_X],
            self.data.qpos[ONI_QPOS_Y],
        ], dtype=np.float64)

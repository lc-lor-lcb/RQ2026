"""
Go2 四足歩行ロボットの強化学習環境（歩行事前学習用）

unitree_rl_mjlab の velocity task を参考にした報酬設計:
  - 速度コマンドへの追従（線速度・角速度）
  - 姿勢の安定性（重力方向の傾き）
  - エネルギー効率（トルク・アクション変化）
  - 足のスリップ防止

観測空間: 45次元
行動空間: 12次元（正規化された目標関節角度オフセット [-1, 1]）
"""
import os

from typing import Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from roboquest.utils.reward_utils import WalkRewardConfig

# デフォルト立ち姿勢 keyframe "home" より
# 順序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#       RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
STANDING_POS = np.array([
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
], dtype=np.float64)

CONTROL_DT = 0.02  # 50 Hz; shared by local training and Colab

ACTION_SCALE = 0.3   # action * ACTION_SCALE + STANDING_POS = 目標関節角度
# KP/KD は go2_posctrl.xml の <position kp=20> と joint damping=0.5 で設定済み。
# 報酬計算用に定数として保持する。
KP = 20.0
KD = 0.5

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "go2")
# walk_scene.xml → go2_posctrl.xml（位置制御）+ 床
MODEL_XML = os.path.join(_MODEL_DIR, "walk_scene.xml")

# 速度コマンドのサンプリング範囲
VEL_CMD_RANGE = {
    "vx":    (-1.0,  1.0),   # 前後 (m/s 相当)
    "vy":    (-0.5,  0.5),   # 左右
    "omega": (-1.0,  1.0),   # 回転 (rad/s 相当)
}

# 足ゼオム名（go2.xml / go2_posctrl.xml に定義されている <geom name="FR" ...> 等）
# アクチュエータ順に合わせて FR, FL, RR, RL の順で並べる
FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]


class Go2WalkEnv(gym.Env):
    """Go2 四足歩行の基本環境（速度コマンド追従）。

    観測 (45次元):
      [0:3]   速度コマンド (vx, vy, omega)
      [3:6]   胴体角速度 xyz
      [6:9]   重力方向ベクトル（ボディフレーム）
      [9:21]  関節角度（立ち姿勢からの相対値）
      [21:33] 関節角速度
      [33:45] 前回行動
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        reward_config: Optional[WalkRewardConfig] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        xml_path: Optional[str] = None,
        randomize_cmd: bool = True,
        command_ranges: Optional[dict] = None,
        action_scale: float = ACTION_SCALE,
        command_mode: str = "uniform",
        joint_stiffness: float = KP,
        joint_damping: float = KD,
        torque_limits: Optional[list] = None,
    ):
        super().__init__()
        self.reward_config = reward_config or WalkRewardConfig()
        if min(self.reward_config.linear_tracking_variance, self.reward_config.angular_tracking_variance) <= 0:
            raise ValueError("Tracking variances must be positive")
        if not np.isfinite(action_scale) or action_scale <= 0:
            raise ValueError("action_scale must be finite and positive")
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        if command_mode not in {"uniform", "axis"}:
            raise ValueError("command_mode must be uniform or axis")
        self.command_mode = command_mode
        self.action_scale = float(action_scale)
        self.randomize_cmd = randomize_cmd
        self.command_ranges = dict(VEL_CMD_RANGE if command_ranges is None else command_ranges)
        self._step_count = 0

        xml = xml_path or MODEL_XML
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = round(CONTROL_DT / self.model.opt.timestep)
        if not np.isclose(self.frame_skip * self.model.opt.timestep, CONTROL_DT):
            raise ValueError("Physics timestep must divide CONTROL_DT")

        obs_dim = 3 + 3 + 3 + 12 + 12 + 12  # = 45
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._last_action = np.zeros(12, dtype=np.float64)
        self._vel_cmd = np.zeros(3, dtype=np.float64)   # [vx, vy, omega]

        # アクチュエータ順の qpos/dof アドレス（qpos 順序とアクチュエータ順序は異なる）
        self._act_qposadr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ], dtype=int)
        self._act_dofadr = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ], dtype=int)
        if not np.isfinite(joint_stiffness) or joint_stiffness <= 0:
            raise ValueError('joint_stiffness must be finite and positive')
        if not np.isfinite(joint_damping) or joint_damping < 0:
            raise ValueError('joint_damping must be finite and nonnegative')
        self.model.actuator_gainprm[:, 0] = joint_stiffness
        self.model.actuator_biasprm[:, 1] = -joint_stiffness
        self.model.dof_damping[self._act_dofadr] = joint_damping
        if torque_limits is not None:
            limits = np.asarray(torque_limits, dtype=float)
            if limits.shape != (3,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
                raise ValueError('torque_limits must contain positive hip/thigh/calf limits')
            limits = np.tile(limits, 4)
            self.model.actuator_forcelimited[:] = True
            self.model.actuator_forcerange[:] = np.stack([-limits, limits], axis=1)

        # 足ゼオム ID の取得（foot slip 報酬用）
        self._foot_geom_ids = []
        for name in FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._foot_geom_ids.append(gid)

        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        else:
            self._renderer = None

        self._knee_body_ids = [self.model.body(name + "_calf").id
                               for name in FOOT_GEOM_NAMES]

    def posture_metrics(self):
        """Flat-floor clearance/contact diagnostics shared by reward and evaluation."""
        nonfoot = set()
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if self.model.geom_bodyid[a] == 0:
                robot = b
            elif self.model.geom_bodyid[b] == 0:
                robot = a
            else:
                continue
            if robot not in self._foot_geom_ids and contact.dist <= 0:
                nonfoot.add(robot)
        return {"nonfoot_contacts": len(nonfoot),
                "knee_heights": self.data.xpos[self._knee_body_ids, 2].copy()}

    # ── 公開 API ─────────────────────────────────────────────────────────

    def set_vel_cmd(self, vx: float, vy: float, omega: float) -> None:
        """外部（高レベルポリシー）から速度コマンドを設定する。"""
        self._vel_cmd = np.array([vx, vy, omega], dtype=np.float64)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # keyframe "home" から起動＋微小ノイズ
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qpos[7:19] += self.np_random.uniform(-0.05, 0.05, 12)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.model.key_ctrl[0]
        mujoco.mj_forward(self.model, self.data)

        # 速度コマンドをランダムサンプリング
        if self.randomize_cmd:
            self._vel_cmd = np.array([
                self.np_random.uniform(*self.command_ranges["vx"]),
                self.np_random.uniform(*self.command_ranges["vy"]),
                self.np_random.uniform(*self.command_ranges["omega"]),
            ], dtype=np.float64)
            if self.command_mode == "axis":
                axis = self.np_random.choice([-1, 0, 1, 2], p=[.1, .4, .25, .25])
                self._vel_cmd[np.arange(3) != axis] = 0.

        self._step_count = 0
        self._last_action = np.zeros(12, dtype=np.float64)
        return self._get_obs().astype(np.float32), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._apply_pd_control(action)

        # 物理刻みから50 Hzの制御周期を計算する。
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps

        if terminated:
            reward -= self.reward_config.fall_penalty

        self._last_action = action.copy()
        obs = self._get_obs().astype(np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    # ── 観測 ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        ang_vel = self.data.qvel[3:6].copy()
        proj_grav = self._projected_gravity()
        # アクチュエータ順で取得してアクション空間と対応させる
        jpos = self.data.qpos[self._act_qposadr] - STANDING_POS
        jvel = self.data.qvel[self._act_dofadr]
        return np.concatenate([
            self._vel_cmd,
            ang_vel,
            proj_grav,
            jpos,
            jvel,
            self._last_action,
        ])

    def _projected_gravity(self) -> np.ndarray:
        """重力ベクトル [0,0,-1] をボディフレームに回転。"""
        return self._world_to_body(np.array([0.0, 0.0, -1.0]))

    def _world_to_body(self, vector: np.ndarray) -> np.ndarray:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return rotation.reshape(3, 3).T @ vector

    # ── 制御 ─────────────────────────────────────────────────────────────

    def _apply_pd_control(self, action: np.ndarray) -> None:
        # 位置目標をアクチュエータへ渡す。保存済みのゲイン・力の上限を
        # 設定した MuJoCo モデルが位置制御を行う（既定値 kp=20, kd=0.5）。
        q_target = STANDING_POS + action * self.action_scale
        limits = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(q_target, limits[:, 0], limits[:, 1])

    # ── 報酬 ─────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        cfg = self.reward_config

        # 1. 線速度追跡（Gaussian）— body フレームに変換してから比較
        # qvel[:3] は MuJoCo free joint のワールドフレーム線速度
        lin_vel_body = self._world_to_body(self.data.qvel[:3])[:2]
        lin_err = np.sum((self._vel_cmd[:2] - lin_vel_body) ** 2)
        r_lin = cfg.lin_vel_weight * float(np.exp(-lin_err / cfg.linear_tracking_variance))
        r_lin += cfg.linear_error_weight * float(lin_err)

        # 2. 角速度追跡（Gaussian）
        ang_vel_z = self.data.qvel[5]
        ang_err = (self._vel_cmd[2] - ang_vel_z) ** 2
        r_ang = cfg.ang_vel_weight * float(np.exp(-ang_err / cfg.angular_tracking_variance))
        r_ang += cfg.angular_error_weight * float(ang_err)

        # 3. 姿勢ペナルティ（重力方向の xy 傾き）
        proj_grav = self._projected_gravity()
        r_orient = cfg.orientation_weight * float(np.sum(proj_grav[:2] ** 2))

        # 4. トルクペナルティ（実際のアクチュエータ力を使用）
        # qfrc_actuator は DOF 順の一般化力。足関節 DOF は freejoint(6) の直後の 12 DOF ではなく
        # _act_dofadr で指定されたアドレスにある。
        actual_tau = self.data.qfrc_actuator[self._act_dofadr]
        r_torque = cfg.torques_weight * float(np.sum(actual_tau ** 2))

        # 5. アクション変化ペナルティ
        r_rate = cfg.action_rate_weight * float(np.sum((action - self._last_action) ** 2))
        velocity_weight = cfg.joint_velocity_weight
        if np.linalg.norm(self._vel_cmd) < .05:
            velocity_weight += cfg.stand_joint_velocity_weight
        r_rate += velocity_weight * float(np.sum(self.data.qvel[self._act_dofadr] ** 2))

        # 6. 足スリップペナルティ
        r_slip = self._foot_slip_penalty(cfg)

        # 7. トロット歩行リズム報酬（対角足が交互に着地）
        r_gait = self._feet_gait_reward(cfg)
        if cfg.diagonal_support_weight and np.linalg.norm(self._vel_cmd) >= .1:
            contacts = [any(c.geom1 == gid or c.geom2 == gid
                           for c in self.data.contact[:self.data.ncon])
                        for gid in self._foot_geom_ids]
            if len(contacts) == 4:
                fr, fl, rr, rl = contacts
                r_gait += cfg.diagonal_support_weight * float(
                    (fr and rl and not fl and not rr) or
                    (fl and rr and not fr and not rl))

        r_height = cfg.base_height_weight * float((self.data.qpos[2] - cfg.base_height_target) ** 2)
        r_vertical = cfg.vertical_velocity_weight * float(self.data.qvel[2] ** 2)
        r_posture = 0.
        if cfg.nonfoot_contact_weight or cfg.knee_height_weight:
            posture = self.posture_metrics()
            r_posture = cfg.nonfoot_contact_weight * posture['nonfoot_contacts']
            r_posture += cfg.knee_height_weight * float(np.sum(
                np.maximum(cfg.knee_height_target - posture['knee_heights'], 0.) ** 2))
        return r_lin + r_ang + r_orient + r_torque + r_rate + r_slip + r_gait + r_height + r_vertical + r_posture

    def _feet_gait_reward(self, cfg: WalkRewardConfig) -> float:
        """トロット歩行リズム報酬。

        対角足ペア (FR+RL, FL+RR) が交互に着地するリズムを参照波と比較して報酬を与える。
        速度コマンドがほぼゼロの場合は全足着地が正解なのでスキップ。
        """
        if cfg.feet_gait_weight == 0 or len(self._foot_geom_ids) < 4:
            return 0.0

        # 速度コマンドが小さい場合はスタンド静止が正解 → gait 報酬をスキップ
        cmd_speed = float(np.linalg.norm(self._vel_cmd))
        if cmd_speed < 0.1:
            return 0.0

        freq = 1.5  # トロット周波数 Hz
        t = self.data.time
        phase = 2.0 * np.pi * freq * t

        # 参照接触確率 [0,1]: FR と RL は同位相、FL と RR は逆位相
        ref_fr_rl = 0.5 * (1.0 + np.sin(phase))   # FR, RL
        ref_fl_rr = 0.5 * (1.0 - np.sin(phase))   # FL, RR

        # _foot_geom_ids の順: FR=0, FL=1, RR=2, RL=3
        refs = [ref_fr_rl, ref_fl_rr, ref_fl_rr, ref_fr_rl]

        reward = 0.0
        for geom_id, ref in zip(self._foot_geom_ids, refs):
            in_contact = float(any(
                c.geom1 == geom_id or c.geom2 == geom_id
                for c in self.data.contact[:self.data.ncon]
            ))
            reward += ref * in_contact + (1.0 - ref) * (1.0 - in_contact)

        return cfg.feet_gait_weight * reward / 4.0

    def _foot_slip_penalty(self, cfg: WalkRewardConfig) -> float:
        if not self._foot_geom_ids or cfg.foot_slip_weight == 0:
            return 0.0
        penalty = 0.0
        for geom_id in self._foot_geom_ids:
            # 接触チェック
            in_contact = any(
                c.geom1 == geom_id or c.geom2 == geom_id
                for c in self.data.contact[:self.data.ncon]
            )
            if in_contact:
                # 足の水平速度
                foot_vel = np.zeros(6)
                mujoco.mj_objectVelocity(
                    self.model, self.data,
                    mujoco.mjtObj.mjOBJ_GEOM, geom_id, foot_vel, 0
                )
                slip = float(np.sum(foot_vel[3:5] ** 2))
                penalty += slip
        return cfg.foot_slip_weight * penalty

    # ── 終了判定 ──────────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        return bool(self.data.qpos[2] < 0.15)  # 胴体高さ < 0.15m = 転倒

    # ── プロパティ ────────────────────────────────────────────────────────

    @property
    def robot_xy(self) -> np.ndarray:
        return self.data.qpos[:2].copy()

    @property
    def robot_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def vel_cmd(self) -> np.ndarray:
        return self._vel_cmd.copy()

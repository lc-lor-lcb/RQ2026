"""
報酬関数の設定クラス

WalkRewardConfig : 歩行事前学習用（unitree_rl_mjlab 参考）
FleeRewardConfig : 階層学習の高レベルポリシー（鬼ごっこ）用
"""
from dataclasses import dataclass


@dataclass
class WalkRewardConfig:
    """Go2 歩行の報酬重み（unitree_rl_mjlab の velocity task 準拠）。

    高校生向けに調整可能なパラメータを選別:
    - 速度追跡の重みを上げると、より速く歩くことを優先
    - 姿勢の重みを上げると、安定した歩行を優先
    """
    # ── 速度追跡（主目標） ─────────────────────────────────────────────
    # コマンド速度への追従（Gaussian 報酬: exp(-error/std)）
    linear_tracking_variance: float = .25
    angular_tracking_variance: float = .5
    lin_vel_weight: float = 1.0    # 線速度追跡の重み
    ang_vel_weight: float = 0.5    # 角速度追跡の重み
    # Gaussian報酬が遠方でほぼ0になる時の補助。負値で誤差二乗ペナルティ。
    linear_error_weight: float = 0.0
    angular_error_weight: float = 0.0

    # ── 安定性ペナルティ ───────────────────────────────────────────────
    # 重力ベクトルの傾き（xy 成分の二乗和）
    orientation_weight: float = -1.0
    base_height_target: float = .26
    base_height_weight: float = 0.0
    vertical_velocity_weight: float = 0.0
    # 足先以外で床を支えない。膝の高さは床から関節中心まで。
    nonfoot_contact_weight: float = 0.0
    knee_height_target: float = .10
    knee_height_weight: float = 0.0

    # ── エネルギー効率ペナルティ ───────────────────────────────────────
    # トルクの二乗和（省エネ）
    torques_weight: float = -2.5e-5
    # アクション変化の二乗和（滑らかな動き）
    action_rate_weight: float = -0.05
    # 実際の関節速度を抑え、細かい高速な足踏みを減らす。
    joint_velocity_weight: float = 0.0
    stand_joint_velocity_weight: float = 0.0

    # ── 歩行品質報酬 ───────────────────────────────────────────────────
    # トロット歩行リズム（対角足交互着地）: 大きいほど歩行らしさを重視
    feet_gait_weight: float = 0.5
    # 時刻を使わず、対角2脚の支持を促す（比較実験用）。
    diagonal_support_weight: float = 0.0
    # 接触中の足水平速度（スリップ防止）
    foot_slip_weight: float = -0.1

    # ── 終了ペナルティ ─────────────────────────────────────────────────
    fall_penalty: float = 10.0


@dataclass
class FleeRewardConfig:
    """鬼ごっこ（高レベルポリシー）用報酬。

    クイックスタート編 のスライダーパラメータがこのクラスに対応。
    """
    # 毎ステップの生存ボーナス（生き延びることへの報酬）
    survival_weight: float = 0.5

    # 鬼との距離に比例した逃げ報酬
    distance_weight: float = 1.0

    # タグされた（鬼に捕まった）時のペナルティ
    tag_penalty: float = 50.0

    # 転倒時のペナルティ
    fall_penalty: float = 20.0


# ── プリセット ──────────────────────────────────────────────────────────────

def make_fast_walker() -> WalkRewardConfig:
    """速さ優先（速度追跡を強化）"""
    return WalkRewardConfig(lin_vel_weight=2.0, ang_vel_weight=0.5)


def make_stable_walker() -> WalkRewardConfig:
    """安定優先（姿勢ペナルティを強化）"""
    return WalkRewardConfig(orientation_weight=-2.0, torques_weight=-5e-5)


def make_aggressive_flee() -> FleeRewardConfig:
    """距離優先（リスクを顧みずに距離を稼ぐ）"""
    return FleeRewardConfig(distance_weight=3.0, survival_weight=0.2)


def make_survival_flee() -> FleeRewardConfig:
    """生存優先（じっくり逃げ続ける）"""
    return FleeRewardConfig(survival_weight=2.0, distance_weight=0.5)

"""
Generation-14 fixed-heading omnidirectional walk evolution (heading-stable).

Gen13 established solid 360-degree translation accuracy (mean_direction_error
~3.07 deg / max ~54.4 deg once the 6 turn-only eval commands are excluded,
0 falls) but investigation of the Gen13 run (walk_gen13_fixed_heading_002)
surfaced two separate problems:

1. EVAL COMMAND SET MISMATCH (cosmetic, not a training bug):
   `build_commands()` (imported unchanged from train_walk_gen11.py) still
   contains 6 commands that require a nonzero omega:
   turn_left_sprint, turn_right_sprint, left_turn, right_turn,
   back_left_turn, back_right_turn. Gen13/Gen14 never train with a nonzero
   omega command (command_ranges["omega"] is pinned to [0.0, 0.0]
   throughout every stage), so feeding these 6 commands directly via
   set_vel_cmd() at eval/video time puts the policy on input it has never
   seen in training (a nonzero commanded omega in the observation), and it
   responds by essentially freezing in place. This produced the "robot just
   stands still" videos and inflated max_direction_error_deg (89.6 deg, from
   `left_turn`) in walk_gen13's evaluation summary. Fix: evaluate and
   record video only for the omega == 0 subset of build_commands()
   (see build_commands_fixed_heading() below).

2. ABSOLUTE HEADING DRIFT (a real training-reward gap):
   Nothing in the reward stack constrains the robot's *absolute* heading
   over an episode:
     - Go2WalkEnv._compute_reward()'s r_ang term tracks *instantaneous*
       angular velocity (qvel[5]) against the commanded omega (always 0.0
       here) via a Gaussian bonus (ang_vel_weight=0.5,
       angular_tracking_variance=0.5) plus a squared-error term whose
       weight (angular_error_weight) defaults to 0.0 in WalkRewardConfig
       and is never overridden by Gen13. With that weight at 0.0, the only
       pressure against yaw rate is the Gaussian bonus, which is nearly
       flat near zero error - so a small, persistent yaw-rate bias costs
       the policy almost nothing per step, yet integrates into a large
       heading change over a 12-60 second episode.
     - r_orient (gravity projected into the body xy-plane) penalizes
       roll/pitch tilt only; a robot that stays upright while yawing
       produces ~zero projected-gravity error regardless of how much it
       has turned.
     - AngleAwareGo2WalkEnv's direction_error_weight term compares
       body-frame actual velocity against body-frame commanded velocity
       every step, so it is *structurally blind* to absolute heading
       drift: as long as the legs walk "forward" relative to whatever way
       the body currently happens to be facing, this term reports ~0
       error even while the body slowly spins in the world frame.
   This is consistent with what was seen in the Gen13 videos (heading
   visibly different at the end of a clip vs. the start) despite good
   walk_summary numbers.

   Fix: FixedHeadingRandomAngleGo2WalkEnv now tracks the body yaw at
   episode reset and adds an explicit heading_drift_weight term that
   penalizes squared deviation from that reference yaw every step,
   independent of the existing rate-based r_ang term.

Everything else (physics, PPO overrides, curriculum shape, omega pinned to
[0.0, 0.0]) is unchanged from Gen13.

Retraining note: --source below defaults to Gen13's own best checkpoint
(warm start - no need to re-learn omnidirectional translation from
scratch), but the full GEN14_PROFILES curriculum is run again starting
from stage 1's own ent_coef/learning_rate. Because each stage already
force-applies its own learning_rate/ent_coef to the reloaded model before
.learn() (the "Fix 1" behavior carried over from Gen13), this gives the
policy a fresh, non-annealed exploration budget to actually unlearn
whatever small yaw-drift habit it settled into under Gen13's reward,
rather than continuing training under Gen13's already very low tail-stage
ent_coef (0.0015), where there may not be enough exploration left to
change gait habits.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm

from scripts import train_speed_walk_base as base
from scripts.train_walk_gen11 import AngleAwareGo2WalkEnv, build_commands, eval_angle_walk


# -----------------------------------------------------------------------------
# Gen14 command set (translation-only subset of Gen11's build_commands())
# -----------------------------------------------------------------------------
def build_commands_fixed_heading() -> dict:
    """Gen13/Gen14はomegaを一切学習していないため、build_commands()が持つ
    旋回系6コマンド (turn_left_sprint, turn_right_sprint, left_turn,
    right_turn, back_left_turn, back_right_turn) を評価・動画対象から除外
    する。これらはomega != 0を要求するが、command_ranges["omega"]は常に
    [0.0, 0.0]で学習しているため、評価時にこれらを直接set_vel_cmd()して
    しまうと方策が学習中に一度も見たことのない入力を受け取ることになり、
    停止に近い挙動になる（Gen13で実際に観測された）。
    """
    return {
        name: cmd
        for name, cmd in build_commands().items()
        if abs(cmd[2]) < 1e-9  # omega成分がゼロのコマンドのみ残す
    }


# -----------------------------------------------------------------------------
# Gen14 environment
# -----------------------------------------------------------------------------
class FixedHeadingRandomAngleGo2WalkEnv(AngleAwareGo2WalkEnv):
    """Gen13の固定方位・全方位並進環境に、絶対ヨードリフトへの直接ペナルティ
    を追加したもの。

    AngleAwareGo2WalkEnv._sample_angle_command()は常にomegaを
    self.command_ranges["omega"]から一様サンプリングしてself.set_vel_cmd()
    に渡す。command_ranges={"omega": [0.0, 0.0]}で構築する限りomegaは常に
    0.0になる（Gen13から変更なし）。

    Gen13にはなかった追加要素：
    - reset()時点のヨー角を self._reference_yaw に保存
    - 毎ステップ、現在のヨー角がその基準からどれだけずれているかを
      (ラップした角度)^2 * heading_drift_weight としてrewardに加算
    """

    def __init__(
        self,
        *args,
        command_switch_steps=150,
        heading_drift_weight=-1.0,
        **kwargs,
    ):
        self.command_switch_steps = max(1, int(command_switch_steps))
        self._command_step_counter = 0
        self.heading_drift_weight = float(heading_drift_weight)
        self._reference_yaw = 0.0
        super().__init__(*args, **kwargs)

        omega_lo, omega_hi = self.command_ranges["omega"]
        if omega_lo != 0.0 or omega_hi != 0.0:
            raise ValueError(
                "FixedHeadingRandomAngleGo2WalkEnv requires "
                "command_ranges['omega'] == [0.0, 0.0] so that "
                "AngleAwareGo2WalkEnv._sample_angle_command() always draws "
                f"omega=0.0; got {list(self.command_ranges['omega'])!r} "
                "instead. Fixed-heading training depends entirely on this "
                "range, since there is no separate omega attribute forced "
                "to zero elsewhere."
            )

    # --------------------------------------------------------
    # Absolute yaw tracking
    # --------------------------------------------------------

    def _current_yaw(self) -> float:
        """MuJoCoのfree jointは qpos[3:7] = [w, x, y, z]（MuJoCo規約）。
        Z-upワールドでのヨー角（ロール・ピッチに依存しない）を返す。
        """
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._command_step_counter = 0
        # エピソード開始時点の向きを、このエピソード全体の基準方位とする。
        # コマンド切り替え(_sample_angle_command)のたびには更新しない:
        # 「毎回指示方向へ正確に並進できるか」とは独立に、「エピソードを
        # 通して体が勝手に回転し続けていないか」を見たいため。
        self._reference_yaw = self._current_yaw()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._command_step_counter += 1

        if (
            self.randomize_cmd
            and not terminated
            and not truncated
            and self._command_step_counter >= self.command_switch_steps
        ):
            self._sample_angle_command()
            self._command_step_counter = 0
            obs = self._get_obs().astype(np.float32)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray) -> float:
        # AngleAwareGo2WalkEnv._compute_reward()（並進方向誤差ペナルティを
        # 含む）→ Go2WalkEnv._compute_reward()（基礎報酬一式）の順で呼ばれる。
        reward = super()._compute_reward(action)

        yaw_drift = self._wrap_angle_rad(
            self._current_yaw() - self._reference_yaw
        )
        reward += self.heading_drift_weight * (yaw_drift ** 2)

        return reward


# -----------------------------------------------------------------------------
# Curriculum
# -----------------------------------------------------------------------------
# steps/speed/angle/direction_error_weight/switch_steps/ent_coef/learning_rate
# はGen13からそのまま引き継ぎ（並進精度は既に良好だったため）。
# heading_drift_weightのみ新規追加。値は暫定の出発点であり、最初の短い
# テストステージの結果を見て調整する前提。
GEN14_PROFILES = {
    "fixed_heading": [
        dict(
            name="fixed_heading_foundation",
            steps=800_000,
            speed=[0.45, 1.20],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-1.5,
            heading_drift_weight=-1.0,
            switch_steps=250,
            ent_coef=0.0025,
            learning_rate=1.00e-5,
        ),
        dict(
            name="fixed_heading_360",
            steps=1_100_000,
            speed=[0.60, 1.35],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-2.0,
            heading_drift_weight=-1.5,
            switch_steps=220,
            ent_coef=0.0023,
            learning_rate=9.5e-6,
        ),
        dict(
            name="fixed_heading_high_speed",
            steps=1_200_000,
            speed=[0.85, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-2.5,
            heading_drift_weight=-2.0,
            switch_steps=180,
            ent_coef=0.0021,
            learning_rate=9.0e-6,
        ),
        dict(
            name="fixed_heading_precision",
            steps=1_200_000,
            speed=[0.45, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-2.5,
            switch_steps=150,
            ent_coef=0.0019,
            learning_rate=8.8e-6,
        ),
        dict(
            name="fixed_heading_fast_switch",
            steps=1_200_000,
            speed=[0.40, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.2,
            heading_drift_weight=-3.0,
            switch_steps=120,
            ent_coef=0.0017,
            learning_rate=8.5e-6,
        ),
        dict(
            name="fixed_heading_full_range",
            steps=1_300_000,
            speed=[0.30, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.2,
            heading_drift_weight=-3.0,
            switch_steps=100,
            ent_coef=0.0016,
            learning_rate=8.2e-6,
        ),
        dict(
            name="final_fixed_heading",
            steps=1_500_000,
            speed=[0.30, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-3.5,
            switch_steps=80,
            ent_coef=0.0015,
            learning_rate=8.0e-6,
        ),
    ]
}


GEN14_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)
GEN14_PPO_OVERRIDES.update({
    "target_kl": 0.007,
    "n_epochs": 5,
})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generation-14 fixed-heading, heading-stable omnidirectional walk evolution."
    )
    parser.add_argument(
        "--source",
        # Gen13の最終チェックポイントからウォームスタートする。
        # 並進精度はGen13で既に良好 (mean_direction_error ~3.07 deg,
        # fall 0件, turn系コマンド除外後) だったため、ゼロから学習し直す
        # 必要はない。カリキュラムは各ステージが自前のent_coef/
        # learning_rateを毎回上書きするため、この重み継続でも
        # ステージ1から探索の余地は確保される。
        default="runs/walk_base/walk_gen13_fixed_heading_002/best",
        help="Gen13 best model directory.",
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(GEN14_PROFILES),
        default="fixed_heading",
    )
    parser.add_argument("--num-envs", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--record-videos", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source model was not found: {source}")

    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen14"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    stages = GEN14_PROFILES[args.profile]

    common_physics = {
        "action_scale": 0.6,
        "joint_stiffness": 60.0,
        "joint_damping": 2.0,
        "torque_limits": [23.7, 23.7, 45.43],
    }

    standard_env_kwargs = {
        **common_physics,
        "command_mode": "uniform",
        "command_ranges": {
            "vx": [-1.4, 1.4],
            "vy": [-1.2, 1.2],
            "omega": [0.0, 0.0],
        },
    }

    history = []

    for stage_index, stage in enumerate(
        tqdm(stages, desc="gen14 stages", unit="stage")
    ):
        print(f"\n=== Gen14 stage {stage_index + 1}/{len(stages)}: {stage['name']} ===", flush=True)
        print(f"steps={stage['steps']}", flush=True)
        print(f"speed={stage['speed']}", flush=True)
        print(f"angle={stage['angle']}", flush=True)
        print(f"omega={stage['omega']}", flush=True)
        print(f"direction_error_weight={stage['direction_error_weight']}", flush=True)
        print(f"heading_drift_weight={stage['heading_drift_weight']}", flush=True)
        print(f"random_angle_switch_steps={stage['switch_steps']}", flush=True)

        def make_env():
            return FixedHeadingRandomAngleGo2WalkEnv(
                randomize_cmd=True,
                angle_ranges=stage["angle"],
                speed_ranges=stage["speed"],
                direction_error_weight=stage["direction_error_weight"],
                direction_error_speed_threshold=0.15,
                actual_speed_threshold=0.05,
                heading_drift_weight=stage["heading_drift_weight"],
                command_switch_steps=stage["switch_steps"],
                **common_physics,
                command_ranges={
                    "vx": [-1.4, 1.4],
                    "vy": [-1.2, 1.2],
                    "omega": [0.0, 0.0],
                },
            )

        # IMPORTANT: explicitly keep the fast Gen11/SubprocVecEnv path.
        vec_raw = make_vec_env(
            make_env,
            n_envs=args.num_envs,
            seed=1000 + stage_index,
            vec_env_cls=SubprocVecEnv,
        )

        if stage_index == 0:
            vec_env = VecNormalize(
                vec_raw,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )
        else:
            vec_env = VecNormalize.load(
                str(run_dir / "walk_model_vecnorm.pkl"),
                vec_raw,
            )
            vec_env.training = True
            vec_env.norm_reward = True

        model = PPO.load(
            run_dir / "walk_model",
            env=vec_env,
            device=args.device,
        )

        # Fix 1 (carried over from Gen13): actually apply this stage's
        # learning_rate and ent_coef to the reloaded model before training
        # on it, rather than relying on whatever was saved with the
        # checkpoint (which would otherwise silently keep Gen13's final,
        # heavily-annealed tail-stage values).
        model.learning_rate = stage["learning_rate"]
        model.lr_schedule = get_schedule_fn(stage["learning_rate"])
        model.ent_coef = stage["ent_coef"]

        ppo_kwargs = dict(
            GEN14_PPO_OVERRIDES,
            device=args.device,
            learning_rate=stage["learning_rate"],
            ent_coef=stage["ent_coef"],
        )

        checkpoint_dir = run_dir / "checkpoints" / stage["name"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointCallback(
            save_freq=max(20_000 // max(args.num_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="walk_gen14",
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps=int(stage["steps"]),
            callback=checkpoint,
            progress_bar=True,
            reset_num_timesteps=False,
        )

        model.save(run_dir / "walk_model")
        vec_env.save(str(run_dir / "walk_model_vecnorm.pkl"))

        params = {
            "kind": "walk",
            "generation": 14,
            "source": str(source),
            "stage": stage,
            "physics": common_physics,
            "walk_env_kwargs": standard_env_kwargs,
            "fixed_heading_training": {
                "enabled": True,
                "commanded_omega_fixed_zero": True,
                "body_turning_is_not_a_training_target": True,
                "speed_range": stage["speed"],
                "angle_range_deg": stage["angle"],
                "omega_range": stage["omega"],
                "command_switch_steps": stage["switch_steps"],
                "continuous_uniform_angle": True,
                "translation_only": True,
                "direction_error_weight": stage["direction_error_weight"],
                "heading_drift_weight": stage["heading_drift_weight"],
                "heading_drift_formula": (
                    "heading_drift_weight * "
                    "wrap(current_yaw - reference_yaw_at_episode_reset)^2"
                ),
                "eval_commands": "build_commands_fixed_heading() (omega==0 subset only)",
            },
            "ppo": ppo_kwargs,
            "vec_env_cls": "SubprocVecEnv",
            "num_envs": args.num_envs,
        }
        (run_dir / "walk_params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        history.append({"stage": stage, "params": params})
        (run_dir / "gen14_curriculum.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        vec_env.close()

    # Gen11-compatible deterministic angle evaluation, but restricted to the
    # omega==0 subset of commands (see build_commands_fixed_heading() above)
    # so turn-only commands never get fed to a model that never trained on
    # a nonzero omega.
    commands = build_commands_fixed_heading()
    walk_rows, walk_summary = eval_angle_walk(
        run_dir,
        commands,
        seeds,
        args.seconds,
        standard_env_kwargs,
    )
    tag_rows, tag_summary = base.eval_tag_direct(
        run_dir,
        commands,
        seeds,
        args.seconds,
    )

    result = {
        "generation": 14,
        "goal": "accurate 360-degree translation with fixed heading, zero commanded yaw, and bounded absolute heading drift",
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }
    (run_dir / "gen14_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in base.REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_gen14_record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.record_videos:
        video_dir = run_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for label, command in commands.items():
            if label == "stand":
                continue
            base.record_walk(
                run_dir,
                video_dir / f"{label}.mp4",
                command=command,
                seconds=min(args.seconds, 12),
                seed=100,
            )

    print(json.dumps({
        "run_dir": str(run_dir),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

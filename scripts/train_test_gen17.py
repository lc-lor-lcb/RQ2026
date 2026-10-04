"""
Generation-17 TEST: reuse the always-zero omega command slot for heading error.

train_test_gen15/16.pyでの検証結果まとめ：
  - heading_drift_weightを強めても(Gen15①)、target_kl/ent_coefを緩めても
    (Gen15②)、angular_error_weightを有効化しても(Gen16、低〜中LR)、絶対
    ヨードリフトはBefore(27.70°)からほとんど動かなかった(27.7〜30.6°)。
  - learning_rateを大きく上げると数字は動いたが、悪化する方向だった
    (highlr: 35.0°、瞬間的にほぼ180°反転するケースも発生)。
  - 学習量を3倍(105万step)にしても改善しなかった(30.6°)。
  - go2_walk_env.pyの観測ベクトル(45次元)には、エピソード開始からの
    累積ヨードリフトを示す情報が一切なく、方策はフィードフォワードMLPで
    記憶を持たないため、閉ループでの補正を学習しようがなかった、という
    仮説がここまでの消去法でかなり強く裏付けられた。

観測次元を拡張する案(45→47)は重みサージェリーが必要でリスクが高いため
一旦保留していたが、次の点に気づいた：

  歩行カリキュラムはcommand_ranges["omega"] = [0.0, 0.0] に固定されている
  ため、観測ベクトルの vel_cmd の3番目の要素（目標omega）は、Gen11から
  今まで一度も0以外の値を取ったことがない「空きスロット」になっている。

このスクリプトでは、観測次元を45のまま変えず、このomegaスロットを
sin(heading_error)（エピソード開始からの累積ヨードリフト）に置き換える。
これにより：
  - 既存のPPO.load()/VecNormalize.load()がそのまま使える
    （テンソル形状が一切変わらないので重みサージェリー不要）
  - 方策は初めて「今どれだけ回転してしまっているか」を直接観測できる

ただし2点、注意して処理する必要がある：
  1. このスロットへの入力重みは、常に0を掛けられ続けてきたため勾配を
     一度も受け取っておらず、初期化時のランダム値のまま残っている。
     いきなり意味のある値を流すと、ランダムな重みとの積で予期しない
     出力が混ざり込み、学習開始直後の挙動が乱れる恐れがある。
     → 学習開始前に、このスロットへの入力重み列だけを明示的に0で
        上書きしてから学習を再開する（zero_out_obs_column()）。
        これで学習開始の瞬間は既存の挙動を完全に再現し、そこから
        徐々にこの新しい信号の使い方を学習させる。
  2. VecNormalizeの正規化統計（このスロットの分散）も「常に0」だった
     ため分散がほぼ0に固まっており、そのままだと新しい値を
     (x-mean)/sqrt(var+eps) で正規化する際に分散がほぼ0で割ることに
     なり数値が爆発する。
     → 学習開始前に、このスロットの mean=0.0, var=1.0 に上書きする。

どちらの処理も、このomegaスロット転用トリックを初めて適用する最初の
stageでのみ実行する（それ以降のstageや、既にこのトリックを適用済みの
モデルを--sourceに渡す場合は--skip-omega-slot-resetで無効化できる）。
"""
from __future__ import annotations

import argparse
import dataclasses
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
from scripts.train_walk_gen11 import wrap_angle_deg
from scripts.train_walk_gen14 import FixedHeadingRandomAngleGo2WalkEnv
from scripts.train_test_gen15 import (
    build_drift_test_commands,
    summarize_with_drift,
    GEN15_PPO_OVERRIDES as GEN17_PPO_OVERRIDES,
)


# -----------------------------------------------------------------------------
# 観測次元を増やさず、vel_cmdのomegaスロット(index=2)をsin(heading_error)に
# 置き換える環境。
# -----------------------------------------------------------------------------
class HeadingInOmegaSlotGo2WalkEnv(FixedHeadingRandomAngleGo2WalkEnv):
    """歩行カリキュラムでは常に0だったomegaコマンドスロットを、
    エピソード開始からの累積ヨードリフト(sin表現)に置き換えた環境。
    観測次元は45のまま変えない。
    """

    OMEGA_SLOT_INDEX = 2  # obs = [vx_cmd, vy_cmd, omega_cmd, ang_vel(3), ...]

    def _heading_error_rad(self) -> float:
        return self._wrap_angle_rad(self._current_yaw() - self._reference_yaw)

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs().astype(np.float32)
        obs[self.OMEGA_SLOT_INDEX] = math.sin(self._heading_error_rad())
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # FixedHeadingRandomAngleGo2WalkEnv.reset()はself._reference_yawを
        # 設定した"後"にobsを組み直していないため、ここで一度作り直して
        # heading_errorが正しく0から始まるようにする。
        obs = self._get_obs()
        return obs, info


def zero_out_obs_column(model: PPO, column_index: int) -> list[str]:
    """observation_spaceの生の次元(45)を直接受け取る全ての重み行列について、
    column_index列目をゼロで上書きする。

    このスロットはこれまで常に0だった（勾配を一度も受け取っていない）ため、
    重みはランダム初期化値のまま残っている。学習再開の瞬間に既存の挙動を
    完全に再現するため、まずこの列を明示的にゼロにする。
    """
    obs_dim = model.observation_space.shape[0]
    state = model.policy.state_dict()
    modified = []
    for key, tensor in state.items():
        if tensor.dim() == 2 and tensor.shape[1] == obs_dim:
            tensor[:, column_index] = 0.0
            modified.append(key)
    model.policy.load_state_dict(state)
    return modified


def prepare_omega_slot_reuse(
    vec_env: VecNormalize,
    model: PPO,
    column_index: int = HeadingInOmegaSlotGo2WalkEnv.OMEGA_SLOT_INDEX,
) -> list[str]:
    vec_env.obs_rms.mean[column_index] = 0.0
    vec_env.obs_rms.var[column_index] = 1.0
    return zero_out_obs_column(model, column_index)


# -----------------------------------------------------------------------------
# Gen15のeval_angle_walk_with_drift()と同じロジックだが、評価時の環境も
# HeadingInOmegaSlotGo2WalkEnvにする（観測の作り方を訓練時と一致させる
# 必要があるため、Gen15のもの(FixedHeadingRandomAngleGo2WalkEnv固定)は
# そのまま使えない）。
# -----------------------------------------------------------------------------
def load_drift_test_env_gen17(folder: Path, env_kwargs: dict):
    base_env = make_vec_env(
        lambda: HeadingInOmegaSlotGo2WalkEnv(
            randomize_cmd=False,
            command_switch_steps=10**9,
            heading_drift_weight=0.0,
            **env_kwargs,
        ),
        n_envs=1,
    )
    norm = VecNormalize.load(str(folder / "walk_model_vecnorm.pkl"), base_env)
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(folder / "walk_model", env=norm, device="cpu")
    return base_env, norm, model


def eval_angle_walk_with_drift_gen17(
    folder: Path,
    commands: dict,
    seeds: list[int],
    seconds: float,
    env_kwargs: dict,
):
    base_env, norm, model = load_drift_test_env_gen17(folder, env_kwargs)
    raw = base_env.envs[0].unwrapped

    rows = []
    try:
        total = len(commands) * len(seeds)
        pbar = tqdm(total=total, desc="gen17 drift eval", unit="case")

        for name, command in commands.items():
            target_vx, target_vy, target_omega = (float(c) for c in command)
            target_speed = math.sqrt(target_vx**2 + target_vy**2)
            target_angle = (
                math.degrees(math.atan2(target_vy, target_vx))
                if target_speed > 1e-6
                else 0.0
            )

            for seed in seeds:
                raw.set_vel_cmd(target_vx, target_vy, target_omega)
                obs, _ = raw.reset(seed=seed)

                reference_yaw = raw._current_yaw()
                start_xy = np.array(raw.data.qpos[0:2], dtype=float).copy()

                velocities = []
                direction_errors = []
                yaw_drifts_deg = []
                heights = []

                terminated = False
                truncated = False
                step = 0

                for step in range(round(seconds / base.CONTROL_DT)):
                    normalized_obs = norm.normalize_obs(obs)
                    action, _ = model.predict(normalized_obs, deterministic=True)
                    obs, _, terminated, truncated, _ = raw.step(action)

                    body_vel = raw._world_to_body(raw.data.qvel[:3])
                    vx = float(body_vel[0])
                    vy = float(body_vel[1])
                    omega = float(raw.data.qvel[5])
                    speed = math.sqrt(vx * vx + vy * vy)

                    if speed > 0.05 and target_speed > 0.15:
                        actual_angle = math.degrees(math.atan2(vy, vx))
                        direction_errors.append(
                            abs(wrap_angle_deg(actual_angle - target_angle))
                        )

                    yaw_drift_deg = math.degrees(
                        raw._wrap_angle_rad(raw._current_yaw() - reference_yaw)
                    )
                    yaw_drifts_deg.append(yaw_drift_deg)

                    velocities.append([vx, vy, omega])
                    heights.append(float(raw.data.qpos[2]))

                    if terminated or truncated:
                        break

                end_xy = np.array(raw.data.qpos[0:2], dtype=float)
                net_disp = end_xy - start_xy
                net_distance = float(np.linalg.norm(net_disp))

                if net_distance > 0.05 and target_speed > 0.15:
                    net_angle = math.degrees(math.atan2(net_disp[1], net_disp[0]))
                    net_direction_error_deg = abs(
                        wrap_angle_deg(net_angle - target_angle)
                    )
                else:
                    net_direction_error_deg = None

                values = np.asarray(velocities, dtype=float)
                target = np.asarray(command, dtype=float)

                rows.append({
                    "command": name,
                    "seed": seed,
                    "target_angle_deg": target_angle if target_speed > 0.01 else None,
                    "mean_direction_error_deg": (
                        float(np.mean(direction_errors)) if direction_errors else None
                    ),
                    "max_direction_error_deg": (
                        float(np.max(direction_errors)) if direction_errors else None
                    ),
                    "max_abs_yaw_drift_deg": (
                        float(np.max(np.abs(yaw_drifts_deg))) if yaw_drifts_deg else None
                    ),
                    "final_yaw_drift_deg": (
                        float(yaw_drifts_deg[-1]) if yaw_drifts_deg else None
                    ),
                    "net_direction_error_deg": net_direction_error_deg,
                    "net_distance_m": net_distance,
                    "fell": bool(terminated),
                    "survived_seconds": float((step + 1) * base.CONTROL_DT),
                    "min_height": float(np.min(heights)) if heights else None,
                    "mean_vx": float(np.mean(values[:, 0])) if len(values) else None,
                    "mean_vy": float(np.mean(values[:, 1])) if len(values) else None,
                    "mean_omega": float(np.mean(values[:, 2])) if len(values) else None,
                    "velocity_rmse": (
                        float(np.sqrt(np.mean((values - target) ** 2)))
                        if len(values)
                        else None
                    ),
                })

                pbar.update(1)

        pbar.close()
    finally:
        norm.close()

    summary = summarize_with_drift(rows)
    return rows, summary


# -----------------------------------------------------------------------------
# 短縮カリキュラム：angular_error_weightはGen16と同様に段階的に強めるが、
# 主眼はheading情報を観測できるようにしたこと自体。LRは暴れを起こさない
# ことが分かっている安全な値(7.0〜8.0e-6)に固定する。
# -----------------------------------------------------------------------------
GEN17_TEST_PROFILES = {
    "heading_in_omega_slot_test": [
        dict(
            name="omega_slot_stage1",
            steps=350_000,
            speed=[0.45, 1.30],
            angle=[-180.0, 180.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-3.5,
            angular_error_weight=-2.0,
            switch_steps=150,
            ent_coef=0.0018,
            learning_rate=8.0e-6,
        ),
        dict(
            name="omega_slot_stage2",
            steps=350_000,
            speed=[0.40, 1.40],
            angle=[-180.0, 180.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-3.5,
            angular_error_weight=-4.0,
            switch_steps=120,
            ent_coef=0.0016,
            learning_rate=7.5e-6,
        ),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generation-17 TEST: repurpose the always-zero omega command "
            "slot to carry sin(heading_error), without changing the "
            "observation dimensionality."
        )
    )
    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_gen14_main_001/best",
        help=(
            "Warm start source. 通常はomegaスロットがまだ転用されていない"
            "Gen14/Gen15/Gen16のbestディレクトリを指定する。"
        ),
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(GEN17_TEST_PROFILES),
        default="heading_in_omega_slot_test",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "学習をスキップし、--sourceのモデルに評価だけを実行する。"
            "--sourceは既にomegaスロット転用済み(Gen17で学習済み)の"
            "モデルを指定すること。"
        ),
    )
    parser.add_argument(
        "--skip-omega-slot-reset",
        action="store_true",
        help=(
            "omegaスロットの重み・VecNormalize統計のリセットをスキップする。"
            "--sourceが既にこのトリックを適用済み(Gen17で学習を継続する場合"
            "など)のときに指定する。"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--eval-speeds", default="1.0,1.4")
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--video-seconds", type=float, default=None)
    parser.add_argument("--run-tag-eval", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f"Source model was not found: {source}\n"
            "Gen14/Gen15/Gen16のbestディレクトリを --source で指定してください。"
        )

    suffix = "walk_gen17_eval_only" if args.eval_only else "walk_gen17_test"
    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    eval_speeds = tuple(float(x) for x in args.eval_speeds.replace(",", " ").split())
    stages = GEN17_TEST_PROFILES[args.profile]

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

    if args.eval_only:
        print(
            "\n=== --eval-only 指定：学習をスキップし、--sourceのモデルに "
            "そのままomegaスロット転用済み評価をかけます。===",
            flush=True,
        )
        stages = []

    for stage_index, stage in enumerate(
        tqdm(stages, desc="gen17 test stages", unit="stage")
    ):
        print(f"\n=== Gen17-TEST stage {stage_index + 1}/{len(stages)}: {stage['name']} ===", flush=True)
        print(f"steps={stage['steps']}", flush=True)
        print(f"angular_error_weight={stage['angular_error_weight']}", flush=True)
        print(f"heading_drift_weight={stage['heading_drift_weight']}", flush=True)

        stage_reward_config = dataclasses.replace(
            base.SPEED_REWARD,
            angular_error_weight=stage["angular_error_weight"],
        )

        def make_env(stage=stage, stage_reward_config=stage_reward_config):
            return HeadingInOmegaSlotGo2WalkEnv(
                randomize_cmd=True,
                angle_ranges=stage["angle"],
                speed_ranges=stage["speed"],
                direction_error_weight=stage["direction_error_weight"],
                direction_error_speed_threshold=0.15,
                actual_speed_threshold=0.05,
                heading_drift_weight=stage["heading_drift_weight"],
                command_switch_steps=stage["switch_steps"],
                reward_config=stage_reward_config,
                **common_physics,
                command_ranges={
                    "vx": [-1.4, 1.4],
                    "vy": [-1.2, 1.2],
                    "omega": [0.0, 0.0],
                },
            )

        vec_raw = make_vec_env(
            make_env,
            n_envs=args.num_envs,
            seed=4000 + stage_index,
            vec_env_cls=SubprocVecEnv,
        )

        vec_env = VecNormalize.load(str(run_dir / "walk_model_vecnorm.pkl"), vec_raw)
        vec_env.training = True
        vec_env.norm_reward = True

        model = PPO.load(run_dir / "walk_model", env=vec_env, device=args.device)

        if stage_index == 0 and not args.skip_omega_slot_reset:
            modified_keys = prepare_omega_slot_reuse(vec_env, model)
            print(
                "[omega-slot reset] zeroed input weight column for keys: "
                f"{modified_keys}",
                flush=True,
            )
            print(
                "[omega-slot reset] vec_env.obs_rms.mean[2]="
                f"{vec_env.obs_rms.mean[2]}, var[2]={vec_env.obs_rms.var[2]}",
                flush=True,
            )

        effective_lr = (
            args.learning_rate if args.learning_rate is not None else stage["learning_rate"]
        )
        model.learning_rate = effective_lr
        model.lr_schedule = get_schedule_fn(effective_lr)
        model.ent_coef = stage["ent_coef"]

        effective_target_kl = (
            args.target_kl if args.target_kl is not None else GEN17_PPO_OVERRIDES["target_kl"]
        )
        model.target_kl = effective_target_kl

        ppo_kwargs = dict(
            GEN17_PPO_OVERRIDES,
            device=args.device,
            learning_rate=effective_lr,
            ent_coef=stage["ent_coef"],
            target_kl=effective_target_kl,
        )

        checkpoint_dir = run_dir / "checkpoints" / stage["name"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointCallback(
            save_freq=max(20_000 // max(args.num_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="walk_gen17_test",
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
            "generation": "17-test",
            "source": str(source),
            "stage": stage,
            "physics": common_physics,
            "walk_env_kwargs": standard_env_kwargs,
            "omega_slot_repurposed": True,
            "purpose": (
                "Observation-side fix (no dimension change): repurpose the "
                "always-zero omega command slot to carry sin(heading_error), "
                "since reward/PPO-hyperparameter changes alone (Gen15/Gen16) "
                "showed zero-to-negative effect on absolute yaw drift."
            ),
            "ppo": ppo_kwargs,
            "vec_env_cls": "SubprocVecEnv",
            "num_envs": args.num_envs,
        }
        (run_dir / "walk_params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        history.append({"stage": stage, "params": params})
        (run_dir / "gen17_test_curriculum.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        vec_env.close()

    # -------------------------------------------------------------------
    # 評価（omegaスロット転用に対応した専用のeval関数を使う）
    # -------------------------------------------------------------------
    commands = build_drift_test_commands(speeds=eval_speeds)
    walk_rows, walk_summary = eval_angle_walk_with_drift_gen17(
        run_dir,
        commands,
        seeds,
        args.seconds,
        standard_env_kwargs,
    )

    tag_summary = None
    tag_rows = None
    if args.run_tag_eval:
        tag_rows, tag_summary = base.eval_tag_direct(
            run_dir,
            commands,
            seeds,
            args.seconds,
        )

    result = {
        "generation": "17-test",
        "goal": (
            "verify whether repurposing the always-zero omega slot to carry "
            "sin(heading_error) (observation-side fix, no dimension change) "
            "finally allows closed-loop yaw-drift correction"
        ),
        "eval_only": args.eval_only,
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }
    (run_dir / "gen17_test_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in base.REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_gen17_test_record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------
    # 動画（デフォルトON）。record_walkはHeadingInOmegaSlotGo2WalkEnvを
    # 直接使わず、Go2WalkEnv向けの汎用録画ヘルパー(base.record_walk)を
    # 使う想定なので、record_walk側がenv_kwargsからenv classを再構築する
    # 実装になっていない場合は、best/walk_params.jsonのomega_slot_
    # repurposed:trueを見て呼び出し側で分岐する必要がある点に注意。
    # -------------------------------------------------------------------
    if not args.skip_videos:
        video_dir = run_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        video_seconds = args.video_seconds or min(args.seconds, 10.0)

        for label, command in commands.items():
            if label == "stand":
                continue
            base.record_walk(
                run_dir,
                video_dir / f"{label}.mp4",
                command=command,
                seconds=video_seconds,
                seed=100,
            )

    print(json.dumps({
        "run_dir": str(run_dir),
        "eval_only": args.eval_only,
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "videos_recorded": (not args.skip_videos),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

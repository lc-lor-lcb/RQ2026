"""
Generation-15 TEST run: drift-metric verification + short training.

このスクリプトはtrain_walk_gen14.pyの分析結果を踏まえて作られた「本番学習用」
ではなく「検証用（test）」スクリプトです。目的は2つだけです。

1. 評価パイプラインの盲点を塞ぐ
   train_walk_gen11.py の eval_angle_walk（Gen11/Gen14で共通して使われてきた
   評価関数）は、
       body_vel = world_to_body(qvel)
       actual_angle = atan2(body_vel.y, body_vel.x)
       error = |actual_angle - target_angle|
   という「現在の姿勢を基準にしたbody-frame」の速度比較しかしていません。
   これは、ロボットが世界座標系でゆっくり回転（ヨードリフト）しながらも
   「その場その場で自分の正面へ真っ直ぐ歩く」だけで誤差がほぼ0になって
   しまう、という構造的な穴を持っています（train_walk_gen14.pyのdocstring
   で報酬側の同じ問題を指摘しているが、評価側にも全く同じ穴が残っている）。
   Gen14はheading_drift_weightという形で学習報酬側にドリフト抑制を入れま
   したが、それが実際に効いたかどうかを検証できる指標がevaluation側に
   一つも無いため、best_gen14_record.jsonの数字だけでは「本当にドリフトが
   直ったのか」を判定できません。

   本スクリプトはeval_angle_walk_with_drift()として、
     - エピソード開始時のヨー角を基準にした絶対ヨードリフト
       (max_abs_yaw_drift_deg / final_yaw_drift_deg)
     - エピソード開始位置から終了位置への正味移動方向と、
       コマンド角度とのズレ (net_direction_error_deg)
   を新たに記録します。あわせて、従来の walk_summary の
   "max_direction_error_deg" が実は「各エピソードの平均誤差の最大値」で
   あって「瞬間最大誤差の最大値」ではなかった集計バグも修正しています
   (summarize_with_drift() 参照)。

2. 学習量を大幅に落とし、動画で目視確認できる形にする
   Gen14のフルカリキュラムは合計830万ステップですが、このスクリプトは
   Gen14のbestモデルからウォームスタートして、2ステージ・合計35万ステップ
   程度の短い追加学習だけを行います（heading_drift_weightをGen14最終値
   より強めて、上の新指標がそれに反応するかどうかを見るのが狙い）。
   目的は収束させることではなく、
     (a) 新しいドリフト指標が実際に意味のある数字を返すか
     (b) heading_drift_weightを強めると数字が動くか
     (c) 問題が集中していた left / back_left 系コマンドの見た目が変わるか
   を素早く確認することです。評価も全方位ではなく、問題が集中していた
   left・back_left・back_right・front_right 系の角度に絞り、デフォルトで
   動画も自動生成します。

本番相当のフル学習をする場合は、ここで得られた知見をtrain_walk_gen14.py
（またはその後継）のカリキュラム・報酬重みにフィードバックしてください。
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
from scripts.train_walk_gen11 import direction_command, wrap_angle_deg
from scripts.train_walk_gen14 import (
    FixedHeadingRandomAngleGo2WalkEnv,
    GEN14_PPO_OVERRIDES as GEN15_PPO_OVERRIDES,
)


# -----------------------------------------------------------------------------
# テスト対象コマンド：全方位ではなく、Gen14の解析で問題が集中していた
# 角度（left / back_left / back_right / front_right）＋forward/backのみに絞る
# -----------------------------------------------------------------------------
DRIFT_TEST_ANGLES = [
    ("forward", 0.0),
    ("left", 90.0),
    ("back_left_112_5", 112.5),
    ("back_left_135", 135.0),
    ("back_left_157_5", 157.5),
    ("back", 180.0),
    ("back_right_202_5", 202.5),
    ("back_right_225", 225.0),
    ("back_right_247_5", 247.5),
    ("right", 270.0),
    ("front_right_315", 315.0),
]


def build_drift_test_commands(speeds: tuple[float, ...] = (1.0, 1.4)) -> dict:
    """Gen14の全方位・全速度(60コマンド超)ではなく、絶対ドリフトの検証に
    必要な角度だけに絞った軽量コマンドセット。stand込みで
    1 + len(DRIFT_TEST_ANGLES) * len(speeds) 個になる。
    """
    commands = {"stand": (0.0, 0.0, 0.0)}
    for name, angle in DRIFT_TEST_ANGLES:
        for speed in speeds:
            commands[f"angle_{name}_{speed}"] = direction_command(angle, speed)
    return commands


# -----------------------------------------------------------------------------
# ドリフト計測つき評価（eval_angle_walkの拡張版）
# -----------------------------------------------------------------------------
def load_drift_test_env(folder: Path, env_kwargs: dict):
    base_env = make_vec_env(
        lambda: FixedHeadingRandomAngleGo2WalkEnv(
            randomize_cmd=False,
            # 評価中はテスト側で固定コマンドをset_vel_cmd()するため、
            # _sample_angle_command()による自動切り替えは無効化しておく。
            command_switch_steps=10**9,
            heading_drift_weight=0.0,  # 評価では報酬値自体は使わない
            **env_kwargs,
        ),
        n_envs=1,
    )

    norm = VecNormalize.load(str(folder / "walk_model_vecnorm.pkl"), base_env)
    norm.training = False
    norm.norm_reward = False

    model = PPO.load(folder / "walk_model", env=norm, device="cpu")

    return base_env, norm, model


def eval_angle_walk_with_drift(
    folder: Path,
    commands: dict,
    seeds: list[int],
    seconds: float,
    env_kwargs: dict,
):
    base_env, norm, model = load_drift_test_env(folder, env_kwargs)
    raw = base_env.envs[0].unwrapped

    rows = []

    try:
        total = len(commands) * len(seeds)
        pbar = tqdm(total=total, desc="gen15 drift eval", unit="case")

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

                # FixedHeadingRandomAngleGo2WalkEnv._current_yaw() /
                # _wrap_angle_rad() をそのまま再利用して、
                # 「エピソード開始時からの絶対ヨードリフト」を追跡する。
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

                # 「結局、正味どちらの方向へ進んだか」。body-frameの瞬間誤差
                # とは独立に、世界座標系での実質的な移動方向のズレを見る。
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


def summarize_with_drift(rows: list[dict]) -> dict:
    """Gen11/Gen14の集計バグ（"max_direction_error_deg"が実は行ごとの
    "平均"誤差の最大値であり、瞬間最大ではなかった）を修正しつつ、
    絶対ヨードリフトと正味移動方向のズレの集計を追加する。
    """
    falls = sum(1 for r in rows if r["fell"])

    mean_errors = [
        r["mean_direction_error_deg"] for r in rows
        if r["mean_direction_error_deg"] is not None
    ]
    row_max_errors = [
        r["max_direction_error_deg"] for r in rows
        if r["max_direction_error_deg"] is not None
    ]
    max_yaw_drifts = [
        r["max_abs_yaw_drift_deg"] for r in rows
        if r["max_abs_yaw_drift_deg"] is not None
    ]
    final_yaw_drifts = [
        abs(r["final_yaw_drift_deg"]) for r in rows
        if r["final_yaw_drift_deg"] is not None
    ]
    net_errors = [
        r["net_direction_error_deg"] for r in rows
        if r["net_direction_error_deg"] is not None
    ]

    return {
        "fall_count": falls,
        # --- 従来互換（ただし呼び方を明示的に変えて誤解を防ぐ） ---
        "mean_of_row_means_direction_error_deg": (
            float(np.mean(mean_errors)) if mean_errors else None
        ),
        "max_of_row_means_direction_error_deg": (
            # Gen11/Gen14のwalk_summary["max_direction_error_deg"]と同じ計算。
            # 名前の通り「行ごとの平均の最大」であり、瞬間最大ではない。
            float(np.max(mean_errors)) if mean_errors else None
        ),
        # --- 修正版：真の瞬間最大誤差 ---
        "mean_of_row_max_direction_error_deg": (
            float(np.mean(row_max_errors)) if row_max_errors else None
        ),
        "true_worst_direction_error_deg": (
            # 全エピソード・全ステップを通じた本当の最大瞬間誤差。
            float(np.max(row_max_errors)) if row_max_errors else None
        ),
        # --- 新規：絶対ヨードリフト（body-frame指標では検出できない） ---
        "mean_max_abs_yaw_drift_deg": (
            float(np.mean(max_yaw_drifts)) if max_yaw_drifts else None
        ),
        "worst_max_abs_yaw_drift_deg": (
            float(np.max(max_yaw_drifts)) if max_yaw_drifts else None
        ),
        "mean_abs_final_yaw_drift_deg": (
            float(np.mean(final_yaw_drifts)) if final_yaw_drifts else None
        ),
        # --- 新規：正味移動方向のズレ ---
        "mean_net_direction_error_deg": (
            float(np.mean(net_errors)) if net_errors else None
        ),
        "worst_net_direction_error_deg": (
            float(np.max(net_errors)) if net_errors else None
        ),
        "min_height": (
            float(min(r["min_height"] for r in rows if r["min_height"] is not None))
            if rows
            else None
        ),
        "mean_survived_seconds": (
            float(np.mean([r["survived_seconds"] for r in rows])) if rows else None
        ),
    }


# -----------------------------------------------------------------------------
# 短縮カリキュラム（テスト用：Gen14合計830万stepに対し、2ステージ計35万step）
# -----------------------------------------------------------------------------
GEN15_TEST_PROFILES = {
    "fixed_heading_drift_test": [
        dict(
            name="drift_test_stage1",
            steps=150_000,
            speed=[0.45, 1.30],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.0,
            # Gen14最終ステージの-3.5よりさらに強め、新指標(max_abs_yaw_drift)
            # がそれに反応して縮むかどうかを見るのがこのテストの主眼。
            heading_drift_weight=-5.0,
            switch_steps=150,
            ent_coef=0.0018,
            learning_rate=8.0e-6,
        ),
        dict(
            name="drift_test_stage2",
            steps=200_000,
            speed=[0.40, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-6.0,
            switch_steps=120,
            ent_coef=0.0016,
            learning_rate=7.5e-6,
        ),
    ],
    # 1回目のテスト(fixed_heading_drift_test)は、絶対ヨードリフトの数値が
    # Before/Afterで完全に一致してしまい、方策がほぼ動いていなかった疑いが
    # 強い。GEN14_PPO_OVERRIDES由来のtarget_kl=0.007がPPOの更新を早期に
    # 打ち切っていた可能性が高いため、ent_coefを大幅に上げて探索を増やす。
    # target_klは--target-kl引数で別途緩めること（例：0.03）。
    "fixed_heading_drift_unstick_test": [
        dict(
            name="drift_unstick_stage1",
            steps=150_000,
            speed=[0.45, 1.30],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-5.0,
            switch_steps=150,
            # 前回の0.0018から一桁近く上げて、実際に方策を動かせる探索量を確保する。
            ent_coef=0.010,
            learning_rate=8.0e-6,
        ),
        dict(
            name="drift_unstick_stage2",
            steps=200_000,
            speed=[0.40, 1.40],
            angle=[-180.0, 180.0],
            omega=[0.0, 0.0],
            direction_error_weight=-3.0,
            heading_drift_weight=-6.0,
            switch_steps=120,
            ent_coef=0.008,
            learning_rate=7.5e-6,
        ),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generation-15 TEST: short warm-start run to verify the new "
            "absolute-yaw-drift eval metric, with video confirmation."
        )
    )
    parser.add_argument(
        "--source",
        # Gen14のbestディレクトリを指定してください（例：
        # runs/walk_base/<gen14のrun名>/best）。
        default="runs/walk_base/walk_gen14_best/best",
        help="Gen14 best model directory (warm start source).",
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(GEN15_TEST_PROFILES),
        default="fixed_heading_drift_test",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "学習ステージを一切スキップし、--sourceのモデルに対して "
            "eval_angle_walk_with_drift()（＋動画）だけを実行する。"
            "Gen14そのものに対する絶対ヨードリフトの「Before」の数値を "
            "取るために使う。"
        ),
    )
    # テスト用途なのでGen14のデフォルト(30)より軽めにしてある。
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    # 前回の2seedでは方向ごとの偏り（系統的な癖か単なるばらつきか）を
    # 判断できなかったため、デフォルトを5seedに増やしてある。
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument(
        "--eval-speeds",
        default="1.0,1.4",
        help="Comma-separated speeds used for build_drift_test_commands().",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Set to disable video recording (videos are ON by default for this test script).",
    )
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=None,
        help="Defaults to min(--seconds, 10).",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help=(
            "GEN15_PPO_OVERRIDES由来のtarget_kl(=0.007、Gen14終盤の微調整用"
            "の小さい値)を上書きする。前回のテストでBefore/Afterの絶対ヨー"
            "ドリフトが完全に一致してしまい、方策がほとんど更新されていな"
            "かった疑いがあるため、fixed_heading_drift_unstick_testと組み"
            "合わせて0.02〜0.03程度に緩めることを推奨。"
        ),
    )
    parser.add_argument(
        "--run-tag-eval",
        action="store_true",
        help="Also run base.eval_tag_direct (OFF by default to keep the test fast).",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f"Source model was not found: {source}\n"
            "Gen14のbestディレクトリを --source で指定してください。"
        )

    suffix = "walk_gen15_eval_only" if args.eval_only else "walk_gen15_test"
    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    eval_speeds = tuple(float(x) for x in args.eval_speeds.replace(",", " ").split())
    stages = GEN15_TEST_PROFILES[args.profile]

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
            "eval_angle_walk_with_drift() をそのまま適用します（Beforeの "
            "数値取得用）。===",
            flush=True,
        )
        stages = []

    for stage_index, stage in enumerate(
        tqdm(stages, desc="gen15 test stages", unit="stage")
    ):
        print(f"\n=== Gen15-TEST stage {stage_index + 1}/{len(stages)}: {stage['name']} ===", flush=True)
        print(f"steps={stage['steps']}", flush=True)
        print(f"heading_drift_weight={stage['heading_drift_weight']}", flush=True)
        print(f"direction_error_weight={stage['direction_error_weight']}", flush=True)

        def make_env(stage=stage):
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

        vec_raw = make_vec_env(
            make_env,
            n_envs=args.num_envs,
            seed=2000 + stage_index,
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

        model.learning_rate = stage["learning_rate"]
        model.lr_schedule = get_schedule_fn(stage["learning_rate"])
        model.ent_coef = stage["ent_coef"]

        ppo_kwargs = dict(
            GEN15_PPO_OVERRIDES,
            device=args.device,
            learning_rate=stage["learning_rate"],
            ent_coef=stage["ent_coef"],
        )

        checkpoint_dir = run_dir / "checkpoints" / stage["name"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointCallback(
            save_freq=max(20_000 // max(args.num_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="walk_gen15_test",
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
            "generation": "15-test",
            "source": str(source),
            "stage": stage,
            "physics": common_physics,
            "walk_env_kwargs": standard_env_kwargs,
            "purpose": (
                "Short verification run: confirm the new absolute-yaw-drift "
                "eval metric behaves as expected, not a full training run."
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
        (run_dir / "gen15_test_curriculum.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        vec_env.close()

    # -------------------------------------------------------------------
    # 評価（絶対ヨードリフト計測つき）
    # -------------------------------------------------------------------
    commands = build_drift_test_commands(speeds=eval_speeds)
    walk_rows, walk_summary = eval_angle_walk_with_drift(
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
        "generation": "15-test",
        "goal": (
            "verify whether the new absolute-yaw-drift / net-direction "
            "metrics detect real drift, using a short warm-start run "
            "from Gen14's best checkpoint"
        ),
        "eval_only": args.eval_only,
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }
    (run_dir / "gen15_test_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in base.REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_gen15_test_record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------
    # 動画（デフォルトON：left / back_left / back_right / front_right など
    # 問題が集中していた方向を目視確認できるようにする）
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
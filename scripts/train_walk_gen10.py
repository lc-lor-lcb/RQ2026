"""
Generation-10 walk evolution.

Goal:
- Use Gen9 as the starting point.
- Build a high-quality omnidirectional walking model.
- Improve directional precision across 360 degrees.
- Reduce the performance gap between forward, reverse,
  lateral, and diagonal movement.
- Improve diagonal consistency.
- Preserve turning while moving.
- Increase training volume over Gen9.
- Do NOT focus on escape behavior yet.
- Do NOT introduce genetic algorithms yet.

Gen10 focus:
- Directional precision
- 360-degree coverage
- More balanced movement quality
- Fine-grained direction evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts import train_speed_walk_base as base


# ============================================================
# Gen10 curriculum
# ============================================================

GEN10_PROFILES = {
    "omnidirectional_precision": [

        # ----------------------------------------------------
        # Stage 1
        # Gen9で獲得した基本歩行能力を維持
        # ----------------------------------------------------
        dict(
            name="foundation_reinforcement",
            steps=650_000,
            ranges={
                "vx": [-1.20, 1.40],
                "vy": [-1.20, 1.20],
                "omega": [-0.15, 0.15],
            },
            mode="uniform",
            ent_coef=0.0028,
            learning_rate=1.10e-5,
        ),

        # ----------------------------------------------------
        # Stage 2
        # 前後方向の高速歩行品質
        #
        # 前進1.4を維持しつつ、
        # 後退側の性能を引き上げる。
        # ----------------------------------------------------
        dict(
            name="forward_reverse_balance",
            steps=800_000,
            ranges={
                "vx": [-1.20, 1.40],
                "vy": [-0.45, 0.45],
                "omega": [-0.18, 0.18],
            },
            mode="uniform",
            ent_coef=0.0026,
            learning_rate=1.05e-5,
        ),

        # ----------------------------------------------------
        # Stage 3
        # 左右移動の高速化
        #
        # 横移動の速度不足を重点的に改善。
        # ----------------------------------------------------
        dict(
            name="lateral_speed_quality",
            steps=850_000,
            ranges={
                "vx": [-0.45, 0.45],
                "vy": [-1.20, 1.20],
                "omega": [-0.18, 0.18],
            },
            mode="uniform",
            ent_coef=0.0025,
            learning_rate=1.00e-5,
        ),

        # ----------------------------------------------------
        # Stage 4
        # 前進 + 横移動
        #
        # 斜め方向で速度が落ちる問題を重点的に改善。
        # ----------------------------------------------------
        dict(
            name="forward_diagonal_precision",
            steps=1_000_000,
            ranges={
                "vx": [0.40, 1.40],
                "vy": [-1.20, 1.20],
                "omega": [-0.25, 0.25],
            },
            mode="uniform",
            ent_coef=0.0023,
            learning_rate=9.8e-6,
        ),

        # ----------------------------------------------------
        # Stage 5
        # 後退 + 横移動
        #
        # 360°化で特に重要な領域。
        # ----------------------------------------------------
        dict(
            name="reverse_diagonal_precision",
            steps=1_000_000,
            ranges={
                "vx": [-1.20, -0.25],
                "vy": [-1.20, 1.20],
                "omega": [-0.25, 0.25],
            },
            mode="uniform",
            ent_coef=0.0023,
            learning_rate=9.8e-6,
        ),

        # ----------------------------------------------------
        # Stage 6
        # 360°全方向
        #
        # Gen9より学習量を増やす。
        # ----------------------------------------------------
        dict(
            name="full_360_precision",
            steps=1_200_000,
            ranges={
                "vx": [-1.20, 1.40],
                "vy": [-1.20, 1.20],
                "omega": [-0.30, 0.30],
            },
            mode="uniform",
            ent_coef=0.0020,
            learning_rate=9.2e-6,
        ),

        # ----------------------------------------------------
        # Stage 7
        # 移動しながらの方向変更
        #
        # まだ逃走ではなく、
        # 歩行モデルそのものの方向変更能力を鍛える。
        # ----------------------------------------------------
        dict(
            name="movement_turn_precision",
            steps=1_050_000,
            ranges={
                "vx": [-1.20, 1.40],
                "vy": [-1.20, 1.20],
                "omega": [-0.90, 0.90],
            },
            mode="uniform",
            ent_coef=0.0018,
            learning_rate=8.8e-6,
        ),

        # ----------------------------------------------------
        # Stage 8
        # 最終統合
        #
        # 全方向・全速度・旋回を統合。
        # ----------------------------------------------------
        dict(
            name="final_omnidirectional_polish",
            steps=1_300_000,
            ranges={
                "vx": [-1.20, 1.40],
                "vy": [-1.20, 1.20],
                "omega": [-0.90, 0.90],
            },
            mode="uniform",
            ent_coef=0.0015,
            learning_rate=8.2e-6,
        ),
    ],
}


# ============================================================
# PPO
# ============================================================

GEN10_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)

GEN10_PPO_OVERRIDES.update({
    "target_kl": 0.0075,
    "n_epochs": 5,
})


# ============================================================
# Direction utilities
# ============================================================

def direction_command(
    angle_deg: float,
    speed: float,
) -> tuple[float, float, float]:
    """
    Convert an angle into a 2D movement command.

    0 degrees:
        forward

    90 degrees:
        left

    180 degrees:
        reverse

    270 degrees:
        right
    """

    rad = math.radians(angle_deg)

    vx = speed * math.cos(rad)
    vy = speed * math.sin(rad)

    return (
        vx,
        vy,
        0.0,
    )


def build_direction_commands() -> dict[str, tuple[float, float, float]]:
    """
    Build fine-grained 360-degree evaluation commands.

    16 directions:
        every 22.5 degrees

    Multiple speed levels are used so that the model is not
    evaluated only at one command magnitude.
    """

    commands: dict[str, tuple[float, float, float]] = {}

    commands["stand"] = (
        0.0,
        0.0,
        0.0,
    )

    # --------------------------------------------------------
    # Cardinal directions
    # --------------------------------------------------------

    commands["forward_0.8"] = (
        0.8,
        0.0,
        0.0,
    )

    commands["forward_1.0"] = (
        1.0,
        0.0,
        0.0,
    )

    commands["forward_1.2"] = (
        1.2,
        0.0,
        0.0,
    )

    commands["forward_1.4"] = (
        1.4,
        0.0,
        0.0,
    )

    commands["reverse_0.8"] = (
        -0.8,
        0.0,
        0.0,
    )

    commands["reverse_1.0"] = (
        -1.0,
        0.0,
        0.0,
    )

    commands["lateral_left_0.8"] = (
        0.0,
        0.8,
        0.0,
    )

    commands["lateral_left_1.0"] = (
        0.0,
        1.0,
        0.0,
    )

    commands["lateral_right_0.8"] = (
        0.0,
        -0.8,
        0.0,
    )

    commands["lateral_right_1.0"] = (
        0.0,
        -1.0,
        0.0,
    )

    # --------------------------------------------------------
    # 16-direction / 22.5-degree evaluation
    # --------------------------------------------------------

    direction_names = [
        "front",
        "front_left_22_5",
        "front_left_45",
        "front_left_67_5",
        "left",
        "back_left_112_5",
        "back_left_135",
        "back_left_157_5",
        "back",
        "back_right_202_5",
        "back_right_225",
        "back_right_247_5",
        "right",
        "front_right_292_5",
        "front_right_315",
        "front_right_337_5",
    ]

    angles = [
        0.0,
        22.5,
        45.0,
        67.5,
        90.0,
        112.5,
        135.0,
        157.5,
        180.0,
        202.5,
        225.0,
        247.5,
        270.0,
        292.5,
        315.0,
        337.5,
    ]

    # --------------------------------------------------------
    # Speed 0.9
    # --------------------------------------------------------

    for name, angle in zip(
        direction_names,
        angles,
    ):
        commands[
            f"angle_{name}_0.9"
        ] = direction_command(
            angle,
            0.9,
        )

    # --------------------------------------------------------
    # Speed 1.1
    # --------------------------------------------------------

    for name, angle in zip(
        direction_names,
        angles,
    ):
        commands[
            f"angle_{name}_1.1"
        ] = direction_command(
            angle,
            1.1,
        )

    # --------------------------------------------------------
    # Speed 1.25
    #
    # 360°方向性能の本命評価。
    # --------------------------------------------------------

    for name, angle in zip(
        direction_names,
        angles,
    ):
        commands[
            f"angle_{name}_1.25"
        ] = direction_command(
            angle,
            1.25,
        )

    # --------------------------------------------------------
    # High-speed diagonal checks
    # --------------------------------------------------------

    commands["diag_front_left_fast"] = (
        1.15,
        1.15,
        0.0,
    )

    commands["diag_front_right_fast"] = (
        1.15,
        -1.15,
        0.0,
    )

    commands["diag_back_left_fast"] = (
        -1.00,
        1.00,
        0.0,
    )

    commands["diag_back_right_fast"] = (
        -1.00,
        -1.00,
        0.0,
    )

    # --------------------------------------------------------
    # Turning
    # --------------------------------------------------------

    commands["turn_in_place_left"] = (
        0.0,
        0.0,
        1.2,
    )

    commands["turn_in_place_right"] = (
        0.0,
        0.0,
        -1.2,
    )

    commands["turn_left_sprint"] = (
        1.25,
        0.0,
        1.0,
    )

    commands["turn_right_sprint"] = (
        1.25,
        0.0,
        -1.0,
    )

    # --------------------------------------------------------
    # Movement + turning
    # --------------------------------------------------------

    commands["forward_left_turn"] = (
        0.9,
        0.65,
        0.8,
    )

    commands["forward_right_turn"] = (
        0.9,
        -0.65,
        -0.8,
    )

    commands["reverse_left_turn"] = (
        -0.75,
        0.60,
        0.7,
    )

    commands["reverse_right_turn"] = (
        -0.75,
        -0.60,
        -0.7,
    )

    return commands


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generation-10 omnidirectional walk evolution."
    )

    parser.add_argument(
        "--source",
        default=(
            "runs/walk_base/"
            "walk_gen9_omnidirectional_001/best"
        ),
        help="Generation-9 walk model to evolve.",
    )

    parser.add_argument(
        "--output-root",
        default="runs/walk_base",
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    parser.add_argument(
        "--profile",
        choices=sorted(GEN10_PROFILES),
        default="omnidirectional_precision",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--seeds",
        default="100,101,102,103,104",
    )

    parser.add_argument(
        "--record-videos",
        action="store_true",
    )

    args = parser.parse_args()

    torch.set_num_threads(1)

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"Source model was not found: {source}"
        )

    # --------------------------------------------------------
    # Run directory
    # --------------------------------------------------------

    run_name = args.run_name or (
        f"{time.strftime('%Y%m%d_%H%M%S')}"
        "_walk_gen10"
    )

    run_dir = (
        Path(args.output_root)
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    base.copy_base(
        source,
        run_dir,
    )

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    seeds = [
        int(x)
        for x in args.seeds.replace(",", " ").split()
    ]

    stages = GEN10_PROFILES[args.profile]

    history = []

    reward_config = base.AGILITY_REWARD

    # ========================================================
    # Curriculum training
    # ========================================================

    for stage in tqdm(
        stages,
        desc="gen10 stages",
        unit="stage",
    ):

        env_kwargs = {
            "action_scale": 0.6,

            # Gen9と同じ物理条件
            "joint_stiffness": 60.0,
            "joint_damping": 2.0,

            "torque_limits": [
                23.7,
                23.7,
                45.43,
            ],

            "command_mode": stage["mode"],
            "command_ranges": stage["ranges"],
        }

        print(
            f"\n=== gen10 stage "
            f"{stage['name']} "
            f"steps={stage['steps']} "
            f"ranges={stage['ranges']} ===",
            flush=True,
        )

        ppo_overrides = dict(
            GEN10_PPO_OVERRIDES,
            device=args.device,
            learning_rate=stage["learning_rate"],
            ent_coef=stage["ent_coef"],
        )

        base.train_policy(
            "walk",
            run_dir,
            reward_config,

            int(stage["steps"]),

            int(args.num_envs),

            ppo_kwargs={},

            seed=0,

            resume=True,

            checkpoint_steps=max(
                20_000,
                int(stage["steps"]) // 2,
            ),

            progress_bar=True,

            walk_env_kwargs=env_kwargs,

            allow_reward_change=True,

            initial_log_std=-2.0,

            ppo_overrides=ppo_overrides,

            freeze_normalization=False,
        )

        # ----------------------------------------------------
        # Save curriculum history
        # ----------------------------------------------------

        history.append({
            "stage": stage,

            "params": json.loads(
                (
                    run_dir
                    / "walk_params.json"
                ).read_text(
                    encoding="utf-8"
                )
            ),
        })

        (
            run_dir
            / "gen10_curriculum.json"
        ).write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # Evaluation
    # ========================================================

    commands = build_direction_commands()

    print(
        f"\nGenerated {len(commands)} evaluation commands.",
        flush=True,
    )

    walk_rows, walk_summary = base.eval_walk(
        run_dir,
        commands,
        seeds,
        args.seconds,
    )

    tag_rows, tag_summary = base.eval_tag_direct(
        run_dir,
        commands,
        seeds,
        args.seconds,
    )

    # ========================================================
    # Result
    # ========================================================

    result = {
        "generation": 10,

        "goal": (
            "build a high-quality omnidirectional "
            "walking model with improved "
            "360-degree directional precision, "
            "balanced movement speed, "
            "diagonal consistency, "
            "and movement-turning quality"
        ),

        "source": str(source),

        "evaluation": {
            "direction_count": 16,
            "direction_spacing_degrees": 22.5,
            "speed_levels": [
                0.9,
                1.1,
                1.25,
            ],
        },

        "walk_summary": walk_summary,

        "tag_summary": tag_summary,

        "walk_rows": walk_rows,

        "tag_rows": tag_rows,
    }

    (
        run_dir
        / "gen10_evaluation.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Best model
    # ========================================================

    best_dir = run_dir / "best"

    best_dir.mkdir(
        exist_ok=True,
    )

    for name in base.REQUIRED:

        shutil.copy2(
            run_dir / name,
            best_dir / name,
        )

    (
        best_dir
        / "best_gen10_record.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Optional videos
    # ========================================================

    if args.record_videos:

        video_dir = run_dir / "videos"

        video_dir.mkdir(
            exist_ok=True,
        )

        for label, command in commands.items():

            if label == "stand":
                continue

            out = (
                video_dir
                / f"{label}.mp4"
            )

            base.record_walk(
                run_dir,
                out,
                command=command,
                seconds=min(
                    args.seconds,
                    12,
                ),
                seed=100,
            )

    # ========================================================
    # Console output
    # ========================================================

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),

                "walk_summary": walk_summary,

                "tag_summary": tag_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
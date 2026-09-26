"""
Generation-9 walk evolution.

Goal:
- Use Gen8 as the starting point.
- Build a stable high-quality 360-degree movement model.
- Preserve forward / reverse / lateral / turning skills.
- Improve diagonal movement consistency.
- Reduce the performance gap between forward and other directions.
- Target roughly 85-95% of forward movement speed in other directions.
- Do NOT focus on escape behavior yet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts import train_speed_walk_base as base


# ============================================================
# Gen9 curriculum
# ============================================================

GEN9_PROFILES = {
    "omnidirectional": [

        # ----------------------------------------------------
        # Stage 1
        # 基本4方向を再確認しながら安定化
        # ----------------------------------------------------
        dict(
            name="axis_reinforcement",
            steps=550_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.15, 0.15],
            },
            mode="uniform",
            ent_coef=0.0030,
            learning_rate=1.15e-5,
        ),

        # ----------------------------------------------------
        # Stage 2
        # 前進 + 横移動
        #
        # 前進速度を維持しつつ、
        # 横方向を混ぜても速度を落としすぎない。
        # ----------------------------------------------------
        dict(
            name="forward_lateral_quality",
            steps=750_000,
            ranges={
                "vx": [0.50, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.20, 0.20],
            },
            mode="uniform",
            ent_coef=0.0028,
            learning_rate=1.05e-5,
        ),

        # ----------------------------------------------------
        # Stage 3
        # 後退 + 横移動
        # ----------------------------------------------------
        dict(
            name="reverse_lateral_quality",
            steps=750_000,
            ranges={
                "vx": [-1.00, -0.30],
                "vy": [-1.00, 1.00],
                "omega": [-0.20, 0.20],
            },
            mode="uniform",
            ent_coef=0.0028,
            learning_rate=1.05e-5,
        ),

        # ----------------------------------------------------
        # Stage 4
        # 360度方向を広く学習
        #
        # Gen8よりここを重視する。
        # ----------------------------------------------------
        dict(
            name="full_360_direction",
            steps=950_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.25, 0.25],
            },
            mode="uniform",
            ent_coef=0.0025,
            learning_rate=1.00e-5,
        ),

        # ----------------------------------------------------
        # Stage 5
        # 斜め方向を重点的に安定化
        #
        # 「前進は速いが斜めになると遅い」
        # という差を縮める。
        # ----------------------------------------------------
        dict(
            name="diagonal_quality",
            steps=900_000,
            ranges={
                "vx": [-1.00, 1.35],
                "vy": [-1.00, 1.00],
                "omega": [-0.30, 0.30],
            },
            mode="uniform",
            ent_coef=0.0023,
            learning_rate=9.5e-6,
        ),

        # ----------------------------------------------------
        # Stage 6
        # 移動しながら方向を変える
        #
        # まだ逃走学習ではない。
        # 「歩行能力そのもの」の完成度を上げる。
        # ----------------------------------------------------
        dict(
            name="movement_turn_quality",
            steps=850_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.80, 0.80],
            },
            mode="uniform",
            ent_coef=0.0020,
            learning_rate=9.0e-6,
        ),

        # ----------------------------------------------------
        # Stage 7
        # 最終統合
        #
        # すべての方向・速度・旋回を混ぜる。
        # ----------------------------------------------------
        dict(
            name="final_omnidirectional_polish",
            steps=1_000_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.80, 0.80],
            },
            mode="uniform",
            ent_coef=0.0018,
            learning_rate=8.5e-6,
        ),
    ],
}


# ============================================================
# PPO
# ============================================================

GEN9_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)

GEN9_PPO_OVERRIDES.update({
    "target_kl": 0.008,
    "n_epochs": 5,
})


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_gen8_omnidirectional_001/best",
        help="Generation-8 walk model to evolve.",
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
        choices=sorted(GEN9_PROFILES),
        default="omnidirectional",
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

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"Source model was not found: {source}"
        )

    # --------------------------------------------------------
    # Run directory
    # --------------------------------------------------------

    run_name = args.run_name or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen9"
    )

    run_dir = Path(args.output_root) / run_name

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

    stages = GEN9_PROFILES[args.profile]

    history = []

    reward_config = base.AGILITY_REWARD

    # ========================================================
    # Curriculum training
    # ========================================================

    for stage in tqdm(
        stages,
        desc="gen9 stages",
        unit="stage",
    ):

        env_kwargs = {
            "action_scale": 0.6,

            # Gen8と同じ物理条件を維持
            # 物理条件を変えず、歩行能力だけを比較する。
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
            f"\n=== gen9 stage "
            f"{stage['name']} "
            f"steps={stage['steps']} "
            f"ranges={stage['ranges']} ===",
            flush=True,
        )

        ppo_overrides = dict(
            GEN9_PPO_OVERRIDES,
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
        # Curriculum history
        # ----------------------------------------------------

        history.append({
            "stage": stage,

            "params": json.loads(
                (
                    run_dir / "walk_params.json"
                ).read_text(
                    encoding="utf-8"
                )
            ),
        })

        (
            run_dir / "gen9_curriculum.json"
        ).write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # Evaluation commands
    # ========================================================

    commands = {

        # ----------------------------------------------------
        # Stand
        # ----------------------------------------------------

        "stand": (
            0.0,
            0.0,
            0.0,
        ),

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        "forward_0.8": (
            0.8,
            0.0,
            0.0,
        ),

        "forward_1.0": (
            1.0,
            0.0,
            0.0,
        ),

        "forward_1.2": (
            1.2,
            0.0,
            0.0,
        ),

        "forward_1.4": (
            1.4,
            0.0,
            0.0,
        ),

        # ----------------------------------------------------
        # Reverse
        # ----------------------------------------------------

        "reverse_0.75": (
            -0.75,
            0.0,
            0.0,
        ),

        "reverse_1.0": (
            -1.0,
            0.0,
            0.0,
        ),

        # ----------------------------------------------------
        # Lateral
        # ----------------------------------------------------

        "lateral_left_0.75": (
            0.0,
            0.75,
            0.0,
        ),

        "lateral_right_0.75": (
            0.0,
            -0.75,
            0.0,
        ),

        "lateral_left_1.0": (
            0.0,
            1.0,
            0.0,
        ),

        "lateral_right_1.0": (
            0.0,
            -1.0,
            0.0,
        ),

        # ----------------------------------------------------
        # Forward + lateral
        #
        # 斜め方向の性能を重点的に確認
        # ----------------------------------------------------

        "forward_lateral_left": (
            0.95,
            0.75,
            0.0,
        ),

        "forward_lateral_right": (
            0.95,
            -0.75,
            0.0,
        ),

        "forward_lateral_left_fast": (
            1.15,
            0.85,
            0.0,
        ),

        "forward_lateral_right_fast": (
            1.15,
            -0.85,
            0.0,
        ),

        # ----------------------------------------------------
        # Reverse + lateral
        # ----------------------------------------------------

        "reverse_lateral_left": (
            -0.75,
            0.75,
            0.0,
        ),

        "reverse_lateral_right": (
            -0.75,
            -0.75,
            0.0,
        ),

        "reverse_lateral_left_fast": (
            -0.90,
            0.85,
            0.0,
        ),

        "reverse_lateral_right_fast": (
            -0.90,
            -0.85,
            0.0,
        ),

        # ----------------------------------------------------
        # 360-degree diagonal movement
        #
        # Gen9の重要評価項目
        # ----------------------------------------------------

        "diag_front_left": (
            0.90,
            0.90,
            0.0,
        ),

        "diag_front_right": (
            0.90,
            -0.90,
            0.0,
        ),

        "diag_back_left": (
            -0.80,
            0.80,
            0.0,
        ),

        "diag_back_right": (
            -0.80,
            -0.80,
            0.0,
        ),

        # ----------------------------------------------------
        # Turning
        # ----------------------------------------------------

        "turn_in_place_left": (
            0.0,
            0.0,
            1.2,
        ),

        "turn_in_place_right": (
            0.0,
            0.0,
            -1.2,
        ),

        "turn_left_sprint": (
            1.25,
            0.0,
            1.0,
        ),

        "turn_right_sprint": (
            1.25,
            0.0,
            -1.0,
        ),

        # ----------------------------------------------------
        # Movement + turning
        # ----------------------------------------------------

        "forward_lateral_turn_left": (
            0.9,
            0.65,
            0.8,
        ),

        "forward_lateral_turn_right": (
            0.9,
            -0.65,
            -0.8,
        ),

        "reverse_lateral_turn_left": (
            -0.75,
            0.60,
            0.7,
        ),

        "reverse_lateral_turn_right": (
            -0.75,
            -0.60,
            -0.7,
        ),
    }

    # ========================================================
    # Evaluation
    # ========================================================

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
        "generation": 9,

        "goal": (
            "build a stable high-quality "
            "360-degree movement model, "
            "targeting roughly 85-95 percent "
            "of forward movement speed "
            "in non-forward directions"
        ),

        "source": str(source),

        "walk_summary": walk_summary,

        "tag_summary": tag_summary,

        "walk_rows": walk_rows,

        "tag_rows": tag_rows,
    }

    (
        run_dir / "gen9_evaluation.json"
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
        exist_ok=True
    )

    for name in base.REQUIRED:

        shutil.copy2(
            run_dir / name,
            best_dir / name,
        )

    (
        best_dir / "best_gen9_record.json"
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
            exist_ok=True
        )

        for label, command in commands.items():

            if label == "stand":
                continue

            out = (
                video_dir /
                f"{label}.mp4"
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
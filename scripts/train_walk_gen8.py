# scripts/train_walk_gen8.py

"""Generation-8 walk evolution.

Goal:
- Preserve Gen7 forward / lateral / reverse primitives.
- Make forward+lateral and reverse+lateral combinations stable.
- Train balanced 2D movement rather than introducing a new movement direction.
- Keep turning ability from the previous generations.
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


GEN8_PROFILES = {
    "direction_blend": [
        # 1. まず4方向の基礎を壊さず安定化
        dict(
            name="axis_stability",
            steps=450_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.15, 0.15],
            },
            mode="uniform",
            ent_coef=0.0035,
            learning_rate=1.2e-5,
        ),

        # 2. 前進＋横移動
        dict(
            name="forward_lateral_blend",
            steps=650_000,
            ranges={
                "vx": [0.30, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.20, 0.20],
            },
            mode="uniform",
            ent_coef=0.0030,
            learning_rate=1.1e-5,
        ),

        # 3. 後退＋横移動
        dict(
            name="reverse_lateral_blend",
            steps=700_000,
            ranges={
                "vx": [-1.00, -0.20],
                "vy": [-1.00, 1.00],
                "omega": [-0.20, 0.20],
            },
            mode="uniform",
            ent_coef=0.0030,
            learning_rate=1.1e-5,
        ),

        # 4. 前後左右を均等に組み合わせる
        dict(
            name="full_direction_blend",
            steps=800_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-0.30, 0.30],
            },
            mode="uniform",
            ent_coef=0.0025,
            learning_rate=1.0e-5,
        ),

        # 5. 実戦的な方向転換＋移動
        dict(
            name="escape_direction_polish",
            steps=650_000,
            ranges={
                "vx": [-1.00, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-1.10, 1.10],
            },
            mode="uniform",
            ent_coef=0.0020,
            learning_rate=9.0e-6,
        ),
    ],
}


GEN8_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)
GEN8_PPO_OVERRIDES.update({
    "target_kl": 0.008,
    "n_epochs": 5,
})


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_gen7_reverse_001/best",
        help="Generation-7 walk model to evolve.",
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
        choices=sorted(GEN8_PROFILES),
        default="direction_blend",
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

    run_name = args.run_name or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen8"
    )

    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base.copy_base(source, run_dir)

    seeds = [
        int(x)
        for x in args.seeds.replace(",", " ").split()
    ]

    stages = GEN8_PROFILES[args.profile]

    history = []

    reward_config = base.AGILITY_REWARD

    for stage in tqdm(
        stages,
        desc="gen8 stages",
        unit="stage",
    ):
        env_kwargs = {
            "action_scale": 0.6,
            "joint_stiffness": 60.0,
            "joint_damping": 2.0,
            "torque_limits": [23.7, 23.7, 45.43],
            "command_mode": stage["mode"],
            "command_ranges": stage["ranges"],
        }

        print(
            f"\n=== gen8 stage {stage['name']} "
            f"steps={stage['steps']} "
            f"ranges={stage['ranges']} ===",
            flush=True,
        )

        ppo_overrides = dict(
            GEN8_PPO_OVERRIDES,
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

        history.append({
            "stage": stage,
            "params": json.loads(
                (run_dir / "walk_params.json").read_text(
                    encoding="utf-8"
                )
            ),
        })

        (
            run_dir / "gen8_curriculum.json"
        ).write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Gen8では「単方向」だけでなく、
    # 前後左右の組み合わせを重点的に評価する。
    commands = {
        "stand": (0.0, 0.0, 0.0),

        # 前進
        "forward_0.8": (0.8, 0.0, 0.0),
        "forward_1.0": (1.0, 0.0, 0.0),
        "forward_1.2": (1.2, 0.0, 0.0),
        "forward_1.4": (1.4, 0.0, 0.0),

        # 後退
        "reverse": (-0.75, 0.0, 0.0),
        "reverse_fast": (-1.0, 0.0, 0.0),

        # 横移動
        "lateral_left": (0.0, 0.75, 0.0),
        "lateral_right": (0.0, -0.75, 0.0),
        "lateral_left_fast": (0.0, 1.0, 0.0),
        "lateral_right_fast": (0.0, -1.0, 0.0),

        # 前進＋横
        "forward_lateral_left": (0.9, 0.65, 0.0),
        "forward_lateral_right": (0.9, -0.65, 0.0),

        # 後退＋横
        "reverse_lateral_left": (-0.75, 0.65, 0.0),
        "reverse_lateral_right": (-0.75, -0.65, 0.0),

        # 強めの斜め移動
        "forward_lateral_left_fast": (1.1, 0.85, 0.0),
        "forward_lateral_right_fast": (1.1, -0.85, 0.0),
        "reverse_lateral_left_fast": (-0.9, 0.85, 0.0),
        "reverse_lateral_right_fast": (-0.9, -0.85, 0.0),

        # 旋回を維持
        "turn_in_place_left": (0.0, 0.0, 1.2),
        "turn_in_place_right": (0.0, 0.0, -1.2),

        "turn_left_sprint": (1.25, 0.0, 1.0),
        "turn_right_sprint": (1.25, 0.0, -1.0),

        # 移動＋旋回＋横移動
        "forward_lateral_turn_left": (0.9, 0.55, 0.8),
        "forward_lateral_turn_right": (0.9, -0.55, -0.8),

        "reverse_lateral_turn_left": (-0.75, 0.55, 0.7),
        "reverse_lateral_turn_right": (-0.75, -0.55, -0.7),
    }

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

    result = {
        "generation": 8,
        "goal": (
            "stabilize forward_reverse_lateral combinations "
            "without losing existing movement skills"
        ),
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }

    (
        run_dir / "gen8_evaluation.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)

    for name in base.REQUIRED:
        shutil.copy2(
            run_dir / name,
            best_dir / name,
        )

    (
        best_dir / "best_gen8_record.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.record_videos:
        video_dir = run_dir / "videos"
        video_dir.mkdir(exist_ok=True)

        for label, command in commands.items():
            if label == "stand":
                continue

            out = video_dir / f"{label}.mp4"

            base.record_walk(
                run_dir,
                out,
                command=command,
                seconds=min(args.seconds, 12),
                seed=100,
            )

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
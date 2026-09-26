"""Generation-7 walk evolution: acquire reliable reverse movement without
sacrificing the Gen6 lateral/forward/turn foundation.
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


# Gen6 established useful lateral movement while retaining the strong
# forward/turn foundation. Gen7 adds reverse movement as a new primitive.
#
# The important point is that reverse is introduced gradually:
#   1. Pure reverse acquisition
#   2. Faster reverse tracking
#   3. Reverse + lateral blending
#   4. Reverse + turning / escape polishing
#
# No new forward-speed target is introduced in Gen7.
GEN7_PROFILES = {
    "reverse": [
        dict(
            name="reverse_acquire",
            steps=800_000,
            ranges={
                "vx": [-0.85, -0.20],
                "vy": [0.0, 0.0],
                "omega": [0.0, 0.0],
            },
            mode="axis",
            imitation=False,
            ent_coef=0.006,
            learning_rate=1.5e-5,
        ),
        dict(
            name="reverse_fast",
            steps=700_000,
            ranges={
                "vx": [-1.10, -0.25],
                "vy": [-0.10, 0.10],
                "omega": [-0.08, 0.08],
            },
            mode="uniform",
            imitation=False,
            ent_coef=0.005,
            learning_rate=1.5e-5,
        ),
        dict(
            name="reverse_lateral_blend",
            steps=700_000,
            ranges={
                "vx": [-1.05, 0.75],
                "vy": [-0.90, 0.90],
                "omega": [-0.30, 0.30],
            },
            mode="uniform",
            imitation=False,
            ent_coef=0.003,
            learning_rate=1.0e-5,
        ),
        dict(
            name="reverse_turn_escape",
            steps=650_000,
            ranges={
                "vx": [-1.05, 1.25],
                "vy": [-0.90, 0.90],
                "omega": [-1.10, 1.10],
            },
            mode="uniform",
            imitation=False,
            ent_coef=0.0025,
            learning_rate=1.0e-5,
        ),
        dict(
            name="reverse_polish",
            steps=550_000,
            ranges={
                "vx": [-1.10, 1.40],
                "vy": [-1.00, 1.00],
                "omega": [-1.30, 1.30],
            },
            mode="uniform",
            imitation=False,
            ent_coef=0.002,
            learning_rate=1.0e-5,
        ),
    ],
}


GEN7_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)
GEN7_PPO_OVERRIDES.update({
    "target_kl": 0.008,
    "n_epochs": 5,
})


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_gen6_lateral_001/best",
        help="Generation-6 walk model to evolve.",
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(GEN7_PROFILES),
        default="reverse",
    )
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--record-videos", action="store_true")

    args = parser.parse_args()

    torch.set_num_threads(1)

    source = Path(args.source)
    run_name = (
        args.run_name
        or f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen7"
    )

    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    stages = GEN7_PROFILES[args.profile]
    history = []

    reward_config = base.AGILITY_REWARD

    for stage in tqdm(stages, desc="gen7 stages", unit="stage"):
        env_kwargs = {
            "action_scale": 0.6,
            "joint_stiffness": 60.0,
            "joint_damping": 2.0,
            "torque_limits": [23.7, 23.7, 45.43],
            "command_mode": stage["mode"],
            "command_ranges": stage["ranges"],
        }

        print(
            f"\n=== gen7 stage {stage['name']} "
            f"steps={stage['steps']} "
            f"ranges={stage['ranges']} ===",
            flush=True,
        )

        ppo_overrides = dict(
            GEN7_PPO_OVERRIDES,
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
            initial_log_std=(
                -1.9
                if stage["name"]
                in {"reverse_acquire", "reverse_fast"}
                else -2.2
            ),
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

        (run_dir / "gen7_curriculum.json").write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Keep the existing Gen6 skills in the evaluation while adding
    # dedicated reverse commands.
    commands = {
        "stand": (0.0, 0.0, 0.0),

        "forward_0.8": (0.8, 0.0, 0.0),
        "forward_1.0": (1.0, 0.0, 0.0),
        "forward_1.2": (1.2, 0.0, 0.0),
        "forward_1.4": (1.4, 0.0, 0.0),

        "lateral_left": (0.0, 0.75, 0.0),
        "lateral_right": (0.0, -0.75, 0.0),
        "lateral_left_fast": (0.0, 1.00, 0.0),
        "lateral_right_fast": (0.0, -1.00, 0.0),

        "forward_lateral_left": (0.90, 0.65, 0.0),
        "forward_lateral_right": (0.90, -0.65, 0.0),

        "reverse": (-0.75, 0.0, 0.0),
        "reverse_fast": (-1.00, 0.0, 0.0),

        "reverse_lateral_left": (-0.75, 0.60, 0.0),
        "reverse_lateral_right": (-0.75, -0.60, 0.0),

        "turn_in_place_left": (0.0, 0.0, 1.20),
        "turn_in_place_right": (0.0, 0.0, -1.20),

        "turn_left_sprint": (1.25, 0.0, 1.0),
        "turn_right_sprint": (1.25, 0.0, -1.0),

        "reverse_turn_left": (-0.75, 0.0, 0.85),
        "reverse_turn_right": (-0.75, 0.0, -0.85),
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
        "generation": 7,
        "goal": (
            "acquire_reliable_reverse_movement_without_losing_"
            "forward_lateral_turn_foundation"
        ),
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }

    (run_dir / "gen7_evaluation.json").write_text(
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

    (best_dir / "best_gen7_record.json").write_text(
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
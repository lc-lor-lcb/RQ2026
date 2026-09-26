"""Generation-4 walk evolution: add lateral escape while preserving turn and sprint ability."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts import train_speed_walk_base as base

# The goal is deliberately narrow: first teach a new escape primitive (in-place
# turning), then blend it back into the already-good sprint/turn gait.  We do not
# try to learn lateral/reverse motion in this generation.
GEN4_PROFILES = {
    "lateral_escape": [
        dict(
            name="lateral_acquire",
            steps=520_000,
            ranges={"vx": [0.0, 0.0], "vy": [-0.65, 0.65], "omega": [0.0, 0.0]},
            mode="axis",
            imitation=False,
        ),
        dict(
            name="lateral_polish",
            steps=420_000,
            ranges={"vx": [0.0, 0.25], "vy": [-0.75, 0.75], "omega": [-0.10, 0.10]},
            mode="axis",
            imitation=False,
        ),
        dict(
            name="turn_and_sprint_restore",
            steps=420_000,
            ranges={"vx": [0.55, 1.45], "vy": [-0.12, 0.12], "omega": [-1.30, 1.30]},
            mode="uniform",
            imitation=False,
        ),
        dict(
            name="lateral_turn_blend",
            steps=420_000,
            ranges={"vx": [0.0, 1.35], "vy": [-0.65, 0.65], "omega": [-1.25, 1.25]},
            mode="uniform",
            imitation=False,
        ),
        dict(
            name="escape_mix_polish",
            steps=320_000,
            ranges={"vx": [-0.10, 1.45], "vy": [-0.70, 0.70], "omega": [-1.30, 1.30]},
            mode="uniform",
            imitation=False,
        ),
    ],
}


# Slightly more conservative than the existing sprint_turn profile because the
# policy is being asked to leave its old forward-only basin of attraction.
GEN4_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)
GEN4_PPO_OVERRIDES.update({
    "learning_rate": 1e-5,
    "target_kl": 0.008,
    "n_epochs": 5,
    "ent_coef": 0.002,
})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_sprint_turn_002/best",
        help="Generation-2 walk model to evolve.",
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--profile", choices=sorted(GEN4_PROFILES), default="lateral_escape")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--record-videos", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    source = Path(args.source)
    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen4"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    stages = GEN4_PROFILES[args.profile]
    history = []

    # AGILITY_REWARD is intentionally used for every stage: it values both
    # linear and angular tracking, unlike the sprint-only reward basin.
    reward_config = base.AGILITY_REWARD

    for stage in tqdm(stages, desc="gen4 stages", unit="stage"):
        env_kwargs = {
            "action_scale": 0.6,
            "joint_stiffness": 60.0,
            "joint_damping": 2.0,
            "torque_limits": [23.7, 23.7, 45.43],
            "command_mode": stage["mode"],
            "command_ranges": stage["ranges"],
        }
        print(
            f"\n=== gen4 stage {stage['name']} steps={stage['steps']} "
            f"ranges={stage['ranges']} ===",
            flush=True,
        )

        # No forward gait teacher here.  The teacher in the original script is
        # explicitly forward-oriented, so using it would bias the new skill back
        # toward vx-dominant motion.
        base.train_policy(
            "walk",
            run_dir,
            reward_config,
            int(stage["steps"]),
            int(args.num_envs),
            ppo_kwargs={},
            seed=0,
            resume=True,
            checkpoint_steps=max(20_000, int(stage["steps"]) // 2),
            progress_bar=True,
            walk_env_kwargs=env_kwargs,
            allow_reward_change=True,
            initial_log_std=-2.2,
            ppo_overrides=dict(GEN4_PPO_OVERRIDES, device=args.device),
            freeze_normalization=False,
        )

        history.append({
            "stage": stage,
            "params": json.loads((run_dir / "walk_params.json").read_text(encoding="utf-8")),
        })
        (run_dir / "gen3_curriculum.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Evaluate the new primitive explicitly, while also checking the old skills.
    commands = {
        "stand": (0.0, 0.0, 0.0),
        "forward_0.8": (0.8, 0.0, 0.0),
        "forward_1.0": (1.0, 0.0, 0.0),
        "forward_1.2": (1.2, 0.0, 0.0),
        "forward_1.4": (1.4, 0.0, 0.0),
        "lateral_left": (0.0, 0.60, 0.0),
        "lateral_right": (0.0, -0.60, 0.0),
        "lateral_left_fast": (0.0, 0.75, 0.0),
        "lateral_right_fast": (0.0, -0.75, 0.0),
        "turn_in_place_left": (0.0, 0.0, 1.20),
        "turn_in_place_right": (0.0, 0.0, -1.20),
        "forward_lateral_left": (0.90, 0.55, 0.0),
        "forward_lateral_right": (0.90, -0.55, 0.0),
        "turn_left_sprint": (1.25, 0.0, 1.0),
        "turn_right_sprint": (1.25, 0.0, -1.0),
    }

    walk_rows, walk_summary = base.eval_walk(run_dir, commands, seeds, args.seconds)
    tag_rows, tag_summary = base.eval_tag_direct(run_dir, commands, seeds, args.seconds)
    result = {
        "generation": 4,
        "goal": "acquire_lateral_escape_without_losing_turn_or_sprint",
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }
    (run_dir / "gen4_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in base.REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_gen4_record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.record_videos:
        video_dir = run_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for label, command in commands.items():
            if label == "stand":
                continue
            out = video_dir / f"{label}.mp4"
            base.record_walk(run_dir, out, command=command, seconds=min(args.seconds, 12), seed=100)

    print(json.dumps({
        "run_dir": str(run_dir),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

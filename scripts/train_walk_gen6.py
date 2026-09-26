"""Generation-5 walk evolution: make lateral slide a real escape primitive."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts import train_speed_walk_base as base


# Gen4 proved that the policy can keep its sprint/turn foundation, but its
# dedicated lateral commands stayed close to zero actual vy.  Gen5 therefore
# spends much more of the curriculum on pure lateral tracking before blending
# the new skill back into the existing fast gait.
GEN5_PROFILES = {
    "lateral_slide": [
        dict(
            name="slide_acquire",
            steps=800_000,
            ranges={"vx": [0.0, 0.0], "vy": [-0.95, 0.95], "omega": [0.0, 0.0]},
            mode="axis",
            imitation=False,
            ent_coef=0.006,
            learning_rate=1.5e-5,
        ),
        dict(
            name="slide_fast",
            steps=700_000,
            ranges={"vx": [-0.10, 0.10], "vy": [-1.10, 1.10], "omega": [-0.08, 0.08]},
            mode="uniform",
            imitation=False,
            ent_coef=0.005,
            learning_rate=1.5e-5,
        ),
        dict(
            name="slide_forward_blend",
            steps=700_000,
            ranges={"vx": [0.0, 1.10], "vy": [-0.95, 0.95], "omega": [-0.30, 0.30]},
            mode="uniform",
            imitation=False,
            ent_coef=0.003,
            learning_rate=1.0e-5,
        ),
        dict(
            name="slide_turn_blend",
            steps=650_000,
            ranges={"vx": [0.15, 1.35], "vy": [-0.90, 0.90], "omega": [-1.10, 1.10]},
            mode="uniform",
            imitation=False,
            ent_coef=0.0025,
            learning_rate=1.0e-5,
        ),
        dict(
            name="escape_slide_polish",
            steps=550_000,
            ranges={"vx": [-0.15, 1.45], "vy": [-1.00, 1.00], "omega": [-1.30, 1.30]},
            mode="uniform",
            imitation=False,
            ent_coef=0.002,
            learning_rate=1.0e-5,
        ),
    ],
}


GEN5_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)
GEN5_PPO_OVERRIDES.update({
    "target_kl": 0.008,
    "n_epochs": 5,
})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_gen4_lateral_001/best",
        help="Generation-4 walk model to evolve.",
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--profile", choices=sorted(GEN5_PROFILES), default="lateral_slide")
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--record-videos", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    source = Path(args.source)
    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen5"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    stages = GEN5_PROFILES[args.profile]
    history = []

    reward_config = base.AGILITY_REWARD

    for stage in tqdm(stages, desc="gen5 stages", unit="stage"):
        env_kwargs = {
            "action_scale": 0.6,
            "joint_stiffness": 60.0,
            "joint_damping": 2.0,
            "torque_limits": [23.7, 23.7, 45.43],
            "command_mode": stage["mode"],
            "command_ranges": stage["ranges"],
        }
        print(
            f"\n=== gen5 stage {stage['name']} steps={stage['steps']} "
            f"ranges={stage['ranges']} ===",
            flush=True,
        )

        ppo_overrides = dict(
            GEN5_PPO_OVERRIDES,
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
            checkpoint_steps=max(20_000, int(stage["steps"]) // 2),
            progress_bar=True,
            walk_env_kwargs=env_kwargs,
            allow_reward_change=True,
            initial_log_std=-1.9 if stage["name"] in {"slide_acquire", "slide_fast"} else -2.2,
            ppo_overrides=ppo_overrides,
            freeze_normalization=False,
        )

        history.append({
            "stage": stage,
            "params": json.loads((run_dir / "walk_params.json").read_text(encoding="utf-8")),
        })
        (run_dir / "gen5_curriculum.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Evaluate the new slide primitive at several strengths, plus the old
    # forward/turn skills that Gen4 already handled well.
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
        "turn_in_place_left": (0.0, 0.0, 1.20),
        "turn_in_place_right": (0.0, 0.0, -1.20),
        "turn_left_sprint": (1.25, 0.0, 1.0),
        "turn_right_sprint": (1.25, 0.0, -1.0),
    }

    walk_rows, walk_summary = base.eval_walk(run_dir, commands, seeds, args.seconds)
    tag_rows, tag_summary = base.eval_tag_direct(run_dir, commands, seeds, args.seconds)
    result = {
        "generation": 5,
        "goal": "make_lateral_slide_usable_without_losing_forward_turn_or_sprint",
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }
    (run_dir / "gen5_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in base.REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_gen5_record.json").write_text(
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

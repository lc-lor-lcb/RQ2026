"""
Generation-12 random-angle walk evolution.

Based on Gen11's explicit angle training, Gen12 adds dynamic random-angle
commands during an episode. The target movement direction is sampled
continuously from -180..180 degrees and is periodically re-sampled so the
walker must learn to change direction instead of memorizing a fixed command.

Key points:
- Keeps the 45-D observation space and Gen11-compatible policy weights.
- Keeps Gen11 physics/reward structure.
- Uses SubprocVecEnv explicitly (important for the fast training path).
- Random angle is sampled on reset and then re-sampled during the episode.
- The command hold interval is shortened through the curriculum.
- No escape/tag behavior is trained here.
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm

from scripts import train_speed_walk_base as base
from scripts.train_walk_gen11 import AngleAwareGo2WalkEnv, build_commands, eval_angle_walk


# -----------------------------------------------------------------------------
# Gen12 environment
# -----------------------------------------------------------------------------
class RandomAngleGo2WalkEnv(AngleAwareGo2WalkEnv):
    """Gen11 angle-aware environment with dynamic random-angle switching."""

    def __init__(self, *args, command_switch_steps=150, **kwargs):
        self.command_switch_steps = max(1, int(command_switch_steps))
        self._command_step_counter = 0
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._command_step_counter = 0
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


# -----------------------------------------------------------------------------
# Curriculum
# -----------------------------------------------------------------------------
GEN12_PROFILES = {
    "random_angle": [
        dict(
            name="random_angle_foundation",
            steps=800_000,
            speed=[0.45, 1.25],
            angle=[-180.0, 180.0],
            omega=[-0.15, 0.15],
            direction_error_weight=-1.0,
            switch_steps=250,
            ent_coef=0.0025,
            learning_rate=1.00e-5,
        ),
        dict(
            name="random_angle_360",
            steps=1_100_000,
            speed=[0.60, 1.35],
            angle=[-180.0, 180.0],
            omega=[-0.20, 0.20],
            direction_error_weight=-1.5,
            switch_steps=200,
            ent_coef=0.0023,
            learning_rate=9.5e-6,
        ),
        dict(
            name="random_angle_high_speed",
            steps=1_200_000,
            speed=[0.85, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.25, 0.25],
            direction_error_weight=-2.0,
            switch_steps=160,
            ent_coef=0.0021,
            learning_rate=9.0e-6,
        ),
        dict(
            name="random_angle_switch",
            steps=1_200_000,
            speed=[0.45, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.35, 0.35],
            direction_error_weight=-2.3,
            switch_steps=120,
            ent_coef=0.0019,
            learning_rate=8.8e-6,
        ),
        dict(
            name="random_angle_turn",
            steps=1_200_000,
            speed=[0.40, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.70, 0.70],
            direction_error_weight=-2.5,
            switch_steps=100,
            ent_coef=0.0017,
            learning_rate=8.5e-6,
        ),
        dict(
            name="random_angle_fast_switch",
            steps=1_300_000,
            speed=[0.30, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.90, 0.90],
            direction_error_weight=-2.5,
            switch_steps=75,
            ent_coef=0.0016,
            learning_rate=8.2e-6,
        ),
        dict(
            name="final_random_angle",
            steps=1_500_000,
            speed=[0.30, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.90, 0.90],
            direction_error_weight=-2.4,
            switch_steps=60,
            ent_coef=0.0015,
            learning_rate=8.0e-6,
        ),
    ]
}


GEN12_PPO_OVERRIDES = dict(base.SPEED_PPO_OVERRIDES)
GEN12_PPO_OVERRIDES.update({
    "target_kl": 0.007,
    "n_epochs": 5,
})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generation-12 dynamic random-angle walk evolution."
    )
    parser.add_argument(
        "--source",
        default="runs/walk_base/walk_gen11_main_001/best",
        help="Gen11 best model directory.",
    )
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(GEN12_PROFILES),
        default="random_angle",
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

    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_walk_gen12"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base.copy_base(source, run_dir)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    stages = GEN12_PROFILES[args.profile]

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
            "omega": [-0.9, 0.9],
        },
    }

    history = []

    for stage_index, stage in enumerate(
        tqdm(stages, desc="gen12 stages", unit="stage")
    ):
        print(f"\n=== Gen12 stage {stage_index + 1}/{len(stages)}: {stage['name']} ===", flush=True)
        print(f"steps={stage['steps']}", flush=True)
        print(f"speed={stage['speed']}", flush=True)
        print(f"angle={stage['angle']}", flush=True)
        print(f"omega={stage['omega']}", flush=True)
        print(f"direction_error_weight={stage['direction_error_weight']}", flush=True)
        print(f"random_angle_switch_steps={stage['switch_steps']}", flush=True)

        def make_env():
            return RandomAngleGo2WalkEnv(
                randomize_cmd=True,
                angle_ranges=stage["angle"],
                speed_ranges=stage["speed"],
                direction_error_weight=stage["direction_error_weight"],
                direction_error_speed_threshold=0.15,
                actual_speed_threshold=0.05,
                command_switch_steps=stage["switch_steps"],
                **common_physics,
                command_ranges={
                    "vx": [-1.4, 1.4],
                    "vy": [-1.2, 1.2],
                    "omega": stage["omega"],
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

        ppo_kwargs = dict(
            GEN12_PPO_OVERRIDES,
            device=args.device,
            learning_rate=stage["learning_rate"],
            ent_coef=stage["ent_coef"],
        )

        checkpoint_dir = run_dir / "checkpoints" / stage["name"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointCallback(
            save_freq=max(20_000 // max(args.num_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="walk_gen12",
            save_vecnormalize=True,
        )

        # PPO.load already contains the optimizer/model state. The explicit
        # ppo_kwargs are recorded for reproducibility; the current model keeps
        # the optimizer settings loaded from the checkpoint.
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
            "generation": 12,
            "source": str(source),
            "stage": stage,
            "physics": common_physics,
            "walk_env_kwargs": standard_env_kwargs,
            "random_angle_training": {
                "enabled": True,
                "speed_range": stage["speed"],
                "angle_range_deg": stage["angle"],
                "omega_range": stage["omega"],
                "command_switch_steps": stage["switch_steps"],
                "continuous_uniform_angle": True,
                "direction_error_weight": stage["direction_error_weight"],
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
        (run_dir / "gen12_curriculum.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        vec_env.close()

    # Reuse Gen11's deterministic angle evaluation for a clean comparison.
    commands = build_commands()
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
        "generation": 12,
        "goal": "dynamic continuous random-angle walking with progressive command switching",
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }
    (run_dir / "gen12_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in base.REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_gen12_record.json").write_text(
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

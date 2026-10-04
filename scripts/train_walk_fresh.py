"""
Fresh walk training from random initialization.

Curriculum:
    1. Foundation walking
    2. Speed tracking
    3. 360-degree direction tracking without commanded turning
    4. Turning
    5. High-speed + turning

IMPORTANT:
- No Gen11/Gen12/Gen13 model is loaded.
- PPO starts from a random policy.
- A short teacher warm-start is used only to establish a basic gait.
- All curriculum stages then continue with PPO.
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

from roboquest.envs.go2_walk_env import CONTROL_DT, Go2WalkEnv
from roboquest.utils.reward_utils import WalkRewardConfig

from scripts.train_walk_gen11 import (
    AngleAwareGo2WalkEnv,
    build_commands,
    eval_angle_walk,
)

from scripts import train_speed_walk_base as base
from scripts.bootstrap_smooth_walk import example_action


# ============================================================
# Reward
# ============================================================

SPEED_REWARD = WalkRewardConfig(
    linear_tracking_variance=0.20,
    angular_tracking_variance=0.10,
    lin_vel_weight=9.0,
    ang_vel_weight=2.0,
    orientation_weight=-10.0,
    base_height_target=0.29,
    base_height_weight=-300.0,
    vertical_velocity_weight=-1.0,
    nonfoot_contact_weight=-2.0,
    knee_height_target=0.08,
    knee_height_weight=-400.0,
    torques_weight=-2.5e-5,
    action_rate_weight=-0.015,
    joint_velocity_weight=-0.001,
    stand_joint_velocity_weight=-0.01,
    feet_gait_weight=0.0,
    diagonal_support_weight=0.4,
    foot_slip_weight=-0.02,
    fall_penalty=40.0,
)

AGILITY_REWARD = WalkRewardConfig(
    linear_tracking_variance=0.18,
    angular_tracking_variance=0.18,
    lin_vel_weight=8.0,
    ang_vel_weight=7.0,
    linear_error_weight=-2.5,
    angular_error_weight=-4.0,
    orientation_weight=-8.0,
    base_height_target=0.29,
    base_height_weight=-260.0,
    vertical_velocity_weight=-1.0,
    nonfoot_contact_weight=-2.0,
    knee_height_target=0.08,
    knee_height_weight=-350.0,
    torques_weight=-2.5e-5,
    action_rate_weight=-0.010,
    joint_velocity_weight=-0.001,
    stand_joint_velocity_weight=-0.01,
    feet_gait_weight=0.0,
    diagonal_support_weight=0.2,
    foot_slip_weight=-0.02,
    fall_penalty=40.0,
)


# ============================================================
# Physics
# ============================================================

COMMON_PHYSICS = {
    "action_scale": 0.6,
    "joint_stiffness": 60.0,
    "joint_damping": 2.0,
    "torque_limits": [23.7, 23.7, 45.43],
}


# ============================================================
# Curriculum
# ============================================================

STAGES = [
    # --------------------------------------------------------
    # 1. Foundation
    # --------------------------------------------------------
    dict(
        name="foundation_walk",
        steps=1_000_000,
        speed=[0.30, 0.60],
        angle=[0.0, 0.0],
        omega=[0.0, 0.0],
        direction_error_weight=0.0,
        switch_steps=500,
        reward="speed",
        learning_rate=1.2e-5,
        ent_coef=0.0025,
    ),

    # --------------------------------------------------------
    # 2. Speed tracking
    # --------------------------------------------------------
    dict(
        name="speed_tracking",
        steps=1_200_000,
        speed=[0.40, 1.00],
        angle=[0.0, 0.0],
        omega=[0.0, 0.0],
        direction_error_weight=0.0,
        switch_steps=400,
        reward="speed",
        learning_rate=1.0e-5,
        ent_coef=0.0022,
    ),

    # --------------------------------------------------------
    # 3. Direction tracking
    # No commanded turning.
    # --------------------------------------------------------
    dict(
        name="direction_tracking_360",
        steps=1_500_000,
        speed=[0.40, 1.10],
        angle=[-180.0, 180.0],
        omega=[0.0, 0.0],
        direction_error_weight=-1.5,
        switch_steps=350,
        reward="speed",
        learning_rate=9.5e-6,
        ent_coef=0.0020,
    ),

    # --------------------------------------------------------
    # 4. Turning
    # --------------------------------------------------------
    dict(
        name="turning",
        steps=1_400_000,
        speed=[0.45, 1.20],
        angle=[-180.0, 180.0],
        omega=[-0.50, 0.50],
        direction_error_weight=-1.8,
        switch_steps=300,
        reward="agility",
        learning_rate=9.0e-6,
        ent_coef=0.0018,
    ),

    # --------------------------------------------------------
    # 5. High speed
    # --------------------------------------------------------
    dict(
        name="high_speed",
        steps=1_500_000,
        speed=[0.80, 1.40],
        angle=[-180.0, 180.0],
        omega=[-0.90, 0.90],
        direction_error_weight=-2.2,
        switch_steps=250,
        reward="agility",
        learning_rate=8.5e-6,
        ent_coef=0.0015,
    ),
]


# ============================================================
# Dynamic random-angle environment
# ============================================================

class FreshRandomAngleGo2WalkEnv(AngleAwareGo2WalkEnv):
    """
    Angle-aware environment used by the fresh curriculum.

    Commands are periodically re-sampled.
    """

    def __init__(
        self,
        *args,
        command_switch_steps=300,
        **kwargs,
    ):
        self.command_switch_steps = max(
            1,
            int(command_switch_steps),
        )
        self._command_step_counter = 0

        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(
            seed=seed,
            options=options,
        )
        self._command_step_counter = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(
            action
        )

        self._command_step_counter += 1

        if (
            self.randomize_cmd
            and not terminated
            and not truncated
            and self._command_step_counter
            >= self.command_switch_steps
        ):
            self._sample_angle_command()
            self._command_step_counter = 0

            obs = self._get_obs().astype(
                np.float32
            )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )


# ============================================================
# Environment factory
# ============================================================

def make_stage_env(stage):
    def make_env():
        reward = (
            SPEED_REWARD
            if stage["reward"] == "speed"
            else AGILITY_REWARD
        )

        return FreshRandomAngleGo2WalkEnv(
            randomize_cmd=True,

            angle_ranges=stage["angle"],
            speed_ranges=stage["speed"],

            direction_error_weight=(
                stage["direction_error_weight"]
            ),

            direction_error_speed_threshold=0.15,
            actual_speed_threshold=0.05,

            command_switch_steps=(
                stage["switch_steps"]
            ),

            reward_config=reward,

            **COMMON_PHYSICS,

            command_ranges={
                "vx": [-1.4, 1.4],
                "vy": [-1.2, 1.2],
                "omega": stage["omega"],
            },
        )

    return make_env


# ============================================================
# Random model creation
# ============================================================

def create_random_model(vec_env, device):
    """
    Create PPO from scratch.

    There is intentionally NO PPO.load().
    """

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1.2e-5,
        n_epochs=5,
        target_kl=0.007,
        ent_coef=0.0025,

        # Same general PPO family used by Gen11-13.
        device=device,

        verbose=1,
    )

    # Start with moderately conservative exploration.
    with torch.no_grad():
        model.policy.log_std.fill_(-2.2)

    return model


# ============================================================
# Teacher warm-start from random initialization
# ============================================================

def teacher_warmstart(
    model,
    vec_env,
    stage,
    sample_steps=30_000,
    updates=2_000,
    seed=1234,
):
    """
    Establish a basic forward gait.

    IMPORTANT:
    The PPO model is already randomly initialized.

    This is NOT loading a previous generation.
    """

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    teacher_env = Go2WalkEnv(
        randomize_cmd=False,
        reward_config=SPEED_REWARD,
        **COMMON_PHYSICS,
        command_mode="uniform",
        command_ranges={
            "vx": stage["speed"],
            "vy": [0.0, 0.0],
            "omega": [0.0, 0.0],
        },
    )

    observations = []
    targets = []

    try:
        obs, _ = teacher_env.reset(seed=seed)

        for step in tqdm(
            range(sample_steps),
            desc="fresh gait warm-start",
            unit="step",
        ):
            local_step = step % 500

            if local_step == 0:
                vx = float(
                    rng.uniform(
                        stage["speed"][0],
                        stage["speed"][1],
                    )
                )

                teacher_env.set_vel_cmd(
                    vx,
                    0.0,
                    0.0,
                )

                obs, _ = teacher_env.reset(
                    seed=seed + step
                )

            action = example_action(
                (local_step % 250) * CONTROL_DT,
                float(teacher_env.vel_cmd[0]),
                frequency=2.0,
                height=0.31,
                lift=0.055,
            )

            observations.append(
                np.asarray(obs, dtype=np.float32)
            )

            targets.append(
                np.asarray(action, dtype=np.float32)
            )

            obs, _, terminated, truncated, _ = (
                teacher_env.step(action)
            )

            if terminated or truncated:
                obs, _ = teacher_env.reset(
                    seed=seed + step + 10000
                )

        observations_np = np.asarray(
            observations,
            dtype=np.float32,
        )

        targets_np = np.asarray(
            targets,
            dtype=np.float32,
        )

        # Update observation statistics.
        vec_env.obs_rms.update(
            observations_np
        )

        x = torch.as_tensor(
            vec_env.normalize_obs(
                observations_np
            ),
            dtype=torch.float32,
        )

        y = torch.as_tensor(
            targets_np,
            dtype=torch.float32,
        )

        actor_parameters = (
            list(
                model.policy.mlp_extractor
                .policy_net.parameters()
            )
            +
            list(
                model.policy.action_net.parameters()
            )
        )

        optimizer = torch.optim.Adam(
            actor_parameters,
            lr=1e-4,
        )

        final_loss = None

        for _ in tqdm(
            range(updates),
            desc="warm-start updates",
            unit="update",
        ):
            idx = torch.as_tensor(
                rng.integers(
                    0,
                    len(x),
                    size=512,
                ),
                dtype=torch.long,
            )

            prediction = model.policy._predict(
                x[idx],
                deterministic=True,
            )

            loss = (
                prediction - y[idx]
            ).square().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = float(
                loss.item()
            )

        with torch.no_grad():
            model.policy.log_std.fill_(-2.5)

        # The PPO optimizer should not retain
        # stale optimizer state from the random model.
        model.policy.optimizer.state.clear()

        print(
            f"warm-start final MSE = {final_loss}",
            flush=True,
        )

    finally:
        teacher_env.close()


# ============================================================
# Evaluation
# ============================================================

def save_evaluation(run_dir, args, standard_env_kwargs):
    commands = build_commands()

    walk_rows, walk_summary = eval_angle_walk(
        run_dir,
        commands,
        args.seeds_list,
        args.seconds,
        standard_env_kwargs,
    )

    tag_rows, tag_summary = base.eval_tag_direct(
        run_dir,
        commands,
        args.seeds_list,
        args.seconds,
    )

    result = {
        "fresh_training": True,
        "initialization": "random",
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }

    path = (
        run_dir
        / "fresh_evaluation.json"
    )

    path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fresh random-initialization "
            "walk curriculum."
        )
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
        "--num-envs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
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

    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help=(
            "Disable the initial gait "
            "teacher warm-start."
        ),
    )

    args = parser.parse_args()

    args.seeds_list = [
        int(x)
        for x in
        args.seeds.replace(",", " ").split()
    ]

    torch.set_num_threads(1)

    run_name = (
        args.run_name
        or
        f"{time.strftime('%Y%m%d_%H%M%S')}"
        "_walk_fresh_001"
    )

    run_dir = (
        Path(args.output_root)
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "\n========================================",
        flush=True,
    )
    print(
        "FRESH TRAINING",
        flush=True,
    )
    print(
        "Random initialization: YES",
        flush=True,
    )
    print(
        "Previous Gen model: NONE",
        flush=True,
    )
    print(
        "========================================\n",
        flush=True,
    )

    history = []

    model = None
    vec_env = None

    for stage_index, stage in enumerate(
        tqdm(
            STAGES,
            desc="fresh curriculum",
            unit="stage",
        )
    ):
        print(
            f"\n=== Fresh stage "
            f"{stage_index + 1}/"
            f"{len(STAGES)}: "
            f"{stage['name']} ===",
            flush=True,
        )

        print(
            f"steps={stage['steps']}",
            flush=True,
        )

        print(
            f"speed={stage['speed']}",
            flush=True,
        )

        print(
            f"angle={stage['angle']}",
            flush=True,
        )

        print(
            f"omega={stage['omega']}",
            flush=True,
        )

        print(
            f"direction_error_weight="
            f"{stage['direction_error_weight']}",
            flush=True,
        )

        make_env = make_stage_env(stage)

        # IMPORTANT:
        # Keep the fast SubprocVecEnv path that
        # fixed the Gen11 performance problem.
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

            model = create_random_model(
                vec_env,
                args.device,
            )

            if not args.no_warmstart:
                teacher_warmstart(
                    model,
                    vec_env,
                    stage,
                    sample_steps=30_000,
                    updates=2_000,
                    seed=1000,
                )

        else:
            vec_env = VecNormalize.load(
                str(
                    run_dir
                    / "walk_model_vecnorm.pkl"
                ),
                vec_raw,
            )

            vec_env.training = True
            vec_env.norm_reward = True

            # Continue the model learned in the
            # previous curriculum stage.
            model.set_env(vec_env)

        # Stage-specific PPO parameters.
        model.learning_rate = (
            stage["learning_rate"]
        )

        model.ent_coef = (
            stage["ent_coef"]
        )

        model.target_kl = 0.007
        model.n_epochs = 5

        checkpoint_dir = (
            run_dir
            / "checkpoints"
            / stage["name"]
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = CheckpointCallback(
            save_freq=max(
                20_000
                // max(args.num_envs, 1),
                1,
            ),
            save_path=str(
                checkpoint_dir
            ),
            name_prefix="walk_fresh",
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps=int(
                stage["steps"]
            ),
            callback=checkpoint,
            progress_bar=True,
            reset_num_timesteps=(
                stage_index == 0
            ),
        )

        model.save(
            run_dir / "walk_model"
        )

        vec_env.save(
            str(
                run_dir
                / "walk_model_vecnorm.pkl"
            )
        )

        params = {
            "kind": "walk",
            "training_type": "fresh_random_initialization",
            "source": None,
            "stage_index": stage_index,
            "stage": stage,
            "physics": COMMON_PHYSICS,
            "walk_env_kwargs": {
                **COMMON_PHYSICS,
                "command_mode": "uniform",
                "command_ranges": {
                    "vx": [-1.4, 1.4],
                    "vy": [-1.2, 1.2],
                    "omega": [-0.9, 0.9],
                },
            },
            "curriculum": STAGES,
            "vec_env_cls": "SubprocVecEnv",
            "num_envs": args.num_envs,
            "device": args.device,
        }

        (
            run_dir
            / "walk_params.json"
        ).write_text(
            json.dumps(
                params,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        history.append(
            {
                "stage": stage,
                "params": params,
            }
        )

        (
            run_dir
            / "fresh_curriculum.json"
        ).write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        vec_env.close()
        vec_env = None

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    standard_env_kwargs = {
        **COMMON_PHYSICS,
        "command_mode": "uniform",
        "command_ranges": {
            "vx": [-1.4, 1.4],
            "vy": [-1.2, 1.2],
            "omega": [-0.9, 0.9],
        },
    }

    result = save_evaluation(
        run_dir,
        args,
        standard_env_kwargs,
    )

    # --------------------------------------------------------
    # Best directory
    # --------------------------------------------------------

    best_dir = (
        run_dir
        / "best"
    )

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
        / "best_fresh_record.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "walk_summary":
                    result["walk_summary"],
                "tag_summary":
                    result["tag_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
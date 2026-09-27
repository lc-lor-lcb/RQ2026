"""
Generation-11 walk evolution.

Goal:
- Start from the Gen10 omnidirectional model.
- Introduce explicit movement-angle commands.
- Train the walker to move in the commanded direction.
- Penalize directional error independently from speed error.
- Preserve high forward speed while improving 360-degree precision.
- Preserve reverse / lateral / diagonal / turning skills.
- Do NOT train escape behavior yet.
- Do NOT use genetic algorithms yet.

Gen11 key changes:
1. Commands are sampled as:
       speed + angle_deg + omega
   instead of independently sampling vx/vy.

2. Direction error penalty:
       actual velocity angle vs commanded angle

3. Directional error is represented as:
       1 - cos(angle_error)
   so:
       0 deg   -> 0 penalty
       90 deg  -> 1 penalty
       180 deg -> 2 penalty

4. The observation remains 45-dimensional:
       [vx, vy, omega, ...]
   so Gen10 weights remain compatible.

Angle convention:
    0°   = forward
    90°  = left
    180° = reverse
    270° = right
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
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
)
from tqdm.auto import tqdm

from roboquest.envs.go2_walk_env import Go2WalkEnv
from roboquest.utils.reward_utils import WalkRewardConfig

from scripts import train_speed_walk_base as base


# ============================================================
# Gen11 direction-aware environment
# ============================================================

class AngleAwareGo2WalkEnv(Go2WalkEnv):
    """
    Gen11-specific walk environment.

    The actual observation stays 45-dimensional.

    During training, commands are generated from:
        speed + angle_deg + omega

    The generated command is converted internally to:
        vx = speed * cos(angle)
        vy = speed * sin(angle)

    A separate directional error penalty is then applied.
    """

    def __init__(
        self,
        *args,
        angle_ranges=None,
        speed_ranges=None,
        direction_error_weight=-2.5,
        direction_error_speed_threshold=0.15,
        actual_speed_threshold=0.05,
        **kwargs,
    ):
        # Go2WalkEnv itself only knows uniform / axis.
        # We perform angle sampling here.
        kwargs["command_mode"] = "uniform"

        # Dummy ranges are required by the parent constructor.
        kwargs["command_ranges"] = {
            "vx": [-1.4, 1.4],
            "vy": [-1.2, 1.2],
            "omega": kwargs.get(
                "command_ranges",
                {"omega": [-0.9, 0.9]},
            ).get("omega", [-0.9, 0.9]),
        }

        super().__init__(*args, **kwargs)

        self.angle_ranges = (
            angle_ranges
            if angle_ranges is not None
            else [-180.0, 180.0]
        )

        self.speed_ranges = (
            speed_ranges
            if speed_ranges is not None
            else [0.0, 1.4]
        )

        self.direction_error_weight = float(
            direction_error_weight
        )

        self.direction_error_speed_threshold = float(
            direction_error_speed_threshold
        )

        self.actual_speed_threshold = float(
            actual_speed_threshold
        )

        self.target_angle_deg = 0.0
        self.target_speed = 0.0

    # --------------------------------------------------------
    # Angle command generation
    # --------------------------------------------------------

    def _sample_angle_command(self) -> None:
        speed = float(
            self.np_random.uniform(
                self.speed_ranges[0],
                self.speed_ranges[1],
            )
        )

        angle_deg = float(
            self.np_random.uniform(
                self.angle_ranges[0],
                self.angle_ranges[1],
            )
        )

        omega_lo, omega_hi = self.command_ranges["omega"]

        omega = float(
            self.np_random.uniform(
                omega_lo,
                omega_hi,
            )
        )

        angle_rad = math.radians(angle_deg)

        vx = speed * math.cos(angle_rad)
        vy = speed * math.sin(angle_rad)

        self.target_speed = speed
        self.target_angle_deg = angle_deg

        self.set_vel_cmd(
            vx,
            vy,
            omega,
        )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        if self.randomize_cmd:
            self._sample_angle_command()

            # Rebuild observation because parent reset generated
            # a temporary uniform command first.
            obs = self._get_obs().astype(np.float32)

        return obs, info

    # --------------------------------------------------------
    # Direction error
    # --------------------------------------------------------

    @staticmethod
    def _wrap_angle_rad(angle: float) -> float:
        return (
            angle + math.pi
        ) % (
            2.0 * math.pi
        ) - math.pi

    def _direction_error(self) -> tuple[float, float]:
        """
        Returns:
            error_rad
            actual_speed
        """

        lin_vel_body = (
            self._world_to_body(
                self.data.qvel[:3]
            )[:2]
        )

        actual_vx = float(lin_vel_body[0])
        actual_vy = float(lin_vel_body[1])

        actual_speed = math.sqrt(
            actual_vx * actual_vx
            + actual_vy * actual_vy
        )

        target_vx = float(self._vel_cmd[0])
        target_vy = float(self._vel_cmd[1])

        target_speed = math.sqrt(
            target_vx * target_vx
            + target_vy * target_vy
        )

        if (
            target_speed
            < self.direction_error_speed_threshold
            or actual_speed
            < self.actual_speed_threshold
        ):
            return 0.0, actual_speed

        target_angle = math.atan2(
            target_vy,
            target_vx,
        )

        actual_angle = math.atan2(
            actual_vy,
            actual_vx,
        )

        error = self._wrap_angle_rad(
            actual_angle - target_angle
        )

        return error, actual_speed

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> float:
        reward = super()._compute_reward(action)

        error_rad, actual_speed = (
            self._direction_error()
        )

        if (
            self.target_speed
            >= self.direction_error_speed_threshold
            and actual_speed
            >= self.actual_speed_threshold
        ):
            # 1 - cos(error)
            #
            # 0°   -> 0
            # 45°  -> ~0.293
            # 90°  -> 1
            # 180° -> 2
            direction_error = (
                1.0 - math.cos(error_rad)
            )

            reward += (
                self.direction_error_weight
                * direction_error
            )

        return reward


# ============================================================
# Gen11 curriculum
# ============================================================

GEN11_PROFILES = {
    "angle_precision": [

        # ----------------------------------------------------
        # Stage 1
        # Gen10の能力を壊さず角度指定へ移行
        # ----------------------------------------------------

        dict(
            name="angle_foundation",
            steps=700_000,
            speed=[0.45, 1.25],
            angle=[-180.0, 180.0],
            omega=[-0.15, 0.15],
            direction_error_weight=-1.0,
            ent_coef=0.0028,
            learning_rate=1.05e-5,
        ),

        # ----------------------------------------------------
        # Stage 2
        # 360°方向精度
        # ----------------------------------------------------

        dict(
            name="angle_360_precision",
            steps=1_100_000,
            speed=[0.60, 1.35],
            angle=[-180.0, 180.0],
            omega=[-0.20, 0.20],
            direction_error_weight=-1.5,
            ent_coef=0.0025,
            learning_rate=1.00e-5,
        ),

        # ----------------------------------------------------
        # Stage 3
        # 高速360°
        # ----------------------------------------------------

        dict(
            name="angle_high_speed",
            steps=1_200_000,
            speed=[0.85, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.25, 0.25],
            direction_error_weight=-2.0,
            ent_coef=0.0022,
            learning_rate=9.5e-6,
        ),

        # ----------------------------------------------------
        # Stage 4
        # 低速～高速を混ぜて汎化
        # ----------------------------------------------------

        dict(
            name="angle_speed_generalization",
            steps=1_100_000,
            speed=[0.30, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.30, 0.30],
            direction_error_weight=-2.5,
            ent_coef=0.0020,
            learning_rate=9.0e-6,
        ),

        # ----------------------------------------------------
        # Stage 5
        # 斜め方向重点
        #
        # 22.5°刻みの方向を学習上も強く経験させる。
        # ----------------------------------------------------

        dict(
            name="angle_diagonal_precision",
            steps=1_200_000,
            speed=[0.75, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.30, 0.30],
            direction_error_weight=-3.0,
            ent_coef=0.0018,
            learning_rate=8.8e-6,
        ),

        # ----------------------------------------------------
        # Stage 6
        # 方向＋旋回
        # ----------------------------------------------------

        dict(
            name="angle_movement_turn",
            steps=1_100_000,
            speed=[0.55, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.85, 0.85],
            direction_error_weight=-2.8,
            ent_coef=0.0017,
            learning_rate=8.5e-6,
        ),

        # ----------------------------------------------------
        # Stage 7
        # 全要素統合
        # ----------------------------------------------------

        dict(
            name="final_angle_integration",
            steps=1_400_000,
            speed=[0.30, 1.40],
            angle=[-180.0, 180.0],
            omega=[-0.90, 0.90],
            direction_error_weight=-2.5,
            ent_coef=0.0015,
            learning_rate=8.0e-6,
        ),
    ]
}


# ============================================================
# PPO
# ============================================================

GEN11_PPO_OVERRIDES = dict(
    base.SPEED_PPO_OVERRIDES
)

GEN11_PPO_OVERRIDES.update({
    "target_kl": 0.007,
    "n_epochs": 5,
})


# ============================================================
# Angle utilities
# ============================================================

def direction_command(
    angle_deg: float,
    speed: float,
    omega: float = 0.0,
) -> tuple[float, float, float]:

    rad = math.radians(angle_deg)

    return (
        speed * math.cos(rad),
        speed * math.sin(rad),
        omega,
    )


def wrap_angle_deg(angle: float) -> float:
    return (
        angle + 180.0
    ) % 360.0 - 180.0


# ============================================================
# Evaluation commands
# ============================================================

def build_commands():

    commands = {
        "stand": (0.0, 0.0, 0.0),
    }

    # --------------------------------------------------------
    # Cardinal
    # --------------------------------------------------------

    cardinal = {
        "forward": 0.0,
        "left": 90.0,
        "back": 180.0,
        "right": 270.0,
    }

    for name, angle in cardinal.items():
        commands[
            f"angle_{name}_1.0"
        ] = direction_command(
            angle,
            1.0,
        )

        commands[
            f"angle_{name}_1.25"
        ] = direction_command(
            angle,
            1.25,
        )

        commands[
            f"angle_{name}_1.4"
        ] = direction_command(
            angle,
            1.4,
        )

    # --------------------------------------------------------
    # 16 directions
    # --------------------------------------------------------

    directions = [
        ("front", 0.0),
        ("front_left_22_5", 22.5),
        ("front_left_45", 45.0),
        ("front_left_67_5", 67.5),
        ("left", 90.0),
        ("back_left_112_5", 112.5),
        ("back_left_135", 135.0),
        ("back_left_157_5", 157.5),
        ("back", 180.0),
        ("back_right_202_5", 202.5),
        ("back_right_225", 225.0),
        ("back_right_247_5", 247.5),
        ("right", 270.0),
        ("front_right_292_5", 292.5),
        ("front_right_315", 315.0),
        ("front_right_337_5", 337.5),
    ]

    for name, angle in directions:

        commands[
            f"angle_{name}_1.0"
        ] = direction_command(
            angle,
            1.0,
        )

        commands[
            f"angle_{name}_1.25"
        ] = direction_command(
            angle,
            1.25,
        )

        commands[
            f"angle_{name}_1.4"
        ] = direction_command(
            angle,
            1.4,
        )

    # --------------------------------------------------------
    # Moving + turning
    # --------------------------------------------------------

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

    commands["left_turn"] = (
        0.0,
        1.0,
        0.8,
    )

    commands["right_turn"] = (
        0.0,
        -1.0,
        -0.8,
    )

    commands["back_left_turn"] = (
        -0.75,
        0.75,
        0.7,
    )

    commands["back_right_turn"] = (
        -0.75,
        -0.75,
        -0.7,
    )

    return commands


# ============================================================
# Angle-aware evaluation
# ============================================================

def load_angle_policy(
    folder: Path,
    env_kwargs: dict,
):

    base_env = make_vec_env(
        lambda: AngleAwareGo2WalkEnv(
            randomize_cmd=False,
            **env_kwargs,
        ),
        n_envs=1,
    )

    norm = VecNormalize.load(
        str(folder / "walk_model_vecnorm.pkl"),
        base_env,
    )

    norm.training = False
    norm.norm_reward = False

    model = PPO.load(
        folder / "walk_model",
        env=norm,
        device="cpu",
    )

    return base_env, norm, model


def eval_angle_walk(
    folder: Path,
    commands: dict,
    seeds: list[int],
    seconds: float,
    env_kwargs: dict,
):

    base_env, norm, model = load_angle_policy(
        folder,
        env_kwargs,
    )

    raw = base_env.envs[0].unwrapped

    rows = []

    try:

        total = (
            len(commands)
            * len(seeds)
        )

        pbar = tqdm(
            total=total,
            desc="gen11 angle eval",
            unit="case",
        )

        for name, command in commands.items():

            target_vx = float(command[0])
            target_vy = float(command[1])
            target_omega = float(command[2])

            target_speed = math.sqrt(
                target_vx ** 2
                + target_vy ** 2
            )

            target_angle = (
                math.degrees(
                    math.atan2(
                        target_vy,
                        target_vx,
                    )
                )
                if target_speed > 1e-6
                else 0.0
            )

            for seed in seeds:

                raw.set_vel_cmd(
                    target_vx,
                    target_vy,
                    target_omega,
                )

                obs, _ = raw.reset(
                    seed=seed
                )

                # reset() with randomize_cmd=False
                # preserves the manually assigned command.

                velocities = []
                direction_errors = []
                heights = []

                terminated = False

                for step in range(
                    round(
                        seconds
                        / base.CONTROL_DT
                    )
                ):

                    normalized_obs = (
                        norm.normalize_obs(
                            obs
                        )
                    )

                    action, _ = model.predict(
                        normalized_obs,
                        deterministic=True,
                    )

                    obs, _, terminated, truncated, _ = (
                        raw.step(action)
                    )

                    body_vel = (
                        raw._world_to_body(
                            raw.data.qvel[:3]
                        )
                    )

                    vx = float(body_vel[0])
                    vy = float(body_vel[1])
                    omega = float(
                        raw.data.qvel[5]
                    )

                    speed = math.sqrt(
                        vx * vx
                        + vy * vy
                    )

                    if (
                        speed > 0.05
                        and target_speed > 0.15
                    ):
                        actual_angle = (
                            math.degrees(
                                math.atan2(
                                    vy,
                                    vx,
                                )
                            )
                        )

                        error = abs(
                            wrap_angle_deg(
                                actual_angle
                                - target_angle
                            )
                        )
                    else:
                        actual_angle = None
                        error = None

                    velocities.append(
                        [vx, vy, omega]
                    )

                    if error is not None:
                        direction_errors.append(
                            error
                        )

                    heights.append(
                        float(
                            raw.data.qpos[2]
                        )
                    )

                    if (
                        terminated
                        or truncated
                    ):
                        break

                values = np.asarray(
                    velocities,
                    dtype=float,
                )

                target = np.asarray(
                    command,
                    dtype=float,
                )

                rows.append({
                    "command": name,
                    "seed": seed,
                    "target_angle_deg": (
                        target_angle
                        if target_speed > 0.01
                        else None
                    ),
                    "mean_direction_error_deg": (
                        float(
                            np.mean(
                                direction_errors
                            )
                        )
                        if direction_errors
                        else None
                    ),
                    "max_direction_error_deg": (
                        float(
                            np.max(
                                direction_errors
                            )
                        )
                        if direction_errors
                        else None
                    ),
                    "fell": bool(
                        terminated
                    ),
                    "survived_seconds": float(
                        (step + 1)
                        * base.CONTROL_DT
                    ),
                    "min_height": float(
                        np.min(heights)
                    )
                    if heights
                    else None,
                    "mean_vx": float(
                        np.mean(
                            values[:, 0]
                        )
                    )
                    if len(values)
                    else None,
                    "mean_vy": float(
                        np.mean(
                            values[:, 1]
                        )
                    )
                    if len(values)
                    else None,
                    "mean_omega": float(
                        np.mean(
                            values[:, 2]
                        )
                    )
                    if len(values)
                    else None,
                    "velocity_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    values
                                    - target
                                ) ** 2
                            )
                        )
                    )
                    if len(values)
                    else None,
                })

                pbar.update(1)

        pbar.close()

    finally:
        norm.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    valid = [
        r for r in rows
        if r["mean_direction_error_deg"]
        is not None
    ]

    errors = [
        r["mean_direction_error_deg"]
        for r in valid
    ]

    falls = sum(
        1 for r in rows
        if r["fell"]
    )

    summary = {
        "passed": bool(
            rows
            and falls == 0
            and (
                not errors
                or float(np.mean(errors))
                <= 15.0
            )
        ),
        "fall_count": falls,
        "mean_direction_error_deg": (
            float(np.mean(errors))
            if errors
            else None
        ),
        "max_direction_error_deg": (
            float(np.max(errors))
            if errors
            else None
        ),
        "min_height": (
            float(
                min(
                    r["min_height"]
                    for r in rows
                    if r["min_height"]
                    is not None
                )
            )
            if rows
            else None
        ),
        "mean_survived_seconds": (
            float(
                np.mean([
                    r[
                        "survived_seconds"
                    ]
                    for r in rows
                ])
            )
            if rows
            else None
        ),
    }

    return rows, summary


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generation-11 angle-aware "
            "omnidirectional walk evolution."
        )
    )

    parser.add_argument(
        "--source",
        default=(
            "runs/walk_base/"
            "walk_gen10_omnidirectional_001/best"
        ),
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
        choices=sorted(GEN11_PROFILES),
        default="angle_precision",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
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
        f"{time.strftime('%Y%m%d_%H%M%S')}"
        "_walk_gen11"
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

    seeds = [
        int(x)
        for x in
        args.seeds.replace(",", " ").split()
    ]

    stages = GEN11_PROFILES[
        args.profile
    ]

    # --------------------------------------------------------
    # Physical parameters
    #
    # Keep Gen10 physics unchanged.
    # --------------------------------------------------------

    common_physics = {
        "action_scale": 0.6,
        "joint_stiffness": 60.0,
        "joint_damping": 2.0,
        "torque_limits": [
            23.7,
            23.7,
            45.43,
        ],
    }

    # Standard env kwargs are deliberately kept compatible
    # with the existing Go2WalkEnv / tag evaluator.
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

    # ========================================================
    # Curriculum
    # ========================================================

    for stage_index, stage in enumerate(
        tqdm(
            stages,
            desc="gen11 stages",
            unit="stage",
        )
    ):

        print(
            f"\n=== Gen11 stage "
            f"{stage_index + 1}/"
            f"{len(stages)}: "
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
            "direction_error_weight="
            f"{stage['direction_error_weight']}",
            flush=True,
        )

        # ----------------------------------------------------
        # Environment factory
        # ----------------------------------------------------

        def make_env():
            return AngleAwareGo2WalkEnv(
                randomize_cmd=True,
                angle_ranges=stage["angle"],
                speed_ranges=stage["speed"],
                direction_error_weight=(
                    stage[
                        "direction_error_weight"
                    ]
                ),
                direction_error_speed_threshold=0.15,
                actual_speed_threshold=0.05,
                **common_physics,
                command_ranges={
                    "vx": [-1.4, 1.4],
                    "vy": [-1.2, 1.2],
                    "omega": stage["omega"],
                },
            )

        # ----------------------------------------------------
        # Load VecNormalize
        # ----------------------------------------------------

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

            model = PPO.load(
                run_dir / "walk_model",
                env=vec_env,
                device=args.device,
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

            model = PPO.load(
                run_dir / "walk_model",
                env=vec_env,
                device=args.device,
            )

        # ----------------------------------------------------
        # PPO configuration
        # ----------------------------------------------------

        ppo_kwargs = dict(
            GEN11_PPO_OVERRIDES,
            device=args.device,
            learning_rate=stage[
                "learning_rate"
            ],
            ent_coef=stage[
                "ent_coef"
            ],
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

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
            name_prefix="walk_gen11",
            save_vecnormalize=True,
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.learn(
            total_timesteps=int(
                stage["steps"]
            ),
            callback=checkpoint,
            progress_bar=True,
            reset_num_timesteps=False,
        )

        # ----------------------------------------------------
        # Save current model
        # ----------------------------------------------------

        model.save(
            run_dir / "walk_model"
        )

        vec_env.save(
            str(
                run_dir
                / "walk_model_vecnorm.pkl"
            )
        )

        # ----------------------------------------------------
        # Save parameters
        # ----------------------------------------------------

        params = {
            "kind": "walk",
            "generation": 11,
            "source": str(source),
            "stage": stage,
            "physics": common_physics,
            "walk_env_kwargs": standard_env_kwargs,
            "angle_training": {
                "enabled": True,
                "speed_range": stage["speed"],
                "angle_range_deg": stage["angle"],
                "omega_range": stage["omega"],
                "direction_error_weight": (
                    stage[
                        "direction_error_weight"
                    ]
                ),
                "direction_error_formula": (
                    "weight * "
                    "(1 - cos(actual_angle - target_angle))"
                ),
            },
            "ppo": ppo_kwargs,
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

        history.append({
            "stage": stage,
            "params": params,
        })

        (
            run_dir
            / "gen11_curriculum.json"
        ).write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        vec_env.close()

    # ========================================================
    # Evaluation
    # ========================================================

    commands = build_commands()

    walk_rows, walk_summary = (
        eval_angle_walk(
            run_dir,
            commands,
            seeds,
            args.seconds,
            standard_env_kwargs,
        )
    )

    # Existing tag evaluation is retained.
    #
    # walk_params.json intentionally contains only
    # standard Go2WalkEnv kwargs, so the existing
    # hierarchical evaluator remains compatible.
    tag_rows, tag_summary = (
        base.eval_tag_direct(
            run_dir,
            commands,
            seeds,
            args.seconds,
        )
    )

    # ========================================================
    # Result
    # ========================================================

    result = {
        "generation": 11,
        "goal": (
            "angle-specified 360-degree "
            "walking with explicit "
            "direction-error penalty"
        ),
        "source": str(source),
        "walk_summary": walk_summary,
        "tag_summary": tag_summary,
        "walk_rows": walk_rows,
        "tag_rows": tag_rows,
    }

    (
        run_dir
        / "gen11_evaluation.json"
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
        best_dir
        / "best_gen11_record.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Videos
    # ========================================================

    if args.record_videos:

        video_dir = (
            run_dir / "videos"
        )

        video_dir.mkdir(
            exist_ok=True
        )

        # Use the existing recorder.
        # walk_params.json contains compatible
        # standard environment parameters.
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
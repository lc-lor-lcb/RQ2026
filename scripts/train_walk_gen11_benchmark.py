"""
Gen11 performance benchmark.

Purpose:
- Compare Gen10-style environment against Gen11 angle-aware environment.
- Isolate the performance cost of:
    1. Normal Go2WalkEnv
    2. Angle command generation without direction penalty
    3. Full Gen11 direction-error penalty

This is NOT a training run for producing a new model.
It is a throughput diagnostic.

Expected interpretation:
    BASE             -> Gen9/Gen10 baseline
    ANGLE_ONLY       -> cost of angle-aware command handling
    ANGLE_PENALTY    -> cost of full Gen11 reward calculation

Use the same source model and num-envs for all tests.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from roboquest.envs.go2_walk_env import Go2WalkEnv
from scripts.train_walk_gen11 import AngleAwareGo2WalkEnv


# ============================================================
# Angle-only environment
# ============================================================

class AngleOnlyGo2WalkEnv(AngleAwareGo2WalkEnv):
    """
    Gen11 angle-command environment WITHOUT the
    direction-error reward calculation.

    This keeps:
        speed + angle -> vx/vy

    but removes:
        actual angle calculation
        atan2()
        cos()
        direction-error penalty
    """

    def _compute_reward(self, action):
        # Skip AngleAwareGo2WalkEnv._compute_reward()
        # and directly use the original Go2WalkEnv reward.
        return Go2WalkEnv._compute_reward(self, action)


# ============================================================
# Common physics
# ============================================================

COMMON_PHYSICS = {
    "action_scale": 0.6,
    "joint_stiffness": 60.0,
    "joint_damping": 2.0,
    "torque_limits": [
        23.7,
        23.7,
        45.43,
    ],
}


# ============================================================
# Environment builders
# ============================================================

def make_base_env():
    """
    Gen10/Gen9-style baseline.
    """

    return Go2WalkEnv(
        randomize_cmd=True,
        command_mode="uniform",
        command_ranges={
            "vx": [-1.4, 1.4],
            "vy": [-1.2, 1.2],
            "omega": [-0.9, 0.9],
        },
        **COMMON_PHYSICS,
    )


def make_angle_only_env():
    """
    Angle command generation,
    but NO direction-error penalty.
    """

    return AngleOnlyGo2WalkEnv(
        randomize_cmd=True,
        angle_ranges=[-180.0, 180.0],
        speed_ranges=[0.30, 1.40],
        direction_error_weight=0.0,
        direction_error_speed_threshold=0.15,
        actual_speed_threshold=0.05,
        command_ranges={
            "vx": [-1.4, 1.4],
            "vy": [-1.2, 1.2],
            "omega": [-0.9, 0.9],
        },
        **COMMON_PHYSICS,
    )


def make_angle_penalty_env():
    """
    Full Gen11 environment.
    """

    return AngleAwareGo2WalkEnv(
        randomize_cmd=True,
        angle_ranges=[-180.0, 180.0],
        speed_ranges=[0.30, 1.40],
        direction_error_weight=-2.5,
        direction_error_speed_threshold=0.15,
        actual_speed_threshold=0.05,
        command_ranges={
            "vx": [-1.4, 1.4],
            "vy": [-1.2, 1.2],
            "omega": [-0.9, 0.9],
        },
        **COMMON_PHYSICS,
    )


# ============================================================
# Benchmark
# ============================================================

def benchmark(
    name: str,
    source: Path,
    env_factory,
    num_envs: int,
    steps: int,
    device: str,
):
    print()
    print("=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)

    print(f"source   : {source}")
    print(f"num_envs : {num_envs}")
    print(f"steps    : {steps}")
    print(f"device   : {device}")
    print()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    vec_raw = make_vec_env(
        env_factory,
        n_envs=num_envs,
        seed=1000,
    )

    # --------------------------------------------------------
    # Load existing normalization
    # --------------------------------------------------------

    vec_env = VecNormalize.load(
        str(source / "walk_model_vecnorm.pkl"),
        vec_raw,
    )

    vec_env.training = True
    vec_env.norm_reward = True

    # --------------------------------------------------------
    # Load existing model
    # --------------------------------------------------------

    model = PPO.load(
        source / "walk_model",
        env=vec_env,
        device=device,
    )

    print("model loaded")
    print()

    # --------------------------------------------------------
    # Warm-up
    #
    # Avoid measuring initialization overhead.
    # --------------------------------------------------------

    warmup_steps = min(
        2_000,
        max(500, steps // 10),
    )

    print(f"warm-up: {warmup_steps:,} steps")

    model.learn(
        total_timesteps=warmup_steps,
        reset_num_timesteps=False,
        progress_bar=False,
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print()
    print(f"benchmark: {steps:,} steps")
    print("starting...")
    print()

    start = time.perf_counter()

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    it_per_sec = steps / elapsed

    env_steps_per_sec = (
        steps * num_envs / elapsed
    )

    print()
    print("-" * 70)
    print(f"{name} RESULT")
    print("-" * 70)

    print(f"wall time          : {elapsed:.2f} sec")
    print(f"SB3 it/s           : {it_per_sec:.2f}")
    print(f"environment steps/s: {env_steps_per_sec:.2f}")

    vec_env.close()

    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "it_per_sec": it_per_sec,
        "environment_steps_per_sec": env_steps_per_sec,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Gen11 performance bottleneck."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to Gen10 best model.",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    args = parser.parse_args()

    torch.set_num_threads(1)

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"Source model not found: {source}"
        )

    required = [
        "walk_model.zip",
        "walk_model_vecnorm.pkl",
    ]

    for name in required:
        path = source / name

        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    print()
    print("=" * 70)
    print("GEN11 PERFORMANCE DIAGNOSTIC")
    print("=" * 70)
    print()
    print("This benchmark does NOT create a new model.")
    print("It only measures training throughput.")
    print()

    results = []

    # ========================================================
    # Test 1
    # ========================================================

    results.append(
        benchmark(
            name="BASE",
            source=source,
            env_factory=make_base_env,
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
        )
    )

    # ========================================================
    # Test 2
    # ========================================================

    results.append(
        benchmark(
            name="ANGLE_ONLY",
            source=source,
            env_factory=make_angle_only_env,
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
        )
    )

    # ========================================================
    # Test 3
    # ========================================================

    results.append(
        benchmark(
            name="ANGLE_PENALTY",
            source=source,
            env_factory=make_angle_penalty_env,
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    baseline = results[0]["it_per_sec"]

    for result in results:
        ratio = (
            result["it_per_sec"]
            / baseline
            if baseline > 0
            else 0.0
        )

        print(
            f"{result['name']:16s} "
            f"{result['it_per_sec']:8.1f} it/s "
            f"({ratio * 100:6.1f}% of BASE)"
        )

    print()
    print("Interpretation:")
    print()

    base_speed = results[0]["it_per_sec"]
    angle_only_speed = results[1]["it_per_sec"]
    penalty_speed = results[2]["it_per_sec"]

    if angle_only_speed < base_speed * 0.70:
        print(
            "[!] ANGLE_ONLY is already much slower."
        )
        print(
            "    The bottleneck is likely the custom "
            "AngleAware environment."
        )
    else:
        print(
            "[OK] Angle command generation itself "
            "does not appear to be the main bottleneck."
        )

    if penalty_speed < angle_only_speed * 0.70:
        print(
            "[!] Direction-error calculation causes "
            "a large slowdown."
        )
        print(
            "    The _direction_error() / _compute_reward() "
            "path is the main suspect."
        )
    else:
        print(
            "[OK] Direction-error penalty does not appear "
            "to be the dominant bottleneck."
        )

    print()
    print(
        "Run the benchmark with the SAME --num-envs and "
        "--device used for Gen9."
    )


if __name__ == "__main__":
    main()
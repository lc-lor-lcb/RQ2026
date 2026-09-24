"""Benchmark a walk base with a simple geometric flee controller.

This does not train a neural flee policy. It answers a more basic question:
with this walk model and command range, can a reasonable hand-written policy
survive long enough to justify using the walk base for Tier2 flee learning?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from scripts.tier2_common import score_summary, summarize_results


def yaw_from_quat(q: np.ndarray) -> float:
    # MuJoCo quaternion order is w, x, y, z.
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def controller_action(env: Go2TagHierarchicalEnv, speed: float, turn_gain: float) -> np.ndarray:
    robot = env._low_env.robot_xy
    oni = env._oni_xy
    away_world = robot - oni
    dist = float(np.linalg.norm(away_world))
    if dist < 1e-6:
        away_world = np.array([1.0, 0.0])
    else:
        away_world = away_world / dist

    yaw = yaw_from_quat(env.data.qpos[3:7])
    desired_yaw = float(np.arctan2(away_world[1], away_world[0]))
    err = (desired_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
    vx = speed
    # If the safe direction is mostly behind, slow down while turning.
    if abs(err) > 1.2:
        vx *= 0.35
    elif abs(err) > 0.7:
        vx *= 0.65
    omega = float(np.clip(turn_gain * err, -0.8, 0.8))
    return np.array([vx, 0.0, omega], dtype=np.float32)


def evaluate_episode(
    walk_dir: Path,
    seed: int,
    speed: float,
    turn_gain: float,
    oni_speed: float,
    seconds: float,
    video_path: Path | None = None,
    fps: int = 5,
) -> dict:
    env = Go2TagHierarchicalEnv(
        low_level_model_path=str(walk_dir / "walk_model"),
        low_level_vecnorm_path=str(walk_dir / "walk_model_vecnorm.pkl"),
        oni_speed=oni_speed,
        render_mode="rgb_array" if video_path else None,
    )
    max_high_steps = int(seconds / 0.2)
    frames = []
    try:
        env.reset(seed=seed)
        distances = []
        min_distance = float("inf")
        info = {}
        for _ in range(max_high_steps):
            action = controller_action(env, speed=speed, turn_gain=turn_gain)
            _, _, terminated, truncated, info = env.step(action)
            dist = float(info.get("oni_distance", env._oni_distance()))
            distances.append(dist)
            min_distance = min(min_distance, dist)
            if video_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame.copy())
            if terminated or truncated:
                break
        timeout = bool(info.get("TimeLimit.truncated", False)) or len(distances) >= max_high_steps
        tagged = bool(info.get("is_tagged", False))
        row = {
            "seed": seed,
            "survived_seconds": float(info.get("survived_seconds", len(distances) * 0.2)),
            "mean_distance": float(sum(distances) / len(distances)) if distances else 0.0,
            "min_distance": float(min_distance if np.isfinite(min_distance) else 0.0),
            "escaped": bool(timeout and not tagged),
            "tagged": tagged,
            "fell": bool((not timeout) and (not tagged)),
        }
    finally:
        env.close()
    if video_path and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=fps)
        row["video"] = str(video_path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-dir", required=True)
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--turn-gain", type=float, default=1.4)
    parser.add_argument("--oni-speed", type=float, default=0.025)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    walk_dir = Path(args.walk_dir)
    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    rows = []
    out_dir = Path(args.output) if args.output else walk_dir / "flee_controller_benchmark"
    for seed in tqdm(seeds, desc="flee controller", unit="episode"):
        video = out_dir / f"seed{seed}_speed{args.speed}.mp4" if args.record else None
        rows.append(evaluate_episode(
            walk_dir=walk_dir,
            seed=seed,
            speed=args.speed,
            turn_gain=args.turn_gain,
            oni_speed=args.oni_speed,
            seconds=args.seconds,
            video_path=video,
        ))
    summary = summarize_results(rows)
    summary["score"] = score_summary(summary)
    output = {
        "walk_dir": str(walk_dir),
        "speed": args.speed,
        "turn_gain": args.turn_gain,
        "oni_speed": args.oni_speed,
        "seeds": seeds,
        "results": rows,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

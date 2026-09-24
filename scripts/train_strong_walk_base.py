"""Train and select a strong base walking model for Tier1/Tier2.

The produced ``best`` folder contains the standard RoboQuest walk files:

    walk_model.zip
    walk_model_vecnorm.pkl
    walk_params.json

Those files can be copied into a Tier1 or Tier2 candidate folder before
training the high-level flee policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tqdm.auto import tqdm

from roboquest.envs.go2_tag_env import CONTROL_HZ, ONI_QPOS_X, ONI_QPOS_Y, ONI_QVEL_X, ONI_QVEL_Y
from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv, N_LOW_STEPS
from roboquest.envs.go2_walk_env import CONTROL_DT, Go2WalkEnv
from scripts.bootstrap_smooth_walk import train_smooth_walk


WALK_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_0.25": (0.25, 0.0, 0.0),
    "forward_0.4": (0.4, 0.0, 0.0),
}

TAG_COMMANDS = {
    "tag_stand": (0.0, 0.0, 0.0),
    "tag_forward_0.25": (0.25, 0.0, 0.0),
    "tag_forward_0.4": (0.4, 0.0, 0.0),
}

PROFILES = {
    "smoke": dict(seeds=[0], sample_steps=6000, updates=800, ppo_steps=2048, seconds=6, eval_seeds=[100]),
    "standard": dict(seeds=[0, 1], sample_steps=30000, updates=4000, ppo_steps=12288, seconds=12, eval_seeds=[100, 101, 102]),
    "strong": dict(seeds=[0, 1, 2], sample_steps=45000, updates=6000, ppo_steps=50000, seconds=20, eval_seeds=[100, 101, 102, 200, 201]),
}


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or value == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_walk_policy(folder: Path):
    params = json.loads((folder / "walk_params.json").read_text(encoding="utf-8"))
    env_kwargs = params.get("walk_env_kwargs", {})
    base = make_vec_env(lambda: Go2WalkEnv(randomize_cmd=False, **env_kwargs), n_envs=1)
    norm = VecNormalize.load(str(folder / "walk_model_vecnorm.pkl"), base)
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(folder / "walk_model", device="cpu")
    return params, base, norm, model


def evaluate_walk_env(folder: Path, seconds: float, seeds: list[int], progress_bar: bool = True) -> tuple[list[dict], dict]:
    _, base, norm, model = load_walk_policy(folder)
    raw = base.envs[0].unwrapped
    rows = []
    try:
        total = len(WALK_COMMANDS) * len(seeds)
        eval_iter = tqdm(total=total, desc=f"{folder.name} walk eval", unit="case", disable=not progress_bar)
        for command_name, command in WALK_COMMANDS.items():
            for seed in seeds:
                eval_iter.set_postfix(command=command_name, seed=seed)
                raw.set_vel_cmd(*command)
                obs, _ = raw.reset(seed=seed)
                heights = []
                velocities = []
                terminated = False
                steps = round(seconds / CONTROL_DT)
                for step in range(steps):
                    action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
                    obs, _, terminated, truncated, _ = raw.step(action)
                    body_vel = raw._world_to_body(raw.data.qvel[:3])
                    heights.append(float(raw.data.qpos[2]))
                    velocities.append([float(body_vel[0]), float(body_vel[1]), float(raw.data.qvel[5])])
                    if terminated or truncated:
                        break
                values = np.asarray(velocities, dtype=float)
                target = np.asarray(command, dtype=float)
                rows.append({
                    "layer": "walk_env",
                    "command": command_name,
                    "seed": seed,
                    "fell": bool(terminated),
                    "survived_seconds": float((step + 1) * CONTROL_DT),
                    "mean_height": float(np.mean(heights)) if heights else None,
                    "min_height": float(np.min(heights)) if heights else None,
                    "mean_vx": float(np.mean(values[:, 0])) if len(values) else None,
                    "mean_vy": float(np.mean(values[:, 1])) if len(values) else None,
                    "mean_omega": float(np.mean(values[:, 2])) if len(values) else None,
                    "velocity_rmse": float(np.sqrt(np.mean((values - target) ** 2))) if len(values) else None,
                })
                eval_iter.update(1)
        eval_iter.close()
    finally:
        norm.close()
    summary = summarize_rows(rows, seconds)
    return rows, summary


def evaluate_tag_env(folder: Path, seconds: float, seeds: list[int], progress_bar: bool = True) -> tuple[list[dict], dict]:
    rows = []
    max_steps = max(1, round(seconds * CONTROL_HZ / N_LOW_STEPS))
    total = len(TAG_COMMANDS) * len(seeds)
    eval_iter = tqdm(total=total, desc=f"{folder.name} tag eval", unit="case", disable=not progress_bar)
    for command_name, command in TAG_COMMANDS.items():
        for seed in seeds:
            eval_iter.set_postfix(command=command_name, seed=seed)
            env = Go2TagHierarchicalEnv(
                low_level_model_path=str(folder / "walk_model"),
                low_level_vecnorm_path=str(folder / "walk_model_vecnorm.pkl"),
                oni_speed=0.0,
                max_episode_steps=max_steps,
                high_level_command_mode="direct",
            )
            try:
                env.reset(seed=seed)
                # This is a walking-base compatibility check, not a flee test.
                # Keep oni out of the fixed forward path so early termination
                # means walking instability rather than accidentally touching it.
                env.data.qpos[ONI_QPOS_X] = -2.2
                env.data.qpos[ONI_QPOS_Y] = 2.2
                env.data.qvel[ONI_QVEL_X] = 0.0
                env.data.qvel[ONI_QVEL_Y] = 0.0
                terminated = False
                truncated = False
                info = {}
                for _ in range(max_steps):
                    _, _, terminated, truncated, info = env.step(np.asarray(command, dtype=np.float32))
                    if terminated or truncated:
                        break
                rows.append({
                    "layer": "tag_env",
                    "command": command_name,
                    "seed": seed,
                    "fell": bool(terminated and not info.get("is_tagged", False)),
                    "tagged": bool(info.get("is_tagged", False)),
                    "survived_seconds": float(info.get("survived_seconds", 0.0)),
                    "mean_height": None,
                    "min_height": float(env.data.qpos[2]),
                    "mean_vx": None,
                    "mean_vy": None,
                    "mean_omega": None,
                    "velocity_rmse": None,
                })
            finally:
                env.close()
            eval_iter.update(1)
    eval_iter.close()
    summary = summarize_rows(rows, seconds)
    return rows, summary


def summarize_rows(rows: list[dict], target_seconds: float) -> dict:
    if not rows:
        return {"passed": False, "fall_count": 999, "mean_survived_seconds": 0.0}
    fall_count = sum(1 for row in rows if row["fell"])
    tagged_count = sum(1 for row in rows if row.get("tagged"))
    mean_seconds = float(np.mean([row["survived_seconds"] for row in rows]))
    min_seconds = float(np.min([row["survived_seconds"] for row in rows]))
    height_values = [row["min_height"] for row in rows if row.get("min_height") is not None]
    min_height = float(np.min(height_values)) if height_values else None
    passed = (
        fall_count == 0
        and tagged_count == 0
        and min_seconds >= target_seconds * 0.98
        and (min_height is None or min_height >= 0.15)
    )
    return {
        "passed": bool(passed),
        "fall_count": int(fall_count),
        "tagged_count": int(tagged_count),
        "mean_survived_seconds": mean_seconds,
        "min_survived_seconds": min_seconds,
        "min_height": min_height,
    }


def score_candidate(walk_summary: dict, tag_summary: dict) -> float:
    fall_penalty = 1000.0 * (walk_summary["fall_count"] + tag_summary["fall_count"])
    survival = walk_summary["mean_survived_seconds"] + 2.0 * tag_summary["mean_survived_seconds"]
    height = 10.0 * max(0.0, float(walk_summary.get("min_height") or 0.0))
    return survival + height - fall_penalty


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_walk_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("walk_model.zip", "walk_model_vecnorm.pkl", "walk_params.json"):
        shutil.copy2(source / name, destination / name)


def record_video(folder: Path, output: Path, seconds: float) -> str | None:
    try:
        from scripts.record_walk import record
        record(folder, output, seconds=seconds)
        return str(output)
    except Exception as exc:  # pragma: no cover - best-effort artifact
        return f"video_failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds, e.g. 0,1,2")
    parser.add_argument("--eval-seeds", default=None, help="Comma-separated evaluation seeds")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--ppo-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing candidate folders")
    parser.add_argument("--record-videos", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    profile = dict(PROFILES[args.profile])
    seeds = parse_int_list(args.seeds, profile["seeds"])
    eval_seeds = parse_int_list(args.eval_seeds, profile["eval_seeds"])
    sample_steps = args.sample_steps if args.sample_steps is not None else profile["sample_steps"]
    updates = args.updates if args.updates is not None else profile["updates"]
    ppo_steps = args.ppo_steps if args.ppo_steps is not None else profile["ppo_steps"]
    seconds = args.seconds if args.seconds is not None else profile["seconds"]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_{args.profile}"
    run_dir = Path(args.output_root) / run_name
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {
        "profile": args.profile,
        "seeds": seeds,
        "eval_seeds": eval_seeds,
        "sample_steps": sample_steps,
        "updates": updates,
        "ppo_steps": ppo_steps,
        "num_envs": args.num_envs,
        "seconds": seconds,
    })

    records = []
    seed_iter = tqdm(list(enumerate(seeds, start=1)), desc="walk candidates", unit="candidate")
    for index, seed in seed_iter:
        candidate = candidates_dir / f"walk_seed{seed}"
        seed_iter.set_postfix(seed=seed)
        print(f"\n[{index}/{len(seeds)}] candidate={candidate} seed={seed}", flush=True)
        if not args.skip_train and not (candidate / "walk_model.zip").exists():
            train_smooth_walk(
                candidate,
                steps=ppo_steps,
                seed=seed,
                sample_steps=sample_steps,
                updates=updates,
                num_envs=args.num_envs,
                progress_bar=True,
            )
        walk_rows, walk_summary = evaluate_walk_env(candidate, seconds=seconds, seeds=eval_seeds)
        tag_rows, tag_summary = evaluate_tag_env(candidate, seconds=seconds, seeds=eval_seeds)
        all_rows = walk_rows + tag_rows
        score = score_candidate(walk_summary, tag_summary)
        video = None
        if args.record_videos:
            video = record_video(candidate, run_dir / "videos" / f"{candidate.name}_forward.mp4", seconds=min(seconds, 12))
        record_item = {
            "candidate": str(candidate),
            "seed": seed,
            "score": score,
            "passed": bool(walk_summary["passed"] and tag_summary["passed"]),
            "walk_summary": walk_summary,
            "tag_summary": tag_summary,
            "video": video,
        }
        write_json(candidate / "strong_walk_evaluation.json", {"summary": record_item, "rows": all_rows})
        records.append(record_item)
        write_json(run_dir / "ranking.json", sorted(records, key=lambda item: item["score"], reverse=True))
        print(json.dumps(record_item, ensure_ascii=False, indent=2), flush=True)

    ranking = sorted(records, key=lambda item: item["score"], reverse=True)
    write_json(run_dir / "ranking.json", ranking)
    if ranking:
        best_source = Path(ranking[0]["candidate"])
        best_dir = run_dir / "best"
        copy_walk_files(best_source, best_dir)
        write_json(best_dir / "best_walk_record.json", ranking[0])
        print(f"\nBest walk copied to: {best_dir}", flush=True)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate", "seed", "score", "passed",
            "walk_falls", "tag_falls", "tagged_count", "walk_mean_seconds", "tag_mean_seconds",
            "walk_min_height", "tag_min_height", "video",
        ])
        writer.writeheader()
        for item in ranking:
            writer.writerow({
                "candidate": item["candidate"],
                "seed": item["seed"],
                "score": item["score"],
                "passed": item["passed"],
                "walk_falls": item["walk_summary"]["fall_count"],
                "tag_falls": item["tag_summary"]["fall_count"],
                "tagged_count": item["tag_summary"].get("tagged_count", 0),
                "walk_mean_seconds": item["walk_summary"]["mean_survived_seconds"],
                "tag_mean_seconds": item["tag_summary"]["mean_survived_seconds"],
                "walk_min_height": item["walk_summary"]["min_height"],
                "tag_min_height": item["tag_summary"]["min_height"],
                "video": item["video"],
            })


if __name__ == "__main__":
    main()

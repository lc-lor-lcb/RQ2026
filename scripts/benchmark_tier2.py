"""Benchmark Tier2 training throughput on this machine."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from roboquest.envs.go2_tag_strategic_train_env import Go2TagStrategicTrainEnv
from roboquest.utils.reward_utils import FleeRewardConfig
from scripts.tier2_common import append_csv, copy_walk_files, load_config, write_json


def available_device(name: str) -> bool:
    return name == "cpu" or (name == "cuda" and torch.cuda.is_available())


def run_case(cfg: dict, device: str, n_envs: int, case_dir: Path) -> dict:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    copy_walk_files(cfg["walk_source_dir"], case_dir)

    reward_cfg = FleeRewardConfig(**cfg.get("reward", {}))
    strategic_cfg = cfg.get("strategic_reward", {})
    walk_model = str(case_dir / "walk_model")
    walk_vecnorm = str(case_dir / "walk_model_vecnorm.pkl")

    def make_env():
        env = Go2TagStrategicTrainEnv(
            low_level_model_path=walk_model,
            low_level_vecnorm_path=walk_vecnorm,
            flee_config=reward_cfg,
            strategic_config=strategic_cfg,
            oni_speed=float(cfg.get("oni_speed", 0.025)),
        )
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=int(cfg.get("seed", 0)))
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    ppo_cfg = dict(cfg["ppo"])
    ppo_cfg["device"] = device
    model = PPO("MlpPolicy", env, seed=int(cfg.get("seed", 0)), verbose=0, **ppo_cfg)
    started = time.monotonic()
    model.learn(total_timesteps=int(cfg["timesteps"]), progress_bar=False)
    elapsed = time.monotonic() - started
    steps_per_sec = float(cfg["timesteps"]) / max(elapsed, 1e-9)
    env.close()
    return {
        "device": device,
        "n_envs": n_envs,
        "timesteps": int(cfg["timesteps"]),
        "elapsed_seconds": elapsed,
        "steps_per_sec": steps_per_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tier2_benchmark.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "runs/tier2_benchmark"))
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for device in cfg.get("devices", ["cpu"]):
        if not available_device(device):
            print(f"skip device={device}: unavailable")
            continue
        for n_envs in cfg.get("n_envs", [1, 4]):
            print(f"benchmark device={device} n_envs={n_envs}")
            row = run_case(cfg, device, int(n_envs), out / f"{device}_{n_envs}")
            print(f"  steps/sec={row['steps_per_sec']:.1f} elapsed={row['elapsed_seconds']:.1f}s")
            results.append(row)
            append_csv(out / "benchmark.csv", row)
    best = max(results, key=lambda r: r["steps_per_sec"]) if results else None
    write_json(out / "benchmark_results.json", {"results": results, "best": best})
    if best:
        print(f"best: device={best['device']} n_envs={best['n_envs']} steps/sec={best['steps_per_sec']:.1f}")


if __name__ == "__main__":
    main()

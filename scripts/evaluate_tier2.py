"""Evaluate a saved Tier2 candidate in the standard compatible environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from roboquest.utils.reward_utils import FleeRewardConfig
from scripts.tier2_common import resolve_run, score_summary, summarize_results, write_json


def evaluate_candidate(candidate_dir: str | Path, seeds: Iterable[int]) -> list[dict]:
    folder = Path(candidate_dir)
    required = [
        "walk_model.zip",
        "walk_model_vecnorm.pkl",
        "flee_model.zip",
        "flee_model_vecnorm.pkl",
        "flee_params.json",
    ]
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    params = json.loads((folder / "flee_params.json").read_text(encoding="utf-8"))

    def factory():
        return Go2TagHierarchicalEnv(
            low_level_model_path=str(folder / "walk_model"),
            low_level_vecnorm_path=str(folder / "walk_model_vecnorm.pkl"),
            flee_config=FleeRewardConfig(**params["reward_config"]),
            oni_speed=float(params.get("oni_speed", 0.025)),
            high_level_command_mode=params.get("high_level_command_mode", "direct"),
            high_level_command_ranges=params.get("high_level_command_ranges"),
            high_level_observation_mode=params.get("high_level_observation_mode", "standard"),
        )

    base = make_vec_env(factory, n_envs=1)
    try:
        env = VecNormalize.load(str(folder / "flee_model_vecnorm.pkl"), base)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(folder / "flee_model"))
        rows = []
        for seed in seeds:
            env.seed(int(seed))
            obs = env.reset()
            distances = []
            min_distance = float("inf")
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                info = infos[0]
                dist = float(info.get("oni_distance", 0.0))
                distances.append(dist)
                min_distance = min(min_distance, dist)
                if dones[0]:
                    timeout = bool(info.get("TimeLimit.truncated", False))
                    tagged = bool(info.get("is_tagged", False))
                    rows.append(
                        dict(
                            seed=int(seed),
                            survived_seconds=float(info.get("survived_seconds", 0.0)),
                            mean_distance=sum(distances) / len(distances),
                            min_distance=min_distance,
                            escaped=timeout,
                            tagged=tagged,
                            fell=not timeout and not tagged,
                        )
                    )
                    break
        return rows
    finally:
        base.close()


def candidate_path(run_dir: Path, candidate: str) -> Path:
    if candidate == "best":
        best = run_dir / "best" / "candidate_dir.txt"
        if best.is_file():
            return Path(best.read_text(encoding="utf-8").strip())
    p = Path(candidate)
    if p.is_dir():
        return p
    p = run_dir / "candidates" / candidate
    if p.is_dir():
        return p
    raise FileNotFoundError(f"candidate not found: {candidate}")


def parse_seeds(value: str, run_dir: Path) -> list[int]:
    if value == "coarse":
        return [100, 101, 102]
    if value == "medium":
        return list(range(100, 110))
    if value == "final":
        return list(range(100, 130))
    return [int(x) for x in value.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", default="best")
    parser.add_argument("--seeds", default="final")
    args = parser.parse_args()
    run_dir = resolve_run(args.run)
    cand = candidate_path(run_dir, args.candidate)
    seeds = parse_seeds(args.seeds, run_dir)
    results = evaluate_candidate(cand, seeds)
    summary = summarize_results(results)
    summary["score"] = score_summary(summary)
    output = {"candidate_dir": str(cand), "seeds": seeds, "results": results, "summary": summary}
    out_path = run_dir / "eval" / f"{cand.name}_{args.seeds}.json"
    write_json(out_path, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

"""Diagnose why flee models terminate immediately.

Run from the project root:

    python -m scripts.diagnose_flee_pipeline --candidate runs/tier2/latest

This does not train. It checks whether the bundled walk model can stand/walk in
the arena and whether a saved flee policy is producing unsafe high-level
commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from scripts.evaluate_tier2 import candidate_path
from scripts.tier2_common import resolve_run


def resolve_candidate(path: str, candidate: str) -> Path:
    p = Path(path)
    if p.exists() and (p / "flee_model.zip").exists():
        return p
    run_dir = resolve_run(path)
    return candidate_path(run_dir, candidate)


def run_fixed_action(candidate_dir: Path, action, mode: str, seed: int = 100, max_high_steps: int = 60):
    params_path = candidate_dir / "flee_params.json"
    oni_speed = 0.0
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        oni_speed = float(params.get("oni_speed", 0.025))
    env = Go2TagHierarchicalEnv(
        low_level_model_path=str(candidate_dir / "walk_model"),
        low_level_vecnorm_path=str(candidate_dir / "walk_model_vecnorm.pkl"),
        oni_speed=oni_speed,
        high_level_command_mode=mode,
    )
    try:
        obs, _ = env.reset(seed=seed)
        infos = []
        for _ in range(max_high_steps):
            obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            infos.append(info)
            if terminated or truncated:
                break
        final = infos[-1]
        return {
            "mode": mode,
            "action": list(action),
            "high_steps": len(infos),
            "survived_seconds": final.get("survived_seconds"),
            "tagged": final.get("is_tagged"),
            "height": float(env.data.qpos[2]),
            "distance": final.get("oni_distance"),
            "velocity_command": final.get("velocity_command"),
            "raw_high_action": final.get("raw_high_action"),
            "terminated_as": "tagged" if final.get("is_tagged") else ("fell_or_other" if len(infos) < max_high_steps else "still_running"),
        }
    finally:
        env.close()


def run_saved_policy(candidate_dir: Path, seed: int = 100, max_high_steps: int = 60):
    required = ["flee_model.zip", "flee_model_vecnorm.pkl", "flee_params.json"]
    missing = [name for name in required if not (candidate_dir / name).exists()]
    if missing:
        return {"saved_policy_error": f"missing {missing}"}

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from roboquest.utils.reward_utils import FleeRewardConfig

    params = json.loads((candidate_dir / "flee_params.json").read_text(encoding="utf-8"))

    def factory():
        return Go2TagHierarchicalEnv(
            low_level_model_path=str(candidate_dir / "walk_model"),
            low_level_vecnorm_path=str(candidate_dir / "walk_model_vecnorm.pkl"),
            flee_config=FleeRewardConfig(**params["reward_config"]),
            oni_speed=float(params.get("oni_speed", 0.025)),
        )

    base = make_vec_env(factory, n_envs=1)
    try:
        env = VecNormalize.load(str(candidate_dir / "flee_model_vecnorm.pkl"), base)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(candidate_dir / "flee_model"))
        env.seed(seed)
        obs = env.reset()
        rows = []
        for _ in range(max_high_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            info = infos[0]
            rows.append({
                "action": np.asarray(action).reshape(-1).astype(float).tolist(),
                "survived_seconds": info.get("survived_seconds"),
                "tagged": info.get("is_tagged"),
                "velocity_command": info.get("velocity_command"),
                "height_unknown_in_vecenv": None,
            })
            if dones[0]:
                break
        return {"saved_policy_steps": len(rows), "saved_policy_trace": rows[:10], "final": rows[-1] if rows else None}
    finally:
        base.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="best")
    parser.add_argument("--run", default="runs/tier2/latest")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    candidate_dir = resolve_candidate(args.run, args.candidate)
    print(f"candidate_dir: {candidate_dir}")
    for name in ["walk_model.zip", "walk_model_vecnorm.pkl", "walk_params.json", "flee_model.zip", "flee_model_vecnorm.pkl", "flee_params.json"]:
        print(f"{name}: {(candidate_dir / name).exists()}")

    tests = [
        ("direct", [0.0, 0.0, 0.0]),
        ("direct", [0.4, 0.0, 0.0]),
        ("direct", [1.0, 0.0, 0.0]),
        ("safe_forward", [0.0, 0.0, 0.0]),
        ("safe_forward", [1.0, 0.0, 0.0]),
    ]
    print("\nfixed action tests:")
    for mode, action in tests:
        print(json.dumps(run_fixed_action(candidate_dir, action, mode, seed=args.seed), ensure_ascii=False, indent=2))

    print("\nsaved policy test:")
    print(json.dumps(run_saved_policy(candidate_dir, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

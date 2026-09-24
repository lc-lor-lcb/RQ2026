"""Record videos for Tier2 candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from roboquest.utils.reward_utils import FleeRewardConfig
from scripts.evaluate_tier2 import candidate_path, parse_seeds
from scripts.tier2_common import resolve_run


def record_one(
    candidate_dir: Path,
    output_path: Path,
    seed: int,
    fps: int = 5,
    freeze_tail_seconds: float = 2.0,
) -> dict:
    params = json.loads((candidate_dir / "flee_params.json").read_text(encoding="utf-8"))
    raw_env_holder = {}

    def factory():
        env = Go2TagHierarchicalEnv(
            low_level_model_path=str(candidate_dir / "walk_model"),
            low_level_vecnorm_path=str(candidate_dir / "walk_model_vecnorm.pkl"),
            flee_config=FleeRewardConfig(**params["reward_config"]),
            oni_speed=float(params.get("oni_speed", 0.025)),
            render_mode="rgb_array",
            high_level_command_mode=params.get("high_level_command_mode", "direct"),
            high_level_command_ranges=params.get("high_level_command_ranges"),
            high_level_observation_mode=params.get("high_level_observation_mode", "standard"),
        )
        raw_env_holder["env"] = env
        return env

    base = make_vec_env(factory, n_envs=1)
    frames = []
    try:
        env = VecNormalize.load(str(candidate_dir / "flee_model_vecnorm.pkl"), base)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(candidate_dir / "flee_model"))
        env.seed(seed)
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            frame = raw_env_holder["env"].render()
            if frame is not None:
                frames.append(frame)
            if dones[0]:
                info = infos[0]
                break
    finally:
        base.close()

    if frames and freeze_tail_seconds > 0:
        frames.extend([frames[-1]] * int(max(0, freeze_tail_seconds) * fps))

    escaped = bool(info.get("TimeLimit.truncated", False))
    tagged = bool(info.get("is_tagged", False))
    fell = (not escaped) and (not tagged)
    reason = "escaped" if escaped else ("tagged" if tagged else "fell")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return {
        "seed": seed,
        "video": str(output_path),
        "frames": len(frames),
        "survived_seconds": float(info.get("survived_seconds", 0.0)),
        "tagged": tagged,
        "escaped": escaped,
        "fell": fell,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", default="best")
    parser.add_argument("--seeds", default="100 101 102")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--freeze-tail-seconds", type=float, default=2.0)
    args = parser.parse_args()
    run_dir = resolve_run(args.run)
    cand = candidate_path(run_dir, args.candidate)
    seeds = parse_seeds(args.seeds, run_dir)
    for seed in seeds:
        out = run_dir / "videos" / f"{cand.name}_seed{seed}.mp4"
        result = record_one(cand, out, seed, fps=args.fps, freeze_tail_seconds=args.freeze_tail_seconds)
        print(result)


if __name__ == "__main__":
    main()

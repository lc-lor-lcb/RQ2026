"""Collect geometric teacher data for Tier2 imitation experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from scripts.geometric_flee_teacher import teacher_action_from_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk_model", default="models/pretrained/smooth_walk/walk_model")
    parser.add_argument("--walk_vecnorm", default="models/pretrained/smooth_walk/walk_model_vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default="runs/tier2_teacher/teacher_data.npz")
    args = parser.parse_args()
    env = Go2TagHierarchicalEnv(args.walk_model, low_level_vecnorm_path=args.walk_vecnorm)
    obs_rows = []
    action_rows = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=1000 + ep)
            for _ in range(env.max_episode_steps):
                action = teacher_action_from_state(env._low_env.robot_xy, env._oni_xy)
                obs_rows.append(obs)
                action_rows.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.asarray(obs_rows, dtype=np.float32), actions=np.asarray(action_rows, dtype=np.float32))
    print(f"saved {len(obs_rows)} samples: {out}")


if __name__ == "__main__":
    main()

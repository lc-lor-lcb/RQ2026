"""
ステップ1: Go2 歩行の事前学習スクリプト

速度コマンド追従（vx, vy, omega）を学習。
このモデルを train_flee.py の低レベルポリシーとして使用する。

使い方（Colab）:
    python scripts/train_walk.py --timesteps 500000 --save_path /tmp/walk_model
"""
import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roboquest.envs.go2_walk_env import Go2WalkEnv
from roboquest.utils.reward_utils import WalkRewardConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="総学習ステップ数（推奨: 500000~2000000）")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="並列環境数")
    parser.add_argument("--save_path", type=str, default="/tmp/walk_model",
                        help="モデルの保存パス")
    parser.add_argument("--log_dir", type=str, default="/tmp/walk_logs",
                        help="ログ保存ディレクトリ")
    # 報酬パラメータ
    parser.add_argument("--lin_vel_weight", type=float, default=1.0)
    parser.add_argument("--ang_vel_weight", type=float, default=0.5)
    parser.add_argument("--orientation_weight", type=float, default=-1.0)
    parser.add_argument("--torques_weight", type=float, default=-2.5e-5)
    parser.add_argument("--action_rate_weight", type=float, default=-0.05)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    reward_cfg = WalkRewardConfig(
        lin_vel_weight=args.lin_vel_weight,
        ang_vel_weight=args.ang_vel_weight,
        orientation_weight=args.orientation_weight,
        torques_weight=args.torques_weight,
        action_rate_weight=args.action_rate_weight,
    )

    def make_env():
        env = Go2WalkEnv(reward_config=reward_cfg, max_episode_steps=500)
        env = Monitor(env, args.log_dir)
        return env

    print(f"並列環境数: {args.n_envs}")
    print(f"総学習ステップ: {args.timesteps:,}")
    print(f"保存先: {args.save_path}.zip")

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // args.n_envs,
        save_path=args.log_dir,
        name_prefix="walk_ckpt",
    )

    print("\n🏃 歩行学習を開始します...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # モデルと正規化パラメータを保存
    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")
    print(f"\n✅ 学習完了！モデルを保存: {args.save_path}.zip")


if __name__ == "__main__":
    main()

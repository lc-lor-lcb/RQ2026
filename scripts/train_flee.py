"""
ステップ2: 鬼ごっこ逃げポリシーの学習（階層学習版）

事前学習済みの歩行モデルを低レベルポリシーとして固定し、
高レベルの「どの方向・速さで逃げるか」を強化学習する。

使い方（Colab）:
    # まず歩行学習を実行
    python scripts/train_walk.py --timesteps 500000

    # 次に鬼ごっこ学習
    python scripts/train_flee.py \
        --walk_model /tmp/walk_model \
        --timesteps 300000 \
        --save_path /tmp/flee_model
"""
import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from roboquest.utils.reward_utils import FleeRewardConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk_model", type=str, default="/tmp/walk_model",
                        help="事前学習済み歩行モデルのパス（.zip なし）")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="総学習ステップ数（高レベルステップ数）")
    parser.add_argument("--n_envs", type=int, default=2,
                        help="並列環境数（各環境が低レベルポリシーを持つためメモリ注意）")
    parser.add_argument("--save_path", type=str, default="/tmp/flee_model",
                        help="モデルの保存パス")
    parser.add_argument("--log_dir", type=str, default="/tmp/flee_logs",
                        help="ログ保存ディレクトリ")
    parser.add_argument("--oni_speed", type=float, default=0.025,
                        help="鬼の移動速度（m/物理ステップ）")
    # 報酬パラメータ
    parser.add_argument("--survival_weight", type=float, default=0.5,
                        help="毎ステップの生存ボーナス")
    parser.add_argument("--distance_weight", type=float, default=1.0,
                        help="鬼との距離に比例した報酬")
    parser.add_argument("--tag_penalty", type=float, default=50.0,
                        help="タグされた時のペナルティ")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    walk_model_path = args.walk_model
    if not os.path.exists(walk_model_path + ".zip"):
        print(f"❌ 歩行モデルが見つかりません: {walk_model_path}.zip")
        print("   先に train_walk.py を実行してください。")
        sys.exit(1)

    flee_cfg = FleeRewardConfig(
        survival_weight=args.survival_weight,
        distance_weight=args.distance_weight,
        tag_penalty=args.tag_penalty,
    )

    def make_env():
        env = Go2TagHierarchicalEnv(
            low_level_model_path=walk_model_path,
            flee_config=flee_cfg,
            oni_speed=args.oni_speed,
        )
        env = Monitor(env, args.log_dir)
        return env

    print(f"歩行モデル: {walk_model_path}.zip")
    print(f"鬼の速度: {args.oni_speed} m/step")
    print(f"総学習ステップ（高レベル）: {args.timesteps:,}")

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000 // args.n_envs,
        save_path=args.log_dir,
        name_prefix="flee_ckpt",
    )

    print("\n🐾 鬼ごっこ逃げ学習を開始します...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")
    print(f"\n✅ 学習完了！モデルを保存: {args.save_path}.zip")


if __name__ == "__main__":
    main()

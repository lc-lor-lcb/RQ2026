"""
鬼ロボットの強化学習訓練スクリプト（運営用）

3段階カリキュラム学習:
  Phase 1: ランダムに動く逃げロボットを追う
  Phase 2: 訓練中の逃げロボットを追う
  Phase 3: 最終版の逃げロボットモデルを追う

使用例:
  python scripts/train_oni.py
  python scripts/train_oni.py --flee-model models/flee_robot/flee_final.zip
"""
import argparse
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from roboquest.envs.go2_walk_env import Go2WalkEnv, STANDING_POS, ACTION_SCALE, KP, KD
from roboquest.utils.reward_utils import RewardConfig

import mujoco


class Go2OniEnv(Go2WalkEnv):
    """鬼ロボット用環境: 逃げロボット（固定またはランダム）を追いかける。"""

    def __init__(
        self,
        flee_model=None,
        flee_speed: float = 0.025,
        max_episode_steps: int = 1000,
    ):
        # 鬼の報酬（距離を縮めるほど高い）
        cfg = RewardConfig(
            distance_weight=-2.0,  # 距離が小さいほど高報酬
            survival_weight=0.1,
            control_weight=0.02,
            forward_weight=0.0,
        )
        super().__init__(reward_config=cfg, max_episode_steps=max_episode_steps)

        self.flee_model = flee_model
        self.flee_speed = flee_speed
        self._flee_pos = np.zeros(2, dtype=np.float64)

        obs_dim = self.observation_space.shape[0] + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(2.0, 4.0)
        self._flee_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        return self._get_full_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 逃げロボットを前方に移動（ランダム方向またはモデル推論）
        self._move_flee()

        # 追いかける報酬: 近いほど高い
        dist = float(np.linalg.norm(self.robot_xy - self._flee_pos))
        r_chase = -self.reward_config.distance_weight * dist * 0.01
        reward += r_chase

        # タグ判定（捕まえた時ボーナス）
        if dist < 0.5:
            reward += 20.0
            terminated = True

        info["flee_distance"] = dist
        return self._get_full_obs(obs), reward, terminated, truncated, info

    def _move_flee(self):
        if self.flee_model is not None:
            pass  # TODO: 実際のモデル推論
        else:
            # ランダムウォーク
            angle = self.np_random.uniform(0, 2 * np.pi)
            self._flee_pos += self.flee_speed * np.array([np.cos(angle), np.sin(angle)])
            self._flee_pos = np.clip(self._flee_pos, -5.0, 5.0)

    def _get_full_obs(self, base_obs):
        rel = self._flee_pos - self.robot_xy
        dist = np.linalg.norm(rel).reshape(1)
        return np.concatenate([base_obs, rel, dist]).astype(np.float32)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    def make_env(rank=0):
        def _init():
            env = Go2OniEnv()
            env = Monitor(env)
            return env
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        policy_kwargs={"net_arch": [256, 256, 128]},
        verbose=1,
    )

    print(f"\n=== 鬼ロボットの学習開始（運営用）===")
    print(f"学習ステップ数: {args.timesteps:,}\n")

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("\n学習を中断しました。")

    model_path = os.path.join(args.save_dir, "oni_final")
    model.save(model_path)
    env.save(os.path.join(args.save_dir, "vec_normalize_oni.pkl"))
    print(f"\n✅ 鬼モデル保存完了: {model_path}.zip")


def main():
    parser = argparse.ArgumentParser(description="鬼ロボット学習スクリプト")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-dir", default="models/oni_robot")
    parser.add_argument("--flee-model", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

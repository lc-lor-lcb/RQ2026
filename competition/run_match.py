"""
大会試合実行スクリプト

チームの逃げロボットモデル vs 運営の鬼モデルで試合を行い結果を返す。

使用例:
  python competition/run_match.py \\
    --flee models/flee_robot/flee_final.zip \\
    --oni   models/oni_robot/oni_final.zip

  # トーナメント（全チーム一括）
  python competition/run_match.py \\
    --tournament \\
    --teams-dir /content/drive/MyDrive/RoboQuest2026 \\
    --oni models/oni_robot/oni_final.zip
"""
import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from roboquest.envs.go2_tag_env import Go2TagEnv
from roboquest.utils.visualization import record_episode, show_video_colab


def load_model(model_path: str, normalize_path: Optional[str] = None):
    """SB3 モデルを読み込む。VecNormalize があれば一緒に読み込む。"""
    model = PPO.load(model_path)
    return model


def run_single_match(
    flee_model_path: str,
    oni_model_path: Optional[str] = None,
    n_episodes: int = 10,
    max_steps: int = 1000,
) -> Dict:
    """1対1の試合を実行してスコアを返す。

    Args:
        flee_model_path: 逃げロボットモデル (.zip)
        oni_model_path: 鬼モデル (.zip)。None の場合はスクリプト鬼。
        n_episodes: 試合エピソード数
        max_steps: 1エピソードの最大ステップ数

    Returns:
        dict: {survival_time, caught_count, avg_distance, score, episodes}
    """
    flee_model = load_model(flee_model_path)
    oni_model = load_model(oni_model_path) if oni_model_path else None

    survival_times = []
    caught_flags = []
    avg_distances = []

    env = Go2TagEnv(max_episode_steps=max_steps)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_distance = 0.0
        step = 0

        for step in range(max_steps):
            action, _ = flee_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_distance += info.get("oni_distance", 0.0)

            if terminated or truncated:
                break

        survival_times.append(step + 1)
        caught_flags.append(int(info.get("is_tagged", 0)))
        avg_distances.append(total_distance / (step + 1))

    env.close()

    score = (
        np.mean(survival_times) * 0.5
        + np.mean(avg_distances) * 100
        - np.sum(caught_flags) * 50
    )

    return {
        "survival_time": float(np.mean(survival_times)),
        "caught_count": int(np.sum(caught_flags)),
        "avg_distance": float(np.mean(avg_distances)),
        "score": float(score),
        "episodes": n_episodes,
    }


def run_tournament(
    teams_dir: str,
    oni_model_path: Optional[str] = None,
    n_episodes: int = 5,
) -> pd.DataFrame:
    """全チームのモデルを読み込んでトーナメントを実行する。

    Args:
        teams_dir: 各チームのモデルが入ったディレクトリ
                   例: teams_dir/チームA/flee_final.zip
        oni_model_path: 鬼モデル
        n_episodes: チームごとの試合エピソード数

    Returns:
        DataFrame: チームランキング表
    """
    # チームモデルを探す
    model_files = glob.glob(os.path.join(teams_dir, "**", "flee_final.zip"), recursive=True)
    if not model_files:
        model_files = glob.glob(os.path.join(teams_dir, "**", "best_model.zip"), recursive=True)

    if not model_files:
        print(f"チームモデルが見つかりません: {teams_dir}")
        return pd.DataFrame()

    results = []
    for model_path in sorted(model_files):
        team_name = os.path.basename(os.path.dirname(model_path))
        print(f"試合中: {team_name} ...", end=" ", flush=True)
        try:
            result = run_single_match(model_path, oni_model_path, n_episodes=n_episodes)
            result["team"] = team_name
            results.append(result)
            print(f"スコア: {result['score']:.1f}")
        except Exception as e:
            print(f"エラー: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "順位"
    return df[["team", "score", "survival_time", "avg_distance", "caught_count"]]


def record_match(
    flee_model_path: str,
    oni_model_path: Optional[str] = None,
    n_steps: int = 1000,
    show_inline: bool = True,
) -> List:
    """試合を録画して Colab でインライン表示する。"""
    flee_model = load_model(flee_model_path)
    env = Go2TagEnv(render_mode="rgb_array")
    frames = record_episode(env, model=flee_model, n_steps=n_steps)
    env.close()

    if show_inline:
        show_video_colab(frames)
    return frames


def main():
    parser = argparse.ArgumentParser(description="鬼ごっこ大会スクリプト")
    parser.add_argument("--flee", required=False, help="逃げモデルパス (.zip)")
    parser.add_argument("--oni", default=None, help="鬼モデルパス (.zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--tournament", action="store_true")
    parser.add_argument("--teams-dir", default="models/teams")
    parser.add_argument("--record", action="store_true", help="試合を録画して表示")
    args = parser.parse_args()

    if args.tournament:
        df = run_tournament(args.teams_dir, args.oni, args.episodes)
        print("\n=== 大会結果 ===")
        print(df.to_string())
        df.to_csv("tournament_results.csv", index=True, encoding="utf-8-sig")
        print("\n結果を tournament_results.csv に保存しました")
    elif args.flee:
        if args.record:
            record_match(args.flee, args.oni)
        else:
            result = run_single_match(args.flee, args.oni, args.episodes)
            print("\n=== 試合結果 ===")
            for k, v in result.items():
                print(f"  {k}: {v}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

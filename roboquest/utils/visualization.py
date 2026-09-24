"""
可視化ユーティリティ（Colab インライン動画・学習曲線）

MuJoCo 公式チュートリアル準拠の方式で動画を Colab 内に表示する。
参考: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb
"""
import os

# Google Colab などヘッドレス環境では EGL オフスクリーンレンダリングを使用する。
# MUJOCO_GL は mujoco を import する前に設定しなければならない。
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from typing import Optional, List

import mujoco
import numpy as np
import matplotlib.pyplot as plt


def record_episode(
    env,
    model=None,
    n_steps: int = 500,
    fps: int = 30,
    camera_name: Optional[str] = None,
) -> List[np.ndarray]:
    """1エピソードを実行してフレームリストを返す。

    Args:
        env: Go2WalkEnv または Go2TagEnv のインスタンス
        model: SB3 学習済みモデル（None の場合はランダム行動）
        n_steps: 録画ステップ数
        fps: 再生フレームレート（表示用、実際の物理には影響しない）
        camera_name: カメラ名（None でデフォルト）

    Returns:
        RGB フレームのリスト (各フレーム: ndarray shape (H, W, 3))
    """
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames = []

    obs, _ = env.reset()
    for _ in range(n_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        renderer.update_scene(env.data)
        frames.append(renderer.render().copy())

        if terminated or truncated:
            obs, _ = env.reset()

    renderer.close()
    return frames


def show_video_colab(frames: List[np.ndarray], fps: int = 30) -> None:
    """Colab でフレームリストをインライン動画として表示する。

    mediapy がインストールされている場合はそちらを使用。
    フォールバックとして matplotlib アニメーションを使用。
    """
    try:
        import mediapy as media
        media.show_video(frames, fps=fps)
    except ImportError:
        _show_video_matplotlib(frames, fps)


def _show_video_matplotlib(frames: List[np.ndarray], fps: int = 30) -> None:
    """matplotlib + IPython で動画を表示（mediapy なし版）"""
    from matplotlib import animation
    from IPython.display import HTML, display

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_array(frame)
        return [img]

    interval = 1000 / fps
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )
    plt.close(fig)
    display(HTML(ani.to_jshtml()))


def plot_training_curve(
    log_path: str,
    title: str = "学習曲線",
    smooth_window: int = 50,
) -> None:
    """SB3 の monitor.csv から学習曲線を描画する。

    Args:
        log_path: Monitor ログファイル (.csv) またはディレクトリのパス
        title: グラフタイトル
        smooth_window: 移動平均のウィンドウサイズ
    """
    import pandas as pd
    import glob

    # ディレクトリが指定された場合は monitor.csv を探す
    if os.path.isdir(log_path):
        csvs = glob.glob(os.path.join(log_path, "**", "monitor.csv"), recursive=True)
        if not csvs:
            print(f"ログが見つかりません: {log_path}")
            return
        log_path = csvs[0]

    df = pd.read_csv(log_path, skiprows=1)
    rewards = df["r"].values
    steps = np.cumsum(df["l"].values)

    # 移動平均
    if len(rewards) >= smooth_window:
        smooth = np.convolve(rewards, np.ones(smooth_window) / smooth_window, mode="valid")
        steps_smooth = steps[smooth_window - 1:]
    else:
        smooth = rewards
        steps_smooth = steps

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, rewards, alpha=0.3, color="steelblue", label="エピソード報酬")
    ax.plot(steps_smooth, smooth, color="steelblue", linewidth=2, label=f"移動平均(n={smooth_window})")
    ax.set_xlabel("学習ステップ数")
    ax.set_ylabel("エピソード報酬")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

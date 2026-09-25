"""前進/後退/横歩きに対応した簡易手本歩容ジェネレータ。

既存の bootstrap_smooth_walk.example_action は前後方向(x)の2リンクIKのみで、
横方向(y)の動きは扱えない(股関節の外転/内転が常に0固定のため)。
本モジュールはこれを拡張し、横方向の脚配置を股関節角度(外転/内転)で近似する。

注意: これは完全な運動学的最適解ではなく、PPOが探索を始めるための
「そこそこ歩ける」初期方策を作るための手本(ウォームスタート用)。
最終的な質はPPOの本学習(train_policy)側で仕上がる想定。
挙動は必ず record_walk.py 等で目視確認してから本学習に進むこと。
"""
from __future__ import annotations

import numpy as np

from roboquest.envs.go2_walk_env import STANDING_POS

LEG_LENGTH = 0.213  # 大腿・下腿リンク長 (bootstrap_smooth_walk.example_action と同じ値)


def omni_example_action(
    time: float,
    vx: float,
    vy: float = 0.0,
    frequency: float = 1.5,
    height: float = 0.31,
    lift: float = 0.05,
) -> np.ndarray:
    """前後(vx)・左右(vy)の速度指令に対する12関節の目標角度(手本)を返す。

    vx のみ: 既存 example_action と同じ前後トロット歩容(符号反転で後退にも対応)。
    vy を含む場合: 同じ遊脚タイミングで、股関節の外転/内転を使って
    足先を左右にもオフセットさせる(簡易近似)。
    """
    speed = float(np.hypot(vx, vy))
    if speed < 0.05:
        x, y = np.zeros(4), np.zeros(4)
        z = np.full(4, -height)
    else:
        # 対角トロットの遊脚タイミングは既存の手本と同一
        phase = (time * frequency + np.array([0.0, 0.5, 0.5, 0.0])) % 1
        swing = phase < 0.5
        u = np.where(swing, phase * 2, (phase - 0.5) * 2)
        extent_x = 1.6 * vx / (frequency * 4)
        extent_y = 1.6 * vy / (frequency * 4)
        x = np.where(swing, -extent_x * np.cos(np.pi * u), extent_x * (1 - 2 * u))
        y = np.where(swing, -extent_y * np.cos(np.pi * u), extent_y * (1 - 2 * u))
        z = -height + np.where(swing, lift * np.sin(np.pi * u), 0.0)

    # 股関節(外転/内転)で足先を横方向オフセット y に近づける簡易近似。
    # 脚全体の長さ(hipからの距離)は変えず、z軸まわりではなくx軸まわりに
    # 角度をつけることで y オフセットを作る想定。
    hip = np.arctan2(y, -z)
    leg_reach = np.sqrt(z * z + y * y)  # 外転後もIKに渡す脚の伸び量は一定に保つ

    knee = -np.arccos(np.clip((x * x + leg_reach * leg_reach - 2 * LEG_LENGTH**2) / (2 * LEG_LENGTH**2), -1, 1))
    thigh = np.arctan2(-x, -leg_reach) - knee / 2

    target = np.stack([hip, thigh, knee], axis=1).reshape(-1)
    return np.clip((target - STANDING_POS) / 0.6, -1, 1).astype(np.float32)
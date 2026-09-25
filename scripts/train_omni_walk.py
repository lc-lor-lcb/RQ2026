"""現在の歩行モデルに、後退・横歩き・その場旋回を追加で学習させるスクリプト (v3)。

v2からの変更点(v2の学習結果が「全指令に対して立ち止まる」方向へ退行したため):
  - v2で使った「3軸を広い範囲で同時に組み合わせる」段階と、独自に重みを
    調整した報酬(OMNI_REWARD_COMBINED)を撤回した。これはこのリポジトリの
    既存プロファイル(standard/strong/race/escape/sprint_turn)のどれにも
    無いやり方で、実績がなかった。
  - 代わりに、既存プロファイルと同じやり方(範囲を狭く始めて段階的に広げる、
    新しい軸は1つずつ足す、報酬は AGILITY_REWARD のまま変えない)を踏襲した
    カリキュラムに作り直した。詳細は configs/walk_omni_v1.yaml を参照。
  - 報酬設定は train_speed_walk_base.AGILITY_REWARD をそのまま import して
    使う(独自の重み調整はしない)。

使い方:
    新規学習:
        python -m scripts.train_omni_walk --config configs/walk_omni_v1.yaml
    途中まで進んだrunを再開(未完了の段階だけ実行):
        python -m scripts.train_omni_walk --config configs/walk_omni_v1.yaml \
            --resume runs/walk_base/20260925_121832_omni

前提:
  - リポジトリのルート(scripts/ と roboquest/ が同じ階層にあるフォルダ)から
    「python -m scripts.train_omni_walk ...」の形式で実行すること。
  - scripts/omni_gait_teacher.py が scripts/ 配下に配置されていること。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tqdm.auto import tqdm

from roboquest.envs.go2_walk_env import CONTROL_DT, Go2WalkEnv
from scripts.notebook_workflow import train_policy
from scripts.omni_gait_teacher import omni_example_action
from scripts.record_walk import record as record_walk
from scripts.tier2_common import load_config
from scripts.train_speed_walk_base import (
    AGILITY_REWARD,
    REQUIRED,
    SPEED_PPO_OVERRIDES,
    copy_base,
    eval_tag_direct,
    eval_walk,
)

ENV_PHYSICS = {
    "action_scale": 0.6,
    "joint_stiffness": 60.0,
    "joint_damping": 2.0,
    "torque_limits": [23.7, 23.7, 45.43],
}

# 各段階直後に挟む簡易評価(1seedだけの短い確認用。本判定はカリキュラム完走後の
# eval_walk/eval_tag_directで行う)。
QUICK_EVAL_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_1.0": (1.0, 0.0, 0.0),
    "reverse_slow": (-0.4, 0.0, 0.0),
    "strafe_left": (0.0, 0.4, 0.0),
    "pivot_left": (0.05, 0.0, 0.8),
    "turn_left_fast": (1.0, 0.0, 0.75),
}


def omni_imitation_warmstart(
    folder: Path,
    env_kwargs: dict,
    sample_steps: int,
    updates: int,
    seed: int,
) -> None:
    """vx・vy 両方に対応した手本で、PPO本学習前のウォームスタートを行う。

    train_speed_walk_base.speed_imitation_warmstart とほぼ同じ構造。
    違いは teacher.set_vel_cmd(vx, vy, omega) の vy を実際に手本生成へ渡す点
    (既存実装は vy を常に 0 固定していた)。
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    base = make_vec_env(lambda: Go2WalkEnv(randomize_cmd=False, reward_config=AGILITY_REWARD, **env_kwargs), n_envs=1, seed=seed)
    env = VecNormalize.load(str(folder / "walk_model_vecnorm.pkl"), base)
    model = PPO.load(folder / "walk_model", env=env, device="cpu")
    teacher = Go2WalkEnv(randomize_cmd=False, reward_config=AGILITY_REWARD, **env_kwargs)

    observations, targets, returns = [], [], []
    discounted_return = 0.0
    vx_lo, vx_hi = env_kwargs["command_ranges"]["vx"]
    vy_lo, vy_hi = env_kwargs["command_ranges"].get("vy", [0.0, 0.0])
    omega_lo, omega_hi = env_kwargs["command_ranges"].get("omega", [0.0, 0.0])

    try:
        pbar = tqdm(range(sample_steps), desc="omni examples", unit="step")
        for step in pbar:
            local_step = step % 500
            if local_step == 0:
                discounted_return = 0.0
                vx = float(rng.uniform(vx_lo, vx_hi))
                vy = float(rng.uniform(vy_lo, vy_hi))
                omega = float(rng.uniform(omega_lo, omega_hi))
                teacher.set_vel_cmd(vx, vy, omega)
                obs, _ = teacher.reset(seed=seed + step)
            action = omni_example_action((local_step % 250) * CONTROL_DT, float(teacher.vel_cmd[0]), float(teacher.vel_cmd[1]))
            observations.append(obs.copy())
            targets.append(action)
            obs, reward, terminated, _, _ = teacher.step(action)
            discounted_return = 0.99 * discounted_return + float(reward)
            returns.append(discounted_return)
            if terminated:
                obs, _ = teacher.reset(seed=seed + step + 17)
        observations_np = np.asarray(observations, dtype=np.float32)
        targets_np = np.asarray(targets, dtype=np.float32)
        env.obs_rms.update(observations_np)
        env.ret_rms.update(np.asarray(returns, dtype=np.float64))
        x = torch.as_tensor(env.normalize_obs(observations_np), device="cpu")
        y = torch.as_tensor(targets_np, device="cpu")
        actor = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
        optimizer = torch.optim.Adam(actor, lr=1e-4)
        pbar = tqdm(range(updates), desc="omni imitation", unit="update")
        for update in pbar:
            idx = torch.as_tensor(rng.integers(0, len(x), 512))
            prediction = model.policy._predict(x[idx], deterministic=True)
            loss = (prediction - y[idx]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(mse=f"{float(loss.item()):.3g}")
        with torch.no_grad():
            model.policy.log_std.fill_(-2.2)
        model.policy.optimizer.state.clear()
        model.save(folder / "walk_model")
        env.save(str(folder / "walk_model_vecnorm.pkl"))
        params = json.loads((folder / "walk_params.json").read_text(encoding="utf-8"))
        history = params.setdefault("omni_imitation", [])
        history.append({
            "sample_steps": sample_steps, "updates": updates, "seed": seed,
            "env_kwargs": env_kwargs, "final_mse": float(loss.item()),
        })
        params["freeze_normalization"] = False
        (folder / "walk_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        teacher.close()
        env.close()


def quick_eval(run_dir: Path) -> dict:
    rows, summary = eval_walk(run_dir, QUICK_EVAL_COMMANDS, [100], seconds=4.0)
    printable = {
        cmd: {
            "vx": round(summary["per_command_mean_vx"].get(cmd, float("nan")), 3),
            "vy": round(summary["per_command_mean_vy"].get(cmd, float("nan")), 3),
            "omega": round(summary["per_command_mean_omega"].get(cmd, float("nan")), 3),
        }
        for cmd in QUICK_EVAL_COMMANDS
    }
    print("  簡易評価(1seed,4秒):", json.dumps(printable, ensure_ascii=False))
    return {"rows": rows, "summary": summary}


def save_checkpoint(run_dir: Path, stage_name: str, quick_eval_result: dict) -> None:
    ckpt_dir = run_dir / "checkpoints" / stage_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(run_dir / name, ckpt_dir / name)
    (ckpt_dir / "quick_eval.json").write_text(
        json.dumps(quick_eval_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_curriculum(cfg: dict, run_dir: Path, num_envs: int, done_stages: set[str]) -> list[dict]:
    history_path = run_dir / "omni_curriculum.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    for stage in tqdm(cfg["stages"], desc="omni stages", unit="stage"):
        name = stage["name"]
        if name in done_stages:
            print(f"\n=== omni stage {name} は完了済みのためスキップ ===", flush=True)
            continue
        env_kwargs = dict(ENV_PHYSICS, command_mode=stage["mode"], command_ranges=stage["ranges"])
        steps = int(stage["steps"])
        print(f"\n=== omni stage {name} steps={steps} mode={stage['mode']} ranges={stage['ranges']} "
              f"imitation={stage.get('imitation', False)} ===", flush=True)
        if stage.get("imitation", False):
            omni_imitation_warmstart(
                run_dir, env_kwargs,
                sample_steps=max(6_000, min(60_000, steps // 3)),
                updates=max(800, min(8_000, steps // 35)),
                seed=2000 + len(history),
            )
        train_policy(
            "walk", run_dir, AGILITY_REWARD, steps, num_envs,
            ppo_kwargs={}, seed=0, resume=True,
            checkpoint_steps=max(20_000, steps // 2), progress_bar=True,
            walk_env_kwargs=env_kwargs, allow_reward_change=True,
            initial_log_std=-2.2,
            ppo_overrides=dict(SPEED_PPO_OVERRIDES, device="auto", learning_rate=2e-5, target_kl=0.015),
            freeze_normalization=False,
        )
        qe = quick_eval(run_dir)
        save_checkpoint(run_dir, name, qe)
        history.append({"stage": stage, "params": json.loads((run_dir / "walk_params.json").read_text(encoding="utf-8"))})
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> checkpoints/{name}/ に保存しました。結果がおかしい場合はここで一旦止めて、"
              f"--resume {run_dir} で(このステージまでを使って)再開できます。", flush=True)
    return history


EVAL_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_0.6": (0.6, 0.0, 0.0),
    "forward_1.0": (1.0, 0.0, 0.0),
    "forward_1.4": (1.4, 0.0, 0.0),
    "turn_left_fast": (1.0, 0.0, 0.75),
    "turn_right_fast": (1.0, 0.0, -0.75),
    "turn_left_sprint": (1.25, 0.0, 1.0),
    "turn_right_sprint": (1.25, 0.0, -1.0),
    "pivot_left": (0.05, 0.0, 0.8),
    "pivot_right": (0.05, 0.0, -0.8),
    "reverse_slow": (-0.4, 0.0, 0.0),
    "reverse_fast": (-0.7, 0.0, 0.0),
    "strafe_left": (0.0, 0.4, 0.0),
    "strafe_right": (0.0, -0.4, 0.0),
    "diag_back_left": (-0.4, 0.3, 0.0),
    "diag_back_right": (-0.4, -0.3, 0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/walk_omni_v1.yaml")
    parser.add_argument("--source", default=None, help="未指定ならconfig内のsourceを使用")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="既存のrun_dirを指定すると、完了済み段階をスキップして続きから実行する")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--record-videos", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(1)

    num_envs = args.num_envs or int(cfg.get("num_envs", 8))
    seconds = args.seconds or float(cfg.get("seconds", 12.0))
    seeds_raw = args.seeds or cfg.get("seeds", [100, 101, 102])
    seeds = [int(x) for x in (seeds_raw.replace(",", " ").split() if isinstance(seeds_raw, str) else seeds_raw)]

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise SystemExit(f"--resume で指定したフォルダが見つかりません: {run_dir}")
        history_path = run_dir / "omni_curriculum.json"
        done_history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        done_stages = {h["stage"]["name"] for h in done_history}
        print(f"再開: {run_dir}  完了済み段階: {sorted(done_stages)}")
    else:
        source = Path(args.source or cfg["source"])
        output_root = Path(cfg.get("output_root", "runs/walk_base"))
        run_name = args.run_name or cfg.get("run_name") or f"{time.strftime('%Y%m%d_%H%M%S')}_omni"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        copy_base(source, run_dir)
        done_stages = set()

    run_curriculum(cfg, run_dir, num_envs, done_stages)

    walk_rows, walk_summary = eval_walk(run_dir, EVAL_COMMANDS, seeds, seconds)
    tag_rows, tag_summary = eval_tag_direct(run_dir, EVAL_COMMANDS, seeds, seconds)
    result = {"walk_summary": walk_summary, "tag_summary": tag_summary, "walk_rows": walk_rows, "tag_rows": tag_rows}
    (run_dir / "speed_walk_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_speed_walk_record.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.record_videos or cfg.get("record_videos", False):
        video_dir = run_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for label, command in EVAL_COMMANDS.items():
            if label == "stand":
                continue
            record_walk(run_dir, video_dir / f"{label}.mp4", command=command, seconds=min(seconds, 12), seed=100)

    print(json.dumps({"run_dir": str(run_dir), "walk_summary": walk_summary, "tag_summary": tag_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
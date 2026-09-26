"""Refine the stable walk base into faster Tier2-oriented walking candidates."""
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

from roboquest.envs.go2_tag_env import CONTROL_HZ, ONI_QPOS_X, ONI_QPOS_Y, ONI_QVEL_X, ONI_QVEL_Y
from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv, N_LOW_STEPS
from roboquest.envs.go2_walk_env import CONTROL_DT, Go2WalkEnv
from roboquest.utils.reward_utils import WalkRewardConfig
from scripts.bootstrap_smooth_walk import example_action
from scripts.notebook_workflow import train_policy
from scripts.record_walk import record as record_walk


REQUIRED = ["walk_model.zip", "walk_model_vecnorm.pkl", "walk_params.json"]

SPEED_REWARD = WalkRewardConfig(
    linear_tracking_variance=0.20,
    angular_tracking_variance=0.10,
    lin_vel_weight=9.0,
    ang_vel_weight=2.0,
    orientation_weight=-10.0,
    base_height_target=0.29,
    base_height_weight=-300.0,
    vertical_velocity_weight=-1.0,
    nonfoot_contact_weight=-2.0,
    knee_height_target=0.08,
    knee_height_weight=-400.0,
    torques_weight=-2.5e-5,
    action_rate_weight=-0.015,
    joint_velocity_weight=-0.001,
    stand_joint_velocity_weight=-0.01,
    feet_gait_weight=0.0,
    diagonal_support_weight=0.4,
    foot_slip_weight=-0.02,
    fall_penalty=40.0,
)

AGILITY_REWARD = WalkRewardConfig(
    linear_tracking_variance=0.18,
    angular_tracking_variance=0.18,
    lin_vel_weight=8.0,
    ang_vel_weight=7.0,
    linear_error_weight=-2.5,
    angular_error_weight=-4.0,
    orientation_weight=-8.0,
    base_height_target=0.29,
    base_height_weight=-260.0,
    vertical_velocity_weight=-1.0,
    nonfoot_contact_weight=-2.0,
    knee_height_target=0.08,
    knee_height_weight=-350.0,
    torques_weight=-2.5e-5,
    action_rate_weight=-0.010,
    joint_velocity_weight=-0.001,
    stand_joint_velocity_weight=-0.01,
    feet_gait_weight=0.0,
    diagonal_support_weight=0.2,
    foot_slip_weight=-0.02,
    fall_penalty=40.0,
)

SPEED_PPO_OVERRIDES = {
    "learning_rate": 1e-5,
    "target_kl": 0.008,
    "n_epochs": 5,
    "ent_coef": 0.0,
}

PROFILES = {
    "smoke": [
        dict(name="vx06", steps=20_000, ranges={"vx": [0.45, 0.65], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
    ],
    "standard": [
        dict(name="vx06", steps=80_000, ranges={"vx": [0.45, 0.65], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="vx08", steps=120_000, ranges={"vx": [0.60, 0.90], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="vx08_turn", steps=120_000, ranges={"vx": [0.60, 0.90], "vy": [0.0, 0.0], "omega": [-0.35, 0.35]}, mode="uniform"),
        dict(name="balanced_polish", steps=100_000, ranges={"vx": [0.30, 0.90], "vy": [0.0, 0.0], "omega": [-0.25, 0.25]}, mode="uniform"),
    ],
    "strong": [
        dict(name="vx06", steps=120_000, ranges={"vx": [0.45, 0.70], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="vx09", steps=200_000, ranges={"vx": [0.70, 1.00], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="vx10_turn", steps=250_000, ranges={"vx": [0.75, 1.10], "vy": [0.0, 0.0], "omega": [-0.45, 0.45]}, mode="uniform"),
        dict(name="vx12", steps=250_000, ranges={"vx": [0.90, 1.25], "vy": [0.0, 0.0], "omega": [-0.35, 0.35]}, mode="uniform"),
        dict(name="balanced_polish", steps=180_000, ranges={"vx": [0.35, 1.10], "vy": [0.0, 0.0], "omega": [-0.35, 0.35]}, mode="uniform"),
    ],
    "race": [
        dict(name="vx12", steps=180_000, ranges={"vx": [1.00, 1.25], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="vx14", steps=220_000, ranges={"vx": [1.20, 1.45], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="vx16", steps=260_000, ranges={"vx": [1.35, 1.65], "vy": [0.0, 0.0], "omega": [0.0, 0.0]}, mode="uniform"),
        dict(name="race_turn", steps=220_000, ranges={"vx": [1.15, 1.55], "vy": [0.0, 0.0], "omega": [-0.25, 0.25]}, mode="uniform"),
        dict(name="race_polish", steps=160_000, ranges={"vx": [0.80, 1.55], "vy": [0.0, 0.0], "omega": [-0.25, 0.25]}, mode="uniform"),
    ],
    "escape": [
        dict(name="turn_low", steps=160_000, ranges={"vx": [0.35, 0.75], "vy": [0.0, 0.0], "omega": [-0.85, 0.85]}, mode="uniform"),
        dict(name="turn_mid", steps=220_000, ranges={"vx": [0.65, 1.15], "vy": [0.0, 0.0], "omega": [-0.85, 0.85]}, mode="uniform"),
        dict(name="turn_fast", steps=260_000, ranges={"vx": [0.90, 1.45], "vy": [0.0, 0.0], "omega": [-0.75, 0.75]}, mode="uniform"),
        dict(name="escape_polish", steps=180_000, ranges={"vx": [0.55, 1.45], "vy": [0.0, 0.0], "omega": [-0.95, 0.95]}, mode="uniform"),
    ],
    "agility": [
        dict(name="turn_axis", steps=320_000, ranges={"vx": [0.00, 0.35], "vy": [0.0, 0.0], "omega": [-1.25, 1.25]}, mode="axis", imitation=False),
        dict(name="lateral_axis", steps=320_000, ranges={"vx": [0.00, 0.45], "vy": [-0.60, 0.60], "omega": [-0.25, 0.25]}, mode="axis", imitation=False),
        dict(name="reverse_axis", steps=260_000, ranges={"vx": [-0.60, 0.35], "vy": [-0.25, 0.25], "omega": [-0.35, 0.35]}, mode="axis", imitation=False),
        dict(name="curve_escape", steps=420_000, ranges={"vx": [0.35, 1.20], "vy": [-0.25, 0.25], "omega": [-1.15, 1.15]}, mode="uniform", imitation=False),
        dict(name="agility_polish", steps=360_000, ranges={"vx": [-0.35, 1.25], "vy": [-0.50, 0.50], "omega": [-1.20, 1.20]}, mode="uniform", imitation=False),
    ],
    "sprint_turn": [
        dict(name="fast_curve_left_right", steps=500_000, ranges={"vx": [0.85, 1.45], "vy": [0.0, 0.0], "omega": [-1.20, 1.20]}, mode="uniform", imitation=False),
        dict(name="danger_turn", steps=420_000, ranges={"vx": [0.55, 1.20], "vy": [0.0, 0.0], "omega": [-1.35, 1.35]}, mode="uniform", imitation=False),
        dict(name="sprint_turn_polish", steps=380_000, ranges={"vx": [0.75, 1.45], "vy": [-0.12, 0.12], "omega": [-1.25, 1.25]}, mode="uniform", imitation=False),
    ],
}


def speed_imitation_warmstart(
    folder: Path,
    env_kwargs: dict,
    sample_steps: int,
    updates: int,
    seed: int,
) -> None:
    """Teach larger forward commands before PPO refinement.

    The stable base was trained mostly around vx=0.4. Commands such as 0.8 or
    1.0 are out-of-distribution, so PPO alone can get a safe but useless
    standing policy. This expands observation normalization and nudges the
    actor toward a faster trot before RL fine-tuning.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    base = make_vec_env(lambda: Go2WalkEnv(randomize_cmd=False, reward_config=SPEED_REWARD, **env_kwargs), n_envs=1, seed=seed)
    env = VecNormalize.load(str(folder / "walk_model_vecnorm.pkl"), base)
    model = PPO.load(folder / "walk_model", env=env, device="cpu")
    teacher = Go2WalkEnv(randomize_cmd=False, reward_config=SPEED_REWARD, **env_kwargs)
    observations = []
    targets = []
    returns = []
    discounted_return = 0.0
    vx_lo, vx_hi = env_kwargs["command_ranges"]["vx"]
    omega_lo, omega_hi = env_kwargs["command_ranges"].get("omega", [0.0, 0.0])
    try:
        pbar = tqdm(range(sample_steps), desc="speed examples", unit="step")
        for step in pbar:
            local_step = step % 500
            if local_step == 0:
                discounted_return = 0.0
                vx = float(rng.uniform(vx_lo, vx_hi))
                omega = float(rng.uniform(omega_lo, omega_hi))
                teacher.set_vel_cmd(vx, 0.0, omega)
                obs, _ = teacher.reset(seed=seed + step)
            action = example_action((local_step % 250) * CONTROL_DT, float(teacher.vel_cmd[0]), frequency=2.0, height=0.31, lift=0.055)
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
        pbar = tqdm(range(updates), desc="speed imitation", unit="update")
        for update in pbar:
            idx = torch.as_tensor(rng.integers(0, len(x), 512))
            prediction = model.policy._predict(x[idx], deterministic=True)
            loss = (prediction - y[idx]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(mse=f"{float(loss.item()):.3g}")
        with torch.no_grad():
            model.policy.log_std.fill_(-2.5)
        model.policy.optimizer.state.clear()
        model.save(folder / "walk_model")
        env.save(str(folder / "walk_model_vecnorm.pkl"))
        params = json.loads((folder / "walk_params.json").read_text(encoding="utf-8"))
        history = params.setdefault("speed_imitation", [])
        history.append({
            "sample_steps": sample_steps,
            "updates": updates,
            "seed": seed,
            "env_kwargs": env_kwargs,
            "final_mse": float(loss.item()),
        })
        params["freeze_normalization"] = False
        (folder / "walk_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        teacher.close()
        env.close()


def copy_base(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, destination / name)
    (destination / "speed_source.json").write_text(json.dumps({
        "source": str(source),
        "copied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_policy(folder: Path):
    params = json.loads((folder / "walk_params.json").read_text(encoding="utf-8"))
    env_kwargs = params.get("walk_env_kwargs", {})
    base = make_vec_env(lambda: Go2WalkEnv(randomize_cmd=False, **env_kwargs), n_envs=1)
    norm = VecNormalize.load(str(folder / "walk_model_vecnorm.pkl"), base)
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(folder / "walk_model", device="cpu")
    return params, base, norm, model


def eval_walk(folder: Path, commands: dict[str, tuple[float, float, float]], seeds: list[int], seconds: float) -> tuple[list[dict], dict]:
    _, base, norm, model = load_policy(folder)
    raw = base.envs[0].unwrapped
    rows = []
    try:
        total = len(commands) * len(seeds)
        pbar = tqdm(total=total, desc=f"{folder.name} speed eval", unit="case")
        for command_name, command in commands.items():
            for seed in seeds:
                pbar.set_postfix(command=command_name, seed=seed)
                raw.set_vel_cmd(*command)
                obs, _ = raw.reset(seed=seed)
                heights = []
                velocities = []
                terminated = False
                for step in range(round(seconds / CONTROL_DT)):
                    action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
                    obs, _, terminated, truncated, _ = raw.step(action)
                    body_vel = raw._world_to_body(raw.data.qvel[:3])
                    heights.append(float(raw.data.qpos[2]))
                    velocities.append([float(body_vel[0]), float(body_vel[1]), float(raw.data.qvel[5])])
                    if terminated or truncated:
                        break
                values = np.asarray(velocities, dtype=float)
                target = np.asarray(command, dtype=float)
                rows.append({
                    "command": command_name,
                    "seed": seed,
                    "fell": bool(terminated),
                    "survived_seconds": float((step + 1) * CONTROL_DT),
                    "mean_height": float(np.mean(heights)) if heights else None,
                    "min_height": float(np.min(heights)) if heights else None,
                    "mean_vx": float(np.mean(values[:, 0])) if len(values) else None,
                    "mean_vy": float(np.mean(values[:, 1])) if len(values) else None,
                    "mean_omega": float(np.mean(values[:, 2])) if len(values) else None,
                    "velocity_rmse": float(np.sqrt(np.mean((values - target) ** 2))) if len(values) else None,
                })
                pbar.update(1)
        pbar.close()
    finally:
        norm.close()
    summary = summarize(rows, seconds)
    return rows, summary


def eval_tag_direct(
    folder: Path,
    commands: dict[str, tuple[float, float, float]],
    seeds: list[int],
    seconds: float,
    max_straight_seconds: float = 3.0,
) -> tuple[list[dict], dict]:
    rows = []
    pbar = tqdm(total=len(commands) * len(seeds), desc=f"{folder.name} tag speed eval", unit="case")
    for command_name, command in commands.items():
        speed = float(np.linalg.norm(command[:2]))
        case_seconds = seconds if speed < 0.1 else min(seconds, max_straight_seconds)
        max_steps = max(1, round(case_seconds * CONTROL_HZ / N_LOW_STEPS))
        for seed in seeds:
            pbar.set_postfix(command=command_name, seed=seed)
            env = Go2TagHierarchicalEnv(
                low_level_model_path=str(folder / "walk_model"),
                low_level_vecnorm_path=str(folder / "walk_model_vecnorm.pkl"),
                oni_speed=0.0,
                max_episode_steps=max_steps,
                high_level_command_mode="direct",
            )
            try:
                env.reset(seed=seed)
                env.data.qpos[ONI_QPOS_X] = -2.2
                env.data.qpos[ONI_QPOS_Y] = 2.2
                env.data.qvel[ONI_QVEL_X] = 0.0
                env.data.qvel[ONI_QVEL_Y] = 0.0
                info = {}
                terminated = False
                for _ in range(max_steps):
                    _, _, terminated, truncated, info = env.step(np.asarray(command, dtype=np.float32))
                    if terminated or truncated:
                        break
                rows.append({
                    "command": command_name,
                    "seed": seed,
                    "fell": bool(terminated and not info.get("is_tagged", False)),
                    "tagged": bool(info.get("is_tagged", False)),
                    "survived_seconds": float(info.get("survived_seconds", 0.0)),
                    "target_seconds": float(case_seconds),
                    "min_height": float(env.data.qpos[2]),
                })
            finally:
                env.close()
            pbar.update(1)
    pbar.close()
    summary = summarize(rows, seconds)
    return rows, summary


def summarize(rows: list[dict], seconds: float) -> dict:
    if not rows:
        return {"passed": False}
    fall_count = sum(1 for row in rows if row.get("fell"))
    tagged_count = sum(1 for row in rows if row.get("tagged"))
    ratios = [
        float(row.get("survived_seconds", 0.0)) / max(float(row.get("target_seconds", seconds)), 1e-6)
        for row in rows
    ]
    min_ratio = min(ratios)
    min_seconds = min(float(row.get("survived_seconds", 0.0)) for row in rows)
    mean_seconds = sum(float(row.get("survived_seconds", 0.0)) for row in rows) / len(rows)
    min_height_values = [float(row["min_height"]) for row in rows if row.get("min_height") is not None]
    min_height = min(min_height_values) if min_height_values else None
    mean_vx = [row.get("mean_vx") for row in rows if row.get("mean_vx") is not None]
    mean_vy = [row.get("mean_vy") for row in rows if row.get("mean_vy") is not None]
    mean_omega = [row.get("mean_omega") for row in rows if row.get("mean_omega") is not None]
    per_command_vx = {}
    per_command_vy = {}
    per_command_omega = {}
    per_command_rmse = {}
    for command in sorted({row.get("command") for row in rows}):
        selected = [row for row in rows if row.get("command") == command]
        vx_values = [row.get("mean_vx") for row in selected if row.get("mean_vx") is not None]
        vy_values = [row.get("mean_vy") for row in selected if row.get("mean_vy") is not None]
        omega_values = [row.get("mean_omega") for row in selected if row.get("mean_omega") is not None]
        rmse_values = [row.get("velocity_rmse") for row in selected if row.get("velocity_rmse") is not None]
        if vx_values:
            per_command_vx[str(command)] = float(np.mean(vx_values))
        if vy_values:
            per_command_vy[str(command)] = float(np.mean(vy_values))
        if omega_values:
            per_command_omega[str(command)] = float(np.mean(omega_values))
        if rmse_values:
            per_command_rmse[str(command)] = float(np.mean(rmse_values))
    high_speed_values = [
        per_command_vx[name]
        for name in ("forward_0.8", "forward_1.0", "forward_1.2")
        if name in per_command_vx
    ]
    return {
        "passed": bool(fall_count == 0 and tagged_count == 0 and min_ratio >= 0.98 and (min_height is None or min_height >= 0.15)),
        "fall_count": fall_count,
        "tagged_count": tagged_count,
        "mean_survived_seconds": mean_seconds,
        "min_survived_seconds": min_seconds,
        "min_survival_ratio": min_ratio,
        "min_height": min_height,
        "mean_vx": float(np.mean(mean_vx)) if mean_vx else None,
        "mean_vy": float(np.mean(mean_vy)) if mean_vy else None,
        "mean_omega": float(np.mean(mean_omega)) if mean_omega else None,
        "per_command_mean_vx": per_command_vx,
        "per_command_mean_vy": per_command_vy,
        "per_command_mean_omega": per_command_omega,
        "per_command_velocity_rmse": per_command_rmse,
        "high_speed_mean_vx": float(np.mean(high_speed_values)) if high_speed_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/walk_base/walk_strong_001/candidates/walk_seed2")
    parser.add_argument("--output-root", default="runs/walk_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--record-videos", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    source = Path(args.source)
    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_speed_{args.profile}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    copy_base(source, run_dir)
    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    stages = PROFILES[args.profile]
    history = []

    for stage in tqdm(stages, desc="speed stages", unit="stage"):
        env_kwargs = {
            "action_scale": 0.6,
            "joint_stiffness": 60.0,
            "joint_damping": 2.0,
            "torque_limits": [23.7, 23.7, 45.43],
            "command_mode": stage["mode"],
            "command_ranges": stage["ranges"],
        }
        print(f"\n=== speed stage {stage['name']} steps={stage['steps']} ranges={stage['ranges']} ===", flush=True)
        # The example gait teacher is forward-only. Disable it for agility
        # stages so omega/vy commands are learned from the tracking reward.
        stage_steps = int(stage["steps"])
        if stage.get("imitation", True):
            speed_imitation_warmstart(
                run_dir,
                env_kwargs,
                sample_steps=max(6_000, min(60_000, stage_steps // 3)),
                updates=max(800, min(8_000, stage_steps // 35)),
                seed=1000 + len(history),
            )
        reward_config = AGILITY_REWARD if args.profile in {"agility", "sprint_turn"} else SPEED_REWARD
        train_policy(
            "walk",
            run_dir,
            reward_config,
            stage_steps,
            int(args.num_envs),
            ppo_kwargs={},
            seed=0,
            resume=True,
            checkpoint_steps=max(20_000, int(stage["steps"]) // 2),
            progress_bar=True,
            walk_env_kwargs=env_kwargs,
            allow_reward_change=True,
            initial_log_std=-2.2,
            ppo_overrides=dict(SPEED_PPO_OVERRIDES, device="auto", learning_rate=2e-5, target_kl=0.015),
            freeze_normalization=False,
        )
        history.append({"stage": stage, "params": json.loads((run_dir / "walk_params.json").read_text(encoding="utf-8"))})
        (run_dir / "speed_curriculum.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    commands = {
        "stand": (0.0, 0.0, 0.0),
        "forward_0.4": (0.4, 0.0, 0.0),
        "forward_0.6": (0.6, 0.0, 0.0),
        "forward_0.8": (0.8, 0.0, 0.0),
        "forward_1.0": (1.0, 0.0, 0.0),
    }
    if args.profile in {"strong", "race", "escape", "agility", "sprint_turn"}:
        commands["forward_1.2"] = (1.2, 0.0, 0.0)
    if args.profile in {"race", "escape", "agility", "sprint_turn"}:
        commands["forward_1.4"] = (1.4, 0.0, 0.0)
        commands["turn_left_fast"] = (1.0, 0.0, 0.75)
        commands["turn_right_fast"] = (1.0, 0.0, -0.75)
        commands["turn_left_sprint"] = (1.25, 0.0, 1.0)
        commands["turn_right_sprint"] = (1.25, 0.0, -1.0)
    if args.profile == "agility":
        commands["turn_left_hard"] = (0.35, 0.0, 1.1)
        commands["turn_right_hard"] = (0.35, 0.0, -1.1)
        commands["strafe_left"] = (0.35, 0.35, 0.0)
        commands["strafe_right"] = (0.35, -0.35, 0.0)
        commands["reverse"] = (-0.35, 0.0, 0.0)
    if args.profile == "race":
        commands["forward_1.6"] = (1.6, 0.0, 0.0)
    walk_rows, walk_summary = eval_walk(run_dir, commands, seeds, args.seconds)
    tag_rows, tag_summary = eval_tag_direct(run_dir, commands, seeds, args.seconds)
    result = {"walk_summary": walk_summary, "tag_summary": tag_summary, "walk_rows": walk_rows, "tag_rows": tag_rows}
    (run_dir / "speed_walk_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(run_dir / name, best_dir / name)
    (best_dir / "best_speed_walk_record.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.record_videos:
        video_dir = run_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for label, command in commands.items():
            if label == "stand":
                continue
            out = video_dir / f"{label}.mp4"
            record_walk(run_dir, out, command=command, seconds=min(args.seconds, 12), seed=100)

    print(json.dumps({"run_dir": str(run_dir), "walk_summary": walk_summary, "tag_summary": tag_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

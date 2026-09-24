"""PC-oriented GA runner for compatible Tier2 flee training."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm

from roboquest.envs.go2_tag_strategic_train_env import Go2TagStrategicTrainEnv
from roboquest.utils.reward_utils import FleeRewardConfig
from scripts.evaluate_tier2 import evaluate_candidate
from scripts.tier2_common import (
    Stopwatch,
    append_csv,
    append_jsonl,
    copy_submission_files,
    copy_walk_files,
    crossover,
    flatten_row,
    format_seconds,
    hash_dict,
    load_config,
    make_experiment_dir,
    mutate_individual,
    sample_from_space,
    score_summary,
    summarize_results,
    write_json,
)


REWARD_KEYS = {"survival_weight", "distance_weight", "tag_penalty", "fall_penalty"}
STRATEGIC_KEYS = {
    "distance_delta_weight",
    "distance_closing_penalty",
    "wall_penalty_weight",
    "danger_wall_penalty_weight",
    "escape_bonus",
    "action_change_weight",
    "lateral_action_weight",
    "yaw_action_weight",
    "stall_penalty_weight",
}
PPO_KEYS = {"learning_rate", "gamma", "gae_lambda", "ent_coef", "n_steps", "batch_size", "n_epochs", "net_arch"}


def _scale_to_raw(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / max(hi - lo, 1e-6) * 2.0 - 1.0, -1.0, 1.0))


def teacher_raw_action(obs: np.ndarray, command_ranges: dict) -> np.ndarray:
    """Simple body-relative escape teacher for PPO warm start.

    The observation's first two values are the oni position relative to the
    robot. In body-relative mode, +x is forward and +y is left. The teacher
    turns toward the direction opposite the oni, then sprints forward.
    """
    rel = np.asarray(obs[:2], dtype=np.float64)
    if np.linalg.norm(rel) < 1e-6:
        rel = np.array([1.0, 0.0])
    away = -rel / max(float(np.linalg.norm(rel)), 1e-6)
    yaw_error = float(np.arctan2(away[1], away[0]))
    vx_cmd = 1.45 if abs(yaw_error) < 0.75 else 0.90
    omega_cmd = float(np.clip(1.35 * yaw_error, -0.95, 0.95))
    vx_lo, vx_hi = command_ranges["vx"]
    vy_lo, vy_hi = command_ranges["vy"]
    om_lo, om_hi = command_ranges["omega"]
    return np.array([
        _scale_to_raw(vx_cmd, vx_lo, vx_hi),
        _scale_to_raw(0.0, vy_lo, vy_hi),
        _scale_to_raw(omega_cmd, om_lo, om_hi),
    ], dtype=np.float32)


def behavior_clone_teacher(
    model: PPO,
    vec_env: VecNormalize,
    make_teacher_env,
    command_ranges: dict,
    *,
    episodes: int,
    updates: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> int:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    teacher_env = make_teacher_env()
    try:
        for ep in range(episodes):
            obs, _ = teacher_env.reset(seed=seed + ep)
            for _ in range(teacher_env.max_episode_steps):
                action = teacher_raw_action(obs, command_ranges)
                obs_rows.append(obs.astype(np.float32))
                action_rows.append(action)
                obs, _, terminated, truncated, _ = teacher_env.step(action)
                if terminated or truncated:
                    break
    finally:
        teacher_env.close()

    if not obs_rows or updates <= 0:
        return 0

    obs_np = np.asarray(obs_rows, dtype=np.float32)
    actions_np = np.asarray(action_rows, dtype=np.float32)
    vec_env.obs_rms.update(obs_np)
    norm_obs_np = vec_env.normalize_obs(obs_np.copy())
    obs_t = torch.as_tensor(norm_obs_np, device=model.device)
    actions_t = torch.as_tensor(actions_np, device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in tqdm(range(updates), desc="teacher BC", unit="update", leave=False):
        idx = torch.as_tensor(rng.integers(0, len(obs_np), size=batch_size), device=model.device)
        pred, _, _ = model.policy(obs_t[idx], deterministic=True)
        loss = torch.nn.functional.mse_loss(pred, actions_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return len(obs_rows)


def split_individual(ind: dict) -> tuple[dict, dict, dict, dict]:
    reward = {k: ind[k] for k in REWARD_KEYS if k in ind}
    strategic = {k: ind[k] for k in STRATEGIC_KEYS if k in ind}
    ppo = {k: ind[k] for k in PPO_KEYS if k in ind}
    train = {
        "timesteps": int(ind.get("flee_timesteps", 100000)),
        "n_envs": int(ind.get("flee_num_envs", 4)),
        "oni_speed": float(ind.get("oni_speed", 0.025)),
    }
    return reward, strategic, ppo, train


def train_candidate(cfg: dict, run_dir: Path, candidate: dict, generation: int, index: int) -> tuple[Path, dict]:
    cid = hash_dict(candidate, "cand_")
    candidate_dir = run_dir / "candidates" / cid
    done_file = candidate_dir / "done.json"
    if done_file.is_file():
        return candidate_dir, json.loads(done_file.read_text(encoding="utf-8"))

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    copy_walk_files(cfg["walk_source_dir"], candidate_dir)
    write_json(candidate_dir / "individual.json", candidate)

    reward_dict, strategic_dict, ppo_dict, train_cfg = split_individual(candidate)
    reward_cfg = FleeRewardConfig(**reward_dict)
    walk_model = str(candidate_dir / "walk_model")
    walk_vecnorm = str(candidate_dir / "walk_model_vecnorm.pkl")
    command_mode = cfg.get("high_level_command_mode", "safe_forward")
    command_ranges = cfg.get("high_level_command_ranges")
    observation_mode = cfg.get("high_level_observation_mode", "standard")
    initial_distance_range = cfg.get("initial_distance_range", (1.5, 2.35))
    max_episode_seconds = float(cfg.get("max_episode_seconds", 60.0))

    def make_env():
        env = Go2TagStrategicTrainEnv(
            low_level_model_path=walk_model,
            low_level_vecnorm_path=walk_vecnorm,
            flee_config=reward_cfg,
            strategic_config=strategic_dict,
            oni_speed=train_cfg["oni_speed"],
            high_level_command_mode=command_mode,
            high_level_command_ranges=command_ranges,
            high_level_observation_mode=observation_mode,
            initial_distance_range=tuple(initial_distance_range),
            max_episode_seconds=max_episode_seconds,
        )
        return Monitor(env)

    def make_teacher_env():
        return Go2TagStrategicTrainEnv(
            low_level_model_path=walk_model,
            low_level_vecnorm_path=walk_vecnorm,
            flee_config=reward_cfg,
            strategic_config=strategic_dict,
            oni_speed=train_cfg["oni_speed"],
            high_level_command_mode=command_mode,
            high_level_command_ranges=command_ranges,
            high_level_observation_mode=observation_mode,
            initial_distance_range=tuple(initial_distance_range),
            max_episode_seconds=max_episode_seconds,
        )

    n_envs = int(train_cfg["n_envs"])
    vec_env_cls = SubprocVecEnv if cfg.get("vec_env_cls", "dummy") == "subproc" else None
    base = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=int(cfg.get("seed", 0)) + generation * 1000 + index,
        vec_env_cls=vec_env_cls,
    )
    env = VecNormalize(base, norm_obs=True, norm_reward=True)
    default_training = cfg.get("default_training", {})
    base_ppo = {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.02,
        "policy_kwargs": {"net_arch": [128, 128]},
        "device": default_training.get("device", "cpu"),
    }
    if "net_arch" in ppo_dict:
        ppo_dict["policy_kwargs"] = {"net_arch": ppo_dict.pop("net_arch")}
    base_ppo.update(ppo_dict)
    started = time.monotonic()
    bc_samples = 0
    try:
        model = PPO("MlpPolicy", env, seed=int(cfg.get("seed", 0)) + index, verbose=0, **base_ppo)
        if not bool(cfg.get("skip_policy_training", False)):
            teacher_cfg = cfg.get("teacher_warmstart", {})
            if teacher_cfg.get("enabled", False):
                bc_samples = behavior_clone_teacher(
                    model,
                    env,
                    make_teacher_env,
                    command_ranges,
                    episodes=int(teacher_cfg.get("episodes", 24)),
                    updates=int(teacher_cfg.get("updates", 1200)),
                    batch_size=int(teacher_cfg.get("batch_size", 512)),
                    lr=float(teacher_cfg.get("learning_rate", 0.0003)),
                    seed=int(cfg.get("seed", 0)) + generation * 1000 + index * 100,
                )
                print(f"teacher warmstart samples={bc_samples}")
            checkpoint_steps = int(default_training.get("checkpoint_steps", 25000))
            cb = CheckpointCallback(
                save_freq=max(checkpoint_steps // max(n_envs, 1), 1),
                save_path=str(candidate_dir / "checkpoints"),
                name_prefix="flee",
                save_vecnormalize=True,
            )
            model.learn(total_timesteps=int(train_cfg["timesteps"]), callback=cb, progress_bar=True)
        else:
            print("skip_policy_training=True: saving untrained high-level policy because command mode ignores policy actions")
        model.save(str(candidate_dir / "flee_model"))
        env.save(str(candidate_dir / "flee_model_vecnorm.pkl"))
    finally:
        env.close()

    params = {
        "kind": "flee",
        "reward_config": asdict(reward_cfg),
        "timesteps": int(train_cfg["timesteps"]),
        "num_envs": n_envs,
        "ppo_kwargs": base_ppo,
        "seed": int(cfg.get("seed", 0)) + index,
        "oni_speed": float(train_cfg["oni_speed"]),
        "total_timesteps": int(train_cfg["timesteps"]),
        "elapsed_seconds": time.monotonic() - started,
        "tier2_candidate_id": cid,
        "strategic_reward": strategic_dict,
        "high_level_command_mode": command_mode,
        "high_level_command_ranges": command_ranges,
        "high_level_observation_mode": observation_mode,
        "initial_distance_range": list(initial_distance_range),
        "max_episode_seconds": max_episode_seconds,
        "teacher_warmstart": cfg.get("teacher_warmstart", {}),
        "teacher_bc_samples": bc_samples,
    }
    write_json(candidate_dir / "flee_params.json", params)

    results = evaluate_candidate(candidate_dir, cfg.get("coarse_eval_seeds", [100, 101, 102]))
    summary = summarize_results(results)
    summary["score"] = score_summary(summary)
    record = {
        "candidate_id": cid,
        "generation": generation,
        "index": index,
        "candidate_dir": str(candidate_dir),
        "individual": candidate,
        "eval_results": results,
        "summary": summary,
    }
    write_json(done_file, record)
    return candidate_dir, record


def load_existing_records(run_dir: Path) -> dict[str, dict]:
    records = {}
    for path in (run_dir / "candidates").glob("cand_*/done.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        records[data["candidate_id"]] = data
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tier2_compat_ga.yaml")
    parser.add_argument("--resume", default=None, help="Existing run directory to continue.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rng = random.Random(int(cfg.get("seed", 0)))
    if args.resume:
        run_dir = Path(args.resume)
        state_path = run_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    else:
        run_dir = make_experiment_dir(cfg.get("output_dir", "runs/tier2"), cfg.get("experiment_name", "compat_ga"))
        write_json(run_dir / "config.json", cfg)
        state = {}

    space = cfg["search_space"]
    population_size = int(cfg.get("population_size", 8))
    generations = int(cfg.get("generations", 10))
    elite_count = int(cfg.get("elite_count", 2))
    mutation_rate = float(cfg.get("mutation_rate", 0.25))
    crossover_rate = float(cfg.get("crossover_rate", 0.65))

    population = state.get("population") or [sample_from_space(space, rng) for _ in range(population_size)]
    best_record = state.get("best_record")
    watch = Stopwatch()
    start_generation = int(state.get("generation", 0))
    progress_path = run_dir / "progress_state.json"
    total_candidates = max(0, generations - start_generation) * population_size
    pbar = tqdm(total=total_candidates, desc="Tier2 GA candidates", unit="candidate")

    for generation in range(start_generation, generations):
        print(f"\n=== Generation {generation + 1}/{generations} ===")
        generation_records = []
        for index, individual in enumerate(population):
            cid = hash_dict(individual, "cand_")
            best_score = None if best_record is None else round(float(best_record["summary"]["score"]), 2)
            write_json(
                progress_path,
                {
                    "status": "running",
                    "generation": generation,
                    "generation_display": f"{generation + 1}/{generations}",
                    "individual_index": index,
                    "individual_display": f"{index + 1}/{population_size}",
                    "candidate_id": cid,
                    "elapsed_seconds": watch.elapsed,
                    "eta_generation": watch.eta(index, population_size),
                    "best_score": best_score,
                    "run_dir": str(run_dir),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            pbar.set_description(f"Gen {generation + 1}/{generations} Ind {index + 1}/{population_size}")
            pbar.set_postfix(candidate=cid[:8], best=best_score)
            print(f"[Generation {generation + 1}/{generations}] Individual {index + 1}/{population_size} {cid}")
            print(f"elapsed={format_seconds(watch.elapsed)} eta_generation={watch.eta(index, population_size)}")
            candidate_dir, record = train_candidate(cfg, run_dir, individual, generation, index)
            summary = record["summary"]
            print(
                "score={score:.2f} mean={mean_survived_seconds:.1f}s worst={worst_survived_seconds:.1f}s "
                "escaped={escaped_count} tagged={tagged_count} fell={fell_count}".format(**summary)
            )
            generation_records.append(record)
            append_jsonl(run_dir / "candidates.jsonl", record)
            row = {
                "candidate_id": record["candidate_id"],
                "generation": generation,
                "index": index,
                "candidate_dir": record["candidate_dir"],
                **flatten_row("", summary),
            }
            append_csv(run_dir / "results.csv", row)
            if best_record is None or summary["score"] > best_record["summary"]["score"]:
                best_record = record
                best_dir = run_dir / "best"
                best_dir.mkdir(exist_ok=True)
                (best_dir / "candidate_dir.txt").write_text(record["candidate_dir"], encoding="utf-8")
                copy_submission_files(record["candidate_dir"], best_dir)
                write_json(best_dir / "best_record.json", best_record)
                print(f"new best: {summary['score']:.2f}")
            write_json(
                progress_path,
                {
                    "status": "finished_candidate",
                    "generation": generation,
                    "generation_display": f"{generation + 1}/{generations}",
                    "individual_index": index,
                    "individual_display": f"{index + 1}/{population_size}",
                    "candidate_id": cid,
                    "last_score": float(summary["score"]),
                    "last_mean_survived_seconds": float(summary["mean_survived_seconds"]),
                    "last_worst_survived_seconds": float(summary["worst_survived_seconds"]),
                    "last_escaped_count": int(summary["escaped_count"]),
                    "last_tagged_count": int(summary["tagged_count"]),
                    "last_fell_count": int(summary["fell_count"]),
                    "best_score": float(best_record["summary"]["score"]) if best_record else None,
                    "elapsed_seconds": watch.elapsed,
                    "run_dir": str(run_dir),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            pbar.update(1)

        ranked = sorted(generation_records, key=lambda r: r["summary"]["score"], reverse=True)
        write_json(run_dir / f"generation_{generation:03d}_ranking.json", ranked)
        elites = [r["individual"] for r in ranked[:elite_count]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent_a = rng.choice(elites or population)
            parent_b = rng.choice(elites or population)
            child = crossover(parent_a, parent_b, crossover_rate, rng)
            child = mutate_individual(child, space, mutation_rate, rng)
            next_population.append(child)
        population = next_population
        state = {"generation": generation + 1, "population": population, "best_record": best_record}
        write_json(run_dir / "state.json", state)
        write_json(
            progress_path,
            {
                "status": "finished_generation",
                "generation": generation + 1,
                "generation_display": f"{generation + 1}/{generations}",
                "best_score": float(best_record["summary"]["score"]) if best_record else None,
                "elapsed_seconds": watch.elapsed,
                "run_dir": str(run_dir),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    pbar.close()
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()

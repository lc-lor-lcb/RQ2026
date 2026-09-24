"""Shared helpers for RoboQuest Tier2 experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable


SUBMISSION_FILES = [
    "walk_model.zip",
    "walk_model_vecnorm.pkl",
    "walk_params.json",
    "flee_model.zip",
    "flee_model_vecnorm.pkl",
    "flee_params.json",
]


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    return data or {}


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | os.PathLike[str], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def append_csv(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    fieldnames = list(row.keys())
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_dict(data: dict[str, Any], prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}" if prefix else digest


def make_experiment_dir(base_dir: str | os.PathLike[str], name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    latest_file = Path(base_dir) / "latest_run.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(str(run_dir), encoding="utf-8")
    return run_dir


def resolve_run(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.name == "latest":
        latest = p.parent / "latest_run.txt"
        if latest.exists():
            return Path(latest.read_text(encoding="utf-8").strip())
    if p.is_dir():
        return p
    latest = p / "latest_run.txt"
    if latest.exists():
        return Path(latest.read_text(encoding="utf-8").strip())
    raise FileNotFoundError(f"Run directory not found: {path}")


def copy_walk_files(source_dir: str | os.PathLike[str], dest_dir: str | os.PathLike[str]) -> None:
    src = Path(source_dir)
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("walk_model.zip", "walk_model_vecnorm.pkl", "walk_params.json"):
        source = src / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, dst / name)


def copy_submission_files(candidate_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> None:
    src = Path(candidate_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in SUBMISSION_FILES:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dst / name)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return dict(
            mean_survived_seconds=0.0,
            worst_survived_seconds=0.0,
            escaped_count=0,
            tagged_count=0,
            fell_count=0,
            mean_distance=0.0,
            min_distance=0.0,
        )
    survived = [float(r.get("survived_seconds", 0.0)) for r in results]
    distances = [float(r.get("mean_distance", 0.0)) for r in results]
    min_distances = [float(r.get("min_distance", r.get("mean_distance", 0.0))) for r in results]
    return dict(
        mean_survived_seconds=sum(survived) / len(survived),
        worst_survived_seconds=min(survived),
        escaped_count=sum(1 for r in results if bool(r.get("escaped", False))),
        tagged_count=sum(1 for r in results if bool(r.get("tagged", False))),
        fell_count=sum(1 for r in results if bool(r.get("fell", False))),
        mean_distance=sum(distances) / len(distances),
        min_distance=min(min_distances),
    )


def score_summary(summary: dict[str, Any]) -> float:
    return (
        float(summary.get("mean_survived_seconds", 0.0))
        + 35.0 * float(summary.get("escaped_count", 0))
        + 2.0 * float(summary.get("mean_distance", 0.0))
        - 12.0 * float(summary.get("tagged_count", 0))
        - 30.0 * float(summary.get("fell_count", 0))
        + 0.25 * float(summary.get("worst_survived_seconds", 0.0))
    )


def sample_from_space(space: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "uniform":
            out[key] = rng.uniform(float(spec[1]), float(spec[2]))
        elif kind == "loguniform":
            lo, hi = math.log(float(spec[1])), math.log(float(spec[2]))
            out[key] = math.exp(rng.uniform(lo, hi))
        elif kind == "choice":
            out[key] = rng.choice(spec[1])
        else:
            raise ValueError(f"Unknown search spec for {key}: {spec}")
    return out


def mutate_individual(individual: dict[str, Any], space: dict[str, Any], rate: float, rng: random.Random) -> dict[str, Any]:
    child = json.loads(json.dumps(individual))
    for key in list(space.keys()):
        if rng.random() < rate:
            child[key] = sample_from_space({key: space[key]}, rng)[key]
    return child


def crossover(a: dict[str, Any], b: dict[str, Any], rate: float, rng: random.Random) -> dict[str, Any]:
    if rng.random() > rate:
        return json.loads(json.dumps(a))
    return {key: (a[key] if rng.random() < 0.5 else b.get(key, a[key])) for key in a.keys()}


def flatten_row(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{k}": json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in data.items()}


class Stopwatch:
    def __init__(self) -> None:
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def eta(self, done: int, total: int) -> str:
        if done <= 0:
            return "unknown"
        rate = self.elapsed / done
        return format_seconds(rate * max(0, total - done))

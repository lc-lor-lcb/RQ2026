"""Rank Tier2 GA candidates from results.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.tier2_common import resolve_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    run_dir = resolve_run(args.run)
    rows = []
    with (run_dir / "results.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_score"] = float(row.get("score", 0.0))
            rows.append(row)
    rows.sort(key=lambda r: r["_score"], reverse=True)
    out = run_dir / "ranking.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys() if k != "_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "_score"})
    for row in rows[: args.top]:
        print(row["candidate_id"], row.get("score"), row.get("mean_survived_seconds"), row.get("escaped_count"))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

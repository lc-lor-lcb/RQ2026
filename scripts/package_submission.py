"""Create a submission-compatible package from a Tier2 candidate."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scripts.evaluate_tier2 import candidate_path
from scripts.tier2_common import SUBMISSION_FILES, copy_submission_files, resolve_run, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", default="best")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = resolve_run(args.run)
    cand = candidate_path(run_dir, args.candidate)
    out = Path(args.output) if args.output else run_dir / "submission_package"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copy_submission_files(cand, out)
    missing = [name for name in SUBMISSION_FILES if not (out / name).is_file()]
    write_json(out / "manifest.json", {"candidate_dir": str(cand), "files": SUBMISSION_FILES, "missing": missing})
    if missing:
        raise FileNotFoundError(f"Missing submission files: {missing}")
    print(f"submission package: {out}")


if __name__ == "__main__":
    main()

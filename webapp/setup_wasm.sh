#!/usr/bin/env bash
# Install pinned browser dependencies for the Colab-only RoboQuest viewer.
#
# This intentionally does not use zalo/mujoco_wasm GitHub releases. The viewer
# uses the official mujoco-js npm package and serves every asset from Colab.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

python scripts/setup_web_vendor.py "$@"

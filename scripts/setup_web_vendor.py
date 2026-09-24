"""
Download pinned browser assets for the Colab-only RoboQuest viewer.

The viewer is served from the Colab runtime, so all JS/WASM dependencies that
are needed at display time are copied into webapp/vendor/.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


MUJOCO_JS_VERSION = "0.0.7"
ONNXRUNTIME_WEB_VERSION = "1.17.0"
THREE_VERSION = "0.160.0"

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
VENDOR = WEBAPP / "vendor"


def _download(url: str) -> bytes:
    print(f"  down {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _extract_member(tgz: bytes, member: str, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as archive:
        try:
            item = archive.getmember(member)
        except KeyError as exc:
            raise RuntimeError(f"{member} was not found in npm package") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(item) as src:
            if src is None:
                raise RuntimeError(f"{member} is not a file")
            with destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def install_mujoco() -> None:
    tgz = _download(
        f"https://registry.npmjs.org/mujoco-js/-/mujoco-js-{MUJOCO_JS_VERSION}.tgz"
    )
    out = VENDOR / "mujoco" / "mujoco_wasm.js"
    _extract_member(tgz, "package/dist/mujoco_wasm.js", out)
    print(f"  ok {out.relative_to(ROOT)}")


def install_ort() -> None:
    tgz = _download(
        "https://registry.npmjs.org/onnxruntime-web/"
        f"-/onnxruntime-web-{ONNXRUNTIME_WEB_VERSION}.tgz"
    )
    files = [
        "ort.min.js",
        "ort-wasm.wasm",
        "ort-wasm-simd.wasm",
    ]
    for filename in files:
        out = VENDOR / "ort" / filename
        _extract_member(tgz, f"package/dist/{filename}", out)
        print(f"  ok {out.relative_to(ROOT)}")


def install_three() -> None:
    tgz = _download(f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz")
    files = [
        ("package/build/three.module.js", VENDOR / "three" / "three.module.js"),
        (
            "package/examples/jsm/controls/OrbitControls.js",
            VENDOR / "three" / "addons" / "controls" / "OrbitControls.js",
        ),
    ]
    for member, out in files:
        _extract_member(tgz, member, out)
        print(f"  ok {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install browser viewer vendor assets")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove webapp/vendor before downloading pinned assets",
    )
    args = parser.parse_args()

    if args.clean and VENDOR.exists():
        shutil.rmtree(VENDOR)

    VENDOR.mkdir(parents=True, exist_ok=True)
    print("Installing RoboQuest web viewer assets...")
    install_mujoco()
    install_ort()
    install_three()
    print("\nDone. webapp/vendor is ready for Colab static serving.")


if __name__ == "__main__":
    # Colab sometimes runs this from a different cwd; keep all paths repo-relative.
    os.chdir(ROOT)
    main()

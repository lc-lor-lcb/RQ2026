"""Print the training devices that Python can actually see."""
from __future__ import annotations

import platform
import subprocess


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            print(f"CUDA device: {idx} - {torch.cuda.get_device_name(idx)}")
            print(f"CUDA capability: {torch.cuda.get_device_capability(idx)}")
            free, total = torch.cuda.mem_get_info(idx)
            print(f"CUDA memory free/total: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print("\n--- nvidia-smi ---")
        print(result.stdout.strip() or result.stderr.strip())
    except FileNotFoundError:
        print("nvidia-smi: not found")


if __name__ == "__main__":
    main()

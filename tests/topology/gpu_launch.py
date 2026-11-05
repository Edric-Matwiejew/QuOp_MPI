"""GPU allocation discovery shared by the test launchers (no QuOp imports)."""

import os
import subprocess
from pathlib import Path


def detect_gpus_per_node():
    """Return the allocated GPU count, or None when it cannot be determined.

    QUOP_TEST_GPUS_PER_NODE is useful when each task sees only one GPU: its
    CUDA_VISIBLE_DEVICES list cannot describe the whole node allocation.
    """
    for name in ("QUOP_TEST_GPUS_PER_NODE", "SLURM_GPUS_ON_NODE"):
        value = os.environ.get(name)
        if value:
            if not value.isdigit() or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            return int(value)

    for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if name in os.environ:
            value = os.environ[name].strip()
            if not value or value == "-1":
                return 0
            return len(value.split(","))

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return len(result.stdout.splitlines())
    except (OSError, subprocess.TimeoutExpired):
        pass

    rocm_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    try:
        result = subprocess.run(
            [str(rocm_root / "bin" / "rocm_agent_enumerator")],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        devices = [line for line in result.stdout.splitlines()
                   if line.startswith("gfx") and line != "gfx000"]
        if result.returncode == 0 and devices:
            return len(devices)
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Preserve the configured Setonix default when discovery is unavailable.
    if os.environ.get("QUOP_PROFILE") == "pawsey-setonix":
        return 8
    return None


if __name__ == "__main__":
    count = detect_gpus_per_node()
    if count is not None:
        print(count)

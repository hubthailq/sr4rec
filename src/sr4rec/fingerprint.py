"""Environment fingerprint written to fingerprint.yaml for every run."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

import torch

from . import __version__

PACKAGES = ("torch", "torchvision", "timm", "spandrel", "numpy", "pandas", "pillow", "scipy", "statsmodels",
            "scikit-learn", "matplotlib", "pydantic", "jinja2", "pyyaml")


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5).stdout.strip() or platform.processor()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _driver() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _git_commit() -> str | None:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict:
    res = {}
    for name in PACKAGES:
        try:
            res[name] = md.version(name)
        except md.PackageNotFoundError:
            res[name] = None
    return res


def environment(device: torch.device) -> dict:
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else (
        "Apple GPU (MPS)" if device.type == "mps" else None)
    return {
        "sr4rec_version": __version__,
        "sr4rec_git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "device": str(device),
        "gpu": gpu,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "nvidia_driver": _driver() if gpu else None,
        "packages": _versions(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def now_text() -> tuple[str, _dt.datetime]:
    now = _dt.datetime.now().astimezone()
    off = now.strftime("%z")
    return f"{now:%Y-%m-%d %H:%M} (UTC{off[:3]}:{off[3:]})", now

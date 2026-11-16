"""Environment / device inspection.

The project runs on a CPU-only 2-vCPU / 1.9 GB RAM sandbox, so the training
profile must be chosen from measured facts, never assumed.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class EnvReport:
    os: str
    python: str
    cpu_count: int
    ram_gb: float
    disk_free_gb: float
    gpu_present: bool
    torch_version: str | None
    cuda_available: bool
    training_profile: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return 0.0


def inspect_environment() -> EnvReport:
    gpu = shutil.which("nvidia-smi") is not None
    torch_version, cuda = None, False
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
    except ImportError:
        pass

    import os

    ram = _ram_gb()
    profile = "cpu_microb budget" if (not cuda or ram < 8) else "gpu_standard"

    return EnvReport(
        os=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=ram,
        disk_free_gb=round(shutil.disk_usage("/").free / 1e9, 1),
        gpu_present=gpu,
        torch_version=torch_version,
        cuda_available=cuda,
        training_profile=profile,
    )


if __name__ == "__main__":
    for k, v in inspect_environment().to_dict().items():
        print(f"{k:20s}: {v}")

"""Canonical filesystem locations for the project.

All paths are anchored to this file instead of the process working directory so
the commands behave the same from a terminal, an IDE, or a test runner.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "basic.cfg"
MODEL_ROOT = PROJECT_ROOT / "models"
RUN_ROOT = PROJECT_ROOT / "runs"
REPORT_ROOT = PROJECT_ROOT / "reports"


def make_run_id(seed: int, now: datetime | None = None) -> str:
    """Return a sortable, practically unique identifier for an experiment."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    return f"{timestamp}-seed{seed}"


def experiment_directories(model_name: str, run_id: str) -> tuple[Path, Path]:
    """Return the checkpoint and TensorBoard directories for an experiment."""
    normalized_name = model_name.lower()
    return (
        MODEL_ROOT / normalized_name / run_id,
        RUN_ROOT / normalized_name / run_id,
    )

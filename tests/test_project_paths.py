from __future__ import annotations

from datetime import datetime

from project_paths import (
    DEFAULT_CONFIG_PATH,
    MODEL_ROOT,
    PROJECT_ROOT,
    RUN_ROOT,
    experiment_directories,
    make_run_id,
)


def test_paths_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert PROJECT_ROOT.is_absolute()
    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "basic.cfg"
    assert DEFAULT_CONFIG_PATH.is_file()
    assert MODEL_ROOT == PROJECT_ROOT / "models"
    assert RUN_ROOT == PROJECT_ROOT / "runs"


def test_experiment_directories_separate_model_and_run():
    model_dir, run_dir = experiment_directories("SNN", "experiment-1")

    assert model_dir == MODEL_ROOT / "snn" / "experiment-1"
    assert run_dir == RUN_ROOT / "snn" / "experiment-1"


def test_run_id_contains_timestamp_and_seed():
    run_id = make_run_id(42, datetime(2026, 9, 15, 23, 30, 1, 123456))

    assert run_id == "20260915-233001-123456-seed42"


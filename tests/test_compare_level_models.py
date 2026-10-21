from __future__ import annotations

import json

import pytest

from compare_level_models import load_comparison, parse_report_spec, write_comparison


def _report(path, model: str, completion: float) -> None:
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "model": model,
                    "total_episodes": 10,
                    "completion_rate": completion,
                    "death_rate": 0.2,
                    "timeout_rate": 0.8 - completion,
                    "mean_reward": 1.5,
                    "median_reward": 1.0,
                    "mean_kills": 0.5,
                    "mean_sectors_visited": 2.0,
                    "mean_spike_rate": 0.01,
                    "mean_q_margin": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )


def test_comparison_load_and_write(tmp_path):
    cnn = tmp_path / "cnn.json"
    crsnn = tmp_path / "crsnn.json"
    _report(cnn, "level_cnn", 0.1)
    _report(crsnn, "crsnn", 0.3)

    rows = load_comparison([f"CNN={cnn}", f"CRSNN={crsnn}"])
    output = tmp_path / "comparison"
    write_comparison(rows, output)

    assert [row["label"] for row in rows] == ["CNN", "CRSNN"]
    assert rows[1]["completion_rate"] == 0.3
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.csv").is_file()
    assert (output / "comparison.png").is_file()


def test_report_spec_requires_label_and_existing_path(tmp_path):
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_report_spec("missing-separator")
    with pytest.raises(FileNotFoundError):
        parse_report_spec(f"model={tmp_path / 'missing.json'}")


def test_duplicate_labels_are_rejected(tmp_path):
    report = tmp_path / "report.json"
    _report(report, "crsnn", 0.2)
    with pytest.raises(ValueError, match="Duplicate report label"):
        load_comparison([f"same={report}", f"same={report}"])

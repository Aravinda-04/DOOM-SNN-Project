"""Compare diagnostic reports from full-level model experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPARISON_FIELDS = (
    "completion_rate",
    "death_rate",
    "timeout_rate",
    "mean_reward",
    "median_reward",
    "mean_kills",
    "mean_sectors_visited",
    "mean_spike_rate",
    "mean_q_margin",
)


def parse_report_spec(spec: str) -> tuple[str, Path]:
    """Parse a `label=path/to/report.json` command-line value."""
    try:
        label, raw_path = spec.split("=", 1)
    except ValueError as error:
        raise ValueError(
            f"Invalid report specification {spec!r}; use LABEL=PATH."
        ) from error
    label = label.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not label:
        raise ValueError("Report label cannot be empty.")
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostic report not found: {path}")
    return label, path


def load_comparison(specs: list[str]) -> list[dict]:
    """Load normalized aggregate rows while preserving caller order."""
    rows = []
    labels = set()
    for spec in specs:
        label, path = parse_report_spec(spec)
        if label in labels:
            raise ValueError(f"Duplicate report label: {label}")
        labels.add(label)
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError(f"Report has no aggregate object: {path}")
        row = {
            "label": label,
            "report": str(path),
            "model": aggregate.get("model", "unknown"),
            "total_episodes": int(aggregate.get("total_episodes", 0)),
        }
        for field in COMPARISON_FIELDS:
            value = aggregate.get(field, 0.0)
            row[field] = float(value)
        rows.append(row)
    return rows


def write_comparison(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps({"models": rows}, indent=2), encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        ("completion_rate", "Completion rate", (0.0, 1.0)),
        ("death_rate", "Death rate", (0.0, 1.0)),
        ("mean_reward", "Mean shaped reward", None),
        ("mean_sectors_visited", "Mean regions visited", None),
    )
    for axis, (field, title, limits) in zip(axes.flat, panels):
        values = [row[field] for row in rows]
        axis.bar(labels, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        if limits is not None:
            axis.set_ylim(*limits)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom")
    fig.suptitle("Full-level model comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Diagnostic report to compare; repeat once per model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_comparison(args.report)
    if len(rows) < 2:
        raise ValueError("At least two reports are required for comparison.")
    write_comparison(rows, args.output.expanduser().resolve())
    print(f"Comparison written to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()

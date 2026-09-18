from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a visual comparison of two diagnostic reports."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-label", default="Floating point")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return report["summary"]


def main() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    labels = [args.baseline_label, args.candidate_label]

    def category_rate(summary: dict, category: str) -> float:
        return summary["spawn_categories"][category]["success_rate"] * 100

    success_metrics = {
        "Overall": [baseline["success_rate"] * 100, candidate["success_rate"] * 100],
        "Hard left": [
            category_rate(baseline, "hard_left"),
            category_rate(candidate, "hard_left"),
        ],
        "Hard right": [
            category_rate(baseline, "hard_right"),
            category_rate(candidate, "hard_right"),
        ],
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x_positions = range(len(success_metrics))
    width = 0.36
    axes[0].bar(
        [x - width / 2 for x in x_positions],
        [values[0] for values in success_metrics.values()],
        width,
        label=labels[0],
    )
    axes[0].bar(
        [x + width / 2 for x in x_positions],
        [values[1] for values in success_metrics.values()],
        width,
        label=labels[1],
    )
    axes[0].set_xticks(list(x_positions), list(success_metrics))
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Hard-spawn behavior")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    reward_values = [baseline["mean_reward"], candidate["mean_reward"]]
    bars = axes[1].bar(labels, reward_values, color=("#4c78a8", "#f58518"))
    axes[1].set_ylabel("Mean reward")
    axes[1].set_title("Mean reward across matched episodes")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(bars, fmt="%.2f")

    figure.suptitle(f"{args.baseline_label} vs {args.candidate_label} verification")
    figure.tight_layout()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

"""Compare saved corridor reports across reward versions using task outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def measures(report):
    rows = report['episodes']
    if not rows:
        raise ValueError('Comparison requires at least one episode per report')
    directions = {
        direction: [row for row in rows if row['turn'] == direction]
        for direction in ('left', 'right')
    }
    successes = {
        direction: sum(row['outcome'] == 'complete' for row in group) / max(1, len(group))
        for direction, group in directions.items()
    }
    actions = Counter(action for row in rows for action in row['actions'])
    total_actions = sum(actions.values())
    names = report['action_names']
    frequencies = [actions[index] / max(1, total_actions) for index in range(len(names))]
    progress = [
        row.get('max_progress', max(
            (min(max(x, 0), 640) + abs(y)) / 1280 for x, y in row['trajectory']
        ))
        for row in rows
    ]
    return successes, frequencies, progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding='utf-8'))
               for path in (args.baseline, args.candidate)]
    if any(report.get('stage') != 'navigation' for report in reports):
        parser.error('This comparison expects navigation reports')
    if reports[0]['action_names'] != reports[1]['action_names']:
        parser.error('Reports must have the same action ordering')
    baseline, candidate = [measures(report) for report in reports]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['Baseline', 'Candidate']
    colors = ['steelblue', 'seagreen']
    x = np.arange(2)
    for index, (label, values, color) in enumerate(zip(labels, (baseline, candidate), colors)):
        axes[0, 0].bar(x + (index-.5)*.35,
                       [values[0][direction] for direction in ('left', 'right')],
                       width=.35, label=label, color=color)
    axes[0, 0].set(xticks=x, xticklabels=['Left', 'Right'], ylim=(0, 1),
                   ylabel='Completion rate', title='Completion by direction')
    axes[0, 0].legend()
    names = reports[0]['action_names']
    y = np.arange(len(names))
    axes[0, 1].barh(y-.17, baseline[1], height=.34, label='Baseline', color=colors[0])
    axes[0, 1].barh(y+.17, candidate[1], height=.34, label='Candidate', color=colors[1])
    axes[0, 1].set(yticks=y, yticklabels=names, xlabel='Fraction of decisions',
                   title='Action mix')
    axes[0, 1].legend()
    axes[1, 0].boxplot([baseline[2], candidate[2]], tick_labels=labels)
    axes[1, 0].set(title='Maximum route progress per episode', ylabel='Route fraction', ylim=(0, 1.1))
    for index, report in enumerate(reports):
        completed = [row['outcome'] == 'complete' for row in report['episodes']]
        axes[1, 1].plot(range(1, len(completed)+1), np.cumsum(completed) / np.arange(1, len(completed)+1),
                        marker='o', label=labels[index], color=colors[index])
    axes[1, 1].set(title='Cumulative completion', xlabel='Episode', ylabel='Completion rate', ylim=(0, 1))
    axes[1, 1].legend()
    fig.suptitle('Corridor policies | task outcomes (rewards omitted)')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()

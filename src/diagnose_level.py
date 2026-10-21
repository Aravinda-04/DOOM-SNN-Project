"""Full-level DOOM diagnostics for active CNN, DRQN, and spiking agents.

Runs a configurable number of greedy episodes on a full level and produces:
  - reports/level/<run-id>/report.json         (aggregate + per-episode stats)
  - reports/level/<run-id>/episodes.csv        (flat per-episode table)
  - reports/level/<run-id>/tensorboard/        (scalar curves)
  - reports/level/<run-id>/trajectory_N.png    (top-down position traces)

Usage (from project root):

    python src/diagnose_level.py \\
        --model crsnn \\
        --checkpoint models/crsnn/<run-id>/best.pth \\
        --episodes 20 --seeds 0 1 2 3 4 --device auto

Render a few episodes for visual verification:

    python src/diagnose_level.py \\
        --model crsnn \\
        --checkpoint models/crsnn/<run-id>/best.pth \\
        --episodes 3 --seeds 0 --render --delay 0.05
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    AGENT_CONTRACT,
    FRAME_STACK,
)
from checkpointing import load_checkpoint
from doom_level_env import DoomLevelEnvironment
from networks import build_model
from project_paths import LEVEL_CONFIG_PATH, REPORT_ROOT
from runtime_utils import resolve_device, set_random_seeds

LEVEL_MODELS = ("crsnn", "rsnn_v2", "level_cnn", "level_drqn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _plot_trajectory(
    positions: list[tuple[float, float]],
    out_path: Path,
    title: str,
    completion: bool,
) -> None:
    """Save a top-down position trace to ``out_path``."""
    if not _HAS_MPL or len(positions) < 2:
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="plasma", s=2, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Step")
    ax.plot(xs[0], ys[0], "go", markersize=8, label="Start")
    ax.plot(xs[-1], ys[-1], "r*" if not completion else "g*",
            markersize=12, label="End")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: DoomLevelEnvironment,
    policy_net: torch.nn.Module,
    device: torch.device,
    render_delay: float = 0.0,
) -> dict:
    """Run one greedy episode and return a result dict."""
    obs = env.reset()
    hidden: Optional[tuple] = None
    done = False
    total_reward = 0.0
    steps = 0
    spike_counts: list[float] = []
    q_margins: list[float] = []
    positions: list[tuple[float, float]] = []
    action_hist = [0] * AGENT_CONTRACT.action_size
    reward_components: dict[str, float] = {}

    while not done:
        t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, spikes, hidden = policy_net(t, hidden)

        q_np = q_values.squeeze(0).cpu().numpy()
        sorted_q = np.sort(q_np)[::-1]
        action = int(np.argmax(q_np))
        margin = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0

        obs, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        action_hist[action] += 1
        opportunities = getattr(policy_net, "spike_opportunities_per_sample", None)
        if opportunities is None:
            spike_counts.append(0.0)
        elif opportunities <= 0:
            raise ValueError("spike_opportunities_per_sample must be positive.")
        else:
            spike_counts.append(float(spikes.item()) / opportunities)
        q_margins.append(margin)

        info = env.diagnostics()
        positions.append((info.get("position_x", 0.0), info.get("position_y", 0.0)))
        for name, value in info.get("last_reward_components", {}).items():
            if name not in {"unscaled_total", "scaled_total"}:
                reward_components[name] = reward_components.get(name, 0.0) + float(value)

        if render_delay > 0:
            time.sleep(render_delay)

    final_info = env.diagnostics()
    mean_spikes = float(np.mean(spike_counts)) if spike_counts else 0.0
    mean_margin = float(np.mean(q_margins)) if q_margins else 0.0

    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminal_reason": final_info.get("terminal_reason", "unknown"),
        "completion": bool(final_info.get("exit_triggered", False)),
        "player_dead": bool(final_info.get("player_dead", False)),
        "kill_count": float(final_info.get("kill_count", 0.0)),
        "item_count": float(final_info.get("item_count", 0.0)),
        "secret_count": float(final_info.get("secret_count", 0.0)),
        "sectors_visited": int(final_info.get("sectors_visited", 0)),
        "episode_time_tics": int(final_info.get("episode_time", 0)),
        "mean_spike_rate": mean_spikes,
        "mean_q_margin": mean_margin,
        "raw_game_reward": reward_components.get("raw_game_reward", 0.0),
        "reward_components": reward_components,
        "action_histogram": action_hist,
        "positions": positions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose a full-level DOOM agent.")
    p.add_argument("--model", choices=LEVEL_MODELS, default="crsnn")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the .pth checkpoint to evaluate.")
    p.add_argument("--episodes", type=int, default=20,
                   help="Episodes per seed.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--config", type=Path, default=LEVEL_CONFIG_PATH,
                   help="ViZDoom config (default: scenarios/level01.cfg).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--render", action="store_true")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to sleep between steps when rendering.")
    p.add_argument("--episode-timeout", type=int, default=None,
                   help="Override scenario timeout in tics for smoke evaluation.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.episode_timeout is not None and args.episode_timeout <= 0:
        raise ValueError("--episode-timeout must be positive.")
    set_random_seeds(args.seed)
    device = resolve_device(args.device)

    # Create output directory.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir = REPORT_ROOT / "level" / f"{args.model}-{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = report_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model     : {args.model.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Report dir: {report_dir}")
    print(f"Seeds     : {args.seeds}  Episodes/seed: {args.episodes}")

    # Load model.
    checkpoint_path = args.checkpoint.expanduser().resolve()
    policy_net = build_model(
        args.model,
        action_size=AGENT_CONTRACT.action_size,
        in_channels=FRAME_STACK,
    ).to(device)
    checkpoint_metadata = load_checkpoint(checkpoint_path, policy_net, device)
    policy_net.eval()

    writer = SummaryWriter(log_dir=str(tb_dir))
    all_episodes: list[dict] = []
    global_ep = 0

    for seed in args.seeds:
        with DoomLevelEnvironment(
            config_file=args.config,
            render=args.render,
            seed=seed,
            episode_timeout=args.episode_timeout,
        ) as env:
            for ep_idx in range(args.episodes):
                result = run_episode(env, policy_net, device, render_delay=args.delay)
                result["seed"] = seed
                result["episode_index"] = ep_idx
                all_episodes.append(result)

                # TensorBoard step = global episode counter.
                writer.add_scalar("diag/reward", result["total_reward"], global_ep)
                writer.add_scalar("diag/raw_game_reward",
                                  result["raw_game_reward"], global_ep)
                writer.add_scalar("diag/steps", result["steps"], global_ep)
                writer.add_scalar("diag/completion",
                                  float(result["completion"]), global_ep)
                writer.add_scalar("diag/kills", result["kill_count"], global_ep)
                writer.add_scalar("diag/sectors_visited",
                                  result["sectors_visited"], global_ep)
                writer.add_scalar("diag/mean_spike_rate",
                                  result["mean_spike_rate"], global_ep)
                writer.add_scalar("diag/mean_q_margin",
                                  result["mean_q_margin"], global_ep)

                # Trajectory plot.
                traj_path = report_dir / f"trajectory_ep{global_ep:04d}.png"
                _plot_trajectory(
                    result["positions"],
                    traj_path,
                    title=(
                        f"seed={seed} ep={ep_idx}  "
                        f"r={result['total_reward']:+.1f}  "
                        f"{'DONE' if result['completion'] else 'DEAD' if result['player_dead'] else 'TIMEOUT'}"
                    ),
                    completion=result["completion"],
                )

                status = (
                    "COMPLETE" if result["completion"]
                    else "DEAD" if result["player_dead"]
                    else "TIMEOUT"
                )
                print(
                    f"  seed={seed} ep={ep_idx+1:>3}  {status}  "
                    f"r={result['total_reward']:+.2f}  "
                    f"kills={result['kill_count']:.0f}  "
                    f"sectors={result['sectors_visited']}  "
                    f"steps={result['steps']}"
                )
                global_ep += 1

    writer.close()

    # ── Aggregate stats ──────────────────────────────────────────────────────
    rewards = [e["total_reward"] for e in all_episodes]
    kills = [e["kill_count"] for e in all_episodes]
    completions = [e["completion"] for e in all_episodes]
    sectors = [e["sectors_visited"] for e in all_episodes]
    spikes = [e["mean_spike_rate"] for e in all_episodes]
    margins = [e["mean_q_margin"] for e in all_episodes]

    aggregate = {
        "model": args.model,
        "agent_contract_sha256": ACTION_CATALOGUE_HASH,
        "action_names": ACTION_NAMES,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_format_version": int(
            checkpoint_metadata.get("format_version", 0)
        ),
        "checkpoint_model_signature": checkpoint_metadata.get("model_signature"),
        "checkpoint_episode": int(checkpoint_metadata.get("episode", -1)),
        "checkpoint_global_step": int(
            checkpoint_metadata.get("global_step", 0)
        ),
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "total_episodes": len(all_episodes),
        "completion_rate": float(np.mean(completions)),
        "death_rate": float(np.mean([e["player_dead"] for e in all_episodes])),
        "timeout_rate": float(np.mean([
            e["terminal_reason"] == "timed_out" for e in all_episodes
        ])),
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_kills": float(np.mean(kills)),
        "mean_sectors_visited": float(np.mean(sectors)),
        "mean_spike_rate": float(np.mean(spikes)),
        "mean_q_margin": float(np.mean(margins)),
    }

    print("\n-- Aggregate Results --")
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")

    # ── Write report.json ────────────────────────────────────────────────────
    # Remove position traces from JSON (too large).
    json_episodes = [
        {k: v for k, v in ep.items() if k != "positions"}
        for ep in all_episodes
    ]
    report = {"aggregate": aggregate, "episodes": json_episodes}
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ── Write episodes.csv ───────────────────────────────────────────────────
    csv_path = report_dir / "episodes.csv"
    fieldnames = [
        "seed", "episode_index", "total_reward", "steps", "completion",
        "terminal_reason", "player_dead", "raw_game_reward", "kill_count",
        "item_count", "secret_count",
        "sectors_visited", "episode_time_tics", "mean_spike_rate", "mean_q_margin",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer_csv.writeheader()
        writer_csv.writerows(json_episodes)

    print(f"\nReport written to: {report_dir}")


if __name__ == "__main__":
    main()

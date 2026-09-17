from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpointing import checkpoint_action_size, load_checkpoint
from env import DoomEnvironment
from eval import find_default_checkpoint
from networks import MODEL_NAMES, build_model
from project_paths import DEFAULT_CONFIG_PATH, REPORT_ROOT
from runtime_utils import resolve_device, set_random_seeds


class SpikeActivityRecorder:
    """Collect spike counts from every LIF layer without changing the model."""

    def __init__(self, model: torch.nn.Module):
        self.spikes: Counter[str] = Counter()
        self.opportunities: Counter[str] = Counter()
        self.handles = []
        for name, module in model.named_modules():
            if name.startswith("lif"):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def record(_module, _inputs, output):
            spike_tensor = output[0] if isinstance(output, tuple) else output
            self.spikes[name] += float(spike_tensor.detach().sum())
            self.opportunities[name] += spike_tensor.numel()

        return record

    def reset(self) -> None:
        self.spikes.clear()
        self.opportunities.clear()

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "spikes": self.spikes[name],
                "opportunities": self.opportunities[name],
                "rate": self.spikes[name] / self.opportunities[name]
                if self.opportunities[name]
                else 0.0,
            }
            for name in sorted(self.opportunities)
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_termination(start_info: dict, end_info: dict) -> tuple[bool, str]:
    start_kills = start_info.get("kill_count") or 0.0
    end_kills = end_info.get("kill_count") or 0.0
    if end_kills > start_kills:
        return True, "kill"
    if end_info.get("player_dead"):
        return False, "death"
    return False, "timeout"


def position_distance(before: dict, after: dict) -> float | None:
    coordinates = (
        before.get("position_x"),
        before.get("position_y"),
        after.get("position_x"),
        after.get("position_y"),
    )
    if any(value is None for value in coordinates):
        return None
    before_x, before_y, after_x, after_y = coordinates
    return math.hypot(after_x - before_x, after_y - before_y)


def classify_spawn(start_info: dict, easy_target_offset: float) -> str:
    """Classify the initial target view without using policy behavior."""
    offset = start_info.get("target_horizontal_offset")
    if not start_info.get("target_visible") or offset is None:
        return "invisible"
    if abs(offset) <= easy_target_offset:
        return "easy"
    return "hard_left" if offset < 0 else "hard_right"


def spawn_matches(spawn_filter: str, spawn_category: str) -> bool:
    if spawn_filter == "all":
        return True
    if spawn_filter == "hard":
        return spawn_category != "easy"
    if spawn_filter == "left":
        return spawn_category == "hard_left"
    if spawn_filter == "right":
        return spawn_category == "hard_right"
    return spawn_category == spawn_filter


def run_episode(
    env: DoomEnvironment,
    policy_net: torch.nn.Module,
    device: torch.device,
    recorder: SpikeActivityRecorder,
    *,
    render: bool,
    delay: float,
    instant_kill_steps: int,
    easy_target_offset: float,
    initial_state: np.ndarray | None = None,
    start_info: dict | None = None,
) -> dict:
    state = env.reset() if initial_state is None else initial_state
    start_info = env.diagnostics() if start_info is None else start_info
    recorder.reset()

    rewards = []
    actions = []
    q_samples = []
    q_margins = []
    positions = []
    stalled_move_steps = 0
    stalled_move_streak = 0
    longest_stalled_move_streak = 0
    longest_action_streak = 0
    longest_attack_streak = 0
    current_attack_streak = 0
    current_action_streak = 0
    previous_action = None
    done = False

    while not done:
        before_info = env.diagnostics()
        if before_info["position_x"] is not None:
            positions.append(
                (before_info["position_x"], before_info["position_y"])
            )

        state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, _ = policy_net(state_tensor)
        q_row = q_values[0].detach().cpu().tolist()
        action = int(q_values.argmax(1).item())
        sorted_q = torch.topk(q_values[0], k=min(2, q_values.shape[1])).values
        q_margin = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0

        next_state, reward, done = env.step(action)
        after_info = env.diagnostics()
        distance = position_distance(before_info, after_info)

        action_name = env.action_names[action]
        is_move = action_name in ("move_left", "move_right")
        is_stalled_move = is_move and distance is not None and distance < 0.01
        if is_stalled_move:
            stalled_move_steps += 1
            stalled_move_streak += 1
            longest_stalled_move_streak = max(
                longest_stalled_move_streak, stalled_move_streak
            )
        else:
            stalled_move_streak = 0

        if action == previous_action:
            current_action_streak += 1
        else:
            current_action_streak = 1
            previous_action = action
        longest_action_streak = max(longest_action_streak, current_action_streak)
        if action_name == "attack":
            current_attack_streak += 1
            longest_attack_streak = max(longest_attack_streak, current_attack_streak)
        else:
            current_attack_streak = 0

        actions.append(action_name)
        rewards.append(float(reward))
        q_samples.append(q_row)
        q_margins.append(q_margin)
        state = next_state
        if render and delay:
            time.sleep(delay)

    end_info = env.diagnostics()
    success, termination = classify_termination(start_info, end_info)
    instant_kill = success and len(actions) <= instant_kill_steps
    initial_offset = start_info["target_horizontal_offset"]
    spawn_category = classify_spawn(start_info, easy_target_offset)
    easy_spawn = spawn_category == "easy"
    action_counts = Counter(actions)
    movement_steps = action_counts["move_left"] + action_counts["move_right"]
    attack_fraction = action_counts["attack"] / len(actions)
    q_array = np.asarray(q_samples, dtype=np.float64)
    spike_activity = recorder.snapshot()
    wall_loop_suspected = not success and (
        longest_stalled_move_streak >= 3
        or (attack_fraction >= 0.7 and longest_attack_streak >= 10)
    )

    result = {
        "reward": float(sum(rewards)),
        "success": success,
        "instant_kill": instant_kill,
        "noninstant_success": success and not instant_kill,
        "easy_spawn": easy_spawn,
        "hard_spawn": not easy_spawn,
        "hard_spawn_success": success and not easy_spawn,
        "spawn_category": spawn_category,
        "initial_target_visible": start_info["target_visible"],
        "initial_target_horizontal_offset": initial_offset,
        "termination": termination,
        "decision_steps": len(actions),
        # ViZDoom resets get_episode_time() to zero after a terminal action.
        "game_tics": len(actions) * 4,
        "ammo_start": start_info["ammo"],
        "ammo_end": end_info["ammo"],
        "ammo_used": (
            start_info["ammo"] - end_info["ammo"]
            if start_info["ammo"] is not None and end_info["ammo"] is not None
            else None
        ),
        "action_counts": dict(action_counts),
        "action_fractions": {
            name: action_counts[name] / len(actions) for name in env.action_names
        },
        "action_sequence": actions,
        "first_20_actions": actions[:20],
        "longest_action_streak": longest_action_streak,
        "longest_attack_streak": longest_attack_streak,
        "stalled_move_steps": stalled_move_steps,
        "stalled_move_fraction": (
            stalled_move_steps / movement_steps if movement_steps else 0.0
        ),
        "longest_stalled_move_streak": longest_stalled_move_streak,
        "wall_loop_suspected": wall_loop_suspected,
        "position_x_span": (
            max(position[0] for position in positions)
            - min(position[0] for position in positions)
            if positions
            else None
        ),
        "position_y_span": (
            max(position[1] for position in positions)
            - min(position[1] for position in positions)
            if positions
            else None
        ),
        "mean_q_values": {
            name: float(q_array[:, index].mean())
            for index, name in enumerate(env.action_names)
        },
        "mean_q_margin": float(np.mean(q_margins)),
        "minimum_q_margin": float(np.min(q_margins)),
        "spike_activity": spike_activity,
    }
    return result


def aggregate_results(results: list[dict], action_names: tuple[str, ...]) -> dict:
    rewards = np.asarray([result["reward"] for result in results])
    successes = np.asarray([result["success"] for result in results], dtype=float)
    instant_kills = sum(result["instant_kill"] for result in results)
    noninstant_successes = sum(result["noninstant_success"] for result in results)
    hard_spawn_episodes = sum(result["hard_spawn"] for result in results)
    hard_spawn_successes = sum(result["hard_spawn_success"] for result in results)
    terminations = Counter(result["termination"] for result in results)
    spawn_counts = Counter(result["spawn_category"] for result in results)
    spawn_successes = Counter(
        result["spawn_category"] for result in results if result["success"]
    )
    action_counts = Counter()
    layer_spikes: Counter[str] = Counter()
    layer_opportunities: Counter[str] = Counter()

    for result in results:
        action_counts.update(result["action_counts"])
        for name, activity in result["spike_activity"].items():
            layer_spikes[name] += activity["spikes"]
            layer_opportunities[name] += activity["opportunities"]

    total_actions = sum(action_counts.values())
    return {
        "episodes": len(results),
        "successes": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "instant_kills": instant_kills,
        "instant_kill_rate": instant_kills / len(results),
        "noninstant_successes": noninstant_successes,
        "noninstant_success_rate": noninstant_successes / len(results),
        "hard_spawn_episodes": hard_spawn_episodes,
        "hard_spawn_successes": hard_spawn_successes,
        "hard_spawn_success_rate": (
            hard_spawn_successes / hard_spawn_episodes if hard_spawn_episodes else 0.0
        ),
        "spawn_categories": {
            category: {
                "episodes": spawn_counts[category],
                "successes": spawn_successes[category],
                "success_rate": spawn_successes[category] / spawn_counts[category],
            }
            for category in sorted(spawn_counts)
        },
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "median_reward": float(np.median(rewards)),
        "minimum_reward": float(rewards.min()),
        "maximum_reward": float(rewards.max()),
        "mean_decision_steps": float(
            np.mean([result["decision_steps"] for result in results])
        ),
        "termination_counts": dict(terminations),
        "wall_loop_episodes": sum(
            result["wall_loop_suspected"] for result in results
        ),
        "action_counts": dict(action_counts),
        "action_fractions": {
            name: action_counts[name] / total_actions if total_actions else 0.0
            for name in action_names
        },
        "spike_rates": {
            name: layer_spikes[name] / layer_opportunities[name]
            if layer_opportunities[name]
            else 0.0
            for name in sorted(layer_opportunities)
        },
    }


def flatten_episode(result: dict) -> dict:
    row = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "action_counts",
            "action_fractions",
            "action_sequence",
            "first_20_actions",
            "mean_q_values",
            "spike_activity",
        }
    }
    row["first_20_actions"] = " ".join(result["first_20_actions"])
    for name, value in result["action_counts"].items():
        row[f"action_count_{name}"] = value
    for name, value in result["action_fractions"].items():
        row[f"action_fraction_{name}"] = value
    for name, value in result["mean_q_values"].items():
        row[f"mean_q_{name}"] = value
    for name, value in result["spike_activity"].items():
        row[f"spike_rate_{name}"] = value["rate"]
    return row


def write_csv(path: Path, results: list[dict]) -> None:
    rows = [flatten_episode(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose an autonomous ViZDoom policy over multiple seeds."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="snn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--instant-kill-steps", type=int, default=2)
    parser.add_argument("--easy-target-offset", type=float, default=0.1)
    parser.add_argument(
        "--spawn-filter",
        choices=("all", "easy", "hard", "left", "right", "invisible"),
        default="all",
        help="Only execute episodes matching this initial-spawn category.",
    )
    parser.add_argument(
        "--max-spawn-attempts",
        type=int,
        default=10_000,
        help="Maximum resets per seed while searching for accepted spawns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if args.instant_kill_steps < 1:
        raise ValueError("--instant-kill-steps must be positive.")
    if not 0 <= args.easy_target_offset <= 1:
        raise ValueError("--easy-target-offset must be between 0 and 1.")
    if args.max_spawn_attempts < args.episodes:
        raise ValueError("--max-spawn-attempts must be at least --episodes.")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates.")

    device = resolve_device(args.device)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else find_default_checkpoint(args.model)
    )
    checkpoint_label = (
        checkpoint_path.parent.name
        if checkpoint_path.name in ("best.pth", "latest.pth")
        else checkpoint_path.stem
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_dir = (
        args.report_dir.expanduser().resolve()
        if args.report_dir is not None
        else REPORT_ROOT / args.model / checkpoint_label / timestamp
    )
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Report directory is not empty: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)

    set_random_seeds(seeds[0])
    action_size = checkpoint_action_size(checkpoint_path)
    policy_net = build_model(args.model, action_size=action_size).to(device)
    checkpoint = load_checkpoint(checkpoint_path, policy_net, device)
    checkpoint_model = checkpoint.get("config", {}).get("model")
    if checkpoint_model and checkpoint_model != args.model:
        raise ValueError(
            f"Checkpoint contains model '{checkpoint_model}', not '{args.model}'."
        )
    policy_net.eval()
    recorder = SpikeActivityRecorder(policy_net)
    writer = SummaryWriter(log_dir=str(report_dir / "tensorboard"))

    print(f"Diagnosing {args.model.upper()} on {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Seeds: {', '.join(map(str, seeds))}")
    print(f"Spawn filter: {args.spawn_filter}")
    print(f"Report: {report_dir}")

    results = []
    action_names: tuple[str, ...] = ()
    global_episode = 0
    spawn_sampling = {}
    try:
        for seed in seeds:
            set_random_seeds(seed)
            with DoomEnvironment(
                config_file=args.config, render=args.render, seed=seed
            ) as env:
                if len(env.actions) != action_size:
                    raise ValueError(
                        f"Checkpoint has {action_size} actions but the scenario has "
                        f"{len(env.actions)}. Use the matching scenario configuration."
                    )
                action_names = env.action_names
                accepted = 0
                attempts = 0
                skipped_categories: Counter[str] = Counter()
                while accepted < args.episodes:
                    if attempts >= args.max_spawn_attempts:
                        raise RuntimeError(
                            f"Seed {seed}: found only {accepted}/{args.episodes} "
                            f"'{args.spawn_filter}' spawns after {attempts} resets. "
                            "Increase --max-spawn-attempts or use a broader filter."
                        )
                    initial_state = env.reset()
                    start_info = env.diagnostics()
                    attempts += 1
                    spawn_category = classify_spawn(
                        start_info, args.easy_target_offset
                    )
                    if not spawn_matches(args.spawn_filter, spawn_category):
                        skipped_categories[spawn_category] += 1
                        continue

                    result = run_episode(
                        env,
                        policy_net,
                        device,
                        recorder,
                        render=args.render,
                        delay=args.delay,
                        instant_kill_steps=args.instant_kill_steps,
                        easy_target_offset=args.easy_target_offset,
                        initial_state=initial_state,
                        start_info=start_info,
                    )
                    result["seed"] = seed
                    result["episode"] = accepted + 1
                    result["spawn_attempt"] = attempts
                    results.append(result)

                    writer.add_scalar("diagnostics/reward", result["reward"], global_episode)
                    writer.add_scalar(
                        "diagnostics/success", int(result["success"]), global_episode
                    )
                    writer.add_scalar(
                        "diagnostics/noninstant_success",
                        int(result["noninstant_success"]),
                        global_episode,
                    )
                    writer.add_scalar(
                        "diagnostics/hard_spawn_success",
                        int(result["hard_spawn_success"]),
                        global_episode,
                    )
                    writer.add_scalar(
                        "diagnostics/decision_steps",
                        result["decision_steps"],
                        global_episode,
                    )
                    writer.add_scalar(
                        "diagnostics/stalled_move_fraction",
                        result["stalled_move_fraction"],
                        global_episode,
                    )
                    writer.add_scalar(
                        "diagnostics/q_margin",
                        result["mean_q_margin"],
                        global_episode,
                    )
                    for name, activity in result["spike_activity"].items():
                        writer.add_scalar(
                            f"diagnostics/spike_rate/{name}",
                            activity["rate"],
                            global_episode,
                        )
                    global_episode += 1
                    accepted += 1
                    print(
                        f"Seed {seed} | Episode {accepted}/{args.episodes} | "
                        f"{result['spawn_category']} | "
                        f"{result['termination']} | Reward {result['reward']:.1f} | "
                        f"Steps {result['decision_steps']} | "
                        f"Stalled moves {result['stalled_move_steps']}"
                    )
                spawn_sampling[str(seed)] = {
                    "attempts": attempts,
                    "accepted": accepted,
                    "skipped": attempts - accepted,
                    "skipped_categories": dict(skipped_categories),
                }
                print(
                    f"Seed {seed} sampling | Accepted {accepted} | "
                    f"Skipped {attempts - accepted} | Attempts {attempts}"
                )
    finally:
        recorder.close()
        writer.close()

    summary = aggregate_results(results, action_names)
    per_seed = {
        str(seed): aggregate_results(
            [result for result in results if result["seed"] == seed], action_names
        )
        for seed in seeds
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "device": str(device),
        "episodes_per_seed": args.episodes,
        "instant_kill_steps": args.instant_kill_steps,
        "easy_target_offset": args.easy_target_offset,
        "spawn_filter": args.spawn_filter,
        "max_spawn_attempts_per_seed": args.max_spawn_attempts,
        "spawn_sampling": spawn_sampling,
        "seeds": seeds,
        "summary": summary,
        "per_seed": per_seed,
        "episodes": results,
    }
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(report_dir / "episodes.csv", results)

    print(
        f"Success rate: {summary['success_rate']:.1%} "
        f"({summary['successes']}/{summary['episodes']})"
    )
    print(
        f"Non-instant success rate: {summary['noninstant_success_rate']:.1%} "
        f"({summary['noninstant_successes']}/{summary['episodes']})"
    )
    print(
        f"Hard-spawn success rate: {summary['hard_spawn_success_rate']:.1%} "
        f"({summary['hard_spawn_successes']}/{summary['hard_spawn_episodes']})"
    )
    print(
        f"Reward: {summary['mean_reward']:.1f} +/- {summary['std_reward']:.1f} | "
        f"Median: {summary['median_reward']:.1f}"
    )
    print(
        f"Wall-loop indicator: {summary['wall_loop_episodes']}/"
        f"{summary['episodes']} episodes"
    )
    print(f"Wrote {report_dir / 'report.json'}")
    print(f"Wrote {report_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()

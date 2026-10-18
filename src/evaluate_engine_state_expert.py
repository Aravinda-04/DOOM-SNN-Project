"""Run the scripted privileged-state expert through the full MAP01 level."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from doom_level_env import DoomLevelEnvironment
from engine_state_config import FactorizedAction, compose_buttons
from engine_state_expert import EngineStateExpert, ExpertConfig
from engine_state_observation import EngineStateBuilder
from map01_route import Map01Route
from project_paths import PROJECT_ROOT


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def action_name(action: FactorizedAction) -> str:
    parts: list[str] = []
    if action.locomotion.name != "NONE":
        parts.append(action.locomotion.name.lower())
    if action.turning.name != "NONE":
        parts.append(f"turn_{action.turning.name.lower()}")
    if action.attack:
        parts.append("attack")
    if action.use:
        parts.append("use")
    return "+".join(parts) if parts else "noop"


def run_expert(
    *,
    dataset_dir: Path,
    seed: int,
    render: bool,
    render_delay: float,
    episode_timeout: int | None,
    progress_interval: int,
    expert_config: ExpertConfig = ExpertConfig(),
) -> dict:
    route = Map01Route.from_demonstrations(dataset_dir)
    expert = EngineStateExpert(expert_config)
    builder = EngineStateBuilder()
    counts: Counter[str] = Counter()
    previous_action = FactorizedAction()
    route_index = 0
    total_reward = 0.0
    steps = 0

    with DoomLevelEnvironment(
        render=render,
        seed=seed,
        episode_timeout=episode_timeout,
    ) as env:
        env.reset()
        expert.reset()
        builder.reset()
        done = False
        while not done:
            info = env.diagnostics()
            position = np.asarray(
                (info["position_x"], info["position_y"]), dtype=np.float32
            )
            route_context = route.context(position, route_index)
            route_index = route_context.progress_index
            state = env.game.get_state()
            labels = () if state is None else state.labels
            observation = builder.observe(
                info,
                labels,
                route_context,
                previous_action,
            )
            action = expert.act(
                observation,
                route_context,
                player_angle=float(info["angle"]),
                ammunition=float(info["ammo2"] + info["ammo3"]),
            )
            _, reward, done = env.step_buttons(compose_buttons(action))
            previous_action = action
            counts[action_name(action)] += 1
            total_reward += reward
            steps += 1
            if render and render_delay:
                time.sleep(render_delay)
            if progress_interval and steps % progress_interval == 0:
                print(
                    f"steps={steps} route={route_context.progress_fraction:.1%} "
                    f"regions={info['sectors_visited']} kills={info['kill_count']:.0f} "
                    f"health={info['health']:.0f} ammo={info['ammo2'] + info['ammo3']:.0f} "
                    f"enemy={observation.visible_enemy_count} "
                    f"stuck={observation.stuck_steps} "
                    f"action={action_name(action)}",
                    flush=True,
                )
        final = env.diagnostics()

    return {
        "completed": bool(final["exit_triggered"]),
        "terminal_reason": final["terminal_reason"],
        "steps": steps,
        "reward": float(total_reward),
        "kills": int(final["kill_count"]),
        "regions": int(final["sectors_visited"]),
        "route_progress": route_index / max(len(route.points) - 1, 1),
        "route_hash": route.sha256,
        "action_counts": dict(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the scripted engine-state expert on MAP01."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.0)
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_delay < 0:
        raise ValueError("--render-delay cannot be negative.")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval cannot be negative.")
    result = run_expert(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        seed=args.seed,
        render=args.render,
        render_delay=args.render_delay,
        episode_timeout=args.episode_timeout,
        progress_interval=args.progress_interval,
    )
    print("Engine-state expert evaluation complete.")
    print(f"Completed      : {result['completed']}")
    print(f"Terminal       : {result['terminal_reason']}")
    print(f"Decisions      : {result['steps']}")
    print(f"Route progress : {result['route_progress']:.1%}")
    print(f"Regions        : {result['regions']}")
    print(f"Kills          : {result['kills']}")
    print(f"Reward         : {result['reward']:+.2f}")
    print(f"Actions        : {result['action_counts']}")
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate synchronized vision-to-state training sequences for Stage 2 Vision Bridge."""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from doom_level_env import DoomLevelEnvironment
from engine_state_config import (
    ENGINE_STATE_CONTRACT_HASH,
    ENGINE_STATE_CONTRACT_VERSION,
    FactorizedAction,
    Locomotion,
    Turning,
    action_targets,
    compose_buttons,
)
from engine_state_observation import (
    EngineStateBuilder,
    visible_enemies,
    visible_health_items,
)
from evaluate_engine_state_policy import load_policy, predicted_action
from health_tactics import HealthTacticsArbitrator, HealthTacticsConfig
from map01_route import Map01Route, RouteContext
from project_paths import PROJECT_ROOT
from runtime_utils import resolve_device, set_random_seeds
from vision_data_schema import (
    VISION_SCHEMA_VERSION,
    append_vision_manifest_entry,
    save_vision_sequence,
)


DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated" / "map01-vision-v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "exports" / "best-engine-state-spiking-snn-v2.pth"
PASSING_SEEDS = (0, 1, 3, 4, 5, 42, 100, 444)


def generate_vision_episode(
    *,
    route: Map01Route,
    seed: int,
    model,
    device: torch.device,
    episode_timeout: int | None = 2100,
    health_arbitrator: HealthTacticsArbitrator | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Simulate one episode capturing aligned (frames, features, validity, targets)."""
    builder = EngineStateBuilder()
    if health_arbitrator is not None:
        health_arbitrator.reset()

    previous_action = FactorizedAction()
    route_index = 0
    frames_list: list[np.ndarray] = []
    features: list[np.ndarray] = []
    feature_valid: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    applied_targets: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    positions: list[tuple[float, float, float]] = []
    route_progress: list[float] = []

    hidden = None
    shotgun_cooldown_ticks = 0
    previous_ammo = None
    step_count = 0

    with DoomLevelEnvironment(
        render=False,
        seed=seed,
        episode_timeout=episode_timeout,
    ) as env:
        current_obs = env.reset()
        builder.reset()
        done = False

        while not done:
            info = env.diagnostics()
            current_ammo = float(info["ammo2"] + info["ammo3"])
            if previous_ammo is not None and current_ammo < previous_ammo:
                shotgun_cooldown_ticks = 28
            previous_ammo = current_ammo
            shotgun_cooldown_ticks = max(0, shotgun_cooldown_ticks - 1)

            position = np.asarray((info["position_x"], info["position_y"]), dtype=np.float32)
            context = route.context(position, route_index)
            route_index = context.progress_index
            state = env.game.get_state()
            labels = () if state is None else state.labels

            # Health detour arbitration
            if health_arbitrator is not None:
                health_items = visible_health_items(
                    labels, player_x=float(info["position_x"]), player_y=float(info["position_y"])
                )
                enemies = visible_enemies(
                    labels, player_x=float(info["position_x"]), player_y=float(info["position_y"])
                )
                nearest_enemy = enemies[0] if enemies else None
                subgoal = health_arbitrator.step(
                    player_health=float(info["health"]),
                    visible_items=health_items,
                    nearest_enemy=nearest_enemy,
                    player_angle=float(info["angle"]),
                )
                if subgoal is not None:
                    target_pos = np.asarray(subgoal.position, dtype=np.float32)
                    delta = target_pos - position
                    dist = float(np.linalg.norm(delta))
                    bearing = float(math.degrees(math.atan2(delta[1], delta[0])) % 360.0)
                    context = RouteContext(
                        progress_index=context.progress_index,
                        progress_fraction=context.progress_fraction,
                        target=target_pos,
                        delta=delta,
                        distance=dist,
                        bearing_degrees=bearing,
                        off_track=context.off_track,
                        use_recommended=False,
                        recovery=False,
                    )

            observation = builder.observe(info, labels, context, previous_action)

            # SNN inference
            policy_input = np.concatenate(
                (observation.features, observation.validity.astype(np.float32))
            )
            tensor = torch.from_numpy(policy_input).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs, hidden = model(tensor, hidden)

            action = predicted_action(
                outputs,
                observation=observation,
                route=context,
                ammunition=current_ammo,
                player_angle=float(info["angle"]),
                step=step_count,
                weapon_cooldown=shotgun_cooldown_ticks,
            )

            # Store current frame stack as uint8 [0, 255]
            uint8_frame = np.clip(current_obs * 255.0, 0, 255).astype(np.uint8)
            frames_list.append(uint8_frame)
            features.append(observation.features)
            feature_valid.append(observation.validity)
            targets.append(action_targets(action))
            applied_targets.append(action_targets(action))
            positions.append(
                (float(info["position_x"]), float(info["position_y"]), float(info["angle"]))
            )
            route_progress.append(float(context.progress_fraction))

            # Step environment
            next_obs, reward, done = env.step_buttons(compose_buttons(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            previous_action = action
            current_obs = next_obs
            step_count += 1

        final = env.diagnostics()

    metadata = {
        "schema_version": VISION_SCHEMA_VERSION,
        "engine_state_contract_version": ENGINE_STATE_CONTRACT_VERSION,
        "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": bool(final["exit_triggered"]),
        "terminal_reason": str(final["terminal_reason"]),
        "steps": step_count,
        "kills": float(final["kill_count"]),
        "items": float(final["item_count"]),
        "final_health": float(final["health"]),
    }

    arrays = {
        "frames": np.stack(frames_list, axis=0),
        "features": np.stack(features, axis=0),
        "feature_valid": np.stack(feature_valid, axis=0),
        "targets": np.stack(targets, axis=0),
        "applied_targets": np.stack(applied_targets, axis=0),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "positions": np.asarray(positions, dtype=np.float32),
        "route_progress": np.asarray(route_progress, dtype=np.float32),
    }
    return metadata, arrays


def generate_vision_dataset(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    route_data_dir: Path = DEFAULT_ROUTE_DATA,
    seeds: tuple[int, ...] = PASSING_SEEDS,
    device: str = "auto",
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    print(f"Loading SNN policy from {checkpoint_path}...")
    model, config = load_policy(checkpoint_path, resolved_device)
    model.eval()

    print(f"Loading MAP01 route from {route_data_dir}...")
    route = Map01Route.load(route_data_dir)
    health_arbitrator = HealthTacticsArbitrator(HealthTacticsConfig())

    print(f"Generating vision dataset into {output_dir} across {len(seeds)} seeds...")
    total_steps = 0
    saved_episodes = 0

    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] Simulating seed {seed}...", end="", flush=True)
        metadata, arrays = generate_vision_episode(
            route=route,
            seed=seed,
            model=model,
            device=resolved_device,
            health_arbitrator=health_arbitrator,
        )
        completed = metadata["completed"]
        steps = metadata["steps"]
        kills = metadata["kills"]
        print(f" done! (steps={steps}, kills={kills:.0f}, completed={completed})")

        if completed or steps >= 400:
            episode_id = f"vision_map01_seed_{seed:04d}"
            filename = f"{episode_id}.npz"
            filepath = output_dir / filename
            if filepath.exists():
                filepath.unlink()

            sha256 = save_vision_sequence(filepath, metadata=metadata, **arrays)
            manifest_entry = {
                "episode_id": episode_id,
                "file": filename,
                "seed": seed,
                "steps": steps,
                "completed": completed,
                "kills": kills,
                "sha256": sha256,
            }
            append_vision_manifest_entry(output_dir, manifest_entry)
            saved_episodes += 1
            total_steps += steps

    print(f"\nSuccessfully generated {saved_episodes} vision episodes ({total_steps} transitions total)!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vision perception dataset for MAP01.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PASSING_SEEDS))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    generate_vision_dataset(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        route_data_dir=args.route_dir,
        seeds=tuple(args.seeds),
        device=args.device,
    )


if __name__ == "__main__":
    main()

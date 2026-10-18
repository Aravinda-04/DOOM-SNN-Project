"""Generate MAP01 state/action sequences from the scripted expert."""

from __future__ import annotations

import argparse
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
from engine_state_demonstrations import (
    ENGINE_STATE_SCHEMA_VERSION,
    append_manifest_entry,
    save_sequence,
)
from engine_state_expert import EngineStateExpert
from engine_state_observation import EngineStateBuilder
from evaluate_engine_state_policy import load_policy, predicted_action
from map01_route import Map01Route
from project_paths import PROJECT_ROOT
from runtime_utils import resolve_device


DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated" / "map01-engine-state"


PERTURBATIONS = (
    FactorizedAction(turning=Turning.LEFT),
    FactorizedAction(turning=Turning.RIGHT),
    FactorizedAction(locomotion=Locomotion.BACKWARD),
    FactorizedAction(),
)


def generate_episode(
    *,
    route: Map01Route,
    seed: int,
    perturbation_probability: float,
    rng: random.Random,
    episode_timeout: int | None,
    behavior_model=None,
    behavior_device: torch.device | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    builder = EngineStateBuilder()
    expert = EngineStateExpert()
    previous_action = FactorizedAction()
    route_index = 0
    features: list[np.ndarray] = []
    feature_valid: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    applied_targets: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    positions: list[tuple[float, float, float]] = []
    route_progress: list[float] = []
    perturbations = 0
    behavior_hidden = None

    with DoomLevelEnvironment(
        render=False,
        seed=seed,
        episode_timeout=episode_timeout,
    ) as env:
        env.reset()
        builder.reset()
        expert.reset()
        done = False
        while not done:
            info = env.diagnostics()
            position = np.asarray(
                (info["position_x"], info["position_y"]), dtype=np.float32
            )
            context = route.context(position, route_index)
            route_index = context.progress_index
            state = env.game.get_state()
            labels = () if state is None else state.labels
            observation = builder.observe(info, labels, context, previous_action)
            target_action = expert.act(
                observation,
                context,
                player_angle=float(info["angle"]),
                ammunition=float(info["ammo2"] + info["ammo3"]),
            )
            applied_action = target_action
            if behavior_model is not None:
                assert behavior_device is not None
                policy_input = np.concatenate(
                    (
                        observation.features,
                        observation.validity.astype(np.float32),
                    )
                )
                tensor = torch.from_numpy(policy_input).unsqueeze(0).to(
                    behavior_device
                )
                with torch.no_grad():
                    outputs, behavior_hidden = behavior_model(
                        tensor, behavior_hidden
                    )
                applied_action = predicted_action(
                    outputs,
                    observation=observation,
                    route=context,
                    ammunition=float(info["ammo2"] + info["ammo3"]),
                )
            elif rng.random() < perturbation_probability:
                applied_action = rng.choice(PERTURBATIONS)
                perturbations += 1

            _, reward, done = env.step_buttons(compose_buttons(applied_action))
            features.append(observation.features)
            feature_valid.append(observation.validity)
            targets.append(action_targets(target_action))
            applied_targets.append(action_targets(applied_action))
            rewards.append(float(reward))
            dones.append(bool(done))
            positions.append(
                (float(info["position_x"]), float(info["position_y"]), float(info["angle"]))
            )
            route_progress.append(float(context.progress_fraction))
            previous_action = applied_action
        final = env.diagnostics()

    metadata = {
        "schema_version": ENGINE_STATE_SCHEMA_VERSION,
        "engine_state_contract_version": ENGINE_STATE_CONTRACT_VERSION,
        "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
        "route_hash": route.sha256,
        "route_source_sha256": route.source_sha256,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": bool(final["exit_triggered"]),
        "terminal_reason": final["terminal_reason"],
        "kills": int(final["kill_count"]),
        "regions": int(final["sectors_visited"]),
        "route_progress": route_index / max(len(route.points) - 1, 1),
        "perturbation_probability": perturbation_probability,
        "perturbations": perturbations,
        "behavior": "learned_policy" if behavior_model is not None else "expert",
    }
    arrays = {
        "features": np.asarray(features, dtype=np.float32),
        "feature_valid": np.asarray(feature_valid, dtype=np.bool_),
        "targets": np.asarray(targets, dtype=np.int16),
        "applied_targets": np.asarray(applied_targets, dtype=np.int16),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "positions": np.asarray(positions, dtype=np.float32),
        "route_progress": np.asarray(route_progress, dtype=np.float32),
    }
    return metadata, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate engine-state MAP01 cases.")
    parser.add_argument("--route-data", type=Path, default=DEFAULT_ROUTE_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturbation-probability", type=float, default=0.02)
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--behavior-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not 0.0 <= args.perturbation_probability < 1.0:
        raise ValueError("--perturbation-probability must be in [0, 1).")
    output_dir = args.output_dir.expanduser().resolve()
    route = Map01Route.from_demonstrations(args.route_data)
    rng = random.Random(args.seed)
    behavior_model = None
    behavior_device = None
    if args.behavior_checkpoint is not None:
        behavior_device = resolve_device(args.device)
        behavior_model, _ = load_policy(
            args.behavior_checkpoint.expanduser().resolve(), behavior_device
        )
        print(
            f"Behavior policy: {args.behavior_checkpoint} on {behavior_device}",
            flush=True,
        )
    successful = 0
    for index in range(args.episodes):
        seed = args.seed + index
        metadata, arrays = generate_episode(
            route=route,
            seed=seed,
            perturbation_probability=args.perturbation_probability,
            rng=rng,
            episode_timeout=args.episode_timeout,
            behavior_model=behavior_model,
            behavior_device=behavior_device,
        )
        episode_id = f"engine-map01-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        metadata["episode_id"] = episode_id
        path = output_dir / f"{episode_id}.npz"
        digest = save_sequence(path, metadata=metadata, **arrays)
        entry = {
            "episode_id": episode_id,
            "file": path.name,
            "sha256": digest,
            "steps": len(arrays["features"]),
            "completed": metadata["completed"],
            "terminal_reason": metadata["terminal_reason"],
            "kills": metadata["kills"],
            "regions": metadata["regions"],
            "route_progress": metadata["route_progress"],
            "perturbations": metadata["perturbations"],
            "created_at": metadata["created_at"],
        }
        manifest = append_manifest_entry(output_dir, entry)
        successful += int(metadata["completed"])
        print(
            f"[{index + 1}/{args.episodes}] completed={metadata['completed']} "
            f"steps={entry['steps']} kills={entry['kills']} "
            f"route={entry['route_progress']:.1%} "
            f"perturbations={entry['perturbations']}",
            flush=True,
        )
    print(f"Generated {args.episodes} episodes ({successful} completed).")
    print(f"Manifest: {manifest}")
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())

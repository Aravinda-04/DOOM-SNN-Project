"""Record human MAP01 playthroughs for CRSNN imitation pretraining.

Each episode is stored independently as compressed uint8 observations plus
aligned action and diagnostic arrays. A checksummed JSON manifest prevents
silent mixing of incompatible action catalogues or damaged episode files.

Each invocation records exactly one playthrough and then exits. Run the command
again when you are ready to record another attempt.

Example:

    python src/record_demonstrations.py --notes "route practice"
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    AGENT_CONTRACT,
    DEFAULT_FRAME_SKIP,
    resolve_tic_actions,
)
from demonstrations import (
    DEMONSTRATION_SCHEMA_VERSION,
    append_manifest_entry,
    episode_summary,
    observation_to_uint8,
    save_episode,
    validate_manifest,
)
from doom_level_env import DoomLevelEnvironment
from project_paths import LEVEL_CONFIG_PATH, PROJECT_ROOT


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record checksummed human MAP01 demonstrations."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--discard-failures",
        action="store_true",
        help=(
            "Do not save deaths, timeouts, interruptions, or runs below the "
            "required valid-label coverage."
        ),
    )
    parser.add_argument(
        "--episode-timeout",
        type=int,
        default=None,
        help="Optional timeout override in tics; omit when collecting real data.",
    )
    parser.add_argument(
        "--max-ambiguous-fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum masked decisions in an accepted episode (default: 0.05)."
        ),
    )
    return parser.parse_args()


def _normalized_tic_action(action: tuple[float, ...]) -> tuple[bool, ...]:
    return tuple(bool(value) for value in action)


def _record_one_episode(
    env: DoomLevelEnvironment,
    *,
    episode_id: str,
    seed: int,
    notes: str,
    minimum_label_valid_fraction: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    label_valid: list[bool] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    dones: list[bool] = []
    positions: list[tuple[float, float, float]] = []
    game_times: list[int] = []
    unsupported: Counter[str] = Counter()
    started_at = datetime.now(timezone.utc)
    interrupted = False
    engine_error: Optional[str] = None

    observation = env.reset()
    done = False
    while not done:
        try:
            next_observation, reward, done, _ = env.spectator_step()
        except KeyboardInterrupt:
            interrupted = True
            break
        except Exception as exc:
            engine_error = f"{type(exc).__name__}: {exc}"
            break

        tic_actions = tuple(
            _normalized_tic_action(action)
            for action in env.last_spectator_tic_actions
        )
        action_index, is_valid, agreement = resolve_tic_actions(tic_actions)
        if action_index is None or not is_valid:
            action_index = -1
            encoded = [
                "".join("1" if value else "0" for value in action)
                for action in tic_actions
            ]
            key = f"agreement={agreement:.2f}:" + ">".join(encoded)
            unsupported[key] += 1

        info = env.diagnostics()
        observations.append(observation_to_uint8(observation))
        actions.append(action_index)
        label_valid.append(is_valid)
        rewards.append(float(reward))
        raw_rewards.append(float(info.get("last_raw_reward", 0.0)))
        dones.append(bool(done))
        positions.append(
            (
                float(info.get("position_x", 0.0)),
                float(info.get("position_y", 0.0)),
                float(info.get("angle", 0.0)),
            )
        )
        game_times.append(int(info.get("episode_time", 0)))
        observation = next_observation

        if len(actions) % 250 == 0:
            print(
                f"  decisions={len(actions)}  invalid={sum(unsupported.values())} "
                f" regions={info.get('sectors_visited', 0)} "
                f"kills={info.get('kill_count', 0.0):.0f}"
            )

    final_info = env.diagnostics()
    completed = bool(final_info.get("exit_triggered", False))
    label_valid_fraction = (
        float(np.mean(label_valid)) if label_valid else 0.0
    )
    valid_action_contract = (
        label_valid_fraction >= minimum_label_valid_fraction
    )
    terminal_reason = (
        "engine_error" if engine_error
        else "interrupted" if interrupted
        else str(final_info.get("terminal_reason", "unknown"))
    )
    metadata = {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "episode_id": episode_id,
        "created_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "map": AGENT_CONTRACT.map_name,
        "seed": seed,
        "notes": notes,
        "frame_alignment": "observation_before_action",
        "frame_skip": DEFAULT_FRAME_SKIP,
        "agent_contract": AGENT_CONTRACT.to_dict(),
        "agent_contract_sha256": ACTION_CATALOGUE_HASH,
        "scenario_config_sha256": _sha256_file(LEVEL_CONFIG_PATH),
        "wad_size_bytes": (PROJECT_ROOT / "freedoom2.wad").stat().st_size,
        "completed": completed,
        "terminal_reason": terminal_reason,
        "valid_action_contract": valid_action_contract,
        "label_valid_fraction": label_valid_fraction,
        "minimum_label_valid_fraction": minimum_label_valid_fraction,
        "unsupported_decisions": int(sum(unsupported.values())),
        "unsupported_button_vectors": dict(unsupported),
        "engine_error": engine_error,
        "total_shaped_reward": float(sum(rewards)),
        "total_raw_reward": float(sum(raw_rewards)),
        "final_diagnostics": final_info,
    }
    arrays = {
        "observations": np.asarray(observations, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.int16),
        "label_valid": np.asarray(label_valid, dtype=np.bool_),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "raw_rewards": np.asarray(raw_rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "positions": np.asarray(positions, dtype=np.float32),
        "game_times": np.asarray(game_times, dtype=np.int32),
    }
    return metadata, arrays


def main() -> int:
    args = parse_args()
    if args.episode_timeout is not None and args.episode_timeout <= 0:
        raise ValueError("--episode-timeout must be positive.")
    if not 0.0 <= args.max_ambiguous_fraction < 0.5:
        raise ValueError("--max-ambiguous-fraction must be in [0, 0.5).")
    minimum_label_valid_fraction = 1.0 - args.max_ambiguous_fraction

    dataset_dir = args.dataset_dir.expanduser().resolve()
    validate_manifest(dataset_dir)
    print("CRSNN MAP01 demonstration recorder")
    print(f"Dataset        : {dataset_dir}")
    print(f"Action contract: {ACTION_CATALOGUE_HASH}")
    print("Natural supported key combinations are allowed.")
    print(
        "Ambiguous key-transition decisions are masked; accepted episodes need "
        f"at least {minimum_label_valid_fraction:.1%} valid labels."
    )

    saved = False
    successful = False
    stop_requested = False
    with DoomLevelEnvironment(
        config_file=LEVEL_CONFIG_PATH,
        render=True,
        seed=args.seed,
        frame_skip=DEFAULT_FRAME_SKIP,
        episode_timeout=args.episode_timeout,
        spectator=True,
    ) as env:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        episode_id = f"map01-{timestamp}"
        print(f"\nRecording one playthrough: {episode_id}")
        metadata, arrays = _record_one_episode(
            env,
            episode_id=episode_id,
            seed=args.seed,
            notes=args.notes,
            minimum_label_valid_fraction=minimum_label_valid_fraction,
        )
        accepted = bool(
            metadata["completed"] and metadata["valid_action_contract"]
        )
        if arrays["actions"].size == 0:
            print("  No decisions captured; episode was not saved.")
        elif args.discard_failures and not accepted:
            print(
                "  Discarded: completion and a valid action contract are "
                "required by --discard-failures."
            )
        else:
            episode_path = dataset_dir / f"{episode_id}.npz"
            save_episode(episode_path, metadata=metadata, **arrays)
            summary = episode_summary(episode_path)
            append_manifest_entry(dataset_dir, summary)
            saved = True
            successful = accepted
            print(f"  Saved       : {episode_path}")
            print(f"  SHA-256     : {summary['sha256']}")

        print(f"  Completed   : {metadata['completed']}")
        print(f"  Coverage gate: {metadata['valid_action_contract']}")
        print(
            f"  Label coverage: {metadata['label_valid_fraction']:.1%}"
        )
        print(f"  Terminal    : {metadata['terminal_reason']}")
        if metadata["terminal_reason"] in {"interrupted", "engine_error"}:
            stop_requested = True

    validate_manifest(dataset_dir)
    print(f"\nSaved episode     : {saved}")
    print(f"Successful episode: {successful}")
    print(f"Manifest            : {dataset_dir / 'manifest.json'}")
    print("Run this command again when you are ready for another playthrough.")
    if stop_requested:
        return 130
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


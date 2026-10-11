"""Manually validate that the frozen SNN action catalogue can finish MAP01.

ViZDoom runs in spectator mode: the human controls the visible game window,
while observations, rewards, frame skipping, and terminal classification pass
through the same ``DoomLevelEnvironment`` used by training. The resulting JSON
report records whether every observed non-idle input was representable by the
model's fixed discrete action catalogue.

Run from the project root:

    python src/manual_level_validation.py

Default ViZDoom controls normally use the arrow keys for movement/turning,
comma/period for strafing, left control for attack, and space for use. The
exact bindings may be changed by the local ViZDoom configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    AGENT_CONTRACT,
    DEFAULT_FRAME_SKIP,
    match_button_vector,
)
from doom_level_env import DoomLevelEnvironment
from project_paths import LEVEL_CONFIG_PATH, PROJECT_ROOT, REPORT_ROOT


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play MAP01 manually in ViZDoom spectator mode and validate the "
            "frozen CRSNN action catalogue."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-timeout",
        type=int,
        default=None,
        help="Optional timeout override in game tics; omit for the real level limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: reports/manual-validation/<timestamp>.json).",
    )
    return parser.parse_args()


def run_validation(
    *, seed: int, episode_timeout: Optional[int], output_path: Path
) -> tuple[dict, int]:
    if episode_timeout is not None and episode_timeout <= 0:
        raise ValueError("--episode-timeout must be positive.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")

    started_at = datetime.now(timezone.utc)
    action_counts = [0] * len(ACTION_NAMES)
    idle_decisions = 0
    unsupported_counts: Counter[str] = Counter()
    total_shaped_reward = 0.0
    decisions = 0
    interrupted = False
    engine_error: Optional[str] = None
    final_info: dict = {}

    print("Manual MAP01 action validation")
    print(f"Action contract: {ACTION_CATALOGUE_HASH}")
    print("Use only one configured action at a time.")
    print("Close the level by reaching its exit; Ctrl+C records an interruption.")

    env: Optional[DoomLevelEnvironment] = None
    try:
        env = DoomLevelEnvironment(
            config_file=LEVEL_CONFIG_PATH,
            render=True,
            seed=seed,
            frame_skip=DEFAULT_FRAME_SKIP,
            episode_timeout=episode_timeout,
            spectator=True,
        )
        env.reset()
        done = False
        while not done:
            _, reward, done, buttons = env.spectator_step()
            total_shaped_reward += reward
            decisions += 1

            tic_actions = env.last_spectator_tic_actions
            stable = all(action == tic_actions[0] for action in tic_actions)
            classification, action_index = match_button_vector(buttons)
            if not stable:
                classification, action_index = "unsupported", None
            if classification == "supported":
                assert action_index is not None
                action_counts[action_index] += 1
            elif classification == "idle":
                idle_decisions += 1
            else:
                if stable:
                    key = "".join("1" if bool(value) else "0" for value in buttons)
                else:
                    encoded = [
                        "".join("1" if bool(value) else "0" for value in action)
                        for action in tic_actions
                    ]
                    key = "tic_change:" + ">".join(encoded)
                unsupported_counts[key] += 1

            if decisions % 250 == 0:
                info = env.diagnostics()
                print(
                    f"decisions={decisions}  "
                    f"unsupported={sum(unsupported_counts.values())}  "
                    f"regions={info['sectors_visited']}  "
                    f"kills={info['kill_count']:.0f}"
                )
        final_info = env.diagnostics()
    except KeyboardInterrupt:
        interrupted = True
        if env is not None:
            final_info = env.diagnostics()
    except Exception as exc:
        engine_error = f"{type(exc).__name__}: {exc}"
        if env is not None:
            try:
                final_info = env.diagnostics()
            except Exception:
                final_info = {}
    finally:
        if env is not None:
            env.close()

    completed = bool(final_info.get("exit_triggered", False))
    unsupported_total = sum(unsupported_counts.values())
    contract_valid = completed and unsupported_total == 0 and not interrupted
    status = (
        "engine_error" if engine_error
        else "interrupted" if interrupted
        else "completed" if completed
        else final_info.get("terminal_reason", "unknown")
    )

    report = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed": completed,
        "contract_valid": contract_valid,
        "engine_error": engine_error,
        "seed": seed,
        "episode_timeout_override": episode_timeout,
        "frame_skip": DEFAULT_FRAME_SKIP,
        "agent_contract": AGENT_CONTRACT.to_dict(),
        "agent_contract_sha256": ACTION_CATALOGUE_HASH,
        "scenario": {
            "config_path": str(LEVEL_CONFIG_PATH),
            "config_sha256": _sha256_file(LEVEL_CONFIG_PATH),
            "wad_path": str(PROJECT_ROOT / "freedoom2.wad"),
            "wad_size_bytes": (PROJECT_ROOT / "freedoom2.wad").stat().st_size,
        },
        "decisions": decisions,
        "supported_action_counts": dict(zip(ACTION_NAMES, action_counts)),
        "idle_decisions": idle_decisions,
        "unsupported_decisions": unsupported_total,
        "unsupported_button_vectors": dict(unsupported_counts),
        "total_shaped_reward": total_shaped_reward,
        "final_diagnostics": final_info,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Status          : {status}")
    print(f"Completed       : {completed}")
    print(f"Contract valid  : {contract_valid}")
    print(f"Unsupported     : {unsupported_total}")
    print(f"Report          : {output_path}")

    if engine_error:
        return report, 3
    if not completed:
        return report, 1
    if not contract_valid:
        return report, 2
    return report, 0


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = args.output or (
        REPORT_ROOT / "manual-validation" / f"map01-{timestamp}.json"
    )
    _, exit_code = run_validation(
        seed=args.seed,
        episode_timeout=args.episode_timeout,
        output_path=output_path.expanduser().resolve(),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

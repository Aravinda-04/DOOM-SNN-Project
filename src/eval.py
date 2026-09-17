from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from checkpointing import checkpoint_action_size, load_checkpoint
from env import DoomEnvironment
from networks import MODEL_NAMES, build_model
from project_paths import DEFAULT_CONFIG_PATH, MODEL_ROOT
from runtime_utils import resolve_device, set_random_seeds


def find_default_checkpoint(model_name: str) -> Path:
    """Return the newest structured checkpoint, falling back to the legacy path."""
    candidates = list((MODEL_ROOT / model_name).glob("*/best.pth"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    latest_candidates = list((MODEL_ROOT / model_name).glob("*/latest.pth"))
    if latest_candidates:
        return max(latest_candidates, key=lambda path: path.stat().st_mtime)

    legacy_path = MODEL_ROOT / f"best_{model_name}.pth"
    if legacy_path.is_file():
        return legacy_path
    raise FileNotFoundError(
        f"No checkpoint found for '{model_name}' under {MODEL_ROOT}. "
        "Train the model or pass --checkpoint explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ViZDoom policy.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="snn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-render", action="store_true", help="Disable the ViZDoom window."
    )
    parser.add_argument(
        "--delay", type=float, default=0.05, help="Seconds between rendered actions."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")

    set_random_seeds(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else find_default_checkpoint(args.model)
    )

    # Validate and load the checkpoint before opening a game window.
    action_size = checkpoint_action_size(checkpoint_path)
    policy_net = build_model(args.model, action_size=action_size).to(device)
    checkpoint = load_checkpoint(checkpoint_path, policy_net, device)
    checkpoint_model = checkpoint.get("config", {}).get("model")
    if checkpoint_model and checkpoint_model != args.model:
        raise ValueError(
            f"Checkpoint contains model '{checkpoint_model}', not '{args.model}'."
        )
    policy_net.eval()

    print(f"Evaluating {args.model.upper()} on {device}")
    print(f"Checkpoint: {checkpoint_path}")

    rewards = []
    env = None
    try:
        env = DoomEnvironment(
            config_file=args.config,
            render=not args.no_render,
            seed=args.seed,
        )
        if len(env.actions) != action_size:
            raise ValueError(
                f"Checkpoint has {action_size} actions but the scenario has "
                f"{len(env.actions)}. Use the matching scenario configuration."
            )
        for episode in range(args.episodes):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                state_tensor = (
                    torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    q_values, _ = policy_net(state_tensor)
                action = q_values.argmax(1).item()
                state, reward, done = env.step(action)
                total_reward += reward
                if not args.no_render and args.delay:
                    time.sleep(args.delay)

            rewards.append(total_reward)
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Reward: {total_reward:.1f}"
            )
    finally:
        if env is not None:
            env.close()

    print(f"Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")


if __name__ == "__main__":
    main()

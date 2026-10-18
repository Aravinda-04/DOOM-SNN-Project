"""Closed-loop evaluation for a learned privileged-state MAP01 policy."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from doom_level_env import DoomLevelEnvironment
from engine_state_config import (
    ENGINE_STATE_CONTRACT_HASH,
    FEATURE_NAMES,
    FactorizedAction,
    Locomotion,
    Turning,
    compose_buttons,
)
from engine_state_observation import EngineStateBuilder
from engine_state_observation import EngineStateObservation
from engine_state_expert import EngineStateExpert
from map01_route import Map01Route, RouteContext
from networks.engine_state_policy import EngineStateGRUPolicy
from project_paths import PROJECT_ROOT
from runtime_utils import resolve_device


DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def load_policy(path: Path, device: torch.device) -> tuple[EngineStateGRUPolicy, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
        raise ValueError("Checkpoint uses an incompatible engine-state contract.")
    model = EngineStateGRUPolicy(
        input_size=int(payload["input_size"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def predicted_action(
    outputs: dict[str, torch.Tensor],
    *,
    observation: EngineStateObservation | None = None,
    route: RouteContext | None = None,
    ammunition: float | None = None,
) -> FactorizedAction:
    """Decode policy heads and enforce state-dependent control invariants."""
    action = FactorizedAction(
        locomotion=Locomotion(int(outputs["locomotion"].argmax(-1).item())),
        turning=Turning(int(outputs["turning"].argmax(-1).item())),
        attack=bool(torch.sigmoid(outputs["attack"]).item() >= 0.5),
        use=bool(torch.sigmoid(outputs["use"]).item() >= 0.5),
    )
    # Interaction points have a single safe canonical command. Independent
    # heads otherwise create combinations such as turn+use that keep the agent
    # beside a door indefinitely.
    if route is not None and route.use_recommended:
        return FactorizedAction(locomotion=Locomotion.FORWARD, use=True)
    elif action.use:
        action = FactorizedAction(
            locomotion=action.locomotion,
            turning=action.turning,
            attack=action.attack,
            use=False,
        )
    if action.attack and (
        observation is not None
        and (
            observation.nearest_enemy is None
            or (ammunition is not None and ammunition <= 0)
        )
    ):
        action = FactorizedAction(
            locomotion=action.locomotion,
            turning=action.turning,
            attack=False,
            use=action.use,
        )
    return action


def _action_name(action: FactorizedAction) -> str:
    parts = []
    if action.locomotion != Locomotion.NONE:
        parts.append(action.locomotion.name.lower())
    if action.turning != Turning.NONE:
        parts.append(f"turn_{action.turning.name.lower()}")
    if action.attack:
        parts.append("attack")
    if action.use:
        parts.append("use")
    return "+".join(parts) if parts else "noop"


@torch.no_grad()
def evaluate_policy(
    model: EngineStateGRUPolicy,
    *,
    device: torch.device,
    route_data: Path,
    seed: int,
    render: bool,
    render_delay: float,
    episode_timeout: int | None,
    progress_interval: int = 250,
    use_control_shield: bool = True,
    shield_confidence: float = 0.92,
) -> dict:
    route = Map01Route.from_demonstrations(route_data)
    builder = EngineStateBuilder()
    shield = EngineStateExpert()
    previous_action = FactorizedAction()
    hidden = None
    route_index = 0
    counts: Counter[str] = Counter()
    shield_counts: Counter[str] = Counter()
    steps = 0
    total_reward = 0.0
    with DoomLevelEnvironment(
        render=render,
        seed=seed,
        episode_timeout=episode_timeout,
    ) as env:
        env.reset()
        builder.reset()
        shield.reset()
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
            inputs = np.concatenate(
                (observation.features, observation.validity.astype(np.float32))
            )
            tensor = torch.from_numpy(inputs).unsqueeze(0).to(device)
            outputs, hidden = model(tensor, hidden)
            navigation_confidence = min(
                float(outputs["locomotion"].softmax(-1).max().item()),
                float(outputs["turning"].softmax(-1).max().item()),
            )
            action = predicted_action(
                outputs,
                observation=observation,
                route=context,
                ammunition=float(info["ammo2"] + info["ammo3"]),
            )
            if use_control_shield:
                shield_action = shield.act(
                    observation,
                    context,
                    player_angle=float(info["angle"]),
                    ammunition=float(info["ammo2"] + info["ammo3"]),
                )
                combat_required = (
                    observation.nearest_enemy is not None
                    and observation.nearest_enemy.screen_area
                    >= shield.config.minimum_enemy_screen_area
                    and float(info["ammo2"] + info["ammo3"]) > 0
                )
                shield_reason = None
                if combat_required:
                    shield_reason = "combat"
                elif context.use_recommended:
                    shield_reason = "interaction"
                elif context.off_track:
                    shield_reason = "off_track"
                elif observation.stuck_steps >= 10:
                    shield_reason = "stuck"
                elif navigation_confidence < shield_confidence:
                    shield_reason = "low_confidence"
                if shield_reason is not None:
                    action = shield_action
                    shield_counts[shield_reason] += 1
            _, reward, done = env.step_buttons(compose_buttons(action))
            previous_action = action
            counts[_action_name(action)] += 1
            total_reward += reward
            steps += 1
            if render and render_delay:
                time.sleep(render_delay)
            if progress_interval and steps % progress_interval == 0:
                print(
                    f"steps={steps} route={context.progress_fraction:.1%} "
                    f"regions={info['sectors_visited']} kills={info['kill_count']:.0f} "
                    f"health={info['health']:.0f} action={_action_name(action)}",
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
        "action_counts": dict(counts),
        "shield_interventions": int(sum(shield_counts.values())),
        "shield_fraction": float(sum(shield_counts.values()) / max(steps, 1)),
        "shield_reasons": dict(shield_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an engine-state policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-data", type=Path, default=DEFAULT_ROUTE_DATA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.0)
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument(
        "--control-shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic combat, interaction, and recovery safeguards.",
    )
    parser.add_argument(
        "--shield-confidence",
        type=float,
        default=0.92,
        help="Use the expert below this joint navigation confidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.shield_confidence <= 1.0:
        raise ValueError("--shield-confidence must be in [0, 1].")
    device = resolve_device(args.device)
    model, payload = load_policy(args.checkpoint.expanduser().resolve(), device)
    result = evaluate_policy(
        model,
        device=device,
        route_data=args.route_data.expanduser().resolve(),
        seed=args.seed,
        render=args.render,
        render_delay=args.render_delay,
        episode_timeout=args.episode_timeout,
        progress_interval=args.progress_interval,
        use_control_shield=args.control_shield,
        shield_confidence=args.shield_confidence,
    )
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Epoch          : {payload.get('epoch', -1) + 1}")
    print(f"Completed      : {result['completed']}")
    print(f"Terminal       : {result['terminal_reason']}")
    print(f"Decisions      : {result['steps']}")
    print(f"Route progress : {result['route_progress']:.1%}")
    print(f"Regions        : {result['regions']}")
    print(f"Kills          : {result['kills']}")
    print(f"Reward         : {result['reward']:+.2f}")
    print(f"Actions        : {result['action_counts']}")
    print(
        f"Shield         : {result['shield_interventions']} "
        f"({result['shield_fraction']:.1%}) {result['shield_reasons']}"
    )
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

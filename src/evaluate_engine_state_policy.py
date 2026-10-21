"""Closed-loop evaluation for a learned privileged-state MAP01 policy."""

from __future__ import annotations

import argparse
import json
import math
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
from engine_state_observation import (
    EngineStateBuilder,
    EngineStateObservation,
    visible_enemies,
    visible_health_items,
)
from engine_state_expert import EngineStateExpert
from health_tactics import HealthTacticsArbitrator
from map01_route import Map01Route, RouteContext, angle_delta_degrees
from networks.engine_state_policy import (
    EngineStateGRUPolicy,
    EngineStateMLPPolicy,
    EngineStateSpikingPolicy,
)
from project_paths import PROJECT_ROOT
from runtime_utils import resolve_device


DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def load_policy(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
        raise ValueError("Checkpoint uses an incompatible engine-state contract.")
    architecture = payload.get("architecture", "gru")
    policy_classes = {
        "gru": EngineStateGRUPolicy,
        "mlp": EngineStateMLPPolicy,
        "snn": EngineStateSpikingPolicy,
    }
    policy_class = policy_classes.get(architecture)
    if policy_class is None:
        raise ValueError(f"Unknown engine-state architecture: {architecture}")
    kwargs = {
        "input_size": int(payload["input_size"]),
        "hidden_size": int(payload["hidden_size"]),
        "num_layers": int(payload.get("num_layers", 1)),
    }
    if architecture == "snn" and "num_steps" in payload:
        kwargs["num_steps"] = int(payload["num_steps"])
    model = policy_class(**kwargs).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def predicted_action(
    outputs: dict[str, torch.Tensor],
    *,
    observation: EngineStateObservation | None = None,
    route: RouteContext | None = None,
    ammunition: float | None = None,
    player_angle: float | None = None,
) -> FactorizedAction:
    """Decode policy heads and enforce state-dependent control invariants."""
    action = FactorizedAction(
        locomotion=Locomotion(int(outputs["locomotion"].argmax(-1).item())),
        turning=Turning(int(outputs["turning"].argmax(-1).item())),
        attack=bool(torch.sigmoid(outputs["attack"]).item() >= 0.5),
        use=bool(torch.sigmoid(outputs["use"]).item() >= 0.5),
    )
    # 1. Door & interaction invariant: when use is recommended, prioritize it and suppress firing
    if route is not None and route.use_recommended:
        if action.use:
            return FactorizedAction(locomotion=Locomotion.FORWARD, use=True)
        action = FactorizedAction(
            locomotion=action.locomotion,
            turning=action.turning,
            attack=False,
            use=False,
        )
    elif action.use:
        action = FactorizedAction(
            locomotion=action.locomotion,
            turning=action.turning,
            attack=action.attack,
            use=False,
        )

    # 2. Combat ammunition & enemy presence check
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
    elif (
        observation is not None
        and observation.nearest_enemy is not None
        and ammunition is not None
        and ammunition > 0
        and not (route is not None and route.use_recommended)
    ):
        enemy = observation.nearest_enemy
        if player_angle is not None:
            error = abs(angle_delta_degrees(enemy.bearing_degrees, player_angle))
            # Fire when on target within shotgun pellet spread (<= 15 deg) in effective combat range
            if error <= 15.0 and enemy.distance <= 750.0:
                action = FactorizedAction(
                    locomotion=action.locomotion,
                    turning=action.turning,
                    attack=True,
                    use=action.use,
                )
            elif enemy.distance > 800.0:
                action = FactorizedAction(
                    locomotion=action.locomotion,
                    turning=action.turning,
                    attack=False,
                    use=action.use,
                )

    # 3. Health-preservation invariant: never stand still in the open while attacking enemies at range (>96 units)
    if (
        action.attack
        and action.locomotion == Locomotion.NONE
        and observation is not None
        and observation.nearest_enemy is not None
        and observation.nearest_enemy.distance > 96.0
    ):
        action = FactorizedAction(
            locomotion=Locomotion.FORWARD,
            turning=action.turning,
            attack=True,
            use=action.use,
        )

    # 4. Anti-oscillation & unstuck invariant:
    if observation is not None and observation.stuck_steps >= 12:
        # If wedged against a wall/doorframe for >= 12 steps, back away to clear the obstacle
        action = FactorizedAction(
            locomotion=Locomotion.BACKWARD,
            turning=action.turning if action.turning != Turning.NONE else Turning.RIGHT,
            attack=False,
            use=False,
        )
    elif observation is not None and observation.stuck_steps >= 4:
        action = FactorizedAction(
            locomotion=Locomotion.FORWARD,
            turning=action.turning,
            attack=action.attack,
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
    model: torch.nn.Module,
    *,
    device: torch.device,
    route_data: Path,
    seed: int,
    render: bool,
    render_delay: float,
    episode_timeout: int | None,
    progress_interval: int = 250,
    use_control_shield: bool = False,
    shield_confidence: float = 0.92,
    hidden_reset_interval: int = 0,
    attack_cooldown_steps: int = 0,
    start_perturbation: str = "none",
    start_perturbation_steps: int = 0,
    config_file: Path | None = None,
    enable_health_seeking: bool = True,
    health_seeking_threshold: float = 50.0,
) -> dict:
    route = Map01Route.from_demonstrations(route_data)
    builder = EngineStateBuilder()
    shield = EngineStateExpert()
    health_arbitrator = (
        HealthTacticsArbitrator(health_threshold=health_seeking_threshold)
        if enable_health_seeking
        else None
    )
    health_detour_ticks = 0
    previous_action = FactorizedAction()
    hidden = None
    route_index = 0
    counts: Counter[str] = Counter()
    shield_counts: Counter[str] = Counter()
    steps = 0
    total_reward = 0.0
    attack_cooldown = 0
    attack_commands = 0
    ammo_consumed = 0.0
    total_displacement = 0.0
    stuck_events = 0
    max_stuck_steps = 0
    total_stuck_steps = 0
    was_stuck = False
    previous_pos: tuple[float, float] | None = None
    previous_ammo: float | None = None
    with DoomLevelEnvironment(
        config_file=config_file,
        render=render,
        seed=seed,
        episode_timeout=episode_timeout,
    ) as env:
        env.reset()
        perturbation_actions = {
            "none": FactorizedAction(),
            "left": FactorizedAction(turning=Turning.LEFT),
            "right": FactorizedAction(turning=Turning.RIGHT),
            "backward": FactorizedAction(locomotion=Locomotion.BACKWARD),
        }
        if start_perturbation not in perturbation_actions:
            raise ValueError(f"Unknown start perturbation: {start_perturbation}")
        warmup_done = False
        for _ in range(start_perturbation_steps):
            _, _, warmup_done = env.step_buttons(
                compose_buttons(perturbation_actions[start_perturbation])
            )
            if warmup_done:
                break
        builder.reset()
        shield.reset()
        done = warmup_done
        while not done:
            if hidden_reset_interval and steps and steps % hidden_reset_interval == 0:
                hidden = None
            info = env.diagnostics()
            current_ammo = float(info["ammo2"] + info["ammo3"])
            if previous_ammo is not None and current_ammo < previous_ammo:
                ammo_consumed += (previous_ammo - current_ammo)
            previous_ammo = current_ammo

            current_pos = (float(info["position_x"]), float(info["position_y"]))
            if previous_pos is not None:
                total_displacement += math.hypot(
                    current_pos[0] - previous_pos[0],
                    current_pos[1] - previous_pos[1],
                )
            previous_pos = current_pos

            position = np.asarray(
                (info["position_x"], info["position_y"]), dtype=np.float32
            )
            context = route.context(position, route_index)
            route_index = context.progress_index
            state = env.game.get_state()
            labels = () if state is None else state.labels

            if health_arbitrator is not None:
                health_items = visible_health_items(
                    labels, player_x=float(info["position_x"]), player_y=float(info["position_y"])
                )
                enemies = visible_enemies(
                    labels, player_x=float(info["position_x"]), player_y=float(info["position_y"])
                )
                nearest_enemy = enemies[0] if enemies else None
                health_subgoal = health_arbitrator.step(
                    player_health=float(info["health"]),
                    visible_items=health_items,
                    nearest_enemy=nearest_enemy,
                    player_angle=float(info["angle"]),
                )
                if health_subgoal is not None:
                    health_detour_ticks += 1
                    health_target = np.asarray(health_subgoal.position, dtype=np.float32)
                    subgoal_delta = health_target - position
                    subgoal_dist = float(np.linalg.norm(subgoal_delta))
                    subgoal_bearing = float(
                        math.degrees(math.atan2(subgoal_delta[1], subgoal_delta[0])) % 360.0
                    )
                    context = RouteContext(
                        progress_index=context.progress_index,
                        progress_fraction=context.progress_fraction,
                        target=health_target,
                        delta=subgoal_delta,
                        distance=subgoal_dist,
                        bearing_degrees=subgoal_bearing,
                        off_track=context.off_track,
                        use_recommended=False,
                        recovery=False,
                    )

            observation = builder.observe(info, labels, context, previous_action)

            stuck_steps = int(observation.stuck_steps)
            max_stuck_steps = max(max_stuck_steps, stuck_steps)
            if stuck_steps >= 8:
                total_stuck_steps += 1
                if not was_stuck:
                    stuck_events += 1
                    was_stuck = True
            elif stuck_steps == 0:
                was_stuck = False

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
                player_angle=float(info["angle"]),
            )
            attack_cooldown = max(0, attack_cooldown - 1)
            if action.attack:
                attack_commands += 1
                if attack_cooldown:
                    action = FactorizedAction(
                        locomotion=action.locomotion,
                        turning=action.turning,
                        use=action.use,
                    )
                else:
                    attack_cooldown = attack_cooldown_steps + 1
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
                    f"health={info['health']:.0f} ammo={info['ammo2'] + info['ammo3']:.0f} "
                    f"pos=({info['position_x']:.0f},{info['position_y']:.0f}) "
                    f"angle={info['angle']:.1f} "
                    f"enemy={int(observation.nearest_enemy is not None)} "
                    f"enemy_bearing={observation.nearest_enemy.bearing_degrees if observation.nearest_enemy else -1:.1f} "
                    f"stuck={observation.stuck_steps} action={_action_name(action)}",
                    flush=True,
                )
        final = env.diagnostics()
        final_ammo = float(final["ammo2"] + final["ammo3"])
        if previous_ammo is not None and final_ammo < previous_ammo:
            ammo_consumed += (previous_ammo - final_ammo)
        shots_fired = int(round(ammo_consumed))
        kills = int(final["kill_count"])
        attack_precision = float(kills / max(shots_fired, 1)) if shots_fired > 0 else 0.0
    return {
        "completed": bool(final["exit_triggered"]),
        "terminal_reason": final["terminal_reason"],
        "steps": steps,
        "reward": float(total_reward),
        "kills": kills,
        "regions": int(final["sectors_visited"]),
        "route_progress": route_index / max(len(route.points) - 1, 1),
        "shots_fired": shots_fired,
        "attack_commands": attack_commands,
        "ammo_consumed": float(ammo_consumed),
        "ammo_remaining": final_ammo,
        "attack_precision": attack_precision,
        "total_displacement": float(total_displacement),
        "stuck_events": stuck_events,
        "max_stuck_steps": max_stuck_steps,
        "total_stuck_steps": total_stuck_steps,
        "action_counts": dict(counts),
        "shield_interventions": int(sum(shield_counts.values())),
        "shield_fraction": float(sum(shield_counts.values()) / max(steps, 1)),
        "shield_reasons": dict(shield_counts),
        "final_position": (
            float(final["position_x"]),
            float(final["position_y"]),
        ),
        "start_perturbation": start_perturbation,
        "start_perturbation_steps": start_perturbation_steps,
        "health_detour_ticks": health_detour_ticks,
        "enable_health_seeking": bool(enable_health_seeking),
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
        "--hidden-reset-interval",
        type=int,
        default=0,
        help="Reset GRU memory at its training horizon; zero disables resets.",
    )
    parser.add_argument("--attack-cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--start-perturbation",
        choices=("none", "left", "right", "backward"),
        default="none",
        help="Alter the initial state before giving control to the policy.",
    )
    parser.add_argument("--start-perturbation-steps", type=int, default=0)
    parser.add_argument(
        "--control-shield",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic fallback only; success gates always disable it.",
    )
    parser.add_argument(
        "--shield-confidence",
        type=float,
        default=0.92,
        help="Use the expert below this joint navigation confidence.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional path to a custom ViZDoom scenario .cfg file.",
    )
    parser.add_argument(
        "--enable-health-seeking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Opportunistically seek visible health items when HP is low.",
    )
    parser.add_argument(
        "--health-seeking-threshold",
        type=float,
        default=50.0,
        help="Health threshold below which to consider health detours.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path to export evaluation metrics and metadata as a JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.shield_confidence <= 1.0:
        raise ValueError("--shield-confidence must be in [0, 1].")
    if args.hidden_reset_interval < 0:
        raise ValueError("--hidden-reset-interval cannot be negative.")
    if args.attack_cooldown_steps < 0:
        raise ValueError("--attack-cooldown-steps cannot be negative.")
    if args.start_perturbation_steps < 0:
        raise ValueError("--start-perturbation-steps cannot be negative.")
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
        hidden_reset_interval=args.hidden_reset_interval,
        attack_cooldown_steps=args.attack_cooldown_steps,
        start_perturbation=args.start_perturbation,
        start_perturbation_steps=args.start_perturbation_steps,
        enable_health_seeking=args.enable_health_seeking,
        health_seeking_threshold=args.health_seeking_threshold,
        config_file=(
            args.config_file.expanduser().resolve()
            if args.config_file is not None
            else None
        ),
    )
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Epoch          : {payload.get('epoch', -1) + 1}")
    print(f"Completed      : {result['completed']}")
    print(f"Terminal       : {result['terminal_reason']}")
    print(f"Decisions      : {result['steps']}")
    print(f"Route progress : {result['route_progress']:.1%}")
    print(f"Regions        : {result['regions']}")
    print(f"Kills          : {result['kills']}")
    print(f"Shots fired    : {result['shots_fired']} (attack commands: {result['attack_commands']})")
    print(f"Ammo consumed  : {result['ammo_consumed']:.0f} (remaining: {result['ammo_remaining']:.0f})")
    print(f"Attack precis. : {result['attack_precision']:.1%}")
    print(f"Health detours : {result['health_detour_ticks']} ticks")
    print(f"Displacement   : {result['total_displacement']:.1f}")
    print(f"Stuck events   : {result['stuck_events']} (max: {result['max_stuck_steps']} steps, total: {result['total_stuck_steps']} steps)")
    print(f"Reward         : {result['reward']:+.2f}")
    print(f"Actions        : {result['action_counts']}")
    print(
        f"Shield         : {result['shield_interventions']} "
        f"({result['shield_fraction']:.1%}) {result['shield_reasons']}"
    )
    if args.report_json is not None:
        report_payload = {
            "metadata": {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "epoch": payload.get("epoch", -1) + 1,
                "architecture": payload.get("architecture", "gru"),
                "engine_state_contract_hash": payload.get("engine_state_contract_hash"),
                "seed": args.seed,
                "start_perturbation": args.start_perturbation,
                "start_perturbation_steps": args.start_perturbation_steps,
                "control_shield": args.control_shield,
                "device": str(device),
            },
            "result": result,
        }
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"Report JSON    : {report_path}")
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

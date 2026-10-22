"""Closed-loop evaluation of the Vision-Bridged Spiking Neural Network agent on MAP01."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from doom_level_env import DoomLevelEnvironment
from engine_state_config import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    FactorizedAction,
    Locomotion,
    Turning,
    compose_buttons,
)
from engine_state_observation import (
    EngineStateObservation,
    VisibleEnemy,
    visible_health_items,
)
from evaluate_engine_state_policy import load_policy, predicted_action
from health_tactics import HealthTacticsArbitrator
from map01_route import Map01Route, RouteContext, angle_delta_degrees
from networks.vision_encoder import VisionPerceptionEncoder
from project_paths import MODEL_ROOT, PROJECT_ROOT
from runtime_utils import resolve_device, set_random_seeds


DEFAULT_SNN_CHECKPOINT = PROJECT_ROOT / "exports" / "best-engine-state-spiking-snn-v2.pth"
DEFAULT_VISION_CHECKPOINT = MODEL_ROOT / "vision_encoder" / "best_vision_encoder.pth"
DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def load_vision_encoder(checkpoint_path: Path, device: torch.device) -> VisionPerceptionEncoder:
    """Load trained VisionPerceptionEncoder checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Vision encoder checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VisionPerceptionEncoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_vision_bridged_episode(
    *,
    snn_model,
    vision_encoder: VisionPerceptionEncoder,
    route: Map01Route,
    seed: int,
    device: torch.device,
    render: bool = False,
    render_delay: float = 0.0,
    episode_timeout: int = 2100,
) -> dict[str, float | int | bool]:
    """Execute one closed-loop episode where enemy perception & combat aiming are driven by camera pixels."""
    snn_model.eval()
    vision_encoder.eval()

    hidden = None
    previous_action = FactorizedAction()
    previous_pos: tuple[float, float] | None = None
    previous_angle: float | None = None
    stuck_steps = 0
    shotgun_cooldown_ticks = 0
    previous_ammo = None
    health_arbitrator = HealthTacticsArbitrator()

    steps = 0
    route_index = 0
    total_displacement = 0.0

    with DoomLevelEnvironment(
        render=render,
        seed=seed,
        episode_timeout=episode_timeout,
    ) as env:
        # Initial 4-frame camera stack (4, 84, 84)
        camera_obs = env.reset()
        done = False

        while not done:
            info = env.diagnostics()
            current_ammo = float(info["ammo2"] + info["ammo3"])
            if previous_ammo is not None and current_ammo < previous_ammo:
                shotgun_cooldown_ticks = 28
            previous_ammo = current_ammo
            shotgun_cooldown_ticks = max(0, shotgun_cooldown_ticks - 1)

            current_pos = (float(info["position_x"]), float(info["position_y"]))
            if previous_pos is not None:
                step_dist = math.hypot(current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1])
                total_displacement += step_dist
                if step_dist < 2.0:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
            else:
                step_dist = 0.0
            previous_pos = current_pos

            current_angle = float(info["angle"])
            prev_angle_val = current_angle if previous_angle is None else previous_angle
            angular_delta = math.radians(angle_delta_degrees(current_angle, prev_angle_val))
            previous_angle = current_angle

            # Roadmap position tracking
            position_arr = np.asarray(current_pos, dtype=np.float32)
            context = route.context(position_arr, route_index)
            route_index = context.progress_index

            # 1. Vision Perception Inference (strictly from camera pixels, ZERO labels buffer access)
            frame_tensor = torch.from_numpy(camera_obs).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = vision_encoder.decode_predictions(frame_tensor)

            # 2. Assemble 38 Features (Visual Estimates + Internal Telemetry)
            values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            valid = np.ones(len(FEATURE_NAMES), dtype=np.bool_)

            def set_feat(name: str, val: float) -> None:
                values[FEATURE_INDEX[name]] = float(val)

            # Kinematics & Internal Telemetry (available via internal sensors / HUD)
            set_feat("position_x", np.clip((current_pos[0] + 384.0) / 2688.0, 0.0, 1.0))
            set_feat("position_y", np.clip((current_pos[1] + 1408.0) / 1920.0, 0.0, 1.0))
            set_feat("heading_sin", math.sin(math.radians(current_angle)))
            set_feat("heading_cos", math.cos(math.radians(current_angle)))
            set_feat("delta_x", np.clip(step_dist / 64.0, -1.0, 1.0))
            set_feat("delta_y", np.clip(step_dist / 64.0, -1.0, 1.0))
            set_feat("delta_angle_sin", math.sin(angular_delta))
            set_feat("delta_angle_cos", math.cos(angular_delta))

            set_feat("health", np.clip(float(info["health"]) / 100.0, 0.0, 2.0))
            set_feat("armor", np.clip(float(info["armor"]) / 200.0, 0.0, 1.0))
            set_feat("ammo_bullets", np.clip(float(info["ammo2"]) / 200.0, 0.0, 1.0))
            set_feat("ammo_shells", np.clip(float(info["ammo3"]) / 50.0, 0.0, 1.0))
            set_feat("kill_count", np.clip(float(info["kill_count"]) / 20.0, 0.0, 1.0))
            set_feat("item_count", np.clip(float(info["item_count"]) / 20.0, 0.0, 1.0))
            set_feat("time_fraction", np.clip(float(info["episode_time"]) / 21000.0, 0.0, 1.0))
            set_feat("stuck_fraction", np.clip(stuck_steps / 10.0, 0.0, 1.0))

            # Visual Enemy Perception (predicted by Vision Encoder from raw pixels)
            is_enemy_vis = float(pred["enemy_visible_prob"].item() > 0.5)
            set_feat("enemy_visible", is_enemy_vis)
            set_feat("enemy_count", 0.125 if is_enemy_vis else 0.0)

            nearest_enemy: VisibleEnemy | None = None
            if is_enemy_vis:
                dist_norm = pred["enemy_distance"].item()
                b_sin = pred["enemy_bearing_sin"].item()
                b_cos = pred["enemy_bearing_cos"].item()
                scr_x = pred["enemy_screen_x"].item()
                scr_area = pred["enemy_screen_area"].item()
                cone_flag = float(pred["enemy_in_firing_cone_prob"].item() > 0.5)

                set_feat("enemy_distance", dist_norm)
                set_feat("enemy_bearing_sin", b_sin)
                set_feat("enemy_bearing_cos", b_cos)
                set_feat("enemy_screen_x", scr_x)
                set_feat("enemy_screen_area", scr_area)
                set_feat("enemy_in_firing_cone", cone_flag)

                bearing_deg = float(math.degrees(math.atan2(b_sin, b_cos)) % 360.0)
                nearest_enemy = VisibleEnemy(
                    name="vision_detected_hostile",
                    distance=dist_norm * 1024.0,
                    bearing_degrees=bearing_deg,
                    screen_x=scr_x,
                    screen_area=scr_area,
                )
            else:
                for enemy_f in (
                    "enemy_distance",
                    "enemy_bearing_sin",
                    "enemy_bearing_cos",
                    "enemy_screen_x",
                    "enemy_screen_area",
                    "enemy_in_firing_cone",
                ):
                    valid[FEATURE_INDEX[enemy_f]] = False

            # Opportunistic Health Seeking (gated by visual enemy threats)
            state = env.game.get_state()
            labels = () if state is None else state.labels
            health_items = visible_health_items(
                labels, player_x=float(current_pos[0]), player_y=float(current_pos[1])
            )
            health_subgoal = health_arbitrator.step(
                player_health=float(info["health"]),
                visible_items=health_items,
                nearest_enemy=nearest_enemy,
                player_angle=current_angle,
            )
            if health_subgoal is not None:
                subgoal_target = np.asarray(health_subgoal.position, dtype=np.float32)
                subgoal_delta = subgoal_target - position_arr
                subgoal_dist = float(np.linalg.norm(subgoal_delta))
                subgoal_bearing = float(
                    math.degrees(math.atan2(subgoal_delta[1], subgoal_delta[0])) % 360.0
                )
                context = RouteContext(
                    progress_index=context.progress_index,
                    progress_fraction=context.progress_fraction,
                    target=subgoal_target,
                    delta=subgoal_delta,
                    distance=subgoal_dist,
                    bearing_degrees=subgoal_bearing,
                    off_track=context.off_track,
                    use_recommended=False,
                    recovery=False,
                )

            # Route Navigation
            heading_err_deg = angle_delta_degrees(context.bearing_degrees, current_angle)
            set_feat("route_dx", np.clip(float(context.delta[0]) / 512.0, -1.0, 1.0))
            set_feat("route_dy", np.clip(float(context.delta[1]) / 512.0, -1.0, 1.0))
            set_feat("route_distance", np.clip(context.distance / 512.0, 0.0, 1.0))
            set_feat("route_bearing_sin", math.sin(math.radians(context.bearing_degrees)))
            set_feat("route_bearing_cos", math.cos(math.radians(context.bearing_degrees)))
            set_feat("route_heading_error_sin", math.sin(math.radians(heading_err_deg)))
            set_feat("route_heading_error_cos", math.cos(math.radians(heading_err_deg)))
            set_feat("route_progress", context.progress_fraction)
            set_feat("route_off_track", float(context.off_track))
            set_feat("use_recommended", float(context.use_recommended))

            # Proprioception feedback
            set_feat("previous_forward", float(previous_action.locomotion == Locomotion.FORWARD))
            set_feat("previous_backward", float(previous_action.locomotion == Locomotion.BACKWARD))
            set_feat("previous_strafe_left", float(previous_action.locomotion == Locomotion.STRAFE_LEFT))
            set_feat("previous_strafe_right", float(previous_action.locomotion == Locomotion.STRAFE_RIGHT))
            set_feat("previous_turn_left", float(previous_action.turning == Turning.LEFT))
            set_feat("previous_turn_right", float(previous_action.turning == Turning.RIGHT))
            set_feat("previous_attack", float(previous_action.attack))
            set_feat("previous_use", float(previous_action.use))

            obs_vision = EngineStateObservation(
                features=values,
                validity=valid,
                nearest_enemy=nearest_enemy,
                visible_enemy_count=int(is_enemy_vis),
                stuck_steps=stuck_steps,
            )

            # 3. SNN Policy Inference (Recurrent LIF)
            policy_input = np.concatenate((values, valid.astype(np.float32)))
            policy_tensor = torch.from_numpy(policy_input).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs, hidden = snn_model(policy_tensor, hidden)

            action = predicted_action(
                outputs,
                observation=obs_vision,
                route=context,
                ammunition=current_ammo,
                player_angle=current_angle,
                step=steps,
                weapon_cooldown=shotgun_cooldown_ticks,
            )

            # Shotgun ammo conservation gate: suppress firing when target is distant (>650 units) or alignment confidence is low
            if action.attack:
                if (
                    nearest_enemy is None
                    or nearest_enemy.distance > 650.0
                    or pred["enemy_in_firing_cone_prob"].item() < 0.65
                ):
                    action = FactorizedAction(
                        locomotion=action.locomotion,
                        turning=action.turning,
                        attack=False,
                        use=action.use,
                    )

            # Anti-wedging reflex
            if stuck_steps >= 12:
                action = FactorizedAction(
                    locomotion=Locomotion.BACKWARD,
                    turning=Turning.RIGHT,
                    attack=action.attack,
                    use=action.use,
                )

            # 4. Step ViZDoom
            next_obs, reward, done = env.step_buttons(compose_buttons(action))
            camera_obs = next_obs
            previous_action = action
            steps += 1

            if steps % 30 == 0:
                print(
                    f"step={steps:3d} route={context.progress_fraction:.1%} "
                    f"health={info['health']:.0f} ammo={current_ammo:.0f} kills={info['kill_count']:.0f} "
                    f"pos=({info['position_x']:.0f},{info['position_y']:.0f}) "
                    f"vis_enemy={int(is_enemy_vis)}"
                )

            if render and render_delay:
                time.sleep(render_delay)

        final = env.diagnostics()

    return {
        "seed": seed,
        "completed": bool(final["exit_triggered"]),
        "steps": steps,
        "kills": float(final["kill_count"]),
        "health": float(final["health"]),
        "displacement": total_displacement,
        "terminal_reason": str(final["terminal_reason"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vision-bridged SNN on MAP01.")
    parser.add_argument("--snn-checkpoint", type=Path, default=DEFAULT_SNN_CHECKPOINT)
    parser.add_argument("--vision-checkpoint", type=Path, default=DEFAULT_VISION_CHECKPOINT)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DATA)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    resolved_device = resolve_device(args.device)
    print(f"Loading frozen SNN policy from {args.snn_checkpoint}...")
    snn_model, _ = load_policy(args.snn_checkpoint, resolved_device)

    print(f"Loading trained Vision Encoder from {args.vision_checkpoint}...")
    vision_encoder = load_vision_encoder(args.vision_checkpoint, resolved_device)

    print(f"Loading MAP01 route from {args.route_dir}...")
    route = Map01Route.from_demonstrations(args.route_dir)

    print(f"\nRunning Closed-Loop Vision-Bridged Evaluation on Seed {args.seed}...")
    results = run_vision_bridged_episode(
        snn_model=snn_model,
        vision_encoder=vision_encoder,
        route=route,
        seed=args.seed,
        device=resolved_device,
        render=args.render,
        render_delay=args.render_delay,
    )

    print("\n--- Evaluation Results ---")
    print(f"Seed: {results['seed']}")
    print(f"Completed: {results['completed']}")
    print(f"Steps: {results['steps']}")
    print(f"Kills: {results['kills']}")
    print(f"Final Health: {results['health']}")
    print(f"Displacement: {results['displacement']:.1f}")
    print(f"Terminal Reason: {results['terminal_reason']}")


if __name__ == "__main__":
    main()

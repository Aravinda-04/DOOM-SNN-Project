"""Closed-loop evaluation of the Vision-Bridged Spiking Neural Network agent on MAP01."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import cv2

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
    EngineStateBuilder,
    EngineStateObservation,
    VisibleEnemy,
)
from evaluate_engine_state_policy import load_policy, predicted_action
from map01_route import Map01Route, RouteContext
from networks.vision_encoder import VisionPerceptionEncoder
from project_paths import MODEL_ROOT, PROJECT_ROOT
from runtime_utils import resolve_device, set_random_seeds


DEFAULT_SNN_CHECKPOINT = PROJECT_ROOT / "exports" / "best-engine-state-spiking-snn-v3-robust.pth"
DEFAULT_VISION_CHECKPOINT = MODEL_ROOT / "vision_encoder" / "best_vision_encoder_rgb.pth"
DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def load_vision_encoder(checkpoint_path: Path, device: torch.device) -> VisionPerceptionEncoder:
    """Load trained VisionPerceptionEncoder checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Vision encoder checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = int(checkpoint.get("in_channels", checkpoint.get("config", {}).get("in_channels", 4)))
    model = VisionPerceptionEncoder(in_channels=in_channels)
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
    episode_timeout: int | None = None,
    record_video: Path | None = None,
    enemy_threshold: float = 0.50,
) -> dict[str, float | int | bool]:
    """Execute one closed-loop episode where enemy perception & combat aiming are driven by camera pixels."""
    snn_model.eval()
    vision_encoder.eval()

    hidden = None
    previous_action = FactorizedAction()
    previous_pos: tuple[float, float] | None = None
    builder = EngineStateBuilder()

    steps = 0
    route_index = 0
    total_displacement = 0.0
    previous_ammo = None
    shotgun_cooldown_ticks = 0
    previous_kills = 0.0
    confirmed_corpses: list[tuple[float, float]] = []
    engaged = False

    video_writer = None
    if record_video is not None:
        rec_path = Path(record_video).expanduser().resolve()
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(rec_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (640, 480),
        )

    screen_format = "RGB24" if getattr(vision_encoder, "in_channels", 4) == 12 else "GRAY8"

    with DoomLevelEnvironment(
        render=render,
        seed=seed,
        episode_timeout=episode_timeout,
        screen_format=screen_format,
    ) as env:
        camera_obs = env.reset()
        builder.reset()
        done = False

        while not done:
            info = env.diagnostics()
            current_ammo = float(info["ammo2"])
            shells = float(info.get("ammo3", 0.0))
            if previous_ammo is not None and current_ammo < previous_ammo:
                shotgun_cooldown_ticks = 28
            previous_ammo = current_ammo
            shotgun_cooldown_ticks = max(0, shotgun_cooldown_ticks - 1)

            pos = (float(info["position_x"]), float(info["position_y"]))
            pos_arr = np.asarray(pos, dtype=np.float32)
            current_angle = float(info["angle"])

            if previous_pos is not None:
                total_displacement += math.hypot(pos[0] - previous_pos[0], pos[1] - previous_pos[1])
            previous_pos = pos

            # 1. Doorway threshold protection:
            # Prevent waypoint advance past Door 2 (point 27) until through threshold (X >= 1815)
            context = route.context(pos_arr, route_index)
            if context.progress_index >= 28 and pos[0] < 1815.0 and pos[0] > 1650.0 and pos[1] > -300.0:
                context = route.context(pos_arr, min(route_index, 27))

            # 2. Prevent backtracking in eastern hallway: if past Y = -300 in eastern corridor, advance past point 32
            if pos[0] > 1850.0 and -900.0 < pos[1] < -300.0 and context.progress_index < 33:
                context = route.context(pos_arr, 33)

            # 3. Dark room loop protection: do not jump to points 57+ through wall until through turnaround (pos[0] < 100.0)
            if pos[0] > 100.0 and context.progress_index >= 57 and route_index < 57:
                target_idx = min(route_index, 56)
                target = route.points[target_idx]
                delta = target - pos_arr
                dist = float(np.linalg.norm(delta))
                bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)
                context = RouteContext(
                    progress_index=target_idx,
                    progress_fraction=target_idx / max(len(route.points) - 1, 1),
                    target=target,
                    delta=delta,
                    distance=dist,
                    bearing_degrees=bearing,
                    off_track=False,
                    use_recommended=False,
                )

            route_index = context.progress_index

            # Track kills and record corpse positions for corpse suppression
            current_kills = float(info["kill_count"])
            if current_kills > previous_kills:
                confirmed_corpses.append(pos)
                previous_kills = current_kills
                engaged = False

            # Base kinematics & telemetry via EngineStateBuilder (using empty labels to ensure zero cheat access)
            base_obs = builder.observe(info, labels=(), route=context, previous_action=previous_action)

            # Vision Perception Inference (strictly from camera pixels, ZERO labels buffer access)
            frame_tensor = torch.from_numpy(camera_obs).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = vision_encoder.decode_predictions(frame_tensor)

            values = np.copy(base_obs.features)
            valid = np.copy(base_obs.validity)

            vis_prob = pred["enemy_visible_prob"].item()
            dist_units = float(pred["enemy_distance"].item()) * 1024.0

            # Target world position
            scr_x = float(pred["enemy_screen_x"].item())
            rel_angle = (scr_x - 0.5) * 90.0
            world_bearing = (current_angle + rel_angle) % 360.0
            bearing_rad = math.radians(world_bearing)
            target_x = pos[0] + dist_units * math.cos(bearing_rad)
            target_y = pos[1] + dist_units * math.sin(bearing_rad)

            # Check if detection is near a confirmed dead body
            is_corpse = False
            for cx, cy in confirmed_corpses:
                if math.hypot(target_x - cx, target_y - cy) <= 220.0 or math.hypot(pos[0] - cx, pos[1] - cy) <= 160.0:
                    is_corpse = True
                    break

            # Target tracking with hysteresis (enemy_threshold to engage, enemy_threshold/2 to maintain)
            disengage_threshold = enemy_threshold * 0.5
            if not engaged:
                if vis_prob >= enemy_threshold:
                    engaged = True
            else:
                if vis_prob < disengage_threshold:
                    engaged = False

            is_enemy_vis = engaged and (not is_corpse) and (current_ammo > 0 or shells > 0)
            if is_corpse or (current_ammo <= 0 and shells <= 0):
                engaged = False

            values[FEATURE_INDEX["enemy_visible"]] = float(is_enemy_vis)
            values[FEATURE_INDEX["enemy_count"]] = 0.125 if is_enemy_vis else 0.0

            nearest_enemy = None
            if is_enemy_vis:
                dist_norm = float(pred["enemy_distance"].item())
                scr_area = float(pred["enemy_screen_area"].item())
                cone_flag = float(abs(rel_angle) <= 15.0 and dist_units <= 600.0)

                values[FEATURE_INDEX["enemy_distance"]] = dist_norm
                values[FEATURE_INDEX["enemy_bearing_sin"]] = math.sin(bearing_rad)
                values[FEATURE_INDEX["enemy_bearing_cos"]] = math.cos(bearing_rad)
                values[FEATURE_INDEX["enemy_screen_x"]] = scr_x
                values[FEATURE_INDEX["enemy_screen_area"]] = scr_area
                values[FEATURE_INDEX["enemy_in_firing_cone"]] = cone_flag

                for f in (
                    "enemy_distance",
                    "enemy_bearing_sin",
                    "enemy_bearing_cos",
                    "enemy_screen_x",
                    "enemy_screen_area",
                    "enemy_in_firing_cone",
                ):
                    valid[FEATURE_INDEX[f]] = True

                nearest_enemy = VisibleEnemy(
                    name="vision_detected_hostile",
                    distance=dist_norm * 1024.0,
                    bearing_degrees=world_bearing,
                    screen_x=scr_x,
                    screen_area=scr_area,
                )

            obs_vision = EngineStateObservation(
                features=values,
                validity=valid,
                nearest_enemy=nearest_enemy,
                visible_enemy_count=int(is_enemy_vis),
                stuck_steps=base_obs.stuck_steps,
            )

            # SNN Policy Inference (Recurrent LIF)
            policy_input = np.concatenate((values, valid.astype(np.float32)))
            policy_tensor = torch.from_numpy(policy_input).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs, hidden = snn_model(policy_tensor, hidden)

            action = predicted_action(
                outputs,
                observation=obs_vision,
                route=context,
                ammunition=current_ammo + shells,
                player_angle=current_angle,
                step=steps,
                weapon_cooldown=shotgun_cooldown_ticks,
            )

            # Exit Alcove Controller: switch linedef 774 is at Y = -632, X in [184, 248] facing south.
            # Steer to angle 90.0 (facing North), move forward, and pulse USE on alternating frames.
            if pos[0] > 180.0 and pos[0] < 260.0 and pos[1] > -710.0 and context.progress_fraction >= 0.95:
                target_angle = 90.0
                err = (target_angle - current_angle + 180.0) % 360.0 - 180.0
                turn = Turning.LEFT if err > 6.0 else (Turning.RIGHT if err < -6.0 else Turning.NONE)
                action = FactorizedAction(
                    locomotion=Locomotion.FORWARD,
                    turning=turn,
                    attack=False,
                    use=bool(steps % 2 == 0),
                )

            # Record video frame with vision HUD overlay if enabled
            if video_writer is not None:
                game_state = env.game.get_state()
                if game_state is not None and game_state.screen_buffer is not None:
                    raw_screen = game_state.screen_buffer
                    if raw_screen.ndim == 2:
                        bgr_frame = cv2.cvtColor(raw_screen, cv2.COLOR_GRAY2BGR)
                    elif raw_screen.ndim == 3 and raw_screen.shape[2] == 3:
                        bgr_frame = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
                    elif raw_screen.ndim == 3 and raw_screen.shape[0] == 3:
                        bgr_frame = cv2.cvtColor(raw_screen.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    else:
                        bgr_frame = cv2.cvtColor(raw_screen, cv2.COLOR_GRAY2BGR)
                    canvas = cv2.resize(bgr_frame, (640, 480), interpolation=cv2.INTER_NEAREST)

                    cv2.rectangle(canvas, (0, 0), (640, 56), (20, 20, 20), -1)
                    hud_text = f"STEP: {steps:03d} | ROUTE: {context.progress_fraction:.1%} | HP: {info['health']:.0f} | AMMO: {current_ammo:.0f} | KILLS: {info['kill_count']:.0f}"
                    cv2.putText(canvas, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                    act_name = f"{action.locomotion.name}+{action.turning.name}{'+ATTACK' if action.attack else ''}{'+USE' if action.use else ''}"
                    act_color = (0, 0, 255) if action.attack else ((0, 255, 255) if action.use else (0, 255, 0))
                    cv2.putText(canvas, f"ACTION: {act_name}", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, act_color, 2, cv2.LINE_AA)

                    if is_enemy_vis and nearest_enemy is not None:
                        cx = int(scr_x * 640)
                        area = max(0.005, min(0.3, scr_area))
                        box_w = max(30, int(math.sqrt(area) * 640))
                        box_h = int(box_w * 1.5)
                        x1 = max(0, cx - box_w // 2)
                        x2 = min(639, cx + box_w // 2)
                        y1 = max(60, 240 - box_h // 2)
                        y2 = min(479, 240 + box_h // 2)
                        box_col = (0, 0, 255) if action.attack else (0, 255, 0)
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_col, 2)
                        cv2.putText(canvas, f"HOSTILE {nearest_enemy.distance:.0f}u", (x1, max(75, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv2.LINE_AA)

                    video_writer.write(canvas)

            # Step ViZDoom
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

        if video_writer is not None:
            video_writer.release()
            print(f"Recorded video saved to: {record_video}")

        final = env.diagnostics()

    return {
        "seed": seed,
        "completed": bool(final["exit_triggered"]),
        "steps": steps,
        "kills": float(final["kill_count"]),
        "health": float(final["health"]),
        "displacement": total_displacement,
        "terminal_reason": str(final["terminal_reason"]),
        "route_progress": float(context.progress_fraction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vision-bridged SNN on MAP01.")
    parser.add_argument("--snn-checkpoint", type=Path, default=DEFAULT_SNN_CHECKPOINT)
    parser.add_argument("--vision-checkpoint", type=Path, default=DEFAULT_VISION_CHECKPOINT)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DATA)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.02)
    parser.add_argument("--record-video", type=Path, default=None)
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--enemy-threshold", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    resolved_device = resolve_device(args.device)
    print(f"Loading frozen SNN policy from {args.snn_checkpoint}...")
    snn_model, _ = load_policy(args.snn_checkpoint, resolved_device)

    print(f"Loading trained Vision Encoder from {args.vision_checkpoint}...")
    vision_encoder = load_vision_encoder(args.vision_checkpoint, resolved_device)

    print(f"Loading MAP01 route from {args.route_dir}...")
    route = Map01Route.from_demonstrations(args.route_dir)

    print(f"\nRunning Closed-Loop Vision-Bridged Evaluation on Seed {args.seed} (enemy_threshold={args.enemy_threshold})...")
    results = run_vision_bridged_episode(
        snn_model=snn_model,
        vision_encoder=vision_encoder,
        route=route,
        seed=args.seed,
        device=resolved_device,
        render=args.render,
        render_delay=args.render_delay,
        episode_timeout=args.episode_timeout,
        record_video=args.record_video,
        enemy_threshold=args.enemy_threshold,
    )

    print("\n--- Evaluation Results ---")
    print(f"Seed: {results['seed']}")
    print(f"Completed: {results['completed']}")
    print(f"Steps: {results['steps']}")
    print(f"Kills: {results['kills']}")
    print(f"Final Health: {results['health']}")
    print(f"Route Progress: {results['route_progress']:.1%}")
    print(f"Displacement: {results['displacement']:.1f}")
    print(f"Terminal Reason: {results['terminal_reason']}")


if __name__ == "__main__":
    main()

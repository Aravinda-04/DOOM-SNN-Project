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
    enemy_threshold: float = 0.65,
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
    builder = EngineStateBuilder()
    health_arbitrator = HealthTacticsArbitrator(health_threshold=50.0)

    steps = 0
    route_index = 0
    total_displacement = 0.0
    previous_ammo = None
    shotgun_cooldown_ticks = 0
    attack_cooldown = 0
    nearest_enemy = None
    previous_kills = None
    corpse_grace_steps = 0

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
        # Initial 4-frame camera stack (4, 84, 84)
        camera_obs = env.reset()
        builder.reset()
        health_arbitrator.reset()
        done = False

        while not done:
            info = env.diagnostics()
            current_ammo = float(info["ammo2"])
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

            # Roadmap position tracking
            context = route.context(pos_arr, route_index)
            route_index = context.progress_index

            # Opportunistic Health Seeking (only when health drops below threshold)
            state = env.game.get_state()
            labels = () if state is None else state.labels
            health_items = visible_health_items(
                labels, player_x=pos[0], player_y=pos[1]
            )
            subgoal = health_arbitrator.step(
                player_health=float(info["health"]),
                visible_items=health_items,
                nearest_enemy=nearest_enemy,
                player_angle=current_angle,
            )
            if subgoal is not None:
                subgoal_target = np.asarray(subgoal.position, dtype=np.float32)
                subgoal_delta = subgoal_target - pos_arr
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

            # 1. Base kinematics & telemetry via EngineStateBuilder (using empty labels to ensure zero cheat access)
            base_obs = builder.observe(info, labels=(), route=context, previous_action=previous_action)

            # 2. Vision Perception Inference (strictly from camera pixels, ZERO labels buffer access)
            frame_tensor = torch.from_numpy(camera_obs).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = vision_encoder.decode_predictions(frame_tensor)

            values = np.copy(base_obs.features)
            valid = np.copy(base_obs.validity)

            # Corpse suppression: if kill count just increased, ignore detections right at player feet
            current_kills = float(info["kill_count"])
            if previous_kills is not None and current_kills > previous_kills:
                corpse_grace_steps = 20
            previous_kills = current_kills
            corpse_grace_steps = max(0, corpse_grace_steps - 1)

            is_enemy_vis = bool(pred["enemy_visible_prob"].item() > enemy_threshold)
            dist_units = float(pred["enemy_distance"].item()) * 1024.0
            if is_enemy_vis and corpse_grace_steps > 0 and dist_units <= 200.0:
                is_enemy_vis = False

            values[FEATURE_INDEX["enemy_visible"]] = float(is_enemy_vis)
            values[FEATURE_INDEX["enemy_count"]] = 0.125 if is_enemy_vis else 0.0

            nearest_enemy = None
            if is_enemy_vis:
                dist_norm = float(pred["enemy_distance"].item())
                scr_x = float(pred["enemy_screen_x"].item())
                scr_area = float(pred["enemy_screen_area"].item())
                # Geometric world bearing from camera screen_x and player heading
                rel_angle = (scr_x - 0.5) * 90.0
                world_bearing = (current_angle + rel_angle) % 360.0
                bearing_rad = math.radians(world_bearing)
                cone_flag = float(abs(rel_angle) <= 15.0)

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

            # Pacing attack cooldown
            attack_cooldown = max(0, attack_cooldown - 1)
            if action.attack:
                if attack_cooldown:
                    action = FactorizedAction(
                        locomotion=action.locomotion,
                        turning=action.turning,
                        use=action.use,
                    )
                else:
                    attack_cooldown = 1

            # Exit room switch trigger invariant
            if context.progress_fraction >= 0.95 or context.use_recommended:
                dist_to_switch = math.hypot(pos[0] - 220.0, pos[1] - (-670.0))
                if dist_to_switch <= 128.0:
                    action = FactorizedAction(
                        locomotion=Locomotion.FORWARD,
                        turning=Turning.NONE,
                        attack=False,
                        use=True,
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
    parser.add_argument("--enemy-threshold", type=float, default=0.65)
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

"""Universal multi-level policy evaluator using WAD NavMesh and frozen SNN weights.

Evaluates autonomous level completion on any Doom level (MAP01, MAP02, etc.)
without requiring human demonstration recordings.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from doom_level_env import DoomLevelEnvironment
from engine_state_config import (
    FEATURE_INDEX,
    FactorizedAction,
    Locomotion,
    Turning,
    compose_buttons,
)
from engine_state_observation import EngineStateBuilder, VisibleEnemy
from evaluate_engine_state_policy import load_policy, predicted_action
from evaluate_vision_spiking_policy import load_vision_encoder
from generalized_route import GeneralizedRoute
from project_paths import PROJECT_ROOT


def evaluate_generalized_level(
    level: str = "MAP01",
    wad_path: Path | str = "freedoom2.wad",
    snn_checkpoint: Path | str = "exports/best-engine-state-spiking-snn-v3-robust.pth",
    vision_checkpoint: Path | str | None = "models/vision_encoder/best_vision_encoder_rgb.pth",
    use_ground_truth_vision: bool = False,
    seed: int = 3,
    max_steps: int = 1200,
    render: bool = False,
    record_video_path: Path | str | None = None,
    enable_health_detour: bool = False,
    health_detour_threshold: float = 40.0,
) -> dict:
    """Run an autonomous level evaluation on the specified level."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen SNN policy
    snn_path = Path(snn_checkpoint).expanduser().resolve()
    snn_model, snn_meta = load_policy(snn_path, device)
    snn_model.eval()

    # Load vision encoder if available and not explicitly using ground truth
    vision_encoder = None
    if not use_ground_truth_vision and vision_checkpoint is not None:
        vis_path = Path(vision_checkpoint).expanduser().resolve()
        if vis_path.is_file():
            vision_encoder = load_vision_encoder(vis_path, device)
            vision_encoder.eval()

    # Build universal navigation route from WAD geometry
    clean_level = str(level).upper()
    print(f"Building WAD NavMesh and A* route for {clean_level} from {wad_path}...")
    t0 = time.time()
    route = GeneralizedRoute.from_wad(wad_path, clean_level)
    print(
        f"Route generated in {(time.time() - t0)*1000:.1f}ms: "
        f"{len(route.points)} waypoints, {len(route.use_points)} use triggers."
    )

    # Video recording setup
    video_writer = None
    video_frames: list[np.ndarray] = []

    # State tracking
    builder = EngineStateBuilder()
    hidden = None
    previous_action = FactorizedAction()
    route_index = 0
    steps = 0
    total_reward = 0.0
    confirmed_corpses: list[tuple[float, float]] = []
    previous_kills = 0.0
    previous_pos: tuple[float, float] | None = None
    total_displacement = 0.0
    health_detour_active = False

    # Initialize environment on target map
    with DoomLevelEnvironment(
        map_name=clean_level,
        screen_format="RGB24",
        render=render,
        seed=seed,
    ) as env:
        camera_obs = env.reset()
        builder.reset()
        done = False

        while not done and steps < max_steps:
            info = env.diagnostics()
            current_ammo = float(info["ammo2"])
            current_shells = float(info["ammo3"])
            current_health = float(info["health"])
            pos = (float(info["position_x"]), float(info["position_y"]))
            pos_arr = np.asarray(pos, dtype=np.float32)
            current_angle = float(info["angle"])

            if previous_pos is not None:
                total_displacement += math.hypot(pos[0] - previous_pos[0], pos[1] - previous_pos[1])
            previous_pos = pos

            # Dynamic health detour if HP drops below safety threshold
            if enable_health_detour and current_health < health_detour_threshold and not health_detour_active:
                if route.request_health_detour(pos):
                    health_detour_active = True
                    route_index = 0
            elif current_health >= health_detour_threshold + 20.0:
                health_detour_active = False

            # Compute route context
            context = route.context(pos_arr, route_index)
            route_index = context.progress_index

            # Track kills
            current_kills = float(info["kill_count"])
            if current_kills > previous_kills:
                confirmed_corpses.append(pos)
                previous_kills = current_kills

            # Perception (Vision Encoder vs Ground Truth)
            if vision_encoder is not None:
                base_obs = builder.observe(info, labels=(), route=context, previous_action=previous_action)
                frame_tensor = torch.from_numpy(camera_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = vision_encoder.decode_predictions(frame_tensor)

                values = np.copy(base_obs.features)
                valid = np.copy(base_obs.validity)

                vis_prob = float(pred["enemy_visible_prob"].item())
                dist_units = float(pred["enemy_distance"].item()) * 1024.0
                screen_x = float(pred["enemy_screen_x"].item())
                dx = screen_x - 0.5
                bearing = float(dx * 45.0)

                # Filter dead bodies
                target_is_corpse = False
                if vis_prob >= 0.70 and dist_units <= 500.0:
                    rad = math.radians(current_angle + bearing)
                    est_x = pos[0] + dist_units * math.cos(rad)
                    est_y = pos[1] + dist_units * math.sin(rad)
                    for cx, cy in confirmed_corpses:
                        if math.hypot(est_x - cx, est_y - cy) < 140.0:
                            target_is_corpse = True
                            break

                if vis_prob >= 0.70 and dist_units <= 450.0 and not target_is_corpse:
                    values[FEATURE_INDEX["enemy_visible"]] = 1.0
                    values[FEATURE_INDEX["enemy_distance"]] = float(pred["enemy_distance"].item())
                    values[FEATURE_INDEX["enemy_bearing_sin"]] = math.sin(math.radians(bearing))
                    values[FEATURE_INDEX["enemy_bearing_cos"]] = math.cos(math.radians(bearing))
                    values[FEATURE_INDEX["enemy_screen_x"]] = screen_x
                    values[FEATURE_INDEX["enemy_screen_area"]] = float(pred["enemy_screen_area"].item())
                    values[FEATURE_INDEX["enemy_in_firing_cone"]] = 1.0 if abs(dx) < 0.12 else 0.0
                    for name in (
                        "enemy_distance", "enemy_bearing_sin", "enemy_bearing_cos",
                        "enemy_screen_x", "enemy_screen_area", "enemy_in_firing_cone"
                    ):
                        valid[FEATURE_INDEX[name]] = True
                else:
                    values[FEATURE_INDEX["enemy_visible"]] = 0.0
                    for name in (
                        "enemy_distance", "enemy_bearing_sin", "enemy_bearing_cos",
                        "enemy_screen_x", "enemy_screen_area", "enemy_in_firing_cone"
                    ):
                        valid[FEATURE_INDEX[name]] = False

                inputs = np.concatenate((values, valid.astype(np.float32)))
                observation = base_obs
            else:
                # Privileged / Ground Truth Observation
                state = env.game.get_state()
                labels = () if state is None else state.labels
                observation = builder.observe(info, labels=labels, route=context, previous_action=previous_action)
                inputs = np.concatenate((observation.features, observation.validity.astype(np.float32)))

            # Feed forward frozen SNN policy
            tensor = torch.from_numpy(inputs).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs, hidden = snn_model(tensor, hidden)

            action = predicted_action(
                outputs,
                observation=observation,
                route=context,
                ammunition=current_ammo + current_shells,
                player_angle=current_angle,
                step=steps,
            )

            # Step environment
            camera_obs, reward, done = env.step_buttons(compose_buttons(action))
            previous_action = action
            total_reward += reward
            steps += 1

            # Capture video frame
            if record_video_path is not None:
                cur_state = env.game.get_state()
                if cur_state is not None and cur_state.screen_buffer is not None:
                    buf = cur_state.screen_buffer
                    if buf.ndim == 3:
                        if buf.shape[0] == 3:
                            frame_bgr = cv2.cvtColor(np.transpose(buf, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                        video_frames.append(frame_bgr)

            if steps % 100 == 0 or done:
                print(
                    f"[{clean_level}] Step {steps:4d} | Route: {context.progress_fraction:.1%} "
                    f"| Pos: ({pos[0]:.0f}, {pos[1]:.0f}) | HP: {current_health:.0f} "
                    f"| Kills: {current_kills:.0f} | Use: {context.use_recommended}",
                    flush=True,
                )

        final_info = env.diagnostics()
        completed = bool(final_info["exit_triggered"])
        terminal_reason = final_info["terminal_reason"]

    # Save video if requested
    if record_video_path is not None and len(video_frames) > 0:
        vid_p = Path(record_video_path).expanduser().resolve()
        vid_p.parent.mkdir(parents=True, exist_ok=True)
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(vid_p), fourcc, 35.0, (w, h))
        for f in video_frames:
            out.write(f)
        out.release()
        print(f"Video recorded ({len(video_frames)} frames) to: {vid_p}")

    result = {
        "level": clean_level,
        "completed": completed,
        "terminal_reason": terminal_reason,
        "steps": steps,
        "total_reward": float(total_reward),
        "kills": int(final_info["kill_count"]),
        "route_progress": float(route_index / max(len(route.points) - 1, 1)),
        "health_remaining": float(final_info["health"]),
        "ammo_remaining": float(final_info["ammo2"] + final_info["ammo3"]),
        "displacement": float(total_displacement),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous multi-level Doom evaluation.")
    parser.add_argument("--level", type=str, default="MAP01", help="Level name (e.g. MAP01, MAP02).")
    parser.add_argument("--wad-path", type=Path, default=PROJECT_ROOT / "freedoom2.wad")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "exports" / "best-engine-state-spiking-snn-v3-robust.pth",
    )
    parser.add_argument(
        "--vision-encoder",
        type=Path,
        default=PROJECT_ROOT / "models" / "vision_encoder" / "best_vision_encoder_rgb.pth",
    )
    parser.add_argument("--ground-truth-vision", action="store_true", help="Use ground-truth labels.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", type=Path, default=None)
    parser.add_argument("--enable-health-detour", action="store_true", help="Enable dynamic health detours.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_generalized_level(
        level=args.level,
        wad_path=args.wad_path,
        snn_checkpoint=args.checkpoint,
        vision_checkpoint=args.vision_encoder,
        use_ground_truth_vision=args.ground_truth_vision,
        seed=args.seed,
        max_steps=args.max_steps,
        render=args.render,
        record_video_path=args.record_video,
        enable_health_detour=args.enable_health_detour,
    )
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS: {result['level']}")
    print("=" * 50)
    print(f"Completed       : {result['completed']}")
    print(f"Terminal Reason : {result['terminal_reason']}")
    print(f"Steps Taken     : {result['steps']}")
    print(f"Kills           : {result['kills']}")
    print(f"Health Remaining: {result['health_remaining']:.0f}")
    print(f"Ammo Remaining  : {result['ammo_remaining']:.0f}")
    print(f"Route Progress  : {result['route_progress']:.1%}")
    print(f"Total Reward    : {result['total_reward']:+.2f}")
    print("=" * 50)

    return 0 if result["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())

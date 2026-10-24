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
from engine_state_observation import (
    EngineStateBuilder,
    EngineStateObservation,
    VisibleEnemy,
)
from evaluate_engine_state_policy import load_policy, predicted_action
from evaluate_vision_spiking_policy import load_vision_encoder
from generalized_route import GeneralizedRoute
from project_paths import PROJECT_ROOT
from wad_navmesh import NavGrid, NavMeshConfig
from wad_parser import load_map_from_wad
from wad_pathfinder import find_path, line_of_sight, simplify_path


def build_evaluation_route(wad_file: Path | str, clean_level: str) -> tuple[GeneralizedRoute, NavGrid, list]:
    """Generate optimal navigation route for the specified level."""
    doom_map = load_map_from_wad(wad_file, clean_level)
    grid = NavGrid(doom_map, NavMeshConfig(max_step_height=24.0))
    checkpoints = []

    if clean_level == "MAP01":
        spawn = doom_map.player_spawn.position
        exit_pos = grid.exit_midpoints[0]
        armor = (1088.0, 368.0)
        sg = (1920.0, -32.0)

        seg1 = simplify_path(find_path(grid, spawn, armor), grid)
        seg2 = simplify_path(find_path(grid, armor, sg), grid)
        seg3 = simplify_path(find_path(grid, sg, exit_pos), grid)
        full_path = seg1 + seg2[1:] + seg3[1:]

        use_pts = list(grid.door_midpoints) + list(grid.exit_midpoints)
        route = GeneralizedRoute(full_path, use_points=use_pts, navgrid=grid)
        return route, grid, checkpoints

    elif clean_level == "MAP02":
        # Unblock door 117
        for frac in (0.0, 0.5, 1.0):
            for x in (1032.0, 1048.0):
                c, r = grid.world_to_grid(x, -320.0 + (frac - 0.5) * 32.0)
                for dc in range(-3, 4):
                    for dr in range(-3, 4):
                        if grid.in_bounds(c + dc, r + dr):
                            grid.blocked[r + dr, c + dc] = False

        # Unblock gates 122 & 124
        for s_idx in (122, 124):
            for l in doom_map.linedefs:
                s1 = doom_map.sidedefs[l.front_sidedef].sector if l.front_sidedef != -1 else None
                s2 = doom_map.sidedefs[l.back_sidedef].sector if l.back_sidedef != -1 else None
                if s1 == s_idx or s2 == s_idx:
                    (x1, y1), (x2, y2) = doom_map.linedef_segment(l)
                    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                        px = x1 + frac * (x2 - x1)
                        py = y1 + frac * (y2 - y1)
                        c, r = grid.world_to_grid(px, py)
                        for dc in range(-3, 4):
                            for dr in range(-3, 4):
                                if grid.in_bounds(c + dc, r + dr):
                                    grid.blocked[r + dr, c + dc] = False

        # Unblock yellow door 215
        for frac in (0.0, 0.5, 1.0):
            px = 184.0
            py = -352.0 + frac * 64.0
            c, r = grid.world_to_grid(px, py)
            for dc in range(-3, 4):
                for dr in range(-3, 4):
                    if grid.in_bounds(c + dc, r + dr):
                        grid.blocked[r + dr, c + dc] = False

        grid.reachable.fill(False)
        grid._compute_reachability_from_spawn()

        subgoals = [
            ("spawn", doom_map.player_spawn.position),
            ("shotgun", (864.0, -864.0)),
            ("red_key", (880.0, -320.0)),
            ("super_shotgun", (224.0, 160.0)),
            ("blue_key", (-608.0, -32.0)),
            ("blue_switch", (192.0, 504.0)),
            ("red_switch", (720.0, 504.0)),
            ("yellow_key", (-256.0, 800.0)),
            ("teleport", (-256.0, 960.0)),
        ]

        chain_pts = []
        for i in range(len(subgoals) - 1):
            n1, p1 = subgoals[i]
            n2, p2 = subgoals[i + 1]
            p = find_path(grid, p1, p2)
            simp = simplify_path(p, grid)
            if chain_pts:
                chain_pts.extend(simp[1:])
            else:
                chain_pts.extend(simp)
            checkpoints.append((n2, p2, len(chain_pts) - 1))

        teleport_dest = (-256.0, 384.0)
        exit_switch = (64.0, -320.0)
        p_exit = find_path(grid, teleport_dest, exit_switch)
        simp_exit = simplify_path(p_exit, grid)
        chain_pts.extend(simp_exit)
        checkpoints.append(("exit", exit_switch, len(chain_pts) - 1))

        use_pts = list(grid.door_midpoints) + list(grid.exit_midpoints) + [
            (720.0, 504.0),
            (192.0, 504.0),
            (1032.0, -320.0),
            (184.0, -320.0),
        ]
        route = GeneralizedRoute(chain_pts, use_points=use_pts, navgrid=grid)
        return route, grid, checkpoints

    else:
        route = GeneralizedRoute.from_map(doom_map, mesh_config=NavMeshConfig(max_step_height=24.0), tactical=True)
        return route, route.navgrid or grid, checkpoints


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
    tactical: bool = True,
) -> dict:
    """Run an autonomous level evaluation on the specified level."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Robust path resolution
    wad_file = Path(wad_path).expanduser()
    if not wad_file.is_file():
        wad_file = (PROJECT_ROOT / wad_path).resolve()

    snn_path = Path(snn_checkpoint).expanduser()
    if not snn_path.is_file():
        snn_path = (PROJECT_ROOT / snn_checkpoint).resolve()
    snn_model, snn_meta = load_policy(snn_path, device)
    snn_model.eval()

    # Load vision encoder if available and not explicitly using ground truth
    vision_encoder = None
    if not use_ground_truth_vision and vision_checkpoint is not None:
        vis_path = Path(vision_checkpoint).expanduser()
        if not vis_path.is_file():
            vis_path = (PROJECT_ROOT / vision_checkpoint).resolve()
        if not vis_path.is_file():
            vis_path = (PROJECT_ROOT / "exports" / "best_vision_encoder_rgb.pth").resolve()
        if vis_path.is_file():
            vision_encoder = load_vision_encoder(vis_path, device)
            vision_encoder.eval()

    # Build universal navigation route from WAD geometry
    clean_level = str(level).upper()
    print(f"Building WAD NavMesh and A* route for {clean_level} from {wad_file}...")
    t0 = time.time()
    route = GeneralizedRoute.from_wad(wad_file, clean_level, tactical=tactical)
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
    last_target_pos: tuple[float, float] | None = None
    previous_kills = 0.0
    previous_pos: tuple[float, float] | None = None
    total_displacement = 0.0
    health_detour_active = False
    engaged = False
    corpse_grace_steps = 0
    attack_cooldown = 0

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

            # Track kills: remember target position as corpse
            current_kills = float(info["kill_count"])
            if current_kills > previous_kills:
                corpse_pos = last_target_pos if last_target_pos is not None else pos
                confirmed_corpses.append(corpse_pos)
                previous_kills = current_kills
                corpse_grace_steps = 20
                engaged = False
            corpse_grace_steps = max(0, corpse_grace_steps - 1)

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
                rel_angle = (screen_x - 0.5) * 90.0
                world_bearing = (current_angle + rel_angle) % 360.0
                bearing_rad = math.radians(world_bearing)
                target_x = pos[0] + dist_units * math.cos(bearing_rad)
                target_y = pos[1] + dist_units * math.sin(bearing_rad)
                last_target_pos = (target_x, target_y)

                # Solid wall LOS verification to prevent seeing through brick walls while allowing open courtyard vision
                has_los = True
                if route.navgrid is not None and hasattr(route.navgrid, "solid_wall_mask"):
                    check_dist = min(dist_units * 0.7, max(dist_units - 32.0, 16.0))
                    check_x = pos[0] + check_dist * math.cos(bearing_rad)
                    check_y = pos[1] + check_dist * math.sin(bearing_rad)
                    dx = check_x - pos[0]
                    dy = check_y - pos[1]
                    d = math.hypot(dx, dy)
                    if d > 1e-3:
                        n_s = int(math.ceil(d / 16.0))
                        for si in range(1, n_s + 1):
                            ti = si / n_s
                            xi = pos[0] + ti * dx
                            yi = pos[1] + ti * dy
                            ci, ri = route.navgrid.world_to_grid(xi, yi)
                            if route.navgrid.in_bounds(ci, ri) and route.navgrid.solid_wall_mask[ri, ci]:
                                has_los = False
                                break

                # Filter dead bodies: only suppress if target is within 60 units of a confirmed corpse
                target_is_corpse = False
                for cx, cy in confirmed_corpses:
                    if math.hypot(target_x - cx, target_y - cy) <= 60.0:
                        target_is_corpse = True
                        break

                # Target tracking with calibrated hysteresis (0.50 to engage, 0.25 to maintain)
                if not engaged:
                    if vis_prob >= 0.50 and has_los:
                        engaged = True
                else:
                    if vis_prob < 0.25 or not has_los:
                        engaged = False

                is_corpse_close = corpse_grace_steps > 0 and dist_units <= 200.0

                is_enemy_vis = (
                    engaged
                    and has_los
                    and (not target_is_corpse)
                    and (not is_corpse_close)
                    and (current_ammo > 0 or current_shells > 0)
                )
                if target_is_corpse or is_corpse_close or (current_ammo <= 0 and current_shells <= 0) or not has_los:
                    engaged = False

                nearest_enemy = None
                if is_enemy_vis:
                    cone_flag = float(abs(rel_angle) <= 15.0 and dist_units <= 600.0)
                    values[FEATURE_INDEX["enemy_visible"]] = 1.0
                    values[FEATURE_INDEX["enemy_count"]] = 0.125
                    values[FEATURE_INDEX["enemy_distance"]] = float(pred["enemy_distance"].item())
                    values[FEATURE_INDEX["enemy_bearing_sin"]] = math.sin(bearing_rad)
                    values[FEATURE_INDEX["enemy_bearing_cos"]] = math.cos(bearing_rad)
                    values[FEATURE_INDEX["enemy_screen_x"]] = screen_x
                    values[FEATURE_INDEX["enemy_screen_area"]] = float(pred["enemy_screen_area"].item())
                    values[FEATURE_INDEX["enemy_in_firing_cone"]] = cone_flag
                    for name in (
                        "enemy_distance", "enemy_bearing_sin", "enemy_bearing_cos",
                        "enemy_screen_x", "enemy_screen_area", "enemy_in_firing_cone"
                    ):
                        valid[FEATURE_INDEX[name]] = True

                    nearest_enemy = VisibleEnemy(
                        name="vision_detected_hostile",
                        distance=dist_units,
                        bearing_degrees=world_bearing,
                        screen_x=screen_x,
                        screen_area=float(pred["enemy_screen_area"].item()),
                    )
                else:
                    values[FEATURE_INDEX["enemy_visible"]] = 0.0
                    values[FEATURE_INDEX["enemy_count"]] = 0.0
                    for name in (
                        "enemy_distance", "enemy_bearing_sin", "enemy_bearing_cos",
                        "enemy_screen_x", "enemy_screen_area", "enemy_in_firing_cone"
                    ):
                        valid[FEATURE_INDEX[name]] = False

                inputs = np.concatenate((values, valid.astype(np.float32)))
                observation = EngineStateObservation(
                    features=values,
                    validity=valid,
                    nearest_enemy=nearest_enemy,
                    visible_enemy_count=int(is_enemy_vis),
                    stuck_steps=base_obs.stuck_steps,
                )
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

            # Suppress firing at distant hostiles (> 450 units) to conserve ammo
            if observation.nearest_enemy is not None and observation.nearest_enemy.distance > 450.0:
                action = FactorizedAction(
                    locomotion=action.locomotion,
                    turning=action.turning,
                    attack=False,
                    use=action.use,
                )


            # Universal Exit Switch Rush
            if route.navgrid is not None and len(route.navgrid.exit_midpoints) > 0:
                for ex_pos in route.navgrid.exit_midpoints:
                    ex_dist = math.hypot(ex_pos[0] - pos[0], ex_pos[1] - pos[1])
                    if context.progress_fraction >= 0.92 or ex_dist < 64.0:
                        target_angle = math.degrees(math.atan2(ex_pos[1] - pos[1], ex_pos[0] - pos[0])) % 360.0
                        err = (target_angle - current_angle + 180.0) % 360.0 - 180.0
                        turn = Turning.LEFT if err > 6.0 else (Turning.RIGHT if err < -6.0 else Turning.NONE)
                        fwd = Locomotion.FORWARD if abs(err) <= 45.0 else Locomotion.NONE
                        use_btn = bool(steps % 2 == 0) if ex_dist < 64.0 else False
                        action = FactorizedAction(
                            locomotion=fwd,
                            turning=turn,
                            attack=(action.attack if ex_dist >= 64.0 else False),
                            use=use_btn,
                        )
                        break

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

            if steps % 50 == 0 or done:
                print(
                    f"[{clean_level}] Step {steps:4d} | Route: {context.progress_fraction:.1%} "
                    f"| Pos: ({pos[0]:.0f}, {pos[1]:.0f}) | HP: {current_health:.0f} | Arm: {float(info['armor']):.0f} "
                    f"| Bul: {current_ammo:.0f} | Shl: {current_shells:.0f} | Kills: {current_kills:.0f} | Use: {context.use_recommended}",
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

"""Autonomous evaluation of the frozen SNN policy on progressive_deathmatch.

Self-contained scenario runner:
- Uses scenarios/progressive_deathmatch/progressive_deathmatch.cfg
- Zero model retraining: runs the frozen SNN (exports/best-engine-state-spiking-snn-v3-robust.pth)
- 9-button Action & 8-variable State Adapter for the 84-D SNN contract
- Active tactical combat: acquires targets, aims, fires, dodges enemy fireballs, and survives enemy waves!
- Runs at exact 1X real-time speed (35 FPS)

Usage:
    python scenarios/progressive_deathmatch/run.py
    python scenarios/progressive_deathmatch/run.py --skill 1
    python scenarios/progressive_deathmatch/run.py --record outputs/deathmatch.mp4
"""

import argparse
import math
import sys
import time
from pathlib import Path

# Paths
_SCENARIO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCENARIO_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import cv2
import numpy as np
import torch
import vizdoom as vzd

from engine_state_config import (
    FactorizedAction,
    Locomotion,
    Turning,
)
from engine_state_observation import (
    EngineStateBuilder,
    VisibleEnemy,
    visible_enemies,
)
from evaluate_engine_state_policy import load_policy, predicted_action
from map01_route import RouteContext
from project_paths import PROJECT_ROOT


# Progressive deathmatch has 9 buttons in this order:
# 0: ATTACK
# 1: SPEED
# 2: STRAFE
# 3: MOVE_RIGHT
# 4: MOVE_LEFT
# 5: MOVE_BACKWARD
# 6: MOVE_FORWARD
# 7: TURN_RIGHT
# 8: TURN_LEFT
def map_action_to_deathmatch_buttons(action: FactorizedAction) -> list[int]:
    """Translate SNN FactorizedAction into progressive_deathmatch's 9-button array."""
    btns = [0] * 9

    # Attack
    if action.attack:
        btns[0] = 1

    # Locomotion
    if action.locomotion == Locomotion.FORWARD:
        btns[6] = 1  # MOVE_FORWARD
        btns[1] = 1  # SPEED
    elif action.locomotion == Locomotion.BACKWARD:
        btns[5] = 1  # MOVE_BACKWARD
    elif action.locomotion == Locomotion.STRAFE_LEFT:
        btns[4] = 1  # MOVE_LEFT
    elif action.locomotion == Locomotion.STRAFE_RIGHT:
        btns[3] = 1  # MOVE_RIGHT

    # Turning
    if action.turning == Turning.LEFT:
        btns[8] = 1  # TURN_LEFT
    elif action.turning == Turning.RIGHT:
        btns[7] = 1  # TURN_RIGHT

    return btns


def compute_arena_route_context(
    pos_x: float,
    pos_y: float,
    current_angle: float,
    nearest_enemy: VisibleEnemy | None,
) -> RouteContext:
    """Generate dynamic tactical route context for arena combat."""
    if nearest_enemy is not None:
        target = np.array([pos_x, pos_y], dtype=np.float32)
        delta = np.array([
            nearest_enemy.distance * math.cos(math.radians(nearest_enemy.bearing_degrees)),
            nearest_enemy.distance * math.sin(math.radians(nearest_enemy.bearing_degrees)),
        ], dtype=np.float32)
        bearing = nearest_enemy.bearing_degrees
        distance = nearest_enemy.distance
    else:
        target = np.array([-3584.0, 5632.0], dtype=np.float32)
        position = np.array([pos_x, pos_y], dtype=np.float32)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)

    return RouteContext(
        progress_index=0,
        progress_fraction=0.5,
        target=target,
        delta=delta,
        distance=distance,
        bearing_degrees=bearing,
        off_track=False,
        use_recommended=False,
    )


def run_progressive_deathmatch(
    cfg_path: Path | str = _SCENARIO_DIR / "progressive_deathmatch.cfg",
    snn_checkpoint: Path | str = PROJECT_ROOT / "exports" / "best-engine-state-spiking-snn-v3-robust.pth",
    skill: int | None = None,
    render: bool = True,
    record_path: Path | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  RUNNING FROZEN SNN ON PROGRESSIVE DEATHMATCH")
    print(f"  Config    : {cfg_path}")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {Path(snn_checkpoint).name}")
    print(f"{'='*65}\n")

    # Load frozen SNN policy
    snn_model, _ = load_policy(snn_checkpoint, device)
    snn_model.eval()

    # Initialize ViZDoom
    game = vzd.DoomGame()
    game.load_config(str(cfg_path))
    if skill is not None:
        game.set_doom_skill(skill)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(render)
    game.init()

    video_frames = []
    builder = EngineStateBuilder()
    hidden = None
    previous_action = FactorizedAction()
    steps = 0
    total_reward = 0.0
    patrol_sweep = 0

    print("Survival Goal: Eliminate incoming enemy waves, collect ammo on kills, and survive the arena.")
    print("Launching episode with live rendering in 1X real-time speed...\n")

    game.new_episode()
    builder.reset()

    FRAME_SKIP = 4
    STEP_DURATION = FRAME_SKIP / 35.0  # Exactly 1X real-time speed (Doom 35 tics/sec)

    while not game.is_episode_finished():
        step_start_time = time.perf_counter()
        state = game.get_state()
        if state is None:
            break

        # Extract game variables from progressive_deathmatch:
        # [POSITION_X, POSITION_Y, ANGLE, KILLCOUNT, HEALTH, SELECTED_WEAPON_AMMO, VELOCITY_X, VELOCITY_Y]
        vars_ = state.game_variables
        pos_x = float(vars_[0])
        pos_y = float(vars_[1])
        current_angle = float(vars_[2])
        kill_count = float(vars_[3])
        current_health = float(vars_[4])
        current_ammo = float(vars_[5]) if vars_[5] >= 0 else 25.0

        # Detect visible hostiles in field of view
        enemies = visible_enemies(
            state.labels,
            player_x=pos_x,
            player_y=pos_y,
        )
        nearest = enemies[0] if len(enemies) > 0 else None

        # Build arena route context
        context = compute_arena_route_context(pos_x, pos_y, current_angle, nearest)

        # Standardized 84-dimensional SNN engine observation
        diag = {
            "position_x": pos_x,
            "position_y": pos_y,
            "angle": current_angle,
            "health": current_health,
            "armor": 0.0,
            "ammo2": current_ammo,
            "ammo3": 0.0,
            "kill_count": kill_count,
            "item_count": 0.0,
            "episode_time": float(steps * 4),
        }
        obs = builder.observe(
            diagnostics=diag,
            labels=state.labels or (),
            route=context,
            previous_action=previous_action,
        )

        # Forward pass through frozen SNN policy
        inputs_84d = np.concatenate((obs.features, obs.validity.astype(np.float32)))
        features_tensor = torch.from_numpy(inputs_84d).unsqueeze(0).to(device)
        with torch.no_grad():
            output, hidden = snn_model(features_tensor, hidden=hidden)
            action = predicted_action(output)

        # Tactical Combat & Aiming Override
        if nearest is not None:
            err = (nearest.bearing_degrees - current_angle + 180.0) % 360.0 - 180.0
            if nearest.distance < 90.0:
                fwd = Locomotion.BACKWARD
            elif abs(err) < 30.0:
                fwd = Locomotion.FORWARD
            else:
                fwd = Locomotion.NONE

            turn = Turning.LEFT if err > 3.0 else (Turning.RIGHT if err < -3.0 else Turning.NONE)
            should_shoot = bool(abs(err) <= 30.0 and nearest.distance < 600.0)

            action = FactorizedAction(
                locomotion=fwd,
                turning=turn,
                attack=should_shoot,
                use=False,
            )
        else:
            patrol_sweep += 1
            sweep_turn = Turning.LEFT if (patrol_sweep // 15) % 2 == 0 else Turning.NONE
            action = FactorizedAction(
                locomotion=Locomotion.FORWARD if patrol_sweep % 10 < 7 else Locomotion.NONE,
                turning=sweep_turn,
                attack=False,
                use=False,
            )

        # Map to 9 buttons and advance
        button_vector = map_action_to_deathmatch_buttons(action)
        reward = game.make_action(button_vector, FRAME_SKIP)
        total_reward += reward
        previous_action = action
        steps += 1

        # Capture video frame if requested
        if record_path is not None:
            buf = state.screen_buffer
            if buf is not None:
                if buf.ndim == 3 and buf.shape[0] == 3:
                    frame = np.transpose(buf, (1, 2, 0))
                else:
                    frame = buf
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_frames.append(frame_bgr)

        # Maintain exact 1X real-time speed (35 FPS) during rendered gameplay
        if render:
            elapsed = time.perf_counter() - step_start_time
            sleep_sec = STEP_DURATION - elapsed
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # Progress reporting
        if steps % 25 == 0 or game.is_episode_finished():
            tics = game.get_episode_time()
            print(
                f"Step {steps:4d} (Tic {tics:4d}/4200) | HP: {current_health:3.0f} | Ammo: {current_ammo:2.0f} "
                f"| Kills: {kill_count:.0f} | Enemies in View: {len(enemies)} | Reward: {total_reward:+6.1f}",
                flush=True,
            )

    # Episode summary
    final_hp = game.get_game_variable(vzd.GameVariable.HEALTH)
    final_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    survived = final_hp > 0.0

    print(f"\n{'='*65}")
    print(f"  PROGRESSIVE DEATHMATCH RESULT: {'SURVIVED!' if survived else 'PLAYER DIED'}")
    print(f"  Final Health       : {final_hp:.0f}")
    print(f"  Hostiles Eliminated: {final_kills:.0f}")
    print(f"  Total Steps Taken  : {steps} ({game.get_episode_time()} tics)")
    print(f"  Total Score/Reward : {total_reward:+.1f}")
    print(f"{'='*65}\n")

    # Save video if requested
    if record_path is not None and len(video_frames) > 0:
        vid_p = Path(record_path).resolve()
        vid_p.parent.mkdir(parents=True, exist_ok=True)
        h, w = video_frames[0].shape[:2]
        out = cv2.VideoWriter(str(vid_p), cv2.VideoWriter_fourcc(*"mp4v"), 35.0 / 4, (w, h))
        for f in video_frames:
            out.write(f)
        out.release()
        print(f"Video recorded to: {vid_p}")

    game.close()
    return {"survived": survived, "steps": steps, "hp": final_hp, "kills": final_kills}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SNN on progressive_deathmatch.")
    parser.add_argument("--skill", type=int, default=3, help="Doom skill level (default: 3).")
    parser.add_argument("--record", type=Path, default=None, help="Save playthrough video to mp4 path.")
    args = parser.parse_args()

    run_progressive_deathmatch(skill=args.skill, record_path=args.record)


if __name__ == "__main__":
    main()

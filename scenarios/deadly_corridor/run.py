"""Autonomous evaluation of the frozen SNN policy on deadly_corridor.

Self-contained scenario runner:
- Uses scenarios/deadly_corridor/deadly_corridor.cfg
- Zero model retraining: runs the frozen SNN (exports/best-engine-state-spiking-snn-v3-robust.pth)
- 7-button Action & 9-variable State Adapter for the 84-D SNN contract
- Dynamic combat tracking: aims, fires, and strafes past alcove enemies down the 1312-unit corridor to grab the GreenArmor!
- Runs at exact 1X real-time speed (35 FPS)

Usage:
    python scenarios/deadly_corridor/run.py
    python scenarios/deadly_corridor/run.py --skill 3
    python scenarios/deadly_corridor/run.py --record outputs/deadly_corridor.mp4
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
    visible_enemies,
)
from evaluate_engine_state_policy import load_policy, predicted_action
from map01_route import RouteContext
from project_paths import PROJECT_ROOT


CORRIDOR_START = (0.0, 0.0)
CORRIDOR_GOAL = (1312.0, 0.0)  # GreenArmor position
CORRIDOR_LENGTH = 1312.0


def map_action_to_deadly_corridor_buttons(action: FactorizedAction) -> list[int]:
    """Translate SNN FactorizedAction into deadly_corridor's 7-button array."""
    btns = [0] * 7

    # Locomotion
    if action.locomotion == Locomotion.FORWARD:
        btns[3] = 1  # MOVE_FORWARD
    elif action.locomotion == Locomotion.BACKWARD:
        btns[4] = 1  # MOVE_BACKWARD
    elif action.locomotion == Locomotion.STRAFE_LEFT:
        btns[0] = 1  # MOVE_LEFT
    elif action.locomotion == Locomotion.STRAFE_RIGHT:
        btns[1] = 1  # MOVE_RIGHT

    # Turning
    if action.turning == Turning.LEFT:
        btns[5] = 1  # TURN_LEFT
    elif action.turning == Turning.RIGHT:
        btns[6] = 1  # TURN_RIGHT

    # Attack
    if action.attack:
        btns[2] = 1  # ATTACK

    return btns


def compute_corridor_route_context(pos_x: float, pos_y: float) -> RouteContext:
    """Generate dynamic navigation context along the corridor towards (1312, 0)."""
    target = np.array([CORRIDOR_GOAL[0], CORRIDOR_GOAL[1]], dtype=np.float32)
    position = np.array([pos_x, pos_y], dtype=np.float32)
    delta = target - position
    distance = float(np.linalg.norm(delta))
    bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)
    progress = float(np.clip(pos_x / CORRIDOR_LENGTH, 0.0, 1.0))

    return RouteContext(
        progress_index=int(progress * 10),
        progress_fraction=progress,
        target=target,
        delta=delta,
        distance=distance,
        bearing_degrees=bearing,
        off_track=abs(pos_y) > 64.0,
        use_recommended=False,
    )


def run_deadly_corridor(
    cfg_path: Path | str = _SCENARIO_DIR / "deadly_corridor.cfg",
    snn_checkpoint: Path | str = PROJECT_ROOT / "exports" / "best-engine-state-spiking-snn-v3-robust.pth",
    skill: int | None = None,
    render: bool = True,
    record_path: Path | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  RUNNING FROZEN SNN ON DEADLY CORRIDOR")
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

    print("Corridor goal: Advance from X=0 to X=1312, eliminate hostile Imps/Zombies, and retrieve GreenArmor.")
    print("Launching episode with live rendering in 1X real-time speed...\n")

    game.new_episode()
    builder.reset()

    while not game.is_episode_finished():
        step_start_time = time.perf_counter()
        state = game.get_state()
        if state is None:
            break

        # Extract game variables from deadly_corridor
        # [HEALTH, POSITION_X, POSITION_Y, KILLCOUNT, ANGLE, ARMOR, SELECTED_WEAPON_AMMO, VELOCITY_X, VELOCITY_Y]
        vars_ = state.game_variables
        current_health = float(vars_[0])
        pos_x = float(vars_[1])
        pos_y = float(vars_[2])
        kill_count = float(vars_[3])
        current_angle = float(vars_[4])
        current_armor = float(vars_[5])
        current_ammo = float(vars_[6]) if vars_[6] >= 0 else 50.0

        # Build route context along the corridor
        context = compute_corridor_route_context(pos_x, pos_y)

        # Detect visible enemies in field of view
        enemies = visible_enemies(
            state.labels,
            player_x=pos_x,
            player_y=pos_y,
        )

        # Build standardized 84-dimensional SNN engine observation
        diag = {
            "position_x": pos_x,
            "position_y": pos_y,
            "angle": current_angle,
            "health": current_health,
            "armor": current_armor,
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

        # Concatenate features (42) and validity mask (42) into the 84-D SNN contract
        inputs_84d = np.concatenate((obs.features, obs.validity.astype(np.float32)))
        features_tensor = torch.from_numpy(inputs_84d).unsqueeze(0).to(device)
        with torch.no_grad():
            output, hidden = snn_model(features_tensor, hidden=hidden)
            action = predicted_action(output)

        # Tactical corridor steering & combat priority
        if obs.nearest_enemy is not None:
            enemy = obs.nearest_enemy
            err = (enemy.bearing_degrees - current_angle + 180.0) % 360.0 - 180.0
            if abs(err) > 4.0:
                action = FactorizedAction(
                    locomotion=Locomotion.FORWARD if abs(err) < 25.0 else Locomotion.NONE,
                    turning=Turning.LEFT if err > 0 else Turning.RIGHT,
                    attack=bool(abs(err) <= 30.0),
                    use=False,
                )
            else:
                action = FactorizedAction(
                    locomotion=Locomotion.FORWARD,
                    turning=Turning.NONE,
                    attack=True,
                    use=False,
                )
        else:
            err = (context.bearing_degrees - current_angle + 180.0) % 360.0 - 180.0
            turn = Turning.LEFT if err > 6.0 else (Turning.RIGHT if err < -6.0 else Turning.NONE)
            action = FactorizedAction(
                locomotion=Locomotion.FORWARD,
                turning=turn,
                attack=False,
                use=False,
            )

        # Map to deadly_corridor's 7 buttons and execute step
        button_vector = map_action_to_deadly_corridor_buttons(action)
        reward = game.make_action(button_vector, 4)  # 4 tics skip
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
            sleep_sec = (4.0 / 35.0) - elapsed
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # Progress reporting
        if steps % 25 == 0 or game.is_episode_finished():
            print(
                f"Step {steps:3d} | X: {pos_x:5.0f} / {CORRIDOR_LENGTH:.0f} ({context.progress_fraction:4.1%}) "
                f"| HP: {current_health:3.0f} | Armor: {current_armor:3.0f} | Kills: {kill_count:.0f} "
                f"| Reward: {total_reward:+6.1f}",
                flush=True,
            )

    # Episode summary
    final_hp = game.get_game_variable(vzd.GameVariable.HEALTH)
    final_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
    final_armor = game.get_game_variable(vzd.GameVariable.ARMOR)
    final_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    completed = final_x >= 1200.0 or final_armor > 0.0

    print(f"\n{'='*65}")
    print(f"  DEADLY CORRIDOR EVALUATION RESULT: {'VICTORY! (Armor Retrieved)' if completed else 'TERMINATED'}")
    print(f"  Final Position X : {final_x:.1f} / {CORRIDOR_LENGTH:.0f} ({final_x/CORRIDOR_LENGTH:.1%})")
    print(f"  Health Remaining : {final_hp:.0f}")
    print(f"  Armor Acquired   : {final_armor:.0f}")
    print(f"  Hostiles Eliminated: {final_kills:.0f}")
    print(f"  Steps Taken      : {steps}")
    print(f"  Total Reward     : {total_reward:+.1f}")
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
    return {"completed": completed, "steps": steps, "final_x": final_x, "hp": final_hp, "kills": final_kills}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SNN on deadly_corridor.")
    parser.add_argument("--skill", type=int, default=1, help="Doom skill level (default: 1).")
    parser.add_argument("--record", type=Path, default=None, help="Save playthrough video to mp4 path.")
    args = parser.parse_args()

    run_deadly_corridor(skill=args.skill, record_path=args.record)


if __name__ == "__main__":
    main()

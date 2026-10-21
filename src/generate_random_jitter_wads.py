"""Generate multiple randomized jittered WAD maps for stress testing."""

import struct
import random
from pathlib import Path

MONSTER_TYPES = {3004, 9, 3001, 3002, 58, 3003, 3005, 3006, 65, 66, 67, 68, 69, 71}

def create_jittered_wad(source_wad: Path, target_wad: Path, seed: int, max_jitter: int = 35) -> int:
    rng = random.Random(seed)
    with open(source_wad, "rb") as f:
        wad_bytes = bytearray(f.read())

    things_offset = 12
    things_size = 2000
    num_things = things_size // 10

    modified = 0
    for i in range(num_things):
        idx = things_offset + i * 10
        x, y, angle, ttype, flags = struct.unpack("<hhhhh", wad_bytes[idx:idx+10])
        if ttype in MONSTER_TYPES:
            dx = rng.choice([-max_jitter, -20, 20, max_jitter])
            dy = rng.choice([-max_jitter, -20, 20, max_jitter])
            new_x = max(-32768, min(32767, x + dx))
            new_y = max(-32768, min(32767, y + dy))
            struct.pack_into("<hh", wad_bytes, idx, new_x, new_y)
            modified += 1

    target_wad.parent.mkdir(parents=True, exist_ok=True)
    with open(target_wad, "wb") as f:
        f.write(wad_bytes)
    return modified

def create_cfg(target_cfg: Path, wad_path: Path):
    target_cfg.parent.mkdir(parents=True, exist_ok=True)
    rel_path = f"../../{wad_path.name}"
    content = f"""# Jittered test scenario for {wad_path.name}
doom_scenario_path = {rel_path}
episode_timeout = 21000
available_buttons = {{MOVE_FORWARD MOVE_BACKWARD MOVE_LEFT MOVE_RIGHT TURN_LEFT TURN_RIGHT ATTACK USE}}
available_game_variables = {{HEALTH ARMOR AMMO2 AMMO3 KILLCOUNT ITEMCOUNT SECRETCOUNT POSITION_X POSITION_Y ANGLE}}
labels_buffer_enabled = true
screen_resolution = RES_160X120
screen_format = GRAY8
render_hud = true
render_crosshair = false
render_weapon = true
render_decals = false
render_particles = false
render_effects_sprites = false
mode = PLAYER
living_reward = 0
death_penalty = 0
doom_skill = 3
"""
    target_cfg.write_text(content, encoding="utf-8")

def main():
    root = Path(".")
    source_wad = root / "freedoom2.wad"
    if not source_wad.exists():
        print(f"Error: {source_wad} not found.")
        return

    for s in [10, 20, 30]:
        wad_name = f"freedoom2_jitter_s{s}.wad"
        wad_file = root / wad_name
        cfg_file = root / "scenarios" / "generated" / f"level01_jitter_s{s}.cfg"
        count = create_jittered_wad(source_wad, wad_file, seed=s, max_jitter=32)
        create_cfg(cfg_file, wad_file)
        print(f"Created {wad_name} (seed={s}, {count} monsters displaced) -> {cfg_file}")

if __name__ == "__main__":
    main()

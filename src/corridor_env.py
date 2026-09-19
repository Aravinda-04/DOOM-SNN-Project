"""Reproducible generated UDMF corridors, independent of basic.wad.

The terminal objective is reaching the far end alive (and killing both enemies
in combat). This is an environment-defined exit region, not a Doom exit switch.
"""
from __future__ import annotations

import random
import struct
import uuid
from pathlib import Path

import numpy as np
import vizdoom as vzd

from env import DoomEnvironment
from project_paths import PROJECT_ROOT

SCENARIO_VERSION = 1
ASSET_DIR = PROJECT_ROOT / "scenarios" / "generated"
CORRIDOR_CONFIG_PATH = PROJECT_ROOT / "scenarios" / "corridor.cfg"
BUTTONS = (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
           vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK)
ACTION_NAMES = ("forward", "strafe_left", "strafe_right", "turn_left",
                "turn_right", "attack", "forward_attack")
ACTIONS = ([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 1])
NAVIGATION_ACTIONS = tuple(range(5))
COMBAT_ACTIONS = tuple(range(len(ACTIONS)))


def allowed_actions(stage):
    if stage == 'navigation':
        return NAVIGATION_ACTIONS
    if stage == 'combat':
        return COMBAT_ACTIONS
    raise ValueError(f'Unknown corridor stage: {stage}')


def case_geometry(case):
    """Eight balanced cases per split; held-out cases change enemy positions."""
    mirror = 1 if case % 2 == 0 else -1
    side = 1 if (case // 2) % 2 == 0 else -1
    held_out = case >= 8
    vertices = [(-128, -128), (-128, 128), (512, 128),
                (512, 768), (768, 768), (768, -128)]
    vertices = [(x, mirror * y) for x, y in vertices]
    if mirror < 0:
        vertices.reverse()  # Keep the interior on the right of one-sided walls.
    enemies = [(280 + (case // 4 % 2) * 80 + 25 * held_out, mirror * side * 70),
               (640 - side * (65 if held_out else 80), mirror * (410 + 45 * held_out))]
    return vertices, enemies, mirror


def map_text(case, combat):
    vertices, enemies, mirror = case_geometry(case)
    blocks = ['namespace = "ZDoom";',
              'sector { heightfloor = 0; heightceiling = 128; '
              'texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; lightlevel = 192; }']
    for x, y in vertices:
        blocks.append(f'vertex {{ x = {x}.0; y = {y}.0; }}')
    for i in range(len(vertices)):
        blocks.append('sidedef { sector = 0; texturemiddle = "STARTAN3"; }')
        blocks.append(f'linedef {{ v1 = {i}; v2 = {(i+1)%len(vertices)}; '
                      f'sidefront = {i}; blocking = true; }}')
    things = [(0, 0, 1, 0), (640, mirror * 690, 2028, 0)]
    if combat:
        things += [(x, y, 3004, 180) for x, y in enemies]  # Two former humans.
    for x, y, kind, angle in things:
        blocks.append(f'thing {{ x = {x}.0; y = {y}.0; type = {kind}; angle = {angle}; '
                      'skill1 = true; skill2 = true; skill3 = true; skill4 = true; '
                      'skill5 = true; single = true; }')
    return '\n'.join(blocks).encode('ascii')


def build_scenario(destination=None):
    """Generate map assets from source; no external map editor is required."""
    destination = Path(destination or ASSET_DIR / f"corridor-v{SCENARIO_VERSION}.wad")
    if destination.exists():
        return destination
    lumps = []
    for combat in (True, False):
        for case in range(16):
            number = case + 1 + (0 if combat else 16)
            lumps.extend([(f'MAP{number:02}', b''), ('TEXTMAP', map_text(case, combat)),
                          ('ENDMAP', b'')])
    body, directory = bytearray(), bytearray()
    for name, data in lumps:
        directory.extend(struct.pack('<ii8s', 12 + len(body), len(data), name.encode()))
        body.extend(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(struct.pack('<4sii', b'PWAD', len(lumps), 12 + len(body))
                            + body + directory)
    return destination


def task_outcome(info, stage, timeout):
    if info['dead']:
        return 'death'
    reached = info['x'] >= 540 and info['y_progress'] >= 640
    if reached and (stage == 'navigation' or info['kills'] >= 2):
        return 'complete'
    if info['tics'] >= timeout or info['engine_finished']:
        return 'timeout'
    return 'running'


class CorridorEnvironment:
    preprocess_frame = DoomEnvironment.preprocess_frame
    action_names = ACTION_NAMES
    actions = ACTIONS

    def __init__(self, stage='navigation', split='train', seed=0, render=False, timeout=None,
                 debug_objects=False, config_file=None, allowed_actions_override=None):
        if stage not in ('navigation', 'combat') or split not in ('train', 'eval'):
            raise ValueError('Invalid corridor stage or split')
        self.stage, self.split = stage, split
        self.allowed_actions = tuple(
            allowed_actions_override if allowed_actions_override is not None
            else allowed_actions(stage)
        )
        self.config_path = Path(config_file or CORRIDOR_CONFIG_PATH).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f'Corridor configuration not found: {self.config_path}')
        self.rng = random.Random(seed)
        build_scenario()
        self.game = vzd.DoomGame()
        # Keep engine settings independent of basic.wad and simultaneous envs.
        self.engine_config = ASSET_DIR / f'engine-{uuid.uuid4().hex}.ini'
        try:
            self.game.load_config(str(self.config_path))
            actual_buttons = tuple(self.game.get_available_buttons())
            if actual_buttons != BUTTONS:
                raise ValueError(
                    f'Corridor button order mismatch in {self.config_path}: '
                    f'expected {BUTTONS}, got {actual_buttons}. '
                    'The seven policy actions depend on this exact button order.'
                )
            if self.game.get_screen_format() != vzd.ScreenFormat.GRAY8:
                raise ValueError('Corridor policy preprocessing requires screen_format = GRAY8')
            required = {vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                        vzd.GameVariable.ANGLE, vzd.GameVariable.HEALTH,
                        vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT}
            missing = required - set(self.game.get_available_game_variables())
            if missing:
                raise ValueError(f'Corridor configuration is missing game variables: {missing}')
            # Only runtime choices override the scenario defaults.
            self.game.set_doom_config_path(str(self.engine_config))
            self.game.set_seed(seed)
            self.game.set_window_visible(render)
            self.game.set_objects_info_enabled(debug_objects)
            if timeout is not None:
                self.game.set_episode_timeout(timeout)
            self.timeout = self.game.get_episode_timeout()
            if self.timeout <= self.game.get_episode_start_time():
                raise ValueError('Corridor timeout must exceed the configured episode start time')
            self.game.init()
        except Exception:
            self.close()
            raise
        self.case = 0
        self.mirror = 1
        self.done = False
        self.best_progress = 0.0
        self.no_progress_steps = 0

    def reset(self, case=None):
        self.case = self.rng.randrange(8) + (8 if self.split == 'eval' else 0) if case is None else case
        if not 0 <= self.case < 16:
            raise ValueError('case must be in [0, 15]')
        self.mirror = case_geometry(self.case)[2]
        number = self.case + 1 + (16 if self.stage == 'navigation' else 0)
        self.game.set_doom_map(f'map{number:02}')
        self.game.new_episode()
        self.done = False
        self.previous = self.info()
        self.best_progress = self.potential(self.previous)
        self.no_progress_steps = 0
        return self.preprocess_frame(self.game.get_state().screen_buffer)

    def info(self):
        def value(variable):
            return float(self.game.get_game_variable(variable))
        return {'x': value(vzd.GameVariable.POSITION_X),
                'y_progress': self.mirror * value(vzd.GameVariable.POSITION_Y),
                'angle': value(vzd.GameVariable.ANGLE),
                'health': value(vzd.GameVariable.HEALTH),
                'ammo': value(vzd.GameVariable.AMMO2),
                'kills': value(vzd.GameVariable.KILLCOUNT),
                'tics': self.game.get_episode_time(), 'dead': self.game.is_player_dead(),
                'engine_finished': self.game.is_episode_finished()}

    @staticmethod
    def potential(info):
        # Distance along the L-shaped route; useful only for training reward.
        return (min(max(info['x'], 0), 640) + max(info['y_progress'], 0)) / 1280

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset before stepping a finished corridor episode')
        if action not in self.allowed_actions:
            raise ValueError(f'Action {action} is unavailable in the {self.stage} stage')
        before = self.previous
        self.game.make_action(self.actions[action], 4)
        after = self.info()
        outcome = task_outcome(after, self.stage, self.timeout)
        self.done = outcome != 'running'
        distance = np.hypot(after['x'] - before['x'], after['y_progress'] - before['y_progress'])
        after['stalled'] = action in (0, 1, 2, 6) and distance < 0.5
        after['shots'] = max(0, before['ammo'] - after['ammo'])
        progress = self.potential(after)
        if progress > self.best_progress + 0.005:
            self.best_progress = progress
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        reward_components = {
            'living': -0.04,
            'kill': 10 * (after['kills'] - before['kills']),
            'completion': 100 if outcome == 'complete' else 0,
            'death': -25 if outcome == 'death' else 0,
            'timeout': -25 if outcome == 'timeout' else 0,
            'no_progress': (
                -0.2 if self.stage == 'navigation' and self.no_progress_steps > 12
                and outcome == 'running' else 0
            ),
        }
        reward = sum(reward_components.values())
        terminal_potential = 0 if self.done else self.potential(after)
        after['shaping'] = 20 * (0.99 * terminal_potential - self.potential(before))
        after['reward_components'] = reward_components
        after['progress'] = progress
        after['best_progress'] = self.best_progress
        after['no_progress_steps'] = self.no_progress_steps
        after['outcome'] = outcome
        state = self.game.get_state()
        observation = self.preprocess_frame(state.screen_buffer) if state else np.zeros((84,84), np.float32)
        self.previous = after
        return observation, reward, self.done, after

    def close(self):
        self.game.close()
        self.engine_config.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

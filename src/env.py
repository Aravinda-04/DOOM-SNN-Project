import vizdoom as vzd
import numpy as np
import cv2
from pathlib import Path

from project_paths import DEFAULT_CONFIG_PATH

class DoomEnvironment:
    def __init__(self, config_file=None, render=False, seed=None):
        """
        Initializes the ViZDoom environment.
        :param config_file: The ViZDoom configuration file to load (e.g., basic.cfg).
        :param render: Whether to show the game window.
        :param seed: Optional ViZDoom random seed.
        """
        self.game = vzd.DoomGame()

        config_path = Path(config_file or DEFAULT_CONFIG_PATH).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"ViZDoom configuration not found: {config_path}")

        try:
            self.game.load_config(str(config_path))
            if seed is not None:
                self.game.set_seed(seed)

            # Optimize performance by reducing resolution since the AI doesn't need 4K
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            # Use grayscale because color is not strictly necessary for basic tasks and saves memory
            self.game.set_screen_format(vzd.ScreenFormat.GRAY8)

            # Display the window if requested
            self.game.set_window_visible(render)

            # Start the engine
            self.game.init()
        except Exception as e:
            self.game.close()
            raise RuntimeError(
                f"Could not initialize ViZDoom with config '{config_path}': {e}"
            ) from e
        
        # Define the action space. For basic.cfg, we have 3 buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK
        # An action is passed as a boolean list corresponding to these buttons
        self.actions = [
            [True, False, False, False, False],   # 0: Move Left
            [False, True, False, False, False],   # 1: Move Right
            [False, False, True, False, False],   # 2: Turn Left
            [False, False, False, True, False],   # 3: Turn Right
            [False, False, False, False, True],   # 4: Attack
        ]
        self.action_names = (
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
            "attack",
        )
        
    def preprocess_frame(self, frame):
        """
        Resizes the raw ViZDoom frame and normalizes the pixel values.
        """
        # Resize to 84x84 (A standard RL input size)
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalize pixel values from 0-255 to 0.0-1.0
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def reset(self):
        """
        Starts a new episode and returns the initial processed frame.
        """
        self.game.new_episode()
        state = self.game.get_state()
        return self.preprocess_frame(state.screen_buffer)

    def step(self, action_idx):
        """
        Takes a step in the environment using the provided action index.
        :returns: next_state, reward, done_flag
        """
        action = self.actions[action_idx]
        
        # Make the action and skip 4 frames (tics) to speed up learning
        reward = self.game.make_action(action, 4)
        done = self.game.is_episode_finished()
        
        if not done:
            state = self.game.get_state()
            next_state = self.preprocess_frame(state.screen_buffer)
        else:
            # If the episode is over, return a blank frame
            next_state = np.zeros((84, 84), dtype=np.float32)
            
        return next_state, reward, done

    def diagnostics(self):
        """Return game variables used for evaluation diagnostics."""
        variables = {
            "ammo": vzd.GameVariable.AMMO2,
            "kill_count": vzd.GameVariable.KILLCOUNT,
            "position_x": vzd.GameVariable.POSITION_X,
            "position_y": vzd.GameVariable.POSITION_Y,
        }
        values = {}
        for name, variable in variables.items():
            try:
                values[name] = float(self.game.get_game_variable(variable))
            except Exception:
                values[name] = None

        target_visible = False
        target_horizontal_offset = None
        if not self.game.is_episode_finished():
            state = self.game.get_state()
            labels = [
                label
                for label in (state.labels if state is not None else [])
                if label.object_name != "DoomPlayer"
            ]
            if labels:
                target = max(labels, key=lambda label: label.width * label.height)
                target_visible = True
                target_center_x = target.x + target.width / 2
                target_horizontal_offset = (target_center_x - 80.0) / 80.0

        values.update(
            {
                "episode_time": int(self.game.get_episode_time()),
                "total_reward": float(self.game.get_total_reward()),
                "player_dead": bool(self.game.is_player_dead()),
                "finished": bool(self.game.is_episode_finished()),
                "target_visible": target_visible,
                "target_horizontal_offset": target_horizontal_offset,
            }
        )
        return values

    def close(self):
        """Shuts down the ViZDoom engine cleanly."""
        self.game.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

if __name__ == "__main__":
    # Quick test to ensure the environment works
    print("Initializing ViZDoom...")
    try:
        env = DoomEnvironment(render=True)
        state = env.reset()
        
        print(f"Success! Initial State Shape: {state.shape}")
        print("Running a few random actions...")
        
        for i in range(50):
            if env.game.is_episode_finished():
                env.reset()
            
            # Choose a random action
            action_idx = np.random.randint(0, 3)
            next_state, reward, done = env.step(action_idx)
            
            print(f"Step {i+1} | Action: {action_idx} | Reward: {reward} | Done: {done}")
            
        env.close()
    except Exception as e:
        print(f"\nFailed to initialize environment. Error: {e}")
        print("Note: If you are missing 'basic.cfg' or 'basic.wad', you can find them in the ViZDoom GitHub repository under 'scenarios/'.")

import vizdoom as vzd
import numpy as np
import cv2

class DoomEnvironment:
    def __init__(self, config_file="basic.cfg", render=False):
        """
        Initializes the ViZDoom environment.
        :param config_file: The ViZDoom configuration file to load (e.g., basic.cfg).
        :param render: Whether to show the game window.
        """
        self.game = vzd.DoomGame()
        
        # Try to load the scenario configuration
        try:
            # We assume basic.cfg is available in the run directory or ViZDoom path
            self.game.load_config(config_file)
        except Exception as e:
            print(f"Warning: Could not load config '{config_file}'. Error: {e}")
            print("Please ensure you have the ViZDoom scenario files (e.g., basic.cfg and basic.wad) in the project root.")
            
        # Optimize performance by reducing resolution since the AI doesn't need 4K
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        # Use grayscale because color is not strictly necessary for basic tasks and saves memory
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        
        # Display the window if requested
        self.game.set_window_visible(render)
        
        # Start the engine
        self.game.init()
        
        # Define the action space. For basic.cfg, we have 3 buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK
        # An action is passed as a boolean list corresponding to these buttons
        self.actions = [
            [True, False, False],  # 0: Move Left
            [False, True, False],  # 1: Move Right
            [False, False, True]   # 2: Attack
        ]
        
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

    def close(self):
        """Shuts down the ViZDoom engine cleanly."""
        self.game.close()

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

import os
import time
import torch
import numpy as np

# Import our custom modules
from env import DoomEnvironment
from network import SpikingQNetwork

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Spiking DQN on {device}...")
    
    # Initialize environment with render=True so we can watch it play
    print("Starting ViZDoom environment (A window should open)...")
    env = DoomEnvironment(render=True)
    action_size = len(env.actions)
    
    # Load the trained model
    policy_net = SpikingQNetwork(action_size=action_size).to(device)
    
    model_path = "models/best_snn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at '{model_path}'.")
        print("Please ensure you have successfully completed a training run first.")
        env.close()
        return
        
    # Load weights (using weights_only=True for safety)
    policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy_net.eval()
    print("Model loaded successfully!")
    
    num_episodes = 10
    
    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                # Format state for network: (Batch=1, Channels=1, H, W)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward pass through the Spiking CNN
                q_values, _ = policy_net(state_t)
                
                # We always take the best action (Epsilon = 0)
                action = q_values.max(1)[1].item()
                
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            
            # Slow down the loop slightly so it's watchable by a human
            time.sleep(0.05)
            
        print(f"Evaluation Episode {i_episode + 1}/{num_episodes} | Total Reward: {total_reward:.1f}")
        
    env.close()
    print("Evaluation completed.")

if __name__ == "__main__":
    main()

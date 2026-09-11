import math
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import our custom modules
from env import DoomEnvironment
from network import SpikingQNetwork

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000
LR = 1e-4
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 5 # Set to 5 for initial testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_model(memory, policy_net, optimizer, criterion):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    # Convert to tensors
    # states are of shape (84, 84), we need (1, 84, 84) to add channel dim
    state_batch = torch.stack([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch_state]).to(device)
    action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(device)
    next_state_batch = torch.stack([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch_next_state]).to(device)
    done_batch = torch.tensor(batch_done, dtype=torch.float32).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_state_values = policy_net(next_state_batch).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    # Compute loss (MSE Loss)
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent exploding gradients (common in BPTT)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    
    optimizer.step()

def main():
    print(f"Starting Spiking DQN training on {device}...")
    
    # Initialize environment and networks
    env = DoomEnvironment(render=False) # Disable render for faster training
    action_size = len(env.actions)
    
    policy_net = SpikingQNetwork(action_size=action_size).to(device)
    # Usually DQN has a target network, but for simplicity in this PoC we use just one or we can add it later.
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0

    for i_episode in range(NUM_EPISODES):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        
        start_time = time.time()
        
        while not done:
            # Epsilon-greedy action selection
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if random.random() > eps_threshold:
                with torch.no_grad():
                    # Format state for network: (Batch=1, Channels=1, H, W)
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = random.randrange(action_size)

            # Step the environment
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store the transition in memory
            memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            optimize_model(memory, policy_net, optimizer, criterion)
            
        episode_time = time.time() - start_time
        print(f"Episode {i_episode + 1}/{NUM_EPISODES} | Reward: {total_reward:.1f} | Epsilon: {eps_threshold:.2f} | Time: {episode_time:.1f}s")

    env.close()
    print("Training test completed successfully!")

if __name__ == "__main__":
    main()

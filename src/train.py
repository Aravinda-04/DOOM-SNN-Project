import os
import math
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import our custom modules
from env import DoomEnvironment
from networks.snn import SpikingQNetwork
from networks.ffnn import FeedForwardQNetwork
from networks.rsnn import RSNNQNetwork

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 3000 # Much faster decay so the agent actually exploits
LR = 1e-4
MEMORY_SIZE = 10000
TAU = 0.005 # Soft update rate
NUM_EPISODES = 200
SPARSITY_WEIGHT = 1e-6 # Lowered so it doesn't overpower the RL loss
REWARD_SCALE = 100.0
MODEL_TYPE = "SNN" # Options: "SNN", "FFNN", "RSNN"

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

def optimize_model(memory, policy_net, target_net, optimizer, criterion):
    if len(memory) < BATCH_SIZE:
        return None
    
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

    # Compute Q(s_t, a) and spk_count
    q_values, spk_count = policy_net(state_batch)
    state_action_values = q_values.gather(1, action_batch).squeeze(1)

    # Compute V(s_{t+1}) for all next states using the target network.
    with torch.no_grad():
        next_q_values, _ = target_net(next_state_batch)
        next_state_values = next_q_values.max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    # Compute loss (MSE Loss) + Sparsity Loss
    mse_loss = criterion(state_action_values, expected_state_action_values)
    
    # spk_count is the total sum of spikes across the batch and time steps.
    # We divide by BATCH_SIZE so the penalty doesn't scale with batch size.
    sparsity_loss = (spk_count / BATCH_SIZE) * SPARSITY_WEIGHT
    loss = mse_loss + sparsity_loss

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent exploding gradients (common in BPTT)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

def apply_weight_constraints(model):
    """
    Applies quantization constraints by clipping weights.
    Relaxed to [-5.0, 5.0] to prevent gradient destruction during early training.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(-5.0, 5.0)

def main():
    print(f"Starting Spiking DQN training on {device}...")
    writer = SummaryWriter('runs/doom_snn')
    
    # Initialize environment and networks
    env = DoomEnvironment(render=False) # Disable render for faster training
    action_size = len(env.actions)
    
    if MODEL_TYPE == "SNN":
        policy_net = SpikingQNetwork(action_size=action_size).to(device)
        target_net = SpikingQNetwork(action_size=action_size).to(device)
    elif MODEL_TYPE == "FFNN":
        policy_net = FeedForwardQNetwork(action_size=action_size).to(device)
        target_net = FeedForwardQNetwork(action_size=action_size).to(device)
    elif MODEL_TYPE == "RSNN":
        policy_net = RSNNQNetwork(action_size=action_size).to(device)
        target_net = RSNNQNetwork(action_size=action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    os.makedirs("models", exist_ok=True)
    best_reward = -float('inf')
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss() # Huber Loss is more stable for DQN than MSE
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
                    q_values, _ = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = random.randrange(action_size)

            # Step the environment
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store the scaled transition in memory for stable Q-learning
            scaled_reward = reward / REWARD_SCALE
            memory.push(state, action, scaled_reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            loss = optimize_model(memory, policy_net, target_net, optimizer, criterion)
            apply_weight_constraints(policy_net)
            
            # Soft update of target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if loss is not None:
                writer.add_scalar('Loss', loss, steps_done)
            
        episode_time = time.time() - start_time
        print(f"Episode {i_episode + 1}/{NUM_EPISODES} | Reward: {total_reward:.1f} | Epsilon: {eps_threshold:.2f} | Time: {episode_time:.1f}s")
        
        # Checkpointing
        if total_reward > best_reward:
            best_reward = total_reward
            save_path = f"models/best_{MODEL_TYPE.lower()}.pth"
            torch.save(policy_net.state_dict(), save_path)
            print(f"--> New best reward: {best_reward:.1f}. Model saved to {save_path}")
            
        # Log episode metrics
        writer.add_scalar('Reward', total_reward, i_episode)
        writer.add_scalar('Epsilon', eps_threshold, i_episode)
        writer.flush()

    env.close()
    writer.close()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

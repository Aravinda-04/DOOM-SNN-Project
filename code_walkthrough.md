# DOOM-SNN-Project: Complete Code Walkthrough

This document breaks down the entire project file-by-file so you can understand exactly how the Spiking Neural Network learns to play DOOM.

## 1. `env.py`: The Environment (The World)
This file is the bridge between our Python code and the ViZDoom game engine. 

- **Initialization (`__init__`)**: We create a `vzd.DoomGame()` instance. We load `basic.cfg`, which defines a simple scenario: kill the monster as fast as possible. We set the resolution to a tiny 160x120 grayscale window because the AI doesn't need 4K graphics—it just needs basic shapes.
- **`preprocess_frame`**: The raw 160x120 game frame is resized to 84x84 (a standard size in Reinforcement Learning to save memory). We also divide the pixel values by 255.0 to squeeze them into a range of `0.0` to `1.0`. This is crucial for the neural network.
- **`step`**: When the AI chooses an action (like Move Left), this function sends it to ViZDoom. We use `make_action(action, 4)`, which means the AI presses the button and holds it for 4 frames (frame skipping). This speeds up training immensely because the game state doesn't change much from a single frame to the next.

## 2. `network.py`: The Spiking Brain
This is where the magic happens. We use `snnTorch` to build a brain inspired by real biology.

- **`SpikingQNetwork` Class**: Instead of normal layers, we use convolutional layers (`nn.Conv2d`) followed by Leaky Integrate-and-Fire neurons (`snn.Leaky`).
- **Leaky Integrate-and-Fire (`snn.Leaky`)**: In a normal neural net, neurons just output a decimal number. A Leaky neuron, however, stores "voltage" (membrane potential). It adds up input over time, and if the voltage crosses a threshold, it fires a binary "spike" (a 1) and resets. If it doesn't get input, it slowly "leaks" voltage away.
- **`spikegen.rate`**: This converts the image pixels into a train of spikes. A bright pixel (value close to 1.0) will fire lots of spikes over time. A dark pixel (0.0) fires no spikes.
- **The `forward` loop**: Real brains process data over time, not all at once. We loop `for step in range(self.num_steps)` (10 times). The image is pushed through the network 10 times. The final output layer's voltage (`mem5`) is accumulated over these 10 steps to represent the "Q-Values" (the AI's estimation of how good each action is).
- **Sparsity Tally**: We count up every single spike generated in the network (`total_spikes`) so we can penalize the AI later if it fires too much (to save power on the FPGA).

## 3. `train.py`: The Learning Loop
This is a Spiking Deep Q-Network (SDQN). It teaches the brain using trial and error.

- **Epsilon-Greedy Exploration (`eps_threshold`)**: The AI uses a random number generator. If the random number is less than Epsilon, it takes a random action. This ensures the AI tries new things instead of just spinning in circles forever. Epsilon slowly decays over time.
- **`ReplayMemory`**: The AI remembers every single move it makes (state, action, reward, next_state) in a buffer. Instead of learning from just the very last frame, it grabs a random batch of 64 past memories and learns from them. This breaks correlation and stabilizes learning.
- **Target Network**: We actually have *two* brains. The `policy_net` is actively learning, and the `target_net` provides the "answers" (expected rewards) to study against. If they were the same brain, the answers would constantly change, confusing the AI. We slowly copy the policy brain into the target brain (a "Soft Update" using `TAU`).
- **`optimize_model`**: This calculates the loss (the error). 
  - `mse_loss`: The difference between what the AI *thought* it would score, and what it *actually* scored.
  - `sparsity_loss`: A penalty for firing too many spikes.
  - The AI uses backpropagation (`loss.backward()`) to tweak its weights to minimize this combined error.
- **`apply_weight_constraints`**: Before deployment to an FPGA chip, the weights must be integers (like -2, 0, 3), not decimals. To prepare the network for this, we aggressively clip the weights between -5.0 and 5.0 to simulate hardware limits.

## 4. `eval.py`: The Final Test
Once training is done, we run this script.

- It loads the exact same game and network, but crucially sets **Epsilon to 0**. 
- `action = q_values.max(1)[1].item()`: The AI looks at the Q-values (the accumulated voltage for Move Left, Move Right, and Attack) and simply picks the one with the highest voltage. No random guessing, just pure learned skill.

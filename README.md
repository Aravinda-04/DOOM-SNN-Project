# DOOM-SNN-Project

Autonomous DOOM gameplay simulation using Spiking Neural Networks (SNNs) in Python, designed for future deployment on the PeraMorphIQ neuromorphic FPGA accelerator.

## Overview
This project sets up a reinforcement learning environment using **ViZDoom** and builds a Spiking Convolutional Neural Network (SCNN) using **snnTorch**. The network learns to play DOOM and is constrained (via quantization and sparsity) to ensure compatibility with FPGA hardware.

## Current Progress & Project Structure
- `src/env.py`: ViZDoom environment setup, observation extraction, and frame preprocessing (grayscale, 84x84).
- `src/network.py`: Spiking Convolutional Neural Network (SCNN) using `snnTorch` with Leaky Integrate-and-Fire (LIF) neurons and rate-coded visual input.
- `src/train.py`: Training loop for the Spiking Deep Q-Network (SDQN) with Hardware Constraints (Sparsity Penalty & Weight Quantization) and TensorBoard logging.
- `requirements.txt`: Python dependencies.
- `proposal.md`: Initial project proposal, architecture details, and FPGA deployment strategy.

## Installation
1. Clone the repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the required ViZDoom scenario files (e.g., `basic.cfg` and `basic.wad`) from the [ViZDoom repository](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios) and place them in the project root.

## Usage
**1. Test the Environment**
Verify that ViZDoom is initializing correctly and capturing frames:
```bash
python src/env.py
```

**2. Test the Spiking Neural Network**
Verify the network structure and forward pass:
```bash
python src/network.py
```

**3. Run the Training Loop**
Start training the Spiking DQN agent:
```bash
python src/train.py
```

**4. Monitor Training with TensorBoard**
To view the training progress (Loss, Reward, Epsilon):
```bash
tensorboard --logdir runs
```

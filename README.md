# DOOM-SNN-Project

Autonomous DOOM gameplay simulation using Spiking Neural Networks (SNNs) in Python, designed for future deployment on the PeraMorphIQ neuromorphic FPGA accelerator.

## Overview
This project sets up a reinforcement learning environment using **ViZDoom** and builds a Spiking Convolutional Neural Network (SCNN) using **snnTorch**. The network learns to play DOOM and is constrained (via quantization and sparsity) to ensure compatibility with FPGA hardware.

## Project Structure
- `src/env.py`: ViZDoom environment setup, observation extraction, and frame preprocessing.
- `requirements.txt`: Python dependencies.
- `proposal.md`: Initial project proposal, architecture details, and FPGA deployment strategy.

## Installation
1. Clone the repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the required ViZDoom scenario files (e.g., `basic.cfg` and `basic.wad`) from the [ViZDoom repository](https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios) and place them in the project root.

## Running the Environment Test
To verify that ViZDoom is initializing correctly and capturing frames:
```bash
python src/env.py
```

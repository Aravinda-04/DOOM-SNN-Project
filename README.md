# DOOM Spiking Neural Network (SNN) Accelerator

This repository contains a Python-based reinforcement learning pipeline designed to train neural networks to play DOOM autonomously. It is built as a **comparative study** between standard Artificial Neural Networks and biological Spiking Neural Networks (SNNs) prior to hardware deployment on an FPGA neuromorphic accelerator.

## Current Progress

**Completed:**
- Set up the ViZDoom environment (`basic.cfg`) with a custom Python wrapper.
- Implemented Spiking (SNN), Recurrent Spiking (RSNN), and Feed-Forward (FFNN) network architectures.
- Developed the Deep Q-Network (DQN) training loop with experience replay and soft target updates.
- Added stability fixes for SNN training (Huber Loss, hardware constraint weight clipping, sparsity loss scaling).
- Implemented a modular team architecture to allow independent training and evaluation of all 3 models.
- Verified that the SNN successfully learns to shoot the monster in under 200 episodes.

**Next Steps:**
- Complete long training runs for all three models (FFNN, SNN, RSNN) and gather TensorBoard logs.
- Analyze the performance differences between the models (training speed, accuracy, power efficiency).
- Export PyTorch weights to an integer-quantized format compatible with the PeraMorphIQ FPGA accelerator.
- Draft the final comparative study report.

## Supported Architectures
The project features a modular architecture to allow easy collaboration. The models are located in `src/networks/`:
1. **Feed-Forward Neural Network (FFNN)**: A standard, non-spiking Convolutional Neural Network baseline (`ffnn.py`).
2. **Spiking Neural Network (SNN)**: A biological network using Leaky Integrate-and-Fire neurons and rate coding (`snn.py`).
3. **Recurrent Spiking Neural Network (RSNN)**: An SNN equipped with internal recurrent memory loops for complex temporal tasks (`rsnn.py`).

## Installation

Make sure you have Python installed, and then install the required dependencies:
```bash
pip install torch torchvision
pip install snntorch
pip install vizdoom
pip install tensorboard
```
*Note: You will also need the DOOM `basic.wad` and `basic.cfg` files in the root directory for the ViZDoom environment to function.*

## How to Use

### 1. Training a Model
Open `src/train.py` and set the `MODEL_TYPE` variable at the top of the file to choose which brain you want to train:
```python
MODEL_TYPE = "SNN" # Options: "SNN", "FFNN", "RSNN"
```
Then, run the training script from the root directory:
```bash
python src/train.py
```
This will automatically save the best performing weights to the `models/` directory (e.g., `models/best_snn.pth`).

### 2. Evaluating a Model
To watch the fully trained AI play DOOM without any random exploration (Epsilon = 0), open `src/eval.py`, ensure the `MODEL_TYPE` matches the brain you want to test, and run:
```bash
python src/eval.py
```

### 3. Visualizing Training Progress
The training script automatically logs metrics like Reward and Epsilon decay. You can view interactive graphs of your AI's learning curve by starting a TensorBoard server:
```bash
tensorboard --logdir runs
```
Navigate to `http://localhost:6006` in your web browser to view the dashboard.

## Deep Dive
For a comprehensive line-by-line breakdown of how the environment, spiking mechanics, and Deep Q-Network training loop work, please refer to the `code_walkthrough.md` file!

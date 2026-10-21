# Project Proposal: Autonomous DOOM Agent using Spiking Neural Networks

## 1. Overview
This proposal outlines our initial approach for developing an autonomous DOOM gameplay simulation using Spiking Neural Networks (SNNs) in Python. The primary goal is to build and verify the software pipeline, keeping hardware-aware constraints in mind for eventual deployment on the PeraMorphIQ neuromorphic accelerator.

## 2. Environment & Interface
*   **Engine:** We will utilize **ViZDoom**, the standard Python-based reinforcement learning environment for DOOM.
    *   *What it does:* It runs the game in the background and outputs frames (pixels) and game variables (health, ammo) at every step, while accepting actions (move, shoot).
    *   *Why it is used:* To train an AI, we need a fast, programmatic way to step through the game, receive the visual state, and measure success (reward).
    *   *Justification:* ViZDoom is the industry standard for vision-based RL. It is incredibly fast, lightweight, and customizable compared to modern game engines.
*   **Action Space:** We will begin with a discrete, simplified action space (e.g., move left/right, attack) to establish baseline learning before scaling up to continuous movements.

## 3. SNN Framework Selection
*   **Framework:** We propose using **snnTorch**.
    *   *What it does:* It is a Python library built on top of PyTorch specifically designed for building and training Spiking Neural Networks (SNNs).
    *   *Why it is used:* Standard PyTorch is for traditional artificial neural networks (ANNs). We need specialized layers that communicate via binary spikes over time, mimicking neuromorphic hardware.
    *   *Justification:* `snnTorch` integrates directly with PyTorch, meaning we can use familiar syntax, powerful data loaders, and existing RL libraries without learning a new ecosystem.

## 4. Visual Encoding (Pixel to Spike)
*   **Approach:** We will implement **Rate Coding** or **Delta Modulation** (event-based encoding) to convert the visual frames into spike trains.
    *   *What it does:* It translates continuous, real-valued pixel data into a sequence of binary spikes spread across multiple time-steps.
    *   *Why it is used:* SNNs and neuromorphic hardware like PeraMorphIQ do not process floating-point numbers; they only understand binary events.
    *   *Justification:* Rate Coding is robust, while Delta Modulation is highly energy-efficient (sparse), making it ideal for hardware deployment. We plan to build a Spiking Convolutional Neural Network (SCNN) to directly extract features from these spikes.

## 5. Network Architecture & Neuron Model
*   **Neuron Type:** We will use **Leaky Integrate-and-Fire (LIF)** neurons.
    *   *What it does:* It models a biological brain cell that integrates incoming spikes over time. It "fires" (outputs a 1) if the sum passes a threshold; otherwise, its internal voltage "leaks" (decays back to zero).
    *   *Why it is used:* It gives the network a sense of memory and time, preventing it from firing based on ancient, irrelevant information.
    *   *Justification:* LIF provides an excellent balance between biological plausibility and computational efficiency, and is a standard on FPGA architectures like PeraMorphIQ.
*   **Architecture:** An SCNN for visual feature extraction, followed by Spiking fully-connected layers to output Q-values or policy probabilities for the agent's actions.

## 6. Training Methodology (Spiking RL)
*   **Algorithm:** We will adapt standard Reinforcement Learning algorithms such as **DQN (Deep Q-Network)** or **PPO (Proximal Policy Optimization)**.
    *   *What it does:* It allows the AI to learn by trial and error, receiving positive rewards for shooting a monster and negative rewards for losing health.
    *   *Why it is used:* We do not have human DOOM gameplay to train via Supervised Learning; the agent must figure out the optimal strategy itself.
    *   *Justification:* DQN has a proven track record of solving DOOM environments, and its adaptation to Spiking DQN (SDQN) is well-documented.
*   **Learning:** We will utilize **Surrogate Gradient Descent** alongside Backpropagation Through Time (BPTT).
    *   *What it does:* It mathematically updates the network's weights during training by "pretending" the non-differentiable spiking step function is smooth.
    *   *Why it is used:* Standard backpropagation fails because spikes are discontinuous jumps (zero gradient).
    *   *Justification:* This is currently the most successful and stable way to train deep, multi-layered SNNs from scratch.

## 7. Hardware-Aware Constraints (For PeraMorphIQ FPGA)
To ensure smooth transition to hardware deployment, our Python model will enforce constraints early on:
*   **Quantization:** Limiting synaptic weights to low-bit precision (e.g., 4-bit or 8-bit).
*   **Sparsity:** Encouraging sparse firing rates to minimize energy consumption.
*   **Time-step limitation:** Keeping the number of simulation time-steps per forward pass low.
    *   *Why these are used & Justification:* An FPGA has limited memory and computing logic. It cannot process millions of high-precision floating-point numbers fast enough. By enforcing these constraints in software during training, we guarantee the final model will fit and run efficiently on the physical PeraMorphIQ board in real-time.

## 8. FPGA Deployment Strategy (The Endgame)

### 1. What is an FPGA?
Most computer chips (like a CPU) have a fixed architecture. They have a set number of arithmetic units, memory registers, and instruction sets baked into the silicon at the factory. When you write a Python script, you are just feeding software instructions into this fixed hardware, asking it to perform calculations sequentially.

An FPGA, on the other hand, is a "blank canvas." It is a massive grid of generic, unconnected logic blocks. Instead of writing software that runs on the chip, you write a Hardware Description Language (like Verilog or VHDL) that tells the FPGA how to literally wire those blocks together.

In short: With a CPU, you write software. With an FPGA, you are literally programming the physical hardware layout of the chip. You can wire it to be a video encoder, a cryptography accelerator, or in this case, a neuromorphic neural network.

### 2. Why are FPGAs used for Neuromorphic Computing (SNNs)?
Spiking Neural Networks (SNNs) simulate how biological brains work. In a brain, millions of neurons operate simultaneously, and they only consume energy when they fire a "spike."

*   **The CPU/GPU Problem:** If you run an SNN in Python on a CPU/GPU, the computer has to run a massive for loop to check the voltage of every single neuron, one by one, at every millisecond of time. This is incredibly inefficient and consumes a lot of power.
*   **The FPGA Solution:** Because an FPGA lets you build custom circuits, you can physically instantiate thousands of "neuron circuits" that all run in parallel. There are no software loops. When a spike enters the FPGA, the physical electricity routes through the custom hardware instantly. It is massively parallel, incredibly fast, and consumes a tiny fraction of the power of a GPU.

### 3. How is the FPGA used in this specific DOOM project?
The PeraMorphIQ accelerator is built on an FPGA. The whole point of the project is to get the DOOM AI running on that physical board. However, trying to "train" an AI directly on an FPGA is incredibly difficult and slow.

Here is how the pipeline works and why our Python code matters:

*   **Software Prototyping (What we are doing now):** We use Python, snnTorch, and ViZDoom because Python is flexible. It allows us to experiment, make mistakes, and use standard tools like Backpropagation (BPTT) and Reinforcement Learning to find the exact network architecture and "weights" that allow the AI to win the game.
*   **Hardware Constraints:** Because an FPGA has a limited number of physical logic gates, it cannot store heavy decimal numbers (like 0.849372...). Therefore, in our Python code, we force the AI to learn using low-bit integers (e.g., -3, 0, +2)—this is called Quantization. We also force it to fire as few spikes as possible (Sparsity) so it fits on the board and saves power.
*   **Deployment (The Endgame):** Once the Python model masters DOOM, we will freeze it. The network architecture and the finalized integer weights will be extracted. Another tool will translate those weights into a hardware configuration file.
*   **Hardware Inference:** That configuration file is loaded onto the PeraMorphIQ FPGA. The FPGA physically morphs its circuits to match our SNN. Now, you plug the DOOM video feed directly into the FPGA, and the FPGA spits out joystick commands in real-time—no Python, no CPU, just pure, low-power hardware logic playing the game autonomously.

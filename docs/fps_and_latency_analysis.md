# FPS and Latency Analysis: Vision-Bridged SNN Pipeline

This document provides the formal mathematical and architectural breakdown of frame rates, neural network inference latencies, simulation throughput, and video playback timing for the Vision-Bridged Spiking Neural Network (SNN) agent on FreeDoom2 MAP01.

---

## 1. Executive Summary of Key Metrics

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Doom Game Engine Native Clock** | **35.0 FPS (Hz)** | ViZDoom / Doom internal game loop rate ($1\text{ tic} \approx 28.57\text{ ms}$) |
| **Agent Decision Frequency (Real-Time)** | **8.75 Hz** | With $\text{frame\_skip} = 4$, agent acts every 4 game tics |
| **Real-Time Step Time Budget** | **114.29 ms** | Maximum allowable latency per decision for real-time play |
| **Vision Encoder Latency (12-ch RGB)** | **4.82 ms** | Processes 4-frame stack $(1, 12, 84, 84)$ in **one pass** |
| **Recurrent LIF SNN Latency** | **30.67 ms** | 12 temporal recurrent integration steps |
| **Total Neural Compute Latency** | **35.49 ms** | Combined forward pass latency ($\text{CNN} + \text{SNN}$) |
| **End-to-End Decision Latency** | **~58.1 ms** | Total time including ViZDoom engine step and telemetry |
| **Live Simulation Throughput** | **17.2 steps/s** | Live agent decision frequency on CUDA |
| **Live Simulated Engine FPS** | **68.9 FPS** | Simulated Doom tics per second ($17.2 \times 4$) |
| **Simulation Speedup Factor** | **1.97×** | Ratio of simulated game speed to human real-time ($68.9 / 35.0$) |
| **Recorded Video Encoding Framerate** | **20.0 FPS** | OpenCV `VideoWriter` playback framerate |
| **Recorded Video Speedup Factor** | **2.29×** | Apparent fast-forward ratio in recorded `.mp4` playback |

---

## 2. In-Game Physics Clock & Agent Timing Budget

Classic Doom operates on a deterministic, fixed-timestep clock:
$$T_{\text{engine}} = 35\text{ tics / second} \implies \Delta t_{\text{tic}} = \frac{1000\text{ ms}}{35} \approx 28.571\text{ ms}$$

The reinforcement learning / imitation environment operates with a frame skip parameter:
$$k = \text{frame\_skip} = 4\text{ tics / decision step}$$

In real-time human play:
- The agent makes a decision every $4$ engine tics.
- The real-time step frequency is:
  $$f_{\text{decision, real-time}} = \frac{35\text{ tics/s}}{4\text{ tics/step}} = 8.75\text{ steps / second}$$
- The maximum permissible latency per decision step to maintain real-time play is:
  $$\Delta t_{\text{budget}} = \frac{1000\text{ ms}}{8.75\text{ steps/s}} = 4 \times 28.571\text{ ms} \approx \mathbf{114.29\text{ ms}}$$

Any pipeline with total step latency $\Delta t_{\text{step}} < 114.29\text{ ms}$ runs faster than real-time.

---

## 3. Neural Network Latency Breakdown

### 3.1 Vision Perception Encoder (12-Channel RGB CNN)
- **Observation Tensor Shape**: `(1, 12, 84, 84)`
  - $4\text{ stacked frames} \times 3\text{ RGB color channels} = 12\text{ input channels}$
- **Inference Mode**: Single forward pass.
  > **Note**: The Vision Encoder does **not** run 4 separate times. The entire 4-frame temporal stack is passed as a single 12-channel tensor through the convolutional layers simultaneously.
- **Measured Latency**:
  $$T_{\text{CNN}} = \mathbf{4.82\text{ ms}} \quad (207.3\text{ standalone inferences/sec})$$

### 3.2 Spiking Neural Network Policy (Recurrent LIF)
- **Input Feature Vector Shape**: `(1, 84)`
  - $42\text{ normalized continuous features} + 42\text{ boolean validity flags} = 84\text{ total inputs}$
- **Architecture**: Multi-layer Leaky Integrate-and-Fire (LIF) network with recurrence.
- **Temporal Integration**: Unrolled across $S = 12$ internal time-steps to accumulate membrane potentials and generate output spike rates for each action head (`locomotion`, `turning`, `attack`, `use`).
- **Measured Latency**:
  $$T_{\text{SNN}} = \mathbf{30.67\text{ ms}} \quad (32.6\text{ standalone inferences/sec})$$

### 3.3 Total Neural Compute Time per Decision
$$T_{\text{neural}} = T_{\text{CNN}} + T_{\text{SNN}} = 4.82\text{ ms} + 30.67\text{ ms} = \mathbf{35.49\text{ ms}}$$

---

## 4. End-to-End Pipeline Step Cycle

A complete closed-loop agent step consists of four sequential stages:

```
[Camera Buffer (12,84,84)]
       │
       ▼ (4.82 ms)
[Vision Encoder CNN] ─────────► Extracts (Screen X, Distance, Area, Visible)
       │
       ▼ (0.20 ms)
[Kinematic Telemetry & Corpse Memory] ──► Produces 84D Feature Vector
       │
       ▼ (30.67 ms)
[Recurrent LIF SNN (12 time-steps)] ───► Emits Factorized Action
       │
       ▼ (22.45 ms)
[ViZDoom Engine Step (4 tics)] ────────► Steps physics & renders new frame
       │
Total: ~58.14 ms
```

1. **Vision Inference**: $4.82\text{ ms}$
2. **Telemetry, Geometry, & Corpse Memory Rejection**: $\sim 0.20\text{ ms}$
3. **LIF SNN Forward Pass**: $30.67\text{ ms}$
4. **ViZDoom Physics Execution ($4$ tics)**: $\sim 22.45\text{ ms}$

$$\Delta t_{\text{total\_step}} = 4.82 + 0.20 + 30.67 + 22.45 \approx \mathbf{58.14\text{ ms}}$$

### 4.1 Live Throughput
$$\text{Throughput} = \frac{1000\text{ ms}}{58.14\text{ ms/step}} \approx \mathbf{17.20\text{ decision steps / second}}$$

### 4.2 Simulated Engine Clock Rate
$$\text{Engine FPS} = 17.20\text{ steps/s} \times 4\text{ tics/step} = \mathbf{68.80\text{ game tics / second}}$$

### 4.3 Simulation Speedup Factor
$$\text{Speedup}_{\text{sim}} = \frac{\text{Engine FPS}}{\text{Native Clock}} = \frac{68.80\text{ tics/s}}{35.00\text{ tics/s}} = \frac{114.29\text{ ms}}{58.14\text{ ms}} \approx \mathbf{1.97\times}$$

The agent completes gameplay tasks nearly twice as fast as real-time human speed.

---

## 5. Why the Recorded MP4 Video Looks Sped Up (2.29× Fast-Forward)

The apparent speedup during video playback is caused by a framerate mismatch in the OpenCV `VideoWriter` configuration.

### 5.1 The Mathematical Cause

In `src/evaluate_vision_spiking_policy.py`:
```python
video_writer = cv2.VideoWriter(
    str(rec_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    20.0,  # Playback Framerate = 20.0 FPS
    (640, 480),
)
```

1. Exactly **1 video frame** is written per agent decision step.
2. Each agent decision step represents **4 Doom game tics**:
   $$\Delta t_{\text{game\_per\_frame}} = \frac{4\text{ tics}}{35\text{ tics/sec}} \approx 0.114286\text{ seconds of in-game time}$$
3. The video player renders **20 recorded frames in 1 real-world second**:
   $$\text{In-Game Seconds Elapsed per Video Second} = 20\text{ frames/s} \times 0.114286\text{ s/frame} \approx \mathbf{2.2857\text{ seconds}}$$
4. The video playback speed ratio is therefore:
   $$\text{Playback Speedup} = \frac{20.0\text{ FPS}}{8.75\text{ FPS}} = \mathbf{2.2857\times\text{ fast-forward}}$$

### 5.2 Concrete Example: The Seed 3 Level Clear Run

| Dimension | In-Game Simulation | Recorded Video Playback |
| :--- | :---: | :---: |
| **Total Decision Steps** | 525 steps | 525 frames |
| **Engine Tics** | $525 \times 4 = 2,100\text{ tics}$ | N/A |
| **Duration** | $\frac{2,100}{35} = \mathbf{60.00\text{ seconds}}$ | $\frac{525}{20.0} = \mathbf{26.25\text{ seconds}}$ |
| **Speedup Ratio** | $1.0\times$ (normal time) | $\frac{60.00\text{ s}}{26.25\text{ s}} = \mathbf{2.29\times}$ |

A full 1-minute in-game run is played back in only 26 seconds.

---

## 6. How to Achieve True 1:1 Normal Real-Time Playback

To produce a recording that plays back at true 1:1 human real-time speed, use either of the following approaches:

### Method A: Set VideoWriter FPS to Match the Agent Decision Rate
Configure the writer to render at exactly $8.75\text{ FPS}$:
```python
video_writer = cv2.VideoWriter(
    str(rec_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    8.75,  # 35 tics/s / 4 tics/step = 8.75 FPS
    (640, 480),
)
```
*Result*: 525 frames played at 8.75 FPS will take exactly $60.0\text{ seconds}$, perfectly matching in-game time.

### Method B: Duplicate Frames for 35 FPS Display
Write each decision frame $4$ times (or capture all 4 intermediate tics) at $35.0\text{ FPS}$:
```python
video_writer = cv2.VideoWriter(
    str(rec_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    35.0,  # Native Doom FPS
    (640, 480),
)

# In the step loop:
for _ in range(4):
    video_writer.write(canvas)
```
*Result*: Produces standard 35 FPS video matching the classic engine clock rate at 1:1 real-time duration.

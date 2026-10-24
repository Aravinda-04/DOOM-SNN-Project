# Vision Perception Bridge: End-to-End Spiking AI Implementation Plan

**Document Version**: 1.0.0  
**Target Milestone**: Stage 2 of the Two-Stage Neuromorphic MAP01 Architecture  
**Author**: Antigravity Pair-Programming Agent & Project Team  
**Date**: October 2026  

---

## 1. Executive Summary

This document defines the engineering and research roadmap for bridging the gap between our **100% autonomous privileged-state Spiking Neural Network policy** (`EngineStateSpikingPolicy`) and **raw visual observations** in ViZDoom FreeDoom2 MAP01.

### Current Milestone Status (Stage 1 Complete)
* **Motor & Decision Policy**: The recurrent Leaky Integrate-and-Fire (`snn.Leaky`) policy ([EngineStateSpikingPolicy](file:///e:/Side%20project/src/networks/engine_state_policy.py#L104-L198)) achieves **100% unshielded completion** on MAP01 with **15/15 kills**, fluid navigation (22 stuck steps), and factorized control readouts.
* **Privileged State Dependency**: The current policy reads an 84-dimensional normalized vector directly from ViZDoom's label buffer and coordinate registers (exact enemy bearing, distance, route progress, and position).
* **The Stage 2 Objective**: Replace direct engine state extraction with a **learned Visual Perception Network (Encoder)** that processes raw $4 \times 84 \times 84$ grayscale camera buffers, predicts the structured scene variables, and feeds them into the proven Spiking Policy.

```text
               STAGE 2: Vision Perception                 STAGE 1: Spiking Control
           ┌────────────────────────────────┐         ┌───────────────────────────────┐
Raw Frames │  Visual Perception Encoder     │ Scene   │  EngineStateSpikingPolicy     │
(84x84x4)  │  (ConvNet or Conv-SNN)         │ Vector  │  (Recurrent LIF Spiking SNN)  │
──────────►│  Predicts:                     ├────────►│  Predicts:                    ├─► ViZDoom
           │  - Enemy detection & screen-X  │ (84-dim)│  - Locomotion (Forward/Strafe)│   Buttons
           │  - Estimated bearing & distance│         │  - Turning (Left/Right)       │
           │  - Corridor / heading error    │         │  - Attack & Use               │
           │  - Obstacle / wall proximity   │         └───────────────────────────────┘
           └────────────────────────────────┘                        ▲
                                                                     │
                                                   [ ALREADY VALIDATED & 100% COMPLETE ]
```

---

## 2. Scientific Motivation & Academic Contribution

### 2.1 Why the Legacy End-to-End CRSNN Failed
Earlier attempts to map raw pixels directly to 22 monolithic compound actions through an end-to-end spiking convolutional network failed (~0% completion):
1. **Entangled Credit Assignment**: When an agent hits a wall or misses an enemy, gradients from action loss cannot distinguish between perceptual errors (e.g., failure to detect an Imp) and temporal motor errors (e.g., failing to back away).
2. **Action Fragmentation**: Monolithic actions (such as `forward+turn_left+attack`) have sparse training distributions, severely punishing surrogate-gradient backpropagation through time.
3. **Black-Box Uninterpretability**: There was no visibility into whether the network actually "saw" the monsters or was blindly oscillating.

### 2.2 The Two-Stage Decoupled Advantage
1. **Biological Realism**: Mirrors mammalian neuroanatomy:
   - **Ventral / Dorsal Visual Streams (Visual Cortex V1–V4, MT)**: Extract "what" (enemy identification) and "where" (spatial distance, bearing, corridor direction).
   - **Motor Cortex / Basal Ganglia (Spiking Controller)**: Translates perceived state into muscle actuations and combat reflexes.
2. **Independent Verification**: Perception accuracy ($mAP$, IoU, depth estimation MAE) can be evaluated and debugged independently from navigation and survival metrics.
3. **Direct Thesis Narrative**: Provides clean ablation comparisons:
   - **Oracle Upper Bound**: SNN driven by ground-truth engine state (100% completion).
   - **Vision-Bridged SNN**: SNN driven by predicted visual state (measuring perceptual degradation).
   - **End-to-End Baseline**: Demonstrating why modular visual-motor decoupling is necessary for complex 3D embodied tasks.

---

## 3. Sensory Interface & Feature Decomposition

The 84-dimensional feature vector is decomposed into **Visually Perceived Features** and **Proprioceptive / Telemetry Features**:

### 3.1 Group A: Visually Perceived Features (Predicted by Vision Encoder)
These features must be inferred purely from the $4 \times 84 \times 84$ frame stack:

| Feature Name | Type | Target Range | Visual Task |
| :--- | :--- | :--- | :--- |
| `enemy_visible` | Binary | $\{0, 1\}$ | Object Detection / Presence Classification |
| `enemy_count` | Continuous | $[0, 1]$ (scaled / 8) | Crowd Density Estimation |
| `enemy_distance` | Continuous | $[0, 1]$ (scaled / 1024) | Monocular Depth Estimation |
| `enemy_bearing_sin` | Continuous | $[-1, 1]$ | Target Azimuth Regression |
| `enemy_bearing_cos` | Continuous | $[-1, 1]$ | Target Azimuth Regression |
| `enemy_screen_x` | Continuous | $[0, 1]$ | Horizontal Bounding Box Center |
| `enemy_screen_area` | Continuous | $[0, 1]$ | Apparent Object Scale |
| `enemy_in_firing_cone`| Continuous | $[0, 1]$ (smooth score) | Crosshair Alignment Score |
| `corridor_heading_error_sin` | Continuous | $[-1, 1]$ | Visual Waypoint / Vanishing Point Heading |
| `corridor_heading_error_cos` | Continuous | $[-1, 1]$ | Visual Waypoint / Vanishing Point Heading |
| `obstacle_front_proximity` | Continuous | $[0, 1]$ | Optical Flow / Collision Proximity |
| `use_recommended` | Binary | $\{0, 1\}$ | Switch / Doorway Visual Cue |

### 3.2 Group B: Proprioceptive & Telemetry Features (Directly Available)
These features do not require visual inference because a real-world embodied agent or player always has direct internal sensors or HUD access:
* **Internal Diagnostics**: `health`, `armor`, `ammo_bullets`, `ammo_shells` (read from ViZDoom game variables / HUD counters).
* **Motor Recurrence**: `previous_forward`, `previous_backward`, `previous_turn_left`, `previous_turn_right`, `previous_attack`, `previous_use`.
* **Causal Odometry**: `delta_x`, `delta_y`, `delta_angle_sin`, `delta_angle_cos`, `stuck_fraction` (derived from internal velocity / IMU / movement integration).

---

## 4. Phase-by-Phase Implementation Roadmap

### Phase 0: Ground-Truth Vision Dataset Generation
**Goal**: Build a synchronized, high-throughput offline dataset pairing raw 4-frame visual observations with ground-truth 84-dim engine-state vectors.

* **Files to Add**:
  - `src/generate_vision_dataset.py`
  - `src/vision_dataset.py`
* **Implementation Details**:
  - Run the scripted expert and recovery agent across MAP01 (seeds 0 through 10, plus perturbed starts).
  - At each step, capture:
    1. Grayscale $84 \times 84 \times 4$ observation tensor (`uint8` format to minimize RAM/disk).
    2. Exact 84-dimensional `EngineStateObservation.features` vector.
    3. `validity` mask array.
  - Store as chunked `.npz` or HDF5 files under `data/generated/vision-map01-v1/`.
  - Target dataset size: 50,000–80,000 transitions (~1.2 GB compressed).
* **Exit Gate**:
  - Dataset generator produces reproducible episodes with zero dropped frames.
  - Unit tests verify pixel-to-state alignment.

---

### Phase 1: Visual Perception Architecture Design
**Goal**: Implement the multi-task Visual Perception Encoder network.

* **Files to Add**:
  - `src/networks/vision_encoder.py`
  - `tests/test_vision_encoder.py`
* **Model Architecture**:
  - **Backbone**: 4-layer Convolutional Feature Extractor with Group Normalization and GELU activations:
    - Conv1: $4 \to 32$, kernel $8 \times 8$, stride 4 (downsamples $84 \times 84 \to 20 \times 20$)
    - Conv2: $32 \to 64$, kernel $4 \times 4$, stride 2 (downsamples $20 \times 20 \to 9 \times 9$)
    - Conv3: $64 \to 64$, kernel $3 \times 3$, stride 1 ($9 \times 9 \to 7 \times 7$)
    - Flatten $\to$ Linear projection to 256-dim latent visual embedding $z_t$.
  - **Factorized Perception Heads**:
    1. **Combat Head** $\to$ outputs `[enemy_visible, enemy_screen_x, enemy_screen_area, enemy_distance, enemy_bearing_sin, enemy_bearing_cos, enemy_in_firing_cone]`.
    2. **Navigation / Corridor Head** $\to$ outputs `[corridor_heading_error_sin, corridor_heading_error_cos, route_distance]`.
    3. **Interaction Head** $\to$ outputs `[use_recommended, obstacle_proximity]`.
* **Neuromorphic Variant (`SpikingVisionEncoder`)**:
  - A parallel implementation replacing Conv2d with `snnTorch` Conv-LIF layers for an all-spiking ablation.
* **Exit Gate**:
  - Forward and backward unit tests pass with clean shapes and zero NaN gradients.

---

### Phase 2: Offline Multi-Task Training & Calibration
**Goal**: Train the vision encoder to high fidelity using multi-task loss.

* **Files to Add**:
  - `src/train_vision_encoder.py`
* **Loss Function Formulation**:
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{vis}} \mathcal{L}_{\text{BCE}}(\hat{v}, v) + \lambda_{\text{box}} \mathcal{L}_{\text{smooth\_L1}}(\hat{x}_{\text{scr}}, x_{\text{scr}}) + \lambda_{\text{ang}} \mathcal{L}_{\text{cosine}}(\hat{\theta}_{\text{enemy}}, \theta_{\text{enemy}}) + \lambda_{\text{nav}} \mathcal{L}_{\text{cosine}}(\hat{\theta}_{\text{corr}}, \theta_{\text{corr}}) + \lambda_{\text{dist}} \mathcal{L}_{\text{MSE}}(\hat{d}, d)$$
* **Training Details**:
  - AdamW optimizer, cosine annealing learning rate ($1 \times 10^{-3} \to 1 \times 10^{-5}$).
  - Mixed precision (PyTorch AMP).
  - Validation split: 80% train, 20% held-out unseen seed episodes.
* **Exit Gate**:
  - Enemy detection F1-score $\ge 92\%$.
  - Screen-X mean absolute error $\le 0.04$ (within 6 pixels on a 160-wide screen).
  - Corridor angle error $\le 10^\circ$.

---

### Phase 3: Closed-Loop Vision-Bridged Evaluation
**Goal**: Connect the trained Visual Perception Encoder to the frozen Spiking Policy and evaluate autonomous MAP01 completion without reading engine labels.

* **Files to Add**:
  - `src/evaluate_vision_spiking_policy.py`
* **Inference Pipeline**:
  ```python
  # Closed-loop step
  frames = env.get_screen_stack()                   # (4, 84, 84)
  visual_features = vision_encoder(frames)          # Predicted visual fields
  telemetry = env.get_proprioceptive_state()        # Health, ammo, previous action
  full_state = compose_engine_state(visual_features, telemetry)

  # Spiking Policy inference (Recurrent LIF)
  action, spk_rec, mem_rec = spiking_policy(full_state, spk_rec, mem_rec)
  env.step(action)
  ```
* **Exit Gate**:
  - Zero access to `labels_buffer` or engine coordinate variables during decision loops.
  - Autonomous unshielded completion of MAP01 on standard seeds.

---

### Phase 4: Perceptual Noise Injection & Policy Fine-Tuning
**Goal**: Ensure the Spiking Policy is robust to real-world visual perception errors (flickering detections, minor depth inaccuracies).

* **Approach**:
  - Evaluate policy sensitivity by injecting Gaussian jitter into predicted enemy bearings and distances.
  - Fine-tune `EngineStateSpikingPolicy` with **Perception Dropout** (randomly masking out visual features during 10% of training steps) so the LIF membrane integrates temporal history across temporary visual dropouts.
* **Exit Gate**:
  - Policy maintains $\ge 80\%$ completion even when perception noise is injected.

---

### Phase 5: Neuromorphic Benchmarking & Energy Analysis
**Goal**: Quantify the energy efficiency and computational advantages of the Spiking Neural Network.

* **Metrics to Measure & Report**:
  1. **Mean Firing Rate**: Average spikes per neuron per decision frame across layers.
  2. **Synaptic Operations (SynOps)**: Count of sparse accumulator operations ($AC$) vs. dense multiply-accumulate ($MAC$) operations.
  3. **Theoretical Energy Consumption**:
     $$E_{\text{SNN}} = \text{SynOps} \times E_{\text{AC}} \quad \text{vs.} \quad E_{\text{ANN}} = \text{FLOPs} \times E_{\text{MAC}}$$
     (Using standard neuromorphic CMOS baseline: $E_{\text{AC}} \approx 0.9\,\text{pJ}$, $E_{\text{MAC}} \approx 4.6\,\text{pJ}$ at 45nm).
  4. **Latency Profiling**: Frame-to-decision time on CPU, GPU, and simulated neuromorphic hardware.
* **Exit Gate**:
  - Publication-ready tables and graphs documenting energy savings and completion benchmarks.

---

## 5. Risk Assessment & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Visual Ambiguity in Dark Corridors** | High | The 4-frame temporal stack provides optical flow motion cues even in low-contrast sectors. |
| **Error Accumulation (Perception Drift)** | High | The proprioceptive telemetry (stuck fraction, delta motion) remains uncorrupted, allowing the existing recovery mechanism to trigger if visual drift causes a wall collision. |
| **Model Size / Latency Overhead** | Medium | Keep the Conv backbone lightweight ($< 1.5\,\text{M}$ parameters) to ensure $> 60\,\text{FPS}$ real-time inference on edge devices. |

---

## 6. Target Academic Deliverables

Upon completion of this plan, the repository will contain:
1. **The First Autonomous SNN for ViZDoom MAP01**: A biologically grounded, spiking neural network agent navigating and clearing a full 3D level from visual inputs.
2. **Comprehensive Empirical Study**:
   - Oracle (Engine State) vs. Vision-Bridged vs. End-to-End CRSNN.
   - Comparative analysis of ANN (MLP/GRU) vs. SNN (LIF Spiking).
3. **Reproducible Codebase & Exported Models**: Checkpoints, automated evaluation scripts, and diagnostic JSON reports ready for publication and peer review.

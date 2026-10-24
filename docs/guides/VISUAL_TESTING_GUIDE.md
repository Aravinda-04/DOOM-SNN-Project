# ViZDoom Visual Evaluation & Generalization Testing Guide

This guide contains all terminal commands to watch the autonomous neural network policies navigate, aim, and clear levels live in the ViZDoom graphical window or export recorded gameplay videos with real-time HUD overlays.

All commands should be executed from the project root (`e:\Side project`) in PowerShell or your terminal.

---

## 0. Prerequisite: Asset Setup (Run Once After Cloning)

Because `.wad` game files are large binaries (~144 MB total) excluded from Git, run the automated setup script once before testing:

```powershell
python setup_assets.py
```
This script will:
1. Auto-download and unpack the official open-source `freedoom2.wad` (if missing).
2. Procedurally compile the spatial jitter test WADs (`s10`, `s20`, `s30`).
3. Verify that your local ViZDoom installation initializes FreeDoom2 MAP01 without error.

---

## 1. Closed-Loop Vision-Bridged SNN (Raw Pixels, Zero Cheat Access)

The **Vision-Bridged SNN** completely removes privileged cheat engine labels (`labels = ()`). A 12-channel RGB convolutional perception encoder predicts enemy screen coordinates and distances directly from raw camera pixels, driving a frozen recurrent LIF Spiking Neural Network policy.

### A. Watch the Vision Agent Play Live (Interactive Window)
Opens a graphical desktop window where you can watch the agent aim, fight, and clear MAP01 live:

```powershell
# Seed 3 (Verified Full Level Clear: 525 steps, 7 kills, 100% route progress)
python src/evaluate_vision_spiking_policy.py `
  --seed 3 `
  --render `
  --render-delay 0.02

# Seed 9 (Verified Full Level Clear: 497 steps, 9 kills, 100% route progress)
python src/evaluate_vision_spiking_policy.py `
  --seed 9 `
  --render `
  --render-delay 0.02
```

### B. Record Video with Live HUD & Bounding Boxes
Runs the agent and saves a full MP4 recording with real-time telemetry (HP, ammo, action state, kills) and enemy bounding box tracking:

```powershell
# Record Seed 3 full clear
python src/evaluate_vision_spiking_policy.py `
  --seed 3 `
  --record-video artifacts/vision_level_clear_seed3.mp4

# Record Seed 9 full clear
python src/evaluate_vision_spiking_policy.py `
  --seed 9 `
  --record-video artifacts/vision_level_clear_seed9.mp4
```

### C. Understanding the Live Bounding Boxes & HUD Overlays

The bounding boxes in the recorded MP4 video are **generated directly by the neural network**, not by engine debug cheats:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP: 090 | ROUTE: 25.0% | HP: 94 | AMMO: 41 | KILLS: 1               │
│ ACTION: FORWARD+NONE+ATTACK                                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                      ┌───────────────┐                                 │
│                      │ HOSTILE 340u  │                                 │
│                      │               │                                 │
│                      │   [ENEMY]     │  <-- RED = Firing / Attacking   │
│                      │               │      GREEN = Tracking Target    │
│                      └───────────────┘                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

1. **How the Boxes Are Created**:
   - The Vision ConvNet (`VisionPerceptionEncoder`) takes the raw 12-channel camera stack and regresses `enemy_screen_x` (horizontal center), `enemy_screen_area` (bounding area), and `enemy_distance`.
   - The HUD reconstructs the 2D box width from $\sqrt{\text{area}}$ and scales height to Doom's humanoid sprite ratio ($1.5 \times \text{width}$).
2. **Color Coding**:
   - **GREEN Bounding Box**: Hostile detected and tracked in crosshairs.
   - **RED Bounding Box**: Hostile aligned within the firing cone ($\le 15^\circ$) and the SNN policy is actively executing an `ATTACK` command.
3. **Corpse Rejection**:
   - The system tracks kill locations and suppresses detections near confirmed corpses, preventing false-positive boxes around dead monsters on the floor.

---

## 2. Privileged Engine-State Baseline Runs (Cheats Enabled)

These runs use ground-truth engine coordinates and object labels (`src/evaluate_engine_state_policy.py`) to demonstrate the upper-bound baseline performance.

### A. High-Completion Run (Seed 0 - Flawless 616 Steps)
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --render `
  --render-delay 0.02 `
  --seed 0
```

### B. Fast Route Clear (Seed 3 - Aggressive 504 Steps)
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --render `
  --render-delay 0.02 `
  --seed 3
```

### C. Test Other Verified Winning Seeds
You can test any of the verified passing seeds by changing `--seed <number>`:
- `--seed 0`
- `--seed 1`
- `--seed 3`
- `--seed 4`
- `--seed 42`
- `--seed 100`
- `--seed 444`

---

## 3. Randomized Jittered Enemy WADs (Spatial Generalization)

These scenarios use custom-compiled WAD files where monster $(X, Y)$ spawn points and angles are physically perturbed across the map.

### A. Level 01 with ±10% Coordinate Jitter (`level01_jitter_s10.cfg`)
Monsters are shifted by up to ±10% from their canonical ambush spots.
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --config-file scenarios/generated/level01_jitter_s10.cfg `
  --render `
  --render-delay 0.02 `
  --seed 0
```

### B. Level 01 with ±20% Coordinate Jitter (`level01_jitter_s20.cfg`)
Significant spatial displacement across hallways and courtyard platforms.
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --config-file scenarios/generated/level01_jitter_s20.cfg `
  --render `
  --render-delay 0.02 `
  --seed 2
```

### C. Level 01 with ±30% Coordinate Jitter (`level01_jitter_s30.cfg`)
Extreme monster layout displacement requiring active visual/range reticle adaptation.
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --config-file scenarios/generated/level01_jitter_s30.cfg `
  --render `
  --render-delay 0.02 `
  --seed 0
```

### D. General Jitter Scenario (`level01_jittered.cfg`)
The original randomized scenario setup.
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --config-file scenarios/generated/level01_jittered.cfg `
  --render `
  --render-delay 0.02 `
  --seed 0
```

---

## 4. Altered-Start Perturbation Recovery Tests

Tests whether the agent can recover when forced off-track or rotated away from the route at the very first step.

### A. Backward Displacement (Forced 6 Steps Backwards)
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --start-perturbation backward `
  --start-perturbation-steps 6 `
  --render `
  --render-delay 0.02 `
  --seed 0
```

### B. Right Angular Shift (Forced 8 Steps Turn Right)
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --start-perturbation right `
  --start-perturbation-steps 8 `
  --render `
  --render-delay 0.02 `
  --seed 0
```

### C. Left Angular Shift (Forced 8 Steps Turn Left)
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --start-perturbation left `
  --start-perturbation-steps 8 `
  --render `
  --render-delay 0.02 `
  --seed 0
```

---

## 5. Comparing SNN vs. MLP Baseline Visually

### Watch Spiking Neural Network (SNN) Policy:
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --render `
  --render-delay 0.02 `
  --seed 0
```

### Watch Multilayer Perceptron (MLP) Baseline Policy:
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-reactive-mlp-recovery.pth `
  --render `
  --render-delay 0.02 `
  --seed 0
```

---

## 6. Control Options & Cheatsheet

### Vision Evaluation Parameters (`evaluate_vision_spiking_policy.py`)
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `--render` | *(Flag)* | Opens interactive graphical window. |
| `--render-delay` | `0.02` | Delays each step by 20ms for smooth real-time viewing (~35 FPS). |
| `--record-video` | `None` | Path to export an MP4 video with real-time HUD and bounding boxes. |
| `--seed` | `3` | ViZDoom PRNG seed (`3` and `9` are verified 100% clears). |
| `--enemy-threshold`| `0.50` | Detection confidence threshold to engage enemies in combat. |
| `--vision-checkpoint` | `best_vision_encoder_rgb.pth` | Path to 12-channel RGB ConvNet perception weights. |
| `--snn-checkpoint` | `best-engine-state-spiking-snn-v3-robust.pth` | Path to frozen recurrent LIF SNN policy weights. |

### Engine-State Evaluation Parameters (`evaluate_engine_state_policy.py`)
| Parameter | Recommended Value | Description |
| :--- | :--- | :--- |
| `--render` | *(Flag)* | Opens the ViZDoom graphical window to watch live. |
| `--render-delay` | `0.02` | Delay between decision steps. |
| `--seed` | `0` (or `1, 3, 4, 42, 100, 444`) | Engine random seed. |
| `--config-file` | `scenarios/generated/level01_jitter_s10.cfg` | Custom scenario configuration file (e.g. for jittered WADs). |
| `--start-perturbation` | `none`, `left`, `right`, `backward` | Injects an initial displacement before neural policy assumes control. |
| `--start-perturbation-steps` | `6` to `8` | Number of forced perturbation steps to apply at start. |

---

## 7. Full Automated Benchmark Suite (Headless)

To evaluate the policy across **all 10 seeds**, **all 3 jittered WADs**, and **all 3 perturbation modes** automatically in batch mode:

```powershell
python src/test_robustness_suite.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --report-file reports/my_robustness_benchmark.md
```

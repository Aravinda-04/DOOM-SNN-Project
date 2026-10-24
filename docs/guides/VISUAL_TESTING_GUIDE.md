# ViZDoom Visual Evaluation & Generalization Testing Guide

This guide contains all terminal commands to watch the autonomous neural network policies navigate, aim, and clear levels live in the ViZDoom graphical window.

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

## 1. Quick Start: Standard MAP01 Visual Runs

In standard MAP01, monster placements are fixed according to original level geometry.

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
You can test any of the 7 verified passing seeds by changing `--seed <number>`:
- `--seed 0`
- `--seed 1`
- `--seed 3`
- `--seed 4`
- `--seed 42`
- `--seed 100`
- `--seed 444`

---

## 2. Randomized Jittered Enemy WADs (Spatial Generalization)

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

## 3. Altered-Start Perturbation Recovery Tests

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

## 4. Comparing SNN vs. MLP Baseline Visually

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

## 5. Control Options & Cheatsheet

| Parameter | Recommended Value | Description |
| :--- | :--- | :--- |
| `--render` | *(Flag)* | Opens the ViZDoom graphical window to watch live. |
| `--render-delay` | `0.02` | Delays each step by 20ms for smooth real-time viewing (~35 FPS). Use `0.01` for 2x speed, or omit for uncapped speed. |
| `--seed` | `0` (or `1, 3, 4, 42, 100, 444`) | Engine random seed (affects damage rolls, monster reaction tics, and spread). |
| `--config-file` | `scenarios/generated/level01_jitter_s10.cfg` | Custom scenario configuration file (e.g. for jittered WADs). |
| `--start-perturbation` | `none`, `left`, `right`, `backward` | Injects an initial displacement before neural policy assumes control. |
| `--start-perturbation-steps` | `6` to `8` | Number of forced perturbation steps to apply at start. |
| `--episode-timeout` | `6000` | Game tics before timing out (6000 tics = 1500 decision steps). |
| `--control-shield` | *(Flag, default off)* | Diagnostic only: enables scripted fallback if confidence drops below threshold. |

---

## 6. Full Automated Benchmark Suite (Headless)

To evaluate the policy across **all 10 seeds**, **all 3 jittered WADs**, and **all 3 perturbation modes** automatically in batch mode:

```powershell
python src/test_robustness_suite.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --report-file reports/my_robustness_benchmark.md
```

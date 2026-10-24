# ViZDoom & SNN Scenario Guide

This document provides a comprehensive overview of all evaluation and playable scenarios in this repository, their objectives, architecture, and exact commands to run them.

All autonomous evaluations use the **frozen Spiking Neural Network (SNN) policy** (`exports/best-engine-state-spiking-snn-v3-robust.pth`) without modifying its weights or its standardized 84-dimensional feature vector contract.

---

## Quick Reference Summary

| Scenario | Type | Goal / Mechanic | 1X Speed | Run Command |
|---|---|---|:---:|---|
| **MAP01** | Full Level | Autonomous navigation & exit trigger | ✅ | `python scratch/watch_map01.py` |
| **MAP02** | Full Level | Interactive human play with mouse/keys | ✅ | `python scratch/play_map02.py` |
| **Deadly Corridor** | Gauntlet | Sprint 1312 units, eliminate snipers, retrieve GreenArmor | ✅ | `python scenarios/deadly_corridor/run.py` |
| **Progressive Deathmatch**| Wave Arena | Multi-wave arena survival, ACS ammo-on-kill | ✅ | `python scenarios/progressive_deathmatch/run.py` |

---

## 1. FreeDoom2 MAP01 ("Outer Earth")

### Overview
MAP01 is the opening level of FreeDoom2. It features indoor corridors, low-tier hitscanners (Zombiemen) and Imps, standard push doors, and an exit switch chamber.

* **Pathfinding**: 100% autonomous WAD NavMesh + A* graph search (`src/wad_navmesh.py`, `src/wad_pathfinder.py`). No human demonstrations or hardcoded waypoints.
* **Status**: Solved in both Ground Truth (privileged engine state) and Vision (RGB CNN encoder) modes.

### How to Run

```bash
# 1. Ground Truth mode (live window, generalized A* pathfinder)
python scratch/watch_map01.py

# 2. Tactical mode (auto-detours to pickup Shotgun and Armor along the route)
python scratch/watch_map01.py --tactical

# 3. Vision mode (uses CNN encoder on RGB camera feed)
python scratch/watch_map01.py --vision

# 4. Record playthrough video to MP4
python scratch/watch_map01.py --record outputs/map01.mp4
```

---

## 2. FreeDoom2 MAP02 ("The Meat Grinder")

### Overview
A non-linear hub level featuring elevated ShotgunGuy snipers, charging Demons/Spectres, switch-lowered steel barrier gates (Gate 150), motorized elevators (Sector 159), teleporters, and all 3 keycards.

* **Door 156 Mechanic**: Push-doors in Doom reverse direction if `USE` is repeatedly held while opening. The interaction layer pulses `USE` only when closed (`ceiling == 96`) and inhibits input while raising.
* **Interactive Human Mode**: High-resolution spectator mode allowing you to play the map directly using keyboard and mouse.

### How to Run

```bash
# Play MAP02 interactively with mouse & keyboard (800x600, 35 FPS)
python scratch/play_map02.py

# Custom difficulty (1=Easy, 3=Normal, 4=Hard)
python scratch/play_map02.py --skill 2

# Higher resolution
python scratch/play_map02.py --res 1024x768
```

#### Controls
* **Movement**: `W`, `A`, `S`, `D` or Arrow Keys
* **Look / Turn**: Mouse or Left/Right Arrow Keys
* **Fire / Attack**: Left Mouse Button or Left `Ctrl`
* **Open Door / Flip Switch**: `Spacebar` or `E`
* **Weapons**: Keys `1` through `7`
* **Quit**: `Esc`

---

## 3. Deadly Corridor

### Scenario Folder: `scenarios/deadly_corridor/`
```
scenarios/deadly_corridor/
├── deadly_corridor.cfg    # Scenario configuration (Skill 1, 1500 tics)
├── deadly_corridor.wad    # Map WAD file
└── run.py                 # Self-contained SNN runner (1X real-time speed)
```

### Overview
A benchmark scenario where the agent spawns at one end of a long hallway $(0, 0)$ lined with alcoves containing hostile Zombiemen and ShotgunGuys. 

* **Objective**: Advance down the 1312-unit hallway, eliminate threats, and retrieve the **GreenArmor** vest at $(1312, 0)$.
* **Adapter Layer**: Maps the SNN's 8 factorized actions $\to$ `deadly_corridor`'s 7 buttons (`MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK`, `MOVE_FORWARD`, `MOVE_BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`).
* **Observation**: Synthesizes the 84-dimensional observation vector with real-time target distance and bearing to $(1312, 0)$.
* **Pacing**: Precision-locked to **1X real-time speed** (35 FPS / 114.3 ms per 4-tic decision step).

### How to Run

```bash
# Standard 1X visual evaluation
python scenarios/deadly_corridor/run.py

# Higher difficulty
python scenarios/deadly_corridor/run.py --skill 3

# Record playthrough video
python scenarios/deadly_corridor/run.py --record outputs/deadly_corridor.mp4
```

---

## 4. Progressive Deathmatch

### Scenario Folder: `scenarios/progressive_deathmatch/`
```
scenarios/progressive_deathmatch/
├── progressive_deathmatch.cfg  # Scenario configuration (Skill 3, 4200 tics)
├── progressive_deathmatch.wad  # Map WAD file (UDMF format with ACS scripts)
├── progressive_deathmatch.dbs  # Doom Builder settings configuration
└── run.py                      # Self-contained SNN runner (1X real-time speed)
```

### Overview
A custom wave-based survival arena map built in GZDoom UDMF format.

* **Objective**: Survive continuous waves of hostile Imps spawning from 12 perimeter spawn spots. The episode lasts up to 4200 tics (2 minutes).
* **Game Logic (ACS Script)**: The built-in ACS script rewards the player with **+2 pistol ammo for every kill**, making aggression and accuracy vital for sustained survival.
* **Combat & Aiming**: 
  * Actively monitors the labels buffer for visible hostiles.
  * Dynamically computes the signed angular error $\Delta\theta = (\theta_{\text{enemy}} - \theta_{\text{player}})$ to swivel crosshairs onto target.
  * Fires when within the $30^\circ$ firing cone; backpedals if hostiles get within 90 units.
  * Sweeps and patrols the arena when no hostiles are currently in line of sight.
* **Adapter Layer**: Bridges the SNN's factorized actions into the 9-button layout (`ATTACK`, `SPEED`, `STRAFE`, `MOVE_RIGHT`, `MOVE_LEFT`, `MOVE_BACKWARD`, `MOVE_FORWARD`, `TURN_RIGHT`, `TURN_LEFT`).
* **Pacing**: Precision-locked to **1X real-time speed** (35 FPS).

### How to Run

```bash
# Standard 1X visual wave-survival evaluation
python scenarios/progressive_deathmatch/run.py

# Lower difficulty for longer survival
python scenarios/progressive_deathmatch/run.py --skill 1

# Record playthrough video
python scenarios/progressive_deathmatch/run.py --record outputs/deathmatch.mp4
```

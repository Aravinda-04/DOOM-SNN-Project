# Autonomous Spiking Neural Network (SNN) Agent for DOOM MAP01

An autonomous neuromorphic agent powered by a recurrent Spiking Neural Network (SNN) that navigates, combats enemies, interacts with switches/doors, and completes **FreeDoom2 MAP01** via the ViZDoom simulator.

The agent operates **without external control shields**, driving motor controls end-to-end through Leaky Integrate-and-Fire (LIF) neurons (`snnTorch`), factorized action spaces, and opportunistic tactical health seeking.

---

## Benchmark Highlights

Evaluated across the standard 10-seed multi-seed benchmark (`best-engine-state-spiking-snn-v2.pth`):

| Evaluation Suite | Completion Rate | Key Metrics | Status |
| :--- | :---: | :---: | :---: |
| **Standard MAP01 (10 Seeds)** | **80.0% (8/10 seeds)** | 13/15 kills, 512–649 steps (~15–18s) | **PASSED** |
| **Spatial Jitter WADs (±30 units)** | **66.7% (2/3 seeds)** | Procedural monster relocations | **PASSED** |
| **Altered-Start Perturbations** | **66.7% (2/3 runs)** | 8-step left/right forced displacement | **PASSED** |
| **Doorway Wall Wedging** | **0 steps stuck** | Clearance refined at Waypoint 40 | **PASSED** |
| **Unit Test Coverage** | **100% (29/29 tests)** | Deterministic policy & tactical checks | **PASSED** |

* **Verified Passing Seeds**: Seed 0, Seed 1, Seed 3, Seed 4, Seed 5, Seed 42, Seed 100, Seed 444.
* **Fastest Clears**: Seed 3 (520 steps, 13 kills), Seed 5 (512 steps, 13 kills).

---

## Quick Start (3 Steps)

### 1. Install Dependencies
Ensure Python 3.10+ (tested up to Python 3.13) is installed:
```powershell
python -m pip install -r requirements.txt
```

### 2. Run Automated Asset Setup
Download the official open-source `freedoom2.wad` game file and procedurally compile test WADs in one command:
```powershell
python setup_assets.py
```
This script automatically:
* Downloads and verifies `freedoom2.wad` (~29 MB) if missing.
* Procedurally generates spatial enemy jitter WADs (`s10`, `s20`, `s30`).
* Runs a smoke test confirming ViZDoom initializes MAP01 properly.

### 3. Watch the Agent Play Live
Launch a visual interactive window (35 FPS):
```powershell
# Seed 3 (520 steps - fast clear)
python src/evaluate_engine_state_policy.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth --seed 3 --render --render-delay 0.02

# Seed 5 (512 steps - aggressive clear)
python src/evaluate_engine_state_policy.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth --seed 5 --render --render-delay 0.02

# Seed 0 (649 steps - high-exploration run)
python src/evaluate_engine_state_policy.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth --seed 0 --render --render-delay 0.02
```

---

## Testing & Benchmarks

### Full Automated Robustness Suite (Headless)
Run the automated benchmark across all 10 standard seeds, jittered WADs, and altered-start perturbations:
```powershell
python src/test_robustness_suite.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth
```
This produces a structured report containing seed-by-seed completion status, kills, shot counts, attack precision, and steps.

### Spatial Jitter WAD Evaluation
Test whether the agent can adapt when enemies are procedurally displaced from their original map spawns:
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --config-file scenarios/generated/level01_jitter_s30.cfg `
  --seed 0 `
  --render `
  --render-delay 0.02
```

### Run Unit Tests
Run the deterministic unit test suite covering observation builders, factorized actions, health arbitrators, and route contexts:
```powershell
pytest -q
```

---

## Technical Architecture

```text
 ┌────────────────────────────────────────────────────────┐
 │                   ViZDoom MAP01 State                  │
 │   - Player Position (X, Y), Heading Angle, Health, Ammo│
 │   - Label Buffer (Enemies, Medikits, Stimpacks)        │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │        EngineStateBuilder + HealthTacticsArbitrator    │
 │   - Normalized Route Deltas & Bearings                 │
 │   - Dynamic Subgoal Blending (opportunistic healing)   │
 │   - Threat Distance & Forward-Cone Safety Filters      │
 └───────────────────────────┬────────────────────────────┘
                             │ (Features + Validity Flags)
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │        EngineStateSpikingPolicy (LIF Neurons)          │
 │   - Input: Linear Projection -> snn.Leaky Layer        │
 │   - Surrogate Gradient: Fast Sigmoid                   │
 │   - Temporal Spike Integration across time steps       │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │               Factorized Action Decoder                │
 │   - Locomotion: None, Forward, Backward, Strafe        │
 │   - Turning: None, Left, Right                         │
 │   - Combat: Attack (trigger on target within 15°)      │
 │   - Interaction: Use (door & switch triggers)          │
 └────────────────────────────────────────────────────────┘
```

### Key Engineering Features
1. **Recurrent Spiking Neurons**: Implemented with `snnTorch` Leaky Integrate-and-Fire (`snn.Leaky`) units that maintain membrane potential state across decision cycles.
2. **Factorized Motor Control**: Rather than combinatorial discrete actions, independent policy heads handle simultaneous locomotion, steering, shooting, and switch activation.
3. **Opportunistic Health Seeking**: When player health falls below 50 HP, the arbitrator detects nearby Medikits/Stimpacks, verifies safety (outside point-blank combat and acid pits), and temporarily routes to the item before seamlessly rejoining the main path.
4. **Doorway Geometry Clearance & Anti-Oscillation**: Waypoint 40 is offset away from the north wall jamb at `(1760, -772)` into the hallway center at `(1840, -830)` and `(1775, -835)`, reducing stuck events to 0.

---

## Repository Structure

```text
├── exports/                       # Verified production checkpoints
│   ├── best-engine-state-spiking-snn-v2.pth   # Active 80% pass rate SNN checkpoint
│   └── best-engine-state-spiking-snn.pth      # SNN v1 checkpoint
├── src/                           # Core source modules
│   ├── action_space.py            # ViZDoom button mapping catalogues
│   ├── doom_level_env.py          # ViZDoom environment wrapper
│   ├── engine_state_config.py     # Action enums & feature contract definitions
│   ├── engine_state_observation.py# Observation builder & enemy/health item detectors
│   ├── evaluate_engine_state_policy.py # Evaluation & interactive render engine
│   ├── health_tactics.py          # Opportunistic health detour arbitrator
│   ├── map01_route.py             # Route waypoints & context calculations
│   ├── networks/
│   │   └── engine_state_policy.py # SNN & MLP architecture definitions
│   ├── test_robustness_suite.py   # Multi-seed & perturbation benchmark suite
│   └── train_engine_state_policy.py # SNN imitation & recurrent training pipeline
├── scenarios/                     # ViZDoom level configuration files
│   ├── level01.cfg                # Standard FreeDoom2 MAP01 configuration
│   └── generated/                 # Procedural jitter scenario configurations
├── tests/                         # Pytest test suite (100% green)
├── setup_assets.py                # Automated game asset downloader & WAD builder
├── VISUAL_TESTING_GUIDE.md        # Comprehensive visual evaluation manual
├── HEALTH_SEEKING_IMPLEMENTATION_PLAN.md # Tactical survival design memo
└── VISION_PERCEPTION_BRIDGE_IMPLEMENTATION_PLAN.md # Stage 2 vision bridge roadmap
```

---

## Research Roadmap

* **Step 1: Tactical Health Seeking & Doorway Clearance**  
  *Status*: **Completed** (Achieved **80.0% completion rate**, up from 60%).
* **Step 2: Vision Perception Bridge**  
  *Status*: **In Progress** ([Implementation Plan](VISION_PERCEPTION_BRIDGE_IMPLEMENTATION_PLAN.md))  
  *Goal*: Replace privileged coordinate state with a CNN visual frame perception encoder to identify monsters and doors directly from raw pixels.
* **Step 3: Generalized Navigation (MAP02+)**  
  *Status*: Planned  
  *Goal*: Transition from static MAP01 waypoints to dynamic BSP / NavMesh A* pathfinding.

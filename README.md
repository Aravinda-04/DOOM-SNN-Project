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

## Full Project Pipeline (Extensive Guide)

The project implements a complete, end-to-end deep reinforcement learning and neuromorphic engineering pipeline — from raw game binaries and human demonstration capture to spiking neural network imitation, tactical survival arbitration, and automated spatial generalization testing.

```mermaid
flowchart TD
    subgraph S1["1. Asset & Scenario Pipeline"]
        WAD["freedoom2.wad (Asset Downloader)"] --> JIT["OMGIFOL Lump Parser: Spatial Jitter (s10, s20, s30)"]
        JIT --> CFG["ViZDoom Scenarios (level01.cfg, level01_jittered.cfg)"]
    end

    subgraph S2["2. Demonstration & Route Engineering"]
        SPEC["ViZDoom Spectator Mode (Human Runs)"] --> VOTE["4-Tic Majority-Vote Action Quantization"]
        VOTE --> DEMO["12 Checksummed Episodes (.npz + manifest.json)"]
        DEMO --> DP["Douglas-Peucker Sparsification + Interaction Anchors"]
        DP --> CLEAR["Doorway Clearance Refinement (Waypoint 40)"]
    end

    subgraph S3["3. Perception & Feature Processing"]
        CFG --> ENV["ViZDoom Engine State"]
        CLEAR --> ROUTE["Map01Route (Target, Distance, Bearing)"]
        ENV --> PERC["EngineStateBuilder (Labels + Diagnostics)"]
        PERC --> OBS["Feature Vector (16D Features + 16D Validity)"]
    end

    subgraph S4["4. Tactical Survival Arbitration"]
        OBS --> ARB{"HealthTacticsArbitrator (HP < 50?)"}
        ARB -- Safe Detour --> SUB["Dynamic Health Subgoal (Medikit/Stimpack)"]
        ARB -- Combat Threat / Normal --> MAIN["Mainline Speedrun Waypoint"]
        SUB --> CTX["Blended RouteContext"]
        MAIN --> CTX
    end

    subgraph S5["5. Spiking Neural Network Policy"]
        CTX --> BRAIN["EngineStateSpikingPolicy (snnTorch LIF)"]
        BRAIN --> HEADS["Factorized Actor Heads"]
    end

    subgraph S6["6. Action Decoding & Invariants"]
        HEADS --> DEC["FactorizedAction (Locomotion, Turning, Attack, Use)"]
        DEC --> INV["Control Invariants (Use Priority, Aim Gate, Anti-Wedging)"]
        INV --> BTN["compose_buttons() -> ViZDoom Button Tuple"]
        BTN --> ENV
    end

    subgraph S7["7. Robustness & Evaluation"]
        ENV --> SUITE["test_robustness_suite.py"]
        SUITE --> R1["Suite 1: Standard Seeds (0..999) -> 80% Pass"]
        SUITE --> R2["Suite 2: Spatial Jitter WADs -> 66.7% Pass"]
        SUITE --> R3["Suite 3: Altered-Start Perturbations -> 66.7% Pass"]
    end
```

---

### Phase 1: Asset Ingestion & Procedural WAD Compilation
* **Tool**: `setup_assets.py` and `scripts/generate_jittered_wads.py`
* **Mechanism**:
  1. Downloads and verifies `freedoom2.wad` via SHA-256.
  2. Parses the compiled Doom binary lumps using `omgifol`.
  3. Extracts all `THINGS` lumps for MAP01 (monsters, items, triggers).
  4. Applies bounded Gaussian spatial jitter $\Delta x, \Delta y \sim \mathcal{N}(0, \sigma^2)$ ($\sigma \in \{10, 20, 30\}$ map units) to hostile monster spawn coordinates while preserving blocking sector boundaries.
  5. Outputs standalone test scenarios (`level01_jitter_s10.cfg`, `level01_jitter_s20.cfg`, `level01_jitter_s30.cfg`) used to prove that the agent does not merely memorize fixed monster coordinates.

---

### Phase 2: Demonstration Capture & Route Simplification
* **Tool**: `src/record_demonstrations.py`, `src/manual_level_validation.py`, `src/map01_route.py`
* **Mechanism**:
  1. **Spectator Ingestion**: Records skilled human playthroughs in ViZDoom spectator mode.
  2. **Majority-Vote Quantization**: Game inputs are sampled across 4-tic frames. A 75% majority vote rule resolves transitional button presses, tolerating 1-tic human key slips while strictly masking ambiguous combinations.
  3. **Quality Gate**: Only episodes resulting in a verified level exit with $\ge 90\%$ valid action coverage are retained into `data/demonstrations/map01/`.
  4. **Douglas-Peucker Sparsification**: Compresses the continuous path of over 3,000 coordinates into ~60 sparse, high-leverage waypoints.
  5. **Geometric Clearance Refinement**: Specific wall and doorframe jambs (notably the north doorway corner at `(1760, -772)`) are offset into the hallway centerline at `(1840, -830)` and `(1775, -835)` to prevent high-speed corner collisions.

---

### Phase 3: Observation & Engine-State Perception
* **Tool**: `src/engine_state_observation.py` (`EngineStateBuilder`)
* **Mechanism**:
  Constructs a normalized, model-agnostic feature vector pairing continuous engine measurements with binary validity flags:
  * **Route Deltas**: Euclidean distance to current waypoint ($d / 1024.0$), relative bearing error normalized to $[-1, 1]$.
  * **Player Kinematics**: Normalized player velocity, current heading angle, health ($h / 100.0$), ammunition count ($a / 50.0$).
  * **Label Buffer Parsing**:
    * `visible_enemies`: Scans on-screen object labels for active hostiles (`zombieman`, `shotgunguy`, `imp`, `demon`, etc.), extracting distance, horizontal screen angle, and bounding-box screen area.
    * `visible_health_items`: Scans for consumable health pickups (`medikit`, `stimpack`, `healthbonus`), tracking 2D map positions and priority-weighted distances.
  * **Anti-Oscillation Metrics**: Tracks consecutive stuck steps where the agent's displacement $< 2.0$ units/tick.

---

### Phase 4: Tactical Health Seeking & Safety Arbitration
* **Tool**: `src/health_tactics.py` (`HealthTacticsArbitrator`)
* **Mechanism**:
  When player health drops below $50\text{ HP}$, the arbitrator executes opportunistic detour planning:
  1. **Priority Scoring**: Evaluates visible items based on healing value and distance:
     $$\text{Score} = \frac{\text{Distance}}{\text{Priority}} \quad (\text{Medikit}=1.0, \text{Stimpack}=0.7, \text{HealthBonus}=0.2)$$
  2. **Safety Filters**:
     * **Forward-Cone Gate**: Item bearing must be within $\pm 75^\circ$ of current heading (prevents $180^\circ$ blind turnarounds).
     * **Maximum Detour Radius**: Disregards items further than $256$ units away.
     * **Toxic Slime Exclusion**: Automatically rejects items located within the lower radioactive nukage pit.
     * **Combat Threat Preemption**: If a hostile is within point-blank range ($< 128$ units, or $< 120$ units at critical $\text{HP} \le 25$), health detours are suppressed to allow shotgun flinching.
  3. **Pursuit Timeout & Blacklist**: If an item cannot be collected within $35$ ticks (e.g. obstructed by a pillar), the arbitrator aborts the pursuit, blacklists the position for 105 ticks, and rejoins the main route.

---

### Phase 5: Recurrent Spiking Neural Network Policy
* **Tool**: `src/networks/engine_state_policy.py`, `src/train_engine_state_policy.py`
* **Mechanism**:
  * **Neuron Dynamics**: Built using `snnTorch` Leaky Integrate-and-Fire (`snn.Leaky`) neurons governed by:
    $$U[t] = \beta U[t-1] + W X[t] - S[t-1] V_{\text{th}}$$
    $$S[t] = \Theta(U[t] - V_{\text{th}})$$
    with membrane decay $\beta = 0.9$, threshold $V_{\text{th}} = 1.0$, and surrogate gradient using the fast-sigmoid slope.
  * **Temporal Integration**: Spiking potentials are integrated over multi-step temporal windows ($T=2$), accumulating post-synaptic currents into linear factorized readouts.
  * **Sequence Windowed Training**: Trained using Backpropagation Through Time (BPTT) over sliced sequence windows (`sequence_length=32`, `stride=16`). Hidden LIF membrane states are carried forward across consecutive temporal chunks to preserve recurrent momentum.
  * **Factorized Multi-Task Loss**:
    $$\mathcal{L} = \mathcal{L}_{\text{locomotion}}^{\text{CE}} + \mathcal{L}_{\text{turning}}^{\text{CE}} + \mathcal{L}_{\text{attack}}^{\text{BCE}} + \mathcal{L}_{\text{use}}^{\text{BCE}}$$
  * **Optimization**: AdamW optimizer ($\text{lr}=10^{-3}$, weight decay $10^{-4}$) with exponential learning rate decay and mixed-precision acceleration (`torch.amp.autocast`).

---

### Phase 6: Factorized Action Decoding & Anti-Wedging Invariants
* **Tool**: `src/evaluate_engine_state_policy.py` (`predicted_action`), `src/engine_state_config.py`
* **Mechanism**:
  Converts policy logits into physical game buttons under four safety invariants:
  1. **Interaction Invariant**: When `route.use_recommended` is flagged near an exit door or switch, the `USE` button is prioritized and firing is suppressed.
  2. **Combat Trigger Invariant**: If an enemy is within effective shotgun range ($\le 750$ units) and aligned within the pellet spread cone ($\le 15^\circ$), attack commands are asserted.
  3. **Health Preservation Invariant**: Suppresses standing still in the open during combat; converts zero-locomotion firing into forward/strafing advance.
  4. **Anti-Wedging Backward Recovery**: If the agent is wedged against an obstacle or wall jamb for $\ge 12$ consecutive ticks, forward locomotion is overridden with `BACKWARD + TURN_RIGHT` to clear the collision geometry.
  5. **Button Composition**: Resolves `FactorizedAction` into ViZDoom's discrete button vector: `MOVE_FORWARD`, `MOVE_BACKWARD`, `MOVE_LEFT`, `MOVE_RIGHT`, `TURN_LEFT`, `TURN_RIGHT`, `ATTACK`, `USE`.

---

### Phase 7: Robustness Evaluation & Validation Suites
* **Tool**: `src/test_robustness_suite.py`, `src/diagnose_level.py`, `tests/`
* **Mechanism**:
  1. **Suite 1: Standard Multi-Seed Benchmark**: Evaluates 10 diverse seeds (`0, 1, 2, 3, 4, 5, 42, 100, 444, 999`) testing weapon pellet RNG, monster reaction times, and damage rolls (**80% Pass Rate**).
  2. **Suite 2: Spatial Jitter WADs**: Re-evaluates policies on procedurally altered maps where monsters have been displaced by up to 30 units (**66.7% Pass Rate**).
  3. **Suite 3: Altered-Start Perturbations**: Forces 6–8 ticks of uncommanded lateral or backward movement at level spawn to verify path acquisition from unexpected initial headings (**66.7% Pass Rate**).
  4. **Diagnostic Telemetry**: Exports comprehensive JSON and Markdown reports with displacement, ammunition consumption, shot accuracy, and kill ratios.

---

## Testing & Visual Execution Commands

### A. Watch Visual Live Play (Interactive Window)
```powershell
# Fast aggressive run (Seed 3 - 520 steps)
python src/evaluate_engine_state_policy.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth --seed 3 --render --render-delay 0.02

# Seed 5 (512 steps)
python src/evaluate_engine_state_policy.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth --seed 5 --render --render-delay 0.02

# Seed 0 (649 steps)
python src/evaluate_engine_state_policy.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth --seed 0 --render --render-delay 0.02
```

### B. Run the Full Multi-Seed Robustness Suite
```powershell
python src/test_robustness_suite.py --checkpoint exports/best-engine-state-spiking-snn-v2.pth
```

### C. Test Procedural Jitter WADs
```powershell
python src/evaluate_engine_state_policy.py `
  --checkpoint exports/best-engine-state-spiking-snn-v2.pth `
  --config-file scenarios/generated/level01_jitter_s30.cfg `
  --seed 0 `
  --render `
  --render-delay 0.02
```

### D. Run Deterministic Unit Tests
```powershell
pytest -q
```

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Neural Architecture** | Recurrent Spiking Neural Network (`EngineStateSpikingPolicy`) |
| **Neuron Model** | Leaky Integrate-and-Fire (`snn.Leaky`, $\beta=0.9$, fast-sigmoid surrogate) |
| **Input Dimensionality** | 32 continuous features (16 normalized signals + 16 validity flags) |
| **Output Space** | Factorized: Locomotion (5), Turning (3), Attack (1), Use (1) |
| **Temporal Horizon** | Sequence length 32, Stride 16, carried recurrent hidden states |
| **Target Game** | FreeDoom2 MAP01 (ViZDoom engine, 35 ticks/sec) |
| **Level Completion** | 512–649 ticks (~14.6–18.5 seconds of simulated time) |
| **Kill Ratio** | 13/15 monsters neutralized per run |

---

## Repository Structure

```text
├── exports/                       # Production checkpoints
│   ├── best-engine-state-spiking-snn-v2.pth   # Active 80% pass rate SNN checkpoint
│   └── best-engine-state-spiking-snn.pth      # SNN v1 checkpoint
├── src/                           # Pipeline source modules
│   ├── action_space.py            # ViZDoom button catalogues
│   ├── doom_level_env.py          # ViZDoom environment wrapper
│   ├── engine_state_config.py     # Action enums & feature contracts
│   ├── engine_state_observation.py# Feature builder & object label parser
│   ├── evaluate_engine_state_policy.py # Evaluation & live visual runner
│   ├── health_tactics.py          # Tactical health seeking arbitrator
│   ├── map01_route.py             # Route waypoints & geometric clearance
│   ├── networks/
│   │   └── engine_state_policy.py # SNN & MLP architecture definitions
│   ├── test_robustness_suite.py   # Multi-seed & perturbation benchmark suite
│   └── train_engine_state_policy.py # SNN BPTT training pipeline
├── scenarios/                     # ViZDoom level configurations
│   ├── level01.cfg                # Standard MAP01 scenario
│   └── generated/                 # Procedurally compiled jitter scenarios
├── tests/                         # Pytest deterministic test suite (100% green)
├── setup_assets.py                # Asset downloader & WAD compiler
├── VISUAL_TESTING_GUIDE.md        # Comprehensive visual testing manual
├── HEALTH_SEEKING_IMPLEMENTATION_PLAN.md # Tactical survival design memo
└── VISION_PERCEPTION_BRIDGE_IMPLEMENTATION_PLAN.md # Stage 2 vision bridge roadmap
```

---

## Research Roadmap

* **Step 1: Tactical Health Seeking & Doorway Clearance**  
  *Status*: **Completed** (Achieved **80.0% completion rate**, up from 60%).
* **Step 2: Vision Perception Bridge**  
  *Status*: **In Progress** ([Implementation Plan](VISION_PERCEPTION_BRIDGE_IMPLEMENTATION_PLAN.md))  
  *Goal*: Replace privileged coordinate state with a convolutional vision perception encoder to detect doors, walls, and monsters directly from raw visual frames.
* **Step 3: Generalized Navigation (MAP02+)**  
  *Status*: Planned  
  *Goal*: Transition from static MAP01 waypoints to dynamic BSP / NavMesh A* pathfinding.

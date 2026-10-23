# Phase 4 Implementation Plan: Generalized Navigation via WAD NavMesh & A* Pathfinding

**Branch**: `feature/generalized-navigation`  
**Milestone**: Phase 4 — Level-Agnostic Autonomous Navigation  
**Goal**: Transition the agent from static human-demonstration waypoints on MAP01 to an automated, map-agnostic WAD NavMesh and A* pathfinder capable of playing **any** Doom level (MAP01, MAP02, etc.) without human playthrough recordings.

---

## 1. Motivation & Architectural Need

Currently, the agent is tethered to FreeDoom2 MAP01:
* The navigation route ([`src/map01_route.py`](file:///e:/Side%20project/src/map01_route.py)) relies on ~60 waypoints extracted from a single human speedrun demonstration.
* Door and switch positions (`DEFAULT_USE_POINTS`) are hardcoded $(X, Y)$ constants.
* To play a new level (like MAP02 or E1M1), a human currently has to record demonstrations and manually annotate door coordinates.

### The Solution: Direct WAD Geometry Ingestion
Doom levels are stored in standard binary WAD files containing full mathematical definitions of every wall, sector, door, and item. 

By building a lightweight **WAD NavMesh and A\* Pathfinding Engine**:
1. The agent reads the level geometry directly from `freedoom2.wad`.
2. It automatically identifies the Player Spawn, Exit Linedefs, Doors, and Health Items.
3. It computes a collision-free waypoint path to the exit, formatting it into the **same frozen 84-dimensional SNN contract** (`RouteContext`).
4. **Zero human demonstrations required** for any level.

---

## 2. 3-Tier System Architecture

The architecture preserves a clean separation of concerns:

```
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 1: GLOBAL MAP PLANNER (wad_navmesh.py / wad_pathfinder.py)       │
│  • Reads raw WAD binary lumps: VERTEXES, LINEDEFS, SECTORS, THINGS      │
│  • Locates Player Start, Exit Switch, Doors, and Medikits              │
│  • Computes collision-free A* path and emits RouteContext waypoints    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ 84D Feature Contract (dx, dy, heading_err, use)
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 2 & 3: REACTIVE BRAIN (Frozen SNN + 12-Channel Vision CNN)       │
│  • Vision Encoder (4.82 ms): Detects moving hostiles from camera pixels │
│  • Recurrent LIF SNN (30.67 ms): Steers along path, aims, and fires    │
│  • Zero retraining needed: SNN contract and weights stay 100% frozen   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 4: MOTOR EXECUTION (ViZDoom Engine)                              │
│  • Steps physics, interacts with door linedefs, and clears the map     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Doom WAD Binary Structure & Entity Extraction

Doom map lumps are stored sequentially in the WAD file directory and can be parsed with standard Python `struct` (zero external dependencies):

| Lump | Bytes per Record | Extracted Data |
| :--- | :---: | :--- |
| **`VERTEXES`** | 4 bytes | 2D coordinates $(X, Y)$ for all polygon vertices (int16). |
| **`LINEDEFS`** | 14 bytes | Line segments connecting vertices. Flags determine if line is a solid wall (1-sided) or a passable portal (2-sided). Special types define doors and exits. |
| **`SECTORS`** | 26 bytes | Floor heights, ceiling heights, and damage types (slime/nukage). Height step-up $\Delta h \le 24$ is walkable; $\Delta h > 24$ requires stairs/lifts. |
| **`THINGS`** | 10 bytes | 2D positions of Player 1 Spawn, Monsters, Medikits, Stimpacks, and Armor. |

### Critical Special Types to Identify Automatically
* **Player 1 Start**: Thing Type `1`.
* **Exit Triggers**: Linedef specials `11` (S1 Exit), `51` (Secret Exit), `52` (W1 Exit).
* **Interactive Doors**: Linedef specials `1` (DR Manual Open), `26` (Blue Key), `27` (Yellow Key), `28` (Red Key), `31` (D1 Open Stay Open), `117` (Fast Door).
* **Lifts / Elevators**: Linedef specials `62` (SR Lift Lower Wait Raise), `88` (WR Lift).
* **Health Pickups**: Thing Type `2011` (Medikit +25 HP), `2012` (Stimpack +10 HP), `2018` (Armor).

---

## 4. Phased Implementation Roadmap

```mermaid
flowchart LR
    P1["Phase 4.1\nWAD Parser\n(Binary lumps to geometry)"] ──► P2["Phase 4.2\nWalkability Grid\n(Collision & portal graph)"] ──► P3["Phase 4.3\nA* Pathfinder\n(Dynamic waypoint route)"] ──► P4["Phase 4.4\nContract Adapter\n(RouteContext integration)"] ──► P5["Phase 4.5\nValidation\n(MAP01 & MAP02 Clears)"]
```

### Phase 4.1: Standalone WAD Lump Parser (`src/wad_parser.py`)
* Implement binary unpacking for `VERTEXES`, `LINEDEFS`, `SIDEDEFS`, `SECTORS`, and `THINGS`.
* Extract level bounding box, solid blocking walls, door portals, player start, and exit lines.
* Write contract test `tests/test_wad_parser.py` validating that MAP01 and MAP02 parse accurately.

### Phase 4.2: 2D Walkability Grid & NavMesh (`src/wad_navmesh.py`)
* Construct a 2D spatial occupancy grid (e.g., $16\text{ units / cell}$, matching Doom's standard grid alignment).
* Rasterize all 1-sided solid linedefs expanded by the player radius ($R = 16\text{ units}$) to prevent wall collisions.
* Mark 2-sided door linedefs as traversable passage nodes flagged with their interaction trigger.
* Mark toxic nukage sectors as high-cost avoidance cells.

### Phase 4.3: Dynamic A* Pathfinding (`src/wad_pathfinder.py`)
* Implement Euclidean A* search from player start $(X_0, Y_0)$ to the exit trigger $(X_{\text{exit}}, Y_{\text{exit}})$.
* Apply Douglas-Peucker path simplification to reduce raw grid cells to smooth, sparse waypoints.
* **Integrated Health Detour**: If player health drops below threshold, insert an A* detour node to the nearest uncollected Medikit from `THINGS` before heading to the exit.

### Phase 4.4: SNN Contract Adapter (`src/generalized_route.py`)
* Build a drop-in replacement for `Map01Route` implementing the identical `context(pos, progress_index)` interface:
  - Returns `RouteContext(target, delta, distance, bearing_degrees, use_recommended)`.
  - Automatically asserts `use_recommended = True` when the agent approaches any door linedef along the A* path.
  - Ensures 100% compatibility with the frozen 84D SNN contract.

### Phase 4.5: MAP01 Parity & Zero-Shot MAP02 Transfer
1. **MAP01 Regression Test**: Verify that the automated WAD pathfinder completes MAP01 with parity to the human-demonstration route.
2. **MAP02 Zero-Shot Evaluation**:
   - Create `scenarios/level02.cfg`.
   - Run the closed-loop vision agent on MAP02 with **zero human demonstrations**:
     ```powershell
     python src/evaluate_generalized_policy.py --level MAP02 --record-video artifacts/map02_autonomous_clear.mp4
     ```

---

## 5. Key Success Criteria

1. **Zero External Dependencies**: Pure Python WAD parsing using `struct` (no external heavyweight C++ DLLs).
2. **Frozen SNN Preservation**: `best-engine-state-spiking-snn-v3-robust.pth` weights and the 84-dimensional contract hash remain untouched.
3. **Automated Door Traversal**: The agent opens doors and triggers the exit switch on MAP02 without any hardcoded $(X, Y)$ door coordinates.
4. **Health Routing Without Retraining**: High-priority Medikits in the WAD are dynamically visited when health is low.

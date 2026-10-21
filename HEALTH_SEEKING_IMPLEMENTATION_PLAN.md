# Opportunistic Health-Seeking Controller: Implementation Plan

**Branch**: `feature/health-seeking-controller`  
**Milestone**: Step 1 — Tactical Survival Enhancement  
**Status**: Ready for Implementation  
**Target Metric**: Increase standard MAP01 pass rate from **70% to $\ge$ 80–90%** by recovering critical HP without retraining the neural network.

---

## 1. Executive Summary & Objective

In our multi-seed benchmark (`exports/best-engine-state-spiking-snn-v2.pth`), the SNN policy achieved **70% completion**, but episodes on Seeds 2, 5, and 7 failed due to cumulative hitscan chip damage in the zigzag corridor and courtyard.

The agent currently sprints along a rigid 1D speedrun trajectory, ignoring nearby Medikits (+25 HP) and Stimpacks (+10 HP) even when health drops to single digits.

### Core Objective
Implement a **dynamic, opportunistic health-seeking controller** that:
1. Detects visible health items in ViZDoom's label buffer when `health < 50`.
2. Temporarily blends an intermediate waypoint (subgoal) toward the item when safe.
3. Automatically rejoins the main route as soon as health is restored or the item is collected.
4. Preserves the active model contract hash (`ENGINE_STATE_CONTRACT_HASH`) so **no neural network retraining is required**.

---

## 2. Architectural Design

```text
 ┌────────────────────────────────────────────────────────┐
 │                   ViZDoom Game State                   │
 │   - Player Health & Position (X, Y, Heading)           │
 │   - Labels Buffer (Medikit, Stimpack, HealthBonus)     │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │            1. Health Item Perception Module            │
 │   - Filters valid items: "medikit", "stimpack", etc.   │
 │   - Computes: distance, relative bearing, screen-x     │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │             2. Tactical Safety Filter                  │
 │   Is item safe to pursue?                              │
 │   - Health < 50?                                      │
 │   - Distance <= 256 units?                             │
 │   - Bearing within forward cone (±75°)?                │
 │   - NOT in active point-blank combat (< 128 units)?   │
 │   - NOT in toxic slime/nukage sector?                  │
 └───────────────────────────┬────────────────────────────┘
               │ NO                          │ YES
               ▼                             ▼
 ┌───────────────────────────┐ ┌──────────────────────────┐
 │  Standard Route Waypoint  │ │  Dynamic Health Subgoal  │
 │  (Next Speedrun Point)    │ │  (Item Coordinates X, Y) │
 └─────────────┬─────────────┘ └────────────┬─────────────┘
               │                            │
               └──────────────┬─────────────┘
                              ▼
 ┌────────────────────────────────────────────────────────┐
 │           EngineStateSpikingPolicy (SNN Brain)         │
 │   Steers toward target, aims at enemies, moves forward │
 └────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### 3.1 Item Perception (`src/engine_state_observation.py`)
Add health item scanning alongside the existing `visible_enemies()` parser:
- **Tokens**:
  - `medikit` (+25 HP, priority 1.0)
  - `stimpack` (+10 HP, priority 0.7)
  - `healthbonus` (+1 HP, priority 0.2, only if directly on path)
- **Data Structure**:
  ```python
  @dataclass(frozen=True)
  class VisibleHealthItem:
      name: str
      heal_amount: int
      distance: float
      bearing_degrees: float
      position: tuple[float, float]
      priority_score: float
  ```

### 3.2 Tactical Safety Rules (DOOM Hazard Guardrails)
To prevent the agent from dying while trying to heal:
1. **Forward Cone Restriction**: Only pursue items within $\pm 75^\circ$ of current heading. Never perform a blind $180^\circ$ U-turn that exposes the agent's back to monsters.
2. **Maximum Detour Radius**: Disregard items further than $256$ units away.
3. **Combat Preemption**: If an enemy is actively within point-blank range ($< 128$ units) and aligned in the firing cone, **suppress the detour** and prioritize the shotgun flinch.
4. **Slime Hazard Rejection**: Blacklist known toxic nukage sector coordinates (e.g. the lower acid pit on MAP01) unless player is above 60 HP.
5. **Detour Timeout**: Limit any single health pursuit to a maximum of **35 ticks (1 second)**. If the item is not collected within 35 ticks (e.g. blocked by geometry), cancel the detour and resume the mainline route.

### 3.3 Dynamic Subgoal Blending (`src/map01_route.py`)
Modify `Map01Route.context()` or provide a `target_subgoal` override:
- When a valid health target is approved:
  - Temporarily substitute `target = item.position`.
  - Recalculate `route_dx, route_dy, route_distance, route_bearing` toward the item.
- As soon as player `health >= 50` or the item vanishes from labels (collected), the target seamlessly reverts to the next speedrun waypoint.
- Because the SNN already knows how to steer toward `route_target`, it navigates to the item using its learned motor reflexes.

---

## 4. Phase-by-Phase Execution Plan

### Phase 1: Item Perception Parser
- **File**: `src/engine_state_observation.py`
- Add `HEALTH_ITEM_TOKENS` and `visible_health_items(labels, player_x, player_y, player_angle)`.
- Unit test with mock label objects in `tests/test_engine_state_policy.py`.

### Phase 2: Tactical Health Arbitrator
- **File**: `src/health_tactics.py` (new clean modular module)
- Implements:
  - `select_health_subgoal(player_health, visible_items, nearest_enemy, current_pos, current_heading)`
  - Rejection filters (distance, angle, combat threat, timeout).

### Phase 3: Integration into Policy Evaluation
- **File**: `src/evaluate_engine_state_policy.py`
- Hook `select_health_subgoal` into the evaluation loop before generating `observation`.
- Add CLI flag `--enable-health-seeking` (default: `True`).

### Phase 4: Benchmark & Visual Validation
- Run multi-seed robustness suite on Seeds 0, 1, 2, 3, 4, 5, 7, 42, 100, 444:
  ```powershell
  python src/test_robustness_suite.py \
      --model-type snn \
      --checkpoint exports/best-engine-state-spiking-snn-v2.pth \
      --seeds 0 1 2 3 4 5 7 42 100 444 \
      --output reports/snn_v2_health_seeking_benchmark.json
  ```
- Visually verify previously failing seeds (Seed 2, Seed 5) to confirm item pickups and survival.

---

## 5. Success Criteria

| Metric | Baseline (Without Health Seeking) | Target (With Health Seeking) |
| :--- | :---: | :---: |
| **MAP01 Seed Pass Rate** | 70% (7/10 passed) | **$\ge$ 80–90% (8 or 9/10 passed)** |
| **Seeds 2, 5, 7 Survival** | Died to low HP | At least 1–2 of these seeds survive |
| **Model Reusability** | N/A | **Zero retraining** (100% compatible with existing weights) |
| **Level Portability** | N/A | Identical logic works on MAP02 without modifications |

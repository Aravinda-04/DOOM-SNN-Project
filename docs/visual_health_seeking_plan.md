# Phase 3 Implementation Plan: Visual Health & Item Seeking

**Objective**: Eliminate combat attrition across MAP01 seeds by training the Vision Perception Encoder to detect Medikits and Stimpacks from camera pixels, re-enabling the `HealthTacticsArbitrator` with **zero privileged engine cheats**.

---

## 1. Problem Statement & Motivation

During our 20-seed benchmark of the Vision-Bridged SNN:
* **Level Clear Rate**: 2/10 seeds on batch 1–10 (Seeds 3 and 9).
* **Failure Analysis**: **100% of non-clearing runs failed due to `died` (combat attrition)** after reaching 50%–67% route progress.
* **Root Cause**: The agent takes unavoidable hitscan chip damage from Zombiemen and Shotgunners. In the privileged baseline, the agent survived by detouring to nearby Medikits (+25 HP) using ViZDoom labels (`visible_health_items(labels)`). In the vision bridge, because labels are disabled (`labels = ()`), health seeking was inactive.

---

## 2. Technical Architecture

The frozen 84-dimensional SNN policy contract (`ENGINE_STATE_CONTRACT_HASH`) **remains 100% untouched**. 

Health seeking does not require modifying the SNN policy. Instead, it operates at the **navigation waypoint layer (`context`)**:

```
[Raw RGB Camera Frames (12, 84, 84)]
                  │
                  ▼
   [VisionPerceptionEncoder (12-ch CNN)]
   ├── Combat Head     ──► (Enemy Screen X, Distance, Bearing)
   └── Health Head     ──► (Health Visible, Screen X, Distance, Item Type)  <-- NEW
                  │
                  ▼
   [Spatial World Reconstruction]
   item_bearing = player_angle + (scr_x - 0.5) * 90°
   item_x = pos_x + dist * cos(item_bearing)
   item_y = pos_y + dist * sin(item_bearing)
                  │
                  ▼
   [HealthTacticsArbitrator (HP < 50)]
   Safety Filters: Distance <= 256u | Forward Cone <= 75° | Slime Exclusion
                  │
                  ├── Safe Detour Available?
                  │   ├── YES ──► Inject (item_x, item_y) as temporary target in `context`
                  │   └── NO  ──► Follow mainline speedrun waypoint in `context`
                  ▼
   [Frozen SNN Policy (v3-robust)]
   Steers smoothly toward target via delta/bearing features in 84D contract
```

---

## 3. Implementation Phases

### Phase 3.1: Ground-Truth Dataset Pipeline (`generate_vision_dataset.py`)
* During offline demonstration capture, ViZDoom's label buffer already parses `visible_health_items(labels)`.
* Save ground-truth health item arrays:
  - `health_visible`: Binary flag (1.0 if Medikit/Stimpack visible within 600 units).
  - `health_screen_x`: Normalized horizontal center $[0.0, 1.0]$.
  - `health_distance`: Normalized distance $d / 1024.0$.
  - `health_is_medikit`: Binary flag (1.0 for Medikit +25 HP, 0.0 for Stimpack +10 HP).

### Phase 3.2: Multi-Task Perception Architecture (`vision_encoder.py`)
* Add a 4th factorized head to `VisionPerceptionEncoder`:
  ```python
  self.health_head = nn.Sequential(
      nn.Linear(embedding_dim, 128),
      nn.GELU(),
      nn.Dropout(p=0.15),
      nn.Linear(128, 4),  # [vis_logit, screen_x, distance, medikit_logit]
  )
  ```
* Compute multi-task losses using masked Smooth L1 and BCE:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{combat}} + \mathcal{L}_{\text{nav}} + \mathcal{L}_{\text{use}} + \lambda_{\text{health}} \mathcal{L}_{\text{health}}$$

### Phase 3.3: Supervised Training & Calibration (`train_vision_encoder.py`)
* Fine-tune `best_vision_encoder_rgb.pth` with the added health head on the 12-channel RGB dataset.
* Verify validation loss and test detection precision on small item sprites.

### Phase 3.4: Closed-Loop Integration (`evaluate_vision_spiking_policy.py`)
* Wire the predicted health targets into `HealthTacticsArbitrator`:
  - When `health_visible_prob >= 0.50` and `player_health < 50`:
  - Reconstruct `VisibleHealthItem` from predicted `(screen_x, distance)`.
  - Pass candidates to `arbitrator.step(...)` to obtain a `HealthSubgoal`.
  - Blend `HealthSubgoal` into `RouteContext`.
* Add HUD visualization: Draw cyan/blue bounding boxes around detected health items.

### Phase 3.5: Multi-Seed Benchmark Evaluation
* Evaluate across Seeds 1 to 20.
* Target: Improve multi-seed pass rate from **20% (2/10)** to **~70%–80% (7–8/10)**.

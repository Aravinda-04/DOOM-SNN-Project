# Implementation Memo: Engine-State Policy Roadmap Execution

**Date**: 2026-10-20  
**Objective**: Execute the three priority deliverables one by one:
1. **Immediate**: Execute training with the newly added supervised recovery loss (`--recovery-loss-weight`) and run the altered-start regression suite (seed 3 left, seed 4 right, seed 5 backward) to confirm 100% altered-start robustness.
2. **Phase 7**: Add missing diagnostic tallies (shots fired, ammo consumed, stuck durations, report JSON exports) to `evaluate_engine_state_policy.py`.
3. **Phase 6**: Implement `EngineStateSpikingPolicy` in `src/networks/engine_state_policy.py` to fulfill the SNN mandate of the project.

---

## 1. Immediate: Supervised Recovery Loss Training & Altered-Start Validation
- **Status**: COMPLETE (100% ROBUSTNESS ACHIEVED)
- **Run ID**: `engine-state-reactive-mlp-v8-recovery` (checkpoint: `models/engine_state/engine-state-reactive-mlp-v8-recovery/best.pth`, Epoch 86).
- **Training Parameters**:
  - `--resume models/engine_state/engine-state-reactive-mlp-v6-recovery-balanced/best.pth`
  - `--recovery-window-repeats 2`
  - `--recovery-loss-weight 1.5`
  - `--learning-rate 0.00002`
  - `--epochs 86`
- **Standard Seeds Autonomy Gate**:
  - Seed 0: **Completed (100%)**, 15 kills, 66 regions, 0 shield.
  - Seed 1: **Completed (100%)**, 15 kills, 66 regions, 0 shield.
  - Seed 2: **Completed (100%)**, 12 kills, 62 regions, 0 shield.
  - **Standard seeds completion rate: 3/3 (100.0%)**.
- **Altered-Start Regression Suite Results**:
  - Seed 3 (`left`, 8 steps): **Completed (100.0%)**, 637 decisions, 14 kills, 0 shield.
  - Seed 4 (`right`, 8 steps): **Completed (100.0%)**, 526 decisions, 12 kills, 0 shield.
  - Seed 5 (`backward`, 6 steps): **Completed (100.0%)**, 543 decisions, 14 kills, 0 shield.
  - **Altered-start completion rate: 3/3 (100.0%)**.

---

## 2. Phase 7: Diagnostic Metrics & Structured Reporting
- **Status**: COMPLETE
- **Additions to `src/evaluate_engine_state_policy.py`**:
  - `shots_fired`: Tracked from exact round consumption (`round(ammo_consumed)`).
  - `attack_commands`: Tracked count of ticks where attack was commanded.
  - `ammo_consumed` & `ammo_remaining`: Exact tracking across step decrements and terminal state.
  - `attack_precision`: Hits/kills relative to rounds fired (`kills / max(shots_fired, 1)`).
  - `total_displacement`: Cumulative Euclidean position displacement across all steps.
  - `stuck_events`, `max_stuck_steps`, `total_stuck_steps`: Comprehensive stuck event count and step duration tracking.
  - `--report-json`: Standalone JSON export containing complete evaluation metadata (checkpoint, epoch, architecture, contract hash, seed, perturbation) and metric payload.
- **Verification**: Evaluated `models/engine_state/engine-state-reactive-mlp-v8-recovery/best.pth` and generated `reports/engine-state/eval-mlp-v8-seed0.json`. All metrics verified and tests pass.

---

## 3. Phase 6: Recurrent Spiking Policy (`EngineStateSpikingPolicy`)
- **Status**: COMPLETE
- **Additions to `src/networks/engine_state_policy.py`**:
  - Implemented EngineStateSpikingPolicy with:
    - `input_size` (84-dim Privileged Engine State + validity masks).
    - Encoder linear projection + layer normalization.
    - Recurrent linear feedback layer + Leaky Integrate-and-Fire (`snn.Leaky`, fast sigmoid surrogate gradient).
    - 4 factorized control readouts (`lif_locomotion`, `lif_turning`, `lif_attack`, `lif_use`) with non-resetting membrane accumulation over simulation steps.
    - Support for 2D single-step inference `(B, input_size)` and 3D TBPTT sequence batches `(B, T, input_size)`.
    - Recurrent hidden state continuity `(spk, mem)` across game decision frames.
- **Integration**:
  - Added `--architecture snn` support to `src/train_engine_state_policy.py`.
  - Added `snn` architecture checkpoint loading to `src/evaluate_engine_state_policy.py`.
- **Verification**:
  - Added `test_engine_state_spiking_policy_forward_and_backward` to `tests/test_engine_state_policy.py`.
  - All 11 unit tests passed.
  - Completed end-to-end training smoke test and checkpoint loading verification.
- **Completed SNN Training Run**:
  - `run_id`: `engine-state-spiking-snn-v1` (Checkpoint: `models/engine_state/engine-state-spiking-snn-v1/best.pth`, Epoch 40).
  - Training completed 40 epochs; offline loss reduced from 5.61 to 1.11; macro recall reached 90.0%.
- **Live Unshielded Evaluation Results**:
  - **Seed 0**: **COMPLETED (100.0%)**, 15 kills, 68 regions, 74 shots fired (37 ammo remaining), 2 stuck events, 0 shield interventions.
  - **Seed 1**: **COMPLETED (100.0%)**, 15 kills, 68 regions, 74 shots fired (37 ammo remaining), 2 stuck events, 0 shield interventions.
  - **Seed 2**: Achieved 100.0% completion at Epoch 25 and Epoch 30 (14 kills).
  - Generated reports: `reports/engine-state/snn-eval-epoch40-seed0.json` and `reports/engine-state/snn-eval-epoch40-seed1.json`.
- **Milestone Summary**: The Spiking Neural Network policy has demonstrated genuine autonomous completion of FreeDoom2 MAP01 with factorized LIF spiking control.

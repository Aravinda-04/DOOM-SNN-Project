# Reliability and experiment-management changes

## Summary

This update fixes path-dependent artifacts, unreliable checkpoint selection,
hard-coded model choices, misleading quantization behavior, and missing runtime
validation. Existing checkpoints and TensorBoard data were not deleted or
overwritten.

## Filesystem and experiment isolation

- Added `src/project_paths.py` as the single source of truth for the repository
  root, scenario config, checkpoint root, and TensorBoard root.
- Paths no longer depend on the terminal or IDE working directory.
- New experiments use `models/<model>/<run-id>/` and
  `runs/<model>/<run-id>/`.
- Run IDs contain a timestamp, microseconds, and seed. An explicitly reused run
  ID is rejected unless `--resume` is supplied.
- Every experiment writes its resolved configuration to `config.json`.
- Existing `models/`, `src/models/`, `runs/`, and `src/runs/` content was
  deliberately preserved.

## Training and checkpoints

- Replaced the source-edited `MODEL_TYPE` variable with a validated `--model`
  argument.
- Added command-line controls for episodes, DQN hyperparameters, evaluation,
  seed, device, scenario configuration, run ID, and resume checkpoint.
- Added a centralized model registry in `src/networks/__init__.py`; invalid
  model names now fail with a clear error.
- `best.pth` is now chosen by mean reward over greedy evaluation episodes.
  Exploratory training episodes can no longer permanently win checkpoint
  selection through a lucky early score.
- `latest.pth` is saved after every episode.
- Structured checkpoints contain model weights, optimizer state, episode,
  global step, best evaluation mean, and resolved run configuration.
- Legacy files such as `models/best_snn.pth`, which contain only a state
  dictionary, remain loadable.
- Target parameters now use `lerp_`, while integer and non-trainable buffers are
  copied directly. This avoids applying floating-point soft-update arithmetic
  to integer neuron buffers.
- Weight clipping is now accurately described as an optional stability clamp,
  not quantization.
- Replay batches are assembled with `numpy.stack` before tensor conversion,
  reducing Python overhead.
- Random seeds are applied to Python, NumPy, PyTorch, CUDA, and ViZDoom.
- Added configurable penalties for movement into a wall and identical actions
  repeated beyond a threshold. These affect the replay target only; reported
  episode reward remains the real ViZDoom reward.
- Greedy checkpoint evaluation now runs across configurable ViZDoom seeds.
- Best-checkpoint selection now prioritizes kill success rate, with mean reward
  as the tie-breaker. Checkpoints also store `best_eval_success` while remaining
  compatible with older files that do not contain it.

Replay memory is not stored in checkpoints because a full image replay buffer
can be hundreds of megabytes. Resumed runs rebuild it and repeat the configured
warm-up period.

## Evaluation

- Added validated CLI model selection, checkpoint selection, episode count,
  device, seed, config, render, and delay options.
- The evaluator finds the newest structured best checkpoint and falls back to
  the legacy root checkpoint.
- Checkpoints are loaded and validated before ViZDoom opens a window.
- Evaluation reports per-episode rewards plus mean and standard deviation.
- Environment resources are closed through `finally` blocks.

## SNN failure diagnostics

- Added `src/diagnose.py` for deterministic, greedy, multi-seed SNN evaluation.
- Reports success from `KILLCOUNT` instead of inferring it only from reward, and
  distinguishes kills, deaths, and timeouts.
- Captures per-episode action counts and traces, action streaks, ammunition use,
  position span, stalled movement, Q-values, and action-selection margins.
- Added a wall-loop heuristic for failed episodes with repeated movement into a
  boundary or long attack-dominated action loops. This is a diagnostic flag,
  not a replacement for viewing representative failed episodes.
- Uses non-invasive forward hooks to measure spikes, neuron opportunities, and
  firing rate for every LIF layer. The evaluated checkpoint is not modified.
- Produces a full JSON report, flat episode CSV, aggregate/per-seed summaries,
  SHA-256 checkpoint identity, and diagnostic TensorBoard events under
  `reports/<model>/<checkpoint>/<timestamp>/`.
- Added `AMMO2`, `KILLCOUNT`, `POSITION_X`, and `POSITION_Y` to the ViZDoom
  scenario variables and exposed read-only environment diagnostics.
- Added tests for outcome classification and correctly weighted spike-rate
  aggregation.

## Environment handling

- `basic.cfg` is resolved from the repository root by default.
- Missing or invalid configurations now raise an immediate contextual error;
  initialization no longer prints a warning and continues in a broken state.
- Added deterministic ViZDoom seeding and context-manager support.

## Spiking-model semantics

- Removed unused `spikegen` imports after the earlier switch from stochastic
  rate coding to deterministic direct coding.
- Documentation now states that RSNN recurrence spans internal neural time
  steps for a single observation. State still resets between game frames.
- True cross-frame RSNN state was not introduced because isolated-transition
  replay is mathematically incompatible with hidden state carried across
  arbitrary samples. That feature requires sequence replay and is a separate
  algorithm change.

## Quantized export

- Added `src/quantization.py` with symmetric per-tensor quantization.
- Added `src/export_quantized.py` supporting 4-, 8-, and 16-bit integer export.
- Export files contain integer tensors, scales, source metadata, and measured
  mean/maximum output error after dequantization.
- The export is an explicit interchange format. It does not claim to be a
  PeraMorphIQ bitstream or vendor deployment package.

## Dependencies and tests

- Pinned the tested Python package versions in `requirements.txt`.
- Added tests for working-directory-independent paths, experiment directory
  layout, run IDs, all three model contracts, invalid model selection,
  structured checkpoint round trips, legacy checkpoint loading, and bounded
  integer quantization.
- Added `pytest.ini` for consistent test discovery.

## New commands

```powershell
python src/train.py --model snn --episodes 200 --seed 0
python src/eval.py --model snn --no-render
python src/diagnose.py --model snn --checkpoint models/snn/snn-100-seed0/best.pth --episodes 20 --seeds 0 1 2 3 4
python src/export_quantized.py --model snn --bits 8 --output exports/snn-int8.pt
python -m pytest
```

## Shaped SNN experiment result

Trained `models/snn/snn-100-shaped-seed0/best.pth` for 100 episodes with an
epsilon floor of 0.2, decay of 5,000 steps, multi-seed validation, an eight-step
repeat threshold, and two-point repeat/stalled-movement penalties. The original
`models/snn/snn-100-seed0/` checkpoints were not modified.

Both best checkpoints were evaluated greedily over the same 100 episodes: 20
episodes for each seed from 0 through 4.

| Metric | Original | Shaped run |
| --- | ---: | ---: |
| Success rate | 65% | 75% |
| Mean reward | -74.36 | -39.60 |
| Reward standard deviation | 216.86 | 201.30 |
| Suspected wall-loop episodes | 28 | 8 |

The improvement is meaningful for this fixed evaluation set, but a 100-episode
training run is not evidence of convergence. Longer runs and additional unseen
seeds are still required before freezing the model for FPGA conversion.

## Turning and spawn-difficulty evaluation

- Expanded the action space from three to five actions by adding `TURN_LEFT`
  and `TURN_RIGHT`, allowing the agent to search beyond its initial view.
- Enabled ViZDoom's labels buffer and measure initial target offset. A target
  within 10% of screen center is an easy spawn by default; an off-center or
  invisible target is a hard spawn.
- Added spawn difficulty and hard-spawn success to JSON, CSV, console, and
  TensorBoard diagnostics.
- Added `--spawn-filter` to execute only `hard`, `easy`, `left`, `right`, or
  `invisible` initial conditions. Rejected resets do not count toward the
  requested episode total. Reports include attempts, skipped categories, and
  success rates for each accepted spawn subtype.
- Added `--max-spawn-attempts` as a per-seed safety bound so rare filters cannot
  search indefinitely.
- Checkpoint selection now ranks hard-spawn success first, total success second,
  and mean reward third. Decision-count-based non-instant success remains a
  descriptive metric but no longer determines selection.
- Loading and quantized export infer output width from checkpoint weights.
  Evaluation rejects action-count mismatches instead of silently mapping old
  outputs to the wrong actions.

The five-action run `models/snn/snn-100-search-seed0/best.pth` was trained for
100 episodes. Corrected evaluation over 100 episodes and seeds 0 through 4
reported 77% total success, 50% hard-spawn success (21/42), mean reward -64.0,
and 4 suspected loop episodes. It demonstrates genuine search behavior, but
50% hard-spawn success is not deployment-ready.

Run `--help` on any command for its complete options.

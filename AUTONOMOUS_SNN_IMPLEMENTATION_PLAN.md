# Autonomous SNN DOOM Agent — Implementation Plan

Last updated: 2026-10-11

## 1. Objective

Build one spiking-neural-network agent that can start FreeDoom2 MAP01 from the
normal spawn and reach the exit without human input.

The canonical model is the existing **CRSNN**. It receives only visual game
frames during inference, carries recurrent spiking state across the episode,
and selects game actions until the level is completed, the player dies, or the
episode times out.

The project is successful when a saved CRSNN checkpoint can be loaded in a
fresh process and completes MAP01 autonomously. The target reliability gate is
at least 8 completions in 10 deterministic evaluation attempts. A single
fully autonomous completion is the minimum capability milestone.

## 2. Scope

### In scope

- One model: `crsnn`.
- One level: FreeDoom2 MAP01 through ViZDoom.
- Visual policy input: four stacked 84 x 84 grayscale frames.
- Recurrent spiking inference state maintained for the entire episode.
- Human demonstration recording.
- Supervised imitation pretraining.
- Reinforcement-learning fine-tuning of the same CRSNN.
- Training-only progress and diagnostic signals from ViZDoom game variables.
- Checkpointing, interruption recovery, deterministic evaluation, and a
  standalone autonomous-play command.
- Changes that make training practical on an RTX 2050 laptop.

### Explicitly out of scope

- Comparing CRSNN with CNN, DRQN, RSNNv2, or historical models.
- Multi-model charts, parameter comparisons, or architecture rankings.
- Proving that SNNs outperform conventional networks.
- Deployment, FPGA work, quantization, pruning, packaging, or cloud setup.
- Supporting multiple WADs or maps before MAP01 is solved.
- Large multi-seed research experiments.

Existing comparison code and historical artifacts should be left intact, but
they are not part of the active work.

## 3. Current foundation

The repository already provides:

- `src/doom_level_env.py`: MAP01 environment, frame stacking, shaped rewards,
  terminal classification, and diagnostics.
- `src/networks/crsnn.py`: canonical cross-frame recurrent spiking Q-network.
- `src/train_level.py`: sequence-replay DQN training with burn-in and TBPTT.
- `src/diagnose_level.py`: greedy checkpoint evaluation and reporting.
- `src/checkpointing.py`: versioned checkpoints with architecture signatures.
- `scenarios/level01.cfg`: active ViZDoom scenario.
- Tests for model shape, recurrent gradients, replay, checkpoints, environment
  outcomes, and short training mechanics.

The model and training mechanics work, but no checkpoint has yet demonstrated
a real MAP01 completion. Pure epsilon-greedy learning from pixels is unlikely
to discover the complete route efficiently on laptop hardware. The active
plan therefore combines imitation learning with RL fine-tuning.

## 4. Canonical agent contract

The model remains `CRSNNQNetwork` and keeps the current interface:

```python
q_values, spike_count, hidden = model(observation, hidden)
```

Required behavior:

- `observation`: float tensor shaped `(batch, 4, 84, 84)` in `[0, 1]`.
- `q_values`: one value for every action in the fixed action catalogue.
- `spike_count`: auxiliary activity used for monitoring and regularization.
- `hidden`: recurrent membrane/spike state passed to the next decision.
- Hidden state starts as `None` at episode reset.
- Hidden state is carried across every frame, including exploratory actions.
- Training detaches state only at explicit TBPTT boundaries.
- Autonomous play uses `torch.no_grad()` and greedy action selection.

The convolutional encoder may remain non-spiking. The recurrent decision
layers must remain spiking, so the resulting policy is a hybrid recurrent SNN.

## 5. Action-space decision

Before collecting the permanent dataset, freeze one action catalogue and use
it everywhere: environment, recorder, dataset metadata, imitation trainer, RL
trainer, checkpoints, diagnostics, and autonomous play.

The current catalogue contains eight single-button controls plus no-op:

1. Move forward.
2. Move backward.
3. Strafe left.
4. Strafe right.
5. Turn left.
6. Turn right.
7. Attack.
8. Use/open.
9. No-op, with no button pressed.

No-op is required because fixed-rate human demonstrations contain decisions in
which no button is pressed. First perform one manual completion using only
these actions. If MAP01 cannot
be played comfortably without simultaneous inputs, extend the fixed catalogue
once with a small number of essential combinations such as forward + turn and
forward + attack. Do not change the catalogue after demonstrations or
checkpoints are created. Store the ordered catalogue and a hash of it in every
dataset and checkpoint.

## 6. Target pipeline

```text
Human completes MAP01 in ViZDoom spectator mode
    -> recorder captures visual observations and actions
    -> successful episode files plus a manifest
    -> sequence-based CRSNN imitation pretraining
    -> best imitation checkpoint
    -> initialize recurrent DQN from that checkpoint
    -> fine-tune with demonstrations and route-progress shaping
    -> deterministic greedy evaluation
    -> best autonomous checkpoint
    -> standalone play_level.py command
```

Coordinates, labels, and route waypoints may be used to create training
rewards and diagnostics. They must never be included in the policy observation
or required during final inference.

## 7. Phased implementation

Status notation: `[ ]` not started, `[~]` in progress, `[x]` complete.

### Phase 0 — Freeze the single-model contract

Status: `[~]`

Implementation update (2026-10-11):

- [x] Added `src/agent_config.py` as the immutable observation/action contract.
- [x] Added SHA-256 contract identity to new training configurations and
  diagnostics.
- [x] Updated `DoomLevelEnvironment` with spectator mode that shares normal
  preprocessing, frame skipping, reward calculation, and terminal handling.
- [x] Added `src/manual_level_validation.py` with JSON evidence reports and
  strict unsupported-combination accounting.
- [x] Passed a one-tic real-engine tool smoke test and the complete 89-test
  suite. The smoke correctly reported `timed_out`, not completion.
- [ ] Complete MAP01 interactively with the tool and produce a valid report.
- [ ] Based on that run, either confirm the eight atomic actions plus no-op or make the
  one permitted pre-dataset catalogue revision.

The provisional contract hash is
`ccf6ef4360c7a20f945a6e1850f62cf38121bc4717aeed9566be7ef6d120978c`.
It is emitted by `python src/manual_level_validation.py` and stored in every
validation report. It must not be declared final until the interactive
completion gate passes.

Tasks:

1. Declare `crsnn` as the only active model in the new workflow.
2. Keep the existing four-frame visual observation contract.
3. Manually verify that the current action catalogue can complete MAP01.
4. Record a real successful exit and confirm it is classified as
   `TerminalReason.COMPLETED` with the completion reward applied.
5. Freeze and hash the ordered action catalogue.
6. Add a small shared agent-configuration object containing observation shape,
   action catalogue, action repeat, frame stack, preprocessing version, and
   model name.

Deliverables:

- [x] `src/agent_config.py`
- [ ] A successful manual environment report.
- [x] Tests proving consistent action ordering across active entry points.

Gate:

- A human can complete MAP01 through the same environment/action interface.
- The engine reports the exit as completion rather than death or timeout.

### Phase 1 — Demonstration recorder

Status: `[~]`

Implementation update (2026-10-11):

- [x] Added `src/record_demonstrations.py` using the shared spectator-mode
  environment pipeline.
- [x] Added `src/demonstrations.py` with versioned schemas, atomic compressed
  episode writes, safe non-pickle loading, SHA-256 manifests, manifest
  verification, uint8 observation encoding, and whole-episode splitting.
- [x] Samples every engine tic inside each four-tic model decision. A key
  change within that window is stored as action `-1` and makes the episode
  invalid instead of receiving a misleading expert label.
- [x] Added no-op as a real catalogue action so idle human decisions have a
  valid imitation label.
- [x] Added failure retention and `--discard-failures`, notes, unique episode
  names, non-overwrite behavior, and interruption reporting.
- [x] Passed an end-to-end one-tic real-engine recording smoke. The generated
  episode and manifest reloaded with valid checksums and correctly reported a
  timeout.
- [x] Full automated suite: 97 tests passed.
- [ ] Collect at least 10 successful, contract-valid MAP01 demonstrations.
- [ ] Replay/inspect three full successful recordings for final alignment QA.

Add `src/record_demonstrations.py` using ViZDoom spectator mode. The recorder
must capture decisions at exactly the same temporal rate and preprocessing used
by the model.

Per-step data:

```text
observation     uint8 [4, 84, 84]
action          integer action index
reward          float32 shaped reward
raw_reward      float32 game reward
done            bool
position        float32 [x, y, angle]   # training metadata only
game_time       integer
```

Per-episode metadata:

```text
schema_version
created_at
map and WAD identity
action_catalogue and hash
preprocessing version
terminal_reason
completed
episode_steps
total_raw_reward
total_shaped_reward
final diagnostics
```

Storage rules:

- Store one compressed `.npz` per episode so a failed write cannot corrupt the
  whole dataset.
- Store a JSON manifest with file hashes and summary statistics.
- Save observations as `uint8`, not float tensors.
- Mark failed/death/timeout demonstrations; do not silently mix them with
  successful expert trajectories.
- Never overwrite an existing episode file.
- Provide `--discard-failures` and `--notes` options.

Target dataset:

- At least 10 successful MAP01 demonstrations.
- Prefer 20 successful demonstrations with variations in combat and movement.
- Split by complete episode: 80% training and 20% validation.
- Never split adjacent frames from one episode across train and validation.

Tests:

- Spectator button vectors map deterministically to catalogue indices.
- Dataset round-trip preserves observation/action sequences and terminal data.
- Manifest hashes detect modified or incomplete files.
- Failed episodes cannot enter the successful-only training split by accident.

Gate:

- Replay three recorded episodes and verify action/observation alignment.
- At least 10 successful, schema-valid demonstrations exist.

### Phase 2 — CRSNN imitation pretraining

Status: `[~]`

Implementation update (2026-10-11):

- [x] Added `src/imitation_data.py` with deterministic whole-episode splits,
  recurrent window indexing, safe episode caching, and normalized
  inverse-frequency action weights.
- [x] Added `src/train_imitation.py` with no-gradient burn-in, explicit TBPTT
  boundary detachment, class-weighted cross-entropy, normalized spike-rate
  regularization, gradient clipping, accumulation, and CUDA AMP.
- [x] Added complete held-out episode evaluation with accuracy, per-action
  support/recall, macro/minimum recall, confusion matrix, and spike rate.
- [x] Added optional closed-loop MAP01 evaluation and TensorBoard logging.
- [x] Extended checkpoints with scaler state, DataLoader generator state,
  validation metrics, closed-loop metrics, and strict resume compatibility.
- [x] Synthetic tests prove episode separation, burn-in gradient isolation,
  finite gradients, action weighting, and tiny recurrent-sequence overfitting.
- [x] Actual CRSNN CUDA/AMP validation produced a finite loss with nine model
  outputs.
- [x] Actual CRSNN CUDA/AMP train-save-resume smoke completed three epochs. The
  smoke checkpoint is under
  `models/crsnn/stabilization-imitation-smoke-20261011/`; it used synthetic
  data and is not gameplay evidence.
- [x] Full automated suite: 103 tests passed.
- [ ] Collect the real successful demonstration dataset.
- [ ] Run the tiny-real-subset overfit gate and full imitation training.
- [ ] Demonstrate repeatable closed-loop route progress from the trained
  imitation checkpoint.

Add `src/train_imitation.py` and a sequence dataset loader.

Training behavior:

1. Sample complete recurrent windows from demonstration episodes.
2. Reconstruct recurrent context with no-gradient burn-in.
3. Apply cross-entropy loss to expert actions over the learning window.
4. Add the existing normalized spike-rate penalty with a small coefficient.
5. Use class weighting or balanced sampling so common forward actions do not
   eliminate rare use, turn, and attack actions.
6. Clip gradients and reject NaN/Inf losses.
7. Evaluate using whole held-out episodes, not shuffled frames.
8. Save `latest.pth` and `best.pth`; rank by held-out action accuracy, rare
   action recall, and closed-loop route progress.

Recommended initial RTX 2050 configuration:

```text
batch_size: 4
sequence_length: 16
burn_in: 4
gradient_accumulation_steps: 4
mixed_precision: enabled on CUDA
checkpoint_each_epoch: true
```

Offline metrics:

- Overall held-out action accuracy.
- Per-action precision and recall.
- Confusion matrix.
- Sequence negative log likelihood.
- Spike rate and silent/saturated layer checks.

Closed-loop validation is mandatory because high offline accuracy can still
fail after one incorrect turn. Every few epochs, run the policy greedily from
the normal spawn and record route progress, death, timeout, and completion.

Checkpoint requirements:

- Store format version, dataset manifest hash, model signature, action
  catalogue hash, preprocessing version, epoch, optimizer/scaler state, and
  validation metrics.
- Support exact resume after interruption.

Tests:

- A tiny synthetic demonstration set can be overfit.
- Loss at later frames produces gradients through earlier recurrent state.
- Burn-in frames do not receive gradients.
- Train/validation episode separation is enforced.
- A saved imitation checkpoint reloads with identical greedy outputs.

Gate:

- The model overfits a tiny real subset as a pipeline check.
- Held-out accuracy and all rare-action recalls exceed a documented baseline.
- Greedy closed-loop play makes repeatable progress beyond the starting area.

### Phase 3 — Demonstration-guided RL fine-tuning

Status: `[ ]`

Extend `src/train_level.py` rather than creating a second incompatible RL
trainer.

Required additions:

1. `--init-checkpoint` loads model weights from imitation training without
   treating the checkpoint as an RL resume.
2. `--resume` continues to restore an interrupted RL run exactly.
3. Seed replay with successful demonstration episodes.
4. Retain a configurable fraction of demonstration windows in each batch so
   rare successful behavior is not immediately forgotten.
5. Add an auxiliary expert-action loss on demonstration samples while keeping
   the DQN TD loss on all samples.
6. Lower initial exploration for an imitation-pretrained policy and decay it
   gradually; do not begin again at fully random epsilon 1.0.
7. Optimize once every configurable number of environment decisions instead
   of necessarily after every decision.
8. Add CUDA automatic mixed precision and gradient scaling.
9. Save replay state or a compact replay snapshot so laptop training can be
   stopped and resumed without repeatedly rebuilding experience.

Recommended starting loss:

```text
total_loss = td_loss
           + imitation_weight * expert_cross_entropy
           + sparsity_weight * normalized_spike_rate
```

Recommended starting schedule, to be tuned from short runs:

```text
epsilon_start: 0.20
epsilon_end: 0.02
optimize_every: 4 decisions
target update: existing soft update
batch_size: 4 or 8 depending on measured VRAM
sequence_length: 16
burn_in: 4
```

Route-progress reward:

- Derive an ordered set of coarse route waypoints from successful
  demonstrations.
- Reward only newly reached furthest progress; never reward oscillating around
  the same waypoint repeatedly.
- Keep exploration, survival, combat, death, and completion components
  separately logged.
- Make completion the dominant positive reward.
- Do not pass waypoint indices or coordinates into the CRSNN.

Curriculum:

1. Keep combat disabled only if a training-only scenario variant is necessary
   to prove basic route following.
2. Restore the normal MAP01 scenario before ranking final checkpoints.
3. Increase episode timeout only when the policy reaches progressively later
   route segments.
4. Promote checkpoints by autonomous completion first, route progress second,
   and shaped reward only as a tie-breaker.

Tests:

- Imitation checkpoints initialize RL while incompatible checkpoints fail.
- Demonstration sampling ratio is respected.
- Expert auxiliary loss applies only to labelled demonstration sequences.
- Newly reached waypoint progress is rewarded once.
- Replay snapshots round-trip without changing stored transitions.
- AMP and non-AMP paths produce finite gradients.

Gate:

- At least one autonomous episode reaches substantially beyond the imitation
  model's failure point.
- No persistent action collapse, unbounded Q-values, or NaN/Inf loss occurs.
- At least one fully autonomous MAP01 completion is recorded.

### Phase 4 — Autonomous play command

Status: `[ ]`

Add `src/play_level.py` as the final user-facing entry point.

Example interface:

```powershell
python src/play_level.py `
  --checkpoint models/crsnn/map01-agent/best.pth `
  --device cuda `
  --render `
  --episodes 1
```

Required behavior:

- Load and validate the checkpoint, model signature, action catalogue, frame
  preprocessing, and scenario identity.
- Reset recurrent state only at the beginning of an episode.
- Select greedy actions with no exploration.
- Run without position, labels, demonstration data, or route waypoints as
  model inputs.
- Print terminal reason, completion, elapsed decisions, raw reward, shaped
  reward, kills, items, and route coverage.
- Optionally save a trajectory and video without affecting actions.
- Return a non-zero process exit code for checkpoint/configuration errors, not
  merely because an individual gameplay attempt fails.

Tests:

- A deterministic fake model produces the expected action sequence.
- Hidden state persists between frames and resets between episodes.
- Incompatible action/preprocessing metadata is rejected before play.
- Headless real-engine smoke execution works from a fresh process.

Gate:

- The best checkpoint completes MAP01 through `play_level.py` without any
  training-only inputs or human intervention.

### Phase 5 — Reliability and final acceptance

Status: `[ ]`

This phase evaluates only the chosen CRSNN; it is not a model comparison.

Protocol:

1. Freeze the checkpoint and resolved agent configuration.
2. Run 10 headless greedy attempts from the standard MAP01 start.
3. Record completion, death, timeout, completion time, and action distribution.
4. Run at least one rendered attempt and visually verify genuine gameplay.
5. Preserve a successful trajectory or video and the exact invocation.

Acceptance levels:

- **Minimum capability:** one uninterrupted autonomous MAP01 completion.
- **Working agent:** at least 5 completions in 10 attempts.
- **Project target:** at least 8 completions in 10 attempts.

Final artifact set:

```text
models/crsnn/map01-agent/
    best.pth
    latest.pth
    config.json
    dataset_manifest_hash.txt

reports/autonomous-map01/
    evaluation.json
    episodes.csv
    successful_trajectory.npz
    successful_run.mp4          # optional
    reproduction.md
```

## 8. Laptop execution strategy

The target development GPU is an RTX 2050, commonly limited to 4 GB VRAM.
Training must be interruptible and memory-conscious.

Rules:

- Begin with batch size 4 and use gradient accumulation.
- Enable mixed precision only after a finite-loss smoke test.
- Store demonstration and replay observations as `uint8`.
- Keep sequence length at 16 initially; do not reduce recurrent context merely
  to force a larger batch.
- Evaluate infrequently during early training and use one evaluation episode
  until the policy shows real route progress.
- Checkpoint at least every epoch for imitation and at a predictable interval
  for RL.
- Record actual seconds per epoch and per episode; calculate ETA from measured
  runtime rather than GPU-name estimates.
- Keep the laptop plugged in, use the high-performance GPU mode, disable
  rendering during training, and monitor temperature and VRAM externally.

Short-run progression:

1. Dataset/model contract validation.
2. Tiny-subset overfit test.
3. One imitation epoch.
4. Five closed-loop evaluation episodes.
5. Short RL fine-tuning run.
6. Continue only when route progress is visible and numerical metrics remain
   stable.

## 9. Metrics that matter

Primary metrics:

- Autonomous level completions.
- Completion rate.
- Furthest route waypoint reached.
- Completion time for successful runs.

Safety/debugging metrics:

- Death and timeout rates.
- Action distribution and longest repeated-action streak.
- Q-value magnitude and margin.
- TD and imitation losses.
- Spike rate per spiking layer.
- GPU memory and wall-clock time.

Shaped reward is a training signal, not the definition of success. A policy
with high shaped reward but no exits has not solved the level.

## 10. Immediate implementation order

The next coding agent should follow this order:

1. [x] Update active documentation so this file is the canonical plan.
2. [~] Validate a real manual MAP01 completion through the active environment;
   the tool is implemented and the interactive completion is pending.
3. [~] Freeze and centralize the action catalogue and preprocessing metadata;
   the nine-action catalogue is centralized and awaits manual confirmation.
4. [x] Implement and test `record_demonstrations.py`.
5. [ ] Collect and validate at least 10 successful demonstrations.
6. [x] Implement the episode-based demonstration dataset loader.
7. [x] Implement and test `train_imitation.py`.
8. [~] Run tiny-overfit and one-epoch laptop smoke tests; synthetic and
   mechanical CUDA tests pass, while real-demo learning awaits the dataset.
9. Add `--init-checkpoint`, demonstration replay mixing, route-progress reward,
   AMP, configurable optimization frequency, and replay persistence to
   `train_level.py`.
10. Fine-tune until the first autonomous completion.
11. Implement `play_level.py` and run the final acceptance protocol.

Do not spend time on baseline models, comparison reports, deployment, or
hardware export while these steps are incomplete.

## 11. Definition of done

The project is done for the current scope when all of the following are true:

- A CRSNN checkpoint starts in a fresh process.
- Its only policy input is the four-frame visual observation.
- It controls the normal MAP01 environment without human intervention.
- It reaches the exit and the environment records `completed`.
- The successful run is reproducible with a documented command.
- The checkpoint, agent configuration, action catalogue, preprocessing
  version, and training lineage are preserved together.
- No comparison or deployment work is required for acceptance.

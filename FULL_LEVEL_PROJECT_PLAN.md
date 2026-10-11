# Full-Level DOOM SNN Project Plan

> **Historical plan:** As of 2026-10-11, active work is governed by
> [AUTONOMOUS_SNN_IMPLEMENTATION_PLAN.md](AUTONOMOUS_SNN_IMPLEMENTATION_PLAN.md).
> The project now targets one CRSNN agent completing MAP01; model comparisons
> and deployment are out of scope. This document is retained as stabilization
> history and technical background.

## 1. Purpose of this document

This document is the implementation and experiment plan for the next phase of
the project. It is intended to let another developer or coding agent understand
the repository, identify the current blockers, and continue without relying on
conversation history.

## Implementation status

Last updated: 2026-10-11

Current effort: **Model stabilization and controlled baselines (Phases 0-5)**

- [x] Phase 0: Preserve and establish the baseline.
- [x] Phase 1: Repository and interface consistency.
- [~] Phase 2: Correct environment semantics and metrics.
- [x] Phase 3: Correct recurrent learning and replay.
- [x] Phase 4: Checkpointing and end-to-end smoke validation.
- [~] Phase 5: Establish controlled model baselines.
- [ ] Phase 6: Curriculum learning.
- [ ] Phase 7: Full-scale training and evaluation.
- [!] Phase 8: Optimization, deployment, or hardware work is deferred by the
  project owner; current work must remain model-focused.

Status notation: `[ ]` not started, `[~]` in progress, `[x]` complete, and
`[!]` blocked. Detailed work notes and verification evidence are recorded in
the relevant phase sections as implementation proceeds.

### Latest verification evidence

- Full suite after the baseline, profiler, checkpoint-signature, report
  comparison, and checkpoint-identity additions: **74 passed**.
- Real ViZDoom contract validation: passed with observation shape
  `(4, 84, 84)` and Q-value shape `(1, 8)`.
- Real-engine timeout classification: passed.
- Real-engine forced-death classification: passed.
- CRSNN smoke training, evaluation, checkpoint creation, and resume: passed.
- Diagnostic JSON/CSV/TensorBoard/trajectory generation: passed.
- Successful MAP01 exit classification is unit-tested but still needs one
  real-engine scripted or manual completion before Phase 2 is complete.

The successful smoke run is
`models/crsnn/stabilization-crsnn-smoke-20261010-v2/`. Its latest checkpoint is
the earlier format version 2 at completed episode index 1 and global step 8.
New checkpoints use format version 3 with an automatic model state-shape
signature. The matching diagnostic report is
`reports/level/crsnn-20261010-193221/`.

The earlier `stabilization-crsnn-smoke-20261010` run stopped before evaluation
because Unicode console symbols were incompatible with the Windows CP1252
console. Console status output was converted to ASCII, and the failed run was
preserved rather than overwritten.

The immediate objective is **not** to launch a long training run. The immediate
objective is to make the new full-level training pipeline correct, measurable,
repeatable, and safe to run for long periods.

## 2. Project objective

Train and evaluate an autonomous agent that can complete FreeDoom2 MAP01 from
visual observations using a recurrent spiking neural network.

The intended active pipeline is:

```text
FreeDoom2 MAP01
    -> ViZDoom
    -> four 84 x 84 grayscale frames
    -> CRSNN or RSNNv2 policy
    -> eight discrete-action Q-values
    -> recurrent DQN training with sequence replay
    -> greedy multi-seed evaluation
```

The eight actions are:

1. Move forward.
2. Move backward.
3. Strafe left.
4. Strafe right.
5. Turn left.
6. Turn right.
7. Attack.
8. Use/open.

Completion rate on held-out seeds is the primary target metric. Reward, kills,
map exploration, episode length, action distribution, spike activity, and
failure type are supporting metrics.

## 3. Current repository state

### Active full-level implementation

- `src/doom_level_env.py`: full-level ViZDoom environment, frame stacking,
  action mapping, reward shaping, and diagnostics.
- `src/networks/crsnn.py`: cross-frame recurrent spiking Q-network.
- `src/networks/rsnn_v2.py`: second cross-frame recurrent spiking Q-network.
- `src/train_level.py`: recurrent DQN trainer with episode-based sequence
  replay.
- `src/diagnose_level.py`: greedy evaluation and report generation.
- `scenarios/level01.cfg`: FreeDoom2 MAP01 configuration.
- `deploy.ps1`: remote server packaging and training launcher.
- `tests/test_level_env.py`: mocked environment contract tests.
- `tests/test_crsnn.py`: recurrent-network contract tests.

### Historical work

The repository also contains artifacts from earlier phases:

- Three-action and five-action combat agents.
- FFNN, SNN, RSNN, compact SNN, and DRQN experiments.
- Hard-spawn aiming experiments.
- Corridor navigation and curriculum experiments.
- Quantized INT8 export and comparison work.
- Historical checkpoints, TensorBoard runs, and diagnostic reports.

These artifacts must not be deleted or overwritten. They are useful baselines
and may contain reusable curriculum code or pretrained weights.

### Git/worktree warning

At the time this plan was written, the working tree is mid-migration:

- Several previously tracked corridor, training, diagnostic, evaluation, and
  quantization files appear deleted.
- The new full-level files are untracked.
- The README and other documentation refer to some files that no longer exist
  in the working tree.
- The current branch is `Coordinates_Generation`.

Before changing code, inspect `git status` and `git diff`. Do not restore,
discard, or overwrite user changes without explicit authorization. In
particular, do not use `git reset --hard`, destructive checkout commands, or
bulk cleanup.

### Test result at discovery and after stabilization

The initial suite produced:

```text
36 passed, 1 failed
```

The failing test expects root-level `basic.cfg`, but that file is absent while
`src/project_paths.py` still defines it as the default scenario. This is a
repository-consistency failure, not a failure of the CRSNN tensor-shape tests.

That path contract has now been migrated to `scenarios/level01.cfg`. After the
model-baseline work, the current suite produces:

```text
74 passed
```

The only emitted warning is an installed `requests` dependency-version warning;
it is not a project test failure.

## 4. Original stabilization findings and resolutions

These problems must be addressed before long training.

### 4.1 Successful exits are never recorded

**Status: implemented; real completion verification remains.** Terminal
classification now uses ViZDoom's explicit death and timeout APIs and treats a
finished, living, non-timeout episode as completion. Death and timeout have
real-engine tests; completion currently has mocked coverage only.

`DoomLevelEnvironment._exit_triggered` is reset and read but is never changed to
`True`. Consequences include:

- Completion rate is always zero.
- The exit bonus is never applied.
- Best-checkpoint selection cannot select for actual level completion.
- Diagnostics can label successful runs as timeouts.

The terminal reason must be classified reliably as one of:

```text
completed | died | timed_out | engine_error
```

For MAP01, investigate whether ViZDoom exposes an explicit completion signal.
If not, classify an episode that ended while the player is alive and before the
configured timeout as completion. Put this logic in one tested helper rather
than duplicating it across training and diagnostics.

### 4.2 Terminal diagnostics discard final statistics

**Status: resolved.** The environment preserves the last readable game-variable
snapshot and exposes terminal reason, raw reward, shaped reward, and individual
reward components.

When the episode has finished, `diagnostics()` currently returns zeros for
kills, items, secrets, position, health, and ammunition. Evaluation therefore
loses the episode outcome at exactly the point when it is recorded.

Maintain a final diagnostic snapshot before state access becomes unavailable.
The terminal snapshot must preserve, at minimum:

- Kill, item, and secret counts.
- Last known position and angle.
- Health and ammunition.
- Episode time.
- Raw game reward and shaped reward separately.
- Terminal reason and completion flag.
- Number of sectors or exploration regions visited.

### 4.3 Recurrence is detached at every game frame

**Status: resolved.** Models keep hidden state connected inside the learning
window. The trainer creates explicit no-gradient burn-in and TBPTT boundaries.
Tests prove both cross-frame gradient flow and boundary detachment.

Both active spiking models call `.detach()` whenever hidden state is supplied.
The trainer consequently does not backpropagate through the configured
sequence window, even though it describes the process as TBPTT.

Hidden-state detachment policy belongs in the training loop:

- Start each sampled replay window with `hidden=None`, or with a no-gradient
  burn-in state.
- Carry hidden state through every learning step without detaching it.
- Detach only at the explicit TBPTT boundary between windows/chunks.
- During inference, carry hidden state for the entire episode under
  `torch.no_grad()`.

Add a gradient-flow test proving that a loss at a later frame can affect an
earlier frame or state within the training window.

### 4.4 Exploration leaves recurrent state stale

**Status: resolved.** Action selection evaluates the policy on every frame and
applies epsilon only after recurrent state has advanced. A regression test
covers the fully random branch.

During epsilon-random action selection, the policy is not evaluated, so its
hidden state is not updated from the current observation. The next greedy
decision then receives a hidden state from an older frame.

Evaluate the policy on every observation to advance hidden state. Epsilon
should choose between the greedy action and a random action only after that
forward pass.

### 4.5 Replay capacity is defined in episodes

**Status: resolved for the active trainer.** Capacity is measured in completed
transitions, observations are stored as `uint8`, redundant next observations
are reconstructed, old episodes are evicted, and estimated observation memory
is logged.

`MEMORY_SIZE = 50_000` currently means up to 50,000 complete episodes, while
each transition stores both a four-frame observation and a four-frame next
observation as float16. This can exhaust RAM long before the nominal capacity.

Redesign replay storage with these requirements:

- Capacity is measured in transitions or an explicit byte budget.
- Eviction removes old complete episodes while respecting the transition
  budget.
- Observations are stored as `uint8` where possible and normalized when
  sampled.
- Avoid storing both `obs` and `next_obs` when the next observation can be
  reconstructed from adjacent episode entries.
- Log current transition count, episode count, and estimated replay memory.
- Reject a sequence length larger than every eligible stored episode with a
  clear error.

Start conservatively, measure real memory usage, and increase capacity only
when justified.

### 4.6 Checkpoint resume state is incomplete

**Status: resolved for model experiments.** Checkpoint format version 2 stores
best completion, latest checkpoints are periodic, final evaluation is forced,
resume compatibility is validated before model loading, interruption writes a
recoverable checkpoint, and a train-save-resume smoke workflow passed.

The trainer restores best mean reward but not best completion rate. This can
cause a resumed run to replace the best checkpoint with an inferior policy.
The `latest.pth` checkpoint is also only saved at evaluation intervals.

Checkpoint requirements:

- Persist best completion rate and all checkpoint-ranking fields.
- Persist model, optimizer, episode, global step, resolved configuration, and
  checkpoint format version.
- Save `latest.pth` at a predictable episode interval and on clean shutdown.
- Save `best.pth` only after a completed deterministic evaluation.
- Validate architecture, input channels, action count, and important training
  configuration before resuming.
- Document that replay memory is intentionally rebuilt, if that remains the
  design.
- Preserve compatibility with historical checkpoints where practical.

### 4.7 Documentation and deployment commands are inconsistent

**Status: active model documentation resolved; deployment intentionally
deferred.** The root README contains only active local model commands. The
previous README is preserved at `docs/legacy_pipeline_readme.md`. Per project
owner direction, do not modify or validate deployment behavior in the current
phase.

Examples currently reference deleted scripts and unsupported arguments.
`deploy.ps1` invokes `train_level.py --stage full`, but `--stage` is not an
accepted argument. The deployment script installs PyTorch 1.12 while the local
requirements pin PyTorch 2.7.1 and describe Python 3.13 as tested.

All documented and deployment commands must be executed at least through
argument validation. There should be one canonical local command and one
canonical server command.

### 4.8 CRSNN and RSNNv2 are effectively duplicates

**Status: resolved architecturally; comparative training remains.** CRSNN keeps
a dense learned 512 x 512 all-to-all recurrent projection. RSNNv2 now uses a
learned element-wise `RLeaky` recurrence. CRSNN has 1,903,736 trainable
parameters; RSNNv2 has 1,641,081. Both share the same observation, output, and
hidden-state interface.

The two current implementations have almost identical layer structure,
recurrent connection, hidden-state handling, and forward logic. Before using
both as experimental variants, define the intended scientific difference. If
there is no meaningful difference, keep one canonical implementation to avoid
misleading comparisons.

### 4.9 Reward signals require validation

**Status: partially resolved.** The invalid label-sector assumption was removed
because installed ViZDoom labels do not expose sector indices. Exploration now
uses coarse training-only position cells, generic items are no longer called
keys, and all reward components are logged separately. Longer behavioral
validation is still required to tune the weights and detect exploitation.

The current reward implementation assumes object labels expose a
`sector_index`; this must be checked against actual ViZDoom label objects.
Key collection is inferred from a generic item-count increase and may reward
ordinary items as keys. Exit reward and death behavior also need real-engine
verification.

Log reward components separately:

```text
raw_game_reward
kill_reward
health_reward
ammo_reward
item_or_key_reward
exploration_reward
step_penalty
death_penalty
completion_reward
total_shaped_reward
```

This makes reward exploitation and incorrect event detection visible.

## 5. Execution plan

## Phase 0: Preserve and establish the baseline

### Tasks

1. Record `git status`, the current commit, Python version, package versions,
   GPU/CUDA details, and available disk/RAM.
2. Inventory historical checkpoints and reports without modifying them.
3. Confirm whether the tracked-file deletions are intentional.
4. Decide whether the old corridor implementation should be restored on a
   separate branch or consulted directly from Git history.
5. Run the test suite with cache writing disabled and save the result.
6. Run `--help` for every active command.

### Deliverable

A short baseline report containing environment details, Git state, tests, and
the exact commands used.

### Gate

No destructive Git operations and no training run longer than a smoke test.

### Implementation result (complete)

- Baseline: branch `Coordinates_Generation`, commit `e55c2d9`.
- Runtime: Python 3.13.1, PyTorch 2.7.1+cu118, snnTorch 1.0.0,
  ViZDoom 1.3.0, NumPy 2.5.2, and OpenCV 4.11.0.
- Hardware: NVIDIA GeForce RTX 2050 with CUDA available.
- The dirty migration state and historical artifacts were preserved.
- No destructive Git operation was used.

## Phase 1: Repository and interface consistency

### Tasks

1. Choose the supported active entry points:
   - `src/train_level.py`
   - `src/diagnose_level.py`
   - Any explicit baseline evaluator added during this phase.
2. Resolve `basic.cfg` consistently:
   - Restore it only if the legacy pipeline remains supported, or
   - Remove it from active path contracts and update tests/documentation.
3. Remove duplicate constants and unused imports from `train_level.py`.
4. Define one common recurrent model return contract, preferably:

   ```python
   q_values, auxiliary_metrics, hidden = model(observation, hidden)
   ```

5. Update README commands to match files that actually exist.
6. Correct `deploy.ps1` arguments and align its Python/PyTorch dependency
   strategy with `requirements.txt`.
7. Add a lightweight configuration validation command that does not start
   training.

### Tests

- All existing tests pass.
- Every documented command accepts its documented arguments.
- Model registry rejects unsupported model names clearly.
- Paths work from the repository root, `src`, and a different working
  directory.

### Gate

The repository must have zero failing tests before moving to environment
semantics.

### Implementation result (complete for local model work)

- The canonical scenario path is `scenarios/level01.cfg`; the nonexistent
  `basic.cfg` active-path contract was removed.
- Active commands are `train_level.py` and `diagnose_level.py`.
- The root README was rewritten for the active model pipeline, while earlier
  user-authored material was preserved under `docs/`.
- Duplicate trainer constants and stale active imports were removed.
- `--validate-only` now performs a real scenario/model contract check.
- Deployment work is explicitly out of scope by project-owner direction.

## Phase 2: Correct environment semantics and metrics

### Tasks

1. Introduce a typed terminal-reason representation.
2. Detect death, timeout, and completion with the real engine.
3. Preserve the final diagnostic snapshot.
4. Separate raw reward, reward components, and scaled training reward.
5. Validate the order and count of configured buttons and game variables after
   ViZDoom initialization.
6. Verify whether player death ends an episode under the current config.
7. Verify what happens when MAP01 exits normally.
8. Validate label-based exploration; replace it if `sector_index` is not
   available.
9. Replace coarse key inference with a reliable signal or rename it to generic
   item reward.
10. Add bounds checking for action indices.

### Tests

Unit tests must cover:

- Death terminal classification.
- Timeout classification.
- Successful completion classification.
- Terminal diagnostics retaining kills and position.
- Each reward component independently.
- Invalid action indices.
- Frame-stack reset and update behavior.
- Missing or malformed scenario configuration.

Real-engine integration tests must cover:

- The scenario starts from any working directory.
- A timeout produces a timeout result.
- A forced or scripted death produces a death result.
- A scripted or manually verified exit produces a completion result.

### Deliverable

An environment smoke report containing one example of every terminal outcome
and the corresponding raw diagnostics.

### Gate

Do not train until a real MAP01 completion is recorded as `completed=True` and
receives the intended completion reward.

### Implementation result (in progress)

- Mocked death, timeout, and completion classification pass.
- Real-engine timeout and forced-death classification pass.
- Final diagnostics retain game variables and reward components.
- Button order, game-variable order, and grayscale format are validated against
  the initialized ViZDoom engine.
- The remaining gate is a real scripted or manually verified MAP01 exit. Do not
  interpret the short timeout smoke run as completion validation.

## Phase 3: Correct recurrent learning and replay

### Tasks

1. Move hidden-state detachment out of model `forward()` and into explicit
   sequence-boundary helpers.
2. Make recurrent state advance on every environment observation, including
   epsilon-random actions.
3. Define correct hidden-state alignment for `Q(s_t)` and target `Q(s_{t+1})`.
4. Redesign replay around a transition or byte budget.
5. Consider a burn-in prefix for sampled sequences:
   - Burn-in frames reconstruct hidden state without gradients.
   - Learning frames contribute TD and sparsity losses.
6. Ensure terminal masks stop bootstrapping and hidden state does not cross an
   episode boundary.
7. Normalize spike loss by actual neuron opportunities rather than only batch
   and sequence length, or clearly rename the metric if it remains a raw spike
   count.
8. Add replay-memory and optimization timing metrics.
9. Confirm target-network buffers are copied safely and floating parameters are
   softly updated.

### Tests

- Later-frame loss propagates to earlier recurrent computation within a
  sequence.
- No gradient crosses a configured TBPTT boundary.
- Hidden state resets between episodes.
- Hidden state advances during random action selection.
- Sampled windows never cross episode boundaries.
- Terminal transitions do not bootstrap.
- Replay eviction respects its transition or byte limit.
- Training one synthetic batch changes policy parameters without changing the
  target network until a target update is applied.

### Deliverable

A deterministic synthetic-sequence test and a memory benchmark showing the
maximum expected replay footprint.

### Gate

Do not start a long run until recurrent gradient flow and replay memory bounds
are demonstrated automatically.

### Implementation result (complete)

- Cross-frame gradients flow within a learning window.
- Burn-in reconstructs hidden state without gradients.
- Explicit detachment prevents gradients crossing TBPTT boundaries.
- Target hidden state is aligned with each `s_t -> s_(t+1)` transition.
- Random exploration still advances recurrent state.
- Replay uses a transition budget and compact `uint8` observations.
- Terminal transitions do not bootstrap.
- Spike regularization and diagnostics use firing rate normalized by neuron
  opportunities rather than raw spike counts.

## Phase 4: Checkpointing and end-to-end smoke validation

### Tasks

1. Upgrade the checkpoint format with completion-ranking metadata.
2. Make save intervals configurable.
3. Add clean interruption handling that writes a recoverable latest
   checkpoint.
4. Validate resume compatibility before opening ViZDoom.
5. Create a tiny CPU-friendly smoke configuration:
   - Few episodes.
   - Small replay threshold.
   - Short timeout.
   - One evaluation seed.
6. Run train -> save -> terminate -> resume -> evaluate -> diagnose.
7. Confirm TensorBoard step counters remain monotonic after resume.

### Tests

- Structured checkpoint round trip.
- Resume preserves episode, global step, optimizer, best completion, and best
  reward.
- Incompatible action count or input channels fail clearly.
- A legacy checkpoint either loads through a documented migration path or is
  rejected with an actionable message.

### Deliverable

One end-to-end smoke-run directory containing:

```text
config.json
latest.pth
best.pth
TensorBoard events
diagnostic report.json
episodes.csv
trajectory plot(s)
```

### Gate

The complete workflow must run twice, including resume, without manual file
editing.

### Implementation result (complete)

- Checkpoint format version 3 stores a state-shape signature and passed
  structured, legacy, and incompatible-architecture tests.
- Incompatible architecture metadata is rejected before model weights load.
- A corrected CRSNN smoke run trained and evaluated for one short episode,
  producing `best.pth`, `latest.pth`, `config.json`, and TensorBoard data.
- The run resumed from episode 1/global step 4 and completed episode 2/global
  step 8 without manual file editing.
- A diagnostic pass loaded `best.pth` and produced JSON, CSV, TensorBoard, and
  trajectory artifacts.

## Phase 5: Establish non-spiking and spiking baselines

**Current status: in progress.** The recurrent spiking variants are now
architecturally distinct. Four-channel `level_cnn` and parameter-matched
`level_drqn` baselines implement the same trainer contract. CRSNN, RSNNv2,
LevelCNN, and the earlier 512-unit DRQN all passed short mechanical training;
the final 128-unit DRQN has passed focused model tests and still needs its
replacement smoke run and controlled learning probe. Attempts to launch that
replacement smoke run on 2026-10-11 were prevented before process creation by
the local execution helper's `setup refresh had errors` failure; this was not a
Python, CUDA, checkpoint, or model exception. No permission prompt should be
used to work around it.

The authoritative resource report is
`reports/model-profile-full-level-20261011/`. Earlier profile directories from
2026-10-10 are superseded because the first undercounted LSTM parameter names
and the second used the intentionally rejected 512-unit, 7.51M-parameter DRQN.

Current software-graph profile:

| Model | Parameters | Recurrent state values | Dense MAC estimate/decision |
| --- | ---: | ---: | ---: |
| LevelCNN | 1,641,080 | 0 | 4,814,848 |
| LevelDRQN-128 | 1,704,056 | 256 | 4,877,312 |
| CRSNN | 1,903,736 | 12,648 | 36,024,320 |
| RSNNv2 | 1,641,081 | 12,648 | 33,402,880 |

The spiking estimates include ten internal neural steps. They describe the
current dense PyTorch execution and do not imply hardware energy or latency.

`src/compare_level_models.py` now accepts repeated
`--report LABEL=path/to/report.json` arguments and creates a normalized JSON,
CSV, and four-panel comparison chart. This keeps completion, failure, reward,
and exploration comparisons consistent across the four model families.

The project needs controlled comparisons before architectural claims can be
made.

### Models

1. Feed-forward CNN DQN using the same four-frame observation.
2. Conventional LSTM/GRU DRQN.
3. Canonical CRSNN.
4. RSNNv2 only if it is made architecturally distinct.

The historical `DRQN` assumes one input channel and has a different return-value
order. A separate `level_drqn` now provides the full-level interface with a
128-unit LSTM selected to keep its parameter count near the CNN and spiking
models. The `level_cnn` baseline uses the identical four-channel convolutional
front end without recurrent state.

### Experimental controls

All models must use the same:

- Training and held-out evaluation seeds.
- Observation preprocessing.
- Eight-action space.
- Reward definition.
- Training-step or environment-step budget.
- Evaluation interval and episode count.
- Checkpoint-ranking rule.
- Reporting format.

Record parameter count, approximate operations, wall-clock training speed,
peak RAM/VRAM, and spike activity where applicable.

### Initial experiment scale

Use staged budgets rather than jumping to 2,000 episodes:

1. Five-episode mechanical smoke run.
2. Twenty-five-episode learning-signal probe.
3. One-hundred-episode baseline comparison.
4. Longer training only if held-out metrics improve beyond random behavior.

### Success criteria

- No NaN or infinite losses/Q-values.
- Replay and process memory remain bounded.
- The policy uses more than one action meaningfully.
- Exploration metrics improve over random behavior.
- At least one baseline produces repeatable partial progress before full-scale
  CRSNN training is approved.

## Phase 6: Curriculum for full MAP01 completion

Learning an entire level directly from pixels and a sparse completion reward is
unlikely to be efficient. Use a curriculum while keeping policy input purely
visual.

### Suggested stages

1. **Control acquisition**
   - Move, turn, avoid walls, and use doors.
2. **Combat**
   - Aim, attack, manage ammunition, and survive encounters.
3. **Short navigation**
   - Complete short routes with one turn or door.
4. **Object interaction**
   - Collect required items or keys and operate doors.
5. **Partial MAP01 routes**
   - Reach successively later landmarks.
6. **Full MAP01**
   - Complete the entire level without privileged inputs.

Historical corridor code may be reusable, but it should be recovered or ported
carefully rather than assumed compatible with the new observation and recurrent
interfaces.

### Curriculum rules

- Privileged coordinates or labels may generate training rewards, curriculum
  stages, demonstrations, or diagnostics.
- Privileged information must not enter the deployed policy observation.
- Evaluation reward must be reported separately from training-only shaping.
- Each stage must have held-out scenarios or seeds.
- Promotion must use held-out success, not training reward.
- Preserve stage checkpoints so regressions can be diagnosed.

### Checkpoint ranking

Recommended lexicographic ranking:

1. Full completion rate.
2. Worst-seed completion rate.
3. Median route progress or exploration coverage.
4. Survival or death rate.
5. Mean unshaped task reward.
6. Mean shaped reward only as a final tie-breaker.

Do not rank primarily by shaped reward because the policy may exploit shaping
without completing the level.

## Phase 7: Full-scale training and evaluation

### Training protocol

1. Freeze code and configuration for each named experiment.
2. Use unique run IDs and never overwrite an existing run.
3. Train at least three independent seeds for credible comparisons.
4. Evaluate periodically on separate fixed seeds.
5. Reserve a final set of unseen seeds until model selection is complete.
6. Monitor process memory, replay memory, GPU memory, steps per second, spike
   rate, Q-value magnitude, loss, and action distribution.
7. Stop runs showing numerical instability, unbounded memory, or action
   collapse; preserve their diagnostics.

### Minimum final evaluation

- Multiple training seeds.
- At least 100 held-out episodes in total.
- Completion, death, and timeout rates.
- Per-seed and worst-seed metrics.
- Median and distribution of completion time.
- Kills, items, and exploration metrics.
- Action histograms and longest repeated-action streaks.
- Q-value margins and numerical stability.
- Spike rates per layer for spiking models.
- Representative successful and failed trajectories.
- Rendered visual inspection of selected episodes.

### Deployment candidate gate

A model becomes a software deployment candidate only when:

- Results are reproducible across independent runs.
- Completion generalizes to unseen seeds.
- No metric or outcome depends on privileged policy inputs.
- Checkpoint identity and configuration are recorded.
- A fresh process can load and evaluate the checkpoint.
- Representative failures are understood and documented.

## Phase 8: Optimization, deployment, or hardware work — deferred

**Project-owner direction (2026-10-10): do not work on deployment yet. Focus on
the model.** This section is retained only as future context and is not part of
the active task sequence.

Hardware deployment, pruning, and quantization should resume only after a
reliable FP32 full-level policy exists. The previous INT8 work is useful but
does not validate fixed-point spiking-neuron dynamics or a hardware runtime.

If hardware deployment returns to scope:

1. Freeze the software reference model and benchmark.
2. Define the actual target accelerator's supported neuron, accumulator,
   precision, and layer rules.
3. Quantize weights and neuron dynamics, not only stored tensors.
4. Compare FP32 and quantized policies on identical episode seeds and starts.
5. Measure outcome agreement, action-sequence agreement, numerical error,
   latency, throughput, memory, power, and resource use.
6. Treat an integer interchange file as an intermediate artifact, not a valid
   hardware deployment by itself.

## 6. Reporting structure

Every experiment should create an isolated directory:

```text
models/<model>/<run-id>/
    config.json
    latest.pth
    best.pth
    environment.json

runs/<model>/<run-id>/
    TensorBoard event files

reports/level/<model>-<run-id>-<timestamp>/
    report.json
    episodes.csv
    summary.md
    trajectory_*.png
```

`environment.json` should include:

- Git commit and dirty-worktree flag.
- Python, PyTorch, snnTorch, and ViZDoom versions.
- Operating system and CUDA/GPU information.
- Scenario/WAD identity where practical.
- Model parameter count.
- Exact command line.

`report.json` should distinguish:

- Training configuration.
- Checkpoint identity.
- Aggregate metrics.
- Per-seed metrics.
- Per-episode results.
- Raw versus shaped reward.
- Terminal reason.

## 7. Testing strategy

Use four levels of testing:

### Unit tests

Pure functions, reward components, terminal classification, replay behavior,
checkpoint metadata, and tensor contracts.

### Mocked integration tests

Environment behavior under controlled ViZDoom states without launching the
engine.

### Real-engine smoke tests

Short deterministic scenarios that prove configuration, action mapping,
terminal outcomes, and reporting with ViZDoom itself.

### Training regression tests

Very short CPU-compatible runs that exercise optimization, saving, resuming,
evaluation, and diagnostic generation. These tests prove mechanics, not
learning quality.

Tests should avoid depending on historical generated artifacts unless the
artifact is an explicit immutable fixture.

## 8. Recommended immediate task order

The next agent should work in this order. Completed stabilization items are
retained so the implementation history is explicit:

1. [x] Inspect Git state and preserve the migration boundary.
2. [x] Fix the missing `basic.cfg` path/test inconsistency.
3. [x] Correct terminal classification and terminal diagnostics.
4. [~] Validate real-engine death, timeout, and completion; only real
   completion remains.
5. [x] Correct TBPTT gradient flow and add boundary tests.
6. [x] Advance hidden state during exploration.
7. [x] Redesign replay capacity and measure its stored observation bytes.
8. [x] Complete checkpoint resume metadata and save behavior.
9. [x] Update active model documentation. Deployment is deferred.
10. [x] Run the complete test suite.
11. [x] Execute train-save-resume-diagnose smoke workflow.
12. [x] Implement compatible four-channel CNN and parameter-matched DRQN
    baselines.
13. [ ] Run controlled FFNN/DRQN/CRSNN/RSNNv2 learning probes.
14. [ ] Decide the curriculum using evidence from those probes.

## 9. Commands to use after stabilization

These reflect the current intended interfaces. Re-verify them after making
changes.

```powershell
# Tests
python -m pytest -q -p no:cacheprovider

# Inspect training arguments
python src/train_level.py --help

# Inspect diagnostic arguments
python src/diagnose_level.py --help

# Example training command; do not run at full scale until all gates pass
python src/train_level.py `
  --model crsnn `
  --run-id crsnn-map01-seed0 `
  --episodes 100 `
  --seed 0 `
  --device auto

# Example diagnostic command
python src/diagnose_level.py `
  --model crsnn `
  --checkpoint models/crsnn/crsnn-map01-seed0/best.pth `
  --episodes 20 `
  --seeds 10 11 12 13 14 `
  --device auto
```

Do not use the older documented `--stage full` argument unless that interface
is deliberately reintroduced and tested.

## 10. Open decisions requiring project-owner confirmation

The following decisions should be confirmed before their consequences become
expensive:

1. Is the corridor implementation intentionally being replaced, or should it
   remain as the curriculum environment?
2. Is FPGA/PeraMorphIQ deployment still a project goal, or is the scope now
   software-only full-level autonomy?
3. Should the legacy combat pipeline remain runnable from the active branch?
4. Is FreeDoom2 MAP01 the fixed benchmark, or should the design support other
   maps and WADs immediately?
5. What hardware and wall-clock budget is available for multi-seed training?

Until answered, prefer reversible changes and keep compatibility where it does
not substantially complicate the corrected full-level pipeline.

## 11. Definition of project milestones

### Milestone A: Mechanically correct

- All tests pass.
- Terminal outcomes are correct.
- Recurrent gradients cross the intended sequence.
- Replay memory is bounded.
- Save/resume/evaluate works end to end.

### Milestone B: Learning demonstrated

- At least one model beats random behavior consistently.
- Results reproduce across multiple seeds.
- Metrics show genuine navigation or task progress rather than reward
  exploitation.

### Milestone C: Full-level capability

- A held-out completion rate is established on MAP01.
- Successful and failed behavior is documented.
- The best checkpoint can be loaded and reproduced from a fresh process.

### Milestone D: Deployment-ready software model

- Performance is stable across unseen seeds.
- Resource requirements and inference speed are known.
- Configuration, checkpoint identity, and evaluation report are archived.

### Milestone E: Hardware-ready model, if required

- Quantized neuron and weight behavior matches the software reference within
  an accepted tolerance.
- End-to-end gameplay is validated on the target runtime or hardware.

## 12. Final guidance for the next agent

Treat existing checkpoints, reports, recordings, and user changes as valuable
data. Do not clean or reorganize them without approval. Make small, testable
changes in dependency order. Correct measurement comes before optimization,
and a short verified run is more useful than a long run whose completion and
recurrent-learning signals are incorrect.

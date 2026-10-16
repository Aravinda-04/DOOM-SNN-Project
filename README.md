# Full-Level DOOM Recurrent SNN Agent

This project is building one recurrent spiking neural-network agent that can
complete FreeDoom2 MAP01 autonomously through ViZDoom. The active pipeline uses
four stacked grayscale frames, a fixed discrete action space, imitation
pretraining, and recurrent DQN fine-tuning.

The repository also contains checkpoints and reports from earlier combat,
hard-spawn, corridor-navigation, conventional neural-network, and quantization
experiments. Those artifacts are historical inputs; the active full-level
pipeline is `train_level.py` and `diagnose_level.py`.

## Current status

The core full-level environment, CRSNN, recurrent training mechanics, and
checkpoint handling are stabilized. The active implementation phase is adding
successful-play demonstration recording, CRSNN imitation pretraining,
demonstration-guided RL fine-tuning, and a standalone autonomous-play command.

No completed full-level CRSNN benchmark has been established yet. Do not treat
historical combat or corridor results as evidence that MAP01 is solved.

See [AUTONOMOUS_SNN_IMPLEMENTATION_PLAN.md](AUTONOMOUS_SNN_IMPLEMENTATION_PLAN.md)
for the active single-agent implementation plan. The older
[FULL_LEVEL_PROJECT_PLAN.md](FULL_LEVEL_PROJECT_PLAN.md) is retained as
stabilization history.

## Active architecture

```text
FreeDoom2 MAP01
    -> ViZDoom
    -> 4 x 84 x 84 grayscale observation
    -> CRSNN
    -> 22 Q-values
    -> atomic movement/combat controls, common combinations, and no-op
```

The recurrent hidden state is carried across game frames. During replay
training, gradients remain connected within the sampled sequence and reset at
the explicit sequence boundary.

## Requirements

- Python 3.13 is the currently tested interpreter.
- `freedoom2.wad` must be present in the repository root.
- The active scenario is `scenarios/level01.cfg`.

Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Validate before training

Run the headless model/environment contract check:

```powershell
python src/train_level.py --model crsnn --device auto --validate-only
```

This initializes ViZDoom, resets MAP01, builds the requested model, and verifies
the `(4, 84, 84)` observation and 22-action Q-value contract. It does not
create a training run.

## Validate manual completion and actions

Before permanent demonstrations are collected, complete MAP01 once through the
same observation, frame-skip, reward, and terminal pipeline used for training:

```powershell
python src/manual_level_validation.py
```

The game opens in ViZDoom spectator mode. Current local bindings use arrow keys
for forward/backward/turning, comma and period for strafing, left control or
mouse button 1 for attack, and space for use. You can play naturally: the
catalogue includes the common movement, turning, attacking, and use
combinations needed for MAP01.

The command creates a timestamped JSON report under
`reports/manual-validation/`. Each four-tic decision is labelled by a 75%
majority, so a one-tic key transition is tolerated. Ambiguous or uncommon
combinations are masked rather than given a false label. A valid report
requires a real MAP01 exit and at least 95% valid label coverage. Exit code `0`
means the action contract passed, `1` means the level was not completed, `2`
means label coverage was too low, and `3` means an engine error occurred.

The frozen contract is defined in `src/agent_config.py`. Its hash is stored in
new training configurations and validation reports so demonstrations and
checkpoints cannot silently use a different output ordering.

## Tests

```powershell
python -m pytest -q -p no:cacheprovider
```

The current suite contains 121 passing tests, including real-engine timeout,
death, and headless spectator-path checks. A real successful exit remains the
manual Phase 0 acceptance gate.

## Record successful demonstrations

Record human playthroughs for CRSNN imitation training with:

```powershell
python src/record_demonstrations.py `
  --discard-failures `
  --notes "successful MAP01 routes"
```

The recorder captures one playthrough and exits, allowing you to rest or review
the result before starting another. Run the command again for each attempt.
With `--discard-failures`, only completed episodes with at least 95% valid
labels are retained. Play normally with the supported combinations. Short
boundary transitions are resolved by majority vote; unresolved decisions are
masked and ignored during imitation learning. Repeat until the manifest has at
least 10 successful episodes. Use `--max-ambiguous-fraction` to change the 5%
default only for an explicitly documented dataset.

The default dataset directory is `data/demonstrations/map01/`. Each episode is
an independent compressed `.npz` containing aligned pre-action uint8 frame
stacks, action indices, per-step label-valid masks, rewards, terminal flags,
positions, and game times.
`manifest.json` stores episode checksums and the frozen agent-contract hash.
Datasets are ignored by Git because recordings can become large.

## Imitation pretraining

After at least two successful demonstrations exist, validate the complete
dataset/model contract without creating a run:

```powershell
python src/train_imitation.py `
  --dataset-dir data/demonstrations/map01 `
  --device cuda `
  --validate-only
```

At least 10 successful demonstrations are recommended before real training.
The RTX 2050 starting configuration is:

```powershell
python src/train_imitation.py `
  --dataset-dir data/demonstrations/map01 `
  --run-id crsnn-map01-imitation `
  --epochs 50 `
  --batch-size 4 `
  --sequence-length 16 `
  --burn-in 4 `
  --stride 16 `
  --gradient-accumulation-steps 4 `
  --closed-loop-interval 5 `
  --closed-loop-episodes 1 `
  --device cuda `
  --amp
```

The trainer splits complete episodes rather than frames, applies no-gradient
burn-in before each recurrent learning window, balances rare actions, and saves
resumable `best.pth` and `latest.pth` checkpoints. Resume with the same data and
hyperparameters while increasing `--epochs`:

```powershell
python src/train_imitation.py `
  --dataset-dir data/demonstrations/map01 `
  --run-id crsnn-map01-imitation `
  --epochs 100 `
  --batch-size 4 `
  --sequence-length 16 `
  --burn-in 4 `
  --stride 16 `
  --gradient-accumulation-steps 4 `
  --closed-loop-interval 5 `
  --closed-loop-episodes 1 `
  --device cuda `
  --amp `
  --resume models/crsnn/crsnn-map01-imitation/latest.pth
```

## Training

Start with a short mechanical run, not the full 2,000-episode target:

```powershell
python src/train_level.py `
  --model crsnn `
  --run-id crsnn-map01-smoke-seed0 `
  --episodes 5 `
  --episode-timeout 140 `
  --eval-interval 5 `
  --eval-episodes 1 `
  --eval-seeds 10 `
  --seed 0 `
  --device auto
```

After smoke validation, a normal run can use:

```powershell
python src/train_level.py `
  --model crsnn `
  --run-id crsnn-map01-seed0 `
  --episodes 2000 `
  --seed 0 `
  --device auto
```

The canonical model for the active objective is:

| Model | Role | Trainable parameters |
| --- | --- | ---: |
| `crsnn` | Dense all-to-all recurrent spiking model | 1,910,918 |

Other architectures remain in the repository as historical or inactive code;
the current plan does not compare them. Run `python src/train_level.py --help`
for the presently implemented RL hyperparameters.

Replay capacity is measured in completed transitions, not episodes.
Observations are stored as `uint8` and decoded when sampled. The default latest
checkpoint interval is one episode.

Each run writes isolated artifacts:

```text
models/<model>/<run-id>/
    config.json
    latest.pth
    best.pth

runs/<model>/<run-id>/
    TensorBoard event files
```

`best.pth` is ranked by deterministic completion rate and then mean shaped
reward. `latest.pth` is the resumable training state. Replay memory is rebuilt
after resume.

Resume by setting the final target episode count:

```powershell
python src/train_level.py `
  --model crsnn `
  --run-id crsnn-map01-seed0 `
  --episodes 2500 `
  --resume models/crsnn/crsnn-map01-seed0/latest.pth `
  --device auto
```

## Diagnostics

Evaluate a checkpoint greedily without human control:

```powershell
python src/diagnose_level.py `
  --model crsnn `
  --checkpoint models/crsnn/crsnn-map01-seed0/best.pth `
  --episodes 10 `
  --seeds 0 `
  --device auto
```

Reports are written beneath `reports/level/` and include JSON, CSV, TensorBoard
metrics, and trajectory plots. They distinguish completion, death, timeout,
raw game reward, shaped reward, kills, exploration, Q margins, and spike
activity.

Use `--render --delay 0.05` with a small episode count for visual inspection.

## TensorBoard

```powershell
tensorboard --logdir runs
```

Training logs include loss, epsilon, episode length, spike activity, replay
episode count, and estimated replay observation memory.

## Important limitations

- Full MAP01 learning has not yet been demonstrated with the new CRSNN.
- Position-based exploration rewards are training-only privileged signals;
  coordinates are not policy inputs.
- Historical INT8 results do not validate current full-level behavior or a
  hardware runtime.
- Deployment and hardware work are intentionally deferred; current work is
  model-focused.
- Demonstration recording and imitation-training code are implemented, but no
  real successful dataset or trained MAP01 checkpoint exists yet. RL
  demonstration fine-tuning and `play_level.py` remain.

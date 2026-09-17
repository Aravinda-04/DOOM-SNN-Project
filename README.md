# DOOM Spiking Neural Network Accelerator

This project trains neural networks to play the ViZDoom `basic` scenario. It
compares a conventional feed-forward CNN (FFNN), a spiking CNN (SNN), and a
recurrent spiking CNN (RSNN) before exporting integer weights for later FPGA
integration.

## Project status

Implemented:

- ViZDoom environment with normalized `84 x 84` grayscale observations.
- FFNN, SNN, and RSNN policies with a shared interface.
- DQN training with experience replay and soft target updates.
- Isolated experiment directories and TensorBoard logs.
- Greedy multi-episode validation for best-checkpoint selection.
- Resumable checkpoints and compatibility with legacy raw state dictionaries.
- Symmetric per-tensor integer export with numerical error reporting.

Still required for hardware deployment:

- Map the integer interchange file to the PeraMorphIQ vendor format.
- Validate supported layers, neuron dynamics, scales, and accumulator widths on
  the target board.
- Measure latency, power, and spike activity on hardware.

## Installation

Python 3.13 is the tested interpreter. Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

The tracked `basic.cfg` and `basic.wad` files must remain in the repository
root. Runtime paths are resolved from the source location, so commands can be
launched from the repository root, `src`, or an IDE.

## Training

Choose the architecture on the command line instead of editing source code:

```powershell
python src/train.py --model snn --episodes 200 --seed 0
python src/train.py --model ffnn --episodes 200 --seed 1
python src/train.py --model rsnn --episodes 200 --seed 2
```

For an SNN run with turning, anti-loop reward shaping, and difficulty-aware
multi-seed checkpoint selection, run this in Command Prompt:

```cmd
cd /d "E:\Side project"
python src\train.py --model snn --episodes 100 --seed 0 --run-id snn-search-seed0 --eps-end 0.2 --eps-decay 5000 --eval-interval 10 --eval-episodes 2 --eval-seeds 0 1 2 --easy-target-offset 0.1 --device auto
```

For balanced hard-left/hard-right fine-tuning with training-only aiming reward
shaping:

```cmd
python src\train.py --model snn --episodes 1000 --seed 0 --run-id snn-1000-balanced-seed0 --resume models\snn\deployment-candidates\snn-hardspawn-91pct.pth --eps-start 1.0 --eps-end 0.1 --eps-decay 10000 --eval-interval 25 --eval-episodes 3 --eval-seeds 0 1 2 3 4 --balanced-training --balanced-evaluation --max-spawn-attempts 1000 --easy-target-offset 0.1 --aim-progress-weight 20 --off-target-attack-penalty 4 --device auto
```

Balanced training alternates accepted hard-left and hard-right starts. Object
labels reward turns that reduce target offset and penalize off-target attacks,
but labels are never passed into the policy network. Balanced evaluation runs
equal samples in both directions, and checkpoint selection maximizes the worse
direction before considering aggregate success and reward.

The replay reward receives a small penalty after eight identical consecutive
actions and when a movement action makes no measurable progress. TensorBoard
continues to show the unmodified game reward. ViZDoom object labels identify
whether the monster starts near the center. Best-checkpoint selection first
maximizes hard-spawn success, then total kill rate and mean reward.

Useful options are available with:

```powershell
python src/train.py --help
```

Each experiment receives a unique run ID and writes to separate directories:

```text
models/<model>/<run-id>/best.pth
models/<model>/<run-id>/latest.pth
models/<model>/<run-id>/config.json
runs/<model>/<run-id>/
```

`best.pth` is selected using greedy multi-seed hard-spawn success, total success,
and then mean reward—not an exploratory training score. `latest.pth` is always
the newest resumable state.

The current action space is `move_left`, `move_right`, `turn_left`,
`turn_right`, and `attack`. Earlier three-action checkpoints require their
matching scenario configuration and cannot be resumed into a five-action model.

Resume a run by setting the final target episode count:

```powershell
python src/train.py --model snn --episodes 400 \
  --resume models/snn/<run-id>/latest.pth
```

The optimizer, episode number, global step, configuration, and best evaluation
score are restored. Replay memory is intentionally rebuilt and must pass the
warm-up threshold again.

## Evaluation

Evaluate the newest structured checkpoint, falling back to the legacy root
checkpoint when necessary:

```powershell
python src/eval.py --model snn
```

Select a particular checkpoint or disable rendering:

```powershell
python src/eval.py --model snn \
  --checkpoint models/snn/<run-id>/best.pth --episodes 10 --no-render
```

The checkpoint is validated before ViZDoom opens a window.

## SNN diagnostics

Run a repeatable 100-episode baseline across five ViZDoom seeds from Windows
Command Prompt:

```cmd
cd /d "E:\Side project"
python src\diagnose.py --model snn --checkpoint models\snn\snn-100-seed0\best.pth --episodes 20 --seeds 0 1 2 3 4 --device auto
```

The command evaluates greedily without rendering and creates a timestamped
folder under `reports/snn/`. Each report contains:

- `report.json`: complete metadata, aggregate and per-seed summaries, action
  traces, Q-value margins, and per-layer SNN spike activity.
- `episodes.csv`: one flat row per episode for plotting or spreadsheet review.
- `tensorboard/`: reward, success, episode length, stalled movement, Q-margin,
  and spike-rate curves.

Open the diagnostic graphs from Command Prompt with:

```cmd
tensorboard --logdir reports
```

The primary metric is hard-spawn success rate, followed by total success,
median reward, and consistency across seeds. A high wall-loop count, a dominant attack
fraction in failed episodes, repeated stalled movement, or a very small
Q-margin indicates a brittle policy even when its mean reward looks good.

To execute only genuinely hard spawns, use:

```cmd
python src\diagnose.py --model snn --checkpoint models\snn\<run-id>\best.pth --episodes 20 --seeds 10 11 12 13 14 --spawn-filter hard --easy-target-offset 0.1 --max-spawn-attempts 1000 --device auto
```

`--episodes` is the number of accepted episodes per seed. Easy resets are
skipped and reported rather than counted. More specific filters are `left`,
`right`, and `invisible`; `all` remains the default. The maximum-attempt limit
prevents an impossible or rare filter from searching forever.

## TensorBoard

```powershell
tensorboard --logdir runs
```

Training and evaluation metrics are separated by architecture and run ID.

## Integer export

Export real integer tensors and per-tensor scales, then measure the output error
introduced by quantization:

```powershell
python src/export_quantized.py --model snn \
  --checkpoint models/snn/<run-id>/best.pth \
  --bits 8 --output exports/snn-int8.pt
```

The result is an integer interchange file, not a vendor-specific FPGA image.

Evaluate the exported integer tensors end to end in ViZDoom after dequantized
reconstruction:

```cmd
python src\diagnose.py --model snn --quantized-export exports\snn-balanced-hardspawn-99pct-int8.pt --episodes 20 --seeds 10 11 12 13 14 --spawn-filter hard --easy-target-offset 0.1 --max-spawn-attempts 1000 --device auto
```

Add `--render --delay 0.05` with fewer episodes for visual verification. Create
a side-by-side report chart with `src/compare_reports.py`; run `--help` for its
report paths and labels.

## Architecture note

The SNN and RSNN currently use deterministic direct coding: continuous pixels
enter the first convolution and LIF neurons generate the first spikes. The RSNN
recurrent state spans the internal neural simulation steps for one observation;
it does not persist across game frames. A true cross-frame recurrent agent would
require sequence replay and explicit hidden-state handling.

## Tests

```powershell
python -m pytest
```

The test suite checks project-root paths, network construction and output
contracts, legacy and structured checkpoints, and integer quantization.

## Legacy artifacts

Earlier runs created both root-level and `src/`-level `models` and `runs`
directories because paths depended on the terminal working directory. They are
left untouched to prevent accidental loss. New code only reads and writes the
root-level structured layout shown above.

See `changes.md` for the complete migration notes.

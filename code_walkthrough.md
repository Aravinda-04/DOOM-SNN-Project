# DOOM SNN project: code walkthrough

## 1. Project paths

`src/project_paths.py` anchors the scenario, checkpoints, and logs to the
repository root. The process working directory no longer changes where files
are read or written.

New experiments use two matching directories:

```text
models/<model>/<run-id>/
runs/<model>/<run-id>/
```

The first contains checkpoints and `config.json`; the second contains
TensorBoard events.

## 2. Environment

`src/env.py` connects Python to ViZDoom. It loads the tracked `basic.cfg`,
selects a `160 x 120` grayscale screen, and exposes three actions: move left,
move right, and attack.

Each raw frame is resized to `84 x 84` and normalized to `[0, 1]`. One agent
action is held for four game tics. Configuration errors are raised immediately,
and callers close the game through a context manager or `finally` block.

## 3. Network registry

`src/networks/__init__.py` maps the CLI names `ffnn`, `snn`, and `rsnn` to their
classes. Training, evaluation, and export all use this registry, so an invalid
name fails before a run starts.

All models accept `(batch, 1, 84, 84)` tensors and return `(q_values,
spike_count)`. The shared contract keeps the DQN code independent of the chosen
architecture.

### FFNN

`src/networks/ffnn.py` is the conventional baseline. Three convolutional ReLU
layers feed a 512-unit fully connected layer and three Q-values. Its spike count
is zero.

### SNN

`src/networks/snn.py` replaces ReLU activations with leaky integrate-and-fire
neurons. The current implementation uses deterministic direct coding: the
normalized image enters the first convolution directly, and the first LIF layer
produces spikes.

The same observation is processed over ten internal neural steps. Hidden-layer
spikes are counted for the sparsity penalty, while output membrane potentials
form the Q-values.

### RSNN

`src/networks/rsnn.py` adds a recurrent connection around the 512-unit spiking
layer. That recurrent state spans the ten internal steps of one observation.
It resets on the next call, so it is not cross-frame memory. Supporting true
episode-level recurrence would require replaying contiguous sequences with
explicit hidden states.

## 4. Training

`src/train.py` implements DQN with:

- Epsilon-greedy exploration.
- Experience replay.
- A policy and soft-updated target network.
- Huber temporal-difference loss.
- Gradient clipping and an optional weight stability clamp.
- A spike-count sparsity penalty for spiking architectures.

Every configured interval, training pauses for greedy evaluation episodes.
Their mean reward selects `best.pth`, preventing a lucky exploratory episode
from freezing the checkpoint. `latest.pth` is saved every episode and includes
optimizer and progress state for resuming.

Replay memory is deliberately not checkpointed because image buffers can be
very large. A resumed run repeats replay warm-up.

## 5. Evaluation

`src/eval.py` validates and loads a checkpoint before starting ViZDoom. It uses
greedy actions, reports every episode reward, and prints the mean and standard
deviation. Rendering can be disabled for automated evaluation.

## 6. Integer export

`src/quantization.py` converts floating tensors into symmetric per-tensor
integers and records a scale for reconstruction. `src/export_quantized.py`
exports 4-, 8-, or 16-bit tensors and compares the original model output with a
dequantized reconstruction.

This is a concrete integer interchange format. A separate adapter is still
needed to satisfy the PeraMorphIQ FPGA toolchain and hardware accumulator rules.

## 7. Tests

The `tests/` directory checks path behavior, the common network interface,
model-name validation, new and legacy checkpoint loading, and quantization
bounds. Run it with `python -m pytest`.

# Corridor software milestone

This is a separate, experimental task. The archived combat policy is preserved.
No corridor policy is currently claimed to be trained to reliable completion,
and neither network has a verified PeraMorphIQ hardware mapping.

## Scenario and curriculum

Scenario defaults are defined in `scenarios/corridor.cfg`, just as the original
task uses `basic.cfg`. It declares the WAD path, initial map, button order, game
variables, screen settings, difficulty, and episode timing. The file is tracked
through an explicit `.gitignore` exception. Relative WAD paths are resolved from
the configuration location, not the terminal's working directory.

Python selects the stage/case, seed, rendering/debug options and action
combinations, and computes task outcomes and rewards. The environment validates
the configured button order, required variables and grayscale format before
starting the engine. `--timeout` overrides the configured timeout only when
explicitly supplied; otherwise the `.cfg` value applies to training and evaluation.
The generated WAD still contains geometry; configuration files do not define maps.

`src/corridor_env.py` generates a UDMF WAD locally, with no map editor or downloaded
assets. ViZDoom supplies its normal game textures and actors. The generated WAD
is ignored by Git and regenerated if absent. Source version is recorded in checkpoints.

The map is an L-shaped corridor, mirrored for left and right turns. Navigation
episodes contain no enemies. Combat episodes contain two former humans at
different positions, including opposite lateral offsets. Cases 0-7 are training
cases, 8-15 are validation cases with changed enemy positions. Navigation uses
the same two mirrored geometries across splits: it is NOT unseen-map testing.
Changing seeds changes engine randomness, not the map geometry.

The objective is reaching the far-end region (x >= 540 and mirrored y >= 640)
alive; combat also requires both kills. A decorative marker marks the far end. This is an
environment-defined exit region, not a door or an actual Doom exit switch.
Forward movement and turning are required. There are no doors, pickups or
additional obstacles in this first milestone. Failure is death or timeout.

Seven discrete actions: forward, strafe left/right, turn left/right, attack,
and forward+attack. Observations remain 84x84 grayscale pixels. Position, health,
ammo and kill counters are used for task scoring, diagnostics and training-only
progress shaping; they are not policy inputs. The debug map is viewer-only.

Navigation now permits only the first five movement/turn actions. The seven-output
network is retained so its weights can initialize the combat stage, which permits
all seven. Exploration, greedy selection, Bellman targets, evaluation and the
environment all enforce the stage-specific action set. New checkpoints record
their allowed actions. Older seven-action navigation checkpoints are still
readable for historical comparison and retain their original action set.

Task reward is -0.04 per decision, +10 per kill, +100 for completion, -25 for
death or timeout, and in navigation only, -0.2 for each ongoing decision after
12 decisions without new route progress. Training adds discounted potential-based progress shaping
with gamma 0.99, coefficient 20 and zero terminal potential; evaluation never
adds it to reported reward. DQN training divides the resulting reward by 25.
All reward components are recorded separately in TensorBoard; evaluation JSON
records the task components and maximum route progress for each episode.
Reports show completion, death, timeout, kills, shots, stalls and trajectories.
Shots are estimated from ammunition decreases; simultaneous ammunition pickups
can cause this estimate to undercount.
Stalls count movement commands with displacement below 0.5 world units, not
proven wall loops. Each decision lasts four game tics.

## Windows CMD workflow

Run from the project directory:

```cmd
cd /d "E:\Side project"
```

Verify map mechanics visually. This controller uses privileged coordinates and
is explicitly a SCRIPTED ENV TEST; it is not a trained SNN demonstration:

```cmd
python src\corridor.py smoke --episodes 8 --render
```

Train a small navigation SNN from scratch using a fresh run ID. The v4
experiment first imitates a coordinate-guided teacher, then gathers corrective
labels on frames reached by the learner (DAgger-style). Corrective rollouts
alternate left/right cases; the teacher takes 20% of actions to keep some
rollouts recoverable. Actions and route sides are balanced when sampling expert
frames. During DQN updates, a weighted expert classification loss rehearses
these labels, reducing the chance that RL immediately forgets a turn.
Coordinates supply labels and scoring only: SNN inputs remain 84x84 images.
This is a controlled experiment, not a proven navigation solution. Inspect
bootstrap and each DAgger/validation chart before spending time on more RL.

```cmd
python src\corridor.py train --stage navigation --model compact_snn --episodes 100 --run-id corridor-nav-dagger-v4-seed0 --demo-episodes 16 --demo-updates 300 --dagger-rounds 2 --dagger-episodes 8 --dagger-updates 200 --dagger-max-steps 120 --dagger-teacher-prob 0.2 --expert-loss-weight 0.25 --epsilon-start 0.3 --device auto
```

Set `--dagger-rounds 0 --expert-loss-weight 0` to ablate corrective training,
or `--demo-episodes 0 --epsilon-start 1` for pure RL. Do not reuse an existing
run ID. The v3 one-sided/failed results remain archived for comparison.

Evaluate the selected checkpoint, including the live dashboard and saved video:

```cmd
python src\corridor.py eval --stage navigation --checkpoint models\corridor\corridor-nav-dagger-v4-seed0\best.pth --episodes 40 --seed 20000 --render
```

Once navigation works, initialize a separate combat run from that checkpoint:

```cmd
python src\corridor.py train --stage combat --model compact_snn --episodes 1000 --run-id corridor-combat-dagger-v4-seed0 --init-checkpoint models\corridor\corridor-nav-dagger-v4-seed0\best.pth --device auto
python src\corridor.py eval --stage combat --checkpoint models\corridor\corridor-combat-dagger-v4-seed0\best.pth --episodes 16 --seed 20000 --render
```

Use `--model snn` to train the existing larger architecture from scratch with
seven outputs for comparison. Five-action basic checkpoints are rejected.
`--init-checkpoint` is a weight warm start, NOT optimizer/replay/RNG resume.
Existing run IDs and evaluation output directories are rejected, preventing
accidental overwrite. Model and number of neural steps must match on warm start.

Models: `models/corridor/<run-id>/{latest,best}.pth`.
Logs: `runs/corridor/<run-id>`; launch `tensorboard --logdir runs/corridor`.
When demonstrations are used, `bootstrap.pth`, `bootstrap.json` and
`bootstrap.png` show behavior before corrective collection or DQN updates.
Each corrective round saves `dagger-N.pth`, `dagger-N.json/png`,
`dagger-N-turns.png`, and `dagger-N-collection.json`.
`best.pth` initially points to the bootstrap and changes only when later
validation improves. Validation
every 25 episodes uses all eight evaluation cases, seed 10000;
best selection orders worst-direction completion first, then overall completion,
then mean task reward. This prevents a one-sided policy from outranking a model
that completes some routes in both directions.
These cases are validation data, not a pristine test set. Use additional seeds
for final evaluation, and introduce additional layouts before claiming map generalization.

Every standalone eval/smoke saves `report.json`, `summary.png`, `dashboard.png`
and `gameplay.avi` beneath a unique `reports/corridor/` directory. Navigation
evaluation also saves `turn-decisions.png`, showing the SNN's first-bend action
accuracy and Q-value margin against the scripted direction when the SNN first
turns at the bend. If a route never turns there, it is marked as missing rather
than counted as a correct decision. A later turn can still succeed, so this is
a diagnostic, not the completion metric. The dashboard
shows gameplay, policy input, action, Q-values, layer spike rates and a debug map.
Video is 8.75 decisions/second. Q-values are not action probabilities; measured
inference latency is PC model-forward time, not FPGA or full-loop latency.
Press Q in the dashboard to stop early (partial video only; no completed report).

## Compact SNN and profiling

The experimental compact model pools to 16x16, then uses 256 -> 128 -> actions
with LIF neurons and ten neural steps. It resets neuron state between decisions,
so it has no persistent navigation memory. Lower size does not imply equal accuracy.
It uses analog input, beta=0.9 and a membrane readout. Pooling, input encoding,
leakage, bias and output interpretation still need hardware agreement.

`compact_conv_snn` is an optional small convolutional comparison if the pooled
model still struggles with a turn direction. It has fewer weights but many more
LIF state values and operations. Neither compact model has confirmed FPGA support.
Compare equal training budgets and the same evaluation cases before selecting one.

```cmd
python src\profile_models.py
python -m pytest
```

The resource report compares weights, LIF state values, hypothetical storage and
dense MAC estimates, with a PNG chart. It is not a synthesis report. The existing
`train.py --model compact_snn` also permits a fresh compact model on the original
combat task; use a new run ID and the same validation settings as the baseline.
Do not load the large SNN weights into the compact architecture.

The first practical acceptance gate is repeated left/right navigation completion,
followed by combat completion across seeds with saved failure videos. A long
training run is not a guarantee of convergence. Full-game training and final FPGA
export remain later work.

## Verification performed (2026-09-19)

- 24 tests passed, including real-engine loading of all 32 maps, enemy counts,
  timeout behavior, terminal guards, checkpoint rejection and compact gradients.
- Scripted navigation: 8/8 evaluation cases completed, 52 decisions each, zero
  stalled moves. This proves traversability only, not learned navigation.
- Compact training smoke: two navigation episodes and two combat warm-start
  episodes, using a deliberately short 140-tic timeout. Optimizer state confirms
  15 updates in the combat run and all saved parameter values are finite.
- Saved navigation policy reloaded and produced a dashboard/video successfully.
- Combat policy evaluated at the normal 1400-tic timeout on two cases: one death,
  one timeout, one kill per episode, zero completions. It is not a trained deliverable.
- All three saved gameplay videos were opened and decoded successfully; dashboard
  and summary images were visually inspected.
- Seven-action weight counts: existing SNN 1,636,864; compact SNN 33,664.
  LIF state counts: 12,647 and 135 respectively, excluding input encoding.

Local artifacts (ignored by Git):

- `reports/corridor/environment-check-final/`: scripted traversal video and plots.
- `reports/corridor/snn-navigation-smoke/`: reloaded navigation SNN dashboard/video.
- `reports/corridor/snn-combat-smoke/`: combat failure video, plot and detailed JSON.
- `reports/model-profile/`: resource comparison PNG and JSON.
- `models/corridor/corridor-smoke-v1/`: navigation smoke checkpoints.
- `models/corridor/corridor-combat-smoke-v1/`: combat smoke checkpoints.

The archived 200-episode compact run reached 3/8 at episode 150 (left turns only)
and 0/8 at episode 200, with attack/turn loops. This motivates the v2 action and
reward changes. The current eight validation cases repeat the same two mirrored
navigation geometries; even 40 episodes on these maps cannot establish unseen
layout generalization. Add genuinely different maps before claiming that result.
Aim for at least 90% overall completion and 80% per direction on a larger
evaluation before starting combat training. A gate is an engineering target,
not evidence of current performance.

Compare an archived validation report against a new one. The chart omits reward
because the reward definitions changed:

```cmd
python src\compare_corridor.py --baseline reports\corridor\corridor-nav-small-seed0\validation-150.json --candidate reports\corridor\corridor-nav-demo-v3-seed0\validation-200.json --output reports\corridor\nav-v1-v3-comparison.png
```

Do not interpret the smoke checkpoints as candidates for deployment.

## Navigation follow-up (2026-09-20)

- The full suite passed 34 tests, including a test that balanced checkpoint
  selection outranks one-sided validation success.
- A 16-demonstration/300-update pooled SNN bootstrap completed 4/8 validation
  routes, all left. After a one-episode RL update it still completed 4/8 left.
  The convolutional comparison also completed 4/8 left after its first update.
- Warm-starting the pooled SNN and training one left and one right route changed
  the successful direction: 4/8 right, 0/4 left. The 50% overall figure is not
  reliable navigation.
- Increasing the pooled SNN imitation phase to 1000 updates did not fix it:
  the bootstrap and one-episode RL follow-up both completed 0/8. Its trajectories
  orbit the first corridor and never enter the second leg, despite using both
  turn actions.
- The mirrored starting observations differ in only 192/7056 pixels; this
  measurement does not prove visual ambiguity later in the route, but it shows
  that the correct turn must be inferred from later observations.
- The comparison and bootstrap charts are at
  `reports/corridor/nav-archived-v3-two-sided-comparison.png` and
  `reports/corridor/corridor-nav-demo-v3-probe1000-seed0/bootstrap.png`.
  `reports/corridor/nav-demo-300-vs-1000.png` directly compares the two
  imitation budgets.

The navigation gate remains unmet. Before combat or a 1000-episode run, the
next experiment should collect corrective teacher labels on states the learner
actually visits (rather than only ideal demonstration paths), and test whether
both turn directions improve on repeated validation and new layouts.

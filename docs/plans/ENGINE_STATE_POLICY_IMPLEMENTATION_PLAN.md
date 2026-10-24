# Engine-State MAP01 Policy Implementation Plan

## Implementation status (2026-10-19)

The first working vertical slice is implemented:

- versioned engine-state and factorized-control contracts;
- direct validated button execution in the MAP01 environment;
- route extraction from successful demonstrations;
- causal state extraction and visible-enemy geometry;
- a scripted expert that completes MAP01;
- compact generated sequence storage and manifest validation;
- expert, perturbed, and learned-policy/DAgger case generation;
- a recurrent GRU policy with four control heads;
- CUDA/AMP training, checkpointing, resume, and closed-loop evaluation;
- a deterministic combat, interaction, recovery, and confidence shield.

The current best confidence-shielded GRU completed MAP01 in 913 decisions with
14 kills and 76 regions visited using a `0.92` confidence threshold. The shield
intervened on 74.8% of decisions. The unshielded GRU has not completed MAP01,
so this is a working privileged hybrid baseline rather than a fully learned
policy. Reducing shield dependence and adding the spiking-policy comparison
remain future phases.

## Objective

Build an agent that completes the full FreeDoom MAP01 level using structured
state read directly from ViZDoom instead of learning from rendered images.

The agent will receive normalized engine telemetry, route context, and visible
object information. It will predict movement, turning, attack, and use controls
through separate output heads. Training cases will be generated automatically
from scripted rollouts and controlled perturbations rather than requiring more
human video or image annotation.

This is a **privileged-state agent**, not a vision-only agent. Its results must
be reported separately from visual-agent results.

## Why this approach is useful

- It removes image preprocessing and visual perception as sources of error.
- It makes enemy direction, navigation progress, and stuck detection explicit.
- It permits balanced generation of combat, door, corner, and recovery cases.
- A small state-vector policy trains and runs much faster than the CRSNN image
  encoder.
- It establishes whether the control, route, and recovery logic can solve the
  level before visual perception is attempted.

## Target runtime pipeline

```text
ViZDoom game variables + visible-object labels + route manager
                           |
                           v
                 normalized state vector
                           |
                           v
                 recurrent control policy
                           |
          +----------------+----------------+
          |                |                |
     locomotion          turning        attack/use
                           |
                           v
                validated button composer
                           |
                           v
                        ViZDoom
```

The game window may still be rendered for human inspection, but rendered
pixels are not policy inputs.

## Scope

The initial system targets the complete MAP01 route from spawn to exit,
including:

- navigation through the entire map;
- combat and shooting while moving or turning;
- door and switch interaction;
- recovery after collisions or missed turns;
- health and ammunition awareness;
- death, timeout, and completion handling.

Supporting other maps is a later project. MAP01-specific route data and
normalization bounds must be explicitly versioned rather than presented as a
general Doom solution.

## Proposed engine-state contract

The final fields will be selected only after an engine audit. Candidate fields
are grouped below.

### Player state

- normalized position X and Y;
- heading represented as sine and cosine, not a discontinuous degree value;
- recent X/Y displacement and angular displacement;
- normalized health, armor, bullet ammunition, and shell ammunition;
- recent damage, healing, ammunition, kill, and item deltas;
- elapsed episode time and remaining timeout fraction;
- causal stuck duration derived from recent displacement.

### Visible combat state

- whether any enemy is visible;
- visible enemy count;
- nearest visible enemy distance;
- nearest visible enemy relative bearing represented as sine/cosine;
- left, centre, and right enemy-presence flags;
- whether the nearest enemy is inside a configurable firing cone;
- recent attack command and recent kill confirmation.

ViZDoom object labels describe visible objects. The implementation must not
pretend they provide positions for enemies hidden behind geometry unless a
separate map-level source is intentionally added and documented.

### Route and interaction state

- current route waypoint identifier;
- relative distance and bearing to the next waypoint;
- route progress fraction;
- whether the waypoint is an interaction point;
- whether a use action is currently allowed or recommended;
- recent waypoint transition and route-regression indicators;
- optional key or inventory state only if it is reliably exposed by the
  selected ViZDoom configuration.

### Validity

Every conditionally available field must have a corresponding validity bit.
For example, enemy distance is invalid when no enemy is visible. Invalid
values must be masked, not silently treated as genuine zeros.

## Factorized control contract

The new policy will not predict one of the existing 22 compound actions.
Instead it will predict:

- locomotion: `none`, `forward`, `backward`, `strafe_left`, `strafe_right`;
- turning: `none`, `left`, `right`;
- attack: `false` or `true`;
- use: `false` or `true`.

A single validated composer will turn these predictions into the configured
ViZDoom button vector. It must reject opposing or unsafe combinations. This
lets all shooting cases train the same attack output regardless of simultaneous
movement.

## Phase 0 - Preserve the visual baseline

### Work

- Keep the engine-state system in separate modules, datasets, checkpoints, and
  reports.
- Record the existing visual policy's per-action recall, kills, regions,
  completion, and stuck duration as comparison metrics.
- Add an explicit `observation_source=engine_state` field to every new
  checkpoint and report.

### Exit gate

- A checkpoint from one observation type cannot be loaded by the other policy.

## Phase 1 - Audit available engine state

### Files

- Add `src/inspect_engine_state.py`.
- Save an audit report under `reports/engine-state-audit/`.

### Work

- Enumerate the game variables exposed by `scenarios/level01.cfg`.
- Inspect object labels throughout representative combat and door situations.
- Determine whether keys, weapons, doors, switches, and enemy identities are
  reliably available.
- Establish valid MAP01 coordinate bounds and heading conventions.
- Measure whether labels remain stable at the four-tic decision rate.

### Exit gate

- Every proposed input field has a documented engine source, normalization,
  missing-value rule, and alignment point.
- Fields not reliably exposed are removed or supplied by an explicitly
  documented MAP01 route module.

## Phase 2 - Define contracts and state extraction

### Files

- Add `src/engine_state_config.py` containing field names, normalization
  constants, validity semantics, contract version, and contract hash.
- Add `src/engine_state_observation.py` for causal history and vector creation.
- Extend `src/doom_level_env.py` with a read-only structured-state method while
  preserving the existing image observation API.
- Add `tests/test_engine_state_observation.py`.

### Requirements

- State must describe the observation before the selected action.
- History-derived fields may use only current and previous engine states.
- Angle wrapping must be tested around zero and 360 degrees.
- All outputs must be finite and normalized to documented ranges.
- Reset must clear displacement, damage, previous-action, and stuck history.

### Exit gate

- Mocked engine states produce exactly the expected vector and validity mask.
- State extraction remains deterministic at episode reset and terminal states.

## Phase 3 - Build and validate the MAP01 route representation

### Files

- Add `src/map01_route.py` with versioned waypoints, connections, interaction
  points, completion conditions, and recovery anchors.
- Add `src/validate_map01_route.py`.
- Add `tests/test_map01_route.py`.

### Work

- Define waypoints in walkable areas from spawn to exit.
- Mark doors, switches, narrow turns, and combat-heavy areas.
- Compute the next waypoint and relative bearing from player position.
- Add hysteresis so waypoint selection does not oscillate near boundaries.
- Define re-entry behavior for off-route positions.

### Exit gate

- A deterministic scripted waypoint follower can traverse the route without a
  learned policy in an enemy-disabled or controlled validation run.
- Starting from tested off-route recovery positions rejoins the route.

The learned policy should not be asked to compensate for a broken route graph.

## Phase 4 - Implement a scripted expert

### Files

- Add `src/engine_state_expert.py`.
- Add `src/evaluate_engine_state_expert.py`.
- Add `tests/test_engine_state_expert.py`.

### Expert priorities

1. Detect terminal and unsafe health conditions.
2. Align with and attack a visible threat when combat is appropriate.
3. Use an interaction point when aligned and in range.
4. Recover when displacement remains below the stuck threshold.
5. Turn toward and move toward the next route waypoint.

The expert must output the same factorized targets used by the learned policy.
Rules and thresholds must be configuration values rather than scattered magic
numbers.

### Exit gate

- Unit scenarios cover enemy-left, enemy-centre, enemy-right, no-ammo, door,
  corner, stuck, and normal waypoint-following cases.
- The scripted expert completes MAP01 in repeated visible evaluations before it
  is used as the main label generator.

If the expert cannot complete the level, generated labels are not trustworthy.

## Phase 5 - Generate sequence training cases

### Files

- Add `src/generate_engine_state_cases.py`.
- Add `src/engine_state_demonstrations.py` with a new independent schema.
- Store generated data under `data/generated/map01-engine-state/`.
- Add schema and temporal-alignment tests.

### Generation modes

- Full expert rollouts from spawn to exit.
- Perturbed rollouts that occasionally delay, reverse, or alter a turn and then
  ask the expert to recover.
- Combat-focused rollouts covering enemy bearing and distance bins.
- Door/use-focused rollouts around interaction points.
- Recovery rollouts near walls, corners, and off-route positions.

Prefer physically reachable states obtained through actual rollouts. If save
states, teleportation, or map scripting are introduced, generated states must
be validated for collision, reachability, and legal game progression.

### Stored arrays

- engine-state vectors and validity masks;
- factorized expert targets;
- rewards and terminal flags;
- position, heading, waypoint ID, and game time;
- case category and perturbation metadata;
- contract hashes and generator configuration.

Rendered frames are optional diagnostics and are not required training inputs.

### Dataset gates

- All manifests and files pass checksums and temporal-alignment validation.
- Every control head has reported support and coverage.
- Attack, use, turning, and recovery cases meet minimum support targets before
  training begins.
- Train and validation splits use complete independent rollouts, not adjacent
  windows from the same rollout.

## Phase 6 - Train the engine-state policy

### Files

- Add `src/networks/engine_state_policy.py`.
- Add `src/engine_state_data.py`.
- Add `src/train_engine_state_policy.py`.

### Model sequence

1. Implement a small GRU reference policy to validate the data and losses.
2. Implement a recurrent spiking policy behind the same input/output contract.
3. Compare both models using identical splits and closed-loop seeds.

A state vector is small enough that training should be substantially faster
than image-based CRSNN training. CPU inference should be practical; CUDA may be
used for batched training.

### Losses

- Cross-entropy for locomotion.
- Cross-entropy for turning.
- Binary cross-entropy or focal loss for attack.
- Binary cross-entropy or focal loss for use.
- Per-head validity masks.
- Optional command-change penalty only if it does not suppress necessary
  shooting or turning.

### Checkpoint selection

Checkpoint ranking must use required-head macro recall and closed-loop results.
Raw overall accuracy cannot select a policy that merely predicts forward.

### Training gates

1. Overfit a tiny generated sequence set to at least 95% recall on every
   represented output head.
2. Pass CPU forward/backward, CUDA AMP, checkpoint, and resume tests.
3. Achieve non-zero held-out recall for attack, use, left turn, right turn, and
   recovery controls.
4. Run visible closed-loop evaluation before starting a long training job.

## Phase 7 - Closed-loop evaluation and diagnostics

### Files

- Add `src/evaluate_engine_state_policy.py`.
- Extend reporting without mixing engine-state and visual-policy leaderboards.

### Required metrics

- completion rate and terminal reason;
- kills, shots, ammunition used, and attack precision;
- route progress and waypoints reached;
- regions visited and displacement;
- doors/use points successfully handled;
- stuck-event count and longest stuck duration;
- per-head prediction distributions and longest repeated command;
- disagreement between the learned policy and scripted expert;
- inference latency on CPU and GPU.

### Exit gate

- The learned agent shoots when a valid target is present.
- It makes measurable progress beyond the previous wall failure.
- It recovers from the fixed wall and off-route test cases.
- It completes MAP01 on fixed evaluation seeds before optimization for speed or
  energy efficiency begins.

## Phase 8 - Corrective case generation

When the learned agent fails:

1. replay or reproduce the state through a valid rollout;
2. allow the scripted expert or a human to recover;
3. save the pre-failure context and recovery sequence;
4. tag it as a corrective case;
5. increase its sampling priority during retraining.

This is the engine-state equivalent of dataset aggregation. It directly adds
states visited by the learned policy rather than generating more copies of the
ideal route.

### Exit gate

- Previously reproducible failures stop recurring on the same evaluation
  scenarios without degrading combat or door behavior.

## Phase 9 - Optional reinforcement-learning fine-tuning

RL is optional because the scripted expert may already provide dense and
balanced supervision. It begins only after the policy can complete MAP01 or
reliably perform every primitive control.

- Initialize from the best imitation checkpoint.
- Explore through factorized control heads rather than uniform random compound
  actions.
- Preserve kill, route-progress, exploration, completion, and death rewards.
- Add a causal stuck penalty.
- Compare against the frozen imitation checkpoint using identical seeds and
  route configuration.

Accept RL changes only when completion or route progress improves without
destroying attack and use behavior.

## Existing data that can be reused

The current 11 image demonstrations contain action indices, positions, viewing
angles, rewards, and terminal flags. They can be converted into:

- factorized action targets;
- route-position examples;
- displacement and stuck-history examples.

They do not contain complete per-step health, ammunition, visible-object, or
route-waypoint state. They therefore supplement but do not replace newly
generated engine-state rollouts. Existing files must remain unchanged and
versioned under their original schema.

## Risks and mitigations

### Engine-label leakage

The policy may receive information unavailable to a human player. This is
intentional for this agent, but every privileged field must be documented and
the result must not be labelled vision-only.

### MAP01 memorization

Position and waypoint inputs make the agent map-specific. Use separate route
versions and do not claim cross-map generalization without new experiments.

### Invalid generated states

Arbitrary teleportation can create impossible or collision-invalid examples.
Prefer actual rollouts and validate every alternate initialization.

### Scripted-expert bias

A learned policy can only imitate the expert's route and combat rules. Add
perturbations, recovery cases, and human corrections where the expert is
insufficient.

### Multiple valid actions

At many positions several actions can succeed. Where appropriate, store soft
targets or permitted-action masks instead of punishing every alternative to a
single expert command.

## Recommended implementation order

1. Audit engine variables and labels.
2. Define and test the engine-state and factorized-control contracts.
3. Build and validate the MAP01 waypoint route.
4. Implement a scripted expert and require it to complete the level.
5. Generate balanced full-route, combat, interaction, and recovery sequences.
6. Overfit a tiny dataset with the GRU reference policy.
7. Train and evaluate the GRU on held-out rollouts.
8. Implement and compare the recurrent spiking policy.
9. Aggregate corrective cases from learned-policy failures.
10. Consider RL only after full-level primitive behavior is reliable.

# Semantic Policy Redesign Implementation Plan

## Objective

Replace the current 22-class end-to-end imitation policy with a staged system
that:

1. converts each visual observation into a compact structured scene state;
2. carries that state through recurrent memory; and
3. predicts movement, turning, attack, and use independently.

The redesign must prove that the agent can learn basic controls, especially
shooting and recovery, before reinforcement learning is introduced.

## Why this redesign is needed

The current policy treats actions such as `attack`, `forward_attack`, and
`turn_right_attack` as unrelated classes. This fragments the shooting signal
and rewards high accuracy on common navigation labels even when combat recall
is zero. Successful demonstrations also contain few observations from the bad
states the autonomous agent reaches, such as facing a wall.

Natural-language sentences are not required. The "description" should be a
versioned numeric structure so it is fast, deterministic, and easy to test.

## Target architecture

```text
84x84 four-frame observation
          |
          v
visual perception model
          |
          v
structured scene state + telemetry
          |
          v
recurrent semantic policy
          |
          +-- locomotion: none / forward / backward / strafe-left / strafe-right
          +-- turning:    none / left / right
          +-- attack:     false / true
          +-- use:        false / true
          |
          v
validated button composer -> ViZDoom
```

The first policy prototype will use perfect semantic targets supplied by the
engine (the **oracle semantic policy**). The learned visual perception model is
connected only after that policy passes closed-loop tests.

## Proposed semantic state

The exact contract will be finalized after auditing the labels available on
FreeDoom MAP01. The initial candidates are:

- enemy visible in left, centre, and right image regions;
- nearest-enemy direction and coarse distance;
- visible enemy count;
- usable object or door cue, where the engine exposes a reliable target;
- obstacle proximity or lack of visual motion;
- normalized health and ammunition;
- recent damage and ammunition changes;
- normalized movement delta and turning delta;
- stuck flag derived causally from recent position history;
- route-memory inputs such as heading encoded as sine/cosine.

Every field will have a validity mask. Unavailable semantic fields must be
masked rather than silently stored as false.

## Phase 0 - Preserve and measure the current baseline

### Work

- Commit the current AMP fix, balanced-sampling work, evaluation telemetry,
  tests, and the intended demonstration manifest without including temporary
  files or local ViZDoom settings.
- Preserve the current imitation checkpoints as comparison artifacts.
- Record the existing baseline metrics: held-out accuracy, macro recall,
  per-action recall, kills, regions, stuck duration, and completion.

### Exit gate

- The branch has a reproducible baseline and no redesign code is mixed into
  the baseline commit.

## Phase 1 - Define versioned semantic and control contracts

### Files

- Add `src/semantic_config.py` for semantic field names, normalization,
  validity rules, schema hash, locomotion classes, turning classes, and button
  composition.
- Extend `src/agent_config.py` only where raw button composition is required;
  retain the old 22-action contract for old checkpoints and datasets.
- Add contract tests in `tests/test_semantic_config.py`.

### Design rules

- Opposing controls, such as turn-left and turn-right, cannot be emitted
  together.
- Attack and use are independent binary targets.
- Existing discrete action indices can be converted into the new target heads,
  allowing the 11 demonstrations to remain useful for control supervision.
- Semantic and action contract hashes must be stored in every new checkpoint.

### Exit gate

- Every old catalogue action converts deterministically to factorized targets.
- Every valid factorized prediction composes to a safe ViZDoom button vector.
- Invalid combinations fail loudly in unit tests.

## Phase 2 - Audit ViZDoom semantic supervision

### Work

- Add `src/inspect_semantic_labels.py` to report visible object names, label
  geometry, game variables, and depth-buffer availability during MAP01.
- Determine which enemy, item, and door targets can be generated reliably.
- If wall proximity is required, evaluate the depth buffer as a training-only
  target. Do not assume the object-label buffer describes walls or map
  geometry.
- Save a short audit report under `reports/semantic-label-audit/`.

### Exit gate

- Each proposed semantic field is classified as engine-derived,
  history-derived, manually annotated, or unavailable.
- Unreliable fields are removed before the dataset schema is finalized.

## Phase 3 - Add an enriched demonstration schema and recorder

### Files

- Introduce demonstration schema version 3 in `src/demonstrations.py` without
  breaking the schema-2 loader.
- Extend `src/record_demonstrations.py` to record observation-aligned:
  `button_targets`, `semantic_targets`, and `semantic_valid` arrays.
- Store new recordings in a separate directory such as
  `data/demonstrations/map01-semantic/`.
- Extend `tests/test_demonstrations.py` and add alignment tests using a mocked
  ViZDoom state.

### Data policy

- Keep observation-before-action alignment.
- Compute stuck and motion targets using only present and past positions, never
  future data.
- Record per-field semantic coverage in the manifest.
- Do not require every episode to be a perfect speedrun. Mark successful route,
  combat practice, door-use practice, and recovery demonstration separately.

### Exit gate

- A recorded episode reloads with matching checksums and contracts.
- Semantic targets, factorized button targets, images, and positions have equal
  sequence lengths and verified temporal alignment.

## Phase 4 - Train the oracle semantic policy

### Files

- Add `src/networks/semantic_policy.py` with a small recurrent reference model
  and a recurrent spiking variant behind the same interface.
- Add `src/semantic_data.py` for sequence windows and per-head targets.
- Add `src/train_semantic_policy.py` for behavior cloning from perfect semantic
  targets.
- Add focused model, loss, AMP, checkpoint, and resume tests.

### Loss

- Cross-entropy for locomotion.
- Cross-entropy for turning.
- Binary cross-entropy or focal loss for attack and use.
- Validity masks per head.
- Metrics must include precision, recall, and confusion matrices for every
  head. Overall accuracy alone is not a checkpoint criterion.

### Training gates

1. Overfit a tiny diagnostic subset to at least 95% recall on every represented
   head. Failure here indicates a model, target, or loss bug.
2. Pass held-out per-head evaluation with non-zero attack and turn recall.
3. Run the oracle policy in visible closed-loop evaluation.
4. Require actual attack decisions, at least one kill across evaluation runs,
   and recovery from a deliberately demonstrated wall state before proceeding.

The checkpoint selector should prioritize required-head macro recall and then
closed-loop progress, not raw common-class accuracy.

## Phase 5 - Train the image-to-semantics perception model

### Files

- Add `src/networks/semantic_perception.py`.
- Add `src/train_semantic_perception.py`.
- Add `src/evaluate_semantic_perception.py` with saved confusion matrices and
  visual overlays.

### Training

- Input: the same four-frame `84x84` observation used by the agent.
- Output: only semantic fields that survived the label audit.
- Split train and validation data by complete episode.
- Use field-appropriate losses and validity masks.
- Add controlled brightness, contrast, and small spatial perturbations without
  changing label meaning.

### Exit gate

- Enemy visibility and direction meet agreed recall thresholds on held-out
  episodes.
- False-positive attack cues are measured explicitly.
- Saved overlays confirm that metrics are not caused by label misalignment.

## Phase 6 - Integrate perception and policy

### Work

- Run three evaluation modes through one interface:
  1. oracle semantics;
  2. predicted semantics;
  3. the old 22-class visual baseline.
- Initially freeze both trained components and connect them.
- Train the policy with a scheduled mixture of true and predicted semantic
  states so it becomes tolerant of perception noise.
- Consider end-to-end fine-tuning only after the frozen pipeline works.

### Closed-loop metrics

- completion and terminal reason;
- kills and shots fired;
- attack precision when an enemy is visible;
- regions visited;
- position displacement;
- stuck events and longest stuck duration;
- action-head distributions and longest repeated command;
- oracle-versus-predicted semantic disagreement.

### Exit gate

- The predicted-semantic agent shoots and turns in the same situations as the
  oracle policy often enough to obtain kills and exceed the old baseline's
  exploration progress.

## Phase 7 - Add corrective demonstration collection

### Work

- Add an intervention mode in which the model controls the game until the human
  takes over.
- Save pre-intervention context, human recovery actions, and an intervention
  marker.
- Sample intervention and recovery windows more frequently during retraining.
- Include wall approaches, wrong turns, missed enemies, door failures, and
  low-ammo combat in the correction set.

### Exit gate

- The same recorded failure scenario no longer produces an indefinitely
  repeated action.
- Closed-loop stuck duration decreases on fixed evaluation seeds.

## Phase 8 - Optional reinforcement-learning fine-tuning

RL begins only after the integrated policy can move, turn, attack, use, and
recover in controlled tests.

- Initialize from the semantic-policy imitation checkpoint.
- Keep separate per-head exploration rather than choosing uniformly among 22
  compound actions.
- Preserve kill, exploration, completion, and death rewards.
- Add a stuck penalty based on causal movement history.
- Compare against the frozen imitation checkpoint on identical seeds.

RL is accepted only if it improves closed-loop completion or progress without
destroying attack and use recall.

## Test and verification strategy

Each phase must include:

- unit tests for contracts, shapes, masks, temporal alignment, and illegal
  controls;
- CPU forward/backward tests;
- CUDA AMP backward tests where CUDA is available;
- checkpoint round-trip and incompatible-contract rejection tests;
- a one-batch `--validate-only` command;
- a tiny overfit test before any multi-hour training run;
- visible closed-loop evaluation at fixed seeds.

No long training run should begin merely because validation loss is finite.
The tiny-overfit and required-head recall gates must pass first.

## Data that can be reused

The 11 existing schema-2 demonstrations can provide:

- images;
- old action indices converted to factorized button targets;
- positions and viewing angles;
- rewards, terminal flags, and causal movement/stuck targets.

They cannot provide reliable historical enemy/door semantic targets because
the object-label buffer was not stored per step. New enriched recordings are
therefore required for perception supervision. Existing data must not be
rewritten or relabelled as if those targets had been captured.

## Recommended implementation order

1. Preserve the current baseline in a clean commit.
2. Implement and test semantic/action contracts.
3. Audit MAP01 labels and finalize the semantic fields.
4. Implement schema 3 and record a short enriched episode.
5. Prove the oracle policy can overfit and act correctly.
6. Collect targeted combat, use, and recovery demonstrations.
7. Train and verify perception.
8. Integrate, evaluate, and collect corrective interventions.
9. Introduce RL only if the imitation pipeline passes its primitive-action
   gates.

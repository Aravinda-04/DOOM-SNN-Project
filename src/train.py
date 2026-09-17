from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from checkpointing import load_checkpoint, save_checkpoint
from env import DoomEnvironment
from networks import MODEL_NAMES, build_model
from project_paths import DEFAULT_CONFIG_PATH, experiment_directories, make_run_id
from runtime_utils import resolve_device, set_random_seeds


# Defaults remain close to the original experiment for backward compatibility.
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10_000
LR = 1e-4
MEMORY_SIZE = 10_000
REPLAY_START_SIZE = 1_000
TAU = 0.001
NUM_EPISODES = 200
SPARSITY_WEIGHT = 1e-9
REWARD_SCALE = 100.0


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    episodes: int
    batch_size: int
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    learning_rate: float
    memory_size: int
    replay_start_size: int
    tau: float
    sparsity_weight: float
    reward_scale: float
    eval_interval: int
    eval_episodes: int
    eval_seeds: tuple[int, ...]
    repeat_action_threshold: int
    repeat_action_penalty: float
    stalled_move_penalty: float
    instant_kill_steps: int
    easy_target_offset: float
    balanced_training: bool
    balanced_evaluation: bool
    max_spawn_attempts: int
    aim_progress_weight: float
    off_target_attack_penalty: float
    weight_clip: float | None
    seed: int
    config_path: str
    run_id: str


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def spawn_category(info: dict, easy_target_offset: float) -> str:
    offset = info.get("target_horizontal_offset")
    if not info.get("target_visible") or offset is None:
        return "invisible"
    if abs(offset) <= easy_target_offset:
        return "easy"
    return "hard_left" if offset < 0 else "hard_right"


def reset_for_spawn(
    env: DoomEnvironment,
    category: str,
    easy_target_offset: float,
    max_attempts: int,
) -> tuple[np.ndarray, dict, int]:
    for attempt in range(1, max_attempts + 1):
        state = env.reset()
        info = env.diagnostics()
        if spawn_category(info, easy_target_offset) == category:
            return state, info, attempt
    raise RuntimeError(
        f"Could not find a {category} spawn after {max_attempts} resets."
    )


def aiming_reward(
    before_info: dict,
    after_info: dict,
    action_name: str,
    *,
    easy_target_offset: float,
    progress_weight: float,
    off_target_attack_penalty: float,
) -> tuple[float, bool]:
    """Training-only privileged reward; labels never enter the policy input."""
    reward = 0.0
    before_offset = before_info.get("target_horizontal_offset")
    after_offset = after_info.get("target_horizontal_offset")
    if (
        before_info.get("target_visible")
        and after_info.get("target_visible")
        and before_offset is not None
        and after_offset is not None
    ):
        reward += progress_weight * (abs(before_offset) - abs(after_offset))

    off_target_attack = action_name == "attack" and (
        not before_info.get("target_visible")
        or before_offset is None
        or abs(before_offset) > easy_target_offset
    )
    if off_target_attack:
        reward -= off_target_attack_penalty
    return reward, off_target_attack


def optimize_model(
    memory: ReplayMemory,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    *,
    batch_size: int,
    replay_start_size: int,
    gamma: float,
    sparsity_weight: float,
    device: torch.device,
) -> float | None:
    if len(memory) < max(batch_size, replay_start_size):
        return None

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    state_batch = torch.from_numpy(np.stack(states)).unsqueeze(1).to(device)
    action_batch = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_state_batch = torch.from_numpy(np.stack(next_states)).unsqueeze(1).to(device)
    done_batch = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values, spike_count = policy_net(state_batch)
    state_action_values = q_values.gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_q_values, _ = target_net(next_state_batch)
        next_state_values = next_q_values.max(1).values

    expected_values = reward_batch + gamma * next_state_values * (1 - done_batch)
    td_loss = criterion(state_action_values, expected_values)
    sparsity_loss = spike_count / batch_size * sparsity_weight
    loss = td_loss + sparsity_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def clip_weights(model: nn.Module, limit: float | None) -> None:
    """Apply an optional stability clamp; this is deliberately not quantization."""
    if limit is None:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.clamp_(-limit, limit)


def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float) -> None:
    """Update floating-point parameters and copy non-trainable buffers safely."""
    with torch.no_grad():
        for target_parameter, policy_parameter in zip(
            target_net.parameters(), policy_net.parameters()
        ):
            target_parameter.lerp_(policy_parameter, tau)
        for target_buffer, policy_buffer in zip(
            target_net.buffers(), policy_net.buffers()
        ):
            target_buffer.copy_(policy_buffer)


def evaluate_policy(
    config_path: Path,
    policy_net: nn.Module,
    device: torch.device,
    episodes_per_seed: int,
    seeds: tuple[int, ...],
    instant_kill_steps: int,
    easy_target_offset: float,
    balanced: bool,
    max_spawn_attempts: int,
) -> dict:
    """Evaluate natural episodes or equal hard-left/hard-right episodes."""
    was_training = policy_net.training
    policy_net.eval()
    rewards = []
    successes = []
    noninstant_successes = []
    hard_spawn_results = []
    directional_results = {"hard_left": [], "hard_right": []}
    try:
        for seed in seeds:
            with DoomEnvironment(config_file=config_path, render=False, seed=seed) as env:
                categories = (
                    ("hard_left", "hard_right") if balanced else (None,)
                )
                for requested_category in categories:
                    for _ in range(episodes_per_seed):
                        if requested_category is None:
                            state = env.reset()
                            start_info = env.diagnostics()
                        else:
                            state, start_info, _ = reset_for_spawn(
                                env,
                                requested_category,
                                easy_target_offset,
                                max_spawn_attempts,
                            )
                        start_kills = start_info["kill_count"] or 0.0
                        category = spawn_category(start_info, easy_target_offset)
                        done = False
                        episode_reward = 0.0
                        decision_steps = 0
                        while not done:
                            state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)
                            with torch.no_grad():
                                q_values, _ = policy_net(state_tensor)
                            action = q_values.argmax(1).item()
                            state, reward, done = env.step(action)
                            episode_reward += reward
                            decision_steps += 1
                        end_kills = env.diagnostics()["kill_count"] or 0.0
                        rewards.append(episode_reward)
                        success = end_kills > start_kills
                        successes.append(success)
                        noninstant_successes.append(success and decision_steps > instant_kill_steps)
                        if category != "easy":
                            hard_spawn_results.append(success)
                        if category in directional_results:
                            directional_results[category].append(success)
    finally:
        policy_net.train(was_training)
    left_rate = float(np.mean(directional_results["hard_left"])) if directional_results["hard_left"] else 0.0
    right_rate = float(np.mean(directional_results["hard_right"])) if directional_results["hard_right"] else 0.0
    return {
        "rewards": rewards,
        "success_rate": float(np.mean(successes)),
        "noninstant_success_rate": float(np.mean(noninstant_successes)),
        "hard_success_rate": float(np.mean(hard_spawn_results)) if hard_spawn_results else 0.0,
        "left_success_rate": left_rate,
        "right_success_rate": right_rate,
        "worst_direction_success_rate": min(left_rate, right_rate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViZDoom DQN policy.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="snn")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--eps-start", type=float, default=EPS_START)
    parser.add_argument("--eps-end", type=float, default=EPS_END)
    parser.add_argument("--eps-decay", type=float, default=EPS_DECAY)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--memory-size", type=int, default=MEMORY_SIZE)
    parser.add_argument("--replay-start-size", type=int, default=REPLAY_START_SIZE)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--sparsity-weight", type=float, default=SPARSITY_WEIGHT)
    parser.add_argument("--reward-scale", type=float, default=REWARD_SCALE)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--repeat-action-threshold", type=int, default=8)
    parser.add_argument("--repeat-action-penalty", type=float, default=2.0)
    parser.add_argument("--stalled-move-penalty", type=float, default=2.0)
    parser.add_argument("--instant-kill-steps", type=int, default=2)
    parser.add_argument("--easy-target-offset", type=float, default=0.1)
    parser.add_argument("--balanced-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--balanced-evaluation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-spawn-attempts", type=int, default=1000)
    parser.add_argument("--aim-progress-weight", type=float, default=20.0)
    parser.add_argument("--off-target-attack-penalty", type=float, default=4.0)
    parser.add_argument("--weight-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.eval_interval <= 0 or args.eval_episodes <= 0:
        raise ValueError("Evaluation interval and episode count must be positive.")
    if args.batch_size <= 0 or args.memory_size < args.batch_size:
        raise ValueError("Memory size must be at least as large as batch size.")
    if args.reward_scale == 0:
        raise ValueError("--reward-scale cannot be zero.")
    if args.eps_decay <= 0:
        raise ValueError("--eps-decay must be positive.")
    if not 0 <= args.eps_end <= args.eps_start <= 1:
        raise ValueError("Epsilon values must satisfy 0 <= end <= start <= 1.")
    if not 0 <= args.gamma <= 1:
        raise ValueError("--gamma must be between 0 and 1.")
    if not 0 <= args.tau <= 1:
        raise ValueError("--tau must be between 0 and 1.")
    if args.weight_clip is not None and args.weight_clip <= 0:
        raise ValueError("--weight-clip must be positive.")
    if not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("--eval-seeds must contain unique values.")
    if args.repeat_action_threshold < 1:
        raise ValueError("--repeat-action-threshold must be positive.")
    if args.repeat_action_penalty < 0 or args.stalled_move_penalty < 0:
        raise ValueError("Action penalties cannot be negative.")
    if args.instant_kill_steps < 1:
        raise ValueError("--instant-kill-steps must be positive.")
    if not 0 <= args.easy_target_offset <= 1:
        raise ValueError("--easy-target-offset must be between 0 and 1.")
    if args.max_spawn_attempts < 1:
        raise ValueError("--max-spawn-attempts must be positive.")
    if args.aim_progress_weight < 0 or args.off_target_attack_penalty < 0:
        raise ValueError("Aiming reward values cannot be negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seeds(args.seed)
    device = resolve_device(args.device)

    run_id = args.run_id or make_run_id(args.seed)
    model_dir, tensorboard_dir = experiment_directories(args.model, run_id)
    if args.resume is None and (model_dir.exists() or tensorboard_dir.exists()):
        raise FileExistsError(
            f"Run '{run_id}' already exists. Choose another --run-id or use --resume."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        model=args.model,
        episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        learning_rate=args.learning_rate,
        memory_size=args.memory_size,
        replay_start_size=args.replay_start_size,
        tau=args.tau,
        sparsity_weight=args.sparsity_weight,
        reward_scale=args.reward_scale,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seeds=tuple(args.eval_seeds),
        repeat_action_threshold=args.repeat_action_threshold,
        repeat_action_penalty=args.repeat_action_penalty,
        stalled_move_penalty=args.stalled_move_penalty,
        instant_kill_steps=args.instant_kill_steps,
        easy_target_offset=args.easy_target_offset,
        balanced_training=args.balanced_training,
        balanced_evaluation=args.balanced_evaluation,
        max_spawn_attempts=args.max_spawn_attempts,
        aim_progress_weight=args.aim_progress_weight,
        off_target_attack_penalty=args.off_target_attack_penalty,
        weight_clip=args.weight_clip,
        seed=args.seed,
        config_path=str(args.config.expanduser().resolve()),
        run_id=run_id,
    )
    config_dict = asdict(config)
    (model_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    print(f"Training {args.model.upper()} on {device}")
    print(f"Run ID: {run_id}")
    print(f"Checkpoints: {model_dir}")
    print(f"TensorBoard: {tensorboard_dir}")

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    env = None
    try:
        env = DoomEnvironment(config_file=args.config, render=False, seed=args.seed)
        action_size = len(env.actions)
        policy_net = build_model(args.model, action_size=action_size).to(device)
        target_net = build_model(args.model, action_size=action_size).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
        criterion = nn.SmoothL1Loss()
        memory = ReplayMemory(args.memory_size)

        start_episode = 0
        steps_done = 0
        best_eval_mean = -float("inf")
        best_eval_success = -float("inf")
        best_eval_noninstant_success = -float("inf")
        best_eval_hard_success = -float("inf")
        best_eval_worst_direction = -float("inf")
        if args.resume is not None:
            resume_state = load_checkpoint(args.resume, policy_net, device, optimizer)
            checkpoint_model = resume_state.get("config", {}).get("model")
            if checkpoint_model and checkpoint_model != args.model:
                raise ValueError(
                    f"Checkpoint contains model '{checkpoint_model}', not '{args.model}'."
                )
            start_episode = int(resume_state.get("episode", -1)) + 1
            steps_done = int(resume_state.get("global_step", 0))
            best_eval_mean = float(
                resume_state.get("best_eval_mean", -float("inf"))
            )
            best_eval_success = float(
                resume_state.get("best_eval_success", -float("inf"))
            )
            best_eval_noninstant_success = float(
                resume_state.get("best_eval_noninstant_success", -float("inf"))
            )
            best_eval_hard_success = float(
                resume_state.get("best_eval_hard_success", -float("inf"))
            )
            best_eval_worst_direction = float(
                resume_state.get("best_eval_worst_direction", -float("inf"))
            )
            print(f"Resuming at episode {start_episode + 1}, step {steps_done}")

        if start_episode >= args.episodes:
            raise ValueError(
                f"Checkpoint is already at episode {start_episode}; "
                f"--episodes must be greater than {start_episode}."
            )

        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        policy_net.train()

        for episode in range(start_episode, args.episodes):
            if args.balanced_training:
                requested_spawn = "hard_left" if episode % 2 == 0 else "hard_right"
                state, _, spawn_attempts = reset_for_spawn(
                    env,
                    requested_spawn,
                    args.easy_target_offset,
                    args.max_spawn_attempts,
                )
            else:
                requested_spawn = "natural"
                spawn_attempts = 1
                state = env.reset()
            total_reward = 0.0
            total_shaping_reward = 0.0
            off_target_attacks = 0
            done = False
            started_at = time.perf_counter()
            last_epsilon = args.eps_start
            previous_action = None
            repeated_actions = 0

            while not done:
                last_epsilon = args.eps_end + (
                    args.eps_start - args.eps_end
                ) * math.exp(-steps_done / args.eps_decay)
                steps_done += 1

                if random.random() > last_epsilon:
                    state_tensor = (
                        torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)
                    )
                    with torch.no_grad():
                        q_values, _ = policy_net(state_tensor)
                    action = q_values.argmax(1).item()
                else:
                    action = random.randrange(action_size)

                before_info = env.diagnostics()
                next_state, reward, done = env.step(action)
                total_reward += reward
                after_info = env.diagnostics()

                if action == previous_action:
                    repeated_actions += 1
                else:
                    repeated_actions = 1
                    previous_action = action

                shaped_reward = reward
                if repeated_actions > args.repeat_action_threshold:
                    shaped_reward -= args.repeat_action_penalty
                if action in (0, 1):
                    before_position = (
                        before_info["position_x"],
                        before_info["position_y"],
                    )
                    after_position = (
                        after_info["position_x"],
                        after_info["position_y"],
                    )
                    if None not in before_position + after_position:
                        displacement = math.hypot(
                            after_position[0] - before_position[0],
                            after_position[1] - before_position[1],
                        )
                        if displacement < 0.01:
                            shaped_reward -= args.stalled_move_penalty
                aim_reward, off_target_attack = aiming_reward(
                    before_info,
                    after_info,
                    env.action_names[action],
                    easy_target_offset=args.easy_target_offset,
                    progress_weight=args.aim_progress_weight,
                    off_target_attack_penalty=args.off_target_attack_penalty,
                )
                shaped_reward += aim_reward
                total_shaping_reward += aim_reward
                off_target_attacks += int(off_target_attack)
                memory.push(
                    state,
                    action,
                    shaped_reward / args.reward_scale,
                    next_state,
                    done,
                )
                state = next_state

                loss = optimize_model(
                    memory,
                    policy_net,
                    target_net,
                    optimizer,
                    criterion,
                    batch_size=args.batch_size,
                    replay_start_size=args.replay_start_size,
                    gamma=args.gamma,
                    sparsity_weight=args.sparsity_weight,
                    device=device,
                )
                if loss is not None:
                    clip_weights(policy_net, args.weight_clip)
                    soft_update(target_net, policy_net, args.tau)
                    writer.add_scalar("train/loss", loss, steps_done)

            elapsed = time.perf_counter() - started_at
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Reward: {total_reward:.1f} | Epsilon: {last_epsilon:.3f} | "
                f"Spawn: {requested_spawn} ({spawn_attempts} resets) | "
                f"Aim shaping: {total_shaping_reward:.1f} | Time: {elapsed:.1f}s"
            )
            writer.add_scalar("train/reward", total_reward, episode)
            writer.add_scalar("train/epsilon", last_epsilon, episode)
            writer.add_scalar("train/aim_shaping_reward", total_shaping_reward, episode)
            writer.add_scalar("train/off_target_attacks", off_target_attacks, episode)

            should_evaluate = (
                (episode + 1) % args.eval_interval == 0
                or episode + 1 == args.episodes
            )
            if should_evaluate:
                evaluation = evaluate_policy(
                    args.config,
                    policy_net,
                    device,
                    args.eval_episodes,
                    tuple(args.eval_seeds),
                    args.instant_kill_steps,
                    args.easy_target_offset,
                    args.balanced_evaluation,
                    args.max_spawn_attempts,
                )
                eval_rewards = evaluation["rewards"]
                eval_success = evaluation["success_rate"]
                eval_noninstant_success = evaluation["noninstant_success_rate"]
                eval_hard_success = evaluation["hard_success_rate"]
                eval_left_success = evaluation["left_success_rate"]
                eval_right_success = evaluation["right_success_rate"]
                eval_worst_direction = evaluation["worst_direction_success_rate"]
                eval_mean = float(np.mean(eval_rewards))
                eval_std = float(np.std(eval_rewards))
                writer.add_scalar("eval/mean_reward", eval_mean, episode)
                writer.add_scalar("eval/std_reward", eval_std, episode)
                writer.add_scalar("eval/success_rate", eval_success, episode)
                writer.add_scalar(
                    "eval/noninstant_success_rate",
                    eval_noninstant_success,
                    episode,
                )
                writer.add_scalar(
                    "eval/hard_spawn_success_rate", eval_hard_success, episode
                )
                writer.add_scalar("eval/hard_left_success_rate", eval_left_success, episode)
                writer.add_scalar("eval/hard_right_success_rate", eval_right_success, episode)
                writer.add_scalar("eval/worst_direction_success_rate", eval_worst_direction, episode)
                print(
                    f"Evaluation | Worst direction: {eval_worst_direction:.1%} | "
                    f"Left: {eval_left_success:.1%} | Right: {eval_right_success:.1%} | "
                    f"Hard-spawn: {eval_hard_success:.1%} | "
                    f"Non-instant: {eval_noninstant_success:.1%} | "
                    f"Success: {eval_success:.1%} | "
                    f"Mean: {eval_mean:.1f} | Std: {eval_std:.1f} | "
                    f"Episodes: {len(eval_rewards)}"
                )
                if (eval_worst_direction, eval_hard_success, eval_mean) > (
                    best_eval_worst_direction,
                    best_eval_hard_success,
                    best_eval_mean,
                ):
                    best_eval_worst_direction = eval_worst_direction
                    best_eval_hard_success = eval_hard_success
                    best_eval_noninstant_success = eval_noninstant_success
                    best_eval_success = eval_success
                    best_eval_mean = eval_mean
                    save_checkpoint(
                        model_dir / "best.pth",
                        policy_net,
                        optimizer,
                        episode=episode,
                        global_step=steps_done,
                        best_eval_mean=best_eval_mean,
                        best_eval_success=best_eval_success,
                        best_eval_noninstant_success=best_eval_noninstant_success,
                        best_eval_hard_success=best_eval_hard_success,
                        best_eval_worst_direction=best_eval_worst_direction,
                        config=config_dict,
                    )
                    print(
                        f"New best: {best_eval_worst_direction:.1%} worst direction, "
                        f"{best_eval_hard_success:.1%} hard-spawn, "
                        f"{best_eval_noninstant_success:.1%} non-instant, "
                        f"{best_eval_success:.1%} total success, "
                        f"mean reward {best_eval_mean:.1f}"
                    )

            save_checkpoint(
                model_dir / "latest.pth",
                policy_net,
                optimizer,
                episode=episode,
                global_step=steps_done,
                best_eval_mean=best_eval_mean,
                best_eval_success=best_eval_success,
                best_eval_noninstant_success=best_eval_noninstant_success,
                best_eval_hard_success=best_eval_hard_success,
                best_eval_worst_direction=best_eval_worst_direction,
                config=config_dict,
            )
            writer.flush()

        print("Training completed successfully.")
    finally:
        if env is not None:
            env.close()
        writer.close()


if __name__ == "__main__":
    main()

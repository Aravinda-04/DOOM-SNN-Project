"""Recurrent DQN trainer for the full DOOM level agent.

Trains a CRSNNQNetwork (or RSNNv2QNetwork) on FreeDoom2 MAP01.

Run from the project root:

    python src/train_level.py --model crsnn --run-id my-run

For a full-level run:

    python src/train_level.py --model crsnn \\
        --run-id crsnn-fullmap-seed0 \\
        --episodes 2000 --seed 0 --device auto

Resume a run:

    python src/train_level.py --model crsnn \\
        --run-id crsnn-fullmap-seed0 --resume models/crsnn/crsnn-fullmap-seed0/latest.pth

See --help for all options.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    AGENT_CONTRACT,
    FRAME_STACK,
    PREPROCESSING_VERSION,
)
from checkpointing import load_checkpoint, save_checkpoint
from doom_level_env import DoomLevelEnvironment
from networks import build_model
from project_paths import (
    LEVEL_CONFIG_PATH,
    PROJECT_ROOT,
    experiment_directories,
    make_run_id,
)
from runtime_utils import resolve_device, set_random_seeds

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20_000
LR = 5e-5
MEMORY_SIZE = 10_000
REPLAY_START_SIZE = 2_000
TAU = 0.001
NUM_EPISODES = 2_000
SPARSITY_WEIGHT = 1e-3
SEQ_LEN = 16            # TBPTT window length (frames per training sequence)
BURN_IN = 4             # preceding frames used to reconstruct recurrent state
EVAL_INTERVAL = 25
EVAL_EPISODES = 3
CHECKPOINT_INTERVAL = 1
GRAD_CLIP = 10.0

LEVEL_MODELS = ("crsnn", "rsnn_v2", "level_cnn", "level_drqn")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelTrainingConfig:
    model: str
    episodes: int
    batch_size: int
    seq_len: int
    burn_in: int
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    learning_rate: float
    memory_size: int
    replay_start_size: int
    tau: float
    sparsity_weight: float
    eval_interval: int
    eval_episodes: int
    eval_seeds: tuple[int, ...]
    checkpoint_interval: int
    episode_timeout: Optional[int]
    seed: int
    config_path: str
    run_id: str
    action_size: int
    in_channels: int
    agent_contract_hash: str
    preprocessing_version: int
    action_names: tuple[str, ...]


@dataclass
class StoredEpisode:
    """Compact replay representation for one episode.

    Each transition stores only its current observation. The next observation
    is the following transition's observation, except for the final transition,
    which uses ``final_observation``. This avoids duplicating every stacked
    observation in memory.
    """

    transitions: list[tuple[np.ndarray, int, np.float32, bool]]
    final_observation: np.ndarray

    @property
    def nbytes(self) -> int:
        observation_bytes = sum(step[0].nbytes for step in self.transitions)
        return observation_bytes + self.final_observation.nbytes


class EpisodeReplayMemory:
    """Episode-based replay buffer for sequence sampling (TBPTT).

    Stores complete episodes and samples random fixed-length windows from them.
    Observations are quantized to uint8 and normalized when sampled.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to retain across completed episodes.
    seq_len:
        Number of transitions that contribute to the training loss.
    burn_in:
        Preceding transitions used without gradients to reconstruct recurrent
        state for a randomly sampled training window.
    """

    def __init__(
        self,
        capacity: int,
        seq_len: int = SEQ_LEN,
        burn_in: int = 0,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.window_len = burn_in + seq_len
        self._episodes: deque[StoredEpisode] = deque()
        self._current: list[tuple[np.ndarray, int, np.float32, bool]] = []
        self._transition_count = 0

    @staticmethod
    def _encode_observation(observation: np.ndarray) -> np.ndarray:
        """Store a normalized observation compactly as uint8."""
        if observation.dtype == np.uint8:
            return np.array(observation, copy=True)
        clipped = np.clip(observation, 0.0, 1.0)
        return np.rint(clipped * 255.0).astype(np.uint8)

    @staticmethod
    def _decode_observation(observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32) / 255.0

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition to the current episode buffer."""
        self._current.append(
            (self._encode_observation(obs), action, np.float32(reward), done)
        )
        if done:
            if len(self._current) >= self.window_len:
                transitions = self._current
                # A single unusually long episode must not defeat the global
                # transition budget. Retaining its terminal tail still yields
                # valid sequence windows with a fresh hidden-state boundary.
                if len(transitions) > self.capacity:
                    transitions = transitions[-self.capacity:]
                episode = StoredEpisode(
                    transitions=transitions,
                    final_observation=self._encode_observation(next_obs),
                )
                self._episodes.append(episode)
                self._transition_count += len(transitions)
                while (
                    self._transition_count > self.capacity
                    and len(self._episodes) > 1
                ):
                    removed = self._episodes.popleft()
                    self._transition_count -= len(removed.transitions)
            self._current = []

    def sample(self, batch_size: int) -> list[list]:
        """Sample ``batch_size`` fixed-length windows from stored episodes."""
        if not self._episodes:
            raise ValueError("Replay memory has no complete eligible episodes.")
        episodes = random.choices(tuple(self._episodes), k=batch_size)
        batch = []
        for ep in episodes:
            transitions = ep.transitions
            start = random.randint(0, len(transitions) - self.window_len)
            sequence = []
            for index in range(start, start + self.window_len):
                obs, action, reward, done = transitions[index]
                next_obs = (
                    transitions[index + 1][0]
                    if index + 1 < len(transitions)
                    else ep.final_observation
                )
                sequence.append(
                    (
                        self._decode_observation(obs),
                        action,
                        reward,
                        self._decode_observation(next_obs),
                        done,
                    )
                )
            batch.append(sequence)
        return batch

    def __len__(self) -> int:
        return self._transition_count

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def estimated_bytes(self) -> int:
        """Bytes occupied by stored observation arrays (excluding overhead)."""
        return sum(episode.nbytes for episode in self._episodes)


# ---------------------------------------------------------------------------
# Optimisation step
# ---------------------------------------------------------------------------

def optimize_model(
    memory: EpisodeReplayMemory,
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
) -> Optional[float]:
    """Sample one mini-batch and perform a gradient step.

    Hidden state is reset to ``None`` at each sequence boundary (TBPTT).
    Gradients are clipped to ``GRAD_CLIP`` for stability in long episodes.
    """
    if (
        len(memory) < max(batch_size, replay_start_size)
        or memory.num_episodes == 0
    ):
        return None

    sequences = memory.sample(batch_size)  # list[list[transition]]
    seq_len = memory.seq_len
    burn_in = memory.burn_in
    window_len = memory.window_len

    # Build tensors: (B, T, C, H, W)
    # Frames were stored as float16 → cast to float32 for compute.
    obs_seqs = np.array(
        [[step[0] for step in seq] for seq in sequences], dtype=np.float32
    )  # (B, T, C, H, W)
    act_seqs = np.array(
        [[step[1] for step in seq] for seq in sequences], dtype=np.int64
    )  # (B, T)
    rew_seqs = np.array(
        [[step[2] for step in seq] for seq in sequences], dtype=np.float32
    )  # (B, T)
    nxt_seqs = np.array(
        [[step[3] for step in seq] for seq in sequences], dtype=np.float32
    )  # (B, T, C, H, W)
    don_seqs = np.array(
        [[step[4] for step in seq] for seq in sequences], dtype=np.float32
    )  # (B, T)

    obs_t = torch.from_numpy(obs_seqs).to(device)     # (B, T, C, H, W)
    act_t = torch.from_numpy(act_seqs).to(device)     # (B, T)
    rew_t = torch.from_numpy(rew_seqs).to(device)     # (B, T)
    nxt_t = torch.from_numpy(nxt_seqs).to(device)     # (B, T, C, H, W)
    don_t = torch.from_numpy(don_seqs).to(device)     # (B, T)

    # Reconstruct recurrent context without gradients. The learning graph then
    # begins at an explicit TBPTT boundary after burn-in.
    total_td_loss = torch.tensor(0.0, device=device)
    total_spikes = torch.tensor(0.0, device=device)

    hidden_policy: Optional[tuple] = None
    hidden_target: Optional[tuple] = None

    with torch.no_grad():
        for t in range(burn_in):
            _, _, hidden_policy = policy_net(obs_t[:, t], hidden_policy)
            _, _, hidden_target = target_net(obs_t[:, t], hidden_target)

    if hidden_policy is not None:
        hidden_policy = tuple(value.detach() for value in hidden_policy)
    if hidden_target is not None:
        hidden_target = tuple(value.detach() for value in hidden_target)

    for t in range(burn_in, window_len):
        obs_frame = obs_t[:, t]  # (B, C, H, W)
        nxt_frame = nxt_t[:, t]  # (B, C, H, W)
        act_frame = act_t[:, t]  # (B,)
        rew_frame = rew_t[:, t]  # (B,)
        don_frame = don_t[:, t]  # (B,)

        q_values, spikes, hidden_policy = policy_net(obs_frame, hidden_policy)
        chosen_q = q_values.gather(1, act_frame.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # Advance the target state with s_t, then evaluate s_(t+1) from
            # that history. Keep the s_t state for the next loop iteration so
            # each observation is incorporated exactly once in the trajectory.
            _, _, target_at_current = target_net(obs_frame, hidden_target)
            next_q, _, _ = target_net(nxt_frame, target_at_current)
            hidden_target = target_at_current
            max_next_q = next_q.max(1).values  # (B,)

        targets = bellman_targets(rew_frame, max_next_q, don_frame, gamma)
        total_td_loss = total_td_loss + criterion(chosen_q, targets)
        total_spikes = total_spikes + spikes

    td_loss = total_td_loss / seq_len
    opportunities = getattr(policy_net, "spike_opportunities_per_sample", None)
    if opportunities is None:
        spike_rate = torch.tensor(0.0, device=device)
    elif opportunities <= 0:
        raise ValueError("spike_opportunities_per_sample must be positive.")
    else:
        spike_rate = total_spikes / (batch_size * seq_len * opportunities)
    sparsity_loss = spike_rate * sparsity_weight
    loss = td_loss + sparsity_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Soft target update
# ---------------------------------------------------------------------------

def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.lerp_(pp, tau)
        for tb, pb in zip(target_net.buffers(), policy_net.buffers()):
            tb.copy_(pb)


def bellman_targets(
    rewards: torch.Tensor,
    max_next_q: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute terminal-aware one-step DQN targets."""
    return rewards + gamma * max_next_q * (1.0 - dones)


def select_action(
    observation: np.ndarray,
    policy_net: nn.Module,
    hidden: Optional[tuple],
    epsilon: float,
    action_size: int,
    device: torch.device,
) -> tuple[int, tuple, float]:
    """Advance recurrent state, then choose a greedy or random action."""
    tensor = torch.from_numpy(observation).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values, spikes, new_hidden = policy_net(tensor, hidden)
    if random.random() > epsilon:
        action = int(q_values.argmax(1).item())
    else:
        action = random.randrange(action_size)
    return action, new_hidden, float(spikes.item())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    env_class,
    config_path: Path,
    policy_net: nn.Module,
    device: torch.device,
    episodes: int,
    seeds: tuple[int, ...],
    episode_timeout: Optional[int] = None,
) -> dict:
    """Run ``episodes`` greedy episodes per seed and return aggregate stats."""
    was_training = policy_net.training
    policy_net.eval()
    all_rewards: list[float] = []
    all_kills: list[float] = []
    all_completions: list[bool] = []
    all_sectors: list[int] = []
    all_lengths: list[int] = []

    try:
        for seed in seeds:
            with env_class(
                config_file=config_path,
                render=False,
                seed=seed,
                episode_timeout=episode_timeout,
            ) as env:
                for _ in range(episodes):
                    obs = env.reset()
                    hidden: Optional[tuple] = None
                    done = False
                    ep_reward = 0.0
                    ep_steps = 0
                    while not done:
                        t = torch.from_numpy(obs).unsqueeze(0).to(device)
                        with torch.no_grad():
                            q_values, _, hidden = policy_net(t, hidden)
                        action = q_values.argmax(1).item()
                        obs, reward, done = env.step(action)
                        ep_reward += reward
                        ep_steps += 1
                    info = env.diagnostics()
                    all_rewards.append(ep_reward)
                    all_kills.append(info.get("kill_count", 0.0))
                    all_completions.append(bool(info.get("exit_triggered", False)))
                    all_sectors.append(info.get("sectors_visited", 0))
                    all_lengths.append(ep_steps)
    finally:
        policy_net.train(was_training)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "median_reward": float(np.median(all_rewards)),
        "completion_rate": float(np.mean(all_completions)),
        "mean_kills": float(np.mean(all_kills)),
        "mean_sectors_visited": float(np.mean(all_sectors)),
        "mean_episode_length": float(np.mean(all_lengths)),
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a full-level DOOM agent with a cross-frame recurrent SNN."
    )
    p.add_argument("--model", choices=LEVEL_MODELS, default="crsnn",
                   help=(
                       "Architecture: level_cnn, level_drqn, crsnn (default), "
                       "or rsnn_v2."
                   ))
    p.add_argument("--episodes", type=int, default=NUM_EPISODES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN,
                   help="TBPTT sequence window length.")
    p.add_argument("--burn-in", type=int, default=BURN_IN,
                   help="No-gradient frames preceding each learning window.")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--eps-start", type=float, default=EPS_START)
    p.add_argument("--eps-end", type=float, default=EPS_END)
    p.add_argument("--eps-decay", type=float, default=EPS_DECAY)
    p.add_argument("--learning-rate", type=float, default=LR)
    p.add_argument("--memory-size", type=int, default=MEMORY_SIZE,
                   help="Maximum completed transitions retained in replay.")
    p.add_argument("--replay-start-size", type=int, default=REPLAY_START_SIZE)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--sparsity-weight", type=float, default=SPARSITY_WEIGHT)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                   help="Episodes between resumable latest checkpoints.")
    p.add_argument("--episode-timeout", type=int, default=None,
                   help="Override scenario timeout in tics (useful for smoke tests).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--render", action="store_true",
                   help="Open ViZDoom window during training (slow).")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate the scenario/model contract, then exit.")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not 0 <= args.gamma <= 1:
        raise ValueError("--gamma must be in [0, 1].")
    if not 0 <= args.eps_end <= args.eps_start <= 1:
        raise ValueError("Epsilon values must satisfy 0 <= end <= start <= 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2.")
    if args.burn_in < 0:
        raise ValueError("--burn-in must be >= 0.")
    if args.memory_size < args.batch_size:
        raise ValueError("--memory-size must be >= --batch-size.")
    if args.memory_size < args.seq_len + args.burn_in:
        raise ValueError("--memory-size must be >= --seq-len + --burn-in.")
    if args.eval_interval <= 0 or args.eval_episodes <= 0:
        raise ValueError("Evaluation interval and episode count must be positive.")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive.")
    if args.episode_timeout is not None and args.episode_timeout <= 0:
        raise ValueError("--episode-timeout must be positive.")
    if not LEVEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Level config not found: {LEVEL_CONFIG_PATH}. "
            "Ensure scenarios/level01.cfg is present."
        )
    wad = PROJECT_ROOT / "freedoom2.wad"
    if not wad.exists():
        raise FileNotFoundError(
            f"FreeDoom WAD not found: {wad}\n"
            "Download from https://freedoom.github.io/download.html "
            "and place freedoom2.wad in the project root."
        )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seeds(args.seed)
    device = resolve_device(args.device)

    if args.validate_only:
        model = build_model(
            args.model,
            action_size=AGENT_CONTRACT.action_size,
            in_channels=FRAME_STACK,
        ).to(device)
        model.eval()
        with DoomLevelEnvironment(
            config_file=LEVEL_CONFIG_PATH,
            render=args.render,
            seed=args.seed,
            episode_timeout=args.episode_timeout,
        ) as validation_env:
            observation = validation_env.reset()
            tensor = torch.from_numpy(observation).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, _, hidden = model(tensor, None)
            if q_values.shape != (1, validation_env.num_actions):
                raise RuntimeError(
                    "Model/environment action contract mismatch: "
                    f"model returned {tuple(q_values.shape)}, "
                    f"environment exposes {validation_env.num_actions} actions."
                )
            if hidden is None:
                raise RuntimeError("Recurrent model did not return hidden state.")
        print(
            "Validation passed: scenario initialized, observation shape "
            f"{observation.shape}, and Q-value shape {tuple(q_values.shape)}."
        )
        return

    run_id = args.run_id or make_run_id(args.seed)
    model_dir, tensorboard_dir = experiment_directories(args.model, run_id)

    if args.resume is None and (model_dir.exists() or tensorboard_dir.exists()):
        raise FileExistsError(
            f"Run '{run_id}' already exists. "
            "Choose a different --run-id or pass --resume."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    config_path = LEVEL_CONFIG_PATH

    # Choose environment class.
    EnvClass = DoomLevelEnvironment
    action_size = AGENT_CONTRACT.action_size
    in_channels = FRAME_STACK

    config = LevelTrainingConfig(
        model=args.model,
        episodes=args.episodes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        burn_in=args.burn_in,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        learning_rate=args.learning_rate,
        memory_size=args.memory_size,
        replay_start_size=args.replay_start_size,
        tau=args.tau,
        sparsity_weight=args.sparsity_weight,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seeds=tuple(args.eval_seeds),
        checkpoint_interval=args.checkpoint_interval,
        episode_timeout=args.episode_timeout,
        seed=args.seed,
        config_path=str(config_path),
        run_id=run_id,
        action_size=action_size,
        in_channels=in_channels,
        agent_contract_hash=ACTION_CATALOGUE_HASH,
        preprocessing_version=PREPROCESSING_VERSION,
        action_names=ACTION_NAMES,
    )
    print(f"Model      : {args.model.upper()}")
    print(f"Device     : {device}")
    print(f"Run ID     : {run_id}")
    print(f"Checkpoints: {model_dir}")
    print(f"TensorBoard: {tensorboard_dir}")

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    env = None
    policy_net = None
    optimizer = None
    last_completed_episode = -1
    steps_done = 0
    best_eval_reward = -float("inf")
    best_completion_rate = -float("inf")
    try:
        # Build networks.
        policy_net = build_model(
            args.model, action_size=action_size, in_channels=in_channels
        ).to(device)
        target_net = build_model(
            args.model, action_size=action_size, in_channels=in_channels
        ).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
        criterion = nn.SmoothL1Loss()
        memory = EpisodeReplayMemory(
            capacity=args.memory_size,
            seq_len=args.seq_len,
            burn_in=args.burn_in,
        )

        start_episode = 0

        if args.resume is not None:
            state = load_checkpoint(
                args.resume,
                policy_net,
                device,
                optimizer,
                expected_config={
                    "model": args.model,
                    "action_size": action_size,
                    "in_channels": in_channels,
                },
            )
            start_episode = int(state.get("episode", -1)) + 1
            steps_done = int(state.get("global_step", 0))
            best_eval_reward = float(state.get("best_eval_mean", -float("inf")))
            best_completion_rate = float(
                state.get(
                    "best_eval_completion",
                    state.get("best_eval_success", -float("inf")),
                )
            )
            print(f"Resuming at episode {start_episode}, step {steps_done}.")

        if start_episode >= args.episodes:
            raise ValueError(
                f"Checkpoint already at episode {start_episode}; "
                f"set --episodes > {start_episode}."
            )

        # Write the resolved configuration only after a resume checkpoint has
        # passed compatibility validation.
        (model_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )

        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        policy_net.train()

        env = DoomLevelEnvironment(
            config_file=config_path,
            render=args.render,
            seed=args.seed,
            episode_timeout=args.episode_timeout,
        )

        for episode in range(start_episode, args.episodes):
            obs = env.reset()
            hidden: Optional[tuple] = None
            done = False
            ep_reward = 0.0
            ep_loss_sum = 0.0
            ep_loss_count = 0
            ep_steps = 0
            ep_spikes = 0.0
            started_at = time.perf_counter()

            while not done:
                # ε-greedy action selection.
                epsilon = args.eps_end + (args.eps_start - args.eps_end) * math.exp(
                    -steps_done / args.eps_decay
                )
                steps_done += 1

                action, hidden, spikes = select_action(
                    obs,
                    policy_net,
                    hidden,
                    epsilon,
                    action_size,
                    device,
                )
                ep_spikes += spikes

                next_obs, reward, done = env.step(action)
                memory.push(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward += reward
                ep_steps += 1

                loss = optimize_model(
                    memory, policy_net, target_net, optimizer, criterion,
                    batch_size=args.batch_size,
                    replay_start_size=args.replay_start_size,
                    gamma=args.gamma,
                    sparsity_weight=args.sparsity_weight,
                    device=device,
                )
                if loss is not None:
                    ep_loss_sum += loss
                    ep_loss_count += 1
                    soft_update(target_net, policy_net, args.tau)

            ep_duration = time.perf_counter() - started_at
            mean_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0.0

            # TensorBoard logging.
            writer.add_scalar("train/reward", ep_reward, episode)
            writer.add_scalar("train/loss", mean_loss, episode)
            writer.add_scalar("train/epsilon", epsilon, episode)
            writer.add_scalar("train/steps", ep_steps, episode)
            writer.add_scalar("train/optimization_updates", ep_loss_count, episode)
            writer.add_scalar("train/episode_duration_seconds", ep_duration, episode)
            writer.add_scalar("train/spike_rate",
                              ep_spikes / (
                                  max(ep_steps, 1)
                                  * getattr(
                                      policy_net,
                                      "spike_opportunities_per_sample",
                                      1,
                                  )
                              ), episode)
            writer.add_scalar("train/episodes_in_memory",
                              memory.num_episodes, episode)
            writer.add_scalar("train/replay_memory_mib",
                              memory.estimated_bytes / (1024 ** 2), episode)

            if (episode + 1) % 10 == 0:
                print(
                    f"[{episode+1:>5}/{args.episodes}] "
                    f"reward={ep_reward:+.2f}  loss={mean_loss:.4f}  "
                    f"epsilon={epsilon:.3f}  steps={ep_steps}  "
                    f"mem_ep={memory.num_episodes}  "
                    f"t={ep_duration:.1f}s"
                )

            # Periodic evaluation.
            if (
                (episode + 1) % args.eval_interval == 0
                or episode + 1 == args.episodes
            ):
                print(f"  -> Evaluating at episode {episode+1}...")
                eval_stats = evaluate_policy(
                    DoomLevelEnvironment,
                    config_path,
                    policy_net,
                    device,
                    episodes=args.eval_episodes,
                    seeds=tuple(args.eval_seeds),
                    episode_timeout=args.episode_timeout,
                )
                mean_r = eval_stats["mean_reward"]
                compl = eval_stats["completion_rate"]
                for k, v in eval_stats.items():
                    writer.add_scalar(f"eval/{k}", v, episode)
                print(
                    f"    eval: reward={mean_r:+.2f}  "
                    f"completion={compl:.1%}  "
                    f"kills={eval_stats['mean_kills']:.1f}  "
                    f"sectors={eval_stats['mean_sectors_visited']:.0f}"
                )

                # Save best checkpoint based on completion rate, then reward.
                is_best = (
                    compl > best_completion_rate
                    or (compl == best_completion_rate and mean_r > best_eval_reward)
                )
                if is_best:
                    best_completion_rate = compl
                    best_eval_reward = mean_r
                    save_checkpoint(
                        model_dir / "best.pth",
                        policy_net, optimizer,
                        episode=episode,
                        global_step=steps_done,
                        best_eval_mean=best_eval_reward,
                        best_eval_completion=best_completion_rate,
                        config=asdict(config),
                    )
                    print(f"     New best saved (completion={compl:.1%})")

            if (
                (episode + 1) % args.checkpoint_interval == 0
                or episode + 1 == args.episodes
            ):
                save_checkpoint(
                    model_dir / "latest.pth",
                    policy_net,
                    optimizer,
                    episode=episode,
                    global_step=steps_done,
                    best_eval_mean=best_eval_reward,
                    best_eval_completion=best_completion_rate,
                    config=asdict(config),
                )
                writer.flush()

            last_completed_episode = episode

        print("\nTraining complete.")
        print(f"Best completion rate: {best_completion_rate:.1%}")
        print(f"Best mean reward    : {best_eval_reward:+.2f}")

    except KeyboardInterrupt:
        if policy_net is not None and optimizer is not None:
            save_checkpoint(
                model_dir / "latest.pth",
                policy_net,
                optimizer,
                episode=last_completed_episode,
                global_step=steps_done,
                best_eval_mean=best_eval_reward,
                best_eval_completion=best_completion_rate,
                config=asdict(config),
            )
            writer.flush()
            print(
                "\nTraining interrupted; recoverable state saved to "
                f"{model_dir / 'latest.pth'}."
            )
        raise
    finally:
        writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

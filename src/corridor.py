"""Independent corridor curriculum, visual evaluation and environment smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from corridor_env import (ACTION_NAMES, SCENARIO_VERSION, CorridorEnvironment,
                          allowed_actions as stage_actions, case_geometry)
from diagnose import SpikeActivityRecorder
from networks import build_model
from project_paths import MODEL_ROOT, REPORT_ROOT, RUN_ROOT
from runtime_utils import resolve_device, set_random_seeds
from train import ReplayMemory, optimize_model, soft_update


def greedy_action(q_values, available_actions):
    """Select among actions permitted by the current task stage."""
    return max(available_actions, key=lambda index: float(q_values[index]))


def validation_score(summary):
    """Prefer success on both mirrored turns over one-sided high reward."""
    by_turn = summary['by_turn']
    worst_turn = min(by_turn[turn]['complete'] / max(1, by_turn[turn]['total'])
                     for turn in ('left', 'right'))
    return (worst_turn, summary['completion_rate'], summary['mean_reward'])


class ExpertDataset:
    """Image/action labels from expert routes and learner-visited states."""

    def __init__(self):
        self.frames = []
        self.labels = []
        self.turns = []
        self.recoveries = []
        self.groups = {}

    def __len__(self):
        return len(self.labels)

    def add(self, observation, action, mirror, *, recovery=False):
        if action not in (0, 3, 4):
            raise ValueError('Navigation expert labels must be forward or turn actions')
        index = len(self.labels)
        turn = 'left' if mirror == 1 else 'right'
        self.frames.append(np.rint(np.clip(observation, 0, 1) * 255).astype(np.uint8))
        self.labels.append(action)
        self.turns.append(turn)
        self.recoveries.append(bool(recovery))
        self.groups.setdefault((action, turn, bool(recovery)), []).append(index)

    def sample(self, rng, batch_size, device):
        """Balance expert actions and route sides; prefer recovery frames when present."""
        actions = rng.choice((0, 3, 4), size=batch_size, p=(0.6, 0.2, 0.2))
        turns = rng.choice(('left', 'right'), size=batch_size)
        recoveries = rng.random(batch_size) < 0.5
        indices = []
        for action, turn, recovery in zip(actions, turns, recoveries):
            alternatives = ((int(action), turn, bool(recovery)),
                            (int(action), turn, not bool(recovery)),
                            (int(action), 'right' if turn == 'left' else 'left', bool(recovery)),
                            (int(action), 'right' if turn == 'left' else 'left', not bool(recovery)))
            candidates = next((self.groups[key] for key in alternatives if self.groups.get(key)), None)
            if candidates is None:
                raise ValueError(f'Expert dataset has no examples for action {action}')
            indices.append(int(rng.choice(candidates)))
        frames = torch.from_numpy(np.stack([self.frames[index] for index in indices])).unsqueeze(1)
        labels = torch.as_tensor([self.labels[index] for index in indices], dtype=torch.long)
        return frames.to(device=device, dtype=torch.float32) / 255.0, labels.to(device)

    def counts(self):
        return {'total': len(self),
                'by_turn': {turn: self.turns.count(turn) for turn in ('left', 'right')},
                'by_action': {ACTION_NAMES[action]: self.labels.count(action)
                              for action in (0, 3, 4)},
                'recovery': sum(self.recoveries)}


def imitation_update(model, optimizer, dataset, rng, device, batch_size):
    frames, labels = dataset.sample(rng, batch_size, device)
    q_values, _ = model(frames)
    loss = torch.nn.functional.cross_entropy(q_values[:, :5] / model.num_steps, labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def dashboard(env, observation, q, spikes, action, info, title, available_actions):
    """A saved video uses the same dashboard as interactive verification."""
    canvas = np.full((600, 1000, 3), 24, np.uint8)
    state = env.game.get_state()
    if state is not None:
        raw = cv2.cvtColor(state.screen_buffer, cv2.COLOR_GRAY2BGR)
        canvas[40:400, :480] = cv2.resize(raw, (480, 360))
    small = cv2.cvtColor(np.uint8(np.clip(observation, 0, 1)*255), cv2.COLOR_GRAY2BGR)
    canvas[40:208, 490:658] = cv2.resize(small, (168,168), interpolation=cv2.INTER_NEAREST)
    def text(message, x, y, color=(220,220,220), scale=0.48):
        cv2.putText(canvas, message, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text(title, 12, 25, scale=0.6)
    text('84x84 policy input', 490, 230)
    text(f'Action: {ACTION_NAMES[action]}', 12, 425)
    required_kills = 2 if env.stage == 'combat' else 0
    text(f"Kills {info['kills']:.0f}/{required_kills} | Health {info['health']:.0f} | Ammo {info['ammo']:.0f}", 12, 450)
    text(f"Outcome: {info.get('outcome', 'running')} | tics {info['tics']}", 12, 475)
    text(f"Route progress: {100*info.get('progress', env.potential(info)):.1f}%", 12, 500)
    text(f"No-progress steps: {info.get('no_progress_steps', 0)}", 12, 525)
    text('Q-values (not probabilities)', 690, 45)
    extent = max(float(np.abs(q).max()), 1e-6)
    for i, (name, value) in enumerate(zip(ACTION_NAMES, q)):
        y = 75 + i*34
        label = f'{name}: {value:.2f}' if i in available_actions else f'{name}: disabled'
        text(label, 690, y, (220,220,220) if i in available_actions else (110,110,110))
        if i not in available_actions:
            continue
        end = 840 + int(140 * value/extent)
        cv2.line(canvas, (840,y+10), (end,y+10), (60,200,90) if i==action else (170,100,80), 6)
    text('Spike rates per neural update', 690, 335)
    for i, (name, layer) in enumerate(spikes.items()):
        text(f"{name}: {100*layer['rate']:.1f}%", 690, 360+i*24)
    # Debug map is privileged information shown to the viewer, never the policy.
    vertices, _, mirror = case_geometry(env.case)
    def point(x,y):
        return (500+int((x+128)*0.17), 580-int((mirror*y+128)*0.24))
    polygon = np.array([point(x,y) for x,y in vertices], np.int32)
    cv2.polylines(canvas, [polygon], True, (150,150,150), 2)
    cv2.circle(canvas, point(info['x'], mirror*info['y_progress']), 4, (0,220,220), -1)
    cv2.circle(canvas, point(640, mirror*660), 6, (0,220,0), -1)
    text('Debug map / green exit', 490, 320)
    return canvas


def scripted_action(env, info):
    """Privileged geometry controller ONLY for verifying navigation mechanics."""
    target = (640, 0) if info['x'] < 590 else (640, 690)
    dy = env.mirror * (target[1] - info['y_progress'])
    angle = math.degrees(math.atan2(dy, target[0] - info['x'])) % 360
    delta = (angle - info['angle'] + 180) % 360 - 180
    if abs(delta) > 8:
        return 3 if delta > 0 else 4
    return 0


def evaluate(model, device, *, stage, episodes, seed, output=None, render=False,
             scripted=False, timeout=None, delay=0.0, available_actions=None):
    available_actions = tuple(available_actions if available_actions is not None else stage_actions(stage))
    rows = []
    writer = None
    recorder = SpikeActivityRecorder(model) if model is not None else None
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        writer = cv2.VideoWriter(str(output/'gameplay.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                 35/4, (1000,600))
        if not writer.isOpened():
            raise RuntimeError('OpenCV could not open the MJPG video writer')
    try:
        with CorridorEnvironment(stage, 'eval', seed, timeout=timeout,
                                 allowed_actions_override=available_actions) as env:
            for episode in range(episodes):
                case = 8 + episode % 8  # Paired turns and sides; reproducible schedule.
                observation = env.reset(case)
                done, reward, steps, stalls, shots = False, 0.0, 0, 0, 0
                latencies, actions, trajectory = [], [], []
                first_bend = None
                reward_totals = {name: 0.0 for name in
                                 ('living', 'kill', 'completion', 'death', 'timeout', 'no_progress', 'alignment')}
                max_progress = 0.0
                info = env.info()
                while not done:
                    q = np.zeros(len(ACTION_NAMES), np.float32)
                    spike_rates = {}
                    if scripted:
                        action = scripted_action(env, info)
                    else:
                        recorder.reset()
                        tensor = torch.from_numpy(observation)[None,None].to(device)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.no_grad():
                            values, _ = model(tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                        latencies.append(1000*(time.perf_counter()-start))
                        q = values[0].cpu().numpy()
                        action = greedy_action(q, available_actions)
                        spike_rates = recorder.snapshot()
                    if stage == 'navigation' and not scripted and first_bend is None:
                        expert_action = scripted_action(env, info)
                        if (info['x'] >= 590 and info['y_progress'] < 200
                                and expert_action in (3, 4) and action in (3, 4)):
                            other = [index for index in available_actions if index != expert_action]
                            first_bend = {
                                'step': steps + 1,
                                'expert_action': ACTION_NAMES[expert_action],
                                'policy_action': ACTION_NAMES[action],
                                'correct': action == expert_action,
                                'q_margin': float(q[expert_action] - max(q[index] for index in other)),
                                'x': info['x'], 'y_progress': info['y_progress'],
                            }
                    if writer is not None or render:
                        frame = dashboard(env, observation, q, spike_rates, action, info,
                                          f'{"SCRIPTED ENV TEST" if scripted else "SNN EVALUATION"} '
                                          f'| {stage} | case {case} | episode {episode+1}',
                                          available_actions)
                        if writer is not None:
                            writer.write(frame)
                            if steps == 0 and episode == 0:
                                cv2.imwrite(str(output/'dashboard.png'), frame)
                        if render:
                            cv2.imshow('Corridor verification (Q to stop)', frame)
                            if cv2.waitKey(max(1, int(delay*1000))) & 0xff == ord('q'):
                                raise KeyboardInterrupt('Evaluation stopped by viewer')
                    observation, task_reward, done, info = env.step(action)
                    reward += task_reward
                    steps += 1
                    stalls += int(info['stalled'])
                    shots += info['shots']
                    max_progress = max(max_progress, info['best_progress'])
                    for name, component in info['reward_components'].items():
                        reward_totals[name] += component
                    reward_totals['alignment'] += info.get('alignment_shaping', 0.0)
                    actions.append(action)
                    trajectory.append([info['x'], env.mirror*info['y_progress']])
                if writer is not None:
                    final = dashboard(env, observation, q, spike_rates, action, info,
                                      f'Episode {episode+1}: {info["outcome"]}', available_actions)
                    for _ in range(9):
                        writer.write(final)
                row = {'episode': episode+1, 'case': case, 'turn': 'left' if env.mirror==1 else 'right',
                       'outcome': info['outcome'], 'reward': reward, 'steps': steps,
                       'kills': info['kills'], 'shots': shots, 'stalled_moves': stalls,
                       'health': info['health'], 'inference_ms': float(np.mean(latencies)) if latencies else None,
                       'actions': actions, 'trajectory': trajectory,
                       'reward_components': reward_totals,
                       'max_progress': max_progress, 'first_bend': first_bend}
                rows.append(row)
                print(f"Episode {episode+1}/{episodes} | {row['turn']} | {row['outcome']} | "
                      f"kills {row['kills']:.0f} | steps {steps} | stalls {stalls}", flush=True)
    finally:
        if recorder is not None:
            recorder.close()
            model.train(was_training)
        if writer is not None:
            writer.release()
        if render:
            cv2.destroyAllWindows()
    summary = {'completion_rate': float(np.mean([r['outcome']=='complete' for r in rows])),
               'mean_reward': float(np.mean([r['reward'] for r in rows])),
               'mean_kills': float(np.mean([r['kills'] for r in rows])),
               'timeout_rate': float(np.mean([r['outcome']=='timeout' for r in rows])),
               'death_rate': float(np.mean([r['outcome']=='death' for r in rows])),
               'mean_shots': float(np.mean([r['shots'] for r in rows])),
               'mean_stalled_moves': float(np.mean([r['stalled_moves'] for r in rows])),
               'mean_max_progress': float(np.mean([r['max_progress'] for r in rows])),
               'by_turn': {
                   turn: {'complete': sum(r['outcome']=='complete' for r in rows if r['turn']==turn),
                          'total': sum(r['turn']==turn for r in rows)}
                   for turn in ('left', 'right')
               }}
    action_counts = {name: sum(action == index for row in rows for action in row['actions'])
                     for index, name in enumerate(ACTION_NAMES)}
    summary['action_counts'] = action_counts
    summary['first_bend'] = {
        'observed': sum(row['first_bend'] is not None for row in rows),
        'correct': sum(row['first_bend'] is not None and row['first_bend']['correct'] for row in rows),
        'by_turn': {
            turn: {
                'observed': sum(row['turn'] == turn and row['first_bend'] is not None for row in rows),
                'correct': sum(row['turn'] == turn and row['first_bend'] is not None
                               and row['first_bend']['correct'] for row in rows),
            }
            for turn in ('left', 'right')
        },
    }
    report = {'stage': stage, 'seed': seed, 'scenario_version': SCENARIO_VERSION,
              'controller': 'privileged_scripted_environment_test' if scripted else 'trained_snn',
              'split': 'eval', 'action_names': ACTION_NAMES,
              'allowed_actions': available_actions, 'summary': summary, 'episodes': rows}
    if output is not None:
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        plot_report(report, output/'summary.png')
        if stage == 'navigation' and not scripted:
            plot_first_bend(report, output/'turn-decisions.png')
    return report


def plot_report(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13,9))
    rows = report['episodes']
    colors = [{'complete':'seagreen','death':'firebrick','timeout':'darkorange'}[r['outcome']] for r in rows]
    axes[0,0].bar(range(1,len(rows)+1), [r['steps'] for r in rows], color=colors)
    axes[0,0].set(xlabel='Episode', ylabel='Decisions', title='Green complete / orange timeout / red death')
    for row, color in zip(rows,colors):
        xy = np.array(row['trajectory'])
        axes[0,1].plot(xy[:,0],xy[:,1],color=color,alpha=.7)
    for case in (8,9):
        vertices = case_geometry(case)[0]
        xy = np.array(vertices+[vertices[0]])
        axes[0,1].plot(xy[:,0],xy[:,1],color='grey',linewidth=.7)
    axes[0,1].set(title='Trajectories in world coordinates', xlabel='x', ylabel='y')
    axes[0,1].set_aspect('equal')
    turns = report['summary']['by_turn']
    axes[1,0].bar(['Left', 'Right'],
                  [turns[direction]['complete']/max(1,turns[direction]['total']) for direction in ('left','right')],
                  color=['steelblue','mediumpurple'])
    axes[1,0].set(title='Completion by turn direction', ylabel='Completion rate', ylim=(0,1))
    counts = report['summary']['action_counts']
    axes[1,1].barh(list(counts), list(counts.values()), color='darkcyan')
    axes[1,1].set(title='Action frequency', xlabel='Decisions')
    fig.suptitle(f"{report['controller']} | {report['stage']} | seed {report['seed']}")
    fig.tight_layout()
    fig.savefig(destination,dpi=150)
    plt.close(fig)


def plot_first_bend(report, destination):
    """Compare the first policy turn at the bend with the expert direction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['episodes']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    probes = report['summary']['first_bend']
    for index, turn in enumerate(('left', 'right')):
        group = probes['by_turn'][turn]
        axes[0].bar(index, group['correct'] / max(1, group['observed']),
                    color='steelblue' if turn == 'left' else 'mediumpurple')
        axes[0].text(index, group['correct'] / max(1, group['observed']) + .03,
                     f"{group['correct']}/{group['observed']}", ha='center')
    axes[0].set(xticks=(0, 1), xticklabels=('Left', 'Right'), ylim=(0, 1.15),
                ylabel='Correct turns / observed policy turns', title='First policy turn at bend')
    for index, row in enumerate(rows, 1):
        probe = row['first_bend']
        if probe is None:
            axes[1].scatter(index, 0, marker='x', color='grey')
        else:
            axes[1].bar(index, probe['q_margin'],
                        color='seagreen' if probe['correct'] else 'firebrick')
    axes[1].axhline(0, color='grey', linewidth=1)
    axes[1].set(xlabel='Episode (grey X = no turn at bend)', ylabel='Expert Q − best other Q',
                title='Decision margin at first bend', xlim=(.3, len(rows) + .7))
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_validation(report, directory, stem):
    (directory/f'{stem}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_report(report, directory/f'{stem}.png')
    if report['stage'] == 'navigation' and report['controller'] == 'trained_snn':
        plot_first_bend(report, directory/f'{stem}-turns.png')


def load_policy(path, device):
    payload = torch.load(path,map_location=device,weights_only=True)
    if payload.get('task') != 'corridor' or tuple(payload.get('action_names',())) != ACTION_NAMES:
        raise ValueError('Use a corridor checkpoint; basic-scenario checkpoints have incompatible actions')
    if payload.get('scenario_version') != SCENARIO_VERSION:
        raise ValueError('Checkpoint scenario version differs from this corridor version')
    model = build_model(payload['model'],action_size=len(ACTION_NAMES),num_steps=payload['num_steps']).to(device)
    model.load_state_dict(payload['model_state_dict'])
    return model, payload


def bootstrap_navigation(model, env, memory, optimizer, device, rng, *, episodes, updates,
                         batch_size, dataset=None):
    """Learn from a scripted route; coordinates provide labels, never policy inputs."""
    dataset = dataset if dataset is not None else ExpertDataset()
    for demonstration in range(episodes):
        observation = env.reset(case=demonstration % 8)
        done = False
        while not done:
            action = scripted_action(env, env.info())
            dataset.add(observation, action, env.mirror)
            next_state, reward, done, info = env.step(action)
            memory.push(observation, action, (reward + info['shaping']) / 25,
                        next_state, done)
            observation = next_state
        if info['outcome'] != 'complete':
            raise RuntimeError('Scripted corridor demonstration did not reach the exit')
    if any(not any(label == action for label in dataset.labels) for action in (0, 3, 4)):
        raise RuntimeError('Demonstrations must include forward and both turn directions')
    losses = []
    model.train()
    for _ in range(updates):
        losses.append(imitation_update(model, optimizer, dataset, rng, device, batch_size))
    return {'episodes': episodes, 'samples': len(dataset),
            'actions': dataset.counts()['by_action'],
            'updates': updates, 'mean_loss': float(np.mean(losses))}


def collect_corrections(model, env, dataset, device, rng, *, episodes, max_steps,
                        teacher_probability, case_offset=0):
    """DAgger-style rollout: label learner-visited images with the expert action."""
    was_training = model.training
    model.eval()
    stats = {'episodes': episodes, 'samples': 0, 'disagreements': 0,
             'teacher_actions_taken': 0, 'completed': 0,
             'by_turn': {'left': 0, 'right': 0}}
    try:
        for rollout in range(episodes):
            observation = env.reset(case=(case_offset + rollout) % 8)
            turn = 'left' if env.mirror == 1 else 'right'
            for _ in range(max_steps):
                expert_action = scripted_action(env, env.info())
                with torch.no_grad():
                    q_values, _ = model(torch.from_numpy(observation)[None,None].to(device))
                learner_action = greedy_action(q_values[0], stage_actions('navigation'))
                disagreed = learner_action != expert_action
                dataset.add(observation, expert_action, env.mirror,
                            recovery=disagreed or env.no_progress_steps >= 3)
                stats['samples'] += 1
                stats['disagreements'] += int(disagreed)
                stats['by_turn'][turn] += 1
                use_teacher = rng.random() < teacher_probability
                action = expert_action if use_teacher else learner_action
                stats['teacher_actions_taken'] += int(use_teacher)
                observation, _, done, info = env.step(action)
                if done:
                    stats['completed'] += int(info['outcome'] == 'complete')
                    break
    finally:
        model.train(was_training)
    return stats


def train(args, device):
    # Separate task root and reject existing run IDs to protect all older models.
    if Path(args.run_id).name != args.run_id or args.run_id in ('.','..'):
        raise ValueError('run-id must be a single directory name')
    directory = MODEL_ROOT/'corridor'/args.run_id
    directory.mkdir(parents=True,exist_ok=False)
    active_actions = stage_actions(args.stage)
    set_random_seeds(args.seed)
    model = build_model(args.model,action_size=len(ACTION_NAMES),num_steps=args.num_steps).to(device)
    if args.init_checkpoint:
        model, payload = load_policy(args.init_checkpoint,device)
        if payload['model'] != args.model or payload['num_steps'] != args.num_steps:
            raise ValueError('Initialization architecture must match --model and --num-steps')
    target = build_model(args.model,action_size=len(ACTION_NAMES),num_steps=args.num_steps).to(device)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    memory = ReplayMemory(args.memory_size)
    criterion = torch.nn.SmoothL1Loss()
    rng = np.random.default_rng(args.seed)
    global_step, best, dagger_round = 0, (-1.0,-1.0,-float('inf')), 0
    expert_dataset = ExpertDataset() if args.demo_episodes else None
    writer = SummaryWriter(str(RUN_ROOT/'corridor'/args.run_id))
    config = vars(args).copy()
    config = {k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    (directory/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    validation_dir = REPORT_ROOT/'corridor'/args.run_id
    validation_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(episode):
        return {'task':'corridor','scenario_version':SCENARIO_VERSION,
                'action_names':ACTION_NAMES,'model':args.model,'num_steps':args.num_steps,
                'stage':args.stage,'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),'episode':episode,
                'global_step':global_step,'config':config,
                'allowed_actions':active_actions,'reward_version':2,
                'dagger_round':dagger_round}

    try:
        with CorridorEnvironment(args.stage,'train',args.seed,timeout=args.timeout) as env:
            if args.demo_episodes:
                if args.stage != 'navigation':
                    raise ValueError('Scripted demonstrations are available only in navigation')
                demo = bootstrap_navigation(
                    model, env, memory, optimizer, device, rng,
                    episodes=args.demo_episodes, updates=args.demo_updates,
                    batch_size=args.batch_size, dataset=expert_dataset,
                )
                target.load_state_dict(model.state_dict())
                (validation_dir/'demonstrations.json').write_text(
                    json.dumps(demo, indent=2), encoding='utf-8')
                writer.add_scalar('bootstrap/mean_loss',demo['mean_loss'],0)
                print(f'Demonstrations | {demo["samples"]} frames | '
                      f'{demo["updates"]} supervised updates | loss {demo["mean_loss"]:.4f}',
                      flush=True)
                report = evaluate(model,device,stage=args.stage,episodes=8,seed=10000,
                                  timeout=args.timeout,available_actions=active_actions)
                save_validation(report, validation_dir, 'bootstrap')
                score = validation_score(report['summary'])
                best = score
                payload = checkpoint_payload(0)
                payload['validation'] = report['summary']
                torch.save(payload,directory/'bootstrap.pth')
                torch.save(payload,directory/'best.pth')
                print(f'Bootstrap validation | {score[1]:.1%} complete | '
                      f'worst direction {score[0]:.1%}', flush=True)
            for dagger_round in range(1, args.dagger_rounds + 1):
                collected = collect_corrections(
                    model, env, expert_dataset, device, rng,
                    episodes=args.dagger_episodes, max_steps=args.dagger_max_steps,
                    teacher_probability=args.dagger_teacher_prob,
                    case_offset=(dagger_round - 1) * args.dagger_episodes,
                )
                losses = [imitation_update(model, optimizer, expert_dataset, rng, device,
                                           args.batch_size) for _ in range(args.dagger_updates)]
                target.load_state_dict(model.state_dict())
                collected['dataset'] = expert_dataset.counts()
                collected['supervised_updates'] = args.dagger_updates
                collected['mean_loss'] = float(np.mean(losses))
                (validation_dir/f'dagger-{dagger_round}-collection.json').write_text(
                    json.dumps(collected, indent=2), encoding='utf-8')
                writer.add_scalar('dagger/disagreement_rate',
                                  collected['disagreements'] / max(1, collected['samples']),
                                  dagger_round)
                writer.add_scalar('dagger/recovery_examples',
                                  collected['dataset']['recovery'], dagger_round)
                report = evaluate(model, device, stage=args.stage, episodes=8, seed=10000,
                                  timeout=args.timeout, available_actions=active_actions)
                save_validation(report, validation_dir, f'dagger-{dagger_round}')
                score = validation_score(report['summary'])
                writer.add_scalar('dagger/completion_rate', score[1], dagger_round)
                writer.add_scalar('dagger/worst_turn_completion', score[0], dagger_round)
                payload = checkpoint_payload(0)
                payload['validation'] = report['summary']
                torch.save(payload, directory/f'dagger-{dagger_round}.pth')
                if score > best:
                    best = score
                    torch.save(payload, directory/'best.pth')
                print(f'DAgger {dagger_round}/{args.dagger_rounds} | '
                      f'{collected["samples"]} corrective frames | '
                      f'{score[1]:.1%} complete | worst direction {score[0]:.1%}',
                      flush=True)
            for episode in range(args.episodes):
                observation = env.reset(case=episode%8)
                done, total, losses = False, 0.0, []
                reward_totals = {name: 0.0 for name in
                                 ('living', 'kill', 'completion', 'death', 'timeout', 'no_progress', 'shaping', 'alignment')}
                while not done:
                    epsilon = .1 + (args.epsilon_start - .1)*math.exp(-global_step/args.eps_decay)
                    if rng.random()<epsilon:
                        action = int(rng.choice(active_actions))
                    else:
                        with torch.no_grad():
                            q,_ = model(torch.from_numpy(observation)[None,None].to(device))
                        action = greedy_action(q[0], active_actions)
                    next_state,reward,done,info = env.step(action)
                    memory.push(observation,action,(reward+info['shaping'])/25,next_state,done)
                    for name, component in info['reward_components'].items():
                        reward_totals[name] += component
                    reward_totals['shaping'] += info['shaping']
                    reward_totals['alignment'] += info.get('alignment_shaping', 0.0)
                    observation = next_state
                    total += reward
                    global_step += 1
                    if global_step % args.train_every == 0:
                        expert_states = expert_actions = None
                        if expert_dataset is not None and args.expert_loss_weight > 0:
                            expert_states, expert_actions = expert_dataset.sample(
                                rng, args.batch_size, device)
                        loss = optimize_model(memory,model,target,optimizer,criterion,
                            batch_size=args.batch_size,replay_start_size=args.replay_start_size,
                            gamma=.99,sparsity_weight=1e-9,device=device,
                            allowed_actions=active_actions,
                            expert_states=expert_states, expert_actions=expert_actions,
                            expert_weight=args.expert_loss_weight if expert_dataset is not None else 0.0)
                        if loss is not None:
                            losses.append(loss)
                            soft_update(target,model,.01)
                writer.add_scalar('train/task_reward',total,episode+1)
                writer.add_scalar('train/epsilon',epsilon,episode+1)
                writer.add_scalar('train/max_progress',info['best_progress'],episode+1)
                for name, value in reward_totals.items():
                    writer.add_scalar(f'train/reward/{name}',value,episode+1)
                if losses:
                    writer.add_scalar('train/loss',float(np.mean(losses)),episode+1)
                print(f'Train {episode+1}/{args.episodes} | {info["outcome"]} | reward {total:.2f}',flush=True)
                payload = checkpoint_payload(episode+1)
                torch.save(payload,directory/'latest.pth')
                if (episode+1)%args.eval_interval==0 or episode+1==args.episodes:
                    report = evaluate(model,device,stage=args.stage,episodes=8,seed=10000,
                                      timeout=args.timeout,available_actions=active_actions)
                    save_validation(report, validation_dir, f'validation-{episode+1}')
                    summary = report['summary']
                    score = validation_score(summary)
                    writer.add_scalar('eval/completion_rate',score[1],episode+1)
                    writer.add_scalar('eval/worst_turn_completion',score[0],episode+1)
                    writer.add_scalar('eval/task_reward',score[2],episode+1)
                    for turn, result in summary['by_turn'].items():
                        writer.add_scalar(f'eval/{turn}_completion',
                                          result['complete']/max(1,result['total']),episode+1)
                    if score>best:
                        best=score
                        payload['validation']=summary
                        torch.save(payload,directory/'best.pth')
    finally:
        writer.close()
    print(f'Checkpoints: {directory}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','eval','smoke'])
    parser.add_argument('--stage',choices=['navigation','combat'],default='navigation')
    parser.add_argument('--model',choices=['snn','compact_snn','compact_conv_snn'],default='compact_snn')
    parser.add_argument('--num-steps',type=int,default=10)
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--init-checkpoint',type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--delay',type=float,default=.1)
    parser.add_argument('--timeout',type=int,default=None,
                        help='Override episode_timeout from scenarios/corridor.cfg')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--memory-size',type=int,default=2000)
    parser.add_argument('--replay-start-size',type=int,default=256)
    parser.add_argument('--train-every',type=int,default=4)
    parser.add_argument('--eval-interval',type=int,default=25)
    parser.add_argument('--eps-decay',type=float,default=20000)
    parser.add_argument('--epsilon-start',type=float,default=1.0)
    parser.add_argument('--demo-episodes',type=int,default=0,
                        help='Navigation-only scripted demonstrations before DQN training')
    parser.add_argument('--demo-updates',type=int,default=200)
    parser.add_argument('--dagger-rounds',type=int,default=0,
                        help='Rounds of learner-visited corrective imitation before DQN')
    parser.add_argument('--dagger-episodes',type=int,default=8,
                        help='Even number of balanced left/right rollouts per round')
    parser.add_argument('--dagger-updates',type=int,default=200)
    parser.add_argument('--dagger-max-steps',type=int,default=120,
                        help='Maximum decisions collected per corrective rollout')
    parser.add_argument('--dagger-teacher-prob',type=float,default=0.2,
                        help='Chance of executing expert action during corrective collection')
    parser.add_argument('--expert-loss-weight',type=float,default=0.0,
                        help='Supervised imitation loss weight during DQN updates')
    args = parser.parse_args()
    for key in ('episodes','num_steps','batch_size','memory_size','replay_start_size','train_every','eval_interval','eps_decay'):
        if getattr(args,key)<=0:
            parser.error(f'{key} must be positive')
    if args.timeout is not None and args.timeout<=0:
        parser.error('--timeout must be positive')
    if not 0.1 <= args.epsilon_start <= 1.0:
        parser.error('--epsilon-start must be between 0.1 and 1.0')
    if args.demo_episodes < 0 or args.demo_updates <= 0:
        parser.error('--demo-episodes must be non-negative and --demo-updates positive')
    if args.demo_episodes and (args.command != 'train' or args.stage != 'navigation'):
        parser.error('Demonstrations require train --stage navigation')
    if args.dagger_rounds < 0 or args.dagger_episodes <= 0 or args.dagger_updates <= 0 or args.dagger_max_steps <= 0:
        parser.error('DAgger rounds must be non-negative and its episodes, updates and max-steps positive')
    if not 0 <= args.dagger_teacher_prob <= 1 or args.expert_loss_weight < 0:
        parser.error('DAgger teacher probability must be in [0,1] and expert loss non-negative')
    if args.dagger_rounds and (args.command != 'train' or args.stage != 'navigation'
                              or args.demo_episodes < 2 or args.dagger_episodes % 2):
        parser.error('DAgger requires navigation training, demonstrations and even corrective episodes')
    if args.expert_loss_weight and (args.command != 'train' or args.stage != 'navigation'
                                   or not args.demo_episodes):
        parser.error('Expert loss requires navigation training with demonstrations')
    if args.memory_size<max(args.batch_size,args.replay_start_size):
        parser.error('Replay memory must fit batch-size and replay-start-size')
    if args.command=='train' and not args.run_id:
        parser.error('train requires --run-id')
    if args.command=='eval' and not args.checkpoint:
        parser.error('eval requires --checkpoint')
    if args.command=='smoke' and args.stage!='navigation':
        parser.error('The scripted smoke test only verifies navigation; use eval for combat')
    torch.set_num_threads(2)
    device=resolve_device(args.device)
    set_random_seeds(args.seed)
    if args.command=='train':
        train(args,device)
    else:
        model,payload=load_policy(args.checkpoint,device) if args.command=='eval' else (None,None)
        if payload and payload['stage']!=args.stage:
            parser.error('--stage must match checkpoint training stage')
        output=args.output or REPORT_ROOT/'corridor'/f'{args.command}-{time.time_ns()}'
        report=evaluate(model,device,stage=args.stage,episodes=args.episodes,seed=args.seed,
                        output=output,render=args.render,scripted=args.command=='smoke',
                        timeout=args.timeout,delay=args.delay,
                        available_actions=(payload.get('allowed_actions', tuple(range(len(ACTION_NAMES))))
                                           if payload else stage_actions(args.stage)))
        if args.checkpoint:
            report['checkpoint']=str(args.checkpoint.resolve())
            report['checkpoint_sha256']=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report['summary'],indent=2))
        print(f'Visual report: {output}')


if __name__=='__main__':
    main()

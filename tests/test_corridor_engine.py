"""Real-engine checks: generated maps load and actor counts match the task."""
import pytest
import numpy as np
import torch

from corridor_env import CORRIDOR_CONFIG_PATH, CorridorEnvironment, build_scenario
from corridor import ExpertDataset, bootstrap_navigation, collect_corrections
from networks import build_model
from train import ReplayMemory


@pytest.mark.parametrize('stage', ['navigation','combat'])
def test_every_corridor_map_loads_with_expected_enemies(stage):
    with CorridorEnvironment(stage, seed=123, debug_objects=True) as env:
        for case in range(16):
            observation=env.reset(case)
            assert observation.shape==(84,84)
            assert observation.min()>=0 and observation.max()<=1
            actors=env.game.get_state().objects
            assert sum(actor.name=='Zombieman' for actor in actors)==(2 if stage=='combat' else 0)
            assert not env.game.is_episode_finished()


def test_real_timeout_and_finished_episode_guard():
    with CorridorEnvironment('navigation',timeout=30) as env:
        env.reset(0)
        done=False
        while not done:
            _,_,done,info=env.step(3)
        assert info['outcome']=='timeout'
        with pytest.raises(RuntimeError,match='reset'):
            env.step(0)


def test_navigation_rejects_attack_and_logs_no_progress_penalty():
    with CorridorEnvironment('navigation',timeout=120) as env:
        env.reset(0)
        with pytest.raises(ValueError,match='unavailable'):
            env.step(5)
        with pytest.raises(ValueError,match='unavailable'):
            env.step(6)
        result=None
        for _ in range(25):
            _, _, done, result = env.step(3)
            if done:
                break
        assert result['reward_components']['no_progress'] == -0.2
        assert result['reward_components']['timeout'] == 0
        while not done:
            _, _, done, result = env.step(3)
        assert result['reward_components']['timeout'] == -25
        assert result['outcome']=='timeout'


def test_config_is_cwd_independent_and_timeout_override_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with CorridorEnvironment() as env:
        assert env.timeout == 1400
        assert env.reset(0).shape == (84,84)
    with CorridorEnvironment(timeout=90) as env:
        assert env.timeout == 90


def test_reordered_buttons_fail_before_engine_start(tmp_path):
    config = CORRIDOR_CONFIG_PATH.read_text(encoding='utf-8')
    config = config.replace('MOVE_FORWARD', 'TEMP_BUTTON').replace('MOVE_LEFT', 'MOVE_FORWARD')
    config = config.replace('TEMP_BUTTON', 'MOVE_LEFT')
    path = tmp_path/'wrong-buttons.cfg'
    path.write_text(config, encoding='utf-8')
    with pytest.raises(ValueError, match='button order mismatch'):
        CorridorEnvironment(config_file=path)


def test_config_timeout_is_used_without_python_override(tmp_path):
    config = CORRIDOR_CONFIG_PATH.read_text(encoding='utf-8')
    build_scenario(tmp_path/'generated/corridor-v1.wad')
    config = config.replace('episode_timeout = 1400', 'episode_timeout = 80')
    path = tmp_path/'short.cfg'
    path.write_text(config, encoding='utf-8')
    with CorridorEnvironment(config_file=path) as env:
        assert env.timeout == 80


def test_scripted_bootstrap_uses_frames_and_all_route_actions():
    model = build_model('compact_snn', action_size=7, num_steps=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    replay = ReplayMemory(200)
    before = model.fc1.weight.detach().clone()
    with CorridorEnvironment('navigation', seed=123) as env:
        result = bootstrap_navigation(model, env, replay, optimizer,
                                      torch.device('cpu'), np.random.default_rng(0),
                                      episodes=2, updates=1, batch_size=8)
    assert result['samples'] == len(replay)
    assert set(result['actions']) == {'forward', 'turn_left', 'turn_right'}
    assert all(transition[1] in (0, 3, 4) for transition in replay.memory)
    assert not torch.equal(before, model.fc1.weight.detach())


def test_corrective_collection_labels_learner_visited_bends_on_both_sides():
    class AlwaysForward(torch.nn.Module):
        def forward(self, states):
            scores = torch.zeros(states.shape[0], 7)
            scores[:, 0] = 1
            return scores, torch.zeros(())

    dataset = ExpertDataset()
    with CorridorEnvironment('navigation', seed=123) as env:
        result = collect_corrections(
            AlwaysForward(), env, dataset, torch.device('cpu'), np.random.default_rng(1),
            episodes=2, max_steps=30, teacher_probability=0,
        )
    assert result['samples'] == 60
    assert result['by_turn'] == {'left': 30, 'right': 30}
    assert result['disagreements'] > 0
    assert {0, 3, 4} == set(dataset.labels)
    assert dataset.counts()['recovery'] > 0

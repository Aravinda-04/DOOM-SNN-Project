import struct

import pytest
import torch
import numpy as np

from corridor_env import (ACTIONS, ACTION_NAMES, NAVIGATION_ACTIONS,
                          allowed_actions, build_scenario, case_geometry, task_outcome)
from corridor import ExpertDataset, greedy_action, load_policy, validation_score
from networks import build_model
from profile_models import profile
from train import ReplayMemory, optimize_model


def test_mirrored_map_and_wad_directory(tmp_path):
    path=build_scenario(tmp_path/'test.wad')
    data=path.read_bytes()
    magic,count,offset=struct.unpack('<4sii',data[:12])
    assert magic==b'PWAD'
    assert count==32*3
    names=[]
    for i in range(count):
        start,size,name=struct.unpack('<ii8s',data[offset+i*16:offset+(i+1)*16])
        assert 12<=start<=offset and start+size<=offset
        names.append(name.rstrip(b'\0'))
    assert names[::3]==[f'MAP{i:02}'.encode() for i in range(1,33)]
    a,_,_=case_geometry(0)
    b,_,_=case_geometry(1)
    assert set(a)=={(x,-y) for x,y in b}
    assert case_geometry(0)[1] != case_geometry(8)[1]


def test_exit_requires_alive_and_combat_kills():
    info={'x':640,'y_progress':660,'dead':False,'kills':0,'tics':100,'engine_finished':False}
    assert task_outcome(info,'navigation',1400)=='complete'
    assert task_outcome(info,'combat',1400)=='running'
    assert task_outcome(dict(info,kills=2),'combat',1400)=='complete'
    assert task_outcome(dict(info,kills=2,dead=True),'combat',1400)=='death'
    assert task_outcome(dict(info,x=0,tics=1400),'navigation',1400)=='timeout'


def test_compact_network_gradients_and_resource_budget():
    model=build_model('compact_snn',action_size=7,num_steps=2)
    q,spikes=model(torch.rand(2,1,84,84))
    assert q.shape==(2,7) and spikes.ndim==0
    q.square().mean().backward()
    assert model.fc1.weight.grad is not None
    assert torch.isfinite(model.fc1.weight.grad).all()
    report=profile(model)
    assert report['weights']==256*128+128*7
    assert report['lif_state_values']==135
    assert len(ACTION_NAMES)==len(ACTIONS)==7
    assert all(len(action)==6 for action in ACTIONS)


def test_reject_combat_checkpoint_for_corridor(tmp_path):
    path=tmp_path/'basic.pth'
    torch.save({'model_state_dict':{}},path)
    with pytest.raises(ValueError,match='incompatible actions'):
        load_policy(path,torch.device('cpu'))


def test_navigation_masks_shooting_in_greedy_and_random_selection():
    q=torch.tensor([3.,2.,1.,0.,-1.,100.,200.])
    assert allowed_actions('navigation') == NAVIGATION_ACTIONS
    assert greedy_action(q, NAVIGATION_ACTIONS) == 0
    assert allowed_actions('combat') == tuple(range(7))
    assert greedy_action(q, allowed_actions('combat')) == 6
    with pytest.raises(ValueError,match='Unknown corridor stage'):
        allowed_actions('unknown')


def test_checkpoint_selection_prefers_balanced_turn_completion():
    one_sided = {'completion_rate': 0.5, 'mean_reward': 50.,
                 'by_turn': {'left': {'complete': 4, 'total': 4},
                             'right': {'complete': 0, 'total': 4}}}
    balanced = {'completion_rate': 0.375, 'mean_reward': -10.,
                'by_turn': {'left': {'complete': 1, 'total': 4},
                            'right': {'complete': 2, 'total': 4}}}
    assert validation_score(balanced) > validation_score(one_sided)


def test_navigation_bellman_target_ignores_disabled_attack_values():
    class ConstantQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values=torch.nn.Parameter(torch.tensor(values,dtype=torch.float32))

        def forward(self, states):
            return self.values.expand(states.shape[0],-1), self.values.new_zeros(())

    class TargetCapture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.expected=None

        def forward(self, actual, expected):
            self.expected=expected.detach().clone()
            return (actual-expected).square().mean()

    memory=ReplayMemory(1)
    frame=torch.zeros(84,84).numpy()
    memory.push(frame,0,0.,frame,False)
    policy=ConstantQ([0]*7)
    target=ConstantQ([1,2,3,4,5,100,200])
    criterion=TargetCapture()
    optimize_model(memory,policy,target,torch.optim.SGD(policy.parameters(),lr=.01),
                   criterion,batch_size=1,replay_start_size=1,gamma=1.,
                   sparsity_weight=0.,device=torch.device('cpu'),
                   allowed_actions=NAVIGATION_ACTIONS)
    assert criterion.expected.item()==5.


def test_expert_dataset_samples_both_turn_actions_and_recovery_frames():
    dataset = ExpertDataset()
    for mirror in (1, -1):
        for action in (0, 3, 4):
            for recovery in (False, True):
                dataset.add(np.full((84, 84), float(recovery)), action, mirror,
                            recovery=recovery)
    frames, labels = dataset.sample(np.random.default_rng(3), 512, torch.device('cpu'))
    assert frames.shape == (512, 1, 84, 84)
    assert {0, 3, 4} == set(labels.tolist())
    assert 150 < int(labels.eq(0).sum()) < 400
    assert 150 < int(frames[:, 0, 0, 0].sum()) < 400
    assert dataset.counts()['recovery'] == 6


def test_expert_loss_changes_q_values_even_when_td_target_is_zero():
    class ConstantQ(torch.nn.Module):
        num_steps = 1

        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(torch.zeros(7))

        def forward(self, states):
            return self.values.expand(states.shape[0], -1), self.values.new_zeros(())

    memory = ReplayMemory(1)
    frame = np.zeros((84, 84), np.float32)
    memory.push(frame, 0, 0., frame, True)
    policy, target = ConstantQ(), ConstantQ()
    loss = optimize_model(
        memory, policy, target, torch.optim.SGD(policy.parameters(), lr=.1),
        torch.nn.SmoothL1Loss(), batch_size=1, replay_start_size=1, gamma=.99,
        sparsity_weight=0., device=torch.device('cpu'),
        allowed_actions=NAVIGATION_ACTIONS,
        expert_states=torch.zeros(1, 1, 84, 84), expert_actions=torch.tensor([3]),
        expert_weight=1.,
    )
    assert loss > 0
    assert policy.values[3] > 0


def test_compact_conv_network_has_gradients_and_fewer_weights_than_dense():
    model=build_model('compact_conv_snn',action_size=7,num_steps=2)
    q,spikes=model(torch.rand(1,1,84,84))
    assert q.shape==(1,7) and spikes.ndim==0
    q.square().mean().backward()
    assert model.conv1.weight.grad is not None
    assert torch.isfinite(model.conv1.weight.grad).all()
    assert profile(model)['weights']<profile(build_model('compact_snn',action_size=7,num_steps=2))['weights']

"""Profile active full-level models for controlled software comparisons."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from networks import build_model
from project_paths import REPORT_ROOT


def profile(model, in_channels=4):
    layers, handles = {}, []

    def hook(name):
        def record(module, inputs, output):
            if name.startswith('lif') and isinstance(output, tuple):
                state = output[1]
                state_values = state.numel() if isinstance(state, torch.Tensor) else 0
                layers.setdefault(name, {'state_values': state_values, 'dense_macs': 0})
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                item = layers.setdefault(name, {'state_values': 0, 'dense_macs': 0})
                fan_in = module.weight[0].numel()
                item['dense_macs'] += output.numel() * fan_in
            elif isinstance(module, nn.LSTMCell):
                item = layers.setdefault(name, {'state_values': 0, 'dense_macs': 0})
                batch = inputs[0].shape[0]
                item['state_values'] = sum(value.numel() for value in output)
                item['dense_macs'] += batch * 4 * module.hidden_size * (
                    module.input_size + module.hidden_size
                )
        return record

    for name,module in model.named_modules():
        if name.startswith('lif') or isinstance(module,(nn.Conv2d,nn.Linear,nn.LSTMCell)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            model(torch.zeros(1, in_channels, 84, 84), None)
    finally:
        for handle in handles:
            handle.remove()
    weights = sum(p.numel() for p in model.parameters() if p.ndim >= 2)
    scalar_parameters = sum(p.numel() for p in model.parameters() if p.ndim < 2)
    parameter_count = sum(p.numel() for p in model.parameters())
    recurrent_state_values = sum(r['state_values'] for r in layers.values())
    return {'parameters': parameter_count,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'matrix_or_kernel_parameters': weights,
            'scalar_or_vector_parameters': scalar_parameters,
            'parameter_fp32_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'hypothetical_int8_parameter_bytes': parameter_count,
            'recurrent_state_values': recurrent_state_values,
            'recurrent_state_fp32_bytes': 4*recurrent_state_values,
            'dense_macs_per_decision': sum(r['dense_macs'] for r in layers.values()),
            'layers':layers}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPORT_ROOT/'model-profile')
    parser.add_argument('--models',nargs='+',
                        default=['level_cnn','level_drqn','crsnn','rsnn_v2'])
    parser.add_argument('--actions',type=int,default=8)
    parser.add_argument('--in-channels',type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    results={}
    for name in args.models:
        model = build_model(
            name,
            action_size=args.actions,
            in_channels=args.in_channels,
        )
        results[name]=profile(model,in_channels=args.in_channels)
    report={'models':results,'notes':[
        'Counts include every spatial LIF state, not just feature channels.',
        'Dense MACs are measured from the executed software graph; spike hardware may use additions.',
        'Pooling, LIF arithmetic, activation buffers, routing, input encoding and scales are excluded from MAC/storage estimates.',
        'LSTMCell estimates include its four input and recurrent gates.',
        'No learning-quality, deployment, or hardware claim is made.']}
    (args.output/'profile.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,metric,title in zip(axes,('parameters','recurrent_state_values','dense_macs_per_decision'),
                               ('Parameter count','Recurrent state values','Dense MAC estimate / decision')):
        ax.bar(results.keys(),[r[metric] for r in results.values()])
        if all(r[metric] > 0 for r in results.values()):
            ax.set_yscale('log')
        ax.set_title(title)
        ax.tick_params(axis='x',rotation=40,labelsize=7)
        for i,r in enumerate(results.values()):
            ax.text(i,r[metric],f"{r[metric]:,}",ha='center',va='bottom',fontsize=8)
    fig.suptitle('Software resource estimates - hardware fit and gameplay accuracy unverified')
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png',dpi=150)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

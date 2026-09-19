"""Measure model storage/state and dense operation estimates; not FPGA synthesis."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from networks import build_model
from project_paths import REPORT_ROOT


def profile(model):
    layers, handles = {}, []

    def hook(name):
        def record(module, inputs, output):
            if name.startswith('lif'):
                layers.setdefault(name, {'state_values': output[1].numel(), 'dense_macs': 0})
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                item = layers.setdefault(name, {'state_values': 0, 'dense_macs': 0})
                fan_in = module.weight[0].numel()
                item['dense_macs'] += output.numel() * fan_in
        return record

    for name,module in model.named_modules():
        if name.startswith('lif') or isinstance(module,(nn.Conv2d,nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            model(torch.zeros(1,1,84,84))
    finally:
        for handle in handles:
            handle.remove()
    weights = sum(p.numel() for name,p in model.named_parameters() if name.endswith('weight'))
    biases = sum(p.numel() for name,p in model.named_parameters() if name.endswith('bias'))
    return {'weights': weights, 'biases': biases,
            'parameter_fp32_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'hypothetical_int8_weight_bytes': weights,
            'lif_state_values': sum(r['state_values'] for r in layers.values()),
            'lif_membrane_fp32_bytes': 4*sum(r['state_values'] for r in layers.values()),
            'dense_macs_per_decision': sum(r['dense_macs'] for r in layers.values()),
            'layers':layers}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPORT_ROOT/'model-profile')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    results={}
    for name in ('snn','compact_snn','compact_conv_snn'):
        for actions in (5,7):
            results[f'{name}-{actions}actions']=profile(build_model(name,action_size=actions))
    report={'models':results,'notes':[
        'Counts include every spatial LIF state, not just feature channels.',
        'Dense MACs are a software-graph estimate including ten neural steps; spike hardware may use additions.',
        'Pooling, LIF arithmetic, activation buffers, routing, input encoding and scales are excluded from MAC/storage estimates.',
        'Compact models use beta=0.9 and a membrane readout; their pooling and hardware compatibility are unverified.',
        'No accuracy equivalence or FPGA resource/timing claim is made.']}
    (args.output/'profile.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,metric,title in zip(axes,('weights','lif_state_values','dense_macs_per_decision'),
                               ('Weight count','LIF state values','Dense MAC estimate / decision')):
        ax.bar(results.keys(),[r[metric] for r in results.values()],
               color=['steelblue','steelblue','seagreen','seagreen','darkorange','darkorange'])
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

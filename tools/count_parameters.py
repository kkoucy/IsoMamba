"""Count unique model parameters without running a CUDA forward pass."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'options/train_IsoMamba.yml')
    args = parser.parse_args()
    import yaml
    from basicsr.models.archs.isomamba_arch import IsoMamba

    with args.config.open() as handle:
        options = dict(yaml.safe_load(handle)['network_g'])
    model_type = options.pop('type')
    if model_type not in ('IsoMamba', 'ClusterMamba'):
        raise ValueError(f'Unsupported network: {model_type}')
    model = IsoMamba(**options)
    stages = Counter()
    parameters = list(model.named_parameters())  # PyTorch deduplicates shared tensors.
    for name, parameter in parameters:
        stages[name.split('.')[0]] += parameter.numel()
    total = sum(p.numel() for _, p in parameters)
    result = {
        'network': options,
        'unique_parameters': total,
        'million_parameters': total / 1e6,
        'trainable_parameters': sum(p.numel() for _, p in parameters if p.requires_grad),
        'by_stage': dict(stages),
        'note': 'Includes all experts and custom SSM tensors; shared encoder references counted once; buffers excluded.',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Run IsoMamba on an image directory without ground-truth targets."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'options/train_IsoMamba.yml')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--param-key', choices=('auto', 'params', 'params_ema', 'raw'), default='auto')
    parser.add_argument('--seed', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {args.input_dir}')
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError('Input and output directories must be different.')
    if not args.weights.is_file():
        raise ValueError(f'Checkpoint does not exist: {args.weights}')
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        raise ValueError('No supported images found.')
    outputs = [args.output_dir / f'{p.stem}.png' for p in files]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Input filenames have duplicate stems; PNG outputs would collide.')
    if any(p.exists() for p in outputs):
        raise FileExistsError('Output images already exist; choose a new output directory.')

    sys.path.insert(0, str(ROOT))
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from PIL import Image
    from tqdm import tqdm
    from basicsr.models.archs.isomamba_arch import IsoMamba

    if not torch.cuda.is_available():
        raise RuntimeError('IsoMamba requires CUDA for its fused selective-scan kernel.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with args.config.open() as handle:
        options = yaml.safe_load(handle)['network_g']
    model_type = options.pop('type')
    if model_type not in ('IsoMamba', 'ClusterMamba'):
        raise ValueError(f'Unsupported model type: {model_type}')
    model = IsoMamba(**options)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    if args.param_key == 'raw':
        state = checkpoint
    elif args.param_key == 'auto':
        key = next((key for key in ('params', 'params_ema') if key in checkpoint), None)
        state = checkpoint[key] if key else checkpoint
        print(f'Checkpoint layout: {key or "raw"}')
    else:
        state = checkpoint[args.param_key]
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for source, destination in tqdm(list(zip(files, outputs)), desc='IsoMamba'):
            with Image.open(source) as image:
                array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda()
            height, width = tensor.shape[-2:]
            # At least 16 pixels keeps the decoder's 2x2 routing pool valid.
            pad_h = max(16, ((height + 7) // 8) * 8) - height
            pad_w = max(16, ((width + 7) // 8) * 8) - width
            mode = 'reflect' if pad_h < height and pad_w < width else 'replicate'
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
            result = model(tensor)[0, :, :height, :width].clamp(0, 1)
            result = result.permute(1, 2, 0).cpu().numpy()
            Image.fromarray(np.rint(result * 255).astype(np.uint8)).save(destination)
    print(f'Saved {len(files)} images to {args.output_dir}')


if __name__ == '__main__':
    main()

"""Portable dataset overrides and accurate command recording for diagnostics."""

import os
from pathlib import Path
import shlex
import sys


def add_dataset_arguments(parser):
    for name in ('uieb', 'lsui'):
        parser.add_argument(f'--{name}-input-dir', type=Path)
        parser.add_argument(f'--{name}-gt-dir', type=Path)


def apply_dataset_arguments(args, datasets):
    for name in ('uieb', 'lsui'):
        input_dir = getattr(args, f'{name}_input_dir')
        gt_dir = getattr(args, f'{name}_gt_dir')
        if (input_dir is None) != (gt_dir is None):
            raise ValueError(f'Provide both --{name}-input-dir and --{name}-gt-dir.')
        if input_dir is not None:
            datasets[name]['input'] = input_dir
            datasets[name]['gt'] = gt_dir


def runtime_command(script):
    bindings = [f'{key}={os.environ[key]}' for key in
                ('CUDA_VISIBLE_DEVICES', 'MPLCONFIGDIR', 'CUBLAS_WORKSPACE_CONFIG')
                if key in os.environ]
    return shlex.join(['env', *bindings, sys.executable, str(script), *sys.argv[1:]])

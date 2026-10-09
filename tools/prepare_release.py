"""Copy a reviewed source allowlist to a new directory; never delete originals."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    '.gitignore', 'README.md', 'README_zh-CN.md', 'requirements.txt',
    'THIRD_PARTY_NOTICES.md', 'train.py', 'test.py', 'options/train_IsoMamba.yml',
)
TREES = {
    'basicsr': {'.py', '.npz'},
    'analysis': {'.py'},
    'tools': {'.py'},
    'docs': {'.md'},
    'assets/figures': {'.svg', '.pdf', '.png'},
    'assets/data': {'.json'},
}


def source_files():
    selected = [ROOT / name for name in FILES]
    for directory, suffixes in TREES.items():
        folder = ROOT / directory
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in folder.rglob('*'):
            if '__pycache__' in path.parts:
                continue
            if path.is_symlink():
                raise ValueError(f'Symlinks must be reviewed before export: {path}')
            if path.is_file() and path.suffix in suffixes:
                selected.append(path)
    for path in selected:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or non-regular source: {path}')
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Source escapes the project: {path}')
    for required in ('architecture', 'cluster_route_isolate', 'hidden_state_separation'):
        for suffix in ('.svg', '.pdf', '.png'):
            if ROOT / f'assets/figures/{required}{suffix}' not in selected:
                raise FileNotFoundError(f'Build the paper figures first: {required}{suffix}')
    if ROOT / 'assets/data/hidden_state_separation.json' not in selected:
        raise FileNotFoundError('Build the portable figure data first.')
    return sorted(set(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    destination = args.output.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing existing destination: {destination}')
    resolved = destination.resolve()
    if resolved == ROOT or ROOT.is_relative_to(resolved):
        raise ValueError('Destination cannot be the project or one of its ancestors.')
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / 'release_dist'):
        raise ValueError('Use an external directory or the ignored release_dist/ folder.')
    selected = source_files()
    records = []
    for source in selected:
        data = source.read_bytes()
        records.append({'path': source.relative_to(ROOT).as_posix(),
                        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    if args.dry_run:
        print(json.dumps(records, indent=2))
        return
    destination.mkdir(parents=True, exist_ok=False)
    for source, record in zip(selected, records):
        target = destination / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'Source changed during export: {source}')
    manifest = {'status': 'release-preparation; license and scientific review pending',
                'files': records, 'total_bytes': sum(r['bytes'] for r in records)}
    (destination / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(records)} files ({manifest["total_bytes"] / 1024**2:.2f} MiB) to {destination}')


if __name__ == '__main__':
    main()

"""Draw implementation schematics and evidence-backed diagnostic figures."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/isomamba-paper-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = '#183249'
BLUE = '#dcecf6'
TEAL = '#d8eee8'
GOLD = '#f8ebce'
EXPERTS = ['#338a9e', '#d29541', '#9272af', '#668f68']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis('off')
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98)
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.07',
                              facecolor=color, edgecolor=INK, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size)


def arrow(ax, start, end, color=INK, dashed=False, bend=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                              linewidth=1, color=color, linestyle='--' if dashed else '-',
                              connectionstyle=f'arc3,rad={bend}'))


def save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(output / f'{stem}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def architecture(output):
    fig, ax = canvas(13.8, 7.4)
    ax.text(0.2, 7.03, 'IsoMamba | global context, isolated recurrent experts', fontsize=17, weight='bold')
    ax.text(0.2, 6.63, 'Implementation schematic · additive skips · independent decoder state streams', fontsize=10)
    left, right, width, height = 1.55, 9.65, 2.65, 0.72
    ys = [5.38, 4.08, 2.78]
    enc = ['Global encoder 1\nC=32 · 4 blocks · H × W',
           'Global encoder 2\nC=64 · 6 blocks · H/2 × W/2',
           'Global encoder 3\nC=128 · 6 blocks · H/4 × W/4']
    dec = ['Isolated decoder 1\n2 experts · 4 blocks',
           'Isolated decoder 2\n3 experts · 6 blocks',
           'Isolated decoder 3\n4 experts · 6 blocks']
    for i, y in enumerate(ys):
        box(ax, left, y, width, height, enc[i])
        box(ax, right, y, width, height, dec[i], TEAL)
        arrow(ax, (left + width, y + .37), (right, y + .37))
        ax.text(6.9, y + .49, 'feature skip (+)', ha='center', fontsize=9)
        arrow(ax, (left + width, y + .13), (right, y + .13), '#768ca0', dashed=True)
        ax.text(6.9, y - .12, 'projection → router  |  terminal state → FiLM', ha='center', fontsize=8)
        if i < 2:
            arrow(ax, (left + width / 2, y), (left + width / 2, ys[i + 1] + height))
            ax.text(left + width / 2 + .15, y - .3, '↓ 2', fontsize=9)
            arrow(ax, (right + width / 2, ys[i + 1] + height), (right + width / 2, y))
            ax.text(right + width / 2 + .15, y - .3, '↑ 2', fontsize=9)
    box(ax, 4.9, 1.22, 4.1, .82, 'Global bottleneck\nC=256 · 8 blocks · H/8 × W/8', GOLD)
    arrow(ax, (left + width / 2, ys[-1]), (4.9, 1.63), bend=.12)
    arrow(ax, (9, 1.63), (right + width / 2, ys[-1]), bend=.12)
    ax.text(3.65, 2.05, '↓ 2', fontsize=9)
    ax.text(10.05, 2.05, '↑ 2', fontsize=9)
    box(ax, .12, 5.38, 1.08, .72, 'Input\nRGB', '#f0f2f5')
    arrow(ax, (1.2, 5.74), (left, 5.74))
    ax.text(.15, 6.25, '3×3 embedding', fontsize=8)
    box(ax, 9.65, 6.33, 2.65, .63, 'Global refinement · 4 blocks', GOLD, size=9)
    arrow(ax, (10.97, 6.1), (10.97, 6.33))
    box(ax, 12.67, 6.05, .94, .9, '3×3\n+ input\nRGB', '#f0f2f5', size=9)
    arrow(ax, (12.3, 6.65), (12.67, 6.65))
    ax.text(.2, .55, 'Dashed paths: encoder cues. Each decoder expert uses its own four-direction selective scan.', fontsize=9)
    ax.text(.2, .22, 'Widths and block counts follow options/train_IsoMamba.yml; no benchmark performance is implied.', fontsize=9)
    save(fig, output, 'architecture')


def routing(output):
    fig, ax = canvas(14, 6.9)
    ax.text(.2, 6.5, 'Cluster → route → isolate', fontsize=18, weight='bold')
    ax.text(.2, 6.08, 'Projection-derived update representation and group-specific recurrent evolution', fontsize=10)
    boxes = [(.2, 2.4, 'Decoder features\n+ detached encoder\nprojection slices', BLUE),
             (3.35, 2.55, 'Directional alignment\nprior fusion + low-rank\nadaptation → Softplus', BLUE),
             (6.65, 2.45, 'Normalize + pool\ncluster centers\nanchor matching', GOLD),
             (9.95, 2.7, 'Full-resolution scores\nTop-1 assignment\ngather group tokens', TEAL)]
    for i, (x, w, label, color) in enumerate(boxes):
        box(ax, x, 4.7, w, .95, label, color, size=10)
        if i < len(boxes) - 1:
            arrow(ax, (x + w, 5.18), (boxes[i + 1][0], 5.18))
    box(ax, .2, 2.9, 2.4, 1.04, 'Illustrative token grid\ncolors indicate groups', '#f0f2f5', size=9)
    for row in range(3):
        for col in range(8):
            ax.add_patch(Rectangle((.3 + col * .27, 1.6 + row * .3), .22, .22,
                                   facecolor=EXPERTS[(col + row * 2) % 3], edgecolor='white'))
    ax.text(.2, 1.27, 'Schematic labels; no data imagery', fontsize=8)
    for i, color in enumerate(EXPERTS[:3]):
        y = 3.42 - i * .86
        box(ax, 4, y, 3.75, .6, f'Expert {i + 1}: independent state stream', '#edf3f5', size=9)
        ax.add_patch(Rectangle((4.02, y + .02), .11, .55, facecolor=color, linewidth=0))
        arrow(ax, (2.7, 2.85), (4, y + .3), color=color)
        arrow(ax, (7.75, y + .3), (9.4, 2.85), color=color)
    arrow(ax, (11.3, 4.7), (6.9, 4.12), bend=-.12)
    ax.text(5.9, 4.21, 'square packing + zero padding', ha='center', fontsize=8)
    box(ax, 9.4, 2.38, 3.25, .94, 'Scatter to original coordinates\n+ feature residual\n→ decoder output', TEAL)
    box(ax, 9.4, 1.12, 3.25, .65, 'Encoder terminal states → FiLM', GOLD, size=9)
    arrow(ax, (9.4, 1.45), (7.75, 1.95), dashed=True)
    ax.text(.2, .65, 'Each expert: projection → depthwise convolution → four-direction scan → FiLM → norm / gate / projection.', fontsize=9)
    ax.text(.2, .23, 'Routing selects indices in this implementation; gradient behavior and logical-packing diagnostics are discussed in the code audit.', fontsize=9)
    save(fig, output, 'cluster_route_isolate')


def load_data(source, data_path):
    if source is not None:
        source_bytes = source.read_bytes()
        raw = json.loads(source_bytes)
        datasets = {}
        for name, values in raw['datasets'].items():
            datasets[name] = {
                'image_count': values['image_count'],
                'modes': {key: {field: mode[field] for field in ('mean', 'std', 'n')}
                          for key, mode in values['modes'].items()},
                'paired_difference': values['paired_difference_isolated_fixed_minus_shared_fixed'],
            }
        data = {
            'source': 'raw_hidden_state_separation/dataset_summary.json',
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'checkpoint': 'ClusterMamba/models/net_g_300000.pth',
            'layer': 'decoder_level3.attention.0',
            'protocol': 'post-hoc; raw normalized h_t; four-direction image mean; logical packing excludes dummy padding',
            'error_bars': 'image-level sample standard deviation',
            'contrast': 'isolated_fixed minus shared_fixed; same expert-0 tokenwise operands',
            'full_isolation': 'descriptive expert-specialization reference, not the fixed-expert contrast',
            'datasets': datasets,
        }
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(data, indent=2) + '\n')
    else:
        data = json.loads(data_path.read_text())
    return data


def diagnostic(output, data):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    names = ['uieb', 'lsui']
    keys = ['shared_fixed', 'isolated_fixed', 'full_isolation']
    labels = ['Shared state\nfixed expert', 'Isolated state\nfixed expert', 'Full isolation\ndescriptive reference']
    colors = ['#8a9aa9', '#338a9e', '#d29541']
    positions = np.arange(2)
    for i, key in enumerate(keys):
        modes = [data['datasets'][name]['modes'][key] for name in names]
        ax.bar(positions + (i - 1) * .23, [m['mean'] for m in modes], .21,
               yerr=[m['std'] for m in modes], capsize=3, label=labels[i],
               color=colors[i], edgecolor='white', linewidth=.6,
               error_kw={'elinewidth': 1, 'ecolor': INK})
        for j, mode in enumerate(modes):
            ax.text(positions[j] + (i - 1) * .23, mode['mean'] + mode['std'] + .018,
                    f'{mode["mean"]:.3f}', ha='center', fontsize=9)
    ax.set_xticks(positions, [f'{n.upper()} (n={data["datasets"][n]["image_count"]})' for n in names])
    ax.set(ylim=(0, 1.05), ylabel='Raw hidden-state separation', axisbelow=True)
    ax.set_title('State organization at fixed expert parameters', loc='left', pad=14, weight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, frameon=False, fontsize=9)
    fig.text(.12, .03, 'One trained checkpoint · post-hoc diagnostic · whiskers: ±1 image-level SD', fontsize=9)
    fig.subplots_adjust(left=.12, right=.98, top=.87, bottom=.27)
    save(fig, output, 'hidden_state_separation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-summary', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / 'assets/data/hidden_state_separation.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets/figures')
    args = parser.parse_args()
    data = load_data(args.source_summary, args.data)
    architecture(args.output_dir)
    routing(args.output_dir)
    diagnostic(args.output_dir, data)
    print(f'Wrote 3 figures × SVG/PDF/PNG to {args.output_dir}')


if __name__ == '__main__':
    main()

# IsoMamba

**State-Isolated Vision Mamba via Update-Tendency Routing for Underwater Image Enhancement**

Yan Luo, Laibin Chang, Zefan Feng, Yi'ang Chen, Kui Jiang, Zhengguo Li

![arXiv pending moderation](https://img.shields.io/badge/arXiv-pending%20moderation-b31b1b)

**arXiv:** The manuscript has been submitted and is awaiting moderation. The paper link and identifier will be added after announcement.

[中文说明](README_zh-CN.md)

IsoMamba groups visual tokens using a model-derived update representation and processes each group with a separate recurrent expert. Encoder terminal states coordinate decoder experts through FiLM modulation. The implementation uses a U-shaped network with additive feature skips and an image residual.

![IsoMamba architecture](assets/figures/architecture.png)

Research code for IsoMamba. The arXiv paper announcement, public checkpoint downloads, license choice, and remaining implementation notes are tracked below. Repository: [kkoucy/IsoMamba](https://github.com/kkoucy/IsoMamba).

## Structure

```text
basicsr/
  models/archs/clusterM_arch.py    # Original checkpoint-compatible network
  models/archs/isomamba_arch.py    # Public IsoMamba alias
  models/image_restoration_model.py
  models/losses/                  # Channel-adaptive L1/SSIM and edge losses
  data/                          # Datasets, augmentation, sampling
  metrics/                       # Validation metrics
  utils/                         # Training infrastructure
options/train_IsoMamba.yml        # Portable configuration
train.py                         # Training entry point
test.py                          # Legacy paired evaluation (see caveats)
tools/infer.py                   # Inference without ground-truth images
assets/figures/architecture.png   # Figure 3 cropped directly from the manuscript
tools/count_parameters.py         # Parameter counting
```

## Environment

The inspected local environment uses Linux, Python 3.10, PyTorch 2.1.2 + CUDA 11.8, torchvision 0.16.2, mamba-ssm 1.1.1, and causal-conv1d 1.1.1. Model imports, strict checkpoint loading and a CUDA forward under THOP were checked in this environment; a fresh installation and checkpoint-based image evaluation still need validation. The fused selective scan requires CUDA.

```bash
git clone https://github.com/kkoucy/IsoMamba.git
cd IsoMamba

conda create -n isomamba python=3.10 -y
conda activate isomamba
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install packaging ninja wheel setuptools
pip install causal-conv1d==1.1.1 --no-build-isolation
pip install mamba-ssm==1.1.1 --no-build-isolation
```

CUDA extension wheels must match Python, PyTorch, CUDA and the C++ ABI; building from source requires a compatible CUDA toolkit and compiler. Install Mamba from the package commands above; no local Mamba source tree or wheel upload is needed. The pinned releases are [mamba-ssm 1.1.1](https://pypi.org/project/mamba-ssm/1.1.1/) and [causal-conv1d 1.1.1](https://pypi.org/project/causal-conv1d/1.1.1/). If a matching prebuilt wheel is unavailable, install a CUDA 11.8 toolkit with `nvcc` and set `CUDA_HOME` to its installation directory before retrying.

Check the installation:

```bash
python -m pip check
python -c "import torch, mamba_ssm, causal_conv1d; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from basicsr.models.archs.isomamba_arch import IsoMamba; print('CUDA:', torch.cuda.is_available()); print('IsoMamba and CUDA extensions imported successfully')"
```

CUDA must report `True` for training and inference.

## Data and training

Obtain the datasets from their original providers and arrange paired inputs/targets with matching filenames. No datasets are redistributed.

```text
datasets/UIEB_LSUI/train/{Input,GT}/
datasets/UIEB/test/{Input,GT}/
```

Edit the paths in `options/train_IsoMamba.yml`, then run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train_IsoMamba.yml
```

The supplied recipe uses 256×256 crops, batch size 2, 300,000 iterations, AdamW at 3e-4, and L1/SSIM/edge weights 8/1/4. It preserves the existing recipe rather than resolving differences with the manuscript. The training entry automatically resumes from the highest numbered state in `experiments/<name>/training_states/`; use a new experiment name for a fresh run. `options/UIE.yml` is the original machine-specific configuration and is excluded from the export.

## Inference

Public checkpoint downloads are pending. Place a compatible checkpoint at `checkpoints/isomamba.pth`:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py \
  --config options/train_IsoMamba.yml \
  --weights checkpoints/isomamba.pth \
  --input-dir datasets/example/Input --output-dir results/example
```

The command preserves image resolution, pads to a multiple of eight, crops the output back, and saves RGB PNGs. It does not resize to the paper's 256×256 benchmark protocol or calculate benchmark scores. Supported checkpoint layouts are `params`, `params_ema`, or a raw state dict; `--param-key` selects the desired layout explicitly. Routing uses randomized center initialization; `--seed` makes initialization repeatable within the same environment, but bitwise reproducibility across CUDA environments is not guaranteed.

`test.py` is retained for historical paired evaluations. It requires targets, only scans lowercase PNG/JPG, uses SSIM `data_range=1.1`, and must not be treated as the final benchmark protocol without review. Training validation uses a separate metric implementation.

## Model complexity

For a **1×3×256×256** input, THOP reports **15.23 M parameters** and **51.65 G operations**.

## Citation and attribution

The supplied author order is recorded above. A formal BibTeX entry will be added when the publication year, venue and public paper identifier are confirmed.

The project includes code derived from BasicSR and uses Mamba, PyTorch, timm and other dependencies. Third-party source provenance and licenses require review before release; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A project license has not yet been selected.

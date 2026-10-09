#!/usr/bin/env python3
"""Analysis-only State Contention diagnostics for IsoMamba.

This script never trains or updates the model.  It reconstructs the recurrent
hidden state used by selective scan in FP32 and compares:

1. a shared state receiving every routed token;
2. parameter-matched isolated states selected by the router label; and
3. optionally, the packed isolated streams used by the actual checkpoint.

The first two modes use identical Mamba parameters and identical scan
coefficients.  Only state sharing changes.  They are a post-hoc diagnostic,
not a trained shared-state enhancement baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# clusterM_arch imports matplotlib during import.  Keep its cache out of the
# user's home directory, which may be read-only on compute nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/isomamba_state_contention_mpl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "options" / "train_IsoMamba.yml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments" / "ClusterMamba_UIEB" / "models" / "net_g_300000.pth"
)
DEFAULT_INPUT = Path("/home/ly/Data/UIE/UIEB/test/Input256")
DEFAULT_GT = Path("/home/ly/Data/UIE/UIEB/test/GT256")
DEFAULT_OUTPUT = REPO_ROOT / "analysis_outputs" / "state_contention"

DIRECTION_NAMES = ("row_forward", "column_forward", "row_reverse", "column_reverse")
PRIMARY_HISTOGRAM_KEYS = (
    ("shared_counterfactual", "intra"),
    ("shared_counterfactual", "inter"),
    ("isolated_counterfactual", "stream"),
)


def natural_key(value: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def stable_image_seed(base_seed: int, basename: str) -> int:
    """Stable across Python processes and independent of dataset subset order."""
    return (int(base_seed) + zlib.crc32(basename.encode("utf-8"))) % (2**31 - 1)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def direction_sequences(spatial: torch.Tensor) -> torch.Tensor:
    """Convert [B,H,W] labels/mask to the exact four SS2D scan orders.

    Returns [B,4,H*W] in row-forward, column-forward, row-reverse,
    column-reverse order.
    """
    if spatial.ndim != 3:
        raise ValueError(f"Expected [B,H,W], got {tuple(spatial.shape)}")
    batch, height, width = spatial.shape
    row = spatial.contiguous().view(batch, height * width)
    column = spatial.transpose(1, 2).contiguous().view(batch, height * width)
    return torch.stack((row, column, row.flip(-1), column.flip(-1)), dim=1)


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def discover_pairs(input_dir: Path, gt_dir: Path) -> List[Tuple[Path, Path]]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    inputs = {p.name: p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions}
    targets = {p.name: p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions}
    missing_gt = sorted(inputs.keys() - targets.keys(), key=natural_key)
    missing_input = sorted(targets.keys() - inputs.keys(), key=natural_key)
    if missing_gt or missing_input:
        raise RuntimeError(
            "Input/GT basename mismatch. "
            f"input_only={missing_gt[:10]}, gt_only={missing_input[:10]}"
        )
    names = sorted(inputs.keys(), key=natural_key)
    if not names:
        raise RuntimeError(f"No paired images found in {input_dir} and {gt_dir}")
    return [(inputs[name], targets[name]) for name in names]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type != "cuda":
        raise RuntimeError(
            "The current ClusterMamba forward uses the fused selective-scan CUDA kernel. "
            "Run in the mamba environment with a visible CUDA GPU."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from basicsr.models.archs.clusterM_arch import ClusterMamba

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    network_options = dict(config["network_g"])
    network_type = network_options.pop("type")
    if network_type not in ("ClusterMamba", "IsoMamba"):
        raise ValueError(f"Expected IsoMamba/ClusterMamba, found {network_type!r}")
    model = ClusterMamba(**network_options)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "params" not in checkpoint:
        raise KeyError(f"{checkpoint_path} has no 'params' state dictionary")
    incompat = model.load_state_dict(checkpoint["params"], strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompat}")
    model.eval().to(device)
    return model


@dataclass
class ScanTensors:
    """Unflattened tensors for four-direction selective scan."""

    scan_x: torch.Tensor  # [B,4,D,L]
    delta_raw: torch.Tensor  # [B,4,D,L], before bias and softplus
    A: torch.Tensor  # [4,D,N]
    B: torch.Tensor  # [B,4,N,L]
    C: torch.Tensor  # [B,4,N,L]
    D: torch.Tensor  # [4,D]
    delta_bias: torch.Tensor  # [4,D]

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        batch, directions, inner, length = self.scan_x.shape
        state = self.A.shape[-1]
        return batch, directions, inner, state, length

    def prefix(self, length: int) -> "ScanTensors":
        length = min(length, self.scan_x.shape[-1])
        return ScanTensors(
            scan_x=self.scan_x[..., :length],
            delta_raw=self.delta_raw[..., :length],
            A=self.A,
            B=self.B[..., :length],
            C=self.C[..., :length],
            D=self.D,
            delta_bias=self.delta_bias,
        )


@torch.inference_mode()
def prepare_scan_tensors(module: torch.nn.Module, x_bhwc: torch.Tensor) -> ScanTensors:
    """Reproduce SS2D_dh0 preprocessing without invoking selective_scan."""
    if x_bhwc.ndim != 4:
        raise ValueError(f"Expected [B,H,W,C], got {tuple(x_bhwc.shape)}")
    batch, height, width, _ = x_bhwc.shape
    length = height * width
    directions = 4

    projected = module.in_proj(x_bhwc)
    scan_feature, _gate = projected.chunk(2, dim=-1)
    scan_feature = scan_feature.permute(0, 3, 1, 2).contiguous()
    scan_feature = module.act(module.conv2d(scan_feature))

    row = scan_feature.view(batch, module.d_inner, length)
    column = scan_feature.transpose(2, 3).contiguous().view(batch, module.d_inner, length)
    first_two = torch.stack((row, column), dim=1)
    scan_x = torch.cat((first_two, first_two.flip(-1)), dim=1)

    projected_scan = torch.einsum(
        "b k d l, k c d -> b k c l",
        scan_x,
        module.x_proj_weight,
    )
    delta_low_rank, variable_B, variable_C = torch.split(
        projected_scan,
        [module.dt_rank, module.d_state, module.d_state],
        dim=2,
    )
    delta_raw = torch.einsum(
        "b k r l, k d r -> b k d l",
        delta_low_rank,
        module.dt_projs_weight,
    )

    return ScanTensors(
        scan_x=scan_x.float(),
        delta_raw=delta_raw.contiguous().float(),
        A=(-torch.exp(module.A_logs.float())).view(directions, module.d_inner, module.d_state),
        B=variable_B.float().view(batch, directions, module.d_state, length),
        C=variable_C.float().view(batch, directions, module.d_state, length),
        D=module.Ds.float().view(directions, module.d_inner),
        delta_bias=module.dt_projs_bias.float().view(directions, module.d_inner),
    )


@torch.inference_mode()
def reference_scan_full(tensors: ScanTensors) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference recurrence returning scan output and final true state."""
    batch, directions, inner, state_size, length = tensors.shape
    hidden = torch.zeros(
        (batch, directions, inner, state_size),
        device=tensors.scan_x.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, directions, inner, length),
        device=tensors.scan_x.device,
        dtype=torch.float32,
    )
    for token in range(length):
        delta = F.softplus(tensors.delta_raw[..., token] + tensors.delta_bias.unsqueeze(0))
        transition = torch.exp(delta.unsqueeze(-1) * tensors.A.unsqueeze(0))
        drive = (
            delta.unsqueeze(-1)
            * tensors.B[:, :, None, :, token]
            * tensors.scan_x[..., token].unsqueeze(-1)
        )
        hidden = transition * hidden + drive
        output[..., token] = (
            (hidden * tensors.C[:, :, None, :, token]).sum(-1)
            + tensors.D.unsqueeze(0) * tensors.scan_x[..., token]
        )
    return output, hidden


@torch.inference_mode()
def validate_reference_recurrence(
    tensors: ScanTensors,
    max_length: int = 128,
    rtol: float = 6e-4,
    atol: float = 2e-3,
) -> Dict[str, Any]:
    """Compare our recurrence against both fused and upstream reference scan."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    sample = tensors.prefix(max_length)
    batch, directions, inner, state_size, length = sample.shape
    scan_x = sample.scan_x.view(batch, directions * inner, length)
    delta = sample.delta_raw.view(batch, directions * inner, length)
    A = sample.A.view(directions * inner, state_size)
    variable_B = sample.B
    variable_C = sample.C
    D = sample.D.view(directions * inner)
    delta_bias = sample.delta_bias.view(directions * inner)

    fused_output, fused_state = selective_scan_fn(
        scan_x,
        delta,
        A,
        variable_B,
        variable_C,
        D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    upstream_output, upstream_state = selective_scan_ref(
        scan_x,
        delta,
        A,
        variable_B,
        variable_C,
        D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    ours_output, ours_state = reference_scan_full(sample)
    ours_output = ours_output.view_as(fused_output)
    ours_state = ours_state.view_as(fused_state)

    checks = {
        "fused_vs_ours_output": torch.allclose(fused_output, ours_output, rtol=rtol, atol=atol),
        "fused_vs_ours_state": torch.allclose(fused_state, ours_state, rtol=rtol, atol=atol),
        "upstream_vs_ours_output": torch.allclose(
            upstream_output, ours_output, rtol=rtol, atol=atol
        ),
        "upstream_vs_ours_state": torch.allclose(
            upstream_state, ours_state, rtol=rtol, atol=atol
        ),
    }
    result: Dict[str, Any] = {
        "length": length,
        "rtol": rtol,
        "atol": atol,
        "checks": checks,
        "fused_output_max_abs": float((fused_output - ours_output).abs().max().item()),
        "fused_output_mean_abs": float((fused_output - ours_output).abs().mean().item()),
        "fused_state_max_abs": float((fused_state - ours_state).abs().max().item()),
        "upstream_output_max_abs": float((upstream_output - ours_output).abs().max().item()),
        "upstream_state_max_abs": float((upstream_state - ours_state).abs().max().item()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Reference recurrence validation failed: {result}")
    return result


@dataclass
class MetricSamples:
    cosine: np.ndarray
    volatility: np.ndarray
    eligible_pairs: int
    near_zero_pairs: int

    @property
    def cosine_pairs(self) -> int:
        return int(self.cosine.size)

    @property
    def negative_count(self) -> int:
        return int(np.count_nonzero(self.cosine < 0.0))

    def row_metrics(self) -> Dict[str, Any]:
        return {
            "eligible_pairs": int(self.eligible_pairs),
            "cosine_pairs": self.cosine_pairs,
            "near_zero_pairs": int(self.near_zero_pairs),
            "near_zero_pair_ratio": (
                float(self.near_zero_pairs / self.eligible_pairs)
                if self.eligible_pairs
                else math.nan
            ),
            "negative_count": self.negative_count,
            "cosine_mean": float(np.mean(self.cosine)) if self.cosine.size else math.nan,
            "cosine_median": float(np.median(self.cosine)) if self.cosine.size else math.nan,
            "negative_ratio": (
                float(self.negative_count / self.cosine.size)
                if self.cosine.size
                else math.nan
            ),
            "volatility_mean": (
                float(np.mean(self.volatility)) if self.volatility.size else math.nan
            ),
        }


def empty_samples() -> MetricSamples:
    return MetricSamples(
        cosine=np.empty((0,), dtype=np.float32),
        volatility=np.empty((0,), dtype=np.float32),
        eligible_pairs=0,
        near_zero_pairs=0,
    )


def merge_samples(samples: Iterable[MetricSamples]) -> MetricSamples:
    materialized = list(samples)
    cosines = [item.cosine for item in materialized if item.cosine.size]
    volatilities = [item.volatility for item in materialized if item.volatility.size]
    return MetricSamples(
        cosine=np.concatenate(cosines) if cosines else np.empty((0,), dtype=np.float32),
        volatility=(
            np.concatenate(volatilities)
            if volatilities
            else np.empty((0,), dtype=np.float32)
        ),
        eligible_pairs=sum(item.eligible_pairs for item in materialized),
        near_zero_pairs=sum(item.near_zero_pairs for item in materialized),
    )


def _samples_from_masks(
    cosine: torch.Tensor,
    volatility: torch.Tensor,
    pair_mask: torch.Tensor,
    cosine_valid: torch.Tensor,
) -> MetricSamples:
    eligible = int(pair_mask.sum().item())
    valid_mask = pair_mask & cosine_valid
    near_zero = int((pair_mask & ~cosine_valid).sum().item())
    cosine_values = cosine[valid_mask].detach().float().cpu().numpy()
    volatility_values = volatility[pair_mask].detach().float().cpu().numpy()
    return MetricSamples(cosine_values, volatility_values, eligible, near_zero)


@torch.inference_mode()
def analyze_shared_recurrence(
    tensors: ScanTensors,
    eps: float,
    labels: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> Dict[Tuple[int, str], MetricSamples]:
    """Analyze a single shared state per direction.

    labels and valid use [B,4,L].  Batch size one is intentionally enforced by
    the experiment protocol.
    """
    batch, directions, inner, state_size, length = tensors.shape
    if batch != 1:
        raise ValueError("State contention analysis requires batch size 1")
    if valid is None:
        valid = torch.ones((batch, directions, length), device=tensors.scan_x.device, dtype=torch.bool)
    else:
        valid = valid.bool()
    if labels is not None and labels.shape != (batch, directions, length):
        raise ValueError(f"Label shape {tuple(labels.shape)} does not match scan length")

    hidden = torch.zeros(
        (batch, directions, inner, state_size),
        device=tensors.scan_x.device,
        dtype=torch.float32,
    )
    cosine = torch.zeros((batch, directions, max(length - 1, 0)), device=tensors.scan_x.device)
    volatility = torch.zeros_like(cosine)
    cosine_valid = torch.zeros_like(cosine, dtype=torch.bool)
    previous_update: Optional[torch.Tensor] = None

    for token in range(length):
        delta = F.softplus(tensors.delta_raw[..., token] + tensors.delta_bias.unsqueeze(0))
        transition = torch.exp(delta.unsqueeze(-1) * tensors.A.unsqueeze(0))
        drive = (
            delta.unsqueeze(-1)
            * tensors.B[:, :, None, :, token]
            * tensors.scan_x[..., token].unsqueeze(-1)
        )
        next_hidden = transition * hidden + drive
        update = next_hidden - hidden
        hidden = next_hidden
        if previous_update is not None:
            norm_current = torch.linalg.vector_norm(update, dim=(-2, -1))
            norm_previous = torch.linalg.vector_norm(previous_update, dim=(-2, -1))
            dot = (update * previous_update).sum(dim=(-2, -1))
            cosine[..., token - 1] = dot / (norm_current * norm_previous + eps)
            volatility[..., token - 1] = torch.linalg.vector_norm(
                update - previous_update, dim=(-2, -1)
            ) / (norm_current + norm_previous + eps)
            cosine_valid[..., token - 1] = (norm_current > eps) & (norm_previous > eps)
        previous_update = update

    pair_valid = valid[..., 1:] & valid[..., :-1]
    result: Dict[Tuple[int, str], MetricSamples] = {}
    for direction in range(directions):
        direction_mask = pair_valid[0, direction]
        if labels is None:
            result[(direction, "stream")] = _samples_from_masks(
                cosine[0, direction],
                volatility[0, direction],
                direction_mask,
                cosine_valid[0, direction],
            )
            continue
        same_group = labels[0, direction, 1:] == labels[0, direction, :-1]
        result[(direction, "intra")] = _samples_from_masks(
            cosine[0, direction],
            volatility[0, direction],
            direction_mask & same_group,
            cosine_valid[0, direction],
        )
        result[(direction, "inter")] = _samples_from_masks(
            cosine[0, direction],
            volatility[0, direction],
            direction_mask & ~same_group,
            cosine_valid[0, direction],
        )
    return result


@torch.inference_mode()
def analyze_isolated_recurrence(
    tensors: ScanTensors,
    labels: torch.Tensor,
    num_groups: int,
    eps: float,
) -> Dict[Tuple[int, str], MetricSamples]:
    """Analyze K independent state buffers using the same scan coefficients."""
    batch, directions, inner, state_size, length = tensors.shape
    if batch != 1:
        raise ValueError("State contention analysis requires batch size 1")
    if labels.shape != (batch, directions, length):
        raise ValueError(f"Label shape {tuple(labels.shape)} does not match scan length")

    states = torch.zeros(
        (batch, directions, num_groups, inner, state_size),
        device=tensors.scan_x.device,
        dtype=torch.float32,
    )
    previous_updates = torch.zeros_like(states)
    has_previous = torch.zeros(
        (batch, directions, num_groups),
        device=tensors.scan_x.device,
        dtype=torch.bool,
    )
    cosine = torch.zeros((batch, directions, length), device=tensors.scan_x.device)
    volatility = torch.zeros_like(cosine)
    eligible = torch.zeros_like(cosine, dtype=torch.bool)
    cosine_valid = torch.zeros_like(cosine, dtype=torch.bool)

    for token in range(length):
        group = labels[..., token].long()
        one_hot = F.one_hot(group, num_classes=num_groups).bool()
        selector = one_hot[..., None, None]
        current_hidden = (states * selector).sum(dim=2)
        previous_update = (previous_updates * selector).sum(dim=2)
        had_previous = (has_previous & one_hot).any(dim=2)

        delta = F.softplus(tensors.delta_raw[..., token] + tensors.delta_bias.unsqueeze(0))
        transition = torch.exp(delta.unsqueeze(-1) * tensors.A.unsqueeze(0))
        drive = (
            delta.unsqueeze(-1)
            * tensors.B[:, :, None, :, token]
            * tensors.scan_x[..., token].unsqueeze(-1)
        )
        next_hidden = transition * current_hidden + drive
        update = next_hidden - current_hidden

        norm_current = torch.linalg.vector_norm(update, dim=(-2, -1))
        norm_previous = torch.linalg.vector_norm(previous_update, dim=(-2, -1))
        dot = (update * previous_update).sum(dim=(-2, -1))
        cosine[..., token] = dot / (norm_current * norm_previous + eps)
        volatility[..., token] = torch.linalg.vector_norm(
            update - previous_update, dim=(-2, -1)
        ) / (norm_current + norm_previous + eps)
        eligible[..., token] = had_previous
        cosine_valid[..., token] = had_previous & (norm_current > eps) & (norm_previous > eps)

        states = torch.where(selector, next_hidden.unsqueeze(2), states)
        previous_updates = torch.where(selector, update.unsqueeze(2), previous_updates)
        has_previous = has_previous | one_hot

    result: Dict[Tuple[int, str], MetricSamples] = {}
    for direction in range(directions):
        result[(direction, "stream")] = _samples_from_masks(
            cosine[0, direction],
            volatility[0, direction],
            eligible[0, direction],
            cosine_valid[0, direction],
        )
    return result


class TargetCapture:
    """Non-mutating forward hooks for one decoder block."""

    def __init__(self, decoder_stage: torch.nn.Module, block_index: int):
        self.stage = decoder_stage
        self.block_index = block_index
        self.rm = decoder_stage.attention[block_index]
        self.handles: List[Any] = []
        self.clear()

    def clear(self) -> None:
        self.block_input: Optional[torch.Tensor] = None
        self.assignments: Optional[torch.Tensor] = None
        self.actual_expert_inputs: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def register(self) -> None:
        if self.handles:
            return

        def rm_pre_hook(_module: torch.nn.Module, args: Tuple[Any, ...]) -> None:
            self.block_input = args[0].detach()
            self.assignments = args[1].detach()

        self.handles.append(self.rm.register_forward_pre_hook(rm_pre_hook))

        for expert_index, expert in enumerate(self.rm.mamba_blocks):
            def expert_pre_hook(
                _module: torch.nn.Module,
                args: Tuple[Any, ...],
                index: int = expert_index,
            ) -> None:
                self.actual_expert_inputs[index].append(args[0].detach())

            self.handles.append(expert.register_forward_pre_hook(expert_pre_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def make_metric_row(
    *,
    image_name: str,
    input_path: Path,
    gt_path: Path,
    checkpoint_path: Path,
    stage: str,
    block: int,
    direction: Any,
    parameter_expert: Any,
    mode: str,
    pair_type: str,
    samples: MetricSamples,
    mse: float,
    psnr: float,
    padding_ratio: float = 0.0,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "image": image_name,
        "input_path": str(input_path),
        "gt_path": str(gt_path),
        "checkpoint": str(checkpoint_path),
        "stage": stage,
        "block": int(block),
        "direction": direction,
        "parameter_expert": parameter_expert,
        "mode": mode,
        "pair_type": pair_type,
        "padding_ratio": float(padding_ratio),
        "mse": float(mse),
        "psnr": float(psnr),
    }
    row.update(samples.row_metrics())
    return row


def finite(values: Iterable[Any]) -> List[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def mean_std(values: Iterable[Any]) -> Tuple[float, float, int]:
    numbers = finite(values)
    if not numbers:
        return math.nan, math.nan, 0
    mean = statistics.fmean(numbers)
    std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    return mean, std, len(numbers)


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    group_keys = (
        "stage",
        "block",
        "direction",
        "parameter_expert",
        "mode",
        "pair_type",
    )
    grouped: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    output: List[Dict[str, Any]] = []
    metrics = (
        "cosine_mean",
        "cosine_median",
        "negative_ratio",
        "volatility_mean",
        "near_zero_pair_ratio",
        "mse",
        "psnr",
    )
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate: Dict[str, Any] = dict(zip(group_keys, key))
        aggregate["num_images"] = len({member["image"] for member in members})
        aggregate["eligible_pairs_sum"] = sum(int(member["eligible_pairs"]) for member in members)
        aggregate["cosine_pairs_sum"] = sum(int(member["cosine_pairs"]) for member in members)
        aggregate["near_zero_pairs_sum"] = sum(int(member["near_zero_pairs"]) for member in members)
        aggregate["negative_count_sum"] = sum(int(member["negative_count"]) for member in members)
        for metric in metrics:
            avg, std, count = mean_std(member[metric] for member in members)
            aggregate[f"{metric}_image_mean"] = avg
            aggregate[f"{metric}_image_std"] = std
            aggregate[f"{metric}_valid_images"] = count

        cosine_weight = aggregate["cosine_pairs_sum"]
        eligible_weight = aggregate["eligible_pairs_sum"]
        aggregate["cosine_mean_micro"] = (
            sum(float(m["cosine_mean"]) * int(m["cosine_pairs"]) for m in members if math.isfinite(float(m["cosine_mean"])))
            / cosine_weight
            if cosine_weight
            else math.nan
        )
        aggregate["negative_ratio_micro"] = (
            aggregate["negative_count_sum"] / cosine_weight if cosine_weight else math.nan
        )
        aggregate["volatility_mean_micro"] = (
            sum(float(m["volatility_mean"]) * int(m["eligible_pairs"]) for m in members if math.isfinite(float(m["volatility_mean"])))
            / eligible_weight
            if eligible_weight
            else math.nan
        )
        output.append(aggregate)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    """Convert NaN/Inf and NumPy scalars to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compute_correlations(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["mode"] == "shared_counterfactual"
        and row["pair_type"] == "inter"
        and str(row["direction"]) == "all"
        and str(row["parameter_expert"]) == "all"
    ]
    conflict = np.asarray([1.0 - float(row["cosine_mean"]) for row in candidates], dtype=np.float64)
    result: List[Dict[str, Any]] = []
    try:
        from scipy import stats
    except ImportError:
        stats = None
    for target_name in ("mse", "psnr"):
        target = np.asarray([float(row[target_name]) for row in candidates], dtype=np.float64)
        valid = np.isfinite(conflict) & np.isfinite(target)
        x, y = conflict[valid], target[valid]
        for method in ("pearson", "spearman"):
            coefficient = p_value = math.nan
            if x.size >= 3 and np.std(x) > 0 and np.std(y) > 0:
                if stats is not None:
                    test = stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
                    coefficient, p_value = float(test.statistic), float(test.pvalue)
                elif method == "pearson":
                    coefficient = float(np.corrcoef(x, y)[0, 1])
            result.append(
                {
                    "conflict_score": "1 - shared_inter_cosine_mean",
                    "target": target_name,
                    "method": method,
                    "coefficient": coefficient,
                    "p_value": p_value,
                    "n": int(x.size),
                }
            )
    return result


def plot_histograms(
    output_dir: Path,
    histogram_counts: Mapping[Tuple[str, str], np.ndarray],
    bins: int,
) -> None:
    import matplotlib.pyplot as plt

    edges = np.linspace(-1.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    labels = {
        ("shared_counterfactual", "intra"): "Shared: intra-group",
        ("shared_counterfactual", "inter"): "Shared: inter-group",
        ("isolated_counterfactual", "stream"): "Isolated stream",
    }
    colors = {
        ("shared_counterfactual", "intra"): "#2b83ba",
        ("shared_counterfactual", "inter"): "#d7191c",
        ("isolated_counterfactual", "stream"): "#1a9641",
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for key in PRIMARY_HISTOGRAM_KEYS:
        counts = histogram_counts.get(key, np.zeros(bins, dtype=np.int64)).astype(np.float64)
        density = counts / (counts.sum() * (edges[1] - edges[0])) if counts.sum() else counts
        axis.plot(centers, density, linewidth=2.0, label=labels[key], color=colors[key])
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.set_xlim(-1.0, 1.0)
    axis.set_xlabel("Cosine similarity of consecutive hidden-state updates")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_distribution.png", dpi=220)
    fig.savefig(output_dir / "cosine_distribution.pdf")
    plt.close(fig)


def build_summary(
    checkpoint: Path,
    input_dir: Path,
    gt_dir: Path,
    stage: str,
    block: int,
    num_images: int,
    validation: Mapping[str, Any],
    aggregate: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# State Contention diagnostic",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Input / GT: `{input_dir}` / `{gt_dir}`",
        f"- Target: `{stage}.attention.{block}`",
        f"- Images: {num_images}",
        "- Interpretation: post-hoc same-parameter state toggle; not a trained shared-state baseline.",
        f"- Reference recurrence validation: {'PASS' if validation else 'not run'}",
        "",
        "## Image-level aggregate",
        "",
        "| Condition | Cosine mean | Negative ratio | Volatility |",
        "|---|---:|---:|---:|",
    ]
    for mode, pair_type, label in (
        ("shared_counterfactual", "intra", "Shared / intra"),
        ("shared_counterfactual", "inter", "Shared / inter"),
        ("isolated_counterfactual", "stream", "Isolated stream"),
    ):
        match = next(
            (
                row
                for row in aggregate
                if row["mode"] == mode
                and row["pair_type"] == pair_type
                and str(row["direction"]) == "all"
                and str(row["parameter_expert"]) == "all"
            ),
            None,
        )
        if match is None:
            continue
        lines.append(
            "| {} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} |".format(
                label,
                float(match["cosine_mean_image_mean"]),
                float(match["cosine_mean_image_std"]),
                float(match["negative_ratio_image_mean"]),
                float(match["negative_ratio_image_std"]),
                float(match["volatility_mean_image_mean"]),
                float(match["volatility_mean_image_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The supplied checkpoint has no trained Shared-state Decoder path.",
            "- Actual packed IsoMamba streams use expert-specific parameters and are reported separately.",
            "- A strict Table-4 comparison still requires the matching shared-state implementation and checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", default="decoder_level3")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--hist-bins", type=int, default=100)
    parser.add_argument("--validation-length", type=int, default=128)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-actual-isolated", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only internal tests")
    return parser.parse_args(argv)


def run_internal_self_test() -> None:
    labels = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    expected = torch.tensor(
        [[[0, 1, 2, 3, 4, 5], [0, 3, 1, 4, 2, 5], [5, 4, 3, 2, 1, 0], [5, 2, 4, 1, 3, 0]]]
    )
    actual = direction_sequences(labels)
    if not torch.equal(actual, expected):
        raise AssertionError(f"Direction permutation mismatch: {actual}")

    # A scalar recurrence with A=0 has h_t=h_{t-1}+delta*B*x.
    scan_x = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    delta_raw = torch.zeros_like(scan_x)
    bias = torch.tensor([[math.log(math.expm1(1.0))]])
    tensors = ScanTensors(
        scan_x=scan_x,
        delta_raw=delta_raw,
        A=torch.zeros((1, 1, 1)),
        B=torch.ones((1, 1, 1, 3)),
        C=torch.ones((1, 1, 1, 3)),
        D=torch.zeros((1, 1)),
        delta_bias=bias,
    )
    output, state = reference_scan_full(tensors)
    torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 3.0, 6.0]))
    torch.testing.assert_close(state.flatten(), torch.tensor([6.0]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_internal_self_test()
        print("Internal CPU self-test: PASS")
        return 0

    device = resolve_device(args.device)
    for path, label in (
        (args.config, "config"),
        (args.checkpoint, "checkpoint"),
        (args.input_dir, "input directory"),
        (args.gt_dir, "GT directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    pairs = discover_pairs(args.input_dir, args.gt_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]
    if args.validate_only:
        pairs = pairs[:1]

    seed_everything(args.seed)
    model = load_model(args.config, args.checkpoint, device)
    if not hasattr(model, args.stage):
        raise AttributeError(f"Model has no stage {args.stage!r}")
    stage = getattr(model, args.stage)
    if not 0 <= args.block < len(stage.attention):
        raise IndexError(f"Block {args.block} outside [0,{len(stage.attention) - 1}]")
    target_rm = stage.attention[args.block]
    num_groups = int(stage.num_clusters)
    capture = TargetCapture(stage, args.block)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_result: Dict[str, Any] = {}
    output_equivalence: Dict[str, Any] = {}
    per_image_rows: List[Dict[str, Any]] = []
    histogram_counts: Dict[Tuple[str, str], np.ndarray] = {
        key: np.zeros(args.hist_bins, dtype=np.int64) for key in PRIMARY_HISTOGRAM_KEYS
    }
    histogram_edges = np.linspace(-1.0, 1.0, args.hist_bins + 1)

    try:
        capture.register()
        for image_index, (input_path, gt_path) in enumerate(pairs):
            image_seed = stable_image_seed(args.seed, input_path.name)
            input_tensor = load_rgb_tensor(input_path).to(device)
            gt_tensor = load_rgb_tensor(gt_path).to(device)
            if input_tensor.shape != gt_tensor.shape:
                raise RuntimeError(
                    f"Shape mismatch for {input_path.name}: "
                    f"{tuple(input_tensor.shape)} vs {tuple(gt_tensor.shape)}"
                )

            # On the first image, prove that analysis hooks are observational.
            if image_index == 0 and not args.skip_validation:
                capture.remove()
                seed_everything(image_seed)
                with torch.inference_mode():
                    output_without_hooks = model(input_tensor)
                capture.register()
                capture.clear()
                seed_everything(image_seed)
                with torch.inference_mode():
                    restored = model(input_tensor)
                difference = (restored - output_without_hooks).abs()
                output_equivalence = {
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                    "allclose": bool(torch.allclose(restored, output_without_hooks, rtol=0.0, atol=0.0)),
                }
                if not output_equivalence["allclose"]:
                    raise RuntimeError(f"Hooks changed model output: {output_equivalence}")
            else:
                capture.clear()
                seed_everything(image_seed)
                with torch.inference_mode():
                    restored = model(input_tensor)

            if capture.block_input is None or capture.assignments is None:
                raise RuntimeError("Target decoder hooks did not capture input and assignments")
            labels_spatial = capture.assignments.argmax(dim=-1).long()
            labels_scan = direction_sequences(labels_spatial)
            if labels_spatial.shape[0] != 1:
                raise RuntimeError("Only batch size 1 is supported")
            if not torch.allclose(capture.assignments.sum(-1), torch.ones_like(labels_spatial, dtype=capture.assignments.dtype)):
                raise RuntimeError("Routing assignments are not one-hot in the forward pass")

            if image_index == 0 and not args.skip_validation:
                first_scan = prepare_scan_tensors(target_rm.mamba_blocks[0], capture.block_input)
                validation_result = validate_reference_recurrence(
                    first_scan,
                    max_length=args.validation_length,
                )
                validation_result["hook_output_equivalence"] = output_equivalence
                with (args.output_dir / "validation.json").open("w", encoding="utf-8") as handle:
                    json.dump(validation_result, handle, indent=2)
                if args.validate_only:
                    print(json.dumps(validation_result, indent=2))
                    return 0

            mse = float(F.mse_loss(restored.clamp(0.0, 1.0), gt_tensor).item())
            psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else math.inf
            image_samples: MutableMapping[Tuple[str, str], List[MetricSamples]] = defaultdict(list)

            for expert_index, expert in enumerate(target_rm.mamba_blocks):
                counterfactual_scan = prepare_scan_tensors(expert, capture.block_input)
                if counterfactual_scan.scan_x.shape[-1] != labels_scan.shape[-1]:
                    raise RuntimeError("Counterfactual scan and routing label lengths differ")

                shared = analyze_shared_recurrence(
                    counterfactual_scan,
                    eps=args.eps,
                    labels=labels_scan,
                )
                isolated = analyze_isolated_recurrence(
                    counterfactual_scan,
                    labels=labels_scan,
                    num_groups=num_groups,
                    eps=args.eps,
                )
                for (direction, pair_type), samples in shared.items():
                    per_image_rows.append(
                        make_metric_row(
                            image_name=input_path.name,
                            input_path=input_path,
                            gt_path=gt_path,
                            checkpoint_path=args.checkpoint,
                            stage=args.stage,
                            block=args.block,
                            direction=DIRECTION_NAMES[direction],
                            parameter_expert=expert_index,
                            mode="shared_counterfactual",
                            pair_type=pair_type,
                            samples=samples,
                            mse=mse,
                            psnr=psnr,
                        )
                    )
                    image_samples[("shared_counterfactual", pair_type)].append(samples)
                    histogram_counts[("shared_counterfactual", pair_type)] += np.histogram(
                        samples.cosine, bins=histogram_edges
                    )[0]
                for (direction, pair_type), samples in isolated.items():
                    per_image_rows.append(
                        make_metric_row(
                            image_name=input_path.name,
                            input_path=input_path,
                            gt_path=gt_path,
                            checkpoint_path=args.checkpoint,
                            stage=args.stage,
                            block=args.block,
                            direction=DIRECTION_NAMES[direction],
                            parameter_expert=expert_index,
                            mode="isolated_counterfactual",
                            pair_type=pair_type,
                            samples=samples,
                            mse=mse,
                            psnr=psnr,
                        )
                    )
                    image_samples[("isolated_counterfactual", pair_type)].append(samples)
                    histogram_counts[("isolated_counterfactual", pair_type)] += np.histogram(
                        samples.cosine, bins=histogram_edges
                    )[0]

            if not args.skip_actual_isolated:
                counts = torch.bincount(labels_spatial.flatten(), minlength=num_groups)
                for expert_index, calls in sorted(capture.actual_expert_inputs.items()):
                    if len(calls) != 1:
                        raise RuntimeError(
                            f"Expected one batch-1 call for expert {expert_index}, got {len(calls)}"
                        )
                    packed_input = calls[0]
                    valid_count = int(counts[expert_index].item())
                    side_h, side_w = packed_input.shape[1:3]
                    if side_h != side_w or side_h * side_w < valid_count:
                        raise RuntimeError("Unexpected packed expert tensor shape")
                    base_valid = torch.zeros(
                        (1, side_h * side_w),
                        device=packed_input.device,
                        dtype=torch.bool,
                    )
                    base_valid[:, :valid_count] = True
                    valid_scan = direction_sequences(base_valid.view(1, side_h, side_w))
                    actual_scan = prepare_scan_tensors(target_rm.mamba_blocks[expert_index], packed_input)
                    actual = analyze_shared_recurrence(
                        actual_scan,
                        eps=args.eps,
                        labels=None,
                        valid=valid_scan,
                    )
                    padding_ratio = 1.0 - valid_count / float(side_h * side_w)
                    for (direction, pair_type), samples in actual.items():
                        per_image_rows.append(
                            make_metric_row(
                                image_name=input_path.name,
                                input_path=input_path,
                                gt_path=gt_path,
                                checkpoint_path=args.checkpoint,
                                stage=args.stage,
                                block=args.block,
                                direction=DIRECTION_NAMES[direction],
                                parameter_expert=expert_index,
                                mode="isolated_actual",
                                pair_type=pair_type,
                                samples=samples,
                                mse=mse,
                                psnr=psnr,
                                padding_ratio=padding_ratio,
                            )
                        )
                        image_samples[("isolated_actual", pair_type)].append(samples)

            # Exact per-image overall rows are built from scalar samples only;
            # full hidden-state sequences have already been discarded.
            for (mode, pair_type), samples_list in image_samples.items():
                combined = merge_samples(samples_list)
                per_image_rows.append(
                    make_metric_row(
                        image_name=input_path.name,
                        input_path=input_path,
                        gt_path=gt_path,
                        checkpoint_path=args.checkpoint,
                        stage=args.stage,
                        block=args.block,
                        direction="all",
                        parameter_expert="all",
                        mode=mode,
                        pair_type=pair_type,
                        samples=combined,
                        mse=mse,
                        psnr=psnr,
                    )
                )

            print(
                f"[{image_index + 1}/{len(pairs)}] {input_path.name}: "
                f"PSNR={psnr:.4f}, captured groups={counts.tolist() if not args.skip_actual_isolated else 'n/a'}"
            )

    finally:
        capture.remove()

    aggregate = aggregate_rows(per_image_rows)
    correlations = compute_correlations(per_image_rows)
    write_csv(args.output_dir / "per_image.csv", per_image_rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    write_csv(args.output_dir / "correlations.csv", correlations)
    plot_histograms(args.output_dir, histogram_counts, args.hist_bins)

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "gt_dir": str(args.gt_dir.resolve()),
        "stage": args.stage,
        "block": args.block,
        "seed": args.seed,
        "eps": args.eps,
        "num_images": len(pairs),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "validation": validation_result,
        "interpretation": (
            "Post-hoc same-parameter state-sharing diagnostic; "
            "not a trained Shared-state Decoder baseline."
        ),
        "aggregate": aggregate,
    }
    with (args.output_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, allow_nan=False)
    summary = build_summary(
        args.checkpoint,
        args.input_dir,
        args.gt_dir,
        args.stage,
        args.block,
        len(pairs),
        validation_result,
        aggregate,
    )
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Analysis complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

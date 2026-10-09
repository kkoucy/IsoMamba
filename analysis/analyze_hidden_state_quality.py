#!/usr/bin/env python3
"""Hidden-State Response Quality diagnostic for an IsoMamba checkpoint.

This analysis-only script compares exactly two target-layer response modes:

* Shared Proxy: all original-grid tokens use expert 0, with one recurrent
  state per scan direction.
* State Isolation: the checkpoint's real Delta-guided labels gather tokens
  into expert-specific packed grids, each expert scans with its own state,
  and responses are pushed back to the original grid.

The measured response is only the selective-scan state projection C_t h_t.
It excludes D_t x_t, gates, residuals, FFNs, FiLM, normalization, output
projection, and subsequent fusion. Shared Proxy is a post-hoc single-expert
diagnostic built from the final IsoMamba checkpoint, not a trained shared-state
decoder baseline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import shlex
import statistics
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/isomamba_hidden_state_quality_mpl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze_state_contention import (  # noqa: E402
    ScanTensors,
    TargetCapture,
    load_model,
    prepare_scan_tensors,
    reference_scan_full,
    seed_everything,
    validate_reference_recurrence,
)
from analysis.runtime_options import (  # noqa: E402
    add_dataset_arguments, apply_dataset_arguments, runtime_command,
)


DEFAULT_CONFIG = REPO_ROOT / "options" / "train_IsoMamba.yml"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "experiments" / "ClusterMamba"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis_outputs" / "hidden_state_quality"
DATASETS = {
    "uieb": {
        "display": "UIEB",
        "input": Path("/home/ly/Data/UIE/UIEB/test/Input256"),
        "gt": Path("/home/ly/Data/UIE/UIEB/test/GT256"),
    },
    "lsui": {
        "display": "LSUI",
        "input": Path("/home/ly/Data/UIE/LSUI/Input256"),
        "gt": Path("/home/ly/Data/UIE/LSUI/GT256"),
    },
    "uieb_original": {
        "display": "UIEB Original",
        "input": Path("/home/ly/Data/UIE/UIEB/test/Input"),
        "gt": Path("/home/ly/Data/UIE/UIEB/test/GT"),
    },
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(value: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def stable_seed(base_seed: int, *parts: str) -> int:
    value = int(base_seed)
    for part in parts:
        value = (value + zlib.crc32(part.encode("utf-8"))) % (2**31 - 1)
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_with_fields(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write a CSV with a header even when there are no data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_progress(path: Path) -> Optional[Tuple[int, str]]:
    """Infer training progress from iteration/step/epoch or conventional suffix."""
    stem = path.stem.lower()
    tagged = re.findall(r"(?:iteration|iter|step|epoch)[_-]?(\d+)", stem)
    if tagged:
        return max(int(item) for item in tagged), "tagged"
    suffix = re.search(r"(?:^|[_-])(\d+)$", stem)
    if suffix:
        return int(suffix.group(1)), "numeric_suffix"
    return None


def discover_checkpoint(root: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates = sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
        ),
        key=lambda path: str(path),
    )
    if not candidates:
        raise FileNotFoundError(f"No .pth/.pt/.ckpt checkpoint found recursively under {root}")
    parsed = [(path, checkpoint_progress(path)) for path in candidates]
    interpretable = [(path, progress) for path, progress in parsed if progress is not None]
    if interpretable:
        selected, progress = max(
            interpretable,
            key=lambda item: (item[1][0], item[0].stat().st_mtime, str(item[0])),
        )
        reason = (
            f"largest parseable training progress ({progress[0]}, "
            f"source={progress[1]}); mtime used only as a tie-breaker"
        )
    else:
        selected = max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))
        progress = None
        reason = "no filename had parseable progress; selected latest modification time"
    details = {
        "search_root": str(root.resolve()),
        "candidate_count": len(candidates),
        "parseable_candidate_count": len(interpretable),
        "selected": str(selected),
        "selected_progress": progress[0] if progress else None,
        "selected_progress_source": progress[1] if progress else None,
        "selection_reason": reason,
    }
    return selected, details


@dataclass(frozen=True)
class ImagePair:
    stem: str
    input_path: Path
    gt_path: Path


def _image_files(directory: Path) -> List[Path]:
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )


def pair_by_stem(input_dir: Path, gt_dir: Path) -> Tuple[List[ImagePair], Dict[str, Any]]:
    input_files = _image_files(input_dir)
    gt_files = _image_files(gt_dir)
    input_map: Dict[str, List[Path]] = {}
    gt_map: Dict[str, List[Path]] = {}
    for path in input_files:
        input_map.setdefault(path.stem, []).append(path)
    for path in gt_files:
        gt_map.setdefault(path.stem, []).append(path)

    duplicate_input = {
        stem: [path.name for path in paths] for stem, paths in input_map.items() if len(paths) > 1
    }
    duplicate_gt = {
        stem: [path.name for path in paths] for stem, paths in gt_map.items() if len(paths) > 1
    }
    ambiguous = set(duplicate_input) | set(duplicate_gt)
    shared_stems = (set(input_map) & set(gt_map)) - ambiguous
    pairs = [
        ImagePair(stem, input_map[stem][0], gt_map[stem][0])
        for stem in sorted(shared_stems, key=natural_key)
    ]
    paired_input_paths = {pair.input_path for pair in pairs}
    paired_gt_paths = {pair.gt_path for pair in pairs}
    report = {
        "input_dir": str(input_dir.resolve()),
        "gt_dir": str(gt_dir.resolve()),
        "input_count": len(input_files),
        "gt_count": len(gt_files),
        "paired_count": len(pairs),
        "unpaired_input_files": [
            path.name for path in input_files if path not in paired_input_paths
        ],
        "unpaired_gt_files": [path.name for path in gt_files if path not in paired_gt_paths],
        "duplicate_input_stems": duplicate_input,
        "duplicate_gt_stems": duplicate_gt,
        "pairing_key": "Path.stem",
    }
    if not pairs:
        raise RuntimeError(f"No unambiguous stem-paired images: {report}")
    return pairs, report


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.inference_mode()
def state_projection_scan(tensors: ScanTensors) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return z_t=C_t h_t in scan order and final true recurrent state, in FP32."""
    batch, directions, inner, state_size, length = tensors.shape
    hidden = torch.zeros(
        (batch, directions, inner, state_size),
        device=tensors.scan_x.device,
        dtype=torch.float32,
    )
    projected = torch.empty(
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
        projected[..., token] = (
            hidden * tensors.C[:, :, None, :, token]
        ).sum(dim=-1)
    return projected, hidden


def inverse_scan_sum(z_scan: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse the four SS2D scan orders and sum them as the production block."""
    if z_scan.ndim != 4 or z_scan.shape[1] != 4:
        raise ValueError(f"Expected z_scan [B,4,D,L], got {tuple(z_scan.shape)}")
    batch, _, inner, length = z_scan.shape
    if length != height * width:
        raise ValueError(f"Scan length {length} != H*W {height * width}")
    inverse = z_scan[:, 2:4].flip(-1)
    row_forward = z_scan[:, 0]
    row_reverse = inverse[:, 0]
    column_forward = (
        z_scan[:, 1]
        .view(batch, inner, width, height)
        .transpose(2, 3)
        .contiguous()
        .view(batch, inner, length)
    )
    column_reverse = (
        inverse[:, 1]
        .view(batch, inner, width, height)
        .transpose(2, 3)
        .contiguous()
        .view(batch, inner, length)
    )
    combined = row_forward + row_reverse + column_forward + column_reverse
    return combined.transpose(1, 2).contiguous().view(batch, height, width, inner)


@torch.inference_mode()
def validate_state_projection(
    tensors: ScanTensors,
    validation_length: int,
) -> Dict[str, Any]:
    base = validate_reference_recurrence(tensors, max_length=validation_length)
    sample = tensors.prefix(validation_length)
    z_only, final_state = state_projection_scan(sample)
    full_reference, reference_state = reference_scan_full(sample)
    reconstructed = z_only + sample.D.unsqueeze(0).unsqueeze(-1) * sample.scan_x
    decomposition_error = (reconstructed - full_reference).abs()
    state_error = (final_state - reference_state).abs()
    base.update(
        {
            "representation": "z_t = C_t h_t only",
            "excluded_terms": [
                "D_t x_t",
                "residual",
                "FFN",
                "FiLM/global-state fusion",
                "out_norm",
                "gate",
                "out_proj",
            ],
            "z_scan_shape": list(z_only.shape),
            "decomposition_max_abs": float(decomposition_error.max().item()),
            "decomposition_mean_abs": float(decomposition_error.mean().item()),
            "z_recurrence_state_max_abs": float(state_error.max().item()),
        }
    )
    if base["decomposition_max_abs"] > 1e-6 or base["z_recurrence_state_max_abs"] > 1e-6:
        raise RuntimeError(f"C*h decomposition validation failed: {base}")
    return base


@torch.inference_mode()
def shared_proxy_response(
    target_rm: torch.nn.Module,
    target_input: torch.Tensor,
) -> Tuple[torch.Tensor, ScanTensors, Dict[str, Any]]:
    """All original-grid tokens through expert 0, one state per direction."""
    if target_input.shape[0] != 1:
        raise ValueError("Analysis supports batch size one")
    height, width = target_input.shape[1:3]
    expert0 = target_rm.mamba_blocks[0]
    tensors = prepare_scan_tensors(expert0, target_input)
    z_scan, final_state = state_projection_scan(tensors)
    z_map = inverse_scan_sum(z_scan, height, width)
    semantics = {
        "parameter_experts_used": [0],
        "cluster_gather_used": False,
        "states_per_direction": 1,
        "direction_states_independent": True,
        "scan_shape": list(z_scan.shape),
        "final_state_shape": list(final_state.shape),
        "z_map_shape": list(z_map.shape),
    }
    return z_map, tensors, semantics


@torch.inference_mode()
def isolated_response(
    target_rm: torch.nn.Module,
    target_input: torch.Tensor,
    labels: torch.Tensor,
    captured_expert_inputs: Mapping[int, List[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reproduce actual gather -> expert packed scan -> push-back for C_t h_t."""
    if target_input.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Analysis supports batch size one")
    _, height, width, channels = target_input.shape
    num_groups = len(target_rm.mamba_blocks)
    flat_input = target_input.view(height * width, channels)
    flat_labels = labels.view(-1)
    output: Optional[torch.Tensor] = None
    expert_details: List[Dict[str, Any]] = []
    gather_max_abs = 0.0
    padding_max_abs = 0.0
    used_experts: List[int] = []

    for expert_index in range(num_groups):
        positions = (flat_labels == expert_index).nonzero(as_tuple=False).squeeze(-1)
        count = int(positions.numel())
        calls = captured_expert_inputs.get(expert_index, [])
        if count == 0:
            if calls:
                raise RuntimeError(f"Empty expert {expert_index} unexpectedly received a call")
            expert_details.append({"expert": expert_index, "token_count": 0, "active": False})
            continue
        if len(calls) != 1:
            raise RuntimeError(
                f"Expected one actual packed call for expert {expert_index}, got {len(calls)}"
            )
        packed_input = calls[0].float()
        side_h, side_w = packed_input.shape[1:3]
        if side_h != side_w or side_h * side_w < count:
            raise RuntimeError(
                f"Expert {expert_index}: packed shape {tuple(packed_input.shape)} invalid for {count}"
            )
        packed_flat = packed_input.view(side_h * side_w, channels)
        expected = flat_input.index_select(0, positions).float()
        gather_error = float((packed_flat[:count] - expected).abs().max().item())
        gather_max_abs = max(gather_max_abs, gather_error)
        if side_h * side_w > count:
            pad_error = float(packed_flat[count:].abs().max().item())
            padding_max_abs = max(padding_max_abs, pad_error)
        tensors = prepare_scan_tensors(target_rm.mamba_blocks[expert_index], packed_input)
        z_scan, final_state = state_projection_scan(tensors)
        packed_z = inverse_scan_sum(z_scan, side_h, side_w).view(side_h * side_w, -1)
        if output is None:
            output = torch.empty(
                (height * width, packed_z.shape[-1]),
                device=packed_z.device,
                dtype=torch.float32,
            )
        output.index_copy_(0, positions, packed_z[:count])
        used_experts.append(expert_index)
        expert_details.append(
            {
                "expert": expert_index,
                "token_count": count,
                "active": True,
                "packed_input_shape": list(packed_input.shape),
                "packed_side": side_h,
                "padding_count": side_h * side_w - count,
                "scan_shape": list(z_scan.shape),
                "final_state_shape": list(final_state.shape),
                "gather_max_abs": gather_error,
            }
        )

    if output is None:
        raise RuntimeError("No active expert produced an isolated response")
    if gather_max_abs != 0.0 or padding_max_abs != 0.0:
        raise RuntimeError(
            f"Actual packed gather mismatch: gather={gather_max_abs}, padding={padding_max_abs}"
        )
    z_map = output.view(1, height, width, -1)
    semantics = {
        "parameter_experts_used": used_experts,
        "cluster_gather_used": True,
        "push_back_used": True,
        "states_per_active_expert_per_direction": 1,
        "direction_states_independent": True,
        "active_expert_count": len(used_experts),
        "gather_max_abs": gather_max_abs,
        "padding_max_abs": padding_max_abs,
        "expert_details": expert_details,
        "z_map_shape": list(z_map.shape),
    }
    return z_map, semantics


def stratified_token_sample(
    labels: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    total = labels.size
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    groups = [int(item) for item in np.unique(labels)]
    shuffled: Dict[int, np.ndarray] = {}
    for group in groups:
        indices = np.flatnonzero(labels == group)
        shuffled[group] = rng.permutation(indices)
    selected: List[int] = []
    cursors = {group: 0 for group in groups}
    # Round-robin sampling gives every active group representation and prevents
    # a dominant group from consuming the budget.
    while len(selected) < maximum:
        progressed = False
        for group in groups:
            cursor = cursors[group]
            if cursor < len(shuffled[group]) and len(selected) < maximum:
                selected.append(int(shuffled[group][cursor]))
                cursors[group] += 1
                progressed = True
        if not progressed:
            break
    return np.asarray(selected, dtype=np.int64)


def sample_unique_pairs(
    token_count: int,
    maximum: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    first, second = np.triu_indices(token_count, k=1)
    if first.size > maximum:
        chosen = rng.choice(first.size, size=maximum, replace=False)
        first, second = first[chosen], second[chosen]
    return first.astype(np.int64), second.astype(np.int64)


def correction_alignment(
    z_flat: torch.Tensor,
    correction_flat: torch.Tensor,
    sampled_positions: np.ndarray,
    pair_first: np.ndarray,
    pair_second: np.ndarray,
    eps: float,
) -> Tuple[float, str]:
    if pair_first.size < 3:
        return math.nan, "fewer_than_3_valid_pairs"
    positions = torch.from_numpy(sampled_positions).to(z_flat.device)
    pair_a = torch.from_numpy(pair_first).to(z_flat.device)
    pair_b = torch.from_numpy(pair_second).to(z_flat.device)
    z = F.normalize(z_flat.index_select(0, positions).float(), p=2, dim=-1, eps=eps)
    correction = correction_flat.index_select(0, positions).float()
    dz = 1.0 - (z.index_select(0, pair_a) * z.index_select(0, pair_b)).sum(dim=-1)
    dc = torch.linalg.vector_norm(
        correction.index_select(0, pair_a) - correction.index_select(0, pair_b),
        dim=-1,
    )
    dz_np = dz.cpu().numpy().astype(np.float64)
    dc_np = dc.cpu().numpy().astype(np.float64)
    valid = np.isfinite(dz_np) & np.isfinite(dc_np)
    dz_np, dc_np = dz_np[valid], dc_np[valid]
    if dz_np.size < 3:
        return math.nan, "fewer_than_3_finite_pairs"
    if np.ptp(dz_np) == 0.0:
        return math.nan, "constant_hidden_distance"
    if np.ptp(dc_np) == 0.0:
        return math.nan, "constant_correction_distance"
    from scipy import stats

    result = stats.spearmanr(dz_np, dc_np)
    coefficient = float(result.statistic)
    if not math.isfinite(coefficient):
        return math.nan, "spearman_non_finite"
    return coefficient, "ok"


def group_separation(
    z_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    eps: float,
) -> Tuple[float, float, float]:
    z_normalized = F.normalize(z_flat.float(), p=2, dim=-1, eps=eps)
    global_center = z_normalized.mean(dim=0)
    between = torch.zeros((), device=z_flat.device, dtype=torch.float32)
    within = torch.zeros_like(between)
    for group in labels_flat.unique(sorted=True):
        group_z = z_normalized[labels_flat == group]
        if group_z.numel() == 0:
            continue
        center = group_z.mean(dim=0)
        between = between + group_z.shape[0] * (center - global_center).square().sum()
        within = within + (group_z - center).square().sum()
    separation = between / (between + within + eps)
    return float(separation.item()), float(between.item()), float(within.item())


@torch.inference_mode()
def group_separation_chunked(
    z_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    num_groups: int,
    eps: float,
    chunk_size: int = 4096,
) -> Tuple[float, float, float]:
    """Two-pass Group Separation without materializing a full normalized copy."""
    feature_dim = z_flat.shape[-1]
    sums = torch.zeros(
        (num_groups, feature_dim), device=z_flat.device, dtype=torch.float32
    )
    counts = torch.zeros((num_groups,), device=z_flat.device, dtype=torch.long)
    global_sum = torch.zeros((feature_dim,), device=z_flat.device, dtype=torch.float32)
    total = int(z_flat.shape[0])
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        normalized = F.normalize(
            z_flat[start:stop].float(), p=2, dim=-1, eps=eps
        )
        chunk_labels = labels_flat[start:stop]
        global_sum += normalized.sum(dim=0)
        for group in range(num_groups):
            mask = chunk_labels == group
            count = int(mask.sum().item())
            if count:
                sums[group] += normalized[mask].sum(dim=0)
                counts[group] += count
        del normalized
    global_center = global_sum / max(total, 1)
    centers = torch.zeros_like(sums)
    active = counts > 0
    centers[active] = sums[active] / counts[active, None]
    between = (
        counts[active].float()[:, None]
        * (centers[active] - global_center).square()
    ).sum()
    within = torch.zeros((), device=z_flat.device, dtype=torch.float32)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        normalized = F.normalize(
            z_flat[start:stop].float(), p=2, dim=-1, eps=eps
        )
        chunk_labels = labels_flat[start:stop]
        within += (normalized - centers.index_select(0, chunk_labels)).square().sum()
        del normalized
    separation = between / (between + within + eps)
    return float(separation.item()), float(between.item()), float(within.item())


def cuda_peak_mb() -> Tuple[float, float]:
    return (
        float(torch.cuda.max_memory_allocated(0) / (1024**2)),
        float(torch.cuda.max_memory_reserved(0) / (1024**2)),
    )


def is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return (
        isinstance(error, RuntimeError)
        and "out of memory" in message
        and ("cuda" in message or "cublas" in message or "cudnn" in message)
    )


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_std(values: Iterable[float]) -> Dict[str, Any]:
    materialized = [float(value) for value in values]
    finite = [value for value in materialized if math.isfinite(value)]
    return {
        "mean": statistics.fmean(finite) if finite else math.nan,
        "std": statistics.stdev(finite) if len(finite) > 1 else (0.0 if finite else math.nan),
        "n": len(finite),
        "excluded": len(materialized) - len(finite),
    }


def paired_statistics(
    shared_values: Sequence[float],
    isolated_values: Sequence[float],
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    shared = np.asarray(shared_values, dtype=np.float64)
    isolated = np.asarray(isolated_values, dtype=np.float64)
    valid = np.isfinite(shared) & np.isfinite(isolated)
    paired_shared = shared[valid]
    paired_isolated = isolated[valid]
    gaps = paired_isolated - paired_shared
    n = gaps.size
    summary: Dict[str, Any] = {
        "paired_n": int(n),
        "excluded_images": int(shared.size - n),
        "gap_mean": float(np.mean(gaps)) if n else math.nan,
        "gap_median": float(np.median(gaps)) if n else math.nan,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_95_ci": [math.nan, math.nan],
        "positive_count": int(np.count_nonzero(gaps > 0.0)),
        "positive_ratio": float(np.mean(gaps > 0.0)) if n else math.nan,
    }
    if n:
        rng = np.random.default_rng(seed)
        # Chunk bootstrap resampling to keep memory bounded for hundreds of images.
        bootstrap_means = np.empty(bootstrap_iterations, dtype=np.float64)
        chunk = 1000
        for start in range(0, bootstrap_iterations, chunk):
            stop = min(start + chunk, bootstrap_iterations)
            indices = rng.integers(0, n, size=(stop - start, n))
            bootstrap_means[start:stop] = gaps[indices].mean(axis=1)
        summary["bootstrap_95_ci"] = [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ]

    tests: Dict[str, Any] = {
        "paired_n": int(n),
        "paired_t_test": {"statistic": math.nan, "p_value": math.nan},
        "wilcoxon_signed_rank": {"statistic": math.nan, "p_value": math.nan},
        "cohens_dz": math.nan,
    }
    if n >= 2:
        from scipy import stats

        t_result = stats.ttest_rel(paired_isolated, paired_shared, nan_policy="omit")
        tests["paired_t_test"] = {
            "statistic": float(t_result.statistic),
            "p_value": float(t_result.pvalue),
        }
        if np.all(gaps == 0.0):
            tests["wilcoxon_signed_rank"] = {"statistic": 0.0, "p_value": 1.0}
        else:
            w_result = stats.wilcoxon(
                paired_isolated,
                paired_shared,
                zero_method="wilcox",
                alternative="two-sided",
            )
            tests["wilcoxon_signed_rank"] = {
                "statistic": float(w_result.statistic),
                "p_value": float(w_result.pvalue),
            }
        gap_std = float(np.std(gaps, ddof=1))
        if gap_std > 0.0:
            tests["cohens_dz"] = float(np.mean(gaps) / gap_std)
        elif np.mean(gaps) != 0.0:
            tests["cohens_dz"] = math.copysign(math.inf, float(np.mean(gaps)))
        else:
            tests["cohens_dz"] = 0.0
    return summary, tests


def aggregate_dataset(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    alignment_shared = [float(row["alignment_shared"]) for row in rows]
    alignment_isolated = [float(row["alignment_isolated"]) for row in rows]
    separation_shared = [float(row["separation_shared"]) for row in rows]
    separation_isolated = [float(row["separation_isolated"]) for row in rows]
    alignment_gap, alignment_tests = paired_statistics(
        alignment_shared,
        alignment_isolated,
        bootstrap_iterations,
        stable_seed(seed, "alignment_bootstrap"),
    )
    separation_gap, separation_tests = paired_statistics(
        separation_shared,
        separation_isolated,
        bootstrap_iterations,
        stable_seed(seed, "separation_bootstrap"),
    )
    aggregate = {
        "image_count": len(rows),
        "correction_alignment": {
            "shared_proxy": mean_std(alignment_shared),
            "state_isolation": mean_std(alignment_isolated),
            "paired_gap_isolated_minus_shared": alignment_gap,
        },
        "group_separation": {
            "shared_proxy": mean_std(separation_shared),
            "state_isolation": mean_std(separation_isolated),
            "paired_gap_isolated_minus_shared": separation_gap,
        },
    }
    tests = {
        "correction_alignment": alignment_tests,
        "group_separation": separation_tests,
        "alternative": "two-sided",
        "gap_definition": "State Isolation - Shared Proxy",
    }
    return aggregate, tests


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NaN"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NaN"


def interpretation(aggregate: Mapping[str, Any], tests: Mapping[str, Any]) -> str:
    alignment = aggregate["correction_alignment"]
    separation = aggregate["group_separation"]
    alignment_up = (
        alignment["state_isolation"]["mean"] > alignment["shared_proxy"]["mean"]
    )
    separation_up = (
        separation["state_isolation"]["mean"] > separation["shared_proxy"]["mean"]
    )
    alignment_p = tests["correction_alignment"]["paired_t_test"]["p_value"]
    separation_p = tests["group_separation"]["paired_t_test"]["p_value"]
    alignment_ci = alignment["paired_gap_isolated_minus_shared"]["bootstrap_95_ci"]
    separation_ci = separation["paired_gap_isolated_minus_shared"]["bootstrap_95_ci"]
    alignment_significant = (
        math.isfinite(float(alignment_p))
        and float(alignment_p) < 0.05
        and float(alignment_ci[0]) > 0.0
    )
    separation_significant = (
        math.isfinite(float(separation_p))
        and float(separation_p) < 0.05
        and float(separation_ci[0]) > 0.0
    )
    if alignment_up and separation_up:
        if alignment_significant and separation_significant:
            claim = (
                "State Isolation has higher Correction Alignment and Group Separation, with "
                "positive bootstrap intervals and paired t-tests below 0.05 for both metrics. "
                "In this checkpoint, state-isolated responses better preserve the Delta-defined "
                "group structure and have a more consistent representational relationship with "
                "GT restoration corrections."
            )
        elif separation_significant:
            claim = (
                "State Isolation has substantially higher Group Separation. Its mean Correction "
                "Alignment is also higher, but the Alignment uncertainty includes zero and the "
                "paired test is not significant at 0.05. This strongly supports improved group "
                "distinction, while evidence of better agreement with the restoration target is "
                "inconclusive for this dataset."
            )
        else:
            claim = (
                "State Isolation has higher means for both metrics, but the paired uncertainty "
                "does not support a firm improvement claim at 0.05."
            )
    elif separation_up and not alignment_up:
        claim = (
            "State Isolation improves mean Group Separation but not mean Correction Alignment. "
            "This supports stronger group distinction only; it does not show better agreement "
            "with the restoration target."
        )
    elif alignment_up and not separation_up:
        claim = (
            "State Isolation improves mean Correction Alignment but not mean Group Separation. "
            "The result does not jointly support improved Delta-defined group structure."
        )
    else:
        claim = (
            "Neither metric has a higher mean under State Isolation. The requested hypothesis "
            "is not supported by this diagnostic."
        )
    return (
        claim
        + " Paired t-test p-values are "
        + f"{fmt(alignment_p)} (Alignment) and {fmt(separation_p)} (Separation); "
        + "effect direction and statistical uncertainty should be considered together."
    )


def dataset_report_markdown(
    display_name: str,
    checkpoint: Path,
    pairing: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    tests: Mapping[str, Any],
    validation: Mapping[str, Any],
    first_shapes: Mapping[str, Any],
) -> str:
    alignment = aggregate["correction_alignment"]
    separation = aggregate["group_separation"]
    a_gap = alignment["paired_gap_isolated_minus_shared"]
    s_gap = separation["paired_gap_isolated_minus_shared"]
    lines = [
        f"# {display_name} Hidden-State Response Quality",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Target layer: `decoder_level3.attention.0`",
        f"- Images analyzed: {aggregate['image_count']}",
        (
            f"- Pairing: inputs={pairing['input_count']}, GT={pairing['gt_count']}, "
            f"paired={pairing['paired_count']} (matched by `Path.stem`)"
        ),
        f"- Unpaired inputs: `{json.dumps(pairing['unpaired_input_files'], ensure_ascii=False)}`",
        f"- Unpaired GT: `{json.dumps(pairing['unpaired_gt_files'], ensure_ascii=False)}`",
        (
            "- Representation: selective-scan state projection `z_t = C_t h_t`; "
            "`D_t x_t`, residual, FFN, FiLM/global fusion, gating, normalization, "
            "and output projection are excluded."
        ),
        (
            "- Limitation: Shared Proxy is a post-hoc single-expert shared-state diagnostic "
            "using expert 0 from the final IsoMamba checkpoint. It is not a trained "
            "Shared-state Decoder baseline."
        ),
        "",
        "## Main results",
        "",
        "| State mode | Correction Alignment ↑ | Group Separation ↑ |",
        "|---|---:|---:|",
        (
            "| Shared Proxy | "
            f"{fmt(alignment['shared_proxy']['mean'])} ± {fmt(alignment['shared_proxy']['std'])} | "
            f"{fmt(separation['shared_proxy']['mean'])} ± {fmt(separation['shared_proxy']['std'])} |"
        ),
        (
            "| State Isolation | "
            f"{fmt(alignment['state_isolation']['mean'])} ± {fmt(alignment['state_isolation']['std'])} | "
            f"{fmt(separation['state_isolation']['mean'])} ± {fmt(separation['state_isolation']['std'])} |"
        ),
        "",
        "## Paired gaps (State Isolation - Shared Proxy)",
        "",
        "| Metric | Mean | Median | Bootstrap 95% CI | Positive | Excluded |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Correction Alignment | {fmt(a_gap['gap_mean'])} | {fmt(a_gap['gap_median'])} | "
            f"[{fmt(a_gap['bootstrap_95_ci'][0])}, {fmt(a_gap['bootstrap_95_ci'][1])}] | "
            f"{a_gap['positive_count']}/{a_gap['paired_n']} ({fmt(a_gap['positive_ratio'] * 100, 2)}%) | "
            f"{a_gap['excluded_images']} |"
        ),
        (
            f"| Group Separation | {fmt(s_gap['gap_mean'])} | {fmt(s_gap['gap_median'])} | "
            f"[{fmt(s_gap['bootstrap_95_ci'][0])}, {fmt(s_gap['bootstrap_95_ci'][1])}] | "
            f"{s_gap['positive_count']}/{s_gap['paired_n']} ({fmt(s_gap['positive_ratio'] * 100, 2)}%) | "
            f"{s_gap['excluded_images']} |"
        ),
        "",
        "## Paired statistical tests",
        "",
        "| Metric | Paired t-test p | Wilcoxon p | Cohen's dz | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("correction_alignment", "Correction Alignment"),
        ("group_separation", "Group Separation"),
    ):
        item = tests[key]
        lines.append(
            f"| {label} | {fmt(item['paired_t_test']['p_value'])} | "
            f"{fmt(item['wilcoxon_signed_rank']['p_value'])} | "
            f"{fmt(item['cohens_dz'])} | {item['paired_n']} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            (
                "- FP32 reference vs fused selective scan: "
                f"max abs={fmt(validation['fused_output_max_abs'], 10)}, "
                f"mean abs={fmt(validation['fused_output_mean_abs'], 10)}."
            ),
            (
                "- `C_t h_t + D_t x_t` decomposition vs full reference: "
                f"max abs={fmt(validation['decomposition_max_abs'], 10)}, "
                f"mean abs={fmt(validation['decomposition_mean_abs'], 10)}."
            ),
            f"- First-image tensor inspection: `{json.dumps(first_shapes)}`",
            "",
            "## Interpretation",
            "",
            interpretation(aggregate, tests),
            "",
        ]
    )
    return "\n".join(lines)


def cross_dataset_report(
    checkpoint: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> str:
    display_names = {
        "uieb": "UIEB-256",
        "lsui": "LSUI-256",
        "uieb_original": "UIEB-Original",
    }
    ordered_keys = ("uieb", "lsui", "uieb_original")
    lines = [
        "# Cross-dataset Hidden-State Response Quality",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Target layer: `decoder_level3.attention.0`",
        (
            "- Shared Proxy is a post-hoc single-expert shared-state diagnostic using "
            "expert 0 from the final IsoMamba checkpoint, not a trained Shared-state Decoder."
        ),
        "- Every dataset/resolution is analyzed independently; no pooled main result is reported.",
        "",
        "| Dataset | State mode | Correction Alignment ↑ | Group Separation ↑ | Successful images |",
        "|---|---|---:|---:|---:|",
    ]
    for key in ordered_keys:
        if key not in results:
            continue
        display = display_names[key]
        aggregate = results[key]["aggregate"]
        a = aggregate["correction_alignment"]
        s = aggregate["group_separation"]
        successful = (
            aggregate["status_counts"]["success"]
            if "status_counts" in aggregate
            else aggregate["image_count"]
        )
        lines.extend(
            [
                (
                    f"| {display} | Shared Proxy | {fmt(a['shared_proxy']['mean'])} ± "
                    f"{fmt(a['shared_proxy']['std'])} | {fmt(s['shared_proxy']['mean'])} ± "
                    f"{fmt(s['shared_proxy']['std'])} | {successful} |"
                ),
                (
                    f"| {display} | State Isolation | {fmt(a['state_isolation']['mean'])} ± "
                    f"{fmt(a['state_isolation']['std'])} | {fmt(s['state_isolation']['mean'])} ± "
                    f"{fmt(s['state_isolation']['std'])} | {successful} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "| Dataset | Alignment gap [95% CI] | Positive ratio | Separation gap [95% CI] | Positive ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ordered_keys:
        if key not in results:
            continue
        display = display_names[key]
        aggregate = results[key]["aggregate"]
        a = aggregate["correction_alignment"]["paired_gap_isolated_minus_shared"]
        s = aggregate["group_separation"]["paired_gap_isolated_minus_shared"]
        lines.append(
            f"| {display} | {fmt(a['gap_mean'])} "
            f"[{fmt(a['bootstrap_95_ci'][0])}, {fmt(a['bootstrap_95_ci'][1])}] | "
            f"{a['positive_count']}/{a['paired_n']} ({fmt(a['positive_ratio'] * 100, 2)}%) | "
            f"{fmt(s['gap_mean'])} "
            f"[{fmt(s['bootstrap_95_ci'][0])}, {fmt(s['bootstrap_95_ci'][1])}] | "
            f"{s['positive_count']}/{s['paired_n']} ({fmt(s['positive_ratio'] * 100, 2)}%) |"
        )
    lines.extend(["", "## Dataset-specific interpretation", ""])
    for key in ordered_keys:
        if key in results:
            aggregate = results[key]["aggregate"]
            successful = (
                aggregate["status_counts"]["success"]
                if "status_counts" in aggregate
                else aggregate["image_count"]
            )
            if successful == 0:
                lines.append(
                    f"- **{display_names[key]}**: no image completed both conditions; "
                    "no aggregate interpretation is made."
                )
                continue
            if key == "uieb_original" and successful < 10:
                lines.append(
                    f"- **{display_names[key]}**: only {successful} images completed both "
                    "conditions; descriptive results only."
                )
                continue
            lines.append(
                f"- **{display_names[key]}**: "
                + interpretation(results[key]["aggregate"], results[key]["tests"])
            )
    lines.append("")
    return "\n".join(lines)


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def analyze_one_dataset(
    *,
    dataset_key: str,
    model: torch.nn.Module,
    checkpoint: Path,
    pairs: Sequence[ImagePair],
    pairing_report: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    max_tokens: int,
    max_pairs: int,
    bootstrap_iterations: int,
    validation_length: int,
    logger: Logger,
) -> Dict[str, Any]:
    display = DATASETS[dataset_key]["display"]
    stage = model.decoder_level3
    target_rm = stage.attention[0]
    num_groups = int(stage.num_clusters)
    capture = TargetCapture(stage, 0)
    rows: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {}
    first_shapes: Dict[str, Any] = {}
    smoke_checks: Dict[str, Any] = {}
    first_shared_semantics: Dict[str, Any] = {}
    first_isolated_semantics: Dict[str, Any] = {}

    capture.register()
    try:
        for image_index, pair in enumerate(pairs):
            image_seed = stable_seed(seed, dataset_key, pair.stem)
            seed_everything(image_seed)
            input_cpu = load_rgb(pair.input_path)
            gt_cpu = load_rgb(pair.gt_path)
            if input_cpu.shape != gt_cpu.shape:
                raise RuntimeError(
                    f"{pair.stem}: Input/GT shape mismatch {tuple(input_cpu.shape)} vs "
                    f"{tuple(gt_cpu.shape)}"
                )
            if input_cpu.shape[-2] % 8 or input_cpu.shape[-1] % 8:
                raise RuntimeError(
                    f"{pair.stem}: image dimensions must be divisible by 8, got "
                    f"{tuple(input_cpu.shape[-2:])}"
                )
            input_tensor = input_cpu.to("cuda:0")
            gt_tensor = gt_cpu.to("cuda:0")
            capture.clear()
            with torch.inference_mode():
                model_output = model(input_tensor)

            if capture.block_input is None or capture.assignments is None:
                raise RuntimeError("Target hooks did not capture block input and routing assignments")
            target_input = capture.block_input.float()
            assignments = capture.assignments
            labels = assignments.argmax(dim=-1).long()
            expected_one_hot = F.one_hot(labels, num_groups).to(assignments.dtype)
            one_hot_error = float((assignments - expected_one_hot).abs().max().item())
            if one_hot_error != 0.0:
                raise RuntimeError(f"Routing assignment forward values are not hard one-hot: {one_hot_error}")

            shared_z, shared_tensors, shared_semantics = shared_proxy_response(
                target_rm, target_input
            )
            isolated_z, isolated_semantics = isolated_response(
                target_rm,
                target_input,
                labels,
                capture.actual_expert_inputs,
            )
            if shared_z.shape != isolated_z.shape:
                raise RuntimeError(
                    f"Response space mismatch: shared={tuple(shared_z.shape)}, "
                    f"isolated={tuple(isolated_z.shape)}"
                )
            target_h, target_w = labels.shape[1:3]
            correction = gt_tensor.float() - input_tensor.float()
            correction_small = F.interpolate(
                correction,
                size=(target_h, target_w),
                mode="area",
            )
            correction_flat = (
                correction_small.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            )
            labels_flat = labels.view(-1)
            shared_flat = shared_z.view(-1, shared_z.shape[-1])
            isolated_flat = isolated_z.view(-1, isolated_z.shape[-1])
            sample_rng = np.random.default_rng(stable_seed(image_seed, "tokens"))
            sampled = stratified_token_sample(
                labels_flat.cpu().numpy(),
                maximum=max_tokens,
                rng=sample_rng,
            )
            pair_rng = np.random.default_rng(stable_seed(image_seed, "pairs"))
            pair_first, pair_second = sample_unique_pairs(
                len(sampled),
                maximum=max_pairs,
                rng=pair_rng,
            )
            alignment_shared, shared_reason = correction_alignment(
                shared_flat,
                correction_flat,
                sampled,
                pair_first,
                pair_second,
                eps=1e-8,
            )
            alignment_isolated, isolated_reason = correction_alignment(
                isolated_flat,
                correction_flat,
                sampled,
                pair_first,
                pair_second,
                eps=1e-8,
            )
            separation_shared, shared_b, shared_w = group_separation(
                shared_flat, labels_flat, eps=1e-8
            )
            separation_isolated, isolated_b, isolated_w = group_separation(
                isolated_flat, labels_flat, eps=1e-8
            )
            counts = torch.bincount(labels_flat, minlength=num_groups).cpu().tolist()
            active_groups = sum(count > 0 for count in counts)

            if image_index == 0:
                first_shared_semantics = shared_semantics
                first_isolated_semantics = isolated_semantics
                validation = validate_state_projection(
                    shared_tensors,
                    validation_length=validation_length,
                )
                # Prove passive hooks are observational: compare with a hook-free forward.
                capture.remove()
                seed_everything(image_seed)
                with torch.inference_mode():
                    output_without_hooks = model(input_tensor)
                capture.register()
                hook_difference = (model_output - output_without_hooks).abs()
                validation["hook_output_equivalence"] = {
                    "max_abs": float(hook_difference.max().item()),
                    "mean_abs": float(hook_difference.mean().item()),
                    "allclose_exact": bool(torch.equal(model_output, output_without_hooks)),
                }
                if not validation["hook_output_equivalence"]["allclose_exact"]:
                    raise RuntimeError(f"Passive hook altered model output: {validation}")
                first_shapes = {
                    "input": list(input_tensor.shape),
                    "target_block_input": list(target_input.shape),
                    "routing_assignment": list(assignments.shape),
                    "routing_label": list(labels.shape),
                    "scan_x": list(shared_tensors.scan_x.shape),
                    "h_t_buffer": [
                        shared_tensors.shape[0],
                        shared_tensors.shape[1],
                        shared_tensors.shape[2],
                        shared_tensors.shape[3],
                    ],
                    "C_t": list(shared_tensors.C.shape),
                    "z_scan": shared_semantics["scan_shape"],
                    "z_feature_map_shared": list(shared_z.shape),
                    "z_feature_map_isolated": list(isolated_z.shape),
                    "correction_full": list(correction.shape),
                    "correction_aligned": list(correction_small.shape),
                    "group_token_counts": counts,
                    "isolated_packed_inputs": [
                        item.get("packed_input_shape")
                        for item in isolated_semantics["expert_details"]
                        if item["active"]
                    ],
                }
                smoke_checks = {
                    "checkpoint_loaded_strictly": True,
                    "pairing_by_stem": pairing_report["pairing_key"] == "Path.stem",
                    "logical_device_is_cuda_0": input_tensor.device == torch.device("cuda:0"),
                    "physical_gpu_binding_env_is_2": os.environ.get("CUDA_VISIBLE_DEVICES") == "2",
                    "reference_recurrence_pass": all(validation["checks"].values()),
                    "shared_uses_only_expert_0": shared_semantics["parameter_experts_used"] == [0],
                    "shared_one_state_per_direction": shared_semantics["states_per_direction"] == 1,
                    "isolated_uses_real_gather": isolated_semantics["cluster_gather_used"],
                    "isolated_uses_multiple_independent_experts": (
                        isolated_semantics["active_expert_count"] > 1
                    ),
                    "actual_gather_exact": isolated_semantics["gather_max_abs"] == 0.0,
                    "z_is_C_h_only": validation["representation"] == "z_t = C_t h_t only",
                    "z_shapes_match": shared_z.shape == isolated_z.shape,
                    "token_positions_match": True,
                    "correction_area_aligned": (
                        correction_small.shape[-2:] == shared_z.shape[1:3]
                    ),
                    "hooks_observational": validation["hook_output_equivalence"]["allclose_exact"],
                }
                if not all(smoke_checks.values()):
                    raise RuntimeError(f"First-image semantic checks failed: {smoke_checks}")
                logger.log(f"[{display}] tensor inspection: {json.dumps(first_shapes)}")
                logger.log(
                    f"[{display}] reference validation: fused max={validation['fused_output_max_abs']:.10g}, "
                    f"mean={validation['fused_output_mean_abs']:.10g}; "
                    f"C*h decomposition max={validation['decomposition_max_abs']:.10g}"
                )

            row = {
                "dataset": display,
                "image_stem": pair.stem,
                "input_file": pair.input_path.name,
                "gt_file": pair.gt_path.name,
                "input_path": str(pair.input_path),
                "gt_path": str(pair.gt_path),
                "checkpoint": str(checkpoint),
                "target_layer": "decoder_level3.attention.0",
                "feature_height": target_h,
                "feature_width": target_w,
                "z_channels": shared_z.shape[-1],
                "group_count": active_groups,
                "group_token_counts": json.dumps(counts),
                "sampled_token_count": len(sampled),
                "sampled_pair_count": len(pair_first),
                "alignment_shared": alignment_shared,
                "alignment_isolated": alignment_isolated,
                "alignment_gap": alignment_isolated - alignment_shared,
                "alignment_shared_status": shared_reason,
                "alignment_isolated_status": isolated_reason,
                "separation_shared": separation_shared,
                "separation_isolated": separation_isolated,
                "separation_gap": separation_isolated - separation_shared,
                "separation_shared_B": shared_b,
                "separation_shared_W": shared_w,
                "separation_isolated_B": isolated_b,
                "separation_isolated_W": isolated_w,
            }
            rows.append(row)
            logger.log(
                f"[{display} {image_index + 1}/{len(pairs)}] {pair.input_path.name}: "
                f"groups={counts}, sampled={len(sampled)}, pairs={len(pair_first)}, "
                f"alignment={fmt(alignment_shared)}/{fmt(alignment_isolated)}, "
                f"separation={fmt(separation_shared)}/{fmt(separation_isolated)}"
            )
            del model_output, shared_z, isolated_z, shared_tensors
    finally:
        capture.remove()

    aggregate, tests = aggregate_dataset(
        rows,
        bootstrap_iterations=bootstrap_iterations,
        seed=stable_seed(seed, dataset_key),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_image_metrics.csv", rows)
    write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_json(output_dir / "statistical_tests.json", tests)
    write_json(output_dir / "pairing_report.json", pairing_report)
    write_json(output_dir / "validation.json", validation)
    write_json(
        output_dir / "smoke_checks.json",
        {
            "checks": smoke_checks,
            "all_pass": all(smoke_checks.values()),
            "first_image_shapes": first_shapes,
            "shared_semantics": first_shared_semantics,
            "isolated_semantics": first_isolated_semantics,
        },
    )
    report = dataset_report_markdown(
        display,
        checkpoint,
        pairing_report,
        aggregate,
        tests,
        validation,
        first_shapes,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "aggregate": aggregate,
        "tests": tests,
        "pairing": pairing_report,
        "validation": validation,
        "first_shapes": first_shapes,
        "smoke_checks": smoke_checks,
    }


ORIGINAL_CSV_FIELDS = (
    "filename",
    "image_stem",
    "input_path",
    "gt_path",
    "height",
    "width",
    "num_pixels",
    "feature_height",
    "feature_width",
    "group_count",
    "group_token_counts",
    "sampled_token_count",
    "sampled_pair_count",
    "alignment_shared",
    "alignment_isolated",
    "alignment_gap",
    "alignment_shared_status",
    "alignment_isolated_status",
    "separation_shared",
    "separation_isolated",
    "separation_gap",
    "shared_peak_allocated_mb",
    "shared_peak_reserved_mb",
    "isolated_peak_allocated_mb",
    "isolated_peak_reserved_mb",
    "status",
    "failure_condition",
    "failure_reason",
)


def original_row_template(pair: ImagePair, height: int, width: int) -> Dict[str, Any]:
    return {
        "filename": pair.input_path.name,
        "image_stem": pair.stem,
        "input_path": str(pair.input_path),
        "gt_path": str(pair.gt_path),
        "height": int(height),
        "width": int(width),
        "num_pixels": int(height * width),
        "feature_height": "",
        "feature_width": "",
        "group_count": "",
        "group_token_counts": "",
        "sampled_token_count": 0,
        "sampled_pair_count": 0,
        "alignment_shared": math.nan,
        "alignment_isolated": math.nan,
        "alignment_gap": math.nan,
        "alignment_shared_status": "not_run",
        "alignment_isolated_status": "not_run",
        "separation_shared": math.nan,
        "separation_isolated": math.nan,
        "separation_gap": math.nan,
        "shared_peak_allocated_mb": math.nan,
        "shared_peak_reserved_mb": math.nan,
        "isolated_peak_allocated_mb": math.nan,
        "isolated_peak_reserved_mb": math.nan,
        "status": "failed",
        "failure_condition": "",
        "failure_reason": "",
    }


def image_dimensions(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def analyze_original_image(
    *,
    pair: ImagePair,
    model: torch.nn.Module,
    capture: TargetCapture,
    target_rm: torch.nn.Module,
    num_groups: int,
    seed: int,
    max_tokens: int,
    max_pairs: int,
    validation_length: int,
    run_validation: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Analyze one untouched original-resolution pair with strict OOM handling."""
    input_height, input_width = image_dimensions(pair.input_path)
    gt_height, gt_width = image_dimensions(pair.gt_path)
    row = original_row_template(pair, input_height, input_width)
    if (input_height, input_width) != (gt_height, gt_width):
        row.update(
            {
                "status": "shape_mismatch",
                "failure_condition": "input_gt_shape_check",
                "failure_reason": (
                    f"Input HxW={input_height}x{input_width}, "
                    f"GT HxW={gt_height}x{gt_width}"
                ),
            }
        )
        return row, {}, {}

    image_seed = stable_seed(seed, "uieb_original", pair.stem)
    seed_everything(image_seed)
    input_tensor = load_rgb(pair.input_path).to("cuda:0")
    gt_tensor = load_rgb(pair.gt_path).to("cuda:0")
    capture.clear()
    validation: Dict[str, Any] = {}
    first_shapes: Dict[str, Any] = {}

    # The checkpoint's normal State Isolation forward is also the observational
    # pass that supplies the real router labels and exact packed expert inputs.
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        with torch.inference_mode():
            model_output = model(input_tensor)
        torch.cuda.synchronize(0)
        isolated_capture_allocated, isolated_capture_reserved = cuda_peak_mb()
        del model_output
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        allocated, reserved = cuda_peak_mb()
        row.update(
            {
                "status": "isolated_oom",
                "failure_condition": "state_isolation_model_forward",
                "failure_reason": str(error),
                "isolated_peak_allocated_mb": allocated,
                "isolated_peak_reserved_mb": reserved,
            }
        )
        capture.clear()
        del input_tensor, gt_tensor
        cleanup_cuda()
        return row, validation, first_shapes

    if capture.block_input is None or capture.assignments is None:
        raise RuntimeError("Original-resolution hooks did not capture target tensors")
    target_input = capture.block_input.float()
    assignments = capture.assignments
    labels = assignments.argmax(dim=-1).long()
    expected_one_hot = F.one_hot(labels, num_groups).to(assignments.dtype)
    if float((assignments - expected_one_hot).abs().max().item()) != 0.0:
        raise RuntimeError("Original-resolution routing values are not hard one-hot")
    feature_height, feature_width = labels.shape[1:3]
    labels_flat = labels.view(-1)
    counts = torch.bincount(labels_flat, minlength=num_groups).cpu().tolist()
    row.update(
        {
            "feature_height": int(feature_height),
            "feature_width": int(feature_width),
            "group_count": sum(count > 0 for count in counts),
            "group_token_counts": json.dumps(counts),
        }
    )
    correction = gt_tensor.float() - input_tensor.float()
    correction_small = F.interpolate(
        correction,
        size=(feature_height, feature_width),
        mode="area",
    )
    correction_flat = (
        correction_small.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    )
    sample_rng = np.random.default_rng(stable_seed(image_seed, "tokens"))
    sampled = stratified_token_sample(
        labels_flat.cpu().numpy(), maximum=max_tokens, rng=sample_rng
    )
    pair_rng = np.random.default_rng(stable_seed(image_seed, "pairs"))
    pair_first, pair_second = sample_unique_pairs(
        len(sampled), maximum=max_pairs, rng=pair_rng
    )
    row["sampled_token_count"] = len(sampled)
    row["sampled_pair_count"] = len(pair_first)

    # Shared Proxy condition. Its recurrent state is newly zero-initialized by
    # state_projection_scan for this image and each of the four directions.
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        shared_z, shared_tensors, shared_semantics = shared_proxy_response(
            target_rm, target_input
        )
        shared_flat = shared_z.view(-1, shared_z.shape[-1])
        alignment_shared, alignment_shared_status = correction_alignment(
            shared_flat,
            correction_flat,
            sampled,
            pair_first,
            pair_second,
            eps=1e-8,
        )
        separation_shared, _, _ = group_separation_chunked(
            shared_flat, labels_flat, num_groups=num_groups, eps=1e-8
        )
        torch.cuda.synchronize(0)
        shared_peak_allocated, shared_peak_reserved = cuda_peak_mb()
        row.update(
            {
                "alignment_shared": alignment_shared,
                "alignment_shared_status": alignment_shared_status,
                "separation_shared": separation_shared,
                "shared_peak_allocated_mb": shared_peak_allocated,
                "shared_peak_reserved_mb": shared_peak_reserved,
            }
        )
        if run_validation:
            validation = validate_state_projection(
                shared_tensors, validation_length=validation_length
            )
            first_shapes = {
                "input": list(input_tensor.shape),
                "target_block_input": list(target_input.shape),
                "routing_assignment": list(assignments.shape),
                "routing_label": list(labels.shape),
                "scan_x": list(shared_tensors.scan_x.shape),
                "h_t_buffer": [
                    shared_tensors.shape[0],
                    shared_tensors.shape[1],
                    shared_tensors.shape[2],
                    shared_tensors.shape[3],
                ],
                "C_t": list(shared_tensors.C.shape),
                "z_feature_map_shared": list(shared_z.shape),
                "correction_aligned": list(correction_small.shape),
                "group_token_counts": counts,
                "shared_semantics": shared_semantics,
            }
        del shared_flat, shared_z, shared_tensors
        cleanup_cuda()
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        allocated, reserved = cuda_peak_mb()
        row.update(
            {
                "status": "shared_oom",
                "failure_condition": "shared_proxy",
                "failure_reason": str(error),
                "shared_peak_allocated_mb": allocated,
                "shared_peak_reserved_mb": reserved,
                "isolated_peak_allocated_mb": isolated_capture_allocated,
                "isolated_peak_reserved_mb": isolated_capture_reserved,
            }
        )
        capture.clear()
        del input_tensor, gt_tensor, target_input, assignments, labels
        del correction, correction_small, correction_flat
        cleanup_cuda()
        return row, validation, first_shapes

    # State Isolation response uses the exact packed tensors captured from the
    # checkpoint's normal gather path, then pushes C_t h_t back to this image.
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        isolated_z, isolated_semantics = isolated_response(
            target_rm,
            target_input,
            labels,
            capture.actual_expert_inputs,
        )
        isolated_flat = isolated_z.view(-1, isolated_z.shape[-1])
        alignment_isolated, alignment_isolated_status = correction_alignment(
            isolated_flat,
            correction_flat,
            sampled,
            pair_first,
            pair_second,
            eps=1e-8,
        )
        separation_isolated, _, _ = group_separation_chunked(
            isolated_flat, labels_flat, num_groups=num_groups, eps=1e-8
        )
        torch.cuda.synchronize(0)
        isolated_response_allocated, isolated_response_reserved = cuda_peak_mb()
        isolated_peak_allocated = max(
            isolated_capture_allocated, isolated_response_allocated
        )
        isolated_peak_reserved = max(
            isolated_capture_reserved, isolated_response_reserved
        )
        row.update(
            {
                "alignment_isolated": alignment_isolated,
                "alignment_isolated_status": alignment_isolated_status,
                "alignment_gap": alignment_isolated - float(row["alignment_shared"]),
                "separation_isolated": separation_isolated,
                "separation_gap": separation_isolated - float(row["separation_shared"]),
                "isolated_peak_allocated_mb": isolated_peak_allocated,
                "isolated_peak_reserved_mb": isolated_peak_reserved,
                "status": "success",
            }
        )
        if run_validation:
            first_shapes["z_feature_map_isolated"] = list(isolated_z.shape)
            first_shapes["isolated_semantics"] = isolated_semantics
        del isolated_flat, isolated_z
        cleanup_cuda()
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        allocated, reserved = cuda_peak_mb()
        row.update(
            {
                "status": "isolated_oom",
                "failure_condition": "state_isolation_response",
                "failure_reason": str(error),
                "isolated_peak_allocated_mb": max(
                    isolated_capture_allocated, allocated
                ),
                "isolated_peak_reserved_mb": max(
                    isolated_capture_reserved, reserved
                ),
            }
        )

    capture.clear()
    del input_tensor, gt_tensor, target_input, assignments, labels
    del correction, correction_small, correction_flat
    cleanup_cuda()
    return row, validation, first_shapes


def original_report_markdown(
    *,
    checkpoint: Path,
    pairing: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    tests: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    counts = aggregate["status_counts"]
    resolution = aggregate["successful_resolution"]
    lines = [
        "# UIEB Original Hidden-State Response Quality",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Target layer: `decoder_level3.attention.0`",
        "- Input images are used at their untouched original H×W: no resize, crop, padding, or tiling.",
        "- Batch size: 1; recurrent buffers are newly initialized for every image and direction.",
        (
            f"- Dataset totals: inputs={pairing['input_count']}, GT={pairing['gt_count']}, "
            f"stem-paired={pairing['paired_count']}, successful={counts['success']}."
        ),
        (
            f"- Status: shared OOM={counts['shared_oom']}, isolated OOM={counts['isolated_oom']}, "
            f"shape mismatch={counts['shape_mismatch']}, other failed={counts['failed']}."
        ),
        (
            "- Shared Proxy remains a post-hoc expert-0 diagnostic from the final IsoMamba "
            "checkpoint, not a trained Shared-state Decoder baseline."
        ),
        "",
    ]
    if counts["success"] == 0:
        lines.extend(
            [
                "No image completed both conditions, so no aggregate metric is reported.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            (
                "- Successful resolution range: "
                f"H={resolution['height_min']}–{resolution['height_max']}, "
                f"W={resolution['width_min']}–{resolution['width_max']}; "
                f"mean H×W={resolution['height_mean']:.2f}×{resolution['width_mean']:.2f}."
            ),
            "",
            "## Main results",
            "",
            "| State mode | Correction Alignment ↑ | Group Separation ↑ | Successful images |",
            "|---|---:|---:|---:|",
        ]
    )
    alignment = aggregate["correction_alignment"]
    separation = aggregate["group_separation"]
    lines.extend(
        [
            (
                f"| Shared Proxy | {fmt(alignment['shared_proxy']['mean'])} ± "
                f"{fmt(alignment['shared_proxy']['std'])} | "
                f"{fmt(separation['shared_proxy']['mean'])} ± "
                f"{fmt(separation['shared_proxy']['std'])} | {counts['success']} |"
            ),
            (
                f"| State Isolation | {fmt(alignment['state_isolation']['mean'])} ± "
                f"{fmt(alignment['state_isolation']['std'])} | "
                f"{fmt(separation['state_isolation']['mean'])} ± "
                f"{fmt(separation['state_isolation']['std'])} | {counts['success']} |"
            ),
            "",
            "## Paired gaps",
            "",
            "| Metric | Mean | Median | Bootstrap 95% CI | Positive ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, label in (
        ("correction_alignment", "Correction Alignment"),
        ("group_separation", "Group Separation"),
    ):
        gap = aggregate[metric]["paired_gap_isolated_minus_shared"]
        lines.append(
            f"| {label} | {fmt(gap['gap_mean'])} | {fmt(gap['gap_median'])} | "
            f"[{fmt(gap['bootstrap_95_ci'][0])}, {fmt(gap['bootstrap_95_ci'][1])}] | "
            f"{gap['positive_count']}/{gap['paired_n']} "
            f"({fmt(gap['positive_ratio'] * 100, 2)}%) |"
        )
    if counts["success"] >= 10:
        lines.extend(
            [
                "",
                "## Paired tests",
                "",
                "| Metric | Paired t-test p | Wilcoxon p | Cohen's dz |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric, label in (
            ("correction_alignment", "Correction Alignment"),
            ("group_separation", "Group Separation"),
        ):
            test = tests[metric]
            lines.append(
                f"| {label} | {fmt(test['paired_t_test']['p_value'])} | "
                f"{fmt(test['wilcoxon_signed_rank']['p_value'])} | "
                f"{fmt(test['cohens_dz'])} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                interpretation(aggregate, tests),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Fewer than 10 images succeeded; only descriptive and per-image results are emphasized.",
            ]
        )
    if validation:
        lines.extend(
            [
                "",
                "## Reference validation",
                "",
                (
                    f"- Fused/reference max abs={fmt(validation['fused_output_max_abs'], 10)}, "
                    f"mean abs={fmt(validation['fused_output_mean_abs'], 10)}."
                ),
                f"- `C_t h_t` decomposition max abs={fmt(validation['decomposition_max_abs'], 10)}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def analyze_original_dataset(
    *,
    model: torch.nn.Module,
    checkpoint: Path,
    pairs: Sequence[ImagePair],
    pairing_report: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    max_tokens: int,
    max_pairs: int,
    bootstrap_iterations: int,
    validation_length: int,
    continue_unsupported_shapes: bool,
    logger: Logger,
) -> Dict[str, Any]:
    stage = model.decoder_level3
    target_rm = stage.attention[0]
    num_groups = int(stage.num_clusters)
    capture = TargetCapture(stage, 0)
    rows: List[Dict[str, Any]] = []
    oom_rows: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {}
    first_shapes: Dict[str, Any] = {}

    unsupported: List[Tuple[ImagePair, int, int]] = []
    for pair in pairs:
        height, width = image_dimensions(pair.input_path)
        gt_height, gt_width = image_dimensions(pair.gt_path)
        if (height, width) == (gt_height, gt_width) and (height % 8 or width % 8):
            unsupported.append((pair, height, width))
    if unsupported and not continue_unsupported_shapes:
        examples = [
            f"{pair.input_path.name}:{height}x{width}"
            for pair, height, width in unsupported[:10]
        ]
        raise RuntimeError(
            f"{len(unsupported)} stem-paired original images have H/W not divisible by 8. "
            "The unchanged checkpoint uses three PixelUnshuffle(2) stages and cannot reach "
            "decoder_level3 without resize/crop/padding. This is a non-OOM architecture "
            f"shape error, so the strict protocol stops. Examples={examples}. "
            "Use --continue-unsupported-shapes-as-failed only with explicit authorization "
            "to record them as failed and continue."
        )

    capture.register()
    try:
        for index, pair in enumerate(pairs):
            height, width = image_dimensions(pair.input_path)
            gt_height, gt_width = image_dimensions(pair.gt_path)
            if (
                (height, width) == (gt_height, gt_width)
                and (height % 8 or width % 8)
                and continue_unsupported_shapes
            ):
                row = original_row_template(pair, height, width)
                row.update(
                    {
                        "status": "failed",
                        "failure_condition": "unchanged_model_resolution_constraint",
                        "failure_reason": (
                            "Original H/W is not divisible by 8; the unchanged model's "
                            "three PixelUnshuffle(2) stages cannot process it without a "
                            "forbidden resize/crop/pad."
                        ),
                    }
                )
                rows.append(row)
                logger.log(
                    f"[UIEB Original {index + 1}/{len(pairs)}] {pair.input_path.name}: "
                    f"failed architecture shape constraint at {height}x{width}"
                )
                continue
            try:
                row, image_validation, shapes = analyze_original_image(
                    pair=pair,
                    model=model,
                    capture=capture,
                    target_rm=target_rm,
                    num_groups=num_groups,
                    seed=seed,
                    max_tokens=max_tokens,
                    max_pairs=max_pairs,
                    validation_length=validation_length,
                    run_validation=not validation,
                )
            except BaseException:
                # Non-OOM errors are never relabeled or skipped.
                capture.clear()
                cleanup_cuda()
                raise
            rows.append(row)
            if image_validation and not validation:
                validation = image_validation
                first_shapes = shapes
            if row["status"] in {"shared_oom", "isolated_oom"}:
                oom_rows.append(
                    {
                        "filename": row["filename"],
                        "height": row["height"],
                        "width": row["width"],
                        "oom_condition": row["failure_condition"],
                        "peak_allocated_mb": (
                            row["shared_peak_allocated_mb"]
                            if row["status"] == "shared_oom"
                            else row["isolated_peak_allocated_mb"]
                        ),
                        "peak_reserved_mb": (
                            row["shared_peak_reserved_mb"]
                            if row["status"] == "shared_oom"
                            else row["isolated_peak_reserved_mb"]
                        ),
                        "error": row["failure_reason"],
                    }
                )
            logger.log(
                f"[UIEB Original {index + 1}/{len(pairs)}] {pair.input_path.name} "
                f"{height}x{width}: status={row['status']}, "
                f"shared_peak={fmt(row['shared_peak_allocated_mb'], 1)}/"
                f"{fmt(row['shared_peak_reserved_mb'], 1)} MB, "
                f"isolated_peak={fmt(row['isolated_peak_allocated_mb'], 1)}/"
                f"{fmt(row['isolated_peak_reserved_mb'], 1)} MB"
            )
            write_csv_with_fields(
                output_dir / "per_image_metrics.csv", rows, ORIGINAL_CSV_FIELDS
            )
            write_csv_with_fields(
                output_dir / "oom_images.csv",
                oom_rows,
                (
                    "filename",
                    "height",
                    "width",
                    "oom_condition",
                    "peak_allocated_mb",
                    "peak_reserved_mb",
                    "error",
                ),
            )
    finally:
        capture.remove()
        capture.clear()
        cleanup_cuda()

    successful = [row for row in rows if row["status"] == "success"]
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("success", "shared_oom", "isolated_oom", "shape_mismatch", "failed")
    }
    if successful:
        metric_aggregate, tests = aggregate_dataset(
            successful,
            bootstrap_iterations=bootstrap_iterations,
            seed=stable_seed(seed, "uieb_original"),
        )
        heights = [int(row["height"]) for row in successful]
        widths = [int(row["width"]) for row in successful]
        resolution = {
            "height_min": min(heights),
            "height_max": max(heights),
            "height_mean": statistics.fmean(heights),
            "width_min": min(widths),
            "width_max": max(widths),
            "width_mean": statistics.fmean(widths),
            "num_pixels_min": min(h * w for h, w in zip(heights, widths)),
            "num_pixels_max": max(h * w for h, w in zip(heights, widths)),
            "num_pixels_mean": statistics.fmean(
                h * w for h, w in zip(heights, widths)
            ),
        }
    else:
        metric_aggregate = {
            "image_count": 0,
            "correction_alignment": {
                "shared_proxy": mean_std([]),
                "state_isolation": mean_std([]),
                "paired_gap_isolated_minus_shared": paired_statistics(
                    [], [], bootstrap_iterations, seed
                )[0],
            },
            "group_separation": {
                "shared_proxy": mean_std([]),
                "state_isolation": mean_std([]),
                "paired_gap_isolated_minus_shared": paired_statistics(
                    [], [], bootstrap_iterations, seed
                )[0],
            },
        }
        tests = {
            "correction_alignment": {"paired_n": 0},
            "group_separation": {"paired_n": 0},
            "inferential_tests_reported": False,
        }
        resolution = {}
    aggregate = {
        **metric_aggregate,
        "dataset_total_input_images": pairing_report["input_count"],
        "stem_paired_count": pairing_report["paired_count"],
        "completed_pair_count": len(rows),
        "status_counts": status_counts,
        "successful_resolution": resolution,
        "original_resolution_policy": (
            "no resize, crop, padding, fixed-size conversion, or tiled inference"
        ),
    }
    if len(successful) < 10:
        tests["inferential_tests_reported"] = False
        tests["note"] = (
            "Fewer than 10 paired-success images; descriptive results only."
        )
    else:
        tests["inferential_tests_reported"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_with_fields(
        output_dir / "per_image_metrics.csv", rows, ORIGINAL_CSV_FIELDS
    )
    write_csv_with_fields(
        output_dir / "oom_images.csv",
        oom_rows,
        (
            "filename",
            "height",
            "width",
            "oom_condition",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "error",
        ),
    )
    write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_json(output_dir / "statistical_tests.json", tests)
    write_json(output_dir / "pairing_report.json", pairing_report)
    write_json(output_dir / "validation.json", validation)
    write_json(output_dir / "first_success_shapes.json", first_shapes)
    report = original_report_markdown(
        checkpoint=checkpoint,
        pairing=pairing_report,
        aggregate=aggregate,
        tests=tests,
        validation=validation,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "aggregate": aggregate,
        "tests": tests,
        "pairing": pairing_report,
        "validation": validation,
        "first_shapes": first_shapes,
    }


def run_self_test() -> None:
    spatial = torch.tensor(
        [[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]],
        dtype=torch.float32,
    )
    # Construct scan-order values and prove inverse_scan_sum aligns all four
    # directions to the same row-major grid.
    row = spatial.view(1, 6, 1).transpose(1, 2)
    column = spatial.permute(0, 3, 2, 1).contiguous().view(1, 1, 6)
    z_scan = torch.stack((row, column, row.flip(-1), column.flip(-1)), dim=1)
    restored = inverse_scan_sum(z_scan, 2, 3)
    torch.testing.assert_close(restored, 4.0 * spatial)

    labels = np.asarray([0] * 90 + [1] * 10)
    sampled = stratified_token_sample(labels, 20, np.random.default_rng(7))
    sampled_counts = np.bincount(labels[sampled], minlength=2)
    if not np.array_equal(sampled_counts, np.asarray([10, 10])):
        raise AssertionError(f"Stratified sample failed: {sampled_counts}")
    a, b = sample_unique_pairs(512, 20000, np.random.default_rng(7))
    if len(a) != 20000 or len(set(zip(a.tolist(), b.tolist()))) != 20000:
        raise AssertionError("Pair sample is not 20,000 unique unordered pairs")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", choices=("all", "uieb", "lsui"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images", type=int, default=0, help="0 analyzes every paired image")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--validation-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        print("Hidden-State Response Quality self-test: PASS")
        return 0
    apply_dataset_arguments(args, DATASETS)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expose exactly one GPU with CUDA_VISIBLE_DEVICES; got {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    if args.checkpoint is None:
        checkpoint, checkpoint_details = discover_checkpoint(args.checkpoint_root)
    else:
        checkpoint = args.checkpoint.resolve()
        checkpoint_details = {
            "search_root": str(args.checkpoint_root.resolve()),
            "selected": str(checkpoint),
            "selection_reason": "explicit --checkpoint override",
        }
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.max_tokens < 2 or args.max_pairs < 3:
        raise ValueError("max-tokens must be >=2 and max-pairs must be >=3")
    if args.bootstrap_iterations != 10000:
        print(
            f"WARNING: bootstrap iterations={args.bootstrap_iterations}; requested final protocol is 10000"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.output_root / "run.log")
    try:
        command = runtime_command(Path(__file__).resolve())
        logger.log(f"Exact checkpoint: {checkpoint}")
        logger.log(f"Checkpoint selection: {checkpoint_details['selection_reason']}")
        logger.log(f"Command: {command}")
        logger.log(
            f"GPU binding: visible={os.environ.get('CUDA_VISIBLE_DEVICES')}, logical={device}, name={torch.cuda.get_device_name(0)}, "
            f"visible_count={torch.cuda.device_count()}"
        )
        selected_datasets = ("uieb", "lsui") if args.dataset == "all" else (args.dataset,)
        pairs_by_dataset: Dict[str, List[ImagePair]] = {}
        pairing_by_dataset: Dict[str, Dict[str, Any]] = {}
        for key in selected_datasets:
            config = DATASETS[key]
            for path in (config["input"], config["gt"]):
                if not path.is_dir():
                    raise FileNotFoundError(path)
            pairs, pairing = pair_by_stem(config["input"], config["gt"])
            pairs_by_dataset[key] = pairs[: args.max_images] if args.max_images > 0 else pairs
            pairing_by_dataset[key] = pairing
            logger.log(
                f"[{config['display']}] inputs={pairing['input_count']}, GT={pairing['gt_count']}, "
                f"paired={pairing['paired_count']}, analyzing={len(pairs_by_dataset[key])}, "
                f"input_only={pairing['unpaired_input_files']}, gt_only={pairing['unpaired_gt_files']}"
            )

        seed_everything(args.seed)
        model = load_model(args.config.resolve(), checkpoint, device)
        target_rm = model.decoder_level3.attention[0]
        logger.log(
            "Target structure: decoder_level3.attention.0, "
            f"experts={len(target_rm.mamba_blocks)}, "
            f"d_model={target_rm.mamba_blocks[0].d_model}, "
            f"d_inner={target_rm.mamba_blocks[0].d_inner}, "
            f"d_state={target_rm.mamba_blocks[0].d_state}, directions=4"
        )
        results: Dict[str, Dict[str, Any]] = {}
        for key in selected_datasets:
            results[key] = analyze_one_dataset(
                dataset_key=key,
                model=model,
                checkpoint=checkpoint,
                pairs=pairs_by_dataset[key],
                pairing_report=pairing_by_dataset[key],
                output_dir=args.output_root / key,
                seed=args.seed,
                max_tokens=args.max_tokens,
                max_pairs=args.max_pairs,
                bootstrap_iterations=args.bootstrap_iterations,
                validation_length=args.validation_length,
                logger=logger,
            )
        cross_report = cross_dataset_report(checkpoint, results)
        (args.output_root / "cross_dataset_report.md").write_text(
            cross_report, encoding="utf-8"
        )
        manifest = {
            "checkpoint": str(checkpoint),
            "checkpoint_discovery": checkpoint_details,
            "config": str(args.config.resolve()),
            "target_layer": "decoder_level3.attention.0",
            "gpu": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": "cuda:0",
                "device_name": torch.cuda.get_device_name(0),
                "visible_device_count": torch.cuda.device_count(),
            },
            "command": command,
            "seed": args.seed,
            "maximum_tokens_per_image": args.max_tokens,
            "maximum_unique_pairs_per_image": args.max_pairs,
            "bootstrap_iterations": args.bootstrap_iterations,
            "datasets": {
                key: {
                    "pairing": pairing_by_dataset[key],
                    "analyzed_count": len(pairs_by_dataset[key]),
                }
                for key in selected_datasets
            },
            "representation": "z_t = C_t h_t only, four directions inverse-scanned and summed",
            "shared_proxy_limitation": (
                "post-hoc single-expert shared-state diagnostic using expert 0 from the "
                "final IsoMamba checkpoint; not a trained Shared-state Decoder baseline"
            ),
        }
        write_json(args.output_root / "run_manifest.json", manifest)
        logger.log("Completed all requested datasets.")
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

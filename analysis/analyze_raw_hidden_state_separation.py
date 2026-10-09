#!/usr/bin/env python3
"""Strict raw recurrent hidden-state separation diagnostic.

This analysis compares three post-hoc recurrence modes at
``decoder_level3.attention.0`` under one checkpoint, one captured layer input,
and one set of hard Delta-routing labels:

1. Shared State, Fixed Expert: expert-0 recurrence operands and one continuous
   state per scan direction.
2. Isolated State, Fixed Expert: the *same tokenwise expert-0 operands* as (1),
   but tokens are logically packed by routing group and each group/direction
   starts from zero.
3. Full State Isolation: the same logical group packing, using the
   corresponding expert's tokenwise operands.

Only the raw FP32 recurrent state

    h_t = exp(softplus(delta_t + bias) A) h_{t-1}
          + softplus(delta_t + bias) B_t x_t

is recorded.  C*h, D*x, gates, residuals, FFNs, FiLM, normalization, output
projection, and all other readouts are excluded.

The fixed-expert control intentionally packs already-computed tokenwise
recurrence operands instead of re-running the expert-0 convolution on a
rearranged square.  Re-running spatial preprocessing after packing would
change x_t, delta_t, and B_t in addition to the predecessor/state graph and
would therefore not isolate the state-organization effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/isomamba_raw_hidden_state_mpl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze_hidden_state_quality import (  # noqa: E402
    DATASETS,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_CONFIG,
    ImagePair,
    Logger,
    discover_checkpoint,
    fmt,
    load_rgb,
    mean_std,
    pair_by_stem,
    paired_statistics,
    seed_everything,
    stable_seed,
    write_csv,
    write_json,
)
from analysis.analyze_state_contention import (  # noqa: E402
    ScanTensors,
    TargetCapture,
    load_model,
    prepare_scan_tensors,
    validate_reference_recurrence,
)
from analysis.runtime_options import (  # noqa: E402
    add_dataset_arguments, apply_dataset_arguments, runtime_command,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis_outputs" / "raw_hidden_state_separation"
TARGET_LAYER = "decoder_level3.attention.0"
DIRECTIONS = 4
MODE_SHARED = "shared_fixed"
MODE_ISOLATED = "isolated_fixed"
MODE_FULL = "full_isolation"
MODES = (MODE_SHARED, MODE_ISOLATED, MODE_FULL)
MODE_LABELS = {
    MODE_SHARED: "Shared State, Fixed Expert",
    MODE_ISOLATED: "Isolated State, Fixed Expert",
    MODE_FULL: "Full State Isolation",
}


def fmt_p_value(value: float) -> str:
    """Keep very small p-values visible instead of rounding them to zero."""
    return f"{value:.3e}" if math.isfinite(value) else "NaN"


@dataclass
class RuntimeEvidence:
    """Validation evidence accumulated across every analyzed image."""

    image_count: int = 0
    maximum_trajectory_final_error: float = 0.0
    maximum_reference_state_error: float = 0.0
    minimum_write_count: int = 1
    maximum_write_count: int = 1
    padding_slots_excluded: int = 0
    tokens_scored: int = 0
    nonfinite_score_count: int = 0
    position_permutation_failures: int = 0
    routing_mutation_failures: int = 0
    input_mutation_failures: int = 0
    expert0_mutation_failures: int = 0
    expert0_operand_failures: int = 0
    padding_processed: int = 0
    stats_count_failures: int = 0
    routing_membership_failures: int = 0


def tensor_digest(tensor: torch.Tensor) -> str:
    """Stable digest used only for validation, never as a model input."""
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def module_parameter_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def scan_to_spatial_indices(
    height: int,
    width: int,
    direction: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Map a direction's scan-time index to row-major spatial token index."""
    if direction not in range(DIRECTIONS):
        raise ValueError(f"direction must be in [0,3], got {direction}")
    row = torch.arange(height * width, device=device, dtype=torch.long).view(height, width)
    base = row.flatten() if direction % 2 == 0 else row.transpose(0, 1).contiguous().flatten()
    return base if direction < 2 else base.flip(0)


def spatial_to_scan_indices(
    height: int,
    width: int,
    direction: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    scan_to_spatial = scan_to_spatial_indices(height, width, direction, device=device)
    inverse = torch.empty_like(scan_to_spatial)
    inverse[scan_to_spatial] = torch.arange(height * width, device=device)
    return inverse


def packed_group_spatial_order(
    labels_flat: torch.Tensor,
    group: int,
    direction: int,
) -> Tuple[torch.Tensor, int]:
    """Return real-token order for a square-packed group, excluding padding.

    The ordering matches SS2D's row, column, reverse-row, and reverse-column
    scans over the logical square.  Dummy square slots are removed *before*
    recurrence so padding cannot alter any real token's raw state.
    """
    if labels_flat.device.type != "cpu":
        raise ValueError("packed_group_spatial_order expects CPU labels")
    positions = (labels_flat == group).nonzero(as_tuple=False).squeeze(-1)
    count = int(positions.numel())
    if count == 0:
        return positions, 0
    side = int(math.ceil(math.sqrt(count)))
    slots = torch.full((side * side,), -1, dtype=torch.long)
    slots[:count] = positions
    square = slots.view(side, side)
    order = square.flatten() if direction % 2 == 0 else square.transpose(0, 1).contiguous().flatten()
    if direction >= 2:
        order = order.flip(0)
    real = order[order >= 0]
    if int(real.numel()) != count or int(torch.unique(real).numel()) != count:
        raise RuntimeError(
            f"Packed order failed for group={group}, direction={direction}, count={count}"
        )
    return real, side * side - count


def validate_captured_group_pack(
    target_input: torch.Tensor,
    labels: torch.Tensor,
    captured_expert_inputs: Mapping[int, List[torch.Tensor]],
    num_groups: int,
) -> Dict[str, Any]:
    """Validate the model's real routing gather and zero-padding operation."""
    if target_input.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Captured pack validation supports batch size one")
    _, height, width, channels = target_input.shape
    flat_input = target_input.view(height * width, channels)
    flat_labels = labels.view(-1)
    gather_max_abs = 0.0
    padding_max_abs = 0.0
    expected_padding = 0
    active_groups: List[int] = []
    for group in range(num_groups):
        positions = (flat_labels == group).nonzero(as_tuple=False).squeeze(-1)
        count = int(positions.numel())
        calls = captured_expert_inputs.get(group, [])
        if count == 0:
            if calls:
                raise RuntimeError(f"Empty routing group {group} received an expert call")
            continue
        if len(calls) != 1:
            raise RuntimeError(
                f"Routing group {group}: expected one packed expert call, got {len(calls)}"
            )
        packed = calls[0].float()
        side = int(math.ceil(math.sqrt(count)))
        if tuple(packed.shape) != (1, side, side, channels):
            raise RuntimeError(
                f"Routing group {group}: packed shape {tuple(packed.shape)} "
                f"!= {(1, side, side, channels)}"
            )
        packed_flat = packed.view(side * side, channels)
        expected = flat_input.index_select(0, positions).float()
        gather_max_abs = max(
            gather_max_abs,
            float((packed_flat[:count] - expected).abs().max().item()),
        )
        padding = side * side - count
        expected_padding += padding
        if padding:
            padding_max_abs = max(
                padding_max_abs,
                float(packed_flat[count:].abs().max().item()),
            )
        active_groups.append(group)
    result = {
        "active_groups": active_groups,
        "gather_max_abs": gather_max_abs,
        "padding_max_abs": padding_max_abs,
        "expected_padding_per_isolated_direction": expected_padding,
        "gather_exact": gather_max_abs == 0.0,
        "padding_zero": padding_max_abs == 0.0,
    }
    if not result["gather_exact"] or not result["padding_zero"]:
        raise RuntimeError(f"Captured group pack validation failed: {result}")
    return result


@dataclass
class DirectionOperands:
    """One direction's token sequence, already ordered for recurrence."""

    x: torch.Tensor  # [L, inner]
    delta_raw: torch.Tensor  # [L, inner]
    variable_B: torch.Tensor  # [L, state]
    A: torch.Tensor  # [inner, state]
    delta_bias: torch.Tensor  # [inner]


def select_direction_operands(
    tensors: ScanTensors,
    direction: int,
    spatial_positions: torch.Tensor,
    height: int,
    width: int,
) -> DirectionOperands:
    """Select tokenwise operands by spatial position without recomputing them."""
    device = tensors.scan_x.device
    positions = spatial_positions.to(device=device, dtype=torch.long)
    inverse = spatial_to_scan_indices(height, width, direction, device=device)
    scan_indices = inverse.index_select(0, positions)
    return DirectionOperands(
        x=tensors.scan_x[0, direction].index_select(-1, scan_indices).transpose(0, 1).contiguous(),
        delta_raw=(
            tensors.delta_raw[0, direction]
            .index_select(-1, scan_indices)
            .transpose(0, 1)
            .contiguous()
        ),
        variable_B=(
            tensors.B[0, direction]
            .index_select(-1, scan_indices)
            .transpose(0, 1)
            .contiguous()
        ),
        A=tensors.A[direction],
        delta_bias=tensors.delta_bias[direction],
    )


@torch.inference_mode()
def raw_hidden_trajectory(
    operands: DirectionOperands,
    *,
    cpu_chunk_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Run FP32 recurrence and offload raw per-step h_t chunks to CPU."""
    length, inner = operands.x.shape
    state = operands.A.shape[-1]
    if length == 0:
        raise ValueError("Cannot run an empty recurrence")
    hidden = torch.zeros(
        (inner, state),
        device=operands.x.device,
        dtype=torch.float32,
    )
    trajectory = torch.empty(
        (length, inner, state),
        device="cpu",
        dtype=torch.float32,
    )
    gpu_steps: List[torch.Tensor] = []
    cpu_write_start = 0
    for token in range(length):
        delta = F.softplus(operands.delta_raw[token] + operands.delta_bias)
        transition = torch.exp(delta.unsqueeze(-1) * operands.A)
        drive = (
            delta.unsqueeze(-1)
            * operands.variable_B[token].unsqueeze(0)
            * operands.x[token].unsqueeze(-1)
        )
        hidden = transition * hidden + drive
        gpu_steps.append(hidden.clone())
        if len(gpu_steps) == cpu_chunk_steps or token + 1 == length:
            cpu_chunk = torch.stack(gpu_steps, dim=0).cpu()
            cpu_write_stop = cpu_write_start + len(gpu_steps)
            trajectory[cpu_write_start:cpu_write_stop].copy_(cpu_chunk)
            cpu_write_start = cpu_write_stop
            gpu_steps.clear()
    final_cpu = hidden.cpu()
    if cpu_write_start != length:
        raise RuntimeError(
            f"Trajectory CPU offload wrote {cpu_write_start} steps, expected {length}"
        )
    if trajectory.shape != (length, inner, state):
        raise RuntimeError(
            f"Raw trajectory shape mismatch: {tuple(trajectory.shape)} "
            f"!= {(length, inner, state)}"
        )
    final_error = float((trajectory[-1] - final_cpu).abs().max().item())
    if final_error > 1e-6:
        raise RuntimeError(f"Trajectory final state mismatch: max_abs={final_error}")
    return trajectory, final_cpu, final_error


@torch.inference_mode()
def raw_reference_final(
    tensors: ScanTensors,
    direction: int,
) -> torch.Tensor:
    """Independent FP32 recurrence final state for full-length shared validation."""
    _, _, inner, state, length = tensors.shape
    hidden = torch.zeros(
        (inner, state),
        device=tensors.scan_x.device,
        dtype=torch.float32,
    )
    for token in range(length):
        delta = F.softplus(
            tensors.delta_raw[0, direction, :, token] + tensors.delta_bias[direction]
        )
        hidden = (
            torch.exp(delta.unsqueeze(-1) * tensors.A[direction]) * hidden
            + delta.unsqueeze(-1)
            * tensors.B[0, direction, :, token].unsqueeze(0)
            * tensors.scan_x[0, direction, :, token].unsqueeze(-1)
        )
    return hidden


@torch.inference_mode()
def hidden_state_separation_two_pass(
    trajectory_spatial_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    num_groups: int,
    *,
    device: torch.device,
    chunk_tokens: int,
    eps: float,
) -> Tuple[float, float, float, List[int]]:
    """Strict normalized separation with CPU storage and two GPU passes."""
    if trajectory_spatial_cpu.device.type != "cpu" or labels_cpu.device.type != "cpu":
        raise ValueError("Trajectory and labels must be CPU-offloaded")
    total = int(trajectory_spatial_cpu.shape[0])
    if labels_cpu.numel() != total:
        raise ValueError("Trajectory/label token counts differ")
    feature_dim = int(trajectory_spatial_cpu.shape[1] * trajectory_spatial_cpu.shape[2])
    sums = torch.zeros((num_groups, feature_dim), device=device, dtype=torch.float32)
    global_sum = torch.zeros((feature_dim,), device=device, dtype=torch.float32)
    counts = torch.zeros((num_groups,), device=device, dtype=torch.long)

    # Pass 1: normalized group and global sums.
    for start in range(0, total, chunk_tokens):
        stop = min(start + chunk_tokens, total)
        raw = trajectory_spatial_cpu[start:stop].to(device, non_blocking=False).view(stop - start, -1)
        normalized = raw / (torch.linalg.vector_norm(raw, dim=1, keepdim=True) + eps)
        chunk_labels = labels_cpu[start:stop].to(device)
        global_sum += normalized.sum(dim=0)
        for group in range(num_groups):
            mask = chunk_labels == group
            count = int(mask.sum().item())
            if count:
                sums[group] += normalized[mask].sum(dim=0)
                counts[group] += count

    if int(counts.sum().item()) != total:
        raise RuntimeError(f"Pass-1 count mismatch: {int(counts.sum())} != {total}")
    active = counts > 0
    centers = torch.zeros_like(sums)
    centers[active] = sums[active] / counts[active].to(torch.float32).unsqueeze(1)
    global_center = global_sum / float(total)
    between = (
        counts[active].to(torch.float32)
        * (centers[active] - global_center).square().sum(dim=1)
    ).sum()

    # Pass 2: strict within-group dispersion.
    within = torch.zeros((), device=device, dtype=torch.float32)
    for start in range(0, total, chunk_tokens):
        stop = min(start + chunk_tokens, total)
        raw = trajectory_spatial_cpu[start:stop].to(device, non_blocking=False).view(stop - start, -1)
        normalized = raw / (torch.linalg.vector_norm(raw, dim=1, keepdim=True) + eps)
        chunk_labels = labels_cpu[start:stop].to(device)
        within += (
            normalized - centers.index_select(0, chunk_labels)
        ).square().sum()

    score = between / (between + within + eps)
    values = (float(score.item()), float(between.item()), float(within.item()))
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"Non-finite raw hidden-state separation: {values}")
    return values[0], values[1], values[2], counts.cpu().tolist()


def build_mode_direction_trajectory(
    *,
    mode: str,
    direction: int,
    labels_cpu: torch.Tensor,
    num_groups: int,
    height: int,
    width: int,
    expert_tensors: Sequence[ScanTensors],
    cpu_chunk_steps: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Produce one spatially aligned [N,inner,state] raw trajectory."""
    total = height * width
    inner = expert_tensors[0].A.shape[1]
    state = expert_tensors[0].A.shape[2]
    output = torch.empty((total, inner, state), dtype=torch.float32, device="cpu")
    write_counts = torch.zeros((total,), dtype=torch.int16)
    source_position_by_output = torch.full((total,), -1, dtype=torch.long)
    padding_excluded = 0
    padding_processed = 0
    maximum_final_error = 0.0
    recurrence_count = 0
    recurrence_real_tokens = 0
    routing_membership_exact = True
    parameter_experts_used: List[int] = []
    order_digest = hashlib.sha256()
    operand_source = (
        "expert0_original_grid"
        if mode in (MODE_SHARED, MODE_ISOLATED)
        else "routed_expert_original_grid"
    )

    if mode == MODE_SHARED:
        order = scan_to_spatial_indices(height, width, direction)
        order_digest.update(order.numpy().tobytes())
        operands = select_direction_operands(
            expert_tensors[0], direction, order, height, width
        )
        trajectory, _, error = raw_hidden_trajectory(
            operands, cpu_chunk_steps=cpu_chunk_steps
        )
        output.index_copy_(0, order, trajectory)
        source_position_by_output.index_copy_(0, order, order)
        write_counts.index_add_(
            0, order, torch.ones_like(order, dtype=write_counts.dtype)
        )
        maximum_final_error = max(maximum_final_error, error)
        recurrence_count = 1
        recurrence_real_tokens = total
        parameter_experts_used = [0]
    else:
        for group in range(num_groups):
            order, excluded = packed_group_spatial_order(labels_cpu, group, direction)
            order_digest.update(group.to_bytes(4, "little", signed=False))
            order_digest.update(order.numpy().tobytes())
            padding_excluded += excluded
            if order.numel() == 0:
                continue
            routing_membership_exact = routing_membership_exact and bool(
                torch.all(labels_cpu.index_select(0, order) == group).item()
            )
            expert_index = 0 if mode == MODE_ISOLATED else group
            operands = select_direction_operands(
                expert_tensors[expert_index],
                direction,
                order,
                height,
                width,
            )
            trajectory, _, error = raw_hidden_trajectory(
                operands, cpu_chunk_steps=cpu_chunk_steps
            )
            output.index_copy_(0, order, trajectory)
            source_position_by_output.index_copy_(0, order, order)
            write_counts.index_add_(
                0, order, torch.ones_like(order, dtype=write_counts.dtype)
            )
            maximum_final_error = max(maximum_final_error, error)
            recurrence_count += 1
            recurrence_real_tokens += int(order.numel())
            parameter_experts_used.append(expert_index)

    minimum_write = int(write_counts.min().item())
    maximum_write = int(write_counts.max().item())
    if minimum_write != 1 or maximum_write != 1:
        raise RuntimeError(
            f"{mode} direction {direction}: each token must be written once, "
            f"observed [{minimum_write}, {maximum_write}]"
        )
    position_mapping_exact = torch.equal(
        source_position_by_output, torch.arange(total, dtype=torch.long)
    )
    if not position_mapping_exact:
        raise RuntimeError(f"{mode} direction {direction}: push-back mapping is not exact")
    return output, {
        "recurrence_count": recurrence_count,
        "recurrence_real_tokens": recurrence_real_tokens,
        "padding_slots_excluded": padding_excluded,
        "padding_slots_processed": padding_processed,
        "tokens_written": int(write_counts.sum().item()),
        "minimum_write_count": minimum_write,
        "maximum_write_count": maximum_write,
        "trajectory_final_max_abs": maximum_final_error,
        "position_mapping_exact": position_mapping_exact,
        "routing_membership_exact": routing_membership_exact,
        "parameter_experts_used": sorted(set(parameter_experts_used)),
        "operand_source": operand_source,
        "group_order_digest": order_digest.hexdigest(),
    }


def analyze_image(
    *,
    dataset_key: str,
    pair: ImagePair,
    model: torch.nn.Module,
    capture: TargetCapture,
    num_groups: int,
    device: torch.device,
    cpu_chunk_steps: int,
    stats_chunk_tokens: int,
    eps: float,
    reference_validation_length: int,
    validate_reference: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run all three modes for one image and return row plus validation evidence."""
    input_cpu = load_rgb(pair.input_path)
    if input_cpu.shape[-2] % 8 or input_cpu.shape[-1] % 8:
        raise RuntimeError(f"Input dimensions must be divisible by 8: {tuple(input_cpu.shape)}")
    input_tensor = input_cpu.to(device)
    input_digest_before = tensor_digest(input_tensor)
    capture.clear()
    with torch.inference_mode():
        model_output = model(input_tensor)
    del model_output
    if capture.block_input is None or capture.assignments is None:
        raise RuntimeError("Hooks did not capture target input/routing")

    target_input = capture.block_input.float()
    assignments = capture.assignments
    labels = assignments.argmax(dim=-1).long()
    expected = F.one_hot(labels, num_groups).to(assignments.dtype)
    if not torch.equal(assignments, expected):
        raise RuntimeError("Routing assignments are not exact hard one-hot values")
    labels_cpu = labels.view(-1).cpu()
    captured_pack = validate_captured_group_pack(
        target_input,
        labels,
        capture.actual_expert_inputs,
        num_groups,
    )
    routing_digest_before = tensor_digest(labels_cpu)
    target_input_digest_before = tensor_digest(target_input)
    _, height, width, _ = target_input.shape
    total = height * width
    if labels_cpu.numel() != total:
        raise RuntimeError("Routing label count does not match target tokens")

    target_rm = model.decoder_level3.attention[0]
    expert0_digest_before = module_parameter_digest(target_rm.mamba_blocks[0])
    expert_tensors = [
        prepare_scan_tensors(expert, target_input)
        for expert in target_rm.mamba_blocks
    ]
    expert0_tensors = expert_tensors[0]
    reference_validation: Dict[str, Any] = {}
    maximum_reference_state_error = 0.0
    if validate_reference:
        reference_validation = validate_reference_recurrence(
            expert0_tensors, max_length=reference_validation_length
        )

    mode_results: Dict[str, Dict[str, Any]] = {
        mode: {"directions": []} for mode in MODES
    }
    mode_runtime: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for direction in range(DIRECTIONS):
            trajectory, runtime = build_mode_direction_trajectory(
                mode=mode,
                direction=direction,
                labels_cpu=labels_cpu,
                num_groups=num_groups,
                height=height,
                width=width,
                expert_tensors=expert_tensors,
                cpu_chunk_steps=cpu_chunk_steps,
            )
            if mode == MODE_SHARED:
                reference_final = raw_reference_final(expert0_tensors, direction).cpu()
                scan_last_position = int(
                    scan_to_spatial_indices(height, width, direction)[-1].item()
                )
                reference_error = float(
                    (trajectory[scan_last_position] - reference_final).abs().max().item()
                )
                maximum_reference_state_error = max(
                    maximum_reference_state_error, reference_error
                )
                if reference_error > 1e-6:
                    raise RuntimeError(
                        f"Shared direction {direction} trajectory/reference final mismatch: "
                        f"{reference_error}"
                    )
            score, between, within, metric_counts = hidden_state_separation_two_pass(
                trajectory,
                labels_cpu,
                num_groups,
                device=device,
                chunk_tokens=stats_chunk_tokens,
                eps=eps,
            )
            expected_counts = torch.bincount(
                labels_cpu, minlength=num_groups
            ).tolist()
            runtime["stats_token_counts"] = metric_counts
            runtime["stats_counts_exact"] = metric_counts == expected_counts
            if metric_counts != expected_counts:
                raise RuntimeError(
                    f"Stats included wrong tokens: {metric_counts} != {expected_counts}"
                )
            mode_results[mode]["directions"].append(
                {
                    "direction": direction,
                    "score": score,
                    "between": between,
                    "within": within,
                }
            )
            mode_runtime[mode].append(runtime)
            del trajectory
        mode_results[mode]["mean"] = statistics.fmean(
            item["score"] for item in mode_results[mode]["directions"]
        )

    input_digest_after = tensor_digest(input_tensor)
    target_input_digest_after = tensor_digest(target_input)
    routing_digest_after = tensor_digest(labels_cpu)
    expert0_digest_after = module_parameter_digest(target_rm.mamba_blocks[0])
    all_scores = [
        item["score"]
        for mode in MODES
        for item in mode_results[mode]["directions"]
    ]
    runtime_items = [
        item for mode in MODES for item in mode_runtime[mode]
    ]
    isolated_order_matches_full = all(
        mode_runtime[MODE_ISOLATED][direction]["recurrence_real_tokens"]
        == mode_runtime[MODE_FULL][direction]["recurrence_real_tokens"]
        == total
        and mode_runtime[MODE_ISOLATED][direction]["routing_membership_exact"]
        and mode_runtime[MODE_FULL][direction]["routing_membership_exact"]
        and mode_runtime[MODE_ISOLATED][direction]["group_order_digest"]
        == mode_runtime[MODE_FULL][direction]["group_order_digest"]
        for direction in range(DIRECTIONS)
    )
    expected_padding = captured_pack["expected_padding_per_isolated_direction"]
    padding_exclusion_exact = (
        all(
            item["padding_slots_excluded"] == 0
            for item in mode_runtime[MODE_SHARED]
        )
        and all(
            item["padding_slots_excluded"] == expected_padding
            for mode in (MODE_ISOLATED, MODE_FULL)
            for item in mode_runtime[mode]
        )
    )
    expert0_operand_semantics_exact = all(
        item["operand_source"] == "expert0_original_grid"
        and item["parameter_experts_used"] == [0]
        for mode in (MODE_SHARED, MODE_ISOLATED)
        for item in mode_runtime[mode]
    )
    all_scan_permutations = all(
        torch.equal(
            torch.sort(scan_to_spatial_indices(height, width, direction)).values,
            torch.arange(total),
        )
        for direction in range(DIRECTIONS)
    )
    all_checks = {
        "same_input": (
            input_digest_before == input_digest_after
            and target_input_digest_before == target_input_digest_after
        ),
        "same_routing": (
            routing_digest_before == routing_digest_after
            and isolated_order_matches_full
            and captured_pack["gather_exact"]
        ),
        "expert0_shared_equal": (
            expert0_digest_before == expert0_digest_after
            and expert0_operand_semantics_exact
        ),
        "no_padding_in_stats": all(
            item["tokens_written"] == total
            and item["recurrence_real_tokens"] == total
            and item["padding_slots_processed"] == 0
            and item["stats_counts_exact"]
            for item in runtime_items
        )
        and padding_exclusion_exact,
        "one_state_per_token": all(
            item["minimum_write_count"] == 1
            and item["maximum_write_count"] == 1
            for item in runtime_items
        ),
        "position_alignment": (
            all_scan_permutations
            and all(item["position_mapping_exact"] for item in runtime_items)
            and captured_pack["gather_exact"]
        ),
        "trajectory_consistency": (
            max(item["trajectory_final_max_abs"] for item in runtime_items) <= 1e-6
            and maximum_reference_state_error <= 1e-6
            and (
                not reference_validation
                or all(reference_validation["checks"].values())
            )
        ),
        "finite_scores": all(math.isfinite(value) for value in all_scores),
    }
    if not all(all_checks.values()):
        raise RuntimeError(f"Raw hidden-state validation failed: {all_checks}")

    group_counts = torch.bincount(labels_cpu, minlength=num_groups).tolist()
    row: Dict[str, Any] = {
        "dataset": DATASETS[dataset_key]["display"],
        "image_stem": pair.stem,
        "input_file": pair.input_path.name,
        "input_path": str(pair.input_path),
        "target_layer": TARGET_LAYER,
        "feature_height": height,
        "feature_width": width,
        "num_tokens": total,
        "active_groups": sum(value > 0 for value in group_counts),
        "group_token_counts": json.dumps(group_counts),
    }
    for mode in MODES:
        row[f"{mode}_mean"] = mode_results[mode]["mean"]
        for item in mode_results[mode]["directions"]:
            direction = item["direction"]
            row[f"{mode}_dir{direction}"] = item["score"]
            row[f"{mode}_dir{direction}_between"] = item["between"]
            row[f"{mode}_dir{direction}_within"] = item["within"]
    row["isolated_minus_shared"] = (
        mode_results[MODE_ISOLATED]["mean"] - mode_results[MODE_SHARED]["mean"]
    )
    row["full_minus_isolated"] = (
        mode_results[MODE_FULL]["mean"] - mode_results[MODE_ISOLATED]["mean"]
    )
    validation = {
        "checks": all_checks,
        "reference_validation": reference_validation,
        "maximum_reference_state_error": maximum_reference_state_error,
        "maximum_trajectory_final_error": max(
            item["trajectory_final_max_abs"] for item in runtime_items
        ),
        "minimum_write_count": min(
            item["minimum_write_count"] for item in runtime_items
        ),
        "maximum_write_count": max(
            item["maximum_write_count"] for item in runtime_items
        ),
        "padding_slots_excluded": sum(
            item["padding_slots_excluded"] for item in runtime_items
        ),
        "tokens_scored": sum(item["tokens_written"] for item in runtime_items),
        "input_digest": input_digest_before,
        "target_input_digest": target_input_digest_before,
        "routing_digest": routing_digest_before,
        "expert0_parameter_digest": expert0_digest_before,
        "captured_group_pack": captured_pack,
        "mode_runtime": mode_runtime,
    }
    del input_tensor, target_input, assignments, labels, expert_tensors
    return row, validation


def update_runtime_evidence(
    evidence: RuntimeEvidence,
    validation: Mapping[str, Any],
) -> None:
    checks = validation["checks"]
    evidence.image_count += 1
    evidence.maximum_trajectory_final_error = max(
        evidence.maximum_trajectory_final_error,
        float(validation["maximum_trajectory_final_error"]),
    )
    evidence.maximum_reference_state_error = max(
        evidence.maximum_reference_state_error,
        float(validation["maximum_reference_state_error"]),
    )
    evidence.minimum_write_count = min(
        evidence.minimum_write_count, int(validation["minimum_write_count"])
    )
    evidence.maximum_write_count = max(
        evidence.maximum_write_count, int(validation["maximum_write_count"])
    )
    evidence.padding_slots_excluded += int(validation["padding_slots_excluded"])
    evidence.tokens_scored += int(validation["tokens_scored"])
    evidence.nonfinite_score_count += 0 if checks["finite_scores"] else 1
    evidence.position_permutation_failures += 0 if checks["position_alignment"] else 1
    evidence.routing_mutation_failures += 0 if checks["same_routing"] else 1
    evidence.input_mutation_failures += 0 if checks["same_input"] else 1
    evidence.expert0_mutation_failures += 0 if checks["expert0_shared_equal"] else 1
    runtime_items = [
        item
        for mode_items in validation["mode_runtime"].values()
        for item in mode_items
    ]
    evidence.expert0_operand_failures += 0 if all(
        item["operand_source"] == "expert0_original_grid"
        and item["parameter_experts_used"] == [0]
        for mode in (MODE_SHARED, MODE_ISOLATED)
        for item in validation["mode_runtime"][mode]
    ) else 1
    evidence.padding_processed += sum(
        int(item["padding_slots_processed"]) for item in runtime_items
    )
    evidence.stats_count_failures += sum(
        0 if item["stats_counts_exact"] else 1 for item in runtime_items
    )
    evidence.routing_membership_failures += sum(
        0 if item["routing_membership_exact"] else 1
        for mode in (MODE_ISOLATED, MODE_FULL)
        for item in validation["mode_runtime"][mode]
    )


def summarize_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    values = {
        mode: [float(row[f"{mode}_mean"]) for row in rows]
        for mode in MODES
    }
    gap, tests = paired_statistics(
        values[MODE_SHARED],
        values[MODE_ISOLATED],
        bootstrap_iterations,
        stable_seed(seed, "raw_hidden_state_bootstrap"),
    )
    summary = {
        "image_count": len(rows),
        "modes": {
            mode: {
                "label": MODE_LABELS[mode],
                **mean_std(values[mode]),
                "directions": {
                    f"direction_{direction}": mean_std(
                        float(row[f"{mode}_dir{direction}"]) for row in rows
                    )
                    for direction in range(DIRECTIONS)
                },
            }
            for mode in MODES
        },
        "paired_difference_isolated_fixed_minus_shared_fixed": gap,
        "descriptive_difference_full_minus_isolated": mean_std(
            float(row["full_minus_isolated"]) for row in rows
        ),
    }
    statistical_tests = {
        "comparison": "Isolated State, Fixed Expert - Shared State, Fixed Expert",
        "alternative": "two-sided",
        **tests,
    }
    return summary, statistical_tests


def validation_report(
    evidence: RuntimeEvidence,
    first_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "same_input": evidence.input_mutation_failures == 0,
        "same_routing": (
            evidence.routing_mutation_failures == 0
            and evidence.routing_membership_failures == 0
        ),
        "expert0_shared_equal": (
            evidence.expert0_mutation_failures == 0
            and evidence.expert0_operand_failures == 0
        ),
        "no_padding_in_stats": (
            evidence.tokens_scored > 0
            and evidence.padding_processed == 0
            and evidence.stats_count_failures == 0
        ),
        "one_state_per_token": (
            evidence.minimum_write_count == 1 and evidence.maximum_write_count == 1
        ),
        "position_alignment": evidence.position_permutation_failures == 0,
        "trajectory_consistency": (
            evidence.maximum_trajectory_final_error <= 1e-6
            and evidence.maximum_reference_state_error <= 1e-6
        ),
        "finite_scores": evidence.nonfinite_score_count == 0,
    }
    report: Dict[str, Any] = {
        **checks,
        "all_pass": all(checks.values()),
        "images_validated": evidence.image_count,
        "evidence": {
            "maximum_trajectory_final_error": evidence.maximum_trajectory_final_error,
            "maximum_reference_state_error": evidence.maximum_reference_state_error,
            "minimum_write_count": evidence.minimum_write_count,
            "maximum_write_count": evidence.maximum_write_count,
            "padding_slots_excluded_before_recurrence_and_stats": (
                evidence.padding_slots_excluded
            ),
            "real_token_state_writes_checked": evidence.tokens_scored,
            "nonfinite_score_count": evidence.nonfinite_score_count,
            "position_permutation_failures": evidence.position_permutation_failures,
            "routing_mutation_failures": evidence.routing_mutation_failures,
            "input_mutation_failures": evidence.input_mutation_failures,
            "expert0_mutation_failures": evidence.expert0_mutation_failures,
            "expert0_operand_failures": evidence.expert0_operand_failures,
            "padding_slots_processed": evidence.padding_processed,
            "stats_count_failures": evidence.stats_count_failures,
            "routing_membership_failures": evidence.routing_membership_failures,
            "first_image_captured_group_pack": first_validation.get(
                "captured_group_pack", {}
            ),
            "first_image_reference_validation": first_validation.get(
                "reference_validation", {}
            ),
        },
    }
    if not report["all_pass"]:
        raise RuntimeError(f"Aggregate validation failed: {report}")
    return report


def latex_table(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Dataset & Shared State & Isolated State & Full Isolation \\",
        r"\midrule",
    ]
    for key in ("uieb", "lsui"):
        if key not in summaries:
            continue
        item = summaries[key]["modes"]
        lines.append(
            f"{DATASETS[key]['display']} & "
            f"{item[MODE_SHARED]['mean']:.6f} $\\pm$ {item[MODE_SHARED]['std']:.6f} & "
            f"{item[MODE_ISOLATED]['mean']:.6f} $\\pm$ {item[MODE_ISOLATED]['std']:.6f} & "
            f"{item[MODE_FULL]['mean']:.6f} $\\pm$ {item[MODE_FULL]['std']:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", "", r"\par\smallskip"])
    gap_parts: List[str] = []
    for key in ("uieb", "lsui"):
        if key not in summaries:
            continue
        gap = summaries[key]["paired_difference_isolated_fixed_minus_shared_fixed"]
        ci = gap["bootstrap_95_ci"]
        gap_parts.append(
            f"{DATASETS[key]['display']}: {gap['gap_mean']:.6f} "
            f"[{ci[0]:.6f}, {ci[1]:.6f}]"
        )
    lines.extend(
        [
            r"\noindent Isolated State $-$ Shared State (95\% bootstrap CI): "
            + "; ".join(gap_parts)
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def dataset_interpretation(
    dataset_key: str,
    summary: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> str:
    gap = summary["paired_difference_isolated_fixed_minus_shared_fixed"]
    ci = gap["bootstrap_95_ci"]
    t_p = tests["paired_t_test"]["p_value"]
    w_p = tests["wilcoxon_signed_rank"]["p_value"]
    mean_gap = gap["gap_mean"]
    supported_positive = (
        mean_gap > 0
        and ci[0] > 0
        and math.isfinite(t_p)
        and t_p < 0.05
        and math.isfinite(w_p)
        and w_p < 0.05
    )
    supported_negative = (
        mean_gap < 0
        and ci[1] < 0
        and math.isfinite(t_p)
        and t_p < 0.05
        and math.isfinite(w_p)
        and w_p < 0.05
    )
    if supported_positive:
        state_effect = (
            "The fixed-expert comparison supports a positive state-isolation effect on "
            "raw h_t separation."
        )
    elif supported_negative:
        state_effect = (
            "The fixed-expert comparison supports lower raw h_t separation under state "
            "isolation; the proposed positive state-isolation effect is not supported."
        )
    else:
        state_effect = (
            "The fixed-expert comparison does not establish a directional state-isolation "
            "effect on raw h_t separation under the joint CI/t-test/Wilcoxon criterion."
        )
    full_gap = summary["descriptive_difference_full_minus_isolated"]
    expert_text = (
        "Full-minus-fixed-isolated is reported only as an expert-specialization reference "
        f"(mean {fmt(full_gap['mean'])} ± {fmt(full_gap['std'])}); it is not part of the "
        "state-effect contrast."
    )
    return (
        f"- **{DATASETS[dataset_key]['display']}**: {state_effect} "
        f"Isolated-minus-shared mean={fmt(mean_gap)}, bootstrap 95% CI="
        f"[{fmt(ci[0])}, {fmt(ci[1])}], paired t p={fmt_p_value(t_p)}, "
        f"Wilcoxon p={fmt_p_value(w_p)}. {expert_text}"
    )


def paper_summary_markdown(
    checkpoint: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    tests: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    lines = [
        "# Raw Hidden-State Separation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Target layer: `{TARGET_LAYER}`",
        "- Representation: raw selective-scan recurrent state `h_t` only.",
        (
            "- Excluded: `C_t h_t`, SSM output/readout, `D_t x_t`, gate, residual, "
            "FFN, FiLM, normalization, output projection, and direction merging."
        ),
        (
            "- Core causal contrast: Isolated State, Fixed Expert minus Shared State, "
            "Fixed Expert. Both reuse identical expert-0 tokenwise `x_t`, `Delta_t`, "
            "`B_t`, `A`, and delta-bias operands; only the recurrence predecessor/state "
            "organization changes."
        ),
        (
            "- Full State Isolation is a descriptive expert-specialization reference and "
            "is not used to identify the state-isolation effect."
        ),
        (
            f"- Validation: `{validation['all_pass']}` across "
            f"{validation['images_validated']} analyzed images."
        ),
        "",
        "## Main results",
        "",
        "| Dataset | Shared State | Isolated State | Full Isolation | Isolated - Shared (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("uieb", "lsui"):
        if key not in summaries:
            continue
        item = summaries[key]
        modes = item["modes"]
        gap = item["paired_difference_isolated_fixed_minus_shared_fixed"]
        lines.append(
            f"| {DATASETS[key]['display']} | "
            f"{fmt(modes[MODE_SHARED]['mean'])} ± {fmt(modes[MODE_SHARED]['std'])} | "
            f"{fmt(modes[MODE_ISOLATED]['mean'])} ± {fmt(modes[MODE_ISOLATED]['std'])} | "
            f"{fmt(modes[MODE_FULL]['mean'])} ± {fmt(modes[MODE_FULL]['std'])} | "
            f"{fmt(gap['gap_mean'])} "
            f"[{fmt(gap['bootstrap_95_ci'][0])}, {fmt(gap['bootstrap_95_ci'][1])}] |"
        )
    lines.extend(["", "## Statistical tests", ""])
    lines.extend(
        [
            "| Dataset | Paired t p | Wilcoxon p | Cohen's dz | n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("uieb", "lsui"):
        if key not in tests:
            continue
        item = tests[key]
        lines.append(
            f"| {DATASETS[key]['display']} | "
            f"{fmt_p_value(item['paired_t_test']['p_value'])} | "
            f"{fmt_p_value(item['wilcoxon_signed_rank']['p_value'])} | "
            f"{fmt(item['cohens_dz'])} | {item['paired_n']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for key in ("uieb", "lsui"):
        if key in summaries:
            lines.append(dataset_interpretation(key, summaries[key], tests[key]))
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            (
                "This is a post-hoc diagnostic of one trained checkpoint, not a separately "
                "trained shared-state architecture. The routing labels define the groups used "
                "by the score, so the result measures conformity of raw h_t to that predefined "
                "routing structure. It does not establish that the routing groups are uniquely "
                "correct or semantically optimal."
            ),
            (
                "Logical packing excludes dummy padding before recurrence. This prevents "
                "padding transitions and rearranged-grid convolutional neighborhoods from "
                "confounding the fixed-expert state-only comparison."
            ),
            "",
            "## Protocol",
            "",
            "```json",
            json.dumps(config["protocol"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    height, width = 2, 3
    expected = (
        [0, 1, 2, 3, 4, 5],
        [0, 3, 1, 4, 2, 5],
        [5, 4, 3, 2, 1, 0],
        [5, 2, 4, 1, 3, 0],
    )
    for direction in range(DIRECTIONS):
        actual = scan_to_spatial_indices(height, width, direction).tolist()
        if actual != expected[direction]:
            raise AssertionError((direction, actual, expected[direction]))

    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0])
    for group in (0, 1):
        original = set((labels == group).nonzero(as_tuple=False).squeeze(-1).tolist())
        for direction in range(DIRECTIONS):
            order, _ = packed_group_spatial_order(labels, group, direction)
            if set(order.tolist()) != original:
                raise AssertionError((group, direction, order, original))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    length, inner, state = 7, 3, 2
    operands = DirectionOperands(
        x=torch.randn(length, inner, device=device),
        delta_raw=torch.randn(length, inner, device=device),
        variable_B=torch.randn(length, state, device=device),
        A=-torch.exp(torch.randn(inner, state, device=device)),
        delta_bias=torch.randn(inner, device=device),
    )
    trajectory, final, error = raw_hidden_trajectory(operands, cpu_chunk_steps=3)
    if trajectory.shape != (length, inner, state) or error > 1e-6:
        raise AssertionError("Raw trajectory self-test failed")
    torch.testing.assert_close(trajectory[-1], final, rtol=0.0, atol=1e-6)
    score, _, _, counts = hidden_state_separation_two_pass(
        trajectory,
        labels,
        2,
        device=device,
        chunk_tokens=3,
        eps=1e-8,
    )
    if not math.isfinite(score) or counts != [4, 3]:
        raise AssertionError((score, counts))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", choices=("all", "uieb", "lsui"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--cpu-chunk-steps", type=int, default=128)
    parser.add_argument("--stats-chunk-tokens", type=int, default=128)
    parser.add_argument("--reference-validation-length", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        print("Raw Hidden-State Separation self-test: PASS")
        return 0
    apply_dataset_arguments(args, DATASETS)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expose exactly one GPU with CUDA_VISIBLE_DEVICES; got {torch.cuda.device_count()}"
        )
    if args.cpu_chunk_steps < 1 or args.stats_chunk_tokens < 1:
        raise ValueError("Chunk sizes must be positive")
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap-iterations must be positive")
    device = torch.device("cuda:0")
    checkpoint, checkpoint_details = (
        discover_checkpoint(args.checkpoint_root)
        if args.checkpoint is None
        else (
            args.checkpoint.resolve(),
            {
                "selected": str(args.checkpoint.resolve()),
                "selection_reason": "explicit --checkpoint override",
            },
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.output_root / "run.log")
    command = runtime_command(Path(__file__).resolve())
    selected = ("uieb", "lsui") if args.dataset == "all" else (args.dataset,)
    protocol = {
        "representation": "raw h_t before all C/D/readout/output transformations",
        "recurrence": (
            "h_t = exp(softplus(delta_raw_t + delta_bias) * A) * h_(t-1) "
            "+ softplus(delta_raw_t + delta_bias) * B_t * x_t"
        ),
        "directions": 4,
        "direction_merge": "none; score each direction then arithmetic mean",
        "shared_fixed": (
            "expert 0 tokenwise operands; one zero-initialized continuous recurrence "
            "per direction"
        ),
        "isolated_fixed": (
            "identical expert 0 tokenwise operands; logical square-pack order per "
            "routing group/direction; independent zero-initialized recurrence"
        ),
        "full_isolation": (
            "corresponding expert tokenwise operands evaluated on the same original-grid "
            "target input; same logical group/direction packing and independent recurrence"
        ),
        "padding": (
            "logical square padding slots are excluded before recurrence and statistics"
        ),
        "normalization": "vec(h)/(L2(vec(h))+eps)",
        "separation": "between/(between+within+eps), computed in two passes",
        "storage": (
            "per-direction raw trajectories recorded in bounded GPU chunks and "
            "offloaded to CPU; modes/directions processed serially"
        ),
        "oom_policy": "fail immediately; no sample, resolution, token, or mode reduction",
        "core_contrast": "isolated_fixed - shared_fixed",
        "expert_reference": (
            "full_isolation is descriptive only and does not identify the state effect"
        ),
        "exclusions": [
            "C_t h_t",
            "SSM output",
            "D_t x_t",
            "gate",
            "residual",
            "FFN",
            "FiLM",
            "normalization",
            "output projection",
        ],
    }
    config: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_discovery": checkpoint_details,
        "model_config": str(args.config.resolve()),
        "target_layer": TARGET_LAYER,
        "datasets": list(selected),
        "dataset_paths": {
            key: {
                "input": str(DATASETS[key]["input"]),
                "gt": str(DATASETS[key]["gt"]),
            }
            for key in selected
        },
        "maximum_images": args.max_images,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "eps": args.eps,
        "cpu_chunk_steps": args.cpu_chunk_steps,
        "stats_chunk_tokens": args.stats_chunk_tokens,
        "reference_validation_length": args.reference_validation_length,
        "command": command,
        "gpu": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "visible_device_count": torch.cuda.device_count(),
        },
        "protocol": protocol,
    }
    write_json(args.output_root / "experiment_config.json", config)

    logger.log(f"Command: {command}")
    logger.log(f"Checkpoint: {checkpoint}")
    logger.log(f"Target layer: {TARGET_LAYER}")
    logger.log(
        "Raw h_t only; no C*h/readout/output terms. "
        "OOM reduction/fallback is disabled."
    )
    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    tests: Dict[str, Dict[str, Any]] = {}
    pairing_reports: Dict[str, Any] = {}
    evidence = RuntimeEvidence()
    first_validation: Optional[Dict[str, Any]] = None
    try:
        pairs_by_dataset: Dict[str, List[ImagePair]] = {}
        for key in selected:
            dataset_config = DATASETS[key]
            for path in (dataset_config["input"], dataset_config["gt"]):
                if not path.is_dir():
                    raise FileNotFoundError(path)
            pairs, pairing = pair_by_stem(
                dataset_config["input"], dataset_config["gt"]
            )
            if args.max_images > 0:
                pairs = pairs[: args.max_images]
            pairs_by_dataset[key] = pairs
            pairing_reports[key] = pairing
            logger.log(
                f"[{DATASETS[key]['display']}] paired={pairing['paired_count']}, "
                f"analyzing={len(pairs)}"
            )

        seed_everything(args.seed)
        model = load_model(args.config.resolve(), checkpoint, device)
        target_rm = model.decoder_level3.attention[0]
        num_groups = len(target_rm.mamba_blocks)
        logger.log(
            f"experts={num_groups}, d_inner={target_rm.mamba_blocks[0].d_inner}, "
            f"d_state={target_rm.mamba_blocks[0].d_state}, directions={DIRECTIONS}"
        )
        capture = TargetCapture(model.decoder_level3, 0)
        capture.register()
        try:
            for key in selected:
                dataset_rows: List[Dict[str, Any]] = []
                for image_index, pair in enumerate(pairs_by_dataset[key]):
                    seed_everything(stable_seed(args.seed, key, pair.stem))
                    row, image_validation = analyze_image(
                        dataset_key=key,
                        pair=pair,
                        model=model,
                        capture=capture,
                        num_groups=num_groups,
                        device=device,
                        cpu_chunk_steps=args.cpu_chunk_steps,
                        stats_chunk_tokens=args.stats_chunk_tokens,
                        eps=args.eps,
                        reference_validation_length=args.reference_validation_length,
                        validate_reference=first_validation is None,
                    )
                    if first_validation is None:
                        first_validation = image_validation
                        logger.log(
                            "First-image smoke validation: "
                            + json.dumps(image_validation["checks"], sort_keys=True)
                        )
                    update_runtime_evidence(evidence, image_validation)
                    dataset_rows.append(row)
                    all_rows.append(row)
                    logger.log(
                        f"[{DATASETS[key]['display']} {image_index + 1}/"
                        f"{len(pairs_by_dataset[key])}] {pair.input_path.name}: "
                        f"shared={fmt(row[f'{MODE_SHARED}_mean'])}, "
                        f"isolated={fmt(row[f'{MODE_ISOLATED}_mean'])}, "
                        f"full={fmt(row[f'{MODE_FULL}_mean'])}, "
                        f"gap={fmt(row['isolated_minus_shared'])}"
                    )
                    # Persist completed work without changing the fixed sample.
                    write_csv(
                        args.output_root / "per_image_hidden_state_separation.csv",
                        all_rows,
                    )
                summary, dataset_tests = summarize_dataset(
                    dataset_rows,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=stable_seed(args.seed, key),
                )
                summaries[key] = summary
                tests[key] = dataset_tests
        finally:
            capture.remove()

        if first_validation is None:
            raise RuntimeError("No image was analyzed")
        validation = validation_report(evidence, first_validation)
        dataset_summary = {
            "datasets": summaries,
            "pairing": pairing_reports,
            "score_definition": (
                "mean over four direction-specific "
                "D_between/(D_between+D_within+eps) scores"
            ),
        }
        write_json(args.output_root / "dataset_summary.json", dataset_summary)
        write_json(args.output_root / "statistical_tests.json", {"datasets": tests})
        write_json(args.output_root / "validation_report.json", validation)
        (args.output_root / "paper_table.tex").write_text(
            latex_table(summaries), encoding="utf-8"
        )
        (args.output_root / "paper_summary.md").write_text(
            paper_summary_markdown(
                checkpoint, summaries, tests, validation, config
            ),
            encoding="utf-8",
        )
        logger.log(
            f"Completed {len(all_rows)} images; aggregate validation all_pass=true."
        )
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

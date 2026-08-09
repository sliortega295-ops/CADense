"""Temporal mesh primitives used by CoFrame.

The functions in this module intentionally do not depend on Wan.  They are
small, deterministic and covered by CPU unit tests so that controller bugs can
be separated from model-integration bugs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

import torch

T = TypeVar("T")


def _validate_budget(num_frames: int, budget: int) -> None:
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    if not 2 <= budget <= num_frames:
        raise ValueError(
            f"budget must satisfy 2 <= budget <= num_frames, got {budget}/{num_frames}"
        )


def _sorted_unique(indices: Iterable[int], num_frames: int) -> list[int]:
    result = sorted({int(index) for index in indices})
    if any(index < 0 or index >= num_frames for index in result):
        raise ValueError(f"frame indices out of range for F={num_frames}: {result}")
    return result


def uniform_indices(num_frames: int, budget: int) -> list[int]:
    """Return exactly ``budget`` approximately uniform indices with endpoints.

    Rounding a linspace can create duplicate indices for short sequences.  We
    fill any missing slots by repeatedly choosing the frame farthest from the
    current set, making the result deterministic for every valid ``F`` and
    ``budget``.
    """

    _validate_budget(num_frames, budget)
    if budget == num_frames:
        return list(range(num_frames))

    raw = torch.linspace(0, num_frames - 1, budget, dtype=torch.float64)
    selected = {int(round(value.item())) for value in raw}
    selected.update((0, num_frames - 1))

    while len(selected) < budget:
        candidate = max(
            (index for index in range(num_frames) if index not in selected),
            key=lambda index: (min(abs(index - chosen) for chosen in selected), -index),
        )
        selected.add(candidate)

    if len(selected) > budget:
        # Endpoints are mandatory. Remove the most redundant interior points.
        while len(selected) > budget:
            interior = [index for index in selected if index not in (0, num_frames - 1)]
            remove = min(
                interior,
                key=lambda index: (
                    min(abs(index - other) for other in selected if other != index),
                    index,
                ),
            )
            selected.remove(remove)

    return sorted(selected)


def frame_neighbors(core_indices: Sequence[int], frame: int) -> tuple[int, int]:
    """Return the enclosing core anchors for an interior frame."""

    core = sorted(int(index) for index in core_indices)
    left = [index for index in core if index < frame]
    right = [index for index in core if index > frame]
    if not left or not right:
        raise ValueError(
            f"frame {frame} is not enclosed by core anchors {core}; endpoints are required"
        )
    return left[-1], right[0]


def interpolate_frame_values(
    anchor_values: torch.Tensor,
    anchor_indices: Sequence[int],
    num_frames: int,
    *,
    frame_dim: int = 1,
) -> torch.Tensor:
    """Piecewise-linearly interpolate arbitrary frame-aligned tensors.

    Args:
        anchor_values: Tensor containing one value per anchor along ``frame_dim``.
        anchor_indices: Sorted or unsorted original frame positions.
        num_frames: Number of frames in the reconstructed sequence.
        frame_dim: Dimension corresponding to anchors.  The default expects
            ``[B, K, ...]`` and also works for ``[1, K, P, C]`` token tensors.

    The first and last frame must be anchors; extrapolation is deliberately not
    hidden because it makes sparse-video errors difficult to interpret.
    """

    indices = _sorted_unique(anchor_indices, num_frames)
    if len(indices) != anchor_values.shape[frame_dim]:
        raise ValueError(
            "anchor count does not match tensor: "
            f"{len(indices)} vs shape[{frame_dim}]={anchor_values.shape[frame_dim]}"
        )
    if indices[0] != 0 or indices[-1] != num_frames - 1:
        raise ValueError(f"interpolation requires endpoint anchors, got {indices}")

    values = anchor_values.movedim(frame_dim, 1)
    output_shape = list(values.shape)
    output_shape[1] = num_frames
    output = torch.empty(output_shape, dtype=values.dtype, device=values.device)

    for segment, (left, right) in enumerate(zip(indices[:-1], indices[1:])):
        left_value = values[:, segment]
        right_value = values[:, segment + 1]
        width = right - left
        if width <= 0:
            raise ValueError(f"non-increasing anchor indices: {indices}")
        for frame in range(left, right + 1):
            alpha = (frame - left) / width
            output[:, frame] = left_value.lerp(right_value, alpha)

    return output.movedim(1, frame_dim)


def gather_frame_tokens(
    hidden: torch.Tensor,
    grid_sizes: torch.Tensor,
    frame_indices: Sequence[int],
) -> tuple[torch.Tensor, int, int, int]:
    """Gather complete frame token slabs from a Wan hidden sequence.

    The pilot deliberately supports batch size one.  This keeps the temporal
    layout explicit and avoids silently mixing padded examples with different
    frame meshes.
    """

    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"expected hidden [1, L, C], got {tuple(hidden.shape)}")
    if grid_sizes.shape != (1, 3):
        raise ValueError(f"expected grid_sizes [1, 3], got {tuple(grid_sizes.shape)}")

    frames, height, width = (int(value) for value in grid_sizes[0].tolist())
    indices = _sorted_unique(frame_indices, frames)
    tokens_per_frame = height * width
    valid_tokens = frames * tokens_per_frame
    if hidden.shape[1] < valid_tokens:
        raise ValueError(
            f"hidden sequence is shorter than grid: {hidden.shape[1]} < {valid_tokens}"
        )

    frame_view = hidden[:, :valid_tokens].reshape(
        1, frames, tokens_per_frame, hidden.shape[-1]
    )
    index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
    selected = frame_view.index_select(1, index_tensor).flatten(1, 2)
    return selected, frames, height, width


def reconstruct_frame_tokens(
    selected_hidden: torch.Tensor,
    selected_indices: Sequence[int],
    *,
    num_frames: int,
    height: int,
    width: int,
    padded_length: int | None = None,
) -> torch.Tensor:
    """Interpolate selected Wan frame slabs back to the full sequence length."""

    tokens_per_frame = height * width
    expected = len(selected_indices) * tokens_per_frame
    if selected_hidden.shape[:2] != (1, expected):
        raise ValueError(
            f"expected selected hidden [1, {expected}, C], got {tuple(selected_hidden.shape)}"
        )
    selected = selected_hidden.reshape(
        1, len(selected_indices), tokens_per_frame, selected_hidden.shape[-1]
    )
    full = interpolate_frame_values(
        selected, selected_indices, num_frames, frame_dim=1
    ).flatten(1, 2)

    if padded_length is None or padded_length == full.shape[1]:
        return full
    if padded_length < full.shape[1]:
        raise ValueError(
            f"padded_length={padded_length} is shorter than valid sequence={full.shape[1]}"
        )
    padding = full.new_zeros(1, padded_length - full.shape[1], full.shape[-1])
    return torch.cat((full, padding), dim=1)


def relative_rms(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Return RMS(error) / RMS(reference) as a Python float."""

    reference_f = reference.float()
    estimate_f = estimate.float()
    numerator = (reference_f - estimate_f).square().mean().sqrt()
    denominator = reference_f.square().mean().sqrt().clamp_min(eps)
    return float((numerator / denominator).item())


def per_frame_relative_rms(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    *,
    frame_dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Vectorized relative RMS with one score per frame."""

    reference = reference.movedim(frame_dim, 1).float()
    estimate = estimate.movedim(frame_dim, 1).float()
    reduce_dims = tuple(dim for dim in range(reference.ndim) if dim not in (0, 1))
    numerator = (reference - estimate).square().mean(dim=reduce_dims).sqrt()
    denominator = reference.square().mean(dim=reduce_dims).sqrt().clamp_min(eps)
    return (numerator / denominator).mean(dim=0)


def leave_one_out_defects(
    selected_values: torch.Tensor,
    selected_indices: Sequence[int],
    core_indices: Sequence[int],
    validators: Sequence[int],
    *,
    frame_dim: int = 1,
) -> dict[int, float]:
    """Measure validator residuals against interpolation from core neighbors.

    ``selected_values`` are outputs already paid for by the current sparse
    block.  No extra DiT call is introduced by this calculation.
    """

    selected = _sorted_unique(selected_indices, max(selected_indices) + 1)
    core = sorted(int(index) for index in core_indices)
    value_by_frame = {
        frame: selected_values.select(frame_dim, position)
        for position, frame in enumerate(selected)
    }
    defects: dict[int, float] = {}
    for validator in validators:
        if validator not in value_by_frame:
            raise ValueError(f"validator {validator} was not computed")
        left, right = frame_neighbors(core, validator)
        if left not in value_by_frame or right not in value_by_frame:
            raise ValueError(
                f"validator neighbors {left}/{right} missing from selected frames {selected}"
            )
        alpha = (validator - left) / (right - left)
        prediction = value_by_frame[left].lerp(value_by_frame[right], alpha)
        defects[int(validator] = relative_rms(value_by_frame[validator], prediction)
    return defects


def temporal_distance(frame: int, selected: Sequence[int]) -> int:
    return min(abs(frame - anchor) for anchor in selected)


def adaptive_mesh_indices(
    num_frames: int,
    budget: int,
    scores: Sequence[float],
    *,
    current: Sequence[int] | None = None,
    distance_power: float = 2.0,
    inertia: float = 0.0,
) -> list[int]:
    """Allocate a fixed-size mesh using risk-weighted farthest-point sampling."""

    _validate_budget(num_frames, budget)
    if len(scores) != num_frames:
        raise ValueError(f"expected {num_frames} scores, got {len(scores)}")
    current_set = set(int(index) for index in (current or ()))
    selected = {0, num_frames - 1}

    while len(selected) < budget:
        candidates = [frame for frame in range(num_frames) if frame not in selected]
        chosen = max(
            candidates,
            key=lambda frame: (
                (max(float(scores[frame]), 0.0) + 1e-6)
                * (temporal_distance(frame, tuple(selected)) + 1) ** distance_power
                * (1.0 + inertia if frame in current_set else 1.0),
                -frame,
            ),
        )
        selected.add(chosen)
    return sorted(selected)

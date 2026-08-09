"""Runtime temporal-RoPE support for unmodified Wan2.1."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Any

import torch

_TEMPORAL_POSITIONS: ContextVar[torch.Tensor | Sequence[float] | None] = ContextVar(
    "coframe_temporal_positions", default=None
)
_PATCHED_MODULES: dict[int, Any] = {}


def temporal_rope_from_positions(
    frequency_table: torch.Tensor, temporal_positions: torch.Tensor | Sequence[float]
) -> torch.Tensor:
    positions = torch.as_tensor(temporal_positions, device=frequency_table.device)
    if positions.ndim != 1:
        raise ValueError(f"temporal positions must be 1-D, got {tuple(positions.shape)}")
    if not torch.is_floating_point(positions):
        return frequency_table[positions.to(dtype=torch.long)]
    inverse_frequency = torch.angle(frequency_table[1]).to(torch.float64)
    phase = torch.outer(positions.to(torch.float64), inverse_frequency)
    return torch.polar(torch.ones_like(phase), phase)


def _rope_apply_with_positions(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    temporal_positions: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    heads, complex_channels = x.size(2), x.size(3) // 2
    split_freqs = freqs.split(
        [
            complex_channels - 2 * (complex_channels // 3),
            complex_channels // 3,
            complex_channels // 3,
        ],
        dim=1,
    )

    output = []
    for sample, (frames, height, width) in enumerate(grid_sizes.tolist()):
        sequence_length = frames * height * width
        if torch.as_tensor(temporal_positions).numel() != frames:
            raise ValueError(
                f"expected {frames} temporal positions, got "
                f"{torch.as_tensor(temporal_positions).numel()}"
            )
        values = torch.view_as_complex(
            x[sample, :sequence_length]
            .to(torch.float64)
            .reshape(sequence_length, heads, -1, 2)
        )
        temporal = temporal_rope_from_positions(split_freqs[0], temporal_positions)
        multipliers = torch.cat(
            [
                temporal.view(frames, 1, 1, -1).expand(
                    frames, height, width, -1
                ),
                split_freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                split_freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(sequence_length, 1, -1)
        values = torch.view_as_real(values * multipliers).flatten(2)
        values = torch.cat((values, x[sample, sequence_length:]))
        output.append(values)
    return torch.stack(output).float()


def install_wan_temporal_rope_patch(model_module: ModuleType) -> None:
    """Patch ``wan.modules.model.rope_apply`` once for scoped gapped RoPE.

    Dense behavior is bit-for-bit delegated to Wan's original function.  Sparse
    positions are active only inside :func:`temporal_position_scope`.
    """

    key = id(model_module)
    if key in _PATCHED_MODULES:
        return
    original = model_module.rope_apply

    def patched(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        positions = _TEMPORAL_POSITIONS.get()
        if positions is None:
            return original(x, grid_sizes, freqs)
        return _rope_apply_with_positions(x, grid_sizes, freqs, positions)

    model_module.rope_apply = patched
    _PATCHED_MODULES[key] = original


@contextmanager
def temporal_position_scope(
    positions: torch.Tensor | Sequence[float] | None,
) -> Iterator[None]:
    token = _TEMPORAL_POSITIONS.set(positions)
    try:
        yield
    finally:
        _TEMPORAL_POSITIONS.reset(token)

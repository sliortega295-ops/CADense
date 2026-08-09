"""Wan2.1 block-sparse execution used by the CoFrame pilot.

This adapter mirrors the public WanModel forward pass while leaving upstream
weights and source files untouched.  The first implementation is intentionally
batch-1 and single-GPU: it is an algorithm-validation path, not yet the final
packed/CUDA-graph latency path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import ModuleType
from typing import Any

import torch
import torch.cuda.amp as amp

from .controller import CoFrameControlller, Selection
from .mesh import (
    gather_frame_tokens,
    interpolate_frame_values,
    leave_one_out_defects,
    reconstruct_frame_tokens,
)
from .rope import install_wan_temporal_rope_patch, temporal_position_scope


@dataclass
class PreparedWanForward:
    hidden: torch.Tensor
    time_embedding: torch.Tensor
    modulation: torch.Tensor
    grid_sizes: torch.Tensor
    sequence_lengths: torch.Tensor
    context: torch.Tensor
    context_lengths: torch.Tensor | None
    original_shapes: list[tuple[int, ...]]
    padded_length: int


@dataclass
class BlockTrace:
    block_index: int
    input_hidden: torch.Tensor
    output_hidden: torch.Tensor


@dataclass
class SparseBlockRecord:
    block_index: int
    selection: Selection
    defects: dict[int, float]
    host_submit_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "selection": self.selection.as_dict(),
            "defects": {str(key): float(value) for key, value in self.defects.items()},
            "host_submit_seconds": self.host_submit_seconds,
        }


@dataclass
class WanForwardResult:
    prediction: torch.Tensor
    traces: dict[int, BlockTrace] = field(default_factory=dict)
    sparse_records: list[SparseBlockRecord] = field(default_factory=list)
    schedule: dict[int, tuple[int, ...]] = field(default_factory=dict)


class Wan21Executor:
    def __init__(self, model: Any, model_module: ModuleType) -> None:
        self.model = model
        self.model_module = model_module
        install_wan_temporal_rope_patch(model_module)

    def prepare(
        self,
        x: list[torch.Tensor],
        *,
        timestep: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
    ) -> PreparedWanForward:
        if len(x) != 1 or len(context) != 1:
            raise ValueError("CoFrame Wan pilot currently supports batch size one")
        model = self.model
        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        embedded = [model.patch_embedding(sample.unsqueeze(0)) for sample in x]
        original_shapes = [tuple(sample.shape) for sample in x]
        grid_sizes = torch.stack(
            [torch.tensor(sample.shape[2:], dtype=torch.long) for sample in embedded]
        )
        flattened = [sample.flatten(2).transpose(1, 2) for sample in embedded]
        sequence_lengths = torch.tensor(
            [sample.size(1) for sample in flattened], dtype=torch.long
        )
        if int(sequence_lengths.max()) > seq_len:
            raise ValueError(
                f"seq_len={seq_len} is smaller than model sequence={int(sequence_lengths.max())}"
            )
        hidden = torch.cat(
            [
                torch.cat(
                    (
                        sample,
                        sample.new_zeros(1, seq_len - sample.size(1), sample.size(2)),
                    ),
                    dim=1,
                )
                for sample in flattened
            ]
        )

        with amp.autocast(dtype=torch.float32):
            time_embedding = model.time_embedding(
                self.model_module.sinusoidal_embedding_1d(
                    model.freq_dim, timestep
                ).float()
            )
            modulation = model.time_projection(time_embedding).unflatten(
                1, (6, model.dim)
            )

        embedded_context = model.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        (
                            sample,
                            sample.new_zeros(
                                model.text_len - sample.size(0), sample.size(1)
                            ),
                        )
                    )
                    for sample in context
                ]
            )
        )
        return PreparedWanForward(
            hidden=hidden,
            time_embedding=time_embedding,
            modulation=modulation,
            grid_sizes=grid_sizes,
            sequence_lengths=sequence_lengths,
            context=embedded_context,
            context_lengths=None,
            original_shapes=original_shapes,
            padded_length=seq_len,
        )

    def _block_kwargs(self, prepared: PreparedWanForward) -> dict[str, Any]:
        return {
            "e": prepared.modulation,
            "seq_lens": prepared.sequence_lengths,
            "grid_sizes": prepared.grid_sizes,
            "freqs": self.model.freqs,
            "context": prepared.context,
            "context_lens": prepared.context_lengths,
        }

    def _finalize(self, hidden: torch.Tensor, prepared: PreparedWanForward) -> torch.Tensor:
        output = self.model.head(hidden, prepared.time_embedding)
        output = self.model.unpatchify(output, prepared.grid_sizes)
        return output[0].float()

    @staticmethod
    def _trace_copy(hidden: torch.Tensor) -> torch.Tensor:
        # Three selected Wan-1.3B blocks fit comfortably in CPU RAM while full
        # all-block traces would not.  fp16 is sufficient for ranking probes.
        return hidden.detach().to(device="cpu", dtype=torch.float16, copy=True)

    def run_dense(
        self,
        prepared: PreparedWanForward,
        *,
        trace_blocks: set[int] | None = None,
    ) -> WanForwardResult:
        trace_blocks = trace_blocks or set()
        hidden = prepared.hidden
        traces: dict[int, BlockTrace] = {}
        kwargs = self._block_kwargs(prepared)
        for block_index, block in enumerate(self.model.blocks):
            input_copy = self._trace_copy(hidden) if block_index in trace_blocks else None
            hidden = block(hidden, **kwargs)
            if input_copy is not None:
                traces[block_index] = BlockTrace(
                    block_index=block_index,
                    input_hidden=input_copy,
                    output_hidden=self._trace_copy(hidden),
                )
        return WanForwardResult(prediction=self._finalize(hidden, prepared), traces=traces)

    def run_sparse_block(
        self,
        *,
        block_index: int,
        hidden: torch.Tensor,
        prepared: PreparedWanForward,
        selected_indices: list[int] | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_hidden, frames, height, width = gather_frame_tokens(
            hidden, prepared.grid_sizes, selected_indices
        )
        tokens_per_frame = height * width
        selected_grid = torch.tensor(
            [[len(selected_indices), height, width]], dtype=torch.long
        )
        selected_lengths = torch.tensor(
            [len(selected_indices) * tokens_per_frame], dtype=torch.long
        )
        kwargs = {
            "e": prepared.modulation,
            "seq_lens": selected_lengths,
            "grid_sizes": selected_grid,
            "freqs": self.model.freqs,
            "context": prepared.context,
            "context_lens": prepared.context_lengths,
        }
        with temporal_position_scope(selected_indices):
            selected_output = self.model.blocks[block_index](selected_hidden, **kwargs)
        reconstructed = reconstruct_frame_tokens(
            selected_output,
            selected_indices,
            num_frames=frames,
            height=height,
            width=width,
            padded_length=prepared.padded_length,
        )
        return reconstructed, selected_output

    def run_static_sparse(
        self,
        prepared: PreparedWanForward,
        *,
        sparse_blocks: set[int],
        schedule: dict[int, tuple[int, ...]],
    ) -> WanForwardResult:
        hidden = prepared.hidden
        dense_kwargs = self._block_kwargs(prepared)
        records: list[SparseBlockRecord] = []
        for block_index, block in enumerate(self.model.blocks):
            if block_index not in sparse_blocks:
                hidden = block(hidden, **dense_kwargs)
                continue
            selected = schedule[block_index]
            start = perf_counter()
            hidden, _ = self.run_sparse_block(
                block_index=block_index,
                hidden=hidden,
                prepared=prepared,
                selected_indices=selected,
            )
            records.append(
                SparseBlockRecord(
                    block_index=block_index,
                    selection=Selection(
                        block_index=block_index,
                        core=tuple(selected),
                        validators=tuple(),
                        selected=tuple(selected),
                    ),
                    defects={},
                    host_submit_seconds=perf_counter() - start,
                )
            )
        return WanForwardResult(
            prediction=self._finalize(hidden, prepared),
            sparse_records=records,
            schedule=schedule,
        )

    def run_coframe(
        self,
        prepared: PreparedWanForward,
        *,
        sparse_blocks: set[int],
        controller: CoFrameController,
    ) -> WanForwardResult:
        hidden = prepared.hidden
        dense_kwargs = self._block_kwargs(prepared)
        records: list[SparseBlockRecord] = []
        schedule: dict[int, tuple[int, ...]] = {}
        for block_index, block in enumerate(self.model.blocks):
            if block_index not in sparse_blocks:
                hidden = block(hidden, **dense_kwargs)
                continue
            selection = controller.select(block_index)
            schedule[block_index] = selection.selected
            start = perf_counter()
            hidden, selected_output = self.run_sparse_block(
                block_index=block_index,
                hidden=hidden,
                prepared=prepared,
                selected_indices=selection.selected,
            )
            frames, height, width = (
                int(value) for value in prepared.grid_sizes[0].tolist()
            )
            selected_frames = selected_output.reshape(
                1,
                len(selection.selected),
                height * width,
                selected_output.shape[-1],
            )
            defects = leave_one_out_defects(
                selected_frames,
                selection.selected,
                selection.core,
                selection.validators,
                frame_dim=1,
            )
            controller.observe(selection, defects)
            records.append(
                SparseBlockRecord(
                    block_index=block_index,
                    selection=selection,
                    defects=defects,
                    host_submit_seconds=perf_counter() - start,
                )
            )
        return WanForwardResult(
            prediction=self._finalize(hidden, prepared),
            sparse_records=records,
            schedule=schedule,
        )

    def local_block_probe(
        self,
        *,
        block_index: int,
        trace: BlockTrace,
        prepared: PreparedWanForward,
        selected_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = prepared.hidden.device
        input_hidden = trace.input_hidden.to(device=device, dtype=prepared.hidden.dtype)
        dense_output = trace.output_hidden.to(device=device, dtype=prepared.hidden.dtype)
        reconstructed, selected_output = self.run_sparse_block(
            block_index=block_index,
            hidden=input_hidden,
            prepared=prepared,
            selected_indices=selected_indices,
        )
        return reconstructed, selected_output, dense_output


def static_schedule(
    sparse_blocks: set[int], indices: list[int]
) -> dict[int, tuple[int, ...]]:
    return {block: tuple(indices) for block in sparse_blocks}

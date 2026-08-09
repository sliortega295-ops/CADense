"""Local causal probes for the CoFrame Wan2.1 pilot.

The probe asks an action-oriented question.  Starting from a Rhyme-selected
``K-1`` core, which candidate frame should receive the final slot of a fixed
``K``-frame block budget?  For every candidate we measure:

* CoFrame's leave-one-out block-output defect;
* Rhyme clean-latent novelty;
* clean-latent interpolation residual;
* the true reduction in dense-block reconstruction error obtained by adding
  that candidate.

Exhaustively scanning candidates is diagnostic only.  The deployed controller
pays for one validator, uses its observed defect, and updates later blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .mesh import (
    interpolate_frame_values,
    leave_one_out_defects,
    per_frame_relative_rms,
    relative_rms,
)
from .selectors import (
    fixed_middle_indices,
    proxy_interpolation_scores,
    rhyme_budgeted_indices,
    rhyme_candidate_scores,
    uniform_indices,
)
from .wan21 import BlockTrace, PreparedWanForward, Wan21Executor


@dataclass(frozen=True)
class ProbeContext:
    prompt_id: str
    prompt: str
    seed: int
    step_index: int
    timestep: float
    block_index: int
    total_budget: int

    def fields(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "seed": int(self.seed),
            "step_index": int(self.step_index),
            "timestep": float(self.timestep),
            "block_index": int(self.block_index),
            "total_budget": int(self.total_budget),
        }


def hidden_frames(hidden: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
    """View Wan hidden tokens as ``[1, F, P, C]`` without padded tokens."""

    frames, height, width = (int(value) for value in grid_sizes[0].tolist())
    valid = frames * height * width
    return hidden[:, :valid].reshape(1, frames, height * width, hidden.shape[-1])


def _selected_output_frames(
    selected_output: torch.Tensor,
    *,
    selected_count: int,
    grid_sizes: torch.Tensor,
) -> torch.Tensor:
    _, height, width = (int(value) for value in grid_sizes[0].tolist())
    return selected_output.reshape(
        1, selected_count, height * width, selected_output.shape[-1]
    )


def _method_row(
    *,
    context: ProbeContext,
    method: str,
    indices: list[int],
    reconstructed: torch.Tensor,
    dense_output: torch.Tensor,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "method",
        **context.fields(),
        "method": method,
        "indices": list(indices),
        "block_relative_rms": relative_rms(dense_output, reconstructed),
        "diagnostic_only": bool(diagnostic_only),
    }


def probe_block_candidates(
    *,
    executor: Wan21Executor,
    prepared: PreparedWanForward,
    trace: BlockTrace,
    clean_proxy: torch.Tensor,
    context: ProbeContext,
) -> list[dict[str, Any]]:
    """Run matched-budget local probes for one Wan block.

    The dense trace fixes the block input.  Every sparse intervention therefore
    differs only in its selected temporal mesh; later blocks and denoising steps
    are deliberately excluded from this G1 mechanism test.
    """

    frames = int(prepared.grid_sizes[0, 0].item())
    if not 3 <= context.total_budget <= frames:
        raise ValueError(
            f"total_budget must be in [3, {frames}], got {context.total_budget}"
        )
    core_budget = context.total_budget - 1
    core = rhyme_budgeted_indices(clean_proxy, core_budget, force_last=True)
    candidates = [frame for frame in range(1, frames - 1) if frame not in core]

    # Core-only output supplies the counterfactual omission error for every
    # candidate.  It is K-1 rather than a matched baseline and is never reported
    # as a deployable speed/quality point.
    core_reconstructed, _, dense_output = executor.local_block_probe(
        block_index=context.block_index,
        trace=trace,
        prepared=prepared,
        selected_indices=core,
    )
    dense_frames = hidden_frames(dense_output, prepared.grid_sizes)
    core_frames = hidden_frames(core_reconstructed, prepared.grid_sizes)
    core_per_frame = per_frame_relative_rms(dense_frames, core_frames)
    core_full_error = relative_rms(dense_output, core_reconstructed)

    rhyme_scores = rhyme_candidate_scores(clean_proxy, core)
    proxy_scores = proxy_interpolation_scores(clean_proxy, core)
    rows: list[dict[str, Any]] = []
    candidate_artifacts: dict[int, tuple[float, float]] = {}

    for candidate in candidates:
        selected = sorted([*core, candidate])
        reconstructed, selected_output, _ = executor.local_block_probe(
            block_index=context.block_index,
            trace=trace,
            prepared=prepared,
            selected_indices=selected,
        )
        selected_frames = _selected_output_frames(
            selected_output,
            selected_count=len(selected),
            grid_sizes=prepared.grid_sizes,
        )
        defect = leave_one_out_defects(
            selected_frames,
            selected,
            core,
            [candidate],
            frame_dim=1,
        )[candidate]
        reconstructed_frames = hidden_frames(reconstructed, prepared.grid_sizes)
        augmented_per_frame = per_frame_relative_rms(dense_frames, reconstructed_frames)
        augmented_full_error = relative_rms(dense_output, reconstructed)
        omission_error = float(core_per_frame[candidate].item())
        selected_exact_error = float(augmented_per_frame[candidate].item())
        full_gain = core_full_error - augmented_full_error
        frame_gain = omission_error - selected_exact_error

        rows.append(
            {
                "kind": "candidate",
                **context.fields(),
                "core_indices": list(core),
                "candidate": int(candidate),
                "selected_indices": list(selected),
                "coframe_defect": float(defect),
                "rhyme_novelty": float(rhyme_scores[candidate]),
                "proxy_interp_error": float(proxy_scores[candidate]),
                "omission_frame_error": omission_error,
                "selected_exact_frame_error": selected_exact_error,
                "frame_error_gain": frame_gain,
                "core_full_error": core_full_error,
                "augmented_full_error": augmented_full_error,
                "full_error_gain": full_gain,
            }
        )
        candidate_artifacts[candidate] = (float(defect), augmented_full_error)
        del reconstructed, selected_output, selected_frames, reconstructed_frames

    # Matched K baselines.
    methods = {
        "fixed_middle": fixed_middle_indices(frames, context.total_budget),
        "uniform_fis_sanity": uniform_indices(frames, context.total_budget),
        "rhyme_budgeted": rhyme_budgeted_indices(
            clean_proxy, context.total_budget, force_last=True
        ),
    }
    for method, indices in methods.items():
        reconstructed, _, _ = executor.local_block_probe(
            block_index=context.block_index,
            trace=trace,
            prepared=prepared,
            selected_indices=indices,
        )
        rows.append(
            _method_row(
                context=context,
                method=method,
                indices=indices,
                reconstructed=reconstructed,
                dense_output=dense_output,
            )
        )

    # Isolate the final-slot decision while keeping the first K-1 Rhyme anchors
    # fixed.  Exhaustive defect and gain scans are diagnostics, not deployable
    # methods, and are explicitly marked as such in every row.
    if candidates:
        rhyme_choice = max(candidates, key=lambda frame: (rhyme_scores[frame], -frame))
        defect_choice = max(
            candidates,
            key=lambda frame: (candidate_artifacts[frame][0], -frame),
        )
        gain_choice = max(
            candidates,
            key=lambda frame: (
                core_full_error - candidate_artifacts[frame][1],
                -frame,
            ),
        )
        for method, choice, diagnostic in (
            ("rhyme_final_slot", rhyme_choice, False),
            ("defect_scan_final_slot", defect_choice, True),
            ("gain_oracle_final_slot", gain_choice, True),
        ):
            indices = sorted([*core, choice])
            reconstructed, _, _ = executor.local_block_probe(
                block_index=context.block_index,
                trace=trace,
                prepared=prepared,
                selected_indices=indices,
            )
            rows.append(
                _method_row(
                    context=context,
                    method=method,
                    indices=indices,
                    reconstructed=reconstructed,
                    dense_output=dense_output,
                    diagnostic_only=diagnostic,
                )
            )
            del reconstructed

    rows.append(
        {
            "kind": "core_diagnostic",
            **context.fields(),
            "method": "rhyme_core_k_minus_1",
            "indices": list(core),
            "block_relative_rms": core_full_error,
            "diagnostic_only": True,
        }
    )
    return rows

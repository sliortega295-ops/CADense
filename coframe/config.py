from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

Method = Literal["dense", "fixed", "rhyme", "coframe"]
KVMode = Literal["anchor_only", "full_kv"]
InterpolationTarget = Literal["delta", "state"]
DefectTarget = Literal["delta", "state"]


@dataclass(slots=True)
class CoFrameConfig:
    """Configuration for CoFrame's Wan2.1 validation path.

    The defaults target Wan2.1-T2V-1.3B-Diffusers with 21 latent frames
    (81 decoded frames), 30 transformer blocks, and a 50-step sampler.
    """

    method: Method = "coframe"
    warmup_steps: int = 5
    num_anchors: int = 9
    force_boundaries: bool = True
    min_anchor_gap: int = 1

    # Only middle blocks are sparse by default. Dense boundary blocks act as
    # error-reset/safety layers and keep the first prototype conservative.
    sparse_block_start: int = 3
    sparse_block_end: int = 27  # exclusive
    block_group_size: int = 3

    kv_mode: KVMode = "anchor_only"
    interpolation_target: InterpolationTarget = "delta"
    defect_target: DefectTarget = "delta"

    # RhymeFlow-style clean-latent prior.
    rhyme_similarity_threshold: float = 0.98
    rhyme_prior_weight: float = 0.35

    # Online risk and mesh refresh.
    risk_ema: float = 0.75
    risk_floor: float = 1.0e-4
    interval_gap_power: float = 2.0
    move_penalty: float = 0.02
    min_refresh_gain: float = 1.0e-4
    max_swaps_per_refresh: int = 1
    refresh_every_groups: int = 1
    defect_clip: float = 10.0

    # Cheap channel sketch for defect measurement. 0 means exact channel RMS.
    sketch_dim: int = 64
    sketch_seed: int = 2026

    # Optional causal diagnostic: recompute a dense block from the same input
    # and correlate CoFrame risk with actual sparse-block error.
    oracle_probe_steps: Sequence[int] = field(default_factory=tuple)
    oracle_probe_blocks: Sequence[int] = field(default_factory=tuple)

    trace_path: str | None = None
    strict_diffusers_version: bool = True

    def validate(self, *, num_blocks: int | None = None, num_frames: int | None = None) -> None:
        if self.method not in {"dense", "fixed", "rhyme", "coframe"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.num_anchors < 1:
            raise ValueError("num_anchors must be positive")
        if self.force_boundaries and self.num_anchors < 2 and (num_frames is None or num_frames > 1):
            raise ValueError("At least two anchors are required when force_boundaries=True")
        if num_frames is not None and self.num_anchors > num_frames:
            raise ValueError(f"num_anchors={self.num_anchors} exceeds latent frames={num_frames}")
        if self.min_anchor_gap < 1:
            raise ValueError("min_anchor_gap must be >= 1")
        if self.block_group_size < 1:
            raise ValueError("block_group_size must be >= 1")
        if not 0.0 <= self.risk_ema < 1.0:
            raise ValueError("risk_ema must be in [0, 1)")
        if self.interval_gap_power <= 0:
            raise ValueError("interval_gap_power must be positive")
        if self.max_swaps_per_refresh < 0:
            raise ValueError("max_swaps_per_refresh must be non-negative")
        if self.refresh_every_groups < 1:
            raise ValueError("refresh_every_groups must be >= 1")
        if self.sketch_dim < 0:
            raise ValueError("sketch_dim must be >= 0")
        if num_blocks is not None:
            if not 0 <= self.sparse_block_start <= num_blocks:
                raise ValueError("sparse_block_start is out of range")
            if not self.sparse_block_start <= self.sparse_block_end <= num_blocks:
                raise ValueError("sparse_block_end is out of range")

    def is_sparse_block(self, block_index: int) -> bool:
        return self.method != "dense" and self.sparse_block_start <= block_index < self.sparse_block_end

    def should_probe(self, step_index: int, block_index: int) -> bool:
        return step_index in set(self.oracle_probe_steps) and block_index in set(self.oracle_probe_blocks)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["oracle_probe_steps"] = list(self.oracle_probe_steps)
        result["oracle_probe_blocks"] = list(self.oracle_probe_blocks)
        return result

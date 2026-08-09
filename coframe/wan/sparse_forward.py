from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from packaging.version import Version

from ..config import CoFrameConfig
from ..controller import AdaptiveMeshController
from ..interpolation import (
    frame_token_indices,
    frames_to_tokens,
    leave_one_out_defects,
    per_frame_relative_rms,
    reconstruct_sparse_block,
    tokens_to_frames,
)
from ..trace import CoFrameTrace


_PROJECTION_CACHE: dict[tuple[str, int | None, str, int, int, int], torch.Tensor] = {}


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    num_frames: int
    height: int
    width: int

    @property
    def tokens_per_frame(self) -> int:
        return self.height * self.width

    @property
    def sequence_length(self) -> int:
        return self.num_frames * self.tokens_per_frame


@dataclass(slots=True)
class TransformerForwardMetadata:
    step_index: int
    block_anchors: dict[int, list[int]]
    refresh_events: list[dict[str, Any]]
    defects: list[dict[str, Any]]
    probes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "block_anchors": {str(key): value for key, value in self.block_anchors.items()},
            "refresh_events": self.refresh_events,
            "defects": self.defects,
            "probes": self.probes,
        }


def require_diffusers_034(strict: bool = True) -> str:
    """Validate the integration target before touching Wan internals."""
    try:
        import diffusers
    except ImportError as exc:  # pragma: no cover - exercised only in GPU env
        raise ImportError("CoFrame's Wan integration requires diffusers==0.34.0") from exc

    found = Version(diffusers.__version__)
    target = Version("0.34.0")
    if found != target:
        message = (
            f"CoFrame currently targets diffusers==0.34.0, but found {found}. "
            "Wan block internals changed after 0.34; install the pinned version or pass "
            "strict_diffusers_version=False only for deliberate porting work."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    return str(found)


def _apply_rotary_emb_subset(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # Mirrors diffusers 0.34.0 WanAttnProcessor2_0. hidden_states is
    # [B,H,L,D], freqs is complex [1,1,L,D/2].
    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
    rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
    output = torch.view_as_real(rotated * freqs).flatten(3, 4)
    return output.type_as(hidden_states)


def _active_self_attention(
    attn: Any,
    query_states: torch.Tensor,
    key_value_states: torch.Tensor,
    rotary_q: torch.Tensor,
    rotary_k: torch.Tensor,
) -> torch.Tensor:
    if getattr(attn, "add_k_proj", None) is not None:
        raise NotImplementedError("The initial CoFrame integration targets Wan T2V self-attention only")

    query = attn.to_q(query_states)
    key = attn.to_k(key_value_states)
    value = attn.to_v(key_value_states)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
    key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
    value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

    query = _apply_rotary_emb_subset(query, rotary_q)
    key = _apply_rotary_emb_subset(key, rotary_k)
    output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    output = output.transpose(1, 2).flatten(2, 3).type_as(query_states)
    output = attn.to_out[0](output)
    output = attn.to_out[1](output)
    return output


def _build_projection(
    hidden_dim: int,
    sketch_dim: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sketch_dim <= 0 or sketch_dim >= hidden_dim:
        return None
    key = (device.type, device.index, str(dtype), int(hidden_dim), int(sketch_dim), int(seed))
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(hidden_dim, sketch_dim, generator=generator, dtype=torch.float32)
    projection /= math.sqrt(float(sketch_dim))
    projection = projection.to(device=device, dtype=dtype)
    _PROJECTION_CACHE[key] = projection
    return projection


def _sparse_block_forward(
    block: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    *,
    anchors: list[int],
    geometry: FrameGeometry,
    config: CoFrameConfig,
    compute_defects: bool,
    projection: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Execute one Wan block exactly on anchors and reconstruct all frames."""
    if temb.ndim != 3:
        raise NotImplementedError("CoFrame v0.1 targets Wan2.1's [B,6,D] timestep modulation")

    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        block.scale_shift_table + temb.float()
    ).chunk(6, dim=1)

    input_frames = tokens_to_frames(hidden_states, geometry.num_frames, geometry.tokens_per_frame)
    active_indices = frame_token_indices(anchors, geometry.tokens_per_frame, device=hidden_states.device)

    exact_anchor_inputs = hidden_states.index_select(1, active_indices)
    if config.kv_mode == "anchor_only":
        normalized_anchor = (
            block.norm1(exact_anchor_inputs.float()) * (1 + scale_msa) + shift_msa
        ).type_as(exact_anchor_inputs)
        key_value_states = normalized_anchor
        rotary_k = rotary_emb.index_select(2, active_indices)
    elif config.kv_mode == "full_kv":
        normalized_full = (
            block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        normalized_anchor = normalized_full.index_select(1, active_indices)
        key_value_states = normalized_full
        rotary_k = rotary_emb
    else:  # pragma: no cover - config validation guards this
        raise ValueError(f"Unsupported kv_mode: {config.kv_mode}")

    rotary_q = rotary_emb.index_select(2, active_indices)
    attention_output = _active_self_attention(
        block.attn1,
        normalized_anchor,
        key_value_states,
        rotary_q,
        rotary_k,
    )

    exact_anchor_states = (
        exact_anchor_inputs.float() + attention_output * gate_msa
    ).type_as(hidden_states)

    normalized_anchor = block.norm2(exact_anchor_states.float()).type_as(exact_anchor_states)
    attention_output = block.attn2(
        hidden_states=normalized_anchor,
        encoder_hidden_states=encoder_hidden_states,
    )
    exact_anchor_states = exact_anchor_states + attention_output

    normalized_anchor = (
        block.norm3(exact_anchor_states.float()) * (1 + c_scale_msa) + c_shift_msa
    ).type_as(exact_anchor_states)
    feed_forward_output = block.ffn(normalized_anchor)
    exact_anchor_states = (
        exact_anchor_states.float() + feed_forward_output.float() * c_gate_msa
    ).type_as(hidden_states)

    exact_anchor_frames = exact_anchor_states.reshape(
        hidden_states.shape[0], len(anchors), geometry.tokens_per_frame, hidden_states.shape[-1]
    )
    reconstructed_frames, exact_anchor_delta = reconstruct_sparse_block(
        input_frames,
        exact_anchor_frames,
        anchors,
        target=config.interpolation_target,
    )

    defects: dict[int, torch.Tensor] = {}
    if compute_defects:
        values = exact_anchor_delta if config.defect_target == "delta" else exact_anchor_frames
        defects = leave_one_out_defects(values, anchors, projection=projection)

    return frames_to_tokens(reconstructed_frames), defects


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    # Frame counts are tiny. A stable double argsort is enough for the causal
    # diagnostic; ties are rare in floating-point error vectors.
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-8) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum().div(denominator + eps).item())


def _probe_entry(
    *,
    step_index: int,
    block_index: int,
    dense_output: torch.Tensor,
    sparse_output: torch.Tensor,
    geometry: FrameGeometry,
    controller: AdaptiveMeshController,
    anchors: list[int],
    defects: dict[int, torch.Tensor],
) -> dict[str, Any]:
    dense_frames = tokens_to_frames(dense_output, geometry.num_frames, geometry.tokens_per_frame)
    sparse_frames = tokens_to_frames(sparse_output, geometry.num_frames, geometry.tokens_per_frame)
    actual_error = per_frame_relative_rms(dense_frames, sparse_frames).detach().cpu()

    anchor_mask = torch.zeros(geometry.num_frames, dtype=torch.bool)
    anchor_mask[torch.tensor(anchors, dtype=torch.long)] = True
    non_anchor_mask = ~anchor_mask
    actual_non_anchor = actual_error[non_anchor_mask]

    prior_base = controller.risk_floor + controller.prior_weight * controller.prior
    prior_predicted = controller.approximation_risk(anchors, prior_base)
    causal_predicted = controller.approximation_risk(anchors, controller.risk)
    current_defect_base = controller.project_defects(defects, anchors)
    current_defect_predicted = controller.approximation_risk(anchors, current_defect_base)

    def correlations(predicted: torch.Tensor) -> tuple[float | None, float | None]:
        values = predicted[non_anchor_mask]
        if values.numel() < 2:
            return None, None
        return _pearson(values, actual_non_anchor), _pearson(_rankdata(values), _rankdata(actual_non_anchor))

    prior_pearson, prior_spearman = correlations(prior_predicted)
    causal_pearson, causal_spearman = correlations(causal_predicted)
    defect_pearson, defect_spearman = correlations(current_defect_predicted)
    return {
        "step": step_index,
        "block": block_index,
        "anchors": list(anchors),
        "actual_frame_error": actual_error.tolist(),
        "prior_expected_error": prior_predicted.tolist(),
        "causal_expected_error": causal_predicted.tolist(),
        "current_defect_expected_error": current_defect_predicted.tolist(),
        "pearson": causal_pearson,
        "spearman": causal_spearman,
        "prior_pearson": prior_pearson,
        "prior_spearman": prior_spearman,
        "defect_pearson": defect_pearson,
        "defect_spearman": defect_spearman,
        "spearman_gain_over_rhyme_prior": (
            None if causal_spearman is None or prior_spearman is None else causal_spearman - prior_spearman
        ),
        "mean_relative_rms": float(actual_error.mean().item()),
        "non_anchor_mean_relative_rms": float(actual_non_anchor.mean().item()),
        "anchor_context_error_mean": float(actual_error[anchor_mask].mean().item()),
        "max_relative_rms": float(actual_error.max().item()),
    }


def coframe_transformer_forward(
    transformer: Any,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    config: CoFrameConfig,
    controller: AdaptiveMeshController,
    step_index: int,
    replay_block_anchors: dict[int, list[int]] | None = None,
    update_controller: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
    trace: CoFrameTrace | None = None,
) -> tuple[torch.Tensor, TransformerForwardMetadata]:
    """Wan2.1 transformer forward with block-conditional sparse frame meshes.

    The conditional CFG branch calls this with ``update_controller=True`` and
    records its per-block schedule. The unconditional branch replays the exact
    same schedule, avoiding branch-dependent token shapes and controller drift.
    """
    require_diffusers_034(config.strict_diffusers_version)
    if attention_kwargs and attention_kwargs.get("scale") is not None:
        raise NotImplementedError("LoRA scale handling is not implemented in CoFrame's custom Wan forward")
    if hidden_states.ndim != 5:
        raise ValueError("hidden_states must be [B,C,F,H,W]")
    if timestep.ndim != 1:
        raise NotImplementedError("CoFrame v0.1 targets Wan2.1 scalar timesteps")

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    if p_t != 1:
        raise NotImplementedError("CoFrame v0.1 assumes temporal patch_size == 1")
    geometry = FrameGeometry(num_frames, height // p_h, width // p_w)
    config.validate(num_blocks=len(transformer.blocks), num_frames=num_frames)
    if controller.num_frames != num_frames:
        raise ValueError("Controller frame count does not match transformer input")

    rotary_emb = transformer.rope(hidden_states)
    hidden_states = transformer.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
        timestep,
        encoder_hidden_states,
        None,
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    projection = _build_projection(
        hidden_states.shape[-1],
        config.sketch_dim,
        config.sketch_seed,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    metadata = TransformerForwardMetadata(
        step_index=step_index,
        block_anchors={},
        refresh_events=[],
        defects=[],
        probes=[],
    )
    group_defects: dict[int, list[float]] = {}
    sparse_group_index = 0

    for block_index, block in enumerate(transformer.blocks):
        if not config.is_sparse_block(block_index):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            continue

        if replay_block_anchors is not None:
            if block_index not in replay_block_anchors:
                raise KeyError(f"Missing replay anchors for sparse block {block_index}")
            anchors = list(replay_block_anchors[block_index])
        else:
            anchors = list(controller.anchors)
        metadata.block_anchors[block_index] = anchors

        block_input = hidden_states
        dense_output = None
        if config.should_probe(step_index, block_index) and replay_block_anchors is None:
            dense_output = block(block_input, encoder_hidden_states, timestep_proj, rotary_emb)

        compute_defects = config.method == "coframe" and replay_block_anchors is None and update_controller
        hidden_states, defects = _sparse_block_forward(
            block,
            block_input,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            anchors=anchors,
            geometry=geometry,
            config=config,
            compute_defects=compute_defects,
            projection=projection,
        )

        if defects:
            defect_values = {frame: float(value.detach().float().item()) for frame, value in defects.items()}
            metadata.defects.append(
                {"step": step_index, "block": block_index, "anchors": anchors, "values": defect_values}
            )
            for frame, value in defect_values.items():
                group_defects.setdefault(frame, []).append(value)

        if dense_output is not None:
            metadata.probes.append(
                _probe_entry(
                    step_index=step_index,
                    block_index=block_index,
                    dense_output=dense_output,
                    sparse_output=hidden_states,
                    geometry=geometry,
                    controller=controller,
                    anchors=anchors,
                    defects=defects,
                )
            )

        relative_in_sparse_region = block_index - config.sparse_block_start + 1
        is_group_boundary = (
            relative_in_sparse_region % config.block_group_size == 0
            or block_index + 1 == config.sparse_block_end
        )
        if is_group_boundary and replay_block_anchors is None:
            sparse_group_index += 1
            if compute_defects and group_defects:
                aggregated = {
                    frame: sum(values) / len(values)
                    for frame, values in group_defects.items()
                    if values
                }
                controller.observe(aggregated, anchors=anchors)
                if sparse_group_index % config.refresh_every_groups == 0:
                    refreshes = controller.refresh()
                    for refresh in refreshes:
                        entry = {
                            "step": step_index,
                            "after_block": block_index,
                            "group": sparse_group_index,
                            **refresh.to_dict(),
                        }
                        metadata.refresh_events.append(entry)
                        if trace is not None:
                            trace.add("mesh_refresh", **entry)
                group_defects = {}

    shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = transformer.proj_out(hidden_states)

    out_channels = transformer.config.out_channels or transformer.config.in_channels
    hidden_states = hidden_states.reshape(
        batch_size,
        geometry.num_frames,
        geometry.height,
        geometry.width,
        p_t,
        p_h,
        p_w,
        out_channels,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if trace is not None:
        trace.add("transformer_forward", **metadata.to_dict())
    return output, metadata

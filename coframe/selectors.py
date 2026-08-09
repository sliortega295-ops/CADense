"""Frame selectors used as CoFrame priors and matched-budget baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional

from .mesh import adaptive_mesh_indices, frame_neighbors, uniform_indices


def frame_descriptors(frames: torch.Tensor) -> torch.Tensor:
    """Convert latent or hidden frames to L2-normalized frame descriptors.

    Supported layouts:
      * ``[C, F, H, W]`` Wan latents;
      * ``[B, F, P, C]`` frame-token tensors (the pilot uses ``B=1``);
      * ``[F, P, C]`` frame-token tensors;
      * ``[F, D]`` precomputed descriptors.

    RhymeFlow computes cosine similarity in one-step clean-latent space.  We
    flatten rather than spatially average by default so that a small moving
    region is not erased before the baseline is evaluated.
    """

    if frames.ndim == 4 and frames.shape[0] == 1:
        # CoFrame's token layout is [1, F, P, C].  Wan latents have C=16, so a
        # leading singleton is unambiguous in the supported pilot workloads.
        values = frames[0].flatten(1)
    elif frames.ndim == 4:
        # Wan latent [C, F, H, W].
        values = frames.permute(1, 0, 2, 3).flatten(1)
    elif frames.ndim == 3:
        values = frames.flatten(1)
    elif frames.ndim == 2:
        values = frames
    else:
        raise ValueError(f"unsupported frame tensor shape: {tuple(frames.shape)}")
    return functional.normalize(values.float(), dim=-1, eps=1e-8)


def cosine_novelty(descriptors: torch.Tensor, reference: int) -> torch.Tensor:
    values = (
        frame_descriptors(descriptors)
        if descriptors.ndim != 2
        else functional.normalize(descriptors.float(), dim=-1, eps=1e-8)
    )
    return 1.0 - values @ values[reference]


def _rhyme_threshold_from_descriptors(
    descriptors: torch.Tensor,
    threshold: float,
    *,
    force_last: bool,
) -> list[int]:
    num_frames = descriptors.shape[0]
    selected = [0]
    preceding = 0
    for frame in range(1, num_frames):
        similarity = float(torch.dot(descriptors[frame], descriptors[preceding]).item())
        if similarity < threshold:
            selected.append(frame)
            preceding = frame
    if force_last and selected[-1] != num_frames - 1:
        selected.append(num_frames - 1)
    return selected


def rhyme_sequential_threshold(
    frames: torch.Tensor,
    threshold: float,
    *,
    force_last: bool = False,
) -> list[int]:
    """RhymeFlow's chronological, nearest-preceding-key threshold rule.

    The first frame is selected.  Every later frame is compared only with the
    nearest preceding selected keyframe and becomes a keyframe when cosine
    similarity falls below ``threshold``.
    """

    return _rhyme_threshold_from_descriptors(
        frame_descriptors(frames), threshold, force_last=force_last
    )


def rhyme_candidate_scores(
    frames: torch.Tensor,
    core_indices: Sequence[int],
) -> list[float]:
    """Cosine novelty to each frame's nearest preceding core keyframe."""

    descriptors = frame_descriptors(frames)
    core = sorted(set(int(index) for index in core_indices))
    if not core or core[0] != 0:
        raise ValueError(f"Rhyme candidate scoring requires frame 0, got {core}")
    scores = [0.0] * descriptors.shape[0]
    core_set = set(core)
    preceding = 0
    for frame in range(descriptors.shape[0]):
        if frame in core_set:
            preceding = frame
            continue
        scores[frame] = float(
            (1.0 - torch.dot(descriptors[frame], descriptors[preceding])).item()
        )
    return scores


def _selected_key_importance(
    descriptors: torch.Tensor,
    selected: Sequence[int],
    frame: int,
) -> float:
    """Novelty of an existing key relative to the preceding key.

    This is used only to repair a threshold-selected set to an exact cardinality.
    It deliberately preserves the one-pass Rhyme inductive bias rather than
    replacing it with a global top-k cosine selector.
    """

    ordered = sorted(int(index) for index in selected)
    position = ordered.index(int(frame))
    if position == 0:
        return float("inf")
    preceding = ordered[position - 1]
    return float((1.0 - torch.dot(descriptors[frame], descriptors[preceding])).item())


def rhyme_budgeted_indices(
    frames: torch.Tensor,
    budget: int,
    *,
    force_last: bool = True,
    threshold_samples: int = 2049,
) -> list[int]:
    """Fixed-budget adaptation of RhymeFlow's sequential selector.

    RhymeFlow itself is threshold-based, so its number of keyframes varies by
    sample.  For latency-matched ablations we search the threshold whose
    sequential set is closest to ``budget`` and repair only discrete ties using
    the same preceding-key cosine novelty.
    """

    descriptors = frame_descriptors(frames)
    num_frames = descriptors.shape[0]
    if not 2 <= budget <= num_frames:
        raise ValueError(f"invalid budget {budget} for F={num_frames}")
    if budget == num_frames:
        return list(range(num_frames))

    best: list[int] | None = None
    best_key: tuple[int, int, float] | None = None
    for threshold in torch.linspace(-1.0, 1.0, threshold_samples).tolist():
        candidate = _rhyme_threshold_from_descriptors(
            descriptors, threshold, force_last=force_last
        )
        # Prefer exact cardinality; then prefer not exceeding the budget; then
        # prefer the smaller threshold for deterministic behavior.
        key = (
            abs(len(candidate) - budget),
            0 if len(candidate) <= budget else 1,
            float(threshold),
        )
        if best_key is None or key < best_key:
            best = candidate
            best_key = key
            if key[0] == 0:
                break
    assert best is not None
    selected = sorted(set(best + ([0, num_frames - 1] if force_last else [0])))

    while len(selected) < budget:
        scores = rhyme_candidate_scores(descriptors, selected)
        candidate = max(
            (frame for frame in range(num_frames) if frame not in selected),
            key=lambda frame: (scores[frame], -frame),
        )
        selected.append(candidate)
        selected.sort()

    while len(selected) > budget:
        removable = [
            frame
            for frame in selected
            if frame != 0 and (not force_last or frame != num_frames - 1)
        ]
        if not removable:
            raise RuntimeError(
                f"cannot reduce Rhyme set {selected} to budget={budget}"
            )
        remove = min(
            removable,
            key=lambda frame: (
                _selected_key_importance(descriptors, selected, frame),
                frame,
            ),
        )
        selected.remove(remove)

    return selected


def rhyme_prior_scores(frames: torch.Tensor) -> list[float]:
    """Dense semantic-change prior used to initialize CoFrame risk."""

    descriptors = frame_descriptors(frames)
    scores = torch.zeros(descriptors.shape[0], dtype=torch.float32)
    scores[0] = 1.0
    if descriptors.shape[0] > 1:
        scores[1:] = 1.0 - (descriptors[1:] * descriptors[:-1]).sum(dim=-1)
        scores[-1] = max(float(scores[-1]), 1e-3)
    return [float(value) for value in scores]


def proxy_interpolation_scores(
    frames: torch.Tensor,
    core_indices: Sequence[int],
) -> list[float]:
    """Input-space interpolation residual for every non-core frame.

    This is an intentionally strong cheap baseline: it asks whether the one-step
    clean latent itself bends away from the current temporal mesh.  CoFrame must
    beat this proxy to justify looking inside a DiT block.
    """

    descriptors = frame_descriptors(frames)
    core = sorted(set(int(index) for index in core_indices))
    if core[0] != 0 or core[-1] != descriptors.shape[0] - 1:
        raise ValueError(f"proxy interpolation requires endpoint anchors, got {core}")
    scores = [0.0] * descriptors.shape[0]
    for frame in range(descriptors.shape[0]):
        if frame in core:
            continue
        left, right = frame_neighbors(core, frame)
        alpha = (frame - left) / (right - left)
        estimate = descriptors[left].lerp(descriptors[right], alpha)
        denominator = descriptors[frame].norm().clamp_min(1e-8)
        scores[frame] = float(((descriptors[frame] - estimate).norm() / denominator).item())
    return scores


def clean_latent_proxy(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor | float | None = None,
    *,
    sigma: torch.Tensor | float | None = None,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """One-step clean-latent estimate for Wan's flow parameterization.

    Wan's UniPC scheduler converts a flow prediction with
    ``x0 = sample - sigma * model_output``.  Passing the scheduler's exact
    ``sigma`` is preferred; ``timestep`` remains available for callers that
    only have Wan's integer ``sigma * num_train_timesteps`` encoding.
    """

    if (timestep is None) == (sigma is None):
        raise ValueError("pass exactly one of timestep or sigma")
    if sigma is None:
        value = float(
            timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        )
        sigma_value = value / float(num_train_timesteps)
    else:
        sigma_value = float(sigma.item() if isinstance(sigma, torch.Tensor) else sigma)
    return latent - sigma_value * velocity


def fixed_middle_indices(num_frames: int, budget: int) -> list[int]:
    """Deterministic fixed-frame baseline with endpoint coverage."""

    if not 2 <= budget <= num_frames:
        raise ValueError(f"invalid budget {budget} for F={num_frames}")
    if budget == num_frames:
        return list(range(num_frames))
    if budget == 2:
        return [0, num_frames - 1]
    interior = budget - 2
    center = (num_frames - 1) / 2
    candidates = sorted(
        range(1, num_frames - 1), key=lambda frame: (abs(frame - center), frame)
    )[:interior]
    return sorted([0, *candidates, num_frames - 1])


def risk_weighted_indices(
    scores: Sequence[float], budget: int, *, current: Sequence[int] | None = None
) -> list[int]:
    return adaptive_mesh_indices(
        len(scores), budget, scores, current=current, distance_power=2.0, inertia=0.1
    )


__all__ = [
    "clean_latent_proxy",
    "cosine_novelty",
    "fixed_middle_indices",
    "frame_descriptors",
    "proxy_interpolation_scores",
    "rhyme_budgeted_indices",
    "rhyme_candidate_scores",
    "rhyme_prior_scores",
    "rhyme_sequential_threshold",
    "risk_weighted_indices",
    "uniform_indices",
]

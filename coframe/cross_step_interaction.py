"""Fail-closed analysis primitives for the cross-step endpoint screen."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from random import Random
from statistics import median
from typing import Any

import numpy as np
import torch

from .cross_step_endpoint import FROZEN_PAIRS, ORIENTATION_ARMS

PROMPT_IDS = tuple(f"p{i}_s0" for i in range(8))
ARM_IDS = (
    "k9_k9", "k6_k9", "k9_k12", "k6_k12",
    "k12_k9", "k9_k6", "k12_k6",
)
ARM_BUDGETS = {
    "k9_k9": (9, 9), "k6_k9": (6, 9), "k9_k12": (9, 12),
    "k6_k12": (6, 12), "k12_k9": (12, 9), "k9_k6": (9, 6),
    "k12_k6": (12, 6),
}
ORIENTATION_ARM_IDS = {
    "6_to_12": {"00": "k9_k9", "10": "k6_k9", "01": "k9_k12", "11": "k6_k12"},
    "12_to_6": {"00": "k9_k9", "10": "k12_k9", "01": "k9_k6", "11": "k12_k6"},
}
MAIN_ORIENTATION = "12_to_6"
SCALAR_RHO_THRESHOLD = 0.25
VECTOR_RHO_THRESHOLD = 0.10
REQUIRED_PROMPTS = 6
REQUIRED_PAIRS = 2
ALIGNMENT_RHO_THRESHOLD = 0.50
ALIGNMENT_SIGN_THRESHOLD = 0.75
EXPECTED_ORIENTATION_ROWS = 8 * 3 * 2
EXPECTED_SINGLETON_ROWS = 8 * 4 * 2

ADVANCE = "ADVANCE_SEQUENTIAL_PLANNER"
SUPPORT = "SUPPORT_CROSS_STEP_INTERACTION_NO_PLANNER_ADVANCE"
STATIC_PRIOR = "SUPPORT_STATIC_BUDGET_TRANSFER_PRIOR"
TEST_PLUS3 = "TEST_PLUS3_OBJECTIVE_LOPO"
REJECT = "REJECT_CURRENT_CROSS_STEP_EXPLANATION"
INCOMPLETE = "INCOMPLETE_CROSS_STEP_ENDPOINT_SCREEN"


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be {'nonnegative ' if nonnegative else ''}finite")
    return result


def normalized_mse(reference: torch.Tensor, approximation: torch.Tensor, *, chunk_size: int = 1_048_576) -> float:
    """FP64-chunked endpoint NMSE."""
    reference = _real_tensor(reference, "reference")
    approximation = _real_tensor(approximation, "approximation")
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have identical shapes")
    numerator = denominator = 0.0
    ref = reference.reshape(-1)
    app = approximation.reshape(-1)
    for start in range(0, ref.numel(), chunk_size):
        r = ref[start:start + chunk_size].double()
        a = app[start:start + chunk_size].double()
        numerator += float((a - r).square().sum())
        denominator += float(r.square().sum())
    return numerator / (denominator + 1e-12)


def temporal_gradient_error(reference: torch.Tensor, approximation: torch.Tensor, *, frame_dim: int = 2, chunk_size: int = 1_048_576) -> float:
    reference = _real_tensor(reference, "reference")
    approximation = _real_tensor(approximation, "approximation")
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have identical shapes")
    if not -reference.ndim <= frame_dim < reference.ndim:
        raise ValueError("frame_dim is outside tensor rank")
    frame_dim %= reference.ndim
    if reference.shape[frame_dim] < 2:
        raise ValueError("endpoint latent requires at least two temporal frames")
    return sqrt(normalized_mse(torch.diff(reference, dim=frame_dim), torch.diff(approximation, dim=frame_dim), chunk_size=chunk_size))


def _real_tensor(value: Any, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu()
    if value.numel() == 0 or not value.dtype.is_floating_point or value.dtype.is_complex:
        raise ValueError(f"{name} must be a non-empty real floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def scalar_factorial(errors: Mapping[str, Any], orientation: str) -> dict[str, float | bool]:
    if orientation not in ORIENTATION_ARM_IDS:
        raise ValueError(f"unknown orientation {orientation!r}")
    mapping = ORIENTATION_ARM_IDS[orientation]
    values = {cell: _finite(errors[arm], f"errors[{arm}]", nonnegative=True) for cell, arm in mapping.items()}
    e00, e10, e01, e11 = (values[cell] for cell in ("00", "10", "01", "11"))
    delta_i, delta_j, observed = e10 - e00, e01 - e00, e11 - e00
    additive = delta_i + delta_j
    interaction = observed - additive
    tau = max(1e-12, .01 * abs(e00))
    rho = abs(interaction) / (abs(delta_i) + abs(delta_j) + tau)
    sign_flip = additive * observed < 0 and abs(additive) > tau and abs(observed) > tau
    return {"e00": e00, "e10": e10, "e01": e01, "e11": e11,
            "delta_i": delta_i, "delta_j": delta_j, "observed": observed,
            "additive": additive, "interaction": interaction, "tau": tau,
            "rho": rho, "sign_flip": sign_flip}


def vector_factorial(states: Mapping[str, Any], orientation: str, *, chunk_size: int = 1_048_576) -> dict[str, float]:
    if orientation not in ORIENTATION_ARM_IDS:
        raise ValueError(f"unknown orientation {orientation!r}")
    mapping = ORIENTATION_ARM_IDS[orientation]
    tensors = {cell: _real_tensor(states[arm], f"states[{arm}]").reshape(-1) for cell, arm in mapping.items()}
    if len({tensor.numel() for tensor in tensors.values()}) != 1:
        raise ValueError("all factorial states must have equal size")

    def norm(coefficients: Sequence[tuple[float, str]]) -> float:
        total = 0.0
        for start in range(0, tensors["00"].numel(), chunk_size):
            value = sum(c * tensors[key][start:start + chunk_size].double() for c, key in coefficients)
            total += float(value.square().sum())
        return sqrt(total)

    ni = norm(((1, "10"), (-1, "00")))
    nj = norm(((1, "01"), (-1, "00")))
    njoint = norm(((1, "11"), (-1, "00")))
    ninteraction = norm(((1, "11"), (-1, "10"), (-1, "01"), (1, "00")))
    return {"norm_delta_i": ni, "norm_delta_j": nj, "norm_delta_joint": njoint,
            "norm_interaction": ninteraction, "rho": ninteraction / (ni + nj + 1e-12)}


def relative_improvement(baseline_error: Any, candidate_error: Any) -> float:
    baseline = _finite(baseline_error, "baseline_error", nonnegative=True)
    candidate = _finite(candidate_error, "candidate_error", nonnegative=True)
    return (baseline - candidate) / (baseline + 1e-12)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("rank input must be a non-empty finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x: Sequence[Any], y: Sequence[Any]) -> float:
    xv = [_finite(v, "x") for v in x]
    yv = [_finite(v, "y") for v in y]
    if len(xv) != len(yv) or len(xv) < 2:
        raise ValueError("Spearman inputs must have matching length >=2")
    xr, yr = _rankdata(xv), _rankdata(yv)
    if float(np.std(xr)) == 0 or float(np.std(yr)) == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def cluster_bootstrap_median(values: Mapping[str, float], *, replicates: int = 20_000, seed: int = 20260813) -> dict[str, float | int]:
    if set(values) != set(PROMPT_IDS):
        raise ValueError("cluster bootstrap requires exactly eight frozen prompt clusters")
    ordered = [_finite(values[p], f"value[{p}]") for p in PROMPT_IDS]
    rng = Random(seed)
    samples = sorted(median([ordered[rng.randrange(8)] for _ in range(8)]) for _ in range(replicates))
    return {"replicates": replicates, "seed": seed, "lower": samples[int(.025 * replicates)],
            "upper": samples[min(replicates - 1, int(.975 * replicates))]}


def aggregate_interactions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen primary and atomic-transfer endpoint gates."""
    expected = {(p, pair.pair_id, o) for p in PROMPT_IDS for pair in FROZEN_PAIRS for o in ORIENTATION_ARM_IDS}
    if len(records) != EXPECTED_ORIENTATION_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_ORIENTATION_ROWS} orientation rows")
    normalized = []
    seen = set()
    for index, row in enumerate(records):
        key = (str(row.get("prompt_id")), str(row.get("pair_id")), str(row.get("orientation")))
        if key in seen:
            raise ValueError(f"duplicate interaction row {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"unexpected interaction row {key}")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != {"endpoint_nmse", "temporal_gradient_error"}:
            raise ValueError(f"row {index} has incomplete endpoint metrics")
        entry = {"prompt_id": key[0], "pair_id": key[1], "orientation": key[2], "metrics": {}}
        for metric, payload in metrics.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"row {index}/{metric} must be an object")
            scalar = payload.get("scalar")
            vector = payload.get("vector")
            if not isinstance(scalar, Mapping):
                raise ValueError(f"row {index}/{metric} lacks scalar factorial")
            if metric == "endpoint_nmse" and not isinstance(vector, Mapping):
                raise ValueError(f"row {index}/{metric} lacks final-latent vector factorial")
            if metric != "endpoint_nmse" and vector is not None:
                raise ValueError(f"row {index}/{metric} must not duplicate final-latent vector factorial")
            entry["metrics"][metric] = {
                "rho_scalar": _finite(scalar.get("rho"), "rho_scalar", nonnegative=True),
                "sign_flip": _require_bool(scalar.get("sign_flip"), "sign_flip"),
                "joint_improvement": _finite(payload.get("joint_improvement"), "joint_improvement"),
            }
            if metric == "endpoint_nmse":
                entry["metrics"][metric]["rho_vector"] = _finite(vector.get("rho"), "rho_vector", nonnegative=True)
        normalized.append(entry)
    if seen != expected:
        raise ValueError(f"interaction grid incomplete: {sorted(expected - seen)[:3]}")

    primary_metrics: dict[str, Any] = {}
    atomic_metrics: dict[str, Any] = {}
    main = [row for row in normalized if row["orientation"] == MAIN_ORIENTATION]
    for metric in ("endpoint_nmse", "temporal_gradient_error"):
        by_prompt = {p: median(row["metrics"][metric]["rho_scalar"] for row in main if row["prompt_id"] == p) for p in PROMPT_IDS}
        pair_medians = {pair.pair_id: median(row["metrics"][metric]["rho_scalar"] for row in main if row["pair_id"] == pair.pair_id) for pair in FROZEN_PAIRS}
        overall = median(by_prompt.values())
        primary_metrics[metric] = {
            "overall_prompt_median_rho": overall,
            "prompt_pass_count": sum(v >= SCALAR_RHO_THRESHOLD for v in by_prompt.values()),
            "pair_pass_count": sum(v >= SCALAR_RHO_THRESHOLD for v in pair_medians.values()),
            "by_prompt": by_prompt, "by_pair": pair_medians,
            "passes_prompt_gate": overall >= SCALAR_RHO_THRESHOLD and sum(v >= SCALAR_RHO_THRESHOLD for v in by_prompt.values()) >= REQUIRED_PROMPTS,
        }
        improvements = {p: median(row["metrics"][metric]["joint_improvement"] for row in main if row["prompt_id"] == p) for p in PROMPT_IDS}
        atomic_metrics[metric] = {"overall_prompt_median_improvement": median(improvements.values()),
                                  "prompt_win_count": sum(v > 0 for v in improvements.values()),
                                  "by_prompt": improvements,
                                  "passes": median(improvements.values()) > 0 and sum(v > 0 for v in improvements.values()) >= REQUIRED_PROMPTS}
    vector_median = median(row["metrics"]["endpoint_nmse"]["rho_vector"] for row in main)
    primary_pass = all(v["passes_prompt_gate"] for v in primary_metrics.values()) and primary_metrics["endpoint_nmse"]["pair_pass_count"] >= REQUIRED_PAIRS and vector_median >= VECTOR_RHO_THRESHOLD
    atomic_pass = all(v["passes"] for v in atomic_metrics.values())
    return {"decision": SUPPORT if primary_pass else REJECT,
            "mechanism_label": "SUPPORT_CROSS_STEP_INTERACTION" if primary_pass else REJECT,
            "primary_gate": {"passes": primary_pass, "metrics": primary_metrics,
                             "endpoint_nmse_vector_rho_median": vector_median,
                             "endpoint_nmse_vector_passes": vector_median >= VECTOR_RHO_THRESHOLD,
                             "required_pair_passes": REQUIRED_PAIRS},
            "atomic_transfer_gate": {"passes": atomic_pass, "metrics": atomic_metrics},
            "reverse_orientation": _orientation_diagnostics(normalized, "6_to_12"),
            "sign_flip_diagnostics": _orientation_diagnostics(normalized, MAIN_ORIENTATION),
            "completeness": {"complete": True, "record_count": len(normalized), "expected": EXPECTED_ORIENTATION_ROWS}}


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _orientation_diagnostics(rows: Sequence[Mapping[str, Any]], orientation: str) -> dict[str, Any]:
    selected = [row for row in rows if row["orientation"] == orientation]
    result = {}
    for metric in ("endpoint_nmse", "temporal_gradient_error"):
        result[metric] = {
            "median_rho_scalar": median(row["metrics"][metric]["rho_scalar"] for row in selected),
            "sign_flip_count": sum(row["metrics"][metric]["sign_flip"] for row in selected),
        }
        if metric == "endpoint_nmse":
            result[metric]["median_rho_vector"] = median(row["metrics"][metric]["rho_vector"] for row in selected)
    return result


def aggregate_alignment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate local singleton effects against endpoint singleton effects."""
    expected = {(p, f"{cell.step}:{cell.group}", k) for p in PROMPT_IDS
                for cell in {pair.source for pair in FROZEN_PAIRS} | {pair.target for pair in FROZEN_PAIRS}
                for k in (6, 12)}
    if len(rows) != EXPECTED_SINGLETON_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_SINGLETON_ROWS} singleton rows")
    normalized, seen = [], set()
    required_fields = ("operator_effect", "propagation_h3_effect", "endpoint_nmse_effect", "temporal_gradient_effect")
    for row in rows:
        key = (str(row.get("prompt_id")), str(row.get("slot")), int(row.get("k", -1)))
        if key in seen:
            raise ValueError(f"duplicate singleton row {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"unexpected singleton row {key}")
        normalized.append({**dict(row), **{field: _finite(row.get(field), field) for field in required_fields}})
    if seen != expected:
        raise ValueError(f"singleton grid incomplete: {sorted(expected - seen)[:3]}")

    results = {}
    for local in ("operator_effect", "propagation_h3_effect"):
        results[local] = {}
        for endpoint in ("endpoint_nmse_effect", "temporal_gradient_effect"):
            by_prompt_rho, by_prompt_sign = {}, {}
            for prompt in PROMPT_IDS:
                subset = [row for row in normalized if row["prompt_id"] == prompt]
                by_prompt_rho[prompt] = spearman([r[local] for r in subset], [r[endpoint] for r in subset])
                by_prompt_sign[prompt] = sum(_same_effect_sign(r[local], r[endpoint]) for r in subset) / len(subset)
            overall_sign = sum(_same_effect_sign(r[local], r[endpoint]) for r in normalized) / len(normalized)
            ci = cluster_bootstrap_median(by_prompt_rho)
            passes = (median(by_prompt_rho.values()) >= ALIGNMENT_RHO_THRESHOLD and ci["lower"] > 0
                      and overall_sign >= ALIGNMENT_SIGN_THRESHOLD
                      and sum(v >= ALIGNMENT_SIGN_THRESHOLD for v in by_prompt_sign.values()) >= REQUIRED_PROMPTS)
            results[local][endpoint] = {"median_within_prompt_spearman": median(by_prompt_rho.values()),
                                       "by_prompt_spearman": by_prompt_rho, "prompt_cluster_bootstrap_ci": ci,
                                       "prompt_balanced_sign_agreement": overall_sign,
                                       "by_prompt_sign_agreement": by_prompt_sign,
                                       "prompt_sign_pass_count": sum(v >= ALIGNMENT_SIGN_THRESHOLD for v in by_prompt_sign.values()),
                                       "passes": passes}
        results[local]["aligned"] = all(results[local][endpoint]["passes"] for endpoint in ("endpoint_nmse_effect", "temporal_gradient_effect"))
    return {"operator_aligned": results["operator_effect"]["aligned"],
            "propagation_h3_aligned": results["propagation_h3_effect"]["aligned"],
            "secondary_only": True, "metrics": results,
            "completeness": {"complete": True, "row_count": len(normalized), "expected": EXPECTED_SINGLETON_ROWS}}


def _same_effect_sign(a: float, b: float) -> bool:
    # Zero agrees only with zero; it must not inflate the directional gate.
    return (a == 0 and b == 0) or (a * b > 0)


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_decision(interaction: Mapping[str, Any], alignment: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen action-label precedence after all independent gates."""
    primary = bool(interaction.get("primary_gate", {}).get("passes"))
    atomic = bool(interaction.get("atomic_transfer_gate", {}).get("passes"))
    plus3 = bool(alignment.get("propagation_h3_aligned"))
    if primary and atomic:
        decision = ADVANCE
        next_step = "design a separately preregistered sequential planner; planner NOT_RUN here"
    elif primary:
        decision = SUPPORT
        next_step = "do not advance a planner until a quality-preserving transfer is demonstrated"
    elif atomic:
        decision = STATIC_PRIOR
        next_step = "test the frozen static early-spend/late-save prior, not an adaptive planner"
    elif plus3:
        decision = TEST_PLUS3
        next_step = "run a separately preregistered plus3-objective LOPO experiment"
    else:
        decision = REJECT
        next_step = "stop the current cross-step explanation and complex-planner line"
    return {
        "decision": decision,
        "mechanism_label": "SUPPORT_CROSS_STEP_INTERACTION" if primary else None,
        "recommended_next_step": next_step,
        "primary_interaction_pass": primary,
        "atomic_transfer_pass": atomic,
        "operator_alignment_pass": bool(alignment.get("operator_aligned")),
        "plus3_alignment_pass": plus3,
    }

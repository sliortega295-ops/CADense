from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import median
from typing import Any

import torch


SCHEMA_VERSION = "coframe.trajectory-interaction-screen.v1"
EXPERIMENT_ID = "trajectory-interaction-screen-20260812"
BASE_TRAJECTORY_K = 9
CHECKPOINTS = ("after_j", "plus_3_dense", "step_end")
PROMPT_IDS = tuple(f"p{index}_s0" for index in range(8))

# The order is part of the preregistration and of the raw-trace contract.
ARM_IDS = (
    "k9_k9",
    "k6_k9",
    "k9_k12",
    "k6_k12",
    "k12_k9",
    "k9_k6",
    "k12_k6",
)
ARM_BUDGETS = {
    "k9_k9": (9, 9),
    "k6_k9": (6, 9),
    "k9_k12": (9, 12),
    "k6_k12": (6, 12),
    "k12_k9": (12, 9),
    "k9_k6": (9, 6),
    "k12_k6": (12, 6),
}


@dataclass(frozen=True, slots=True)
class Orientation:
    orientation_id: str
    arm_00: str
    arm_10: str
    arm_01: str
    arm_11: str


ORIENTATIONS = {
    "6_to_12": Orientation(
        orientation_id="6_to_12",
        arm_00="k9_k9",
        arm_10="k6_k9",
        arm_01="k9_k12",
        arm_11="k6_k12",
    ),
    "12_to_6": Orientation(
        orientation_id="12_to_6",
        arm_00="k9_k9",
        arm_10="k12_k9",
        arm_01="k9_k6",
        arm_11="k12_k6",
    ),
}


@dataclass(frozen=True, slots=True)
class PairSpec:
    pair_id: str
    step: int
    group_i: int
    group_j: int
    distance: str


PAIR_SPECS = (
    PairSpec("step05_g0_g1_adjacent", 5, 0, 1, "adjacent"),
    PairSpec("step05_g0_g7_long", 5, 0, 7, "long"),
    PairSpec("step20_g3_g4_adjacent", 20, 3, 4, "adjacent"),
    PairSpec("step20_g0_g7_long", 20, 0, 7, "long"),
    PairSpec("step40_g6_g7_adjacent", 40, 6, 7, "adjacent"),
    PairSpec("step40_g0_g7_long", 40, 0, 7, "long"),
)
PAIR_BY_ID = {pair.pair_id: pair for pair in PAIR_SPECS}

SCALAR_RHO_THRESHOLD = 0.25
VECTOR_RHO_THRESHOLD = 0.10
REQUIRED_PROMPT_PASSES = 6
REQUIRED_STEP_PASSES = 2
EXPECTED_ORIENTATION_RECORDS = len(PROMPT_IDS) * len(PAIR_SPECS) * len(ORIENTATIONS)

SUPPORT = "SUPPORT_STRONG_WITHIN_STEP_TRAJECTORY_INTERACTION"
LOCAL_ONLY = "LOCAL_STATE_DEPENDENCE_ONLY"
NO_STRONG = "NO_STRONG_WITHIN_STEP_INTERACTION"


def validate_pair_plan(payload: Mapping[str, Any]) -> tuple[PairSpec, ...]:
    """Validate the frozen screen plan and return its typed pair specs.

    This validator intentionally rejects scientifically plausible alternatives:
    changing a pair after seeing results would change the preregistered screen.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory interaction plan must be an object")
    required = {"schema_version", "experiment_id", "base_trajectory_k", "arms", "pairs"}
    if set(payload) != required:
        raise ValueError(f"trajectory interaction plan keys must be exactly {sorted(required)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID!r}")
    if type(payload["base_trajectory_k"]) is not int or payload["base_trajectory_k"] != BASE_TRAJECTORY_K:
        raise ValueError("base_trajectory_k must be exactly 9")
    if payload["arms"] != list(ARM_IDS):
        raise ValueError(f"arms must be exactly {list(ARM_IDS)}")

    raw_pairs = payload["pairs"]
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(PAIR_SPECS):
        raise ValueError("trajectory interaction plan must contain exactly six pairs")
    expected_keys = {"pair_id", "step", "group_i", "group_j", "distance"}
    normalized: list[PairSpec] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw_pairs):
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError(f"pair {index} keys must be exactly {sorted(expected_keys)}")
        pair_id = value["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen_ids:
            raise ValueError(f"pair {index} requires a unique non-empty pair_id")
        seen_ids.add(pair_id)
        step, group_i, group_j = value["step"], value["group_i"], value["group_j"]
        if any(type(item) is not int for item in (step, group_i, group_j)):
            raise ValueError(f"pair {pair_id}: step and group indices must be integers")
        if step not in {5, 20, 40} or not 0 <= group_i < group_j < 8:
            raise ValueError(f"pair {pair_id}: step/groups are outside the frozen domain")
        distance = value["distance"]
        expected_distance = "adjacent" if group_j - group_i == 1 else "long" if group_j - group_i == 7 else None
        if distance != expected_distance:
            raise ValueError(f"pair {pair_id}: distance label does not match its groups")
        normalized.append(PairSpec(pair_id, step, group_i, group_j, distance))

    if tuple(normalized) != PAIR_SPECS:
        raise ValueError("trajectory interaction pairs differ from the frozen six-pair plan")
    return tuple(normalized)


def _orientation(value: str | Orientation) -> Orientation:
    if isinstance(value, Orientation):
        expected = ORIENTATIONS.get(value.orientation_id)
        if expected != value:
            raise ValueError("orientation differs from the frozen arm mapping")
        return value
    try:
        return ORIENTATIONS[str(value)]
    except KeyError as error:
        raise ValueError(f"orientation must be one of {tuple(ORIENTATIONS)}") from error


def _finite_scalar(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be {'nonnegative and ' if nonnegative else ''}finite")
    return result


def scalar_interaction(
    arm_errors: Mapping[str, float],
    orientation: str | Orientation,
) -> dict[str, float | bool]:
    """Compute a scalar factorial interaction from common-reference errors."""
    spec = _orientation(orientation)
    required = (spec.arm_00, spec.arm_10, spec.arm_01, spec.arm_11)
    missing = [arm for arm in required if arm not in arm_errors]
    if missing:
        raise ValueError(f"missing scalar errors for arms {missing}")
    e00, e10, e01, e11 = (
        _finite_scalar(arm_errors[arm], name=f"error[{arm}]", nonnegative=True) for arm in required
    )
    delta_i = e10 - e00
    delta_j = e01 - e00
    observed = e11 - e00
    additive = delta_i + delta_j
    interaction = observed - additive
    tau = max(1.0e-12, 0.01 * abs(e00))
    denominator = abs(delta_i) + abs(delta_j) + tau
    rho = abs(interaction) / denominator
    sign_flip = additive * observed < 0.0 and abs(additive) > tau and abs(observed) > tau
    return {
        "e00": e00,
        "e10": e10,
        "e01": e01,
        "e11": e11,
        "delta_i": delta_i,
        "delta_j": delta_j,
        "observed": observed,
        "delta_joint": observed,
        "additive": additive,
        "additive_prediction": additive,
        "interaction": interaction,
        "I_scalar": interaction,
        "tau": tau,
        "denominator": denominator,
        "rho": rho,
        "rho_scalar": rho,
        "sign_flip": sign_flip,
    }


def _as_vector(value: Any, *, name: str) -> torch.Tensor:
    try:
        result = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be tensor-like") from error
    if result.numel() == 0 or not (result.dtype.is_floating_point or result.dtype.is_complex):
        raise ValueError(f"{name} must be a non-empty real floating-point tensor")
    if result.dtype.is_complex:
        raise ValueError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result.detach().reshape(-1)


def _squared_l2_chunked(terms: Sequence[tuple[float, torch.Tensor]], *, chunk_size: int) -> float:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    size = terms[0][1].numel()
    total = 0.0
    for start in range(0, size, chunk_size):
        stop = min(size, start + chunk_size)
        value = sum(coefficient * tensor[start:stop].to(torch.float64) for coefficient, tensor in terms)
        total += float(value.square().sum().item())
    return total


def vector_interaction(
    arm_states: Mapping[str, Any],
    orientation: str | Orientation,
    *,
    chunk_size: int = 1_048_576,
) -> dict[str, float]:
    """Compute the full-state vector interaction with float64 chunked norms."""
    spec = _orientation(orientation)
    required = (spec.arm_00, spec.arm_10, spec.arm_01, spec.arm_11)
    missing = [arm for arm in required if arm not in arm_states]
    if missing:
        raise ValueError(f"missing hidden states for arms {missing}")
    h00, h10, h01, h11 = (_as_vector(arm_states[arm], name=f"state[{arm}]") for arm in required)
    shapes = {tensor.numel() for tensor in (h00, h10, h01, h11)}
    if len(shapes) != 1:
        raise ValueError("all arm hidden states must have the same shape")

    norm_delta_i = sqrt(_squared_l2_chunked(((1.0, h10), (-1.0, h00)), chunk_size=chunk_size))
    norm_delta_j = sqrt(_squared_l2_chunked(((1.0, h01), (-1.0, h00)), chunk_size=chunk_size))
    norm_delta_joint = sqrt(_squared_l2_chunked(((1.0, h11), (-1.0, h00)), chunk_size=chunk_size))
    norm_interaction = sqrt(
        _squared_l2_chunked(((1.0, h11), (-1.0, h10), (-1.0, h01), (1.0, h00)), chunk_size=chunk_size)
    )
    denominator = norm_delta_i + norm_delta_j + 1.0e-12
    rho = norm_interaction / denominator
    return {
        "norm_delta_i": norm_delta_i,
        "norm_delta_j": norm_delta_j,
        "norm_delta_joint": norm_delta_joint,
        "norm_interaction": norm_interaction,
        "denominator": denominator,
        "rho": rho,
        "rho_vector": rho,
    }


def _rho(checkpoint: Mapping[str, Any], kind: str) -> float:
    value = checkpoint.get(kind)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint requires nested {kind!r} diagnostics")
    key = "rho_scalar" if kind == "scalar" else "rho_vector"
    if key not in value and "rho" not in value:
        raise ValueError(f"checkpoint {kind} diagnostics require {key}")
    return _finite_scalar(value.get(key, value.get("rho")), name=key, nonnegative=True)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_interaction_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen prompt-clustered gate to 96 orientation records.

    Incomplete or malformed inputs raise instead of silently becoming a
    scientific negative result.
    """
    if len(records) != EXPECTED_ORIENTATION_RECORDS:
        raise ValueError(f"expected exactly {EXPECTED_ORIENTATION_RECORDS} orientation records")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        try:
            prompt_id = str(record["prompt_id"])
            pair_id = str(record["pair_id"])
            orientation_id = str(record["orientation"])
            checkpoints = record["checkpoints"]
        except KeyError as error:
            raise ValueError(f"record {index} is missing {error.args[0]}") from error
        if prompt_id not in PROMPT_IDS:
            raise ValueError(f"record {index} has an unexpected prompt_id")
        if pair_id not in PAIR_BY_ID:
            raise ValueError(f"record {index} has an unexpected pair_id")
        if orientation_id not in ORIENTATIONS:
            raise ValueError(f"record {index} has an unexpected orientation")
        key = (prompt_id, pair_id, orientation_id)
        if key in seen:
            raise ValueError(f"duplicate orientation record {key}")
        seen.add(key)
        pair = PAIR_BY_ID[pair_id]
        for name, expected in (
            ("step", pair.step),
            ("group_i", pair.group_i),
            ("group_j", pair.group_j),
            ("distance", pair.distance),
        ):
            if record.get(name) != expected:
                raise ValueError(f"record {key} has {name}={record.get(name)!r}, expected {expected!r}")
        if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(CHECKPOINTS):
            raise ValueError(f"record {key} must contain exactly checkpoints {CHECKPOINTS}")
        checkpoint_values: dict[str, dict[str, float | bool]] = {}
        for checkpoint_name in CHECKPOINTS:
            checkpoint = checkpoints[checkpoint_name]
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"record {key} checkpoint {checkpoint_name} must be an object")
            scalar_rho = _rho(checkpoint, "scalar")
            vector_rho = _rho(checkpoint, "vector")
            scalar = checkpoint["scalar"]
            sign_flip = scalar.get("meaningful_sign_flip", scalar.get("sign_flip", False))
            if not isinstance(sign_flip, bool):
                raise ValueError(f"record {key} checkpoint {checkpoint_name} sign_flip must be boolean")
            checkpoint_values[checkpoint_name] = {
                "rho_scalar": scalar_rho,
                "rho_vector": vector_rho,
                "sign_flip": sign_flip,
            }
        normalized.append(
            {
                "prompt_id": prompt_id,
                "pair_id": pair_id,
                "step": pair.step,
                "distance": pair.distance,
                "orientation": orientation_id,
                "checkpoints": checkpoint_values,
            }
        )

    expected = {
        (prompt_id, pair.pair_id, orientation_id)
        for prompt_id in PROMPT_IDS
        for pair in PAIR_SPECS
        for orientation_id in ORIENTATIONS
    }
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"orientation record grid is incomplete; missing={missing[:3]}")

    prompt_aggregates: dict[str, dict[str, dict[str, float | int | bool]]] = {}
    for prompt_id in PROMPT_IDS:
        prompt_aggregates[prompt_id] = {}
        prompt_rows = [row for row in normalized if row["prompt_id"] == prompt_id]
        for checkpoint_name in CHECKPOINTS:
            scalar_values = [row["checkpoints"][checkpoint_name]["rho_scalar"] for row in prompt_rows]
            vector_values = [row["checkpoints"][checkpoint_name]["rho_vector"] for row in prompt_rows]
            sign_flips = [row["checkpoints"][checkpoint_name]["sign_flip"] for row in prompt_rows]
            prompt_aggregates[prompt_id][checkpoint_name] = {
                "record_count": len(prompt_rows),
                "median_rho_scalar": median(scalar_values),
                "median_rho_vector": median(vector_values),
                "sign_flip_rate": sum(sign_flips) / len(sign_flips),
            }

    def scalar_prompt_gate(checkpoint_name: str) -> dict[str, Any]:
        values = [
            float(prompt_aggregates[prompt_id][checkpoint_name]["median_rho_scalar"])
            for prompt_id in PROMPT_IDS
        ]
        pass_count = sum(value >= SCALAR_RHO_THRESHOLD for value in values)
        overall = median(values)
        return {
            "threshold": SCALAR_RHO_THRESHOLD,
            "overall_prompt_median": overall,
            "prompt_pass_count": pass_count,
            "required_prompt_pass_count": REQUIRED_PROMPT_PASSES,
            "passes": overall >= SCALAR_RHO_THRESHOLD and pass_count >= REQUIRED_PROMPT_PASSES,
        }

    after_j_gate = scalar_prompt_gate("after_j")
    plus_3_scalar_gate = scalar_prompt_gate("plus_3_dense")
    plus_3_vectors = [row["checkpoints"]["plus_3_dense"]["rho_vector"] for row in normalized]
    plus_3_vector_gate = {
        "threshold": VECTOR_RHO_THRESHOLD,
        "record_count": len(plus_3_vectors),
        "median": median(plus_3_vectors),
        "passes": median(plus_3_vectors) >= VECTOR_RHO_THRESHOLD,
    }

    def stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scalar_values = [row["checkpoints"]["plus_3_dense"]["rho_scalar"] for row in rows]
        vector_values = [row["checkpoints"]["plus_3_dense"]["rho_vector"] for row in rows]
        scalar_value, vector_value = median(scalar_values), median(vector_values)
        return {
            "record_count": len(rows),
            "median_rho_scalar": scalar_value,
            "median_rho_vector": vector_value,
            "passes": scalar_value >= SCALAR_RHO_THRESHOLD and vector_value >= VECTOR_RHO_THRESHOLD,
        }

    marginal_strata: dict[str, dict[str, Any]] = {}
    for orientation_id in ORIENTATIONS:
        marginal_strata[f"orientation:{orientation_id}"] = stratum(
            [row for row in normalized if row["orientation"] == orientation_id]
        )
    for distance in ("adjacent", "long"):
        marginal_strata[f"distance:{distance}"] = stratum(
            [row for row in normalized if row["distance"] == distance]
        )
    marginal_gate = all(value["passes"] for value in marginal_strata.values())

    step_strata = {
        str(step): stratum([row for row in normalized if row["step"] == step]) for step in (5, 20, 40)
    }
    step_pass_count = sum(value["passes"] for value in step_strata.values())
    step_gate = step_pass_count >= REQUIRED_STEP_PASSES

    strong = all(
        (
            after_j_gate["passes"],
            plus_3_scalar_gate["passes"],
            plus_3_vector_gate["passes"],
            marginal_gate,
            step_gate,
        )
    )
    decision = SUPPORT if strong else LOCAL_ONLY if after_j_gate["passes"] else NO_STRONG

    checkpoint_overall: dict[str, Any] = {}
    for checkpoint_name in CHECKPOINTS:
        scalars = [row["checkpoints"][checkpoint_name]["rho_scalar"] for row in normalized]
        vectors = [row["checkpoints"][checkpoint_name]["rho_vector"] for row in normalized]
        flips = [row["checkpoints"][checkpoint_name]["sign_flip"] for row in normalized]
        checkpoint_overall[checkpoint_name] = {
            "rho_scalar": _summary(scalars),
            "rho_vector": _summary(vectors),
            "sign_flip_rate": sum(flips) / len(flips),
        }

    return {
        "decision": decision,
        "completeness": {
            "prompt_count": len(PROMPT_IDS),
            "pair_count": len(PAIR_SPECS),
            "orientation_count": len(ORIENTATIONS),
            "orientation_record_count": len(normalized),
            "expected_orientation_record_count": EXPECTED_ORIENTATION_RECORDS,
            "complete": True,
        },
        "thresholds": {
            "rho_scalar": SCALAR_RHO_THRESHOLD,
            "rho_vector": VECTOR_RHO_THRESHOLD,
            "required_prompt_passes": REQUIRED_PROMPT_PASSES,
            "required_step_passes": REQUIRED_STEP_PASSES,
        },
        "gate": {
            "after_j_scalar": after_j_gate,
            "plus_3_dense_scalar": plus_3_scalar_gate,
            "plus_3_dense_vector": plus_3_vector_gate,
            "all_marginal_strata_pass": marginal_gate,
            "step_strata_pass_count": step_pass_count,
            "step_strata_pass": step_gate,
            "passes": strong,
        },
        "prompt_aggregates": prompt_aggregates,
        "checkpoint_overall": checkpoint_overall,
        "marginal_strata": marginal_strata,
        "step_strata": step_strata,
    }

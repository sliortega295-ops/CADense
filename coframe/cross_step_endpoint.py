from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .budget import schedule_key


SCHEMA_VERSION = "coframe.cross-step-endpoint-screen.v1"
EXPERIMENT_ID = "cross-step-endpoint-screen-20260813"
BASE_K = 9
BUDGETS = (6, 9, 12, 21)
TOTAL_STEPS = 50
WARMUP_STEPS = 5
GROUP_COUNT = 8


@dataclass(frozen=True, slots=True)
class Cell:
    step: int
    group: int

    @property
    def key(self) -> str:
        return schedule_key(self.step, self.group)


@dataclass(frozen=True, slots=True)
class Pair:
    pair_id: str
    source: Cell
    target: Cell


FROZEN_PAIRS = (
    Pair("step22_g0_to_step44_g5", Cell(22, 0), Cell(44, 5)),
    Pair("step22_g0_to_step47_g3", Cell(22, 0), Cell(47, 3)),
    Pair("step22_g0_to_step49_g2", Cell(22, 0), Cell(49, 2)),
)
PAIR_BY_ID = {pair.pair_id: pair for pair in FROZEN_PAIRS}

ORIENTATION_ARMS = {
    "6_to_12": {"00": (9, 9), "10": (6, 9), "01": (9, 12), "11": (6, 12)},
    "12_to_6": {"00": (9, 9), "10": (12, 9), "01": (9, 6), "11": (12, 6)},
}
FROZEN_ARMS = {
    "k9_k9": (9, 9),
    "k6_k9": (6, 9),
    "k9_k12": (9, 12),
    "k6_k12": (6, 12),
    "k12_k9": (12, 9),
    "k9_k6": (9, 6),
    "k12_k6": (12, 6),
}
FROZEN_ORIENTATIONS = {
    "6_to_12": {"h00": "k9_k9", "h10": "k6_k9", "h01": "k9_k12", "h11": "k6_k12"},
    "12_to_6": {"h00": "k9_k9", "h10": "k12_k9", "h01": "k9_k6", "h11": "k12_k6"},
}


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def full_k9_schedule() -> dict[str, int]:
    """Return the complete 360-slot K9 schedule rather than relying on fallback."""
    return {
        schedule_key(step, group): BASE_K
        for step in range(WARMUP_STEPS, TOTAL_STEPS)
        for group in range(GROUP_COUNT)
    }


def _cell_from_mapping(value: Mapping[str, Any], *, name: str) -> Cell:
    if set(value) != {"step", "group"}:
        raise ValueError(f"{name} must contain exactly step and group")
    if type(value["step"]) is not int or type(value["group"]) is not int:
        raise ValueError(f"{name} step/group must be integers")
    return Cell(int(value["step"]), int(value["group"]))


def _parse_pair(value: Mapping[str, Any]) -> Pair:
    # Accept only two unambiguous wire shapes so the runtime can consume the
    # preregistration without coupling its scientific checks to cosmetic keys.
    if set(value) == {"pair_id", "source", "target"}:
        return Pair(
            str(value["pair_id"]),
            _cell_from_mapping(value["source"], name="source"),
            _cell_from_mapping(value["target"], name="target"),
        )
    if set(value) == {"pair_id", "i", "j"}:
        return Pair(
            str(value["pair_id"]),
            _cell_from_mapping(value["i"], name="i"),
            _cell_from_mapping(value["j"], name="j"),
        )
    if set(value) == {"pair_id", "step_i", "group_i", "step_j", "group_j"}:
        return Pair(
            str(value["pair_id"]),
            Cell(int(value["step_i"]), int(value["group_i"])),
            Cell(int(value["step_j"]), int(value["group_j"])),
        )
    raise ValueError("each pair must use source/target or step_i/group_i/step_j/group_j fields")


def validate_runtime_plan(payload: Mapping[str, Any]) -> tuple[Pair, ...]:
    """Fail closed on the experiment identity and the three frozen causal pairs."""
    if not isinstance(payload, Mapping):
        raise ValueError("cross-step endpoint plan must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID!r}")
    if int(payload.get("base_trajectory_k", -1)) != BASE_K:
        raise ValueError("base_trajectory_k must be exactly 9")
    if tuple(int(value) for value in payload.get("budget_values", ())) != BUDGETS:
        raise ValueError("budget_values must be exactly [6,9,12,21]")
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list):
        raise ValueError("arms must be a list")
    arms = {
        str(item.get("id", item.get("arm_id"))): (int(item["k_i"]), int(item["k_j"]))
        for item in raw_arms
    }
    if arms != FROZEN_ARMS:
        raise ValueError("logical arms differ from the frozen seven-arm factorial")
    orientations = payload.get("orientations")
    if not isinstance(orientations, Mapping):
        raise ValueError("orientations must be an object")
    expected_roles = {"12_to_6": "primary", "6_to_12": "reverse_control"}
    if {str(name): mapping.get("role") for name, mapping in orientations.items()} != expected_roles:
        raise ValueError("orientation roles differ from primary/reverse-control contract")
    normalized_orientations = {
        str(name): {str(key): str(value) for key, value in mapping.items() if key != "role"}
        for name, mapping in orientations.items()
    }
    if normalized_orientations != FROZEN_ORIENTATIONS:
        raise ValueError("orientations differ from the frozen mappings")
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(FROZEN_PAIRS):
        raise ValueError("cross-step endpoint plan must contain exactly three pairs")
    pairs = tuple(_parse_pair(item) for item in raw_pairs)
    if pairs != FROZEN_PAIRS:
        raise ValueError(f"pairs differ from the frozen plan: {FROZEN_PAIRS!r}")
    return pairs


def build_physical_runs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deduplicate three seven-arm factorials into fifteen physical trajectories."""
    pairs = validate_runtime_plan(payload)
    base = full_k9_schedule()
    schedules_by_hash: dict[str, dict[str, int]] = {}
    physical_by_hash: dict[str, str] = {}
    logical_arm_map: dict[str, dict[str, str]] = {}

    def register(schedule: dict[str, int]) -> str:
        if set(schedule) != set(base):
            raise RuntimeError("physical schedule does not contain all 360 slots")
        if set(schedule.values()) - set(BUDGETS):
            raise RuntimeError("physical schedule contains a budget outside the frozen set")
        digest = _sha256_json(schedule)
        if digest not in physical_by_hash:
            run_id = f"sparse_{len(physical_by_hash):02d}_{digest[:12]}"
            physical_by_hash[digest] = run_id
            schedules_by_hash[digest] = schedule
        return physical_by_hash[digest]

    baseline_id = register(dict(base))
    for pair in pairs:
        pair_arms: dict[str, str] = {}
        for arm_id, (k_source, k_target) in FROZEN_ARMS.items():
                schedule = dict(base)
                schedule[pair.source.key] = int(k_source)
                schedule[pair.target.key] = int(k_target)
                pair_arms[arm_id] = register(schedule)
        logical_arm_map[pair.pair_id] = pair_arms

    physical_runs = []
    for digest, run_id in physical_by_hash.items():
        schedule = schedules_by_hash[digest]
        changed = {key: value for key, value in schedule.items() if value != BASE_K}
        physical_runs.append(
            {
                "run_id": run_id,
                "schedule_id": run_id,
                "method": "adaptive_k",
                "schedule": schedule,
                "schedule_sha256": digest,
                "changed_slots": changed,
                "budget_overrides": changed,
            }
        )
    physical_runs.sort(key=lambda item: item["run_id"])
    if len(physical_runs) != 15:
        raise RuntimeError(f"expected 15 unique sparse schedules, got {len(physical_runs)}")
    if physical_runs[0]["run_id"] != baseline_id:
        raise RuntimeError("K9 baseline must be the first physical sparse run")
    return {
        "schema_version": "coframe.cross-step-endpoint-runtime.v1",
        "experiment_id": EXPERIMENT_ID,
        "plan_sha256": _sha256_json(payload),
        "physical_runs": physical_runs,
        "logical_arm_map": logical_arm_map,
        "baseline_run_id": baseline_id,
        "parity_repeat_run_id": f"{baseline_id}_parity_repeat",
        "dense_run_id": "dense_reference",
    }


def validate_runtime_manifest(manifest: Mapping[str, Any]) -> None:
    runs = manifest.get("physical_runs")
    if not isinstance(runs, list) or len(runs) != 15:
        raise ValueError("runtime manifest requires exactly 15 sparse physical runs")
    ids = [str(item["run_id"]) for item in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("runtime manifest contains duplicate run IDs")
    schedules = []
    for item in runs:
        schedule = {str(key): int(value) for key, value in item["schedule"].items()}
        if set(schedule) != set(full_k9_schedule()):
            raise ValueError("runtime manifest schedule is incomplete")
        digest = _sha256_json(schedule)
        if item.get("schedule_sha256") != digest:
            raise ValueError("runtime manifest schedule hash mismatch")
        schedules.append(digest)
    if len(schedules) != len(set(schedules)):
        raise ValueError("runtime manifest physical schedules are not unique")
    logical = manifest.get("logical_arm_map")
    if not isinstance(logical, Mapping) or set(logical) != set(PAIR_BY_ID):
        raise ValueError("runtime manifest logical arm map differs from frozen pairs")
    valid_ids = set(ids)
    for pair_id, mapping in logical.items():
        expected = set(FROZEN_ARMS)
        if set(mapping) != expected or set(mapping.values()) - valid_ids:
            raise ValueError(f"logical arm map is invalid for {pair_id}")

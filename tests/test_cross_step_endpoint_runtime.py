import copy
import json
from pathlib import Path

import pytest
from coframe.cross_step_endpoint import (
    FROZEN_PAIRS,
    build_physical_runs,
    full_k9_schedule,
    validate_runtime_manifest,
    validate_runtime_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    candidates = (
        ROOT / "configs" / "cross_step_endpoint_screen.json",
        Path("/tmp/cross_step_endpoint_screen.json.draft"),
    )
    path = next((value for value in candidates if value.exists()), None)
    if path is None:
        raise RuntimeError("cross-step endpoint protocol fixture is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_plan_builds_fifteen_unique_complete_schedules_and_logical_map():
    manifest = build_physical_runs(_plan())
    validate_runtime_manifest(manifest)
    runs = manifest["physical_runs"]
    assert len(runs) == 15
    assert len({item["schedule_sha256"] for item in runs}) == 15
    assert all(set(item["schedule"]) == set(full_k9_schedule()) for item in runs)
    assert all(sum(value != 9 for value in item["schedule"].values()) <= 2 for item in runs)
    assert set(manifest["logical_arm_map"]) == {pair.pair_id for pair in FROZEN_PAIRS}
    for mapping in manifest["logical_arm_map"].values():
        assert set(mapping) == {
            "k9_k9", "k6_k9", "k9_k12", "k6_k12", "k12_k9", "k9_k6", "k12_k6"
        }
    baseline = next(item for item in runs if item["run_id"] == manifest["baseline_run_id"])
    assert baseline["budget_overrides"] == {}


def test_shared_source_singletons_are_deduplicated_but_targets_and_joints_are_pair_specific():
    manifest = build_physical_runs(_plan())
    maps = list(manifest["logical_arm_map"].values())
    assert len({mapping["k9_k9"] for mapping in maps}) == 1
    assert len({mapping["k6_k9"] for mapping in maps}) == 1
    assert len({mapping["k12_k9"] for mapping in maps}) == 1
    for arm in ("k9_k12", "k6_k12", "k9_k6", "k12_k6"):
        assert len({mapping[arm] for mapping in maps}) == 3


def test_runtime_plan_and_manifest_fail_closed_on_pair_arm_role_and_hash_changes():
    plan = _plan()
    changed_pair = copy.deepcopy(plan)
    changed_pair["pairs"][0]["i"]["step"] = 20
    with pytest.raises(ValueError, match="frozen plan"):
        validate_runtime_plan(changed_pair)

    changed_arm = copy.deepcopy(plan)
    changed_arm["arms"][1]["k_i"] = 12
    with pytest.raises(ValueError, match="logical arms"):
        validate_runtime_plan(changed_arm)

    changed_role = copy.deepcopy(plan)
    changed_role["orientations"]["12_to_6"]["role"] = "reverse_control"
    with pytest.raises(ValueError, match="roles"):
        validate_runtime_plan(changed_role)

    manifest = build_physical_runs(plan)
    manifest["physical_runs"][0]["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_runtime_manifest(manifest)

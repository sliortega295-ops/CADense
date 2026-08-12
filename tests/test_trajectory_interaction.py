import copy
import json
from pathlib import Path

import pytest
import torch

from coframe.trajectory_interaction import (
    LOCAL_ONLY,
    NO_STRONG,
    SUPPORT,
    aggregate_interaction_records,
    scalar_interaction,
    validate_pair_plan,
    vector_interaction,
)


ROOT = Path(__file__).resolve().parents[1]


def test_scalar_interaction_is_zero_for_an_additive_response():
    result = scalar_interaction(
        {"k9_k9": 1.0, "k6_k9": 2.0, "k9_k12": 3.0, "k6_k12": 4.0},
        "6_to_12",
    )
    assert result["delta_i"] == 1.0
    assert result["delta_j"] == 2.0
    assert result["observed"] == 3.0
    assert result["additive"] == 3.0
    assert result["I_scalar"] == 0.0
    assert result["rho_scalar"] == 0.0
    assert result["sign_flip"] is False


def test_scalar_interaction_reports_only_a_meaningful_sign_flip():
    result = scalar_interaction(
        {"k9_k9": 10.0, "k6_k9": 11.0, "k9_k12": 11.0, "k6_k12": 9.0},
        "6_to_12",
    )
    assert result["tau"] == pytest.approx(0.1)
    assert result["additive"] == 2.0
    assert result["observed"] == -1.0
    assert result["I_scalar"] == -3.0
    assert result["rho_scalar"] == pytest.approx(3.0 / 2.1)
    assert result["sign_flip"] is True

    below_tau = scalar_interaction(
        {"k9_k9": 10.0, "k6_k9": 10.02, "k9_k12": 10.02, "k6_k12": 9.95},
        "6_to_12",
    )
    assert below_tau["additive"] > 0.0 and below_tau["observed"] < 0.0
    assert below_tau["sign_flip"] is False


def test_reverse_orientation_uses_the_frozen_12_to_6_arms():
    result = scalar_interaction(
        {"k9_k9": 1.0, "k12_k9": 2.0, "k9_k6": 4.0, "k12_k6": 8.0},
        "12_to_6",
    )
    assert result["delta_i"] == 1.0
    assert result["delta_j"] == 3.0
    assert result["observed"] == 7.0
    assert result["I_scalar"] == 3.0


def test_vector_interaction_uses_full_state_factorial_residual():
    additive = vector_interaction(
        {
            "k9_k9": torch.tensor([3.0, 4.0]),
            "k6_k9": torch.tensor([4.0, 4.0]),
            "k9_k12": torch.tensor([3.0, 6.0]),
            "k6_k12": torch.tensor([4.0, 6.0]),
        },
        "6_to_12",
        chunk_size=1,
    )
    assert additive["norm_delta_i"] == pytest.approx(1.0)
    assert additive["norm_delta_j"] == pytest.approx(2.0)
    assert additive["rho_vector"] == pytest.approx(0.0)

    nonlinear = vector_interaction(
        {
            "k9_k9": [3.0, 4.0],
            "k6_k9": [4.0, 4.0],
            "k9_k12": [3.0, 6.0],
            "k6_k12": [5.0, 6.0],
        },
        "6_to_12",
        chunk_size=1,
    )
    assert nonlinear["norm_interaction"] == pytest.approx(1.0)
    assert nonlinear["rho_vector"] == pytest.approx(1.0 / (3.0 + 1.0e-12))


def test_pair_plan_validation_is_fail_closed():
    plan = json.loads((ROOT / "configs" / "trajectory_interaction_screen.json").read_text())
    pairs = validate_pair_plan(plan)
    assert len(pairs) == 6
    assert [(pair.step, pair.group_i, pair.group_j) for pair in pairs] == [
        (5, 0, 1),
        (5, 0, 7),
        (20, 3, 4),
        (20, 0, 7),
        (40, 6, 7),
        (40, 0, 7),
    ]

    changed_arm = copy.deepcopy(plan)
    changed_arm["arms"][-1] = "k21_k6"
    with pytest.raises(ValueError, match="arms must be exactly"):
        validate_pair_plan(changed_arm)

    reversed_groups = copy.deepcopy(plan)
    reversed_groups["pairs"][0]["group_i"] = 2
    with pytest.raises(ValueError, match="distance label|frozen"):
        validate_pair_plan(reversed_groups)

    changed_step = copy.deepcopy(plan)
    changed_step["pairs"][0]["step"] = 10
    with pytest.raises(ValueError, match="outside the frozen domain"):
        validate_pair_plan(changed_step)


PAIR_ROWS = (
    ("step05_g0_g1_adjacent", 5, 0, 1, "adjacent"),
    ("step05_g0_g7_long", 5, 0, 7, "long"),
    ("step20_g3_g4_adjacent", 20, 3, 4, "adjacent"),
    ("step20_g0_g7_long", 20, 0, 7, "long"),
    ("step40_g6_g7_adjacent", 40, 6, 7, "adjacent"),
    ("step40_g0_g7_long", 40, 0, 7, "long"),
)


def make_records(*, after_by_prompt, plus_scalar=0.4, plus_vector=0.2):
    records = []
    for prompt_index in range(8):
        for pair_id, step, group_i, group_j, distance in PAIR_ROWS:
            for orientation in ("6_to_12", "12_to_6"):
                checkpoints = {}
                for name, scalar_rho, vector_rho in (
                    ("after_j", after_by_prompt[prompt_index], 0.2),
                    ("plus_3_dense", plus_scalar, plus_vector),
                    ("step_end", 0.1, 0.05),
                ):
                    checkpoints[name] = {
                        "scalar": {"rho_scalar": scalar_rho, "sign_flip": False},
                        "vector": {"rho_vector": vector_rho},
                    }
                records.append(
                    {
                        "prompt_id": f"p{prompt_index}_s0",
                        "pair_id": pair_id,
                        "step": step,
                        "group_i": group_i,
                        "group_j": group_j,
                        "distance": distance,
                        "orientation": orientation,
                        "checkpoints": checkpoints,
                    }
                )
    return records


def test_prompt_clustered_gate_supports_only_when_all_frozen_conditions_pass():
    # Six prompt clusters pass after_j. The two low clusters contain as many
    # raw rows as any high cluster, so they cannot be hidden by pseudo-replication.
    records = make_records(after_by_prompt=[0.4] * 6 + [0.1] * 2)
    result = aggregate_interaction_records(records)
    assert result["decision"] == SUPPORT
    assert result["gate"]["after_j_scalar"]["prompt_pass_count"] == 6
    assert result["gate"]["after_j_scalar"]["overall_prompt_median"] == pytest.approx(0.4)
    assert result["gate"]["plus_3_dense_vector"]["record_count"] == 96
    assert result["gate"]["step_strata_pass_count"] == 3


def test_prompt_gate_does_not_replace_six_clusters_with_pooled_raw_rows():
    # Five complete high prompts already contribute 60/96 high raw records and
    # therefore make the pooled median pass. The frozen prompt-cluster gate
    # still correctly fails because it requires at least six prompt clusters.
    result = aggregate_interaction_records(
        make_records(after_by_prompt=[0.4] * 5 + [0.1] * 3, plus_scalar=0.4, plus_vector=0.2)
    )
    assert result["checkpoint_overall"]["after_j"]["rho_scalar"]["median"] == pytest.approx(0.4)
    assert result["gate"]["after_j_scalar"]["prompt_pass_count"] == 5
    assert result["gate"]["after_j_scalar"]["passes"] is False
    assert result["decision"] == NO_STRONG


def test_two_of_three_step_strata_are_sufficient_but_both_orientations_are_required():
    records = make_records(after_by_prompt=[0.4] * 8)
    for record in records:
        if record["step"] == 40:
            record["checkpoints"]["plus_3_dense"]["scalar"]["rho_scalar"] = 0.1
            record["checkpoints"]["plus_3_dense"]["vector"]["rho_vector"] = 0.05
    two_steps = aggregate_interaction_records(records)
    assert two_steps["gate"]["step_strata_pass_count"] == 2
    assert two_steps["decision"] == SUPPORT

    orientation_failure = make_records(after_by_prompt=[0.4] * 8)
    for record in orientation_failure:
        if record["orientation"] == "12_to_6":
            record["checkpoints"]["plus_3_dense"]["scalar"]["rho_scalar"] = 0.1
            record["checkpoints"]["plus_3_dense"]["vector"]["rho_vector"] = 0.05
    orientation_result = aggregate_interaction_records(orientation_failure)
    assert orientation_result["gate"]["plus_3_dense_scalar"]["passes"] is True
    assert orientation_result["gate"]["all_marginal_strata_pass"] is False
    assert orientation_result["decision"] == LOCAL_ONLY


def test_gate_distinguishes_local_only_from_no_strong_interaction():
    local = aggregate_interaction_records(
        make_records(after_by_prompt=[0.4] * 8, plus_scalar=0.1, plus_vector=0.05)
    )
    assert local["decision"] == LOCAL_ONLY
    assert local["gate"]["after_j_scalar"]["passes"] is True
    assert local["gate"]["plus_3_dense_scalar"]["passes"] is False

    no_strong = aggregate_interaction_records(
        make_records(after_by_prompt=[0.1] * 8, plus_scalar=0.4, plus_vector=0.2)
    )
    assert no_strong["decision"] == NO_STRONG
    assert no_strong["gate"]["after_j_scalar"]["passes"] is False


def test_aggregate_rejects_missing_duplicate_and_changed_pair_records():
    records = make_records(after_by_prompt=[0.4] * 8)
    with pytest.raises(ValueError, match="exactly 96"):
        aggregate_interaction_records(records[:-1])

    duplicate = copy.deepcopy(records)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_interaction_records(duplicate)

    changed = copy.deepcopy(records)
    changed[0]["group_j"] = 2
    with pytest.raises(ValueError, match="expected"):
        aggregate_interaction_records(changed)

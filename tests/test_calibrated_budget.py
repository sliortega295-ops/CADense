import pytest
import json
from pathlib import Path

from coframe.calibrated_budget import optimize_exact_budget_schedule
from coframe.config import CoFrameConfig


ROOT = Path(__file__).resolve().parents[1]


def test_exact_budget_optimizer_prefers_complementary_6_and_12():
    costs = {
        "5:0": {6: 0.0, 9: 5.0, 12: 9.0, 21: 20.0},
        "5:1": {6: 9.0, 9: 5.0, 12: 0.0, 21: 20.0},
    }
    result = optimize_exact_budget_schedule(costs, target_average_k=9.0)
    assert result.schedule == {"5:0": 6, "5:1": 12}
    assert result.average_k == 9.0
    assert result.objective == 0.0
    assert result.uniform_k9_objective == 10.0


def test_exact_budget_optimizer_rejects_changed_budget_set():
    costs = {"5:0": {6: 1.0, 9: 0.0, 12: 1.0}}
    with pytest.raises(ValueError, match="preregistered"):
        optimize_exact_budget_schedule(costs, budgets=(6, 9, 12))


def test_calibrated_budget_probe_contract_is_fail_closed():
    config = CoFrameConfig(
        method="adaptive_k",
        adaptive_k_policy="step_block",
        kv_mode="full_kv",
        calibrated_budget_probe_mode="surface",
    )
    config.validate(num_frames=21, num_blocks=30)
    with pytest.raises(ValueError, match="full_kv"):
        CoFrameConfig(
            method="adaptive_k",
            adaptive_k_policy="step_block",
            calibrated_budget_probe_mode="surface",
        ).validate(num_frames=21, num_blocks=30)


def test_trajectory_interaction_contract_is_fail_closed():
    plan = json.loads((ROOT / "configs" / "trajectory_interaction_screen.json").read_text())
    config = CoFrameConfig(
        method="adaptive_k",
        adaptive_k_policy="step_block",
        kv_mode="full_kv",
        trajectory_interaction_plan=plan,
    )
    config.validate(num_frames=21, num_blocks=30)
    with pytest.raises(ValueError, match="cannot be combined"):
        CoFrameConfig(
            method="adaptive_k",
            adaptive_k_policy="step_block",
            kv_mode="full_kv",
            trajectory_interaction_plan=plan,
            calibrated_budget_probe_mode="surface",
        ).validate(num_frames=21, num_blocks=30)
    with pytest.raises(ValueError, match="non-K9"):
        CoFrameConfig(
            method="adaptive_k",
            adaptive_k_policy="step_block",
            kv_mode="full_kv",
            trajectory_interaction_plan=plan,
            adaptive_k_schedule={"5:0": 12},
        ).validate(num_frames=21, num_blocks=30)
    with pytest.raises(ValueError, match="step_block"):
        CoFrameConfig(
            method="adaptive_k",
            adaptive_k_policy="mean_defect",
            adaptive_k_thresholds=(0.1, 0.2, 0.3),
            kv_mode="full_kv",
            calibrated_budget_probe_mode="current",
        ).validate(num_frames=21, num_blocks=30)

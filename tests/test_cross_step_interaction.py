import copy

import pytest
import torch

from coframe.cross_step_endpoint import FROZEN_PAIRS
from coframe.cross_step_interaction import (
    ADVANCE,
    REJECT,
    SUPPORT,
    TEST_PLUS3,
    aggregate_alignment,
    aggregate_interactions,
    apply_decision,
    normalized_mse,
    scalar_factorial,
    temporal_gradient_error,
    vector_factorial,
)


def test_endpoint_metrics_are_dense_referenced_and_finite():
    reference = torch.tensor([[[[[1.0]], [[2.0]], [[4.0]]]]])
    approximation = torch.tensor([[[[[1.0]], [[3.0]], [[4.0]]]]])
    assert normalized_mse(reference, approximation, chunk_size=1) == pytest.approx(1 / 21)
    assert temporal_gradient_error(reference, approximation, frame_dim=2, chunk_size=1) == pytest.approx((2 / 5) ** .5)
    with pytest.raises(ValueError, match="NaN/Inf"):
        normalized_mse(reference, torch.full_like(reference, float("nan")))


def test_scalar_factorial_additive_and_nonadditive_sign_flip():
    additive = scalar_factorial(
        {"k9_k9": 10, "k12_k9": 8, "k9_k6": 11, "k12_k6": 9}, "12_to_6"
    )
    assert additive["interaction"] == pytest.approx(0)
    assert additive["rho"] == pytest.approx(0)
    assert additive["sign_flip"] is False

    nonlinear = scalar_factorial(
        {"k9_k9": 10, "k12_k9": 8, "k9_k6": 11, "k12_k6": 12}, "12_to_6"
    )
    assert nonlinear["additive"] == pytest.approx(-1)
    assert nonlinear["observed"] == pytest.approx(2)
    assert nonlinear["interaction"] == pytest.approx(3)
    assert nonlinear["rho"] == pytest.approx(3 / 3.1)
    assert nonlinear["sign_flip"] is True


def test_vector_factorial_uses_fp64_chunked_full_latent_residual():
    states = {
        "k9_k9": torch.tensor([0.0, 0.0]),
        "k12_k9": torch.tensor([1.0, 0.0]),
        "k9_k6": torch.tensor([0.0, 2.0]),
        "k12_k6": torch.tensor([1.0, 2.0]),
    }
    assert vector_factorial(states, "12_to_6", chunk_size=1)["rho"] == pytest.approx(0)
    states["k12_k6"] = torch.tensor([2.0, 2.0])
    assert vector_factorial(states, "12_to_6", chunk_size=1)["rho"] == pytest.approx(1 / 3)


def make_interaction_records(*, main_rho=.4, reverse_rho=.1, main_joint=.1):
    rows = []
    for prompt in range(8):
        for pair in FROZEN_PAIRS:
            for orientation, rho in (("12_to_6", main_rho), ("6_to_12", reverse_rho)):
                rows.append({
                    "prompt_id": f"p{prompt}_s0",
                    "pair_id": pair.pair_id,
                    "orientation": orientation,
                    "metrics": {
                        "endpoint_nmse": {
                            "scalar": {"rho": rho, "sign_flip": False},
                            "vector": {"rho": .2 if orientation == "12_to_6" else .05},
                            "joint_improvement": main_joint if orientation == "12_to_6" else -.1,
                        },
                        "temporal_gradient_error": {
                            "scalar": {"rho": rho, "sign_flip": False},
                            "joint_improvement": main_joint if orientation == "12_to_6" else -.1,
                        },
                    },
                })
    return rows


def test_primary_and_atomic_gates_use_only_frozen_main_orientation():
    result = aggregate_interactions(make_interaction_records())
    assert result["primary_gate"]["passes"] is True
    assert result["atomic_transfer_gate"]["passes"] is True
    assert result["reverse_orientation"]["endpoint_nmse"]["median_rho_scalar"] == pytest.approx(.1)

    # Reverse-control strength cannot rescue a failed primary orientation.
    failed = aggregate_interactions(make_interaction_records(main_rho=.1, reverse_rho=.9))
    assert failed["primary_gate"]["passes"] is False


def test_interaction_aggregate_rejects_missing_duplicate_and_nonfinite():
    records = make_interaction_records()
    with pytest.raises(ValueError, match="exactly 48"):
        aggregate_interactions(records[:-1])
    duplicate = copy.deepcopy(records)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_interactions(duplicate)
    nonfinite = copy.deepcopy(records)
    nonfinite[0]["metrics"]["endpoint_nmse"]["scalar"]["rho"] = float("nan")
    with pytest.raises(ValueError, match="nonnegative finite"):
        aggregate_interactions(nonfinite)


def make_alignment_rows(*, reverse_endpoint=False):
    cells = sorted({pair.source for pair in FROZEN_PAIRS} | {pair.target for pair in FROZEN_PAIRS}, key=lambda c: (c.step, c.group))
    rows = []
    for prompt in range(8):
        index = 0
        for cell in cells:
            for k in (6, 12):
                index += 1
                value = float(index + prompt / 100)
                endpoint = -value if reverse_endpoint else value
                rows.append({
                    "prompt_id": f"p{prompt}_s0", "slot": cell.key, "k": k,
                    "operator_effect": value,
                    "propagation_h3_effect": value * 2,
                    "endpoint_nmse_effect": endpoint,
                    "temporal_gradient_effect": endpoint * 3,
                })
    return rows


def test_alignment_requires_all_64_unique_singletons_and_both_outcomes():
    result = aggregate_alignment(make_alignment_rows())
    assert result["operator_aligned"] is True
    assert result["propagation_h3_aligned"] is True
    for predictor in ("operator_effect", "propagation_h3_effect"):
        for outcome in ("endpoint_nmse_effect", "temporal_gradient_effect"):
            metric = result["metrics"][predictor][outcome]
            assert metric["median_within_prompt_spearman"] == pytest.approx(1)
            assert metric["prompt_cluster_bootstrap_ci"]["lower"] > 0
            assert metric["prompt_balanced_sign_agreement"] == pytest.approx(1)

    failed = aggregate_alignment(make_alignment_rows(reverse_endpoint=True))
    assert failed["operator_aligned"] is False
    assert failed["propagation_h3_aligned"] is False

    with pytest.raises(ValueError, match="exactly 64"):
        aggregate_alignment(make_alignment_rows()[:-1])
    duplicate = make_alignment_rows()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_alignment(duplicate)


def test_frozen_decision_precedence():
    aligned = {"operator_aligned": False, "propagation_h3_aligned": True}
    both = aggregate_interactions(make_interaction_records())
    assert apply_decision(both, aligned)["decision"] == ADVANCE

    primary_only = aggregate_interactions(make_interaction_records(main_joint=-.1))
    assert apply_decision(primary_only, aligned)["decision"] == SUPPORT

    neither = aggregate_interactions(make_interaction_records(main_rho=.1, main_joint=-.1))
    assert apply_decision(neither, aligned)["decision"] == TEST_PLUS3
    assert apply_decision(neither, {"operator_aligned": True, "propagation_h3_aligned": False})["decision"] == REJECT

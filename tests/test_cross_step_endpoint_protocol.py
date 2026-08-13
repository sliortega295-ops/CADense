import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "cross_step_endpoint_screen.json"
DOC_PATH = ROOT / "docs" / "CROSS_STEP_ENDPOINT_SCREEN.md"
EXPECTED_ARMS = [
    ("k9_k9", 9, 9),
    ("k6_k9", 6, 9),
    ("k9_k12", 9, 12),
    ("k6_k12", 6, 12),
    ("k12_k9", 12, 9),
    ("k9_k6", 9, 6),
    ("k12_k6", 12, 6),
]
EXPECTED_PAIRS = [
    ("step22_g0_to_step44_g5", (22, 0), (44, 5)),
    ("step22_g0_to_step47_g3", (22, 0), (47, 3)),
    ("step22_g0_to_step49_g2", (22, 0), (49, 2)),
]
EXPECTED_PROMPTS = [
    "A red toy car moves smoothly from left to right across a wooden table.",
    "A red toy car makes a sharp U-turn around a blue cube.",
    "A gymnast performs a fast cartwheel and lands upright.",
    "Two dancers cross paths, exchange positions, and continue in opposite directions.",
    "A brown dog jumps to catch a small yellow ball thrown through the air.",
    "A hummingbird rapidly moves between two red flowers while the camera remains fixed.",
    "The camera circles around a stationary bronze statue in a plaza.",
    "A cup sits on a table while steam rises slowly; the rest of the scene remains still.",
]


def load_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_frozen_cross_step_plan_schema_and_factorials():
    plan = load_plan()
    assert plan["schema_version"] == "coframe.cross-step-endpoint-screen.v1"
    assert plan["experiment_id"] == "cross-step-endpoint-screen-20260813"
    assert plan["base_trajectory_k"] == 9
    assert plan["budget_values"] == [6, 9, 12, 21]
    assert [
        (item["arm_id"], item["k_i"], item["k_j"])
        for item in plan["arms"]
    ] == EXPECTED_ARMS
    assert plan["orientations"] == {
        "12_to_6": {
            "role": "primary",
            "h00": "k9_k9",
            "h10": "k12_k9",
            "h01": "k9_k6",
            "h11": "k12_k6",
        },
        "6_to_12": {
            "role": "reverse_control",
            "h00": "k9_k9",
            "h10": "k6_k9",
            "h01": "k9_k12",
            "h11": "k6_k12",
        },
    }
    assert [
        (
            item["pair_id"],
            (item["i"]["step"], item["i"]["group"]),
            (item["j"]["step"], item["j"]["group"]),
        )
        for item in plan["pairs"]
    ] == EXPECTED_PAIRS


def test_frozen_cross_step_prompt_runtime_and_completeness_contracts():
    plan = load_plan()
    assert [(item["prompt_id"], item["text"]) for item in plan["prompts"]] == [
        (f"p{index}_s0", prompt) for index, prompt in enumerate(EXPECTED_PROMPTS)
    ]
    assert plan["seed"] == 0
    runtime = plan["runtime_contract"]
    assert (runtime["height"], runtime["width"], runtime["decoded_frames"]) == (480, 832, 81)
    assert runtime["latent_frames"] == 21
    assert runtime["denoising_steps"] == 50
    assert runtime["dense_warmup_steps"] == 5
    assert (runtime["sparse_block_start_inclusive"], runtime["sparse_block_end_exclusive"]) == (3, 27)
    assert runtime["block_group_size"] == 3
    assert runtime["kv_mode"] == "full_kv"
    assert runtime["interpolation_target"] == "delta"
    assert runtime["anchor_selection"] == "uniform_select"
    assert runtime["force_boundaries"] is True
    assert runtime["assigned_physical_gpus"] == [1, 2, 3, 4]
    assert runtime["ddp"] is False and runtime["latency"] == "NOT_REPORTED"

    complete = plan["completeness"]
    assert complete["logical_arm_records"] == 8 * 3 * 7 == 168
    assert complete["orientation_records"] == 8 * 3 * 2 == 48
    assert complete["unique_physical_schedules_per_prompt"] == 1 + 2 + 3 * (2 + 2) == 15
    assert complete["unique_physical_schedules_total"] == 120
    assert complete["dense_reference_runs"] == 8


def test_frozen_cross_step_fairness_surface_and_gate_contracts():
    plan = load_plan()
    arms = {item["arm_id"]: (item["k_i"], item["k_j"]) for item in plan["arms"]}
    for joint in ("k6_k12", "k12_k6"):
        assert sum(arms[joint]) == 2 * plan["base_trajectory_k"]

    surface = plan["surface_input"]
    assert surface["external_path"] == "inputs/calibrated_step_block/budget_error_surface.csv"
    assert surface["sha256"] == "de0c409905a0f77b341001559edb6bb10ee0750cf2fab66f12f25528a63819b5"
    assert surface["join_key"] == ["prompt_id", "step", "group", "k"]
    assert surface["expected_rows"] == 11520
    assert surface["required_metric_columns"] == [
        "operator_nmse",
        "propagation_h3_relative_l2",
    ]

    gate = plan["gate"]
    assert gate["primary_orientation"] == "12_to_6"
    assert gate["reverse_control_orientation"] == "6_to_12"
    interaction = gate["interaction"]
    assert interaction["scalar_metrics"] == [
        "endpoint_nmse",
        "endpoint_temporal_gradient_relative_l2",
    ]
    assert interaction["rho_threshold"] == 0.25
    assert interaction["prompt_pass_min"] == 6
    assert interaction["endpoint_nmse_pair_median_pass_min"] == 2
    assert interaction["final_latent_vector_rho_pooled_median_min"] == 0.10
    assert interaction["reverse_control_can_rescue"] is False
    alignment = gate["objective_alignment"]
    assert alignment["deduplicated_singleton_effect_count"] == 64
    assert alignment["predictors"] == [
        "operator_nmse_delta",
        "propagation_h3_relative_l2_delta",
    ]
    assert alignment["outcomes"] == [
        "endpoint_nmse_delta",
        "endpoint_temporal_gradient_relative_l2_delta",
    ]
    assert alignment["median_within_prompt_spearman_min"] == 0.5
    assert alignment["prompt_cluster_bootstrap_ci_lower_strictly_positive"] is True
    assert alignment["prompt_balanced_sign_agreement_min"] == 0.75
    assert alignment["prompt_sign_agreement_min"] == 0.75
    assert alignment["prompt_sign_pass_min"] == 6
    assert alignment["operator_only_aligned_can_rescue_old_schedule"] is False
    assert plan["decision_labels"] == [
        "ADVANCE_SEQUENTIAL_PLANNER",
        "SUPPORT_CROSS_STEP_INTERACTION_NO_PLANNER_ADVANCE",
        "SUPPORT_STATIC_BUDGET_TRANSFER_PRIOR",
        "TEST_PLUS3_OBJECTIVE_LOPO",
        "REJECT_CURRENT_CROSS_STEP_EXPLANATION",
        "INCOMPLETE_CROSS_STEP_ENDPOINT_SCREEN",
    ]


def test_protocol_hash_manifest_matches_frozen_files():
    entries = {}
    for line in (ROOT / "configs" / "CROSS_STEP_ENDPOINT_PROTOCOL.sha256").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        entries[relative] = digest
    assert set(entries) == {
        "configs/cross_step_endpoint_screen.json",
        "docs/CROSS_STEP_ENDPOINT_SCREEN.md",
    }
    for relative, expected in entries.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected

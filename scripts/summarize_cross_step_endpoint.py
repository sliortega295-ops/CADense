#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import torch

from coframe.cross_step_endpoint import (
    FROZEN_ARMS,
    FROZEN_PAIRS,
    build_physical_runs,
    full_k9_schedule,
    validate_runtime_manifest,
    validate_runtime_plan,
)
from coframe.cross_step_interaction import (
    ARM_IDS,
    ARM_BUDGETS,
    INCOMPLETE,
    ORIENTATION_ARM_IDS,
    PROMPT_IDS,
    aggregate_alignment,
    aggregate_interactions,
    apply_decision,
    normalized_mse,
    relative_improvement,
    scalar_factorial,
    sha256_file,
    temporal_gradient_error,
    vector_factorial,
)

LATENT_SCHEMA = "coframe.cross-step-endpoint-latents.v1"
SUMMARY_SCHEMA = "coframe.cross-step-endpoint-run-summary.v1"
SURFACE_SHA256 = "de0c409905a0f77b341001559edb6bb10ee0750cf2fab66f12f25528a63819b5"


def _canonical_sha(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = json.dumps(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(header + tensor.numpy().tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    _check_json_finite(payload, str(path))
    return payload


def _check_json_finite(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_json_finite(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_finite(item, f"{name}[{index}]")


def _load_latent(path: Path, *, expected_prompt: str, expected_run: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing latent: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch <2.6 compatibility.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != LATENT_SCHEMA:
        raise ValueError(f"{path}: latent schema mismatch")
    if payload.get("prompt_id") != expected_prompt or payload.get("run_id") != expected_run:
        raise ValueError(f"{path}: prompt/run identity mismatch")
    tensor = payload.get("latents")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise ValueError(f"{path}: latents must be FP32 tensor")
    if list(tensor.shape) != [1, 16, 21, 60, 104] or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{path}: latent shape or finiteness mismatch")
    if payload.get("final_latent_sha256") != _tensor_sha(tensor):
        raise ValueError(f"{path}: final latent fingerprint mismatch")
    return tensor, {key: value for key, value in payload.items() if key != "latents"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_surface(path: Path, contract: dict[str, Any]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    expected_sha = str(contract.get("sha256"))
    if expected_sha != SURFACE_SHA256 or sha256_file(path) != expected_sha:
        raise ValueError("Phase-A surface SHA256 mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = list(contract.get("schema_columns", []))
        if reader.fieldnames != expected_columns:
            raise ValueError("Phase-A surface schema/order mismatch")
        raw = list(reader)
    if len(raw) != 11_520 or contract.get("expected_rows") != 11_520:
        raise ValueError(f"Phase-A surface requires exactly 11520 rows, found {len(raw)}")
    expected_grid = {
        (prompt, step, group, k)
        for prompt in PROMPT_IDS for step in range(5, 50) for group in range(8) for k in (6, 9, 12, 21)
    }
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    numeric = (
        "operator_nmse", "operator_relative_l2", "operator_reference_energy",
        "propagation_h3_nmse", "propagation_h3_relative_l2", "propagation_h3_reference_energy",
    )
    for index, item in enumerate(raw, start=2):
        try:
            prompt, step, group, k = item["prompt_id"], int(item["step"]), int(item["group"]), int(item["k"])
            key = (prompt, step, group, k)
            if key in rows:
                raise ValueError(f"duplicate Phase-A join key {key}")
            if item["slot"] != f"{step}:{group}":
                raise ValueError(f"row {index}: slot mismatch")
            if (int(item["block_start"]), int(item["block_end"])) != (3 + 3 * group, 5 + 3 * group):
                raise ValueError(f"row {index}: group block bounds mismatch")
            anchors = json.loads(item["anchors"])
            if (not isinstance(anchors, list) or len(anchors) != k or anchors != sorted(set(anchors))
                    or anchors[0] != 0 or anchors[-1] != 20):
                raise ValueError(f"row {index}: anchor set mismatch")
            parsed = {name: float(item[name]) for name in numeric}
            if any(not math.isfinite(value) or value < 0 for value in parsed.values()):
                raise ValueError(f"row {index}: non-finite/negative metric")
            if not item["trace_path"]:
                raise ValueError(f"row {index}: empty provenance path")
            rows[key] = {**item, **parsed, "anchors": anchors}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Phase-A row {index}: {exc}") from exc
    if set(rows) != expected_grid:
        raise ValueError(f"Phase-A surface grid mismatch; missing={len(expected_grid - set(rows))}, extra={len(set(rows) - expected_grid)}")
    return rows


def _validate_plan(plan: dict[str, Any], plan_path: Path) -> tuple[dict[str, Any], str]:
    validate_runtime_plan(plan)
    if {item["prompt_id"] for item in plan.get("prompts", [])} != set(PROMPT_IDS):
        raise ValueError("plan prompt IDs differ from frozen eight")
    if plan.get("seed") != 0:
        raise ValueError("plan seed must be zero")
    source = plan.get("surface_input")
    if not isinstance(source, dict):
        raise ValueError("plan lacks frozen surface contract")
    protocol_manifest = plan_path.parent / "CROSS_STEP_ENDPOINT_PROTOCOL.sha256"
    if not protocol_manifest.is_file():
        raise ValueError("missing protocol checksum manifest")
    expected_manifest_lines = {
        f"{sha256_file(plan_path)}  configs/cross_step_endpoint_screen.json",
        f"{sha256_file(plan_path.parents[0].parent / 'docs' / 'CROSS_STEP_ENDPOINT_SCREEN.md')}  docs/CROSS_STEP_ENDPOINT_SCREEN.md",
    }
    actual_lines = {line.strip() for line in protocol_manifest.read_text().splitlines() if line.strip()}
    if actual_lines != expected_manifest_lines:
        raise ValueError("protocol checksum manifest mismatch")
    return source, _canonical_sha(plan)


def _run_payload(prompt_root: Path, prompt: str, run_id: str) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    run_dir = prompt_root / "runs" / run_id
    summary = _read_json(run_dir / "summary.json")
    trace = _read_json(run_dir / "trace.json")
    tensor, latent_meta = _load_latent(run_dir / "latents.pt", expected_prompt=prompt, expected_run=run_id)
    if summary.get("schema_version") != SUMMARY_SCHEMA or summary.get("status") != "success" or not summary.get("finite"):
        raise ValueError(f"{prompt}/{run_id}: summary status/schema failure")
    if summary.get("latent_shape") != [1, 16, 21, 60, 104] or summary.get("latent_frame_dim") != 2:
        raise ValueError(f"{prompt}/{run_id}: summary latent contract failure")
    identity_fields = (
        "prompt_id", "run_id", "kind", "initial_latent_sha256", "final_latent_sha256",
        "physical_schedule_sha256", "source_commit", "model_id", "model_fingerprint",
        "runtime_config_sha256", "source_fingerprint", "plan_sha256",
    )
    if any(summary.get(key) != latent_meta.get(key) for key in identity_fields):
        raise ValueError(f"{prompt}/{run_id}: summary/latent fingerprints mismatch")
    return tensor, summary, trace


def analyze(root: Path, plan_path: Path, surface_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = _read_json(plan_path)
    surface_contract, plan_sha = _validate_plan(plan, plan_path)
    surface = _load_surface(surface_path, surface_contract)
    protocol_manifest_sha = sha256_file(plan_path.parent / "CROSS_STEP_ENDPOINT_PROTOCOL.sha256")
    prompt_text = {item["prompt_id"]: item["text"] for item in plan["prompts"]}

    arms_out: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    singletons: list[dict[str, Any]] = []
    global_commits: set[str] = set()
    global_models: set[str] = set()
    global_initials: set[str] = set()

    for prompt in PROMPT_IDS:
        prompt_root = root / prompt
        manifest = _read_json(prompt_root / "runtime_manifest.json")
        validate_runtime_manifest(manifest)
        if manifest.get("plan_sha256") != plan_sha or manifest.get("protocol_manifest_sha256") != protocol_manifest_sha:
            raise ValueError(f"{prompt}: runtime plan/protocol fingerprint mismatch")
        expected_manifest = build_physical_runs(plan)
        for field in ("physical_runs", "logical_arm_map", "baseline_run_id", "parity_repeat_run_id", "dense_run_id"):
            if manifest.get(field) != expected_manifest.get(field):
                raise ValueError(f"{prompt}: runtime manifest field {field} differs from frozen plan")
        logical_sidecar = _read_json(prompt_root / "logical_arm_map.json")
        if logical_sidecar != manifest["logical_arm_map"]:
            raise ValueError(f"{prompt}: logical-arm sidecar mismatch")
        status = _read_json(prompt_root / "run_status.json")
        expected_runs = {item["run_id"] for item in manifest["physical_runs"]} | {manifest["dense_run_id"], manifest["parity_repeat_run_id"]}
        if status.get("status") != "success" or set(status.get("completed", [])) != expected_runs or status.get("failed"):
            raise ValueError(f"{prompt}: incomplete run status")
        parity = status.get("parity", {})
        if parity.get("torch_equal") is not True or parity.get("max_abs_difference") != 0.0:
            raise ValueError(f"{prompt}: parity status failed")
        source = _read_json(prompt_root / "source.json")
        if source.get("plan_sha256") != plan_sha or source.get("runtime_invariants", {}).get("prompt") != prompt_text[prompt]:
            raise ValueError(f"{prompt}: source contract mismatch")

        tensors: dict[str, torch.Tensor] = {}
        summaries: dict[str, dict[str, Any]] = {}
        traces: dict[str, Any] = {}
        for run_id in sorted(expected_runs):
            tensors[run_id], summaries[run_id], traces[run_id] = _run_payload(prompt_root, prompt, run_id)
        source_fps = {summary["source_fingerprint"] for summary in summaries.values()}
        if source_fps != {status.get("source_fingerprint")} or len(source_fps) != 1:
            raise ValueError(f"{prompt}: source fingerprint mismatch across physical runs")
        initial_fps = {summary["initial_latent_sha256"] for summary in summaries.values()}
        if initial_fps != {source.get("initial_latent_sha256")}:
            raise ValueError(f"{prompt}: initial latent fingerprint mismatch")
        global_initials |= initial_fps
        global_commits |= {summary["source_commit"] for summary in summaries.values()}
        global_models |= {summary["model_fingerprint"] for summary in summaries.values()}
        baseline_id, dense_id, parity_id = manifest["baseline_run_id"], manifest["dense_run_id"], manifest["parity_repeat_run_id"]
        if not torch.equal(tensors[baseline_id], tensors[parity_id]):
            raise ValueError(f"{prompt}: analyzer parity tensor mismatch")
        dense = tensors[dense_id]

        physical_errors: dict[str, dict[str, float]] = {}
        for item in manifest["physical_runs"]:
            run_id = item["run_id"]
            schedule_path = prompt_root / "runs" / run_id / "physical_schedule.json"
            deployed = _read_json(schedule_path)
            if deployed != item["schedule"] or _canonical_sha(deployed) != item["schedule_sha256"]:
                raise ValueError(f"{prompt}/{run_id}: deployed schedule/hash mismatch")
            audit = summaries[run_id].get("cfg_schedule_audit", {})
            if not (audit.get("conditional_schedule_matches_manifest") and audit.get("unconditional_replay_runtime_assertion_passed")):
                raise ValueError(f"{prompt}/{run_id}: CFG schedule audit failed")
            physical_errors[run_id] = {
                "endpoint_nmse": normalized_mse(dense, tensors[run_id]),
                "temporal_gradient_error": temporal_gradient_error(dense, tensors[run_id], frame_dim=2),
            }

        for pair in FROZEN_PAIRS:
            arm_runs = manifest["logical_arm_map"][pair.pair_id]
            if set(arm_runs) != set(ARM_IDS):
                raise ValueError(f"{prompt}/{pair.pair_id}: logical arms incomplete")
            arm_tensors: dict[str, torch.Tensor] = {}
            arm_errors: dict[str, dict[str, float]] = {}
            for arm in ARM_IDS:
                run_id = arm_runs[arm]
                run_spec = next(item for item in manifest["physical_runs"] if item["run_id"] == run_id)
                expected_schedule = full_k9_schedule()
                expected_schedule[pair.source.key], expected_schedule[pair.target.key] = ARM_BUDGETS[arm]
                if run_spec["schedule"] != expected_schedule:
                    raise ValueError(f"{prompt}/{pair.pair_id}/{arm}: physical reuse/schedule mismatch")
                arm_tensors[arm] = tensors[run_id]
                arm_errors[arm] = physical_errors[run_id]
                arms_out.append({
                    "prompt_id": prompt, "pair_id": pair.pair_id, "arm_id": arm,
                    "physical_run_id": run_id, "source_slot": pair.source.key,
                    "target_slot": pair.target.key, "k_source": ARM_BUDGETS[arm][0],
                    "k_target": ARM_BUDGETS[arm][1], **arm_errors[arm],
                })
            for orientation in ORIENTATION_ARM_IDS:
                metrics: dict[str, Any] = {}
                for metric in ("endpoint_nmse", "temporal_gradient_error"):
                    scalar = scalar_factorial({arm: values[metric] for arm, values in arm_errors.items()}, orientation)
                    payload: dict[str, Any] = {
                        "scalar": scalar,
                        "joint_improvement": relative_improvement(
                            arm_errors[ORIENTATION_ARM_IDS[orientation]["00"]][metric],
                            arm_errors[ORIENTATION_ARM_IDS[orientation]["11"]][metric],
                        ),
                    }
                    if metric == "endpoint_nmse":
                        payload["vector"] = vector_factorial(arm_tensors, orientation)
                    metrics[metric] = payload
                interactions.append({"prompt_id": prompt, "pair_id": pair.pair_id,
                                     "orientation": orientation, "metrics": metrics})

        # Singleton endpoint effects are deduplicated by physical slot and K.
        unique_cells = sorted({pair.source for pair in FROZEN_PAIRS} | {pair.target for pair in FROZEN_PAIRS}, key=lambda cell: (cell.step, cell.group))
        baseline_errors = physical_errors[baseline_id]
        physical_by_changed = {
            tuple(sorted(item["changed_slots"].items())): item["run_id"] for item in manifest["physical_runs"]
        }
        for cell in unique_cells:
            for k in (6, 12):
                run_id = physical_by_changed.get(((cell.key, k),))
                if run_id is None:
                    raise ValueError(f"{prompt}/{cell.key}/K{k}: singleton physical run missing")
                local_k = surface[(prompt, cell.step, cell.group, k)]
                local_9 = surface[(prompt, cell.step, cell.group, 9)]
                singletons.append({
                    "prompt_id": prompt, "slot": cell.key, "step": cell.step, "group": cell.group,
                    "k": k, "physical_run_id": run_id,
                    "operator_effect": local_k["operator_nmse"] - local_9["operator_nmse"],
                    "propagation_h3_effect": local_k["propagation_h3_relative_l2"] - local_9["propagation_h3_relative_l2"],
                    "endpoint_nmse_effect": physical_errors[run_id]["endpoint_nmse"] - baseline_errors["endpoint_nmse"],
                    "temporal_gradient_effect": physical_errors[run_id]["temporal_gradient_error"] - baseline_errors["temporal_gradient_error"],
                })

    if len(global_commits) != 1 or len(global_models) != 1 or len(global_initials) != 1:
        raise ValueError("cross-prompt source/model/initial-latent fingerprint mismatch")
    if (len(arms_out), len(interactions), len(singletons)) != (168, 48, 64):
        raise ValueError("logical-arm/orientation/singleton completeness mismatch")
    interaction_result = aggregate_interactions(interactions)
    alignment_result = aggregate_alignment(singletons)
    decision = apply_decision(interaction_result, alignment_result)
    result = {
        "schema_version": "coframe.cross-step-endpoint-summary.v1",
        **decision,
        "primary_interaction": interaction_result["primary_gate"],
        "atomic_transfer": interaction_result["atomic_transfer_gate"],
        "objective_alignment": alignment_result,
        "reverse_control": interaction_result["reverse_orientation"],
        "sign_flip_diagnostics": interaction_result["sign_flip_diagnostics"],
        "completeness": {
            "status": "COMPLETE", "prompt_jobs": 8, "dense_references": 8,
            "unique_physical_sparse_runs": 120, "logical_arm_records": 168,
            "orientation_records": 48, "singleton_effects": 64,
        },
        "provenance": {
            "plan_sha256": plan_sha, "protocol_manifest_sha256": protocol_manifest_sha,
            "surface_sha256": sha256_file(surface_path), "surface_rows": 11520,
            "source_commit": next(iter(global_commits)), "model_fingerprint": next(iter(global_models)),
            "initial_latent_sha256": next(iter(global_initials)),
        },
        "latency": "NOT_REPORTED",
        "not_run": ["sequential_planner", "mpc", "online_selector", "video_decode", "perceptual_video_metrics"],
        "errors": [],
    }
    return result, arms_out, interactions, singletons


def _flatten_interactions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = {key: record[key] for key in ("prompt_id", "pair_id", "orientation")}
        for metric, payload in record["metrics"].items():
            scalar = payload["scalar"]
            for key in ("e00", "e10", "e01", "e11", "delta_i", "delta_j", "observed", "additive", "interaction", "tau", "rho", "sign_flip"):
                row[f"{metric}_{key}"] = scalar[key]
            row[f"{metric}_joint_improvement"] = payload["joint_improvement"]
            if "vector" in payload:
                for key, value in payload["vector"].items():
                    row[f"final_latent_vector_{key}"] = value
        rows.append(row)
    return rows


def _report(result: dict[str, Any]) -> str:
    primary, atomic, alignment = result["primary_interaction"], result["atomic_transfer"], result["objective_alignment"]
    lines = [
        "# CoFrame Cross-Step Endpoint Interaction Screen", "", "## Decision", "",
        f"**`{result['decision']}`**", "",
        f"- mechanism label: `{result['mechanism_label'] or 'NOT_SUPPORTED'}`",
        f"- recommended next step: {result['recommended_next_step']}", "",
        "## Frozen gates", "",
        f"- primary cross-step interaction: **{primary['passes']}**",
        f"- atomic K12->K6 transfer: **{atomic['passes']}**",
        f"- operator objective aligned: **{alignment['operator_aligned']}**",
        f"- +3 objective aligned: **{alignment['propagation_h3_aligned']}**", "",
        "| Primary metric | prompt-median rho | prompt passes | pair passes |", "|---|---:|---:|---:|",
    ]
    for name, value in primary["metrics"].items():
        lines.append(f"| {name} | {value['overall_prompt_median_rho']:.6f} | {value['prompt_pass_count']}/8 | {value['pair_pass_count']}/3 |")
    lines += [
        f"| final latent vector | {primary['endpoint_nmse_vector_rho_median']:.6f} | - | - |", "",
        "## Integrity and boundary", "",
        "- 8/8 prompt jobs, 120/120 sparse physical runs, 168/168 logical arms, 48/48 orientations, and 64/64 singleton effects audited.",
        "- Phase-A surface SHA/schema/all 11,520 rows and 96 required join keys were validated fail closed.",
        "- Source/model/plan fingerprints, CFG schedules, K9 parity repeats, finite FP32 latents and dense references were checked.",
        "- Latency is `NOT_REPORTED`; planner, MPC, online selector, decode and perceptual metrics are `NOT_RUN`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the frozen cross-step endpoint screen.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result, arms, interactions, singletons = analyze(args.root, args.plan, args.surface_csv)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_csv(output_dir / "arm_metrics.csv", arms)
        (output_dir / "interaction_records.json").write_text(json.dumps(interactions, indent=2) + "\n", encoding="utf-8")
        _write_csv(output_dir / "interaction_records.csv", _flatten_interactions(interactions))
        _write_csv(output_dir / "singleton_alignment.csv", singletons)
        (output_dir / "report.md").write_text(_report(result), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        incomplete = {
            "schema_version": "coframe.cross-step-endpoint-summary.v1",
            "decision": INCOMPLETE, "completeness": {"status": "INCOMPLETE"},
            "errors": [{"type": type(exc).__name__, "message": str(exc)}],
        }
        args.output.write_text(json.dumps(incomplete, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(incomplete, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

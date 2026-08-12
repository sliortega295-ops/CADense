#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from coframe.trajectory_interaction import (
    ARM_IDS,
    CHECKPOINTS,
    ORIENTATIONS,
    PAIR_BY_ID,
    PAIR_SPECS,
    PROMPT_IDS,
    aggregate_interaction_records,
    validate_pair_plan,
)


def _find_run(root: Path, prompt_id: str) -> Path:
    candidates = sorted((root / prompt_id).glob("*_trajectory_interaction_uniform_k9"))
    if len(candidates) != 1:
        raise ValueError(f"{prompt_id}: expected one trajectory-interaction run, found {len(candidates)}")
    return candidates[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _render_report(result: dict[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# CoFrame Trajectory Interaction Screen",
        "",
        "## Decision",
        "",
        f"**`{result['decision']}`**",
        "",
        "This preregistered screen tests only within-step pairwise budget interaction. "
        "It does not evaluate a planner, cross-step interaction, endpoint video quality, or latency.",
        "",
        "## Frozen contract and completeness",
        "",
        f"- prompts: {result['completeness']['prompt_count']}/8",
        f"- pair probes: {result['completeness']['pair_probe_count']}/48",
        f"- arm records: {result['completeness']['arm_record_count']}/336",
        f"- orientation records: {result['completeness']['orientation_record_count']}/96",
        f"- main trajectory: Uniform K=9; probe branches deployed: false",
        f"- causal/integrity errors: {len(result['errors'])}",
        f"- latency: NOT_REPORTED",
        "",
        "## Primary gate",
        "",
        "| Requirement | Value | Gate |",
        "|---|---:|---:|",
        f"| after-j prompt-median scalar rho | {gate['after_j_scalar']['overall_prompt_median']:.6f}; prompts {gate['after_j_scalar']['prompt_pass_count']}/8 | {'Pass' if gate['after_j_scalar']['passes'] else 'Fail'} |",
        f"| +3 prompt-median scalar rho | {gate['plus_3_dense_scalar']['overall_prompt_median']:.6f}; prompts {gate['plus_3_dense_scalar']['prompt_pass_count']}/8 | {'Pass' if gate['plus_3_dense_scalar']['passes'] else 'Fail'} |",
        f"| +3 all-cell vector rho median | {gate['plus_3_dense_vector']['median']:.6f} | {'Pass' if gate['plus_3_dense_vector']['passes'] else 'Fail'} |",
        f"| orientation/distance strata | all pass={gate['all_marginal_strata_pass']} | {'Pass' if gate['all_marginal_strata_pass'] else 'Fail'} |",
        f"| step strata | {gate['step_strata_pass_count']}/3 | {'Pass' if gate['step_strata_pass'] else 'Fail'} |",
        "",
        "## Diagnostics",
        "",
        "| Checkpoint | Scalar rho median | Vector rho median | Meaningful sign flips |",
        "|---|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        stats = result["checkpoint_overall"][checkpoint]
        lines.append(
            f"| {checkpoint} | {stats['rho_scalar']['median']:.6f} | "
            f"{stats['rho_vector']['median']:.6f} | {100.0 * stats['sign_flip_rate']:.2f}% |"
        )
    lines += [
        "",
        "The scalar interaction is descriptive because squared error can contain cross terms even "
        "when vector effects add. The vector factorial residual is retained as the state-space check.",
        "",
        "## Claim boundary",
        "",
        "A support result motivates a separately preregistered sequential or receding-horizon planner. "
        "A negative result rejects only strong within-step interaction under these frozen pairs; it does "
        "not establish that dynamic K is useless or exclude cross-step effects.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-csv", type=Path, required=True)
    parser.add_argument("--orientation-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_pair_plan(plan)
    errors: list[str] = []
    arm_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    pair_probe_count = 0

    for prompt_id in PROMPT_IDS:
        run_dir = _find_run(args.root, prompt_id)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
        config = summary.get("config", {})
        if config.get("trajectory_interaction_plan") != plan:
            errors.append(f"{prompt_id}: runtime plan differs from frozen plan")
        if config.get("adaptive_k_schedule") not in ({}, None):
            if any(int(value) != 9 for value in config["adaptive_k_schedule"].values()):
                errors.append(f"{prompt_id}: deployed schedule contains non-K9 entry")
        probes: list[dict[str, Any]] = []
        for event in trace.get("events", []):
            if event.get("event") != "transformer_forward":
                continue
            budget_events = event.get("budget_events", [])
            if any(int(entry.get("assigned_k", 9)) != 9 for entry in budget_events if "assigned_k" in entry):
                errors.append(f"{prompt_id}: main trajectory contains non-K9 budget")
            probes.extend(event.get("trajectory_interaction_probes", []))
        if len(probes) != len(PAIR_SPECS):
            errors.append(f"{prompt_id}: expected six pair probes, found {len(probes)}")
        seen_pairs: set[str] = set()
        for probe in probes:
            pair_id = str(probe.get("pair_id"))
            if pair_id not in PAIR_BY_ID or pair_id in seen_pairs:
                errors.append(f"{prompt_id}: unexpected or duplicate pair {pair_id}")
                continue
            seen_pairs.add(pair_id)
            spec = PAIR_BY_ID[pair_id]
            pair_probe_count += 1
            if not (probe.get("matched_prefix") and probe.get("state_continuation") and probe.get("deployed") is False):
                errors.append(f"{prompt_id}/{pair_id}: causal branch marker failed")
            if probe.get("executed_arms") != list(ARM_IDS):
                errors.append(f"{prompt_id}/{pair_id}: arm order differs from frozen plan")
            arms = probe.get("arms", {})
            if set(arms) != set(ARM_IDS):
                errors.append(f"{prompt_id}/{pair_id}: missing/extra arms")
                continue
            for arm_id in ARM_IDS:
                arm = arms[arm_id]
                if any(int(value) != 9 for value in arm.get("intermediate_group_budgets", [])):
                    errors.append(f"{prompt_id}/{pair_id}/{arm_id}: non-K9 intermediate group")
                if set(arm.get("checkpoints", {})) != set(CHECKPOINTS):
                    errors.append(f"{prompt_id}/{pair_id}/{arm_id}: checkpoint set mismatch")
                    continue
                row: dict[str, Any] = {
                    "prompt_id": prompt_id,
                    "pair_id": pair_id,
                    "step": spec.step,
                    "group_i": spec.group_i,
                    "group_j": spec.group_j,
                    "distance": spec.distance,
                    "arm_id": arm_id,
                    "k_i": int(arm["k_i"]),
                    "k_j": int(arm["k_j"]),
                }
                for checkpoint in CHECKPOINTS:
                    metric = arm["checkpoints"][checkpoint]
                    row[f"{checkpoint}_nmse"] = _finite(metric["normalized_mse"], f"{prompt_id}/{pair_id}/{arm_id}/{checkpoint}")
                local = arm["j_local"]
                row["j_local_native_nmse"] = _finite(local["native_normalized_mse"], "j_local_native_nmse")
                row["j_local_common_k9_nmse"] = _finite(local["common_k9_input_normalized_mse"], "j_local_common_k9_nmse")
                arm_rows.append(row)
            factorials = probe.get("factorials", {})
            if set(factorials) != set(ORIENTATIONS):
                errors.append(f"{prompt_id}/{pair_id}: factorial orientation mismatch")
                continue
            for orientation in ORIENTATIONS:
                payload = factorials[orientation]
                checkpoints = payload.get("checkpoints", {})
                record = {
                    "prompt_id": prompt_id,
                    "pair_id": pair_id,
                    "step": spec.step,
                    "group_i": spec.group_i,
                    "group_j": spec.group_j,
                    "distance": spec.distance,
                    "orientation": orientation,
                    "checkpoints": checkpoints,
                    "j_local_common_normalized_scalar": payload["j_local_common_normalized_scalar"],
                }
                orientation_rows.append(record)

    if errors:
        raise ValueError("; ".join(errors[:20]))
    if len(arm_rows) != 336 or pair_probe_count != 48:
        raise ValueError(f"completeness failure: pair_probes={pair_probe_count}, arms={len(arm_rows)}")
    aggregate = aggregate_interaction_records(orientation_rows)

    flat_orientation_rows: list[dict[str, Any]] = []
    for record in orientation_rows:
        row = {key: record[key] for key in ("prompt_id", "pair_id", "step", "group_i", "group_j", "distance", "orientation")}
        for checkpoint in CHECKPOINTS:
            scalar = record["checkpoints"][checkpoint]["scalar"]
            vector = record["checkpoints"][checkpoint]["vector"]
            row[f"{checkpoint}_rho_scalar"] = _finite(scalar["rho"], "rho_scalar")
            row[f"{checkpoint}_rho_vector"] = _finite(vector["rho"], "rho_vector")
            row[f"{checkpoint}_interaction"] = _finite(scalar["interaction"], "interaction")
            row[f"{checkpoint}_sign_flip"] = bool(scalar["meaningful_sign_flip"])
        local = record["j_local_common_normalized_scalar"]
        row["j_local_rho_scalar"] = _finite(local["rho"], "j_local_rho_scalar")
        row["j_local_interaction"] = _finite(local["interaction"], "j_local_interaction")
        row["j_local_sign_flip"] = bool(local["meaningful_sign_flip"])
        flat_orientation_rows.append(row)

    result = {
        "schema_version": "coframe.trajectory_interaction.summary.v1",
        **aggregate,
        "completeness": {
            **aggregate["completeness"],
            "pair_probe_count": pair_probe_count,
            "arm_record_count": len(arm_rows),
        },
        "j_local_common_normalized_scalar": {
            "count": len(flat_orientation_rows),
            "median_rho": median(row["j_local_rho_scalar"] for row in flat_orientation_rows),
            "sign_flip_rate": sum(row["j_local_sign_flip"] for row in flat_orientation_rows) / len(flat_orientation_rows),
        },
        "errors": [],
        "latency": "NOT_REPORTED",
        "scope": "within-step pairwise counterfactual interaction; no planner, endpoint, or cross-step claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.arm_csv, arm_rows)
    _write_csv(args.orientation_csv, flat_orientation_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

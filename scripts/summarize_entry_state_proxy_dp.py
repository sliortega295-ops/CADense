from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


BASELINES = ("fixed", "fis", "rhyme", "gap_only")
EXPECTED_STEPS = (5, 20, 40)
EXPECTED_BLOCKS = (8, 14, 20)
EXPECTED_PROMPTS = 8
EXPECTED_CELLS = EXPECTED_PROMPTS * len(EXPECTED_STEPS) * len(EXPECTED_BLOCKS)

# Preregistered qualitative gate made explicit before looking at screen data.
MIN_CELL_WIN_RATE = 0.60
MIN_PROMPT_WIN_COUNT = 6
MIN_STEP_BLOCK_POSITIVE_SLICES = 9
MIN_BEST_BASELINE_HEADROOM_RECOVERY = 0.20
EPS = 1.0e-12


def _summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(data),
        "mean": mean(data),
        "median": median(data),
        "min": min(data),
        "max": max(data),
    }


def _prompt_id(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"p\d+_s0", part):
            return part.split("_", 1)[0]
    raise ValueError(f"Cannot infer prompt id from {path}")


def _read_rows(root: Path) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    run_contracts: list[dict[str, Any]] = []
    for trace_path in sorted(root.rglob("trace.json")):
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        config = payload.get("run", {}).get("config", {})
        if not config.get("probe_entry_state_proxy_dp", False):
            continue
        prompt_id = _prompt_id(trace_path.relative_to(root))
        run_contracts.append(
            {
                "prompt_id": prompt_id,
                "trace": str(trace_path),
                "method": config.get("method"),
                "refresh_signal": config.get("refresh_signal"),
                "sketch_dim": config.get("sketch_dim"),
                "warmup_steps": config.get("warmup_steps"),
                "num_anchors": config.get("num_anchors"),
                "probe_steps": config.get("oracle_probe_steps"),
                "probe_blocks": config.get("oracle_probe_blocks"),
                "probe_curvature_signals": config.get("probe_curvature_signals"),
            }
        )
        for event in payload.get("events", []):
            if event.get("event") != "transformer_forward" or not event.get("probes"):
                continue
            entry = event.get("entry_state_proxy_dp") or {}
            for probe in event["probes"]:
                mesh_only = probe["mesh_only"]
                proxy_anchors = list(probe["entry_state_proxy_anchors"])
                row: dict[str, Any] = {
                    "prompt_id": prompt_id,
                    "trace_path": str(trace_path),
                    "step": int(probe["step"]),
                    "block": int(probe["block"]),
                    "proxy_anchors": proxy_anchors,
                    "entry_event_anchors": list(entry.get("anchors", [])),
                    "proxy_nmse": float(probe["mesh_entry_state_proxy_dp_nmse"]),
                    "proxy_relative_l2": float(probe["mesh_entry_state_proxy_dp_relative_l2"]),
                    "oracle_nmse": float(probe["mesh_oracle_nmse"]),
                    "oracle_anchors": list(mesh_only["oracle"]["anchors"]),
                    "oracle_regret": float(probe["mesh_entry_state_proxy_dp_oracle_nmse_regret"]),
                    "entry_objective_nmse": float(entry.get("normalized_mse", float("nan"))),
                }
                for name in BASELINES:
                    row[f"{name}_anchors"] = list(mesh_only[name]["anchors"])
                    row[f"{name}_nmse"] = float(mesh_only[name]["normalized_mse"])
                    row[f"relative_improvement_vs_{name}"] = float(
                        probe[f"mesh_entry_state_proxy_dp_relative_improvement_over_{name}"]
                    )
                    recovery = probe.get(f"mesh_entry_state_proxy_dp_headroom_recovery_vs_{name}")
                    row[f"headroom_recovery_vs_{name}"] = None if recovery is None else float(recovery)
                operators = probe.get("counterfactual_operator", {})
                for name in (*BASELINES, "entry_state_proxy_dp"):
                    operator = operators.get(name, {})
                    realized = operator.get("realized_block_delta", {})
                    row[f"operator_nmse_{name}"] = realized.get("normalized_mse")
                    propagation = operator.get("propagation", {})
                    for horizon in (1, 3):
                        row[f"propagation_h{horizon}_{name}"] = propagation.get(str(horizon), {}).get("relative_l2")
                best_name = min(BASELINES, key=lambda name: row[f"{name}_nmse"])
                best_nmse = float(row[f"{best_name}_nmse"])
                row["best_baseline"] = best_name
                row["best_baseline_nmse"] = best_nmse
                row["joint_win"] = row["proxy_nmse"] < best_nmse
                denominator = best_nmse - row["oracle_nmse"]
                row["headroom_recovery_vs_best"] = (
                    None if denominator <= EPS else (best_nmse - row["proxy_nmse"]) / denominator
                )
                if proxy_anchors != row["entry_event_anchors"]:
                    errors.append(f"{prompt_id} step={row['step']} block={row['block']}: probe/event mesh mismatch")
                rows.append(row)
    return rows, errors, run_contracts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CoFrame Entry-State Proxy-DP Signal Screen",
        "",
        f"**Decision: `{result['decision']}`**",
        "",
        "## Completeness",
        "",
        f"- Prompt runs: {result['completeness']['prompt_count']}/8",
        f"- Probe cells: {result['completeness']['cell_count']}/72",
        f"- Same-step mesh reuse groups: {result['mesh_reuse']['valid_groups']}/{result['mesh_reuse']['group_count']}",
        f"- Contract errors: {len(result['contract_errors'])}",
        "",
        "## Paired mesh NMSE",
        "",
        "| Baseline | Median relative improvement | Cell wins | Prompt wins | Median headroom recovery |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in BASELINES:
        payload = result["paired_vs_baseline"][name]
        lines.append(
            f"| {name} | {payload['relative_improvement']['median']:.4f} | "
            f"{payload['cell_win_count']}/{payload['cell_count']} | "
            f"{payload['prompt_win_count']}/8 | "
            f"{payload['headroom_recovery']['median'] if payload['headroom_recovery']['median'] is not None else 'N/A'} |"
        )
    joint = result["joint_vs_best_baseline"]
    lines += [
        "",
        "## Strongest-baseline and oracle gap",
        "",
        f"- Joint cell win rate: {joint['cell_win_rate']:.4f}",
        f"- Joint prompt wins: {joint['prompt_win_count']}/8",
        f"- Positive step/block slices: {joint['positive_step_block_slices']}/9",
        f"- Median oracle headroom recovery vs per-cell best baseline: {joint['headroom_recovery']['median']}",
        f"- Median Proxy-DP oracle NMSE regret: {result['oracle_regret']['median']}",
        "",
        "## Gate failures",
        "",
    ]
    if result["failed_gates"]:
        lines.extend(f"- {item}" for item in result["failed_gates"])
    else:
        lines.append("- None")
    lines += [
        "",
        "The Entry-State mesh was computed once after zero-indexed dense block 2 and reused only for matched-input counterfactual probes. It was never deployed into the denoising trajectory. Latency is NOT REPORTED.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Entry-State Proxy-DP matched-input probes.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    args = parser.parse_args()

    rows, contract_errors, run_contracts = _read_rows(args.root)
    prompts = sorted({row["prompt_id"] for row in rows})
    cells = {(row["prompt_id"], row["step"], row["block"]) for row in rows}
    expected_cells = {
        (f"p{prompt}", step, block)
        for prompt in range(EXPECTED_PROMPTS)
        for step in EXPECTED_STEPS
        for block in EXPECTED_BLOCKS
    }
    missing_cells = sorted(expected_cells - cells)
    duplicate_count = len(rows) - len(cells)

    for contract in run_contracts:
        expected = {
            "method": "coframe",
            "refresh_signal": "none",
            "sketch_dim": 64,
            "warmup_steps": 5,
            "num_anchors": 9,
            "probe_steps": list(EXPECTED_STEPS),
            "probe_blocks": list(EXPECTED_BLOCKS),
            "probe_curvature_signals": False,
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                contract_errors.append(
                    f"{contract['prompt_id']}: config {key}={contract.get(key)!r}, expected {value!r}"
                )

    grouped_meshes: dict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    for row in rows:
        grouped_meshes[(row["prompt_id"], row["step"])].append(tuple(row["proxy_anchors"]))
        if len(row["proxy_anchors"]) != 9 or row["proxy_anchors"][0] != 0 or row["proxy_anchors"][-1] != 20:
            contract_errors.append(
                f"{row['prompt_id']} step={row['step']} block={row['block']}: invalid K=9 boundary mesh"
            )
        if row["oracle_nmse"] > row["proxy_nmse"] + 1.0e-8:
            contract_errors.append(
                f"{row['prompt_id']} step={row['step']} block={row['block']}: exact oracle worse than proxy"
            )
    valid_reuse = sum(len(meshes) == 3 and len(set(meshes)) == 1 for meshes in grouped_meshes.values())
    if valid_reuse != len(grouped_meshes):
        contract_errors.append("Proxy-DP mesh was not identical across blocks 8/14/20 within every prompt-step")

    paired: dict[str, Any] = {}
    for name in BASELINES:
        improvements = [row[f"relative_improvement_vs_{name}"] for row in rows]
        recoveries = [
            row[f"headroom_recovery_vs_{name}"]
            for row in rows
            if row[f"headroom_recovery_vs_{name}"] is not None
        ]
        cell_wins = [row["proxy_nmse"] < row[f"{name}_nmse"] for row in rows]
        prompt_medians = {
            prompt: median(
                row[f"relative_improvement_vs_{name}"] for row in rows if row["prompt_id"] == prompt
            )
            for prompt in prompts
        }
        paired[name] = {
            "relative_improvement": _summary(improvements),
            "absolute_nmse_reduction": _summary(
                row[f"{name}_nmse"] - row["proxy_nmse"] for row in rows
            ),
            "headroom_recovery": _summary(recoveries),
            "cell_count": len(cell_wins),
            "cell_win_count": sum(cell_wins),
            "cell_win_rate": sum(cell_wins) / len(cell_wins) if cell_wins else 0.0,
            "prompt_median_relative_improvement": prompt_medians,
            "prompt_win_count": sum(value > 0 for value in prompt_medians.values()),
        }

    prompt_joint_medians = {
        prompt: median(
            (row["best_baseline_nmse"] - row["proxy_nmse"]) / (row["best_baseline_nmse"] + EPS)
            for row in rows
            if row["prompt_id"] == prompt
        )
        for prompt in prompts
    }
    slice_analysis: dict[str, Any] = {}
    positive_slices = 0
    for step in EXPECTED_STEPS:
        for block in EXPECTED_BLOCKS:
            subset = [row for row in rows if row["step"] == step and row["block"] == block]
            joint_improvements = [
                (row["best_baseline_nmse"] - row["proxy_nmse"]) / (row["best_baseline_nmse"] + EPS)
                for row in subset
            ]
            med = median(joint_improvements) if joint_improvements else None
            positive_slices += int(med is not None and med > 0)
            slice_analysis[f"step{step}.block{block}"] = {
                "count": len(subset),
                "joint_relative_improvement": _summary(joint_improvements),
                "joint_win_rate": (
                    sum(row["joint_win"] for row in subset) / len(subset) if subset else None
                ),
                "proxy_nmse": _summary(row["proxy_nmse"] for row in subset),
                "oracle_regret": _summary(row["oracle_regret"] for row in subset),
            }

    best_recoveries = [
        row["headroom_recovery_vs_best"]
        for row in rows
        if row["headroom_recovery_vs_best"] is not None
    ]
    joint_wins = sum(bool(row["joint_win"]) for row in rows)
    joint = {
        "cell_count": len(rows),
        "cell_win_count": joint_wins,
        "cell_win_rate": joint_wins / len(rows) if rows else 0.0,
        "prompt_median_relative_improvement": prompt_joint_medians,
        "prompt_win_count": sum(value > 0 for value in prompt_joint_medians.values()),
        "positive_step_block_slices": positive_slices,
        "headroom_recovery": _summary(best_recoveries),
    }

    failed_gates: list[str] = []
    if len(prompts) != EXPECTED_PROMPTS or len(rows) != EXPECTED_CELLS or missing_cells or duplicate_count:
        failed_gates.append("completeness: require exactly 8 prompts and 72 unique probe cells")
    if contract_errors:
        failed_gates.append("contract: configuration, oracle, K=9 boundary, or mesh-reuse validation failed")
    for name in BASELINES:
        if paired[name]["cell_win_rate"] < MIN_CELL_WIN_RATE:
            failed_gates.append(f"{name}: cell win rate below {MIN_CELL_WIN_RATE:.0%}")
        if paired[name]["prompt_win_count"] < MIN_PROMPT_WIN_COUNT:
            failed_gates.append(f"{name}: positive prompt count below {MIN_PROMPT_WIN_COUNT}/8")
    if joint["cell_win_rate"] < MIN_CELL_WIN_RATE:
        failed_gates.append(f"joint strongest-baseline cell win rate below {MIN_CELL_WIN_RATE:.0%}")
    if joint["prompt_win_count"] < MIN_PROMPT_WIN_COUNT:
        failed_gates.append(f"joint strongest-baseline prompt wins below {MIN_PROMPT_WIN_COUNT}/8")
    if positive_slices < MIN_STEP_BLOCK_POSITIVE_SLICES:
        failed_gates.append("not all 9 denoising-step/block slices have positive median joint improvement")
    best_recovery_median = joint["headroom_recovery"]["median"]
    if best_recovery_median is None or best_recovery_median < MIN_BEST_BASELINE_HEADROOM_RECOVERY:
        failed_gates.append(
            f"median oracle headroom recovery vs strongest baseline below {MIN_BEST_BASELINE_HEADROOM_RECOVERY:.0%}"
        )

    result = {
        "schema_version": "coframe.entry_state_proxy_dp.v1",
        "decision": "SUPPORT_ENTRY_STATE_PROXY_DP" if not failed_gates else "REJECT_ENTRY_STATE_PROXY_DP",
        "gate": {
            "cell_win_rate_each_and_joint": MIN_CELL_WIN_RATE,
            "prompt_win_count_each_and_joint": MIN_PROMPT_WIN_COUNT,
            "positive_step_block_slices": MIN_STEP_BLOCK_POSITIVE_SLICES,
            "median_headroom_recovery_vs_best": MIN_BEST_BASELINE_HEADROOM_RECOVERY,
            "tuned_after_results": False,
        },
        "completeness": {
            "prompt_count": len(prompts),
            "prompts": prompts,
            "run_count": len(run_contracts),
            "cell_count": len(rows),
            "unique_cell_count": len(cells),
            "missing_cells": [list(cell) for cell in missing_cells],
            "duplicate_count": duplicate_count,
        },
        "mesh_reuse": {
            "source_block": 2,
            "group_count": len(grouped_meshes),
            "valid_groups": valid_reuse,
            "all_blocks_reused_same_mesh": valid_reuse == len(grouped_meshes) == 24,
        },
        "paired_vs_baseline": paired,
        "joint_vs_best_baseline": joint,
        "oracle_regret": _summary(row["oracle_regret"] for row in rows),
        "proxy_nmse": _summary(row["proxy_nmse"] for row in rows),
        "oracle_nmse": _summary(row["oracle_nmse"] for row in rows),
        "by_step_block": slice_analysis,
        "run_contracts": run_contracts,
        "contract_errors": contract_errors,
        "failed_gates": failed_gates,
        "latency": "NOT_REPORTED",
        "selector_deployed": False,
    }

    csv_output = args.csv_output or args.output.with_name("probe_cells.csv")
    report_output = args.report_output or args.output.with_name("REPORT.md")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(csv_output, rows)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

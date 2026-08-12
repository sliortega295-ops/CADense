from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch


PROMPT_RE = re.compile(r"p\d+_s0")
PROMPTS = tuple(f"p{index}_s0" for index in range(8))
EXPECTED_SLOTS = {(step, group) for step in range(5, 50) for group in range(8)}
REQUIRED_PROMPT_WINS = 6
MAX_BUDGET_RELATIVE_ERROR = 0.05


def prompt_id(path: Path) -> str:
    match = PROMPT_RE.search(str(path))
    if not match:
        raise ValueError(f"cannot infer prompt id from {path}")
    return match.group(0)


def finite_summary(values: list[float]) -> dict[str, Any]:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(data),
        "mean": mean(data),
        "median": median(data),
        "min": min(data),
        "max": max(data),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload["latents"] if isinstance(payload, dict) else payload
    return value.detach().float().cpu()


def endpoint_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"latent shape mismatch {reference.shape} vs {candidate.shape}")
    difference = candidate - reference
    reference_energy = reference.square().sum().clamp_min(1.0e-12)
    nmse = float((difference.square().sum() / reference_energy).item())
    relative_l2 = math.sqrt(max(0.0, nmse))
    cosine = float(torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item())
    reference_dt = reference[:, :, 1:] - reference[:, :, :-1]
    candidate_dt = candidate[:, :, 1:] - candidate[:, :, :-1]
    temporal = float(
        ((candidate_dt - reference_dt).norm() / reference_dt.norm().clamp_min(1.0e-12)).item()
    )
    return {
        "endpoint_nmse": nmse,
        "endpoint_relative_l2": relative_l2,
        "endpoint_cosine": cosine,
        "endpoint_temporal_gradient_relative_l2": temporal,
    }


def extract_group_rows(trace_path: Path, *, expected_mode: str) -> tuple[list[dict[str, Any]], float, list[str]]:
    trace = load_json(trace_path)
    config = trace.get("run", {}).get("config", {})
    errors: list[str] = []
    if config.get("calibrated_budget_probe_mode") != expected_mode:
        errors.append(
            f"{trace_path}: mode={config.get('calibrated_budget_probe_mode')!r}, expected {expected_mode!r}"
        )
    rows: list[dict[str, Any]] = []
    block_budgets: list[int] = []
    for event in trace.get("events", []):
        if event.get("event") != "transformer_forward":
            continue
        block_budgets.extend(len(value) for value in event.get("block_anchors", {}).values())
        for probe in event.get("budget_group_probes", []):
            assigned = int(probe["trajectory_assigned_k"])
            candidate = probe.get("candidates", {}).get(str(assigned))
            if candidate is None:
                errors.append(f"{trace_path}: missing assigned K={assigned} candidate")
                continue
            rows.append(
                {
                    "step": int(probe["step"]),
                    "group": int(probe["group"]),
                    "slot": f"{int(probe['step'])}:{int(probe['group'])}",
                    "k": assigned,
                    "operator_nmse": float(candidate["operator_delta"]["normalized_mse"]),
                    "operator_relative_l2": float(candidate["operator_delta"]["relative_l2"]),
                    "propagation_h3_nmse": float(candidate["propagation_h3"]["normalized_mse"]),
                    "propagation_h3_relative_l2": float(candidate["propagation_h3"]["relative_l2"]),
                }
            )
    avg_k = mean(block_budgets) if block_budgets else float("nan")
    slots = {(row["step"], row["group"]) for row in rows}
    if len(rows) != 360 or slots != EXPECTED_SLOTS:
        errors.append(f"{trace_path}: group rows={len(rows)}, unique slots={len(slots)}, expected 360")
    return rows, avg_k, errors


def relative_improvement(baseline: float, method: float) -> float:
    return (float(baseline) - float(method)) / (float(baseline) + 1.0e-12)


def paired_gate(values: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    improvements = [item[f"improvement_{metric}"] for item in values]
    wins = sum(value > 0.0 for value in improvements)
    return {
        "metric": metric,
        "relative_improvement": finite_summary(improvements),
        "prompt_win_count": wins,
        "prompt_count": len(improvements),
        "passes": wins >= REQUIRED_PROMPT_WINS and median(improvements) > 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# CoFrame Calibrated Step-Block Budget Schedule",
        "",
        f"**Decision: `{result['decision']}`**",
        "",
        "## Completeness and budget",
        "",
        f"- Held-out folds: {result['completeness']['fold_count']}/8",
        f"- Uniform group cells: {result['completeness']['uniform_group_cell_count']}/2880",
        f"- Scheduled group cells: {result['completeness']['scheduled_group_cell_count']}/2880",
        f"- All actual budgets within 5% of K=9: {result['budget_gate']['all_folds_within_5pct']}",
        f"- Median scheduled average K: {result['budget_gate']['average_k']['median']}",
        f"- Contract errors: {len(result['errors'])}",
        "",
        "## Paired held-out results",
        "",
        "| Metric | Median relative improvement | Prompt wins | Gate |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "operator_nmse",
        "propagation_h3_relative_l2",
        "endpoint_nmse",
        "endpoint_temporal_gradient_relative_l2",
    ):
        gate = result["metric_gates"][key]
        lines.append(
            f"| {key} | {gate['relative_improvement']['median']:.6f} | "
            f"{gate['prompt_win_count']}/8 | {gate['passes']} |"
        )
    lines += ["", "## Failures", ""]
    if result["failed_gates"]:
        lines.extend(f"- {value}" for value in result["failed_gates"])
    else:
        lines.append("- None")
    lines += [
        "",
        "Schedules were optimized independently in eight LOPO folds using only the other seven prompts. The budget set, exact mean-K constraint, solver, and decision gate were fixed before held-out evaluation. Latency is NOT REPORTED.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize held-out calibrated step/group schedules.")
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    plan = load_json(args.plan)
    errors = list(plan.get("errors", []))
    prompt_results: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for pid in PROMPTS:
        surface_dir = next(iter(sorted((args.surface_root / pid).glob("*_surface_uniform_k9"))), None)
        dense_dir = next(iter(sorted((args.eval_root / pid).glob("*_dense"))), None)
        scheduled_dir = next(iter(sorted((args.eval_root / pid).glob("*_calibrated_step_block"))), None)
        if surface_dir is None or dense_dir is None or scheduled_dir is None:
            errors.append(f"{pid}: missing surface, dense, or scheduled run directory")
            continue
        uniform_rows, uniform_avg_k, uniform_errors = extract_group_rows(
            surface_dir / "trace.json",
            expected_mode="surface",
        )
        scheduled_rows, scheduled_avg_k, scheduled_errors = extract_group_rows(
            scheduled_dir / "trace.json",
            expected_mode="current",
        )
        errors.extend(uniform_errors + scheduled_errors)
        uniform_by_slot = {row["slot"]: row for row in uniform_rows}
        scheduled_by_slot = {row["slot"]: row for row in scheduled_rows}
        fold = plan.get("folds", {}).get(pid)
        if fold is None:
            errors.append(f"{pid}: missing LOPO fold plan")
            continue
        if pid in fold.get("training_prompt_ids", []) or fold.get("heldout_used_for_optimization") is not False:
            errors.append(f"{pid}: held-out leakage marker failed")
        expected_schedule = {str(key): int(value) for key, value in fold["schedule"].items()}
        actual_schedule = {slot: int(row["k"]) for slot, row in scheduled_by_slot.items()}
        if actual_schedule != expected_schedule:
            errors.append(f"{pid}: runtime schedule differs from the frozen LOPO schedule")

        for slot in sorted(uniform_by_slot, key=lambda value: tuple(map(int, value.split(":")))):
            uniform = uniform_by_slot[slot]
            scheduled = scheduled_by_slot[slot]
            cell_rows.append(
                {
                    "prompt_id": pid,
                    "step": uniform["step"],
                    "group": uniform["group"],
                    "uniform_k": uniform["k"],
                    "scheduled_k": scheduled["k"],
                    "uniform_operator_nmse": uniform["operator_nmse"],
                    "scheduled_operator_nmse": scheduled["operator_nmse"],
                    "uniform_propagation_h3_relative_l2": uniform["propagation_h3_relative_l2"],
                    "scheduled_propagation_h3_relative_l2": scheduled["propagation_h3_relative_l2"],
                }
            )

        dense_latent = load_latent(dense_dir / "latents.pt")
        uniform_latent = load_latent(surface_dir / "latents.pt")
        scheduled_latent = load_latent(scheduled_dir / "latents.pt")
        uniform_endpoint = endpoint_metrics(dense_latent, uniform_latent)
        scheduled_endpoint = endpoint_metrics(dense_latent, scheduled_latent)
        uniform_operator = mean(row["operator_nmse"] for row in uniform_rows)
        scheduled_operator = mean(row["operator_nmse"] for row in scheduled_rows)
        uniform_propagation = mean(row["propagation_h3_relative_l2"] for row in uniform_rows)
        scheduled_propagation = mean(row["propagation_h3_relative_l2"] for row in scheduled_rows)
        item: dict[str, Any] = {
            "prompt_id": pid,
            "train_prompt_ids": fold["training_prompt_ids"],
            "uniform_average_k": uniform_avg_k,
            "scheduled_average_k": scheduled_avg_k,
            "scheduled_average_k_relative_error_vs_9": abs(scheduled_avg_k - 9.0) / 9.0,
            "budget_match_within_5pct": abs(scheduled_avg_k - 9.0) / 9.0 <= MAX_BUDGET_RELATIVE_ERROR,
            "uniform_operator_nmse": uniform_operator,
            "scheduled_operator_nmse": scheduled_operator,
            "uniform_propagation_h3_relative_l2": uniform_propagation,
            "scheduled_propagation_h3_relative_l2": scheduled_propagation,
            **{f"uniform_{key}": value for key, value in uniform_endpoint.items()},
            **{f"scheduled_{key}": value for key, value in scheduled_endpoint.items()},
        }
        for metric in (
            "operator_nmse",
            "propagation_h3_relative_l2",
            "endpoint_nmse",
            "endpoint_temporal_gradient_relative_l2",
        ):
            item[f"improvement_{metric}"] = relative_improvement(
                item[f"uniform_{metric}"],
                item[f"scheduled_{metric}"],
            )
        prompt_results.append(item)

    metric_gates = {
        metric: paired_gate(prompt_results, metric)
        for metric in (
            "operator_nmse",
            "propagation_h3_relative_l2",
            "endpoint_nmse",
            "endpoint_temporal_gradient_relative_l2",
        )
    }
    budget_matches = [item["budget_match_within_5pct"] for item in prompt_results]
    budget_gate = {
        "target_average_k": 9.0,
        "allowed_relative_error": MAX_BUDGET_RELATIVE_ERROR,
        "average_k": finite_summary([item["scheduled_average_k"] for item in prompt_results]),
        "all_folds_within_5pct": len(budget_matches) == 8 and all(budget_matches),
    }
    failed_gates: list[str] = []
    if len(prompt_results) != 8 or len(cell_rows) != 8 * 360:
        failed_gates.append("completeness: require 8 held-out folds and 2880 cells per policy")
    if errors:
        failed_gates.append("contract or integrity errors are present")
    if not budget_gate["all_folds_within_5pct"]:
        failed_gates.append("fairness: at least one held-out schedule is outside K=9 +/-5%")
    for metric, gate in metric_gates.items():
        if not gate["passes"]:
            failed_gates.append(
                f"{metric}: require positive median paired improvement and at least 6/8 prompt wins"
            )

    result = {
        "schema_version": "coframe.calibrated_step_block.heldout_summary.v1",
        "decision": (
            "SUPPORT_CALIBRATED_STEP_BLOCK_BUDGET"
            if not failed_gates
            else "REJECT_CALIBRATED_STEP_BLOCK_BUDGET"
        ),
        "gate": {
            "budget_values": [6, 9, 12, 21],
            "budget_relative_tolerance": MAX_BUDGET_RELATIVE_ERROR,
            "required_prompt_wins_per_metric": REQUIRED_PROMPT_WINS,
            "required_median_direction_per_metric": "positive",
            "metrics": list(metric_gates),
            "tuned_after_results": False,
        },
        "completeness": {
            "fold_count": len(prompt_results),
            "uniform_group_cell_count": len(cell_rows),
            "scheduled_group_cell_count": len(cell_rows),
        },
        "budget_gate": budget_gate,
        "metric_gates": metric_gates,
        "prompt_results": prompt_results,
        "failed_gates": failed_gates,
        "errors": errors,
        "latency": "NOT_REPORTED",
        "online_signal_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.cells_csv, cell_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

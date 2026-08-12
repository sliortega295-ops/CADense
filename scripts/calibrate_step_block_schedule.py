from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from coframe.calibrated_budget import optimize_exact_budget_schedule


PROMPT_RE = re.compile(r"p\d+_s0")
BUDGETS = (6, 9, 12, 21)
STEPS = tuple(range(5, 50))
GROUPS = tuple(range(8))
EXPECTED_PROMPTS = tuple(f"p{index}_s0" for index in range(8))
EXPECTED_ROWS = len(EXPECTED_PROMPTS) * len(STEPS) * len(GROUPS) * len(BUDGETS)


def prompt_id(path: Path) -> str:
    match = PROMPT_RE.search(str(path))
    if not match:
        raise ValueError(f"cannot infer prompt id from {path}")
    return match.group(0)


def finite_summary(values: list[float]) -> dict[str, Any]:
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


def load_surface(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    run_contracts: list[dict[str, Any]] = []
    errors: list[str] = []
    for trace_path in sorted(root.rglob("trace.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        config = trace.get("run", {}).get("config", {})
        if config.get("calibrated_budget_probe_mode") != "surface":
            continue
        pid = prompt_id(trace_path)
        contract = {
            "prompt_id": pid,
            "trace_path": str(trace_path),
            "method": config.get("method"),
            "adaptive_k_policy": config.get("adaptive_k_policy"),
            "adaptive_k_values": config.get("adaptive_k_values"),
            "warmup_steps": config.get("warmup_steps"),
            "num_anchors": config.get("num_anchors"),
            "sparse_block_start": config.get("sparse_block_start"),
            "sparse_block_end": config.get("sparse_block_end"),
            "block_group_size": config.get("block_group_size"),
            "kv_mode": config.get("kv_mode"),
            "interpolation_target": config.get("interpolation_target"),
        }
        run_contracts.append(contract)
        expected = {
            "method": "adaptive_k",
            "adaptive_k_policy": "step_block",
            "adaptive_k_values": list(BUDGETS),
            "warmup_steps": 5,
            "num_anchors": 9,
            "sparse_block_start": 3,
            "sparse_block_end": 27,
            "block_group_size": 3,
            "kv_mode": "full_kv",
            "interpolation_target": "delta",
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                errors.append(f"{pid}: {key}={contract.get(key)!r}, expected {value!r}")
        for event in trace.get("events", []):
            if event.get("event") != "transformer_forward":
                continue
            for probe in event.get("budget_group_probes", []):
                step = int(probe["step"])
                group = int(probe["group"])
                if int(probe.get("trajectory_assigned_k", -1)) != 9:
                    errors.append(f"{pid} {step}:{group}: Phase-A trajectory is not K=9")
                if probe.get("matched_input") is not True or probe.get("deployed") is not False:
                    errors.append(f"{pid} {step}:{group}: counterfactual contract marker failed")
                if tuple(int(value) for value in probe.get("evaluated_budgets", [])) != BUDGETS:
                    errors.append(f"{pid} {step}:{group}: evaluated budget set changed")
                for budget in BUDGETS:
                    candidate = probe.get("candidates", {}).get(str(budget))
                    if not candidate:
                        errors.append(f"{pid} {step}:{group}: missing K={budget}")
                        continue
                    operator = candidate["operator_delta"]
                    propagation = candidate["propagation_h3"]
                    row = {
                        "prompt_id": pid,
                        "step": step,
                        "group": group,
                        "slot": f"{step}:{group}",
                        "block_start": int(probe["block_start"]),
                        "block_end": int(probe["block_end"]),
                        "k": budget,
                        "operator_nmse": float(operator["normalized_mse"]),
                        "operator_relative_l2": float(operator["relative_l2"]),
                        "operator_reference_energy": float(operator["reference_energy"]),
                        "propagation_h3_nmse": float(propagation["normalized_mse"]),
                        "propagation_h3_relative_l2": float(propagation["relative_l2"]),
                        "propagation_h3_reference_energy": float(propagation["reference_energy"]),
                        "anchors": list(candidate["anchors"]),
                        "trace_path": str(trace_path),
                    }
                    if not all(
                        math.isfinite(row[key]) and row[key] >= 0.0
                        for key in (
                            "operator_nmse",
                            "operator_relative_l2",
                            "operator_reference_energy",
                            "propagation_h3_nmse",
                            "propagation_h3_relative_l2",
                            "propagation_h3_reference_energy",
                        )
                    ):
                        errors.append(f"{pid} {step}:{group} K={budget}: non-finite or negative metric")
                    rows.append(row)
    return rows, run_contracts, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact-budget LOPO step/group schedules.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--schedule-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, run_contracts, errors = load_surface(args.root)
    prompts = sorted({row["prompt_id"] for row in rows})
    keys = Counter((row["prompt_id"], row["step"], row["group"], row["k"]) for row in rows)
    expected_keys = {
        (pid, step, group, budget)
        for pid in EXPECTED_PROMPTS
        for step in STEPS
        for group in GROUPS
        for budget in BUDGETS
    }
    missing = sorted(expected_keys - set(keys))
    duplicates = sorted(key for key, count in keys.items() if count != 1)
    if tuple(prompts) != EXPECTED_PROMPTS:
        errors.append(f"prompt set {prompts!r} does not match preregistered prompts")
    if len(rows) != EXPECTED_ROWS or missing or duplicates:
        errors.append(
            f"surface completeness failed rows={len(rows)}/{EXPECTED_ROWS}, "
            f"missing={len(missing)}, duplicates={len(duplicates)}"
        )

    rows.sort(key=lambda row: (row["prompt_id"], row["step"], row["group"], row["k"]))
    write_csv(args.surface_csv, rows)
    surface_bytes = args.surface_csv.read_bytes()
    surface_sha256 = hashlib.sha256(surface_bytes).hexdigest()

    by_budget = {
        str(budget): {
            "operator_nmse": finite_summary([row["operator_nmse"] for row in rows if row["k"] == budget]),
            "propagation_h3_relative_l2": finite_summary(
                [row["propagation_h3_relative_l2"] for row in rows if row["k"] == budget]
            ),
        }
        for budget in BUDGETS
    }
    monotonic_cells = 0
    cell_count = len(EXPECTED_PROMPTS) * len(STEPS) * len(GROUPS)
    grouped: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["prompt_id"], row["step"], row["group"])][row["k"]] = row
    for values in grouped.values():
        series = [values[budget]["operator_nmse"] for budget in BUDGETS]
        monotonic_cells += int(all(right <= left + 1.0e-12 for left, right in zip(series[:-1], series[1:])))
    k21_max = max((row["operator_nmse"] for row in rows if row["k"] == 21), default=None)
    if k21_max is None or k21_max > 1.0e-6:
        errors.append(f"K=21 dense-equivalence sanity failed: max operator NMSE={k21_max}")

    args.schedule_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, Any] = {}
    for heldout in EXPECTED_PROMPTS:
        train_ids = [pid for pid in EXPECTED_PROMPTS if pid != heldout]
        costs: dict[str, dict[int, float]] = {}
        for step in STEPS:
            for group in GROUPS:
                slot = f"{step}:{group}"
                costs[slot] = {
                    budget: mean(
                        grouped[(pid, step, group)][budget]["operator_nmse"]
                        for pid in train_ids
                    )
                    for budget in BUDGETS
                }
        optimized = optimize_exact_budget_schedule(
            costs,
            budgets=BUDGETS,
            target_average_k=9.0,
        )
        schedule_path = args.schedule_dir / f"{heldout}.json"
        schedule_path.write_text(json.dumps(optimized.schedule, indent=2) + "\n", encoding="utf-8")
        heldout_uniform = mean(
            grouped[(heldout, step, group)][9]["operator_nmse"]
            for step in STEPS
            for group in GROUPS
        )
        heldout_scheduled = mean(
            grouped[(heldout, step, group)][optimized.schedule[f"{step}:{group}"]]["operator_nmse"]
            for step in STEPS
            for group in GROUPS
        )
        fold = {
            "heldout_prompt_id": heldout,
            "training_prompt_ids": train_ids,
            "heldout_used_for_optimization": False,
            "surface_sha256": surface_sha256,
            "schedule_path": str(schedule_path),
            **optimized.to_dict(),
            # Post-calibration diagnostic only; never fed back into the schedule.
            "offline_heldout_surface_uniform_k9_mean_operator_nmse": heldout_uniform,
            "offline_heldout_surface_scheduled_mean_operator_nmse": heldout_scheduled,
            "offline_heldout_surface_relative_improvement": (
                (heldout_uniform - heldout_scheduled) / (heldout_uniform + 1.0e-12)
            ),
        }
        if fold["average_k"] != 9.0 or len(fold["schedule"]) != len(STEPS) * len(GROUPS):
            errors.append(f"{heldout}: exact-budget schedule invariant failed")
        if heldout in fold["training_prompt_ids"]:
            errors.append(f"{heldout}: held-out prompt leaked into training ids")
        folds[heldout] = fold

    result = {
        "schema_version": "coframe.calibrated_step_block.lopo_plan.v1",
        "surface": {
            "run_count": len(run_contracts),
            "prompt_count": len(prompts),
            "row_count": len(rows),
            "expected_row_count": EXPECTED_ROWS,
            "unique_key_count": len(keys),
            "missing_count": len(missing),
            "duplicate_count": len(duplicates),
            "sha256": surface_sha256,
            "by_budget": by_budget,
            "operator_monotonic_cell_count": monotonic_cells,
            "operator_monotonic_cell_rate": monotonic_cells / cell_count,
            "k21_max_operator_nmse": k21_max,
        },
        "optimization_contract": {
            "objective": "sum of training-prompt mean group operator NMSE",
            "budget_values": list(BUDGETS),
            "slot_count": len(STEPS) * len(GROUPS),
            "target_average_k": 9.0,
            "target_total_k": 9 * len(STEPS) * len(GROUPS),
            "constraint": "exact total K; stronger than the held-out +/-5% fairness gate",
            "solver": "exact dynamic programming",
            "lopo": True,
            "heldout_used_for_schedule": False,
            "tuned_after_results": False,
        },
        "run_contracts": run_contracts,
        "folds": folds,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from coframe.budget import select_budget, schedule_key


PROMPT_RE = re.compile(r"p\d+_s\d+")
DEFAULT_BUDGETS = (6, 9, 12, 21)
DEFAULT_QUANTILES = (0.35, 0.80, 0.95)


def prompt_id_from_path(path: Path) -> str:
    match = PROMPT_RE.search(str(path))
    return match.group(0) if match else path.parent.name


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        value = 0.5 * (left + right - 1)
        for slot in range(left, right):
            result[order[slot]] = value
        left = right
    return result


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom <= 1.0e-12:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rank(x), rank(y))


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of an empty list")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def centered(records: list[dict[str, Any]], predictor: str, target: str) -> tuple[list[float], list[float]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(int(record["step"]), int(record["block"]))].append(record)
    x, y = [], []
    for group in groups.values():
        px = mean(float(record[predictor]) for record in group)
        py = mean(float(record[target]) for record in group)
        for record in group:
            x.append(float(record[predictor]) - px)
            y.append(float(record[target]) - py)
    return x, y


def correlation_summary(records: list[dict[str, Any]], predictor: str, target: str) -> dict[str, Any]:
    usable = [record for record in records if record.get(predictor) is not None and record.get(target) is not None]
    x = [float(record[predictor]) for record in usable]
    y = [float(record[target]) for record in usable]
    cx, cy = centered(usable, predictor, target)
    prompts = sorted({str(record["prompt_id"]) for record in usable})
    lopo = []
    lopo_centered = []
    for heldout in prompts:
        train = [record for record in usable if record["prompt_id"] != heldout]
        tx = [float(record[predictor]) for record in train]
        ty = [float(record[target]) for record in train]
        value = spearman(tx, ty)
        if value is not None:
            lopo.append(value)
        tcx, tcy = centered(train, predictor, target)
        value = spearman(tcx, tcy)
        if value is not None:
            lopo_centered.append(value)
    return {
        "count": len(usable),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "step_block_centered_spearman": spearman(cx, cy),
        "lopo_spearman_median": median(lopo) if lopo else None,
        "lopo_spearman_min": min(lopo) if lopo else None,
        "lopo_spearman_max": max(lopo) if lopo else None,
        "lopo_centered_spearman_median": median(lopo_centered) if lopo_centered else None,
        "lopo_centered_spearman_min": min(lopo_centered) if lopo_centered else None,
        "lopo_centered_spearman_max": max(lopo_centered) if lopo_centered else None,
    }


def load_runs(root: Path, *, refresh_signal: str, kv_mode: str) -> list[dict[str, Any]]:
    runs = []
    for trace_path in sorted(root.rglob("trace.json")):
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        config = payload.get("run", {}).get("config", {})
        if config.get("refresh_signal") != refresh_signal:
            continue
        if config.get("kv_mode") != kv_mode:
            continue
        runs.append({
            "prompt_id": prompt_id_from_path(trace_path),
            "path": str(trace_path),
            "config": config,
            "events": [event for event in payload.get("events", []) if event.get("event") == "transformer_forward"],
        })
    return runs


def defect_values_by_group(event: dict[str, Any], *, start: int, end: int, group_size: int) -> dict[int, list[float]]:
    groups: dict[int, list[float]] = defaultdict(list)
    for entry in event.get("defects", []):
        block = int(entry["block"])
        if not start <= block < end:
            continue
        group = (block - start) // group_size
        groups[group].extend(float(value) for value in entry.get("values", {}).values())
    return dict(groups)


def extract_records(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    probe_records: list[dict[str, Any]] = []
    causal_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        start = int(config.get("sparse_block_start", 3))
        end = int(config.get("sparse_block_end", 27))
        group_size = int(config.get("block_group_size", 3))
        events = sorted(run["events"], key=lambda event: int(event.get("step_index", -1)))
        previous_step_last: list[float] | None = None
        for event in events:
            step = int(event["step_index"])
            groups = defect_values_by_group(event, start=start, end=end, group_size=group_size)
            max_group = (end - start - 1) // group_size
            for group in range(max_group + 1):
                source = groups.get(group - 1) if group > 0 else previous_step_last
                if source:
                    causal_observations[run["prompt_id"]].append({
                        "step": step,
                        "group": group,
                        "key": schedule_key(step, group),
                        "mean_defect": mean(source),
                        "max_defect": max(source),
                    })
            if groups.get(max_group):
                previous_step_last = groups[max_group]

            for probe in event.get("probes", []):
                block = int(probe["block"])
                group = (block - start) // group_size
                source = groups.get(group - 1) if group > 0 else previous_step_last
                if not source:
                    continue
                probe_records.append({
                    "prompt_id": run["prompt_id"],
                    "step": step,
                    "block": block,
                    "source_group": group - 1,
                    "mean_prev_group_defect": mean(source),
                    "max_prev_group_defect": max(source),
                    "mesh_fixed_nmse": probe.get("mesh_fixed_nmse"),
                    "operator_nmse": probe.get("block_delta_normalized_mse"),
                    "propagation_h3": probe.get("propagated_relative_l2_h3"),
                })
    return probe_records, causal_observations


def build_fold_plans(
    observations: dict[str, list[dict[str, Any]]],
    *,
    budgets: tuple[int, ...],
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    prompts = sorted(observations)
    folds = {}
    for heldout in prompts:
        train = [item for prompt in prompts if prompt != heldout for item in observations[prompt]]
        fold: dict[str, Any] = {}
        for stat in ("mean_defect", "max_defect"):
            values = [float(item[stat]) for item in train]
            thresholds = [quantile(values, q) for q in quantiles]
            assigned = [select_budget(value, thresholds, budgets) for value in values]
            fold[f"{stat}_thresholds"] = thresholds
            fold[f"{stat}_calibration_avg_k"] = mean(assigned) if assigned else None
        thresholds = fold["mean_defect_thresholds"]
        by_key: dict[str, list[float]] = defaultdict(list)
        for item in train:
            by_key[str(item["key"])].append(float(item["mean_defect"]))
        schedule = {
            key: select_budget(mean(values), thresholds, budgets)
            for key, values in sorted(by_key.items())
        }
        fold["step_block_schedule"] = schedule
        fold["step_block_calibration_avg_k"] = mean(schedule.values()) if schedule else None
        fold["train_prompt_ids"] = [prompt for prompt in prompts if prompt != heldout]
        folds[heldout] = fold
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1d causal lag test and LOPO budget calibration")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--refresh-signal", default="defect")
    parser.add_argument("--kv-mode", default="full_kv")
    args = parser.parse_args()

    runs = load_runs(args.root, refresh_signal=args.refresh_signal, kv_mode=args.kv_mode)
    probe_records, observations = extract_records(runs)
    metrics = {}
    for predictor in ("mean_prev_group_defect", "max_prev_group_defect"):
        for target in ("mesh_fixed_nmse", "operator_nmse", "propagation_h3"):
            metrics[f"{predictor}_vs_{target}"] = correlation_summary(probe_records, predictor, target)

    primary = metrics["mean_prev_group_defect_vs_operator_nmse"]
    centered_signal = primary.get("step_block_centered_spearman")
    lopo_signal = primary.get("lopo_centered_spearman_median")
    pass_gate = (
        centered_signal is not None and centered_signal >= 0.25
        and lopo_signal is not None and lopo_signal >= 0.20
    )
    decision = "RUN_ADAPTIVE_K_SCREEN" if pass_gate else "STOP_AND_RETHINK_BUDGET_SIGNAL"

    result = {
        "schema_version": "coframe.stage1d.lagged.v1",
        "run_count": len(runs),
        "prompt_count": len(observations),
        "probe_record_count": len(probe_records),
        "causal_contract": "previous completed block-group defect -> next block-group budget",
        "metrics": metrics,
        "gate": {
            "heuristic_not_preregistered": True,
            "primary": "mean_prev_group_defect_vs_operator_nmse",
            "required_step_block_centered_spearman": 0.25,
            "required_lopo_centered_spearman_median": 0.20,
            "decision": decision,
        },
    }
    plans = {
        "schema_version": "coframe.stage1d.budget_plan.v1",
        "budget_values": list(DEFAULT_BUDGETS),
        "budget_quantiles": list(DEFAULT_QUANTILES),
        "target_mean_k": 9.0,
        "kv_mode": args.kv_mode,
        "folds": build_fold_plans(observations, budgets=DEFAULT_BUDGETS, quantiles=DEFAULT_QUANTILES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.plan_output.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

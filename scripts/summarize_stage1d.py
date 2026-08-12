from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch


FOLD_RE = re.compile(r"p\d+_s\d+")


def fold_id(path: Path) -> str:
    match = FOLD_RE.search(str(path))
    return match.group(0) if match else path.parent.name


def finite_median(values):
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return median(usable) if usable else None


def latent_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    ref_energy = ref.square().mean().clamp_min(1.0e-12)
    nmse = float((diff.square().mean() / ref_energy).item())
    rel_l2 = float((diff.norm() / ref.norm().clamp_min(1.0e-12)).item())
    cosine = float(torch.nn.functional.cosine_similarity(ref.flatten(), cand.flatten(), dim=0).item())
    if ref.shape[2] > 1:
        ref_dt = ref[:, :, 1:] - ref[:, :, :-1]
        cand_dt = cand[:, :, 1:] - cand[:, :, :-1]
        temporal_rel_l2 = float(((cand_dt - ref_dt).norm() / ref_dt.norm().clamp_min(1.0e-12)).item())
    else:
        temporal_rel_l2 = 0.0
    return {
        "endpoint_nmse": nmse,
        "endpoint_relative_l2": rel_l2,
        "endpoint_cosine": cosine,
        "endpoint_temporal_gradient_relative_l2": temporal_rel_l2,
    }


def policy_name(summary: dict[str, Any]) -> str:
    config = summary.get("config", {})
    method = summary.get("method", config.get("method", "unknown"))
    if method == "adaptive_k":
        values = [int(value) for value in config.get("adaptive_k_values", [])]
        # The dedicated calibration pass uses the identical adaptive runtime but
        # constrains the only possible budget to K=9. It is therefore the clean
        # fixed-budget baseline for Stage-1d.
        if values == [9]:
            return "fixed_k9"
        return str(config.get("adaptive_k_policy", "adaptive_k"))
    if method == "fixed":
        return "fixed_k9"
    return str(method)


def read_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_metrics(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policy = policy_name(summary)
    trace_path = summary_path.parent / "trace.json"
    anchor_budgets: list[int] = []
    operator_nmse: list[float] = []
    propagation_h3: list[float] = []
    budget_events: list[dict[str, Any]] = []
    if trace_path.exists():
        trace = read_trace(trace_path)
        for event in trace.get("events", []):
            if event.get("event") != "transformer_forward":
                continue
            for anchors in event.get("block_anchors", {}).values():
                anchor_budgets.append(len(anchors))
            budget_events.extend(event.get("budget_events", []))
            for probe in event.get("probes", []):
                value = probe.get("block_delta_normalized_mse")
                if value is not None:
                    operator_nmse.append(float(value))
                value = probe.get("propagated_relative_l2_h3")
                if value is not None:
                    propagation_h3.append(float(value))
    avg_k = mean(anchor_budgets) if anchor_budgets else None
    return {
        "fold_id": fold_id(summary_path),
        "policy": policy,
        "summary_path": str(summary_path),
        "latent_path": str(summary_path.parent / "latents.pt"),
        "avg_k": avg_k,
        "avg_k_relative_error_vs_9": None if avg_k is None else abs(avg_k - 9.0) / 9.0,
        "budget_match_within_5pct": None if avg_k is None else abs(avg_k - 9.0) / 9.0 <= 0.05,
        "median_operator_nmse": finite_median(operator_nmse),
        "median_propagation_h3": finite_median(propagation_h3),
        "budget_event_count": len(budget_events),
        "denoise_time_sec": summary.get("denoise_time_sec"),
    }


def load_latent(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload["latents"] if isinstance(payload, dict) else payload
    return value.detach().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage-1d budget-matched adaptive-K runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [run_metrics(path) for path in sorted(args.root.rglob("summary.json"))]
    dense_by_fold = {run["fold_id"]: run for run in runs if run["policy"] == "dense"}
    dense_latents: dict[str, torch.Tensor] = {}
    for fold, run in dense_by_fold.items():
        if Path(run["latent_path"]).exists():
            dense_latents[fold] = load_latent(run["latent_path"])

    for run in runs:
        if run["policy"] == "dense":
            run.update({
                "endpoint_nmse": 0.0,
                "endpoint_relative_l2": 0.0,
                "endpoint_cosine": 1.0,
                "endpoint_temporal_gradient_relative_l2": 0.0,
            })
            continue
        reference = dense_latents.get(run["fold_id"])
        latent_path = Path(run["latent_path"])
        if reference is not None and latent_path.exists():
            run.update(latent_metrics(reference, load_latent(str(latent_path))))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        buckets[run["policy"]].append(run)

    aggregate = {}
    for policy, items in sorted(buckets.items()):
        aggregate[policy] = {
            "run_count": len(items),
            "median_avg_k": finite_median(item.get("avg_k") for item in items),
            "all_budget_matched_within_5pct": all(
                item.get("budget_match_within_5pct") is True for item in items if item.get("avg_k") is not None
            ) if any(item.get("avg_k") is not None for item in items) else None,
            "median_operator_nmse": finite_median(item.get("median_operator_nmse") for item in items),
            "median_propagation_h3": finite_median(item.get("median_propagation_h3") for item in items),
            "median_endpoint_nmse": finite_median(item.get("endpoint_nmse") for item in items),
            "median_endpoint_relative_l2": finite_median(item.get("endpoint_relative_l2") for item in items),
            "median_endpoint_cosine": finite_median(item.get("endpoint_cosine") for item in items),
            "median_endpoint_temporal_gradient_relative_l2": finite_median(
                item.get("endpoint_temporal_gradient_relative_l2") for item in items
            ),
        }

    fixed = aggregate.get("fixed_k9")
    if fixed:
        for policy in ("step_block", "mean_defect", "max_defect"):
            current = aggregate.get(policy)
            if not current:
                continue
            for metric in ("median_operator_nmse", "median_propagation_h3", "median_endpoint_nmse"):
                base = fixed.get(metric)
                value = current.get(metric)
                if base not in (None, 0) and value is not None:
                    current[f"relative_improvement_over_fixed_{metric}"] = (base - value) / base

    result = {
        "schema_version": "coframe.stage1d.summary.v1",
        "latency_policy": "recorded_but_not_a_primary_claim_unless_gpus_are_exclusive",
        "runs": runs,
        "aggregate": aggregate,
        "primary_decision_rule": (
            "mean_defect is supported only if budget matched, it beats fixed_k9 and step_block on operator NMSE, "
            "retains the sign at +3 propagation and endpoint NMSE, and max_defect does not explain the gain better."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

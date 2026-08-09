"""Analysis utilities for CoFrame JSONL probe outputs."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean
from typing import Any

from .metrics import pearson, spearman


GROUP_KEYS = ("prompt_id", "seed", "step_index", "block_index")
PREDICTORS = ("coframe_defect", "rhyme_novelty", "proxy_interp_error")
TARGETS = ("omission_frame_error", "frame_error_gain", "full_error_gain")


def _group(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)
    return groups


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if isfinite(value)]
    return mean(finite) if finite else float("nan")


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("kind") == "candidate"]
    groups = _group(candidates)
    group_summaries: list[dict[str, Any]] = []
    regrets: dict[str, list[float]] = {predictor: [] for predictor in PREDICTORS}
    oracle_capture: dict[str, list[float]] = {predictor: [] for predictor in PREDICTORS}

    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {name: value for name, value in zip(GROUP_KEYS, key)}
        summary["count"] = len(items)
        for target in TARGETS:
            target_values = [float(item[target]) for item in items]
            for predictor in PREDICTORS:
                predictor_values = [float(item[predictor]) for item in items]
                summary[f"spearman.{target}.{predictor}"] = spearman(
                    predictor_values, target_values
                )
                summary[f"pearson.{target}.{predictor}"] = pearson(
                    predictor_values, target_values
                )

        oracle = max(float(item["full_error_gain"]) for item in items)
        summary["oracle_full_error_gain"] = oracle
        for predictor in PREDICTORS:
            chosen = max(
                items,
                key=lambda item: (float(item[predictor]), -int(item["candidate"])),
            )
            achieved = float(chosen["full_error_gain"])
            regret = oracle - achieved
            capture = achieved / oracle if oracle > 1e-12 else float("nan")
            summary[f"chosen_candidate.{predictor}"] = int(chosen["candidate"])
            summary[f"chosen_gain.{predictor}"] = achieved
            summary[f"regret.{predictor}"] = regret
            summary[f"oracle_capture.{predictor}"] = capture
            regrets[predictor].append(regret)
            oracle_capture[predictor].append(capture)
        group_summaries.append(summary)

    aggregate: dict[str, Any] = {
        "candidate_rows": len(candidates),
        "groups": len(groups),
    }
    for target in TARGETS:
        for predictor in PREDICTORS:
            aggregate[f"mean_spearman.{target}.{predictor}"] = _safe_mean(
                [
                    float(group[f"spearman.{target}.{predictor}"])
                    for group in group_summaries
                ]
            )
            aggregate[f"mean_pearson.{target}.{predictor}"] = _safe_mean(
                [
                    float(group[f"pearson.{target}.{predictor}"])
                    for group in group_summaries
                ]
            )
    for predictor in PREDICTORS:
        aggregate[f"mean_regret.{predictor}"] = _safe_mean(regrets[predictor])
        aggregate[f"mean_oracle_capture.{predictor}"] = _safe_mean(
            oracle_capture[predictor]
        )

    defect_corr = aggregate.get(
        "mean_spearman.full_error_gain.coframe_defect", float("nan")
    )
    rhyme_corr = aggregate.get(
        "mean_spearman.full_error_gain.rhyme_novelty", float("nan")
    )
    proxy_corr = aggregate.get(
        "mean_spearman.full_error_gain.proxy_interp_error", float("nan")
    )
    strongest_baseline = max(
        [value for value in (rhyme_corr, proxy_corr) if isfinite(value)],
        default=float("nan"),
    )
    aggregate["gate_correlation_threshold"] = 0.55
    aggregate["gate_margin_threshold"] = 0.15
    aggregate["gate_defect_correlation_pass"] = bool(
        isfinite(defect_corr) and defect_corr >= 0.55
    )
    aggregate["gate_defect_margin_pass"] = bool(
        isfinite(defect_corr)
        and isfinite(strongest_baseline)
        and defect_corr - strongest_baseline >= 0.15
    )
    aggregate["gate_pass"] = bool(
        aggregate["gate_defect_correlation_pass"]
        and aggregate["gate_defect_margin_pass"]
    )
    return {"aggregate": aggregate, "groups": group_summaries}


def summarize_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("kind") not in ("method", "core_diagnostic"):
            continue
        values[str(row["method"])].append(float(row["block_relative_rms"]))
    return {
        method: {
            "count": len(errors),
            "mean_block_relative_rms": mean(errors),
            "min_block_relative_rms": min(errors),
            "max_block_relative_rms": max(errors),
        }
        for method, errors in sorted(values.items())
    }

"""Small dependency-light metrics for probe analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import sqrt
from typing import Any

import torch


def relative_l2(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    reference_f = reference.float().reshape(-1)
    estimate_f = estimate.float().reshape(-1)
    return float(
        ((reference_f - estimate_f).norm() / reference_f.norm().clamp_min(eps)).item()
    )


def cosine_similarity(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    reference_f = reference.float().reshape(-1)
    estimate_f = estimate.float().reshape(-1)
    denominator = reference_f.norm() * estimate_f.norm()
    return float((torch.dot(reference_f, estimate_f) / denominator.clamp_min(eps)).item())


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator_x = sqrt(sum((a - mean_x) ** 2 for a in x))
    denominator_y = sqrt(sum((b - mean_y) ** 2 for b in y))
    if denominator_x == 0 or denominator_y == 0:
        return float("nan")
    return numerator / (denominator_x * denominator_y)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_average_ranks(x), _average_ranks(y))


def grouped_correlations(
    rows: Iterable[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    target: str,
    predictors: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, items in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {key: value for key, value in zip(group_keys, group)}
        summary["count"] = len(items)
        target_values = [float(item[target]) for item in items]
        for predictor in predictors:
            predictor_values = [float(item[predictor]) for item in items]
            summary[f"spearman_{predictor}"] = spearman(
                predictor_values, target_values
            )
            summary[f"pearson_{predictor}"] = pearson(
                predictor_values, target_values
            )
        output.append(summary)
    return output

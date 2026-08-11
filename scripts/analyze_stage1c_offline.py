from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


PROMPT_RE = re.compile(r"p\d+_s\d+")


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        rank = 0.5 * (left + right - 1)
        for slot in range(left, right):
            ranks[order[slot]] = rank
        left = right
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom <= 1e-12:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    return _pearson(_rank(x), _rank(y))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _prompt_id(trace: Path, root: Path) -> str:
    for part in trace.relative_to(root).parts:
        if PROMPT_RE.fullmatch(part):
            return part
    return trace.parent.parent.name


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def _anchor_displacement(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    scale = max(1, max(max(left), max(right)))
    return mean(abs(a - b) for a, b in zip(sorted(left), sorted(right))) / scale


def _pair_stats(groups: Iterable[list[tuple[int, ...]]]) -> dict[str, Any]:
    jaccards: list[float] = []
    displacements: list[float] = []
    exact = 0
    pairs = 0
    for meshes in groups:
        for left, right in combinations(meshes, 2):
            pairs += 1
            score = _jaccard(left, right)
            jaccards.append(score)
            disp = _anchor_displacement(left, right)
            if math.isfinite(disp):
                displacements.append(disp)
            exact += int(left == right)
    return {
        "pair_count": pairs,
        "mean_jaccard": mean(jaccards) if jaccards else None,
        "median_jaccard": median(jaccards) if jaccards else None,
        "exact_set_rate": exact / pairs if pairs else None,
        "mean_normalized_anchor_displacement": mean(displacements) if displacements else None,
        "median_normalized_anchor_displacement": median(displacements) if displacements else None,
    }


def _collect(root: Path, signal: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in sorted(root.rglob("trace.json")):
        payload = json.loads(trace.read_text(encoding="utf-8"))
        config = payload.get("run", {}).get("config", {})
        if config.get("refresh_signal", "unknown") != signal:
            continue
        prompt_id = _prompt_id(trace, root)
        for event in payload.get("events", []):
            if event.get("event") != "transformer_forward":
                continue
            defect_map: dict[tuple[int, int], list[float]] = {}
            for record in event.get("defects", []):
                values = [_finite(v) for v in (record.get("values") or {}).values()]
                defect_map[(int(record["step"]), int(record["block"]))] = [v for v in values if v is not None]
            for probe in event.get("probes", []):
                step, block = int(probe["step"]), int(probe["block"])
                oracle = probe.get("mesh_only", {}).get("oracle", {})
                anchors = oracle.get("anchors")
                if anchors is None:
                    continue
                raw_defects = defect_map.get((step, block), [])
                if not raw_defects:
                    projected = [_finite(v) for v in probe.get("current_defect_expected_error", [])]
                    raw_defects = [v for v in projected if v is not None and v > 0]
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "step": step,
                        "block": block,
                        "oracle_anchors": tuple(int(v) for v in anchors),
                        "mean_defect": mean(raw_defects) if raw_defects else None,
                        "max_defect": max(raw_defects) if raw_defects else None,
                        "mesh_oracle_nmse": _finite(probe.get("mesh_oracle_nmse")),
                        "mesh_fixed_nmse": _finite(probe.get("mesh_fixed_nmse")),
                        "operator_nmse": _finite(probe.get("block_delta_normalized_mse")),
                        "propagation_h3": _finite(probe.get("propagated_relative_l2_h3")),
                    }
                )
    return rows


def _correlation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictors = ("mean_defect", "max_defect")
    targets = ("mesh_oracle_nmse", "mesh_fixed_nmse", "operator_nmse", "propagation_h3")
    result: dict[str, Any] = {}
    prompts = sorted({row["prompt_id"] for row in rows})
    for predictor in predictors:
        for target in targets:
            valid = [row for row in rows if row[predictor] is not None and row[target] is not None]
            x = [float(row[predictor]) for row in valid]
            y = [float(row[target]) for row in valid]
            lopo: list[float] = []
            for held_out in prompts:
                subset = [row for row in valid if row["prompt_id"] != held_out]
                corr = _spearman(
                    [float(row[predictor]) for row in subset],
                    [float(row[target]) for row in subset],
                )
                if corr is not None:
                    lopo.append(corr)
            result[f"{predictor}_vs_{target}"] = {
                "count": len(valid),
                "pearson": _pearson(x, y),
                "spearman": _spearman(x, y),
                "lopo_spearman_median": median(lopo) if lopo else None,
                "lopo_spearman_min": min(lopo) if lopo else None,
                "lopo_spearman_max": max(lopo) if lopo else None,
            }
    return result


def _oracle_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    by_prompt_step: dict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    by_prompt_block: dict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    for row in rows:
        mesh = row["oracle_anchors"]
        by_cell[(row["step"], row["block"])].append(mesh)
        by_prompt_step[(row["prompt_id"], row["step"])].append(mesh)
        by_prompt_block[(row["prompt_id"], row["block"])].append(mesh)

    consensus: dict[str, Any] = {}
    for key, meshes in sorted(by_cell.items()):
        unique = sorted(set(meshes))
        medoid = max(unique, key=lambda candidate: mean(_jaccard(candidate, other) for other in meshes))
        consensus[f"step{key[0]}_block{key[1]}"] = {
            "medoid_anchors": list(medoid),
            "prompt_count": len(meshes),
            "exact_medoid_support": sum(mesh == medoid for mesh in meshes),
            "mean_jaccard_to_medoid": mean(_jaccard(medoid, mesh) for mesh in meshes),
        }

    return {
        "same_step_block_across_prompts": _pair_stats(by_cell.values()),
        "same_prompt_step_across_blocks": _pair_stats(by_prompt_step.values()),
        "same_prompt_block_across_steps": _pair_stats(by_prompt_block.values()),
        "consensus_meshes": consensus,
    }


def _decision(oracle: dict[str, Any], correlations: dict[str, Any]) -> dict[str, Any]:
    cross_prompt = oracle["same_step_block_across_prompts"]
    jaccard = cross_prompt.get("median_jaccard") or 0.0
    displacement = cross_prompt.get("median_normalized_anchor_displacement")
    displacement = displacement if displacement is not None else 1.0
    schedule_evidence = jaccard >= 0.78 and displacement <= 0.08

    risk_candidates = []
    for name in (
        "mean_defect_vs_propagation_h3",
        "max_defect_vs_propagation_h3",
        "mean_defect_vs_operator_nmse",
        "max_defect_vs_operator_nmse",
        "mean_defect_vs_mesh_fixed_nmse",
        "max_defect_vs_mesh_fixed_nmse",
    ):
        payload = correlations.get(name, {})
        corr = payload.get("spearman")
        lopo = payload.get("lopo_spearman_median")
        if corr is not None and lopo is not None:
            risk_candidates.append((name, float(corr), float(lopo)))
    strongest = max(risk_candidates, key=lambda item: item[1], default=None)
    block_risk_evidence = bool(strongest and strongest[1] >= 0.50 and strongest[2] >= 0.40)

    priorities: list[str] = []
    if schedule_evidence:
        priorities.append("TEST_CALIBRATED_STEP_BLOCK_MESH")
    if block_risk_evidence:
        priorities.append("TEST_ADAPTIVE_K_OR_DENSE_BLOCK_GATING")
    if not priorities:
        priorities.append("RUN_INPUT_CURVATURE_SIGNAL_SCREEN")
    return {
        "heuristic_not_preregistered": True,
        "oracle_prompt_stability_supported": schedule_evidence,
        "defect_magnitude_block_risk_supported": block_risk_evidence,
        "strongest_defect_block_risk_correlation": strongest,
        "next_action_priority": priorities,
        "note": "These thresholds are triage heuristics. Inspect the paired distributions before making a paper claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1c zero-GPU analysis of Stage-1b traces.")
    parser.add_argument("--root", type=Path, required=True, help="Stage-1b output root containing trace.json files")
    parser.add_argument("--signal", default="defect", choices=["defect", "none", "gap_only", "shuffled"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _collect(args.root, args.signal)
    if not rows:
        raise SystemExit(f"No Stage-1b probes found for refresh_signal={args.signal!r} under {args.root}")
    oracle = _oracle_report(rows)
    correlations = _correlation_report(rows)
    result = {
        "schema_version": "coframe.stage1c.offline.v1",
        "input_root": str(args.root),
        "refresh_signal": args.signal,
        "cell_count": len(rows),
        "prompt_count": len({row["prompt_id"] for row in rows}),
        "oracle_structure": oracle,
        "defect_magnitude_block_risk": correlations,
        "decision": _decision(oracle, correlations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

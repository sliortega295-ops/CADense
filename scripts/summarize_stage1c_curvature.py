from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


POLICIES = (
    "gap_only",
    "input_curvature",
    "shuffled_input_curvature",
    "previous_delta_curvature",
    "shuffled_previous_delta_curvature",
)


def _action(payload: dict[str, Any] | None) -> tuple[Any, Any, bool] | None:
    if not payload:
        return None
    chosen = payload.get("predicted_best_swap")
    if not chosen:
        return None
    return chosen.get("removed"), chosen.get("added"), bool(chosen.get("noop", False))


def _iter_probes(root: Path):
    for trace in sorted(root.rglob("trace.json")):
        payload = json.loads(trace.read_text(encoding="utf-8"))
        signal = payload.get("run", {}).get("config", {}).get("refresh_signal", "unknown")
        for event in payload.get("events", []):
            if event.get("event") != "transformer_forward":
                continue
            for probe in event.get("probes", []):
                yield signal, probe


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "positive_rate": None, "harmful_rate": None}
    return {
        "count": len(values),
        "median": median(values),
        "mean": mean(values),
        "positive_rate": sum(value > 0 for value in values) / len(values),
        "harmful_rate": sum(value < 0 for value in values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage-1c curvature signal probes.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    action_matches: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for trajectory, probe in _iter_probes(args.root):
        swaps = probe.get("swap_decision", {})
        oracle = swaps.get("gap_only", {}).get("oracle_best_swap")
        actionable = oracle is not None
        for policy in POLICIES:
            payload = swaps.get(policy)
            if not payload:
                continue
            recovery = payload.get("gain_recovery")
            regret = payload.get("normalized_regret")
            if actionable and recovery is not None:
                buckets[trajectory][f"gain_recovery.{policy}"].append(float(recovery))
            if regret is not None:
                buckets[trajectory][f"normalized_regret.{policy}"].append(float(regret))
            chosen = payload.get("predicted_best_swap") or {}
            if actionable and chosen:
                true_gain = chosen.get("true_gain")
                if true_gain is not None:
                    buckets[trajectory][f"chosen_true_gain.{policy}"].append(float(true_gain))

        for base, shuffled in (
            ("input_curvature", "shuffled_input_curvature"),
            ("previous_delta_curvature", "shuffled_previous_delta_curvature"),
        ):
            left, right = _action(swaps.get(base)), _action(swaps.get(shuffled))
            if left is not None and right is not None:
                action_matches[trajectory][base].append(int(left == right))

    result: dict[str, Any] = {"schema_version": "coframe.stage1c.curvature_summary.v1", "trajectories": {}}
    for trajectory, metrics in sorted(buckets.items()):
        payload: dict[str, Any] = {name: _summary(values) for name, values in sorted(metrics.items())}
        for base, values in action_matches.get(trajectory, {}).items():
            payload[f"same_action_rate.{base}_vs_shuffled"] = sum(values) / len(values) if values else None

        def med(name: str) -> float | None:
            values = metrics.get(name, [])
            return median(values) if values else None

        gap = med("gain_recovery.gap_only")
        input_curv = med("gain_recovery.input_curvature")
        shuffled_input = med("gain_recovery.shuffled_input_curvature")
        prev_curv = med("gain_recovery.previous_delta_curvature")
        shuffled_prev = med("gain_recovery.shuffled_previous_delta_curvature")
        payload["paired_direction"] = {
            "input_curvature_minus_gap_median": None if input_curv is None or gap is None else input_curv - gap,
            "input_curvature_minus_shuffled_median": None if input_curv is None or shuffled_input is None else input_curv - shuffled_input,
            "previous_delta_curvature_minus_gap_median": None if prev_curv is None or gap is None else prev_curv - gap,
            "previous_delta_curvature_minus_shuffled_median": None if prev_curv is None or shuffled_prev is None else prev_curv - shuffled_prev,
        }
        result["trajectories"][trajectory] = payload

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

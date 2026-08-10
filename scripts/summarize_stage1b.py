from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def probes_from_trace(path: Path):
    payload = json.loads(path.read_text())
    signal = payload.get("run", {}).get("config", {}).get("refresh_signal", "unknown")
    for event in payload.get("events", []):
        if event.get("event") != "transformer_forward":
            continue
        for probe in event.get("probes", []):
            yield signal, probe


def safe_median(values):
    values = [float(v) for v in values if v is not None]
    return median(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    buckets = defaultdict(lambda: defaultdict(list))
    for trace in args.root.rglob("trace.json"):
        for signal, probe in probes_from_trace(trace):
            current = probe.get("mesh_current_nmse")
            for baseline in ("rhyme", "fis", "fixed"):
                base = probe.get(f"mesh_{baseline}_nmse")
                if current is not None and base not in (None, 0):
                    buckets[signal][f"mesh_improvement_over_{baseline}"].append((float(base) - float(current)) / float(base))
                op = probe.get(f"operator_nmse_relative_improvement_over_{baseline}")
                if op is not None:
                    buckets[signal][f"operator_improvement_over_{baseline}"].append(op)
                for horizon in (1, 3):
                    prop = probe.get(f"propagation_relative_improvement_over_{baseline}_h{horizon}")
                    if prop is not None:
                        buckets[signal][f"prop_h{horizon}_improvement_over_{baseline}"].append(prop)
            for policy, field in (
                ("defect", "swap_post_observation_gain_recovery"),
                ("gap_only", "swap_gap_only_gain_recovery"),
                ("shuffled", "swap_shuffled_gain_recovery"),
            ):
                value = probe.get(field)
                if value is not None:
                    buckets[signal][f"local_swap_gain_recovery_{policy}"].append(value)

    result = {}
    for signal, metrics in sorted(buckets.items()):
        result[signal] = {name: {"count": len(values), "median": safe_median(values), "win_rate": sum(float(v) > 0 for v in values) / len(values) if values else None, "harmful_rate": sum(float(v) < 0 for v in values) / len(values) if values else None} for name, values in sorted(metrics.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

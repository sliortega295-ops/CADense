#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:?prompt required}
OUT_ROOT=${2:?output root required}
SEED=${3:-0}
RUN_ID=${4:?run id such as p0_s0 required}
PLAN=${5:?trajectory interaction plan json required}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Fail closed on the preregistered six-pair interaction screen. The probe arms
# branch counterfactually from a deployed uniform-K9 trajectory; the plan is
# metadata only and must not contain a learned or prompt-specific schedule.
python - "$PLAN" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("trajectory interaction plan must be a JSON object")

required_top_level = {
    "schema_version",
    "experiment_id",
    "base_trajectory_k",
    "arms",
    "pairs",
}
if set(payload) != required_top_level:
    raise SystemExit(
        "trajectory interaction plan keys must be exactly "
        f"{sorted(required_top_level)}"
    )
if not isinstance(payload["schema_version"], str) or not payload["schema_version"]:
    raise SystemExit("trajectory interaction plan requires a non-empty schema_version")
if not isinstance(payload["experiment_id"], str) or not payload["experiment_id"]:
    raise SystemExit("trajectory interaction plan requires a non-empty experiment_id")
if payload["base_trajectory_k"] != 9:
    raise SystemExit("trajectory interaction plan requires base_trajectory_k=9")

expected_arms = [
    "k9_k9",
    "k6_k9",
    "k9_k12",
    "k6_k12",
    "k12_k9",
    "k9_k6",
    "k12_k6",
]
if payload["arms"] != expected_arms:
    raise SystemExit(f"trajectory interaction arms must be exactly {expected_arms}")

expected_pairs = [
    (5, 0, 1, "adjacent"),
    (5, 0, 7, "long"),
    (20, 3, 4, "adjacent"),
    (20, 0, 7, "long"),
    (40, 6, 7, "adjacent"),
    (40, 0, 7, "long"),
]
pairs = payload["pairs"]
if not isinstance(pairs, list) or len(pairs) != len(expected_pairs):
    raise SystemExit("trajectory interaction plan must contain exactly six pairs")

seen_ids = set()
actual_pairs = []
required_pair_keys = {"pair_id", "step", "group_i", "group_j", "distance"}
for index, pair in enumerate(pairs):
    if not isinstance(pair, dict) or set(pair) != required_pair_keys:
        raise SystemExit(
            f"pair {index} keys must be exactly {sorted(required_pair_keys)}"
        )
    pair_id = pair["pair_id"]
    if not isinstance(pair_id, str) or not pair_id or pair_id in seen_ids:
        raise SystemExit(f"pair {index} requires a unique non-empty pair_id")
    seen_ids.add(pair_id)
    step = pair["step"]
    group_i = pair["group_i"]
    group_j = pair["group_j"]
    distance = pair["distance"]
    if not all(type(value) is int for value in (step, group_i, group_j)):
        raise SystemExit(f"pair {pair_id}: step and groups must be integers")
    if not (step in {5, 20, 40} and 0 <= group_i < group_j < 8):
        raise SystemExit(f"pair {pair_id}: step/groups are outside the preregistered domain")
    actual_pairs.append((step, group_i, group_j, distance))

if actual_pairs != expected_pairs:
    raise SystemExit(f"trajectory interaction pairs must be exactly {expected_pairs}")
print(
    "trajectory_interaction_plan_ok "
    f"experiment_id={payload['experiment_id']} pairs={len(pairs)} arms={len(expected_arms)}"
)
PY

mkdir -p "$OUT_ROOT"

# The deployed path is the preregistered uniform-K9 Adaptive-K step/group
# runtime. All seven arms are counterfactual and discarded after measurement;
# no video decode or latency comparison is requested for this signal screen.
python -m coframe.cli \
  --prompt "$PROMPT" --seed "$SEED" \
  --height 480 --width 832 --num-frames 81 \
  --steps 50 --guidance-scale 5 --flow-shift 3 \
  --warmup-steps 5 --num-anchors 9 \
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
  --kv-mode full_kv \
  --interpolation-target delta --defect-target delta \
  --method adaptive_k \
  --adaptive-k-policy step_block \
  --adaptive-k-values 6,9,12,21 \
  --trajectory-interaction-plan-json "$PLAN" \
  --local-files-only \
  --output-dir "$OUT_ROOT" \
  --run-name "${RUN_ID}_trajectory_interaction_uniform_k9"

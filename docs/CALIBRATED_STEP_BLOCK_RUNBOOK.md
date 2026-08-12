# Calibrated Step-Block Budget Schedule

This experiment tests a prompt-independent `K(step, block-group)` schedule. It
does not use defect, curvature, Proxy-DP, or any other online signal.

## Fixed contract

- Wan2.1-T2V-1.3B-Diffusers, 480x832, 81 decoded / 21 latent frames;
- 50 denoising steps with five dense warm-up steps;
- sparse blocks 3-26, grouped as eight groups of three blocks;
- `full_kv`, delta interpolation, K in `{6,9,12,21}`;
- eight prompts, seed 0;
- four independent GPU jobs, no DDP;
- latency is not a primary result and is reported as `NOT_REPORTED`.

## Phase A: matched-input error surface

For each prompt:

```bash
bash scripts/run_calibrated_budget_surface_wan21.sh \
  "<prompt>" outputs/calibrated_step_block/surface/p0_s0 0 p0_s0
```

The deployed trajectory is uniform K=9. At every `(step, group)` entry, the
runtime discards four counterfactual branches at K=6/9/12/21. Each branch runs
the complete three-block sparse group, compares its cumulative group delta to
the complete three-block dense reference, and then propagates both states
through three identical dense blocks. These probes never affect generation.

Expected surface size is `8 prompts x 45 steps x 8 groups x 4 K = 11,520`
rows. Build the schedules only after this exact completeness gate passes:

```bash
python scripts/calibrate_step_block_schedule.py \
  --root outputs/calibrated_step_block/surface \
  --output outputs/calibrated_step_block/plan/lopo_plan.json \
  --surface-csv outputs/calibrated_step_block/plan/budget_error_surface.csv \
  --schedule-dir outputs/calibrated_step_block/plan/schedules
```

## Phase B: exact-budget LOPO calibration

For each held-out prompt, only the other seven prompts contribute costs. The
cost of a slot/K pair is their mean operator NMSE. Exact dynamic programming
minimizes the sum over all 360 slots under total K=3240, i.e. exact average K=9.
No held-out row may affect its schedule.

The primary objective is operator NMSE. +3 propagation is preserved for
diagnosis and held-out evaluation but is not used to choose the schedule.

## Phase C: held-out evaluation

For every fold:

```bash
bash scripts/run_calibrated_budget_eval_wan21.sh \
  "<prompt>" outputs/calibrated_step_block/eval/p0_s0 0 \
  outputs/calibrated_step_block/plan/schedules/p0_s0.json p0_s0
```

This runs a dense endpoint reference and the frozen held-out step/group
schedule through the existing Adaptive-K `step_block` runtime. The Phase-A
uniform trajectory is the K=9 baseline.

Aggregate with:

```bash
python scripts/summarize_calibrated_step_block.py \
  --surface-root outputs/calibrated_step_block/surface \
  --eval-root outputs/calibrated_step_block/eval \
  --plan outputs/calibrated_step_block/plan/lopo_plan.json \
  --output outputs/calibrated_step_block/summary.json \
  --cells-csv outputs/calibrated_step_block/heldout_group_cells.csv \
  --report reports/calibrated_step_block/REPORT.md
```

## Preregistered decision gate

Output `SUPPORT_CALIBRATED_STEP_BLOCK_BUDGET` only if all conditions hold:

1. all eight actual schedule averages lie within 5% of K=9;
2. mean group operator NMSE has positive median paired improvement and wins at
   least 6/8 held-out prompts;
3. mean +3 propagation relative-L2 has the same requirements;
4. dense-referenced endpoint latent NMSE has the same requirements;
5. endpoint temporal-gradient relative-L2 has the same requirements;
6. all folds, traces, schedules, and held-out leakage checks are complete.

Otherwise output `REJECT_CALIBRATED_STEP_BLOCK_BUDGET`. Do not change the K
set, total-budget constraint, objective, LOPO split, or decision gate after
inspecting results.

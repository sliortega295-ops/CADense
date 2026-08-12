# Stage-1d: Causal Adaptive Exact-Frame Budget

Stage-1d tests whether defect magnitude should control **how much exact computation** a future block group receives. It does not use defect localization to choose frame positions.

## Causal contract

For sparse group `g`, collect leave-one-out defects after the group finishes. Only this completed observation may control the exact-frame budget of group `g+1`. The first sparse group starts at K=9; the final group of one denoising step may causally control the first sparse group of the next step.

Frame placement is deterministic uniform placement for every K. The mechanism screen uses `full_kv` so changing K primarily changes the number of exact query frames rather than simultaneously changing K/V context.

## Phase A: zero-GPU gate on preserved Stage-1b traces

```bash
python scripts/analyze_stage1d_lagged.py \
  --root <STAGE1B_ROOT> \
  --output outputs/stage1d/lagged_stage1b.json \
  --plan-output outputs/stage1d/stage1b_screen_plan.json
```

Proceed only if `gate.decision == RUN_ADAPTIVE_K_SCREEN`. Otherwise stop and report the failed lagged-causality test.

## Phase B: clean uniform-K9 calibration

Run the same eight prompts, seed 0. Each GPU can process prompts independently; no DDP is needed.

```bash
bash scripts/run_stage1d_calibration_wan21.sh \
  "<prompt>" outputs/stage1d_gpu/calibration/p0_s0 0 p0_s0
```

This uses the adaptive runtime with its only allowed budget set to `{9}`. It therefore serves both as the clean fixed-K9 baseline and as the source of defect distributions for final LOPO calibration. It deliberately does not reuse Stage-1b remeshed defect distributions.

After all eight calibration runs finish:

```bash
python scripts/analyze_stage1d_lagged.py \
  --root outputs/stage1d_gpu/calibration \
  --output outputs/stage1d/calibration_lagged.json \
  --plan-output outputs/stage1d/budget_plan.json
```

Require this clean calibration analysis to emit `RUN_ADAPTIVE_K_SCREEN` as well. If it does not, stop before adaptive GPU evaluation.

## Phase C: held-out Adaptive-K evaluation

For each held-out prompt, `budget_plan.json` contains thresholds/schedules calibrated only on the other seven prompts. Run:

```bash
bash scripts/run_stage1d_adaptive_k_wan21.sh \
  "<prompt>" outputs/stage1d_gpu/eval/p0_s0 0 \
  outputs/stage1d/budget_plan.json p0_s0
```

This evaluates four trajectories per prompt:

- dense endpoint reference;
- prompt-independent step/group budget schedule;
- previous-group mean-defect Adaptive-K (primary);
- previous-group max-defect Adaptive-K (ablation).

The separate Phase-B calibration trajectory is the fixed K=9 baseline. Default adaptive budgets are `{6,9,12,21}`; thresholds are LOPO-calibrated with quantiles `{0.35,0.80,0.95}`, targeting mean K approximately 9.

## Phase D: aggregate

Use a root containing both `calibration/` and `eval/`:

```bash
python scripts/summarize_stage1d.py \
  --root outputs/stage1d_gpu \
  --output outputs/stage1d_gpu/summary.json
```

The primary result is supported only if mean-defect Adaptive-K:

1. stays within 5% of the K=9 average exact-frame budget;
2. beats the uniform K=9 calibration baseline on realized operator NMSE;
3. beats the prompt-independent step/group schedule;
4. keeps the same improvement sign after +3 dense propagation and at the dense-referenced endpoint;
5. is not explained equally well or better by the max-defect ablation.

Do not tune thresholds after looking at held-out outcomes. If budget matching fails, report it as a failed fairness gate rather than silently changing thresholds. Latency may be recorded but is not a primary claim when GPUs are shared. Only after this mechanism gate passes should `anchor_only` be used for a real speed study.

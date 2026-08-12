# Wan2.1-1.3B experiment plan

## Experimental contract

Use one environment and record its exact package versions, GPU model, driver, CUDA version, model revision, prompt, negative prompt, seed, and scheduler settings. The canonical contract is:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`;
- Diffusers 0.34.0;
- 480 × 832, 81 decoded frames / 21 latent frames;
- 50 denoising steps, flow shift 3, CFG 5;
- BF16 transformer and FP32 VAE;
- first/last 3 transformer blocks dense;
- sparse blocks 3–26, groups of 3;
- 9 exact anchors;
- 5 dense warmup steps;
- latent-only denoising timing for the primary systems table.

Every comparison must reuse the same initial noise. The supplied runner seeds a CUDA generator and saves final latents.

## Stage A — integration and numerical safety

Run CPU tests, then a low-resolution eight-step smoke test. Verify:

- dense custom sampler matches the ordinary Diffusers pipeline when both use the same scheduler and latents;
- conditional and unconditional sparse branches replay identical block schedules;
- exact anchors remain exact after reconstruction;
- no NaN/Inf appears in hidden states, defects, controller risk, or final latents.

The smoke test is not evidence of video quality or speedup.

## Stage B — causal signal test

Use at least eight prompt–seed pairs, chosen to include:

- smooth camera/object motion;
- a sharp direction change;
- articulated human/animal motion;
- two-object interaction;
- a small fast-moving object;
- a mostly static scene.

Probe three denoising stages and three block depths. A reasonable first grid is steps `5,20,40` and blocks `8,14,20` in the 50-step setting.

Run both `anchor_only` and `full_kv`.

Primary diagnostic quantities, in priority order:

1. matched-budget mesh-only block-delta NMSE for current CoFrame, original Rhyme, fixed, and exact DP oracle meshes;
2. Rhyme-to-oracle NMSE headroom recovery plus absolute oracle regret;
3. one-swap decision Spearman, chosen-action gain recovery, normalized regret, and top-1/no-op correctness;
4. realized sparse-operator block-delta NMSE and non-anchor CVaR-10;
5. exact-anchor context error;
6. error after `+1` and `+3` subsequent dense blocks;
7. frame-risk correlations as secondary explanatory diagnostics.

Interpretation:

- lower mesh-only error but worse realized error under `anchor_only` identifies K/V context removal as the main confounder;
- positive frame-risk correlation but poor swap regret means the signal does not induce the right controller action;
- local improvement that disappears or reverses after `+1/+3` blocks indicates an error-propagation problem;
- failure to beat Rhyme in mesh-only NMSE under `full_kv` directly falsifies the proposed adaptive frame-placement benefit.

Do not use one arbitrary Spearman threshold as the gate. Treat prompt–seed–step–block cells as paired and require positive median CoFrame-minus-Rhyme mesh improvement, a favorable win rate, and a prompt-clustered bootstrap interval. Correlation alone is insufficient.

## Stage C — matched-budget quality

Run `dense`, `fixed`, `rhyme`, and `coframe` with identical anchor count, warmup, sparse block range, K/V mode, interpolation target, prompts, and seeds.

First measure:

- final-latent normalized MSE / relative L2 and cosine to dense;
- temporal-gradient relative L2 to expose motion/transition distortion;
- frame-error CVaR-10, P95, and maximum;
- decoded PSNR/SSIM/LPIPS or video feature distance after the mechanism gate;
- temporal consistency and task-specific VBench/T2V-CompBench scores on the eventual larger set.

The key comparison is **CoFrame versus Rhyme**, not CoFrame versus fixed.

Required ablations:

- Rhyme initialization vs uniform initialization;
- no refresh vs refresh;
- delta vs state interpolation;
- anchor-only vs full K/V;
- defect on delta vs defect on state;
- block group size 1/3/6;
- 6/8/9/12 anchors;
- dense boundary blocks 0/3/5;
- random sketch dimension 0/32/64/128.

## Stage D — latency

Measure after one unreported warmup generation. Run at least five repetitions per method and report median plus interquartile range.

Synchronize CUDA immediately before and after denoising. Report separately:

- text encoding;
- denoising;
- VAE decode;
- total generation;
- controller/defect overhead if instrumented;
- peak allocated and reserved memory.

Do not infer latency from FLOPs. Compare `anchor_only` and `full_kv`; only `anchor_only` is expected to reduce self-attention K/V work substantially.

A systems-positive result requires an actual denoising speedup over static Rhyme at matched quality or a quality gain at nearly identical time. If CoFrame improves quality but dynamic mesh changes reduce kernel efficiency, freeze mesh changes to a small profiled set of anchor shapes in the next implementation.

## Stage E — few-step transfer

Wan2.1 base validates the block-level mechanism but is not the final few-step target. Only after Stage B/C succeeds should the same operator be ported to a public distilled 4–8-step Wan-compatible checkpoint.

In a few-step model, adaptation opportunities move from denoising time to block depth. Use a one-step dense analysis pass or the first denoising step for Rhyme initialization, then refresh across block groups. Ensure at least one sparse step remains; the current runner enforces this for smoke tests.

## Result table template

| Method | Anchors | Mesh NMSE ↓ | Headroom ↑ | Swap regret ↓ | Block CVaR ↓ | Prop. +3 ↓ | End Rel-L2 ↓ | Denoise s ↓ | Speedup ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 21 | 0 | — | — | 0 | 0 | 0 | — | 1.00× |
| Fixed | 9 | — | — | — | — | — | — | — | — |
| Rhyme | 9 | — | 0.00 | — | — | — | — | — | — |
| CoFrame | 9 | — | — | — | — | — | — | — | — |
| Oracle mesh | 9 | — | 1.00 | 0 | — | — | — | — | — |

Exact metric definitions and edge-case handling are specified in [`METRICS.md`](METRICS.md).


## Stage B2 — source attribution after the 4xL40 pilot

The first 4xL40 Stage-1 release showed a strong adaptive-mesh signal but an under-calibrated one-swap controller.  Do **not** weaken RhymeFlow or FIS-DiT: both remain strong paper baselines.  The next experiment asks where CoFrame's gain comes from.

Use the same eight prompts and seeds.  Run `scripts/run_stage1b_wan21.sh` in `full_kv` first.  It evaluates four trajectories with the same Rhyme initialization:

- `refresh_signal=none`: no online remeshing;
- `gap_only`: remesh from interval geometry with no semantic/defect evidence;
- `shuffled`: preserve defect magnitudes while destroying frame alignment;
- `defect`: the real CoFrame signal.

Every probe additionally executes **matched-input** Rhyme-selector, fixed, and FIS-style sparse counterfactuals from the exact same block input and propagates all states through the same +1/+3 dense blocks.  This closes the missing comparison in Stage 1.

FIS uses the published interleaved residue rule `r_l=(l-l0) mod n`, boundary anchors, and state interpolation; an exact-K fill/trim is applied only so mechanism comparisons use the same exact-frame budget.  `fis_dense_tail_steps` is explicit rather than hidden.  For final paper tables, also reproduce the official FIS-DiT and full RhymeFlow systems with their authors' recommended settings; the in-repo `rhyme` method is a selector-controlled baseline under CoFrame's sparse operator, not a claim of reproducing the entire asynchronous RhymeFlow scheduler.

Primary Stage-B2 decisions:

1. true defect must beat `gap_only` and `shuffled` on matched-input mesh NMSE and harmful-swap rate;
2. CoFrame must beat Rhyme and FIS on realized block-delta NMSE from the same input;
3. the advantage should retain the same sign after +1/+3 dense propagation;
4. only then run `scripts/run_baselines_wan21.sh` for endpoint fidelity and warmed latency.


## Stage C0 / Stage-1c — mechanism pivot after Stage-1b

Stage-1b falsified the current leave-one-out defect **localization** mechanism: true and shuffled defects produced nearly the same actions. Do not tune that signal further. First reuse the existing Stage-1b traces with no GPU:

```bash
python scripts/analyze_stage1c_offline.py \
  --root outputs/stage1b \
  --signal defect \
  --output outputs/stage1c/offline_analysis.json
```

The analysis separates (a) prompt dependence of the exact DP oracle mesh from step/block dependence and (b) whether the global defect magnitude can still predict block-level operator or +3 propagation risk. The script emits triage flags, not paper claims.

Only if neither a calibrated step/block schedule nor block-risk gating is supported should GPU time be spent on per-frame signal discovery. The first bounded screen is input temporal curvature plus previous-block delta curvature. These are computed from states that already exist; the probe ranks hypothetical swaps but **does not deploy them**. Run static-Rhyme and gap-only trajectories:

```bash
bash scripts/run_stage1c_curvature_wan21.sh "<prompt>" outputs/stage1c_curvature/p0_s0 0
python scripts/summarize_stage1c_curvature.py \
  --root outputs/stage1c_curvature \
  --output outputs/stage1c_curvature/summary.json
```

A curvature signal is worth implementing as a real controller only if it beats gap-only on actionable-cell gain recovery / regret and also beats its own shuffled control. A high same-action rate with shuffled curvature rejects the localization mechanism just as in Stage-1b.


## Stage-1d — causal adaptive exact-frame budget

Stage-1c suggests defect magnitude may be useful as a scalar block-risk signal even though defect localization was rejected. Stage-1d therefore tests **how much exact computation** to allocate, not where to place defect-driven anchors.

First run the zero-GPU lag test on the preserved Stage-1b `full_kv` traces:

```bash
python scripts/analyze_stage1d_lagged.py \
  --root <STAGE1B_ROOT> \
  --output outputs/stage1d/lagged_analysis.json \
  --plan-output outputs/stage1d/budget_plan.json
```

The causal contract is previous completed block-group defect -> next block-group budget. Only `RUN_ADAPTIVE_K_SCREEN` permits GPU execution. LOPO folds calibrate thresholds from the other seven prompts. The default budgets are `{6,9,12,21}` with calibration quantiles `{0.35,0.80,0.95}`, whose intended mean exact-frame count is approximately 9 under `full_kv`.

For each held-out prompt run dense, static K=9, step/block-only schedule, previous-group mean-defect adaptive K, and max-defect ablation:

```bash
bash scripts/run_stage1d_adaptive_k_wan21.sh \
  "<prompt>" outputs/stage1d_gpu/p0_s0 0 \
  outputs/stage1d/budget_plan.json p0_s0
```

Then aggregate:

```bash
python scripts/summarize_stage1d.py \
  --root outputs/stage1d_gpu \
  --output outputs/stage1d_gpu/summary.json
```

Primary support requires mean-defect adaptation to remain within 5% of the K=9 exact-frame budget, beat static K=9 and the prompt-independent step/block schedule on realized operator NMSE, retain the sign after +3 dense propagation and at the dense-referenced endpoint, and survive the max-defect ablation. Latency is not a primary claim unless GPUs are exclusive. This Stage uses `full_kv` to keep mechanism comparisons close to linear in query-frame count; `anchor_only` speed validation follows only after the mechanism gate passes.

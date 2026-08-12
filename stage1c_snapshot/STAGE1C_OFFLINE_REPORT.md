# CoFrame Stage-1c Offline Triage Report

> **Gate: `TEST_ADAPTIVE_K_OR_DENSE_BLOCK_GATING`.**
> The official analyzer did not emit `RUN_INPUT_CURVATURE_SIGNAL_SCREEN`; no curvature GPU experiment was launched.

## Execution contract

- Source commit: `d8bb14c66331908bbca4a00347d5bf7fd4b711c2`.
- Input: the preserved Stage-1b root with 32 traces and 32 summaries; `signal=defect` selects 8 traces / 72 probes.
- GPU policy: `CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=void`, and `torch.cuda.is_available()==False`.
- Repository tests: 27/27 passed with GPU hidden.
- Official analyzer exit code: 0; stderr empty; output schema `coframe.stage1c.offline.v1`.

## 1. Are exact DP oracle meshes prompt-invariant?

No. For a fixed `(step, block)`, cross-prompt exact meshes have:

- median Jaccard: **0.385**;
- exact-set rate: **8.33%**;
- median normalized anchor displacement: **0.033**.

The low set overlap and 8.33% exact match reject a single prompt-independent exact oracle mesh. The small displacement says many disagreements are nearby anchor shifts, not entirely different temporal regions.

## 2. What drives oracle-mesh variation?

| Comparison holding the other axes fixed | Pair count | Median Jaccard | Exact rate | Median normalized displacement |
|---|---:|---:|---:|---:|
| Prompt variation: same step/block, different prompts | 252 | 0.385 | 8.33% | 0.033 |
| Block variation: same prompt/step, different blocks | 72 | 1.000 | 52.78% | 0.000 |
| Step variation: same prompt/block, different steps | 72 | 0.500 | 25.00% | 0.031 |

By median set distance, the ordering is **prompt > denoising step > block depth**. Block depth is the weakest factor: within one prompt and step, over half the block pairs are exactly identical and the median anchor displacement is zero.

## 3. Defect magnitude as a block-risk signal

| Predictor -> target | Pearson | Spearman | LOPO Spearman median [min,max] | Step/block-centered Spearman |
|---|---:|---:|---:|---:|
| mean_defect -> mesh_oracle_nmse | 0.710 | 0.730 | 0.727 [0.709,0.761] | 0.576 |
| mean_defect -> mesh_fixed_nmse | 0.742 | 0.764 | 0.760 [0.735,0.799] | 0.529 |
| mean_defect -> operator_nmse | 0.590 | 0.641 | 0.633 [0.602,0.679] | 0.437 |
| mean_defect -> propagation_h3 | -0.359 | -0.212 | -0.216 [-0.280,-0.155] | 0.309 |
| max_defect -> mesh_oracle_nmse | 0.725 | 0.753 | 0.752 [0.725,0.776] | 0.176 |
| max_defect -> mesh_fixed_nmse | 0.770 | 0.798 | 0.796 [0.768,0.824] | 0.208 |
| max_defect -> operator_nmse | 0.656 | 0.702 | 0.704 [0.668,0.724] | 0.118 |
| max_defect -> propagation_h3 | -0.358 | -0.282 | -0.284 [-0.338,-0.255] | -0.064 |

The strongest official pooled relationship is `max_defect -> fixed-mesh NMSE` (Spearman 0.798; LOPO median 0.796). `max_defect -> realized operator NMSE` is also LOPO-stable in the pooled cells (0.702; LOPO minimum 0.668).

Conditioning on `(step, block)` changes the picture: max-defect Spearman drops to 0.208 for fixed-mesh NMSE and 0.118 for operator NMSE, while mean-defect retains 0.529 and 0.437. Much of the pooled max-defect result is therefore schedule/cell structure; mean defect is the more plausible within-cell adaptive signal to screen.

The pooled +3 correlations are negative, and the sign changes for mean defect after cell centering. This instability means neither magnitude statistic is yet a demonstrated downstream +3 error predictor.

## 4. Leave-one-prompt-out stability

- Correlation conclusions are stable: the strongest risk correlation remains between 0.768 and 0.824 across held-out prompts.
- Full-vs-LOO oracle medoid exact stability: **59.72%**; mean Jaccard: **0.765**.
- A seven-prompt medoid exactly predicts the held-out prompt's oracle only **0.00%** of the time; mean Jaccard: **0.366**.

Thus the risk-correlation result is LOPO-stable, while exact prompt-independent mesh calibration is not.

## Triage decision and next step

- `TEST_CALIBRATED_STEP_BLOCK_MESH`: **not supported** by exact cross-prompt oracle stability.
- `TEST_ADAPTIVE_K_OR_DENSE_BLOCK_GATING`: **supported as the next bounded test**.
- `RUN_INPUT_CURVATURE_SIGNAL_SCREEN`: **not emitted; NOT RUN**.

The next bounded experiment should compare: (1) a step/block-only K-or-dense schedule, (2) that schedule plus mean-defect adaptation, and (3) a max-defect variant. Defect magnitude may decide **how much exact computation** a block receives, not **which frames** to select. Thresholds and budgets must be chosen with leave-one-prompt-out calibration, and downstream +3/endpoint quality must remain a separate gate.

## Limitations

- The official thresholds are triage heuristics, not preregistered paper claims.
- There are only eight prompts and nine probed step/block cells per prompt.
- Pooled correlations can contain step/block structure; the supplement reports cell-centered correlations, but a future budget-matched gating test is still required.

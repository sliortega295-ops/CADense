# CoFrame: 4xL40 initial exploration report

## Decision

**Stage 1 completed, but the preregistered continuation gate failed. Stage 2 is `Not Run`.**

The frame-position signal is strong: under `full_kv`, CoFrame lowers mesh-only NMSE relative to the static Rhyme mesh by a median **39.59%**, wins **67/72 cells (93.06%)**, and has a prompt-clustered bootstrap 95% interval of **[30.36%, 51.95%]**. It also beats the fixed mesh by a median **21.00%** in **68/72 cells (94.44%)**.

The controller-action evidence does not pass, however. Among 33 `full_kv` cells with a positive oracle one-swap opportunity, chosen-action gain recovery is positive in only 6, zero in 12, and negative in 15; its median is **0.00**, below the required strictly-positive direction. The current trace also does not contain a matched-input realized Rhyme sparse-operator counterfactual, so comparative operator and `+1/+3` propagation improvement is **Not Run**, not a claim.

## Experimental contract

- Repository: `https://github.com/sliortega295-ops/CADense.git`
- Remote checkout: `/data/jiayu/yongyan_liu/CoFrame`
- Branch and source commit: `main`, `915fb855446770e9cad91cb66b6ef46040deb389`
- Tracked source diff during all measured runs: empty
- Host: `user-NF5280M6`; 4x NVIDIA L40, 46,068 MiB each; driver `580.173.02`
- Python `3.10.19`; PyTorch `2.13.0+cu130`; Diffusers `0.34.0`; Transformers `4.53.2`; Accelerate `1.14.0`
- Model: local validated `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` conversion at `/data/jiayu/yongyan_liu/models/Wan2.1-T2V-1.3B-Diffusers`
- Upstream model revision: `0fad780a534b6463e45facd96134c9f345acfa5b`
- Transformer dtype: BF16; 480x832, 81 decoded/21 latent frames, 50 steps, CFG 5.0, flow shift 3.0
- Warm-up 5 steps, K=9, sparse blocks `[3,27)`, block group size 3
- Prompt-seed tasks were independent processes: GPUs 0-3 handled one complete pair each; no DDP or tensor parallelism
- No video decoding or VBench was run.

### Environment repairs retained as evidence

1. System Python 3.12 lacked `ensurepip`; the private `.venv` was therefore created from the user's existing Python 3.10 interpreter. The failed venv is preserved as `.venv.failed-py312-no-ensurepip-20260809`.
2. The unconstrained dependency initially resolved to Transformers 5.14.1, which is incompatible with Diffusers 0.34.0 (`FLAX_WEIGHTS_NAME` import failure). Transformers was pinned to 4.53.2; `pip check`, Wan imports, and all 22 tests then passed.
3. `/data/models/Wan2.1-T2V-1.3B-Diffusers` contained broken snapshot symlinks. The experiment used the user-owned complete 17 GB conversion above and registered its pinned revision in the user-owned Hugging Face cache. No shared model files were modified.
4. The GPUs were initially occupied by `/data/anqian` jobs. The launchers waited for three consecutive idle samples and did not terminate or alter those jobs.

## Smoke gate

Command contract: GPU 0 only, 256x448, 33 frames, 8 steps, warm-up 2, K=5, latent-only.

| Check | Result |
|---|---|
| Unit tests | 22 passed |
| Completion | success |
| Dense / sparse denoising steps | 2 / 6 |
| Latent shape | `[1,16,9,32,56]` |
| Finite latents | yes |
| Initial -> final anchors | `[0,1,2,5,8]` -> `[0,1,3,5,8]` |
| Peak allocated / reserved | 14.98 / 15.47 GB |
| Required artifacts | `latents.pt`, `summary.json`, `trace.json` present |
| CFG schedule | runtime replay equality guard passed; no divergence exception |
| Post-run GPU residue | returned to 14 MiB |

The smoke test verifies integration and bounded memory only; it is not quality or speed evidence.

## Stage 1 results

Each mode has 8 prompts x 3 steps x 3 blocks = 72 paired cells. All 16 runs completed, all 144 expected cells were present, all final latents were finite, and there were no OOMs or abnormal traces.

### Primary metrics

| Metric | `full_kv` | `anchor_only` |
|---|---:|---:|
| Median mesh NMSE improvement vs Rhyme | **39.59%** | **36.57%** |
| Mesh win rate vs Rhyme | **93.06%** | **94.44%** |
| Prompt-clustered bootstrap 95% CI | [30.36%, 51.95%] | [20.58%, 54.19%] |
| Median headroom recovery | 0.9999 | 1.0000 |
| Median oracle NMSE regret | 5.90e-6 | 2.72e-6 |
| Median mesh improvement vs fixed | **21.00%** | **19.95%** |
| Win rate vs fixed | **94.44%** | **93.06%** |
| Fixed improvement vs Rhyme | 30.41% | 28.32% |
| Exact oracle anchor set at probes | 38/72 | 44/72 |

The fixed selector itself beats Rhyme by about 30% under `full_kv`. Thus the large CoFrame-vs-Rhyme number partly reflects a weak, highly front-clustered static Rhyme initialization. CoFrame still beats fixed, so the result is not *only* a bad-baseline artifact.

### Controller-action diagnostics

| Metric | `full_kv` | `anchor_only` |
|---|---:|---:|
| Actionable one-swap cells | 33 | 27 |
| Positive / zero / negative gain recovery | 6 / 12 / 15 | 4 / 8 / 15 |
| Median gain recovery | **0.00** | **-0.198** |
| Median post-observation regret | 3,971.5 | 0.0 |
| Median prior regret | 17,551.5 | 13,765.7 |
| Post regret lower than prior | 76.39% | 69.44% |
| Top-1 exact rate | 45.83% | 56.94% |
| No-op correctness | 72.22% | 83.33% |

The defect observation often reduces regret relative to the static prior, but the selected action is still harmful or a missed opportunity in most actionable cells. This directly fails the preregistered controller gate.

The controller exposes 360 refresh decisions per run. Changed meshes range from 13 to 261 times per `full_kv` run, and late probes frequently equal the DP oracle. This is promising but also a warning: the apparent near-oracle mesh may come from very aggressive repeated remeshing rather than a well-calibrated one-swap decision rule.

### Sparse operator and propagation

| Metric (median) | `full_kv` | `anchor_only` |
|---|---:|---:|
| Block-delta NMSE | 0.1127 | 0.1131 |
| Block-delta relative L2 | 0.3358 | 0.3363 |
| Non-anchor CVaR-10 | 0.5148 | 0.4661 |
| Exact-anchor delta error | **0.0020** | **0.1297** |
| Propagation `+1` relative L2 | 0.0955 | 0.1075 |
| Propagation `+3` relative L2 | 0.0800 | 0.0904 |

Under `full_kv`, realized operator NMSE is almost identical to mesh-only NMSE (median ratio 1.00016; Spearman 1.0), so the selected mesh transfers cleanly to the local operator. Under `anchor_only`, the ratio rises to 1.199 and exact-anchor error is roughly 64x larger, identifying removal of non-anchor K/V context as a real confounder.

The absolute error contracts after dense downstream blocks rather than exploding. However, the trace lacks the matched-input Rhyme operator and Rhyme propagation states, so it cannot establish a comparative CoFrame-vs-Rhyme advantage after `+1/+3` blocks. That item remains `Not Run`.

### Dependence on denoising stage (`full_kv`)

| Step | Median mesh improvement vs Rhyme | Win rate | Headroom recovery | Median swap gain recovery |
|---:|---:|---:|---:|---:|
| 5 | 12.17% | 79.17% | 0.731 | 0.000 |
| 20 | 51.95% | 100% | 1.000 | 0.000 |
| 40 | 54.81% | 100% | 1.000 | -0.520 |

The position benefit is weakest immediately after warm-up. All five `full_kv` mesh losses occur at step 5, block 20. Later meshes are much better, but late one-swap decisions are not better calibrated.

### Diagnostic timing (not a deployment speedup result)

| Mode | Median denoise time | Mean denoise time | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| `anchor_only` | 160.83 s | 159.72 s | 17.04 GB | 18.51 GB |
| `full_kv` | 201.71 s | 200.85 s | 17.04 GB | 18.51 GB |

`anchor_only` is about 1.25x faster than `full_kv` in this probe harness. These runs include oracle diagnostics and are not warmed repeated latency trials; no dense or Rhyme timing baseline was run, so no end-to-end speedup claim is made.

## Preregistered gate

| Requirement (`full_kv`) | Threshold | Observed | Decision |
|---|---:|---:|---|
| Median mesh NMSE reduction | >=10% | 39.59% | pass |
| Cell win rate | >=60% | 93.06% | pass |
| Median headroom recovery | >=0.20 | 0.9999 | pass |
| Swap gain recovery | positive | 0.00 median | **fail** |
| Operator and `+1/+3` improve vs Rhyme | comparative direction | direct comparator absent | **Not Run** |

Overall decision: **FAIL / do not enter Stage 2 under the stated protocol.** Therefore `dense`, `fixed`, `rhyme`, and `coframe` endpoint-fidelity experiments for 16 prompt-seed pairs are explicitly `Not Run`.

## Recommended next bounded experiment

Do not tune on one prompt. Reuse the same 8 prompts and add matched-input probes for:

1. static Rhyme;
2. static fixed;
3. CoFrame with no refresh;
4. gap-only refresh with the defect signal removed;
5. shuffled-defect refresh;
6. full defect refresh.

At each probe, execute current and Rhyme meshes from the same block input and propagate both through the same `+1/+3` dense blocks. This separates three possibilities: a genuinely useful defect, a generic drive toward a regular mesh, or a weak Rhyme initialization. Preregister controller acceptance using actionable-cell gain recovery and harmful-swap rate before rerunning full trajectories.

## Artifacts

Primary remote paths:

- `outputs/probe/probe_cells.csv`
- `outputs/probe/probe_summary.json`
- `outputs/probe/probe_validation.json`
- `outputs/probe/probe_additional_diagnostics.json`
- `outputs/probe/p{0..7}_s0/coframe_{anchor_only,full_kv}_seed0/{latents.pt,summary.json,trace.json}`
- `outputs/smoke/wan21_1.3b_coframe_seed0/{latents.pt,summary.json,trace.json}`
- `logs/environment_freeze.txt`, smoke/probe logs, GPU telemetry, and preserved failure logs

The raw traces and latents remain remote. A compact evidence archive excludes only `latents.pt`; a separate manifest records every latent's path, size, and SHA-256.

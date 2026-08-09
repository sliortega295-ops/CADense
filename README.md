# CoFrame

**Rhyme-initialized, self-validating adaptive frame meshes for video diffusion transformers.**

CoFrame studies a narrower question than generic “important-frame selection”:

> Under a fixed number of exactly evaluated frames, can a video DiT use errors observed inside its own blocks to move the frame mesh toward intervals where interpolation is actually failing?

The first target is **Wan2.1-T2V-1.3B at 480p**. This repository is an algorithm-validation implementation: it establishes causal signal, quality behavior, and a reproducible latency contract before introducing custom packed kernels.

## Method in one picture

After a short dense warm-up, CoFrame uses RhymeFlow-style one-step clean-latent cosine selection as a strong content prior. For a block budget of `K` frames:

1. Rhyme initializes `K - V` persistent core anchors.
2. `V` validator frames are chosen inside high-risk temporal intervals. Validators are included in `K`; they are **not extra DiT calls**.
3. The block runs only on the selected frame-token slabs, with their original temporal RoPE positions.
4. A validator output is compared with the linear interpolation of its two neighboring core outputs:

   `defect(v) = || y_v - Interp(y_left, y_right) || / || y_v ||`.

5. The measured defect updates the risk field and reallocates the core mesh for later blocks. The conditional schedule is replayed exactly for the unconditional CFG branch.

This makes RhymeFlow the initialization and strongest frame-selection baseline, while FIS-style uniform selection is retained only as a sanity baseline because our earlier experiments found it close to fixed selection.

## What is implemented

- A runtime temporal-RoPE patch for the **unmodified official Wan2.1 source tree**.
- Batch-1 Wan block execution on selected full-frame token slabs.
- Piecewise-linear reconstruction to the complete 21-frame latent sequence.
- Rhyme sequential selection and a matched-cardinality adaptation.
- Fixed-budget CoFrame control with validators counted in the same `K` budget.
- A causal G1 probe that measures whether defect predicts the **actual error reduction from adding a candidate frame**.
- Dense, fixed, uniform, Rhyme, and CoFrame generation paths with matched prompts, seeds, schedules, timing, and metadata.
- CPU unit tests for mesh, selector, controller, RoPE, and analysis logic.

The current executor uses Python gather/interpolation and dynamic shapes. Its first purpose is to validate the algorithm; it is not yet the final CUDA-graph/packed latency implementation.

## Setup

Python 3.10 or 3.11 and a CUDA environment supported by the official Wan2.1 repository are recommended.

```bash
bash scripts/bootstrap_wan21.sh

# Official Wan2.1-T2V-1.3B checkpoint
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir ./models/Wan2.1-T2V-1.3B
```

`bootstrap_wan21.sh` pins the official Wan checkout to commit
`9737cba9c1c3c4d04b33fcad41c111989865d315`. CoFrame does not modify files in that checkout.

Run the CPU tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## First experiment: G1 mechanism probe

The probe follows a dense trajectory and freezes selected `(denoising step, DiT block)` inputs. It compares CoFrame defect against Rhyme novelty and clean-latent interpolation residual, using true local block-error reduction as the target.

```bash
python scripts/run_wan21_probe.py \
  --wan-repo ./third_party/Wan2.1 \
  --ckpt-dir ./models/Wan2.1-T2V-1.3B \
  --output-root ./outputs/probe_smoke \
  --max-prompts 4 \
  --seeds 42 \
  --step-indices 5,25,45 \
  --block-indices 4,15,26 \
  --total-budget 9

python scripts/analyze_probe.py \
  --input-root ./outputs/probe_smoke
```

The pre-declared G1 gate is:

- dense-adapter relative L2 no larger than `1e-4`;
- mean within-cell Spearman between CoFrame defect and full-block error gain at least `0.55`;
- at least `0.15` higher than the stronger of Rhyme novelty and clean-proxy interpolation residual.

The exhaustive `defect_scan_final_slot` and `gain_oracle_final_slot` rows are diagnostic upper bounds. They are explicitly excluded from deployable latency claims.

## Matched generation

Run each method in a separate process so model loading and allocator state do not contaminate comparisons:

```bash
for method in dense fixed uniform rhyme coframe; do
  python scripts/run_wan21_generate.py \
    --wan-repo ./third_party/Wan2.1 \
    --ckpt-dir ./models/Wan2.1-T2V-1.3B \
    --output-root ./outputs/generation_smoke \
    --method "${method}" \
    --max-prompts 4 \
    --seeds 42 \
    --sparse-blocks 4:28 \
    --total-budget 9 \
    --validator-count 1
done
```

Every unit writes `video.mp4` and `metadata.json`. Metadata contains the exact per-step/per-block mesh, observed defects, denoising wall time, CUDA-event time, VAE time, peak memory, prompt, and seed.

The default contract mirrors the established Wan2.1-1.3B setup: 81 pixel frames, 21 latent frames, 50 denoising steps, shift 8, CFG 6, and 832×480 output. Wan2.1-1.3B is used here to validate the frame/block mechanism; a later few-step checkpoint is required before making the paper's final few-step claim.

## Repository layout

```text
coframe/                 Core mesh, selectors, controller, Wan adapter, probes
scripts/                 Wan setup, G1 probe, analysis, matched generation
configs/                 Pilot prompts and canonical experiment contract
docs/                    Method assumptions and staged experiment protocol
tests/                   Dependency-light CPU tests
```

## Current scientific boundary

CoFrame does **not** claim that uniform/FIS selection is a strong baseline, nor that generic error-aware refresh is new. The intended claim is specific: a leave-one-out **block-output interpolation defect** provides actionable information beyond a strong Rhyme clean-latent prior, and that information can control a fixed-budget temporal mesh.

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for the stop conditions and scale-up plan.

## License

Apache-2.0. Wan2.1 remains subject to its own repository and model licenses.

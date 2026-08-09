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

Primary diagnostic quantities:

1. current-defect Spearman with dense block intervention error;
2. causal-risk Spearman;
3. static Rhyme-prior Spearman;
4. CoFrame gain over Rhyme prior;
5. anchor context error.

Interpretation:

- strong defect correlation in `full_kv` but weak correlation in `anchor_only` identifies K/V context removal as the main confounder;
- weak correlation in both modes falsifies the current block-interpolation defect;
- high correlation but no mesh-quality gain points to a controller or temporal-credit problem rather than a signal problem.

Suggested continuation gate: mean defect or causal-risk Spearman at least 0.5, with a clear positive margin over the Rhyme prior. Treat this as an engineering gate, not a paper threshold.

## Stage C — matched-budget quality

Run `dense`, `fixed`, `rhyme`, and `coframe` with identical anchor count, warmup, sparse block range, K/V mode, interpolation target, prompts, and seeds.

First measure:

- final-latent relative L2 and cosine to dense;
- per-frame latent error and its maximum;
- decoded LPIPS / video feature distance if available;
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

| Method | Anchors | KV | Warmup | Sparse blocks | Rel-L2 ↓ | Frame max ↓ | Quality ↑ | Denoise s ↓ | Speedup ↑ |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| Dense | 21 | full | — | — | 0 | 0 | — | — | 1.00× |
| Fixed | 9 | anchor | 5 | 3–26 | — | — | — | — | — |
| Rhyme | 9 | anchor | 5 | 3–26 | — | — | — | — | — |
| CoFrame | 9 | anchor | 5 | 3–26 | — | — | — | — | — |

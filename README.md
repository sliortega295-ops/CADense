# CoFrame

**Self-Validating Adaptive Frame Meshes for Sparse Video Diffusion**

CoFrame is a training-free research prototype for validating block-conditional sparse-frame computation on **Wan2.1-T2V-1.3B**.

The key question is deliberately narrow and falsifiable:

> Starting from a strong RhymeFlow-style clean-latent frame selector, can block-level leave-one-out interpolation defects improve the frame mesh under the same exact-frame budget?

Our prior experiments found FIS-DiT-style rotating uniform selection close to a fixed-frame baseline, while RhymeFlow-style content-aware selection was substantially stronger. CoFrame therefore uses Rhyme as the strong initialization/baseline rather than treating uniform rotation as the main prior.

## Methods

| Method | Initial anchors | Online refresh | Role |
|---|---|---:|---|
| `dense` | all latent frames | no | quality / latency reference |
| `fixed` | uniform fixed budget | no | static baseline |
| `rhyme` | clean-latent sequential cosine | no | strong content-aware baseline |
| `coframe` | same Rhyme initialization | block LOO defect | proposed method |

CoFrame keeps the anchor budget fixed. For neighboring exact anchors `a < v < b`, the exact update at `v` is compared with interpolation from `a,b`. This leave-one-out residual becomes an online block-level error signal. A one-anchor swap controller then moves the temporal mesh only when the estimated interpolation cost decreases.

The default reconstructs **block deltas**, not full hidden states:

```text
h_next[f] ~= h[f] + Interp_A(delta_h[A])[f]
```

This preserves each skipped frame's incoming state and approximates only the update introduced by the current block.

Two self-attention context modes are implemented:

- `anchor_only`: anchor queries attend only to anchor K/V; this is the actual compute-saving path.
- `full_kv`: anchor queries attend to all K/V; this diagnostic separates temporal interpolation error from loss of non-anchor context.

For CFG, the conditional branch chooses/updates the mesh and the unconditional branch replays the exact same per-block anchor schedule.

## Wan2.1 validation contract

The canonical first experiment uses:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `diffusers==0.34.0`
- 480x832
- 81 decoded frames / 21 latent frames
- 50 denoising steps
- 5 dense warmup steps
- 9 exact anchors
- sparse DiT blocks 3--26
- block group size 3
- BF16 transformer

The implementation intentionally fails closed on unsupported Diffusers versions because Wan attention internals changed after 0.34.0.

## Install

```bash
git clone https://github.com/sliortega295-ops/CADense.git
cd CADense
pip install -r requirements-wan.txt
pip install -e .
pytest -q
```

The research project and Python package are named **CoFrame / `coframe`**. The current GitHub repository slug remains `CADense`; the code does not depend on the slug.

## Run the four-way comparison

```bash
bash scripts/run_ablation_wan21.sh \
  "A red toy car turns sharply around a blue cube on a wooden table." \
  outputs/wan21_main \
  0
```

This runs `dense`, `fixed`, `rhyme`, and `coframe` under the same prompt, seed, sampler, warmup, sparse-block range, anchor budget, and sparse operator, then compares final latent fidelity and denoising latency.

## Run the decisive causal probe

```bash
bash scripts/run_probe_wan21.sh \
  "A gymnast performs a fast cartwheel while a yellow ball rolls behind her." \
  outputs/wan21_probe \
  0
```

The probe recomputes selected dense blocks from the **same block input** and reports:

- Rhyme-prior correlation with true non-anchor block error;
- causal CoFrame-risk correlation;
- current leave-one-out-defect correlation;
- CoFrame gain over the Rhyme prior;
- exact-anchor error, exposing context-restriction error.

Run both `anchor_only` and `full_kv`. If the signal is strong only under `full_kv`, then lost K/V context rather than interpolation is the dominant problem. If the defect is weak in both modes, the current CoFrame hypothesis is falsified.

## Repository layout

```text
coframe/
  selection.py          Rhyme-style and fixed selectors
  interpolation.py      temporal interpolation + leave-one-out defect
  controller.py         fixed-budget adaptive mesh
  wan/
    sparse_forward.py   sparse Wan block/model forward
    pipeline.py         shared dense/sparse denoising loop
  cli.py
  compare.py
scripts/
tests/
docs/
```

See `docs/METHOD.md` for the method specification and `docs/EXPERIMENT_PLAN.md` for the staged validation protocol.

## Current scope

This is a mechanism-validation implementation, not yet an optimized production kernel. Dynamic gather/scatter uses PyTorch operations; CUDA Graphs and shape-specialized kernels are later latency work. Wan2.1 base sampling is used first to validate the block-level signal. A distilled 4--8-step model should be ported only after the causal signal and matched-budget quality gates pass.

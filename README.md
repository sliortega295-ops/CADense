# CoFrame

**ODE-Path-Aware Frame Budgets with Self-Validating Sparse Video Diffusion**

CoFrame is a training-free research prototype for block-conditional sparse-frame computation in **Wan2.1-T2V-1.3B**. The current proposed path separates two decisions:

> How many frames should receive exact computation at the next denoising step, and where should those exact frames be placed inside each sparse block group?

`--method coframe_ode` uses current trajectory signals to allocate the next step's frame budget. Inside that step, already-computed exact anchors validate their own residual interpolation through a leave-one-out defect, and the resulting risk field remeshes the next block group. The earlier fixed-K variant remains available as `--method coframe` for reproducibility. The code targets `diffusers==0.34.0` and does not modify model weights.

## Historical mechanism experiments

Our prior experiments found that FIS-DiT's rotating uniform selection behaved similarly to a fixed-frame baseline, while RhymeFlow's clean-latent cosine selector was meaningfully stronger. CoFrame therefore does **not** present uniform rotation as its main prior. Instead:

1. **Rhyme initializes the mesh.** A one-step clean-latent proxy supplies semantic transition scores and initial anchors.
2. **Sparse Wan blocks validate their own approximation.** Exact interior anchors act as leave-one-out validators: each is predicted from its two neighboring anchors and compared with its exact block update.
3. **The mesh changes at fixed budget.** A risk-weighted one-swap split/merge controller moves anchors only when measured interpolation cost decreases.
4. **Latency remains measurable.** Dense, fixed, Rhyme, and CoFrame share one sampler, one model, one seed, one warmup contract, and the same sparse operator.

This isolates CoFrame's incremental contribution over the already-good Rhyme selector.

## Implemented methods

| Method | Initial anchors | Online refresh | Purpose |
|---|---|---:|---|
| `dense` | all frames | no | quality and latency reference |
| `fixed` | uniform fixed budget | no | static frame-selection baseline |
| `rhyme` | clean-latent sequential cosine | no | strong content-aware baseline |
| `coframe` | same Rhyme initialization | leave-one-out block defect | legacy mechanism ablation |
| `coframe_ode` | LOO residual-defect remeshing | step-level ODE/path budget + group-level self-validation | current proposed path |

All sparse methods can use either:

- `anchor_only`: anchor queries attend only to anchor keys/values; largest theoretical saving, but includes context-restriction error.
- `full_kv`: anchor queries attend to all frame keys/values; smaller saving, but cleanly diagnoses interpolation error.

The default reconstructs **block deltas** rather than full states:

\[
\widetilde h^{l+1}_f = h^l_f + I_A\!\left(\Delta h^l_A\right)_f.
\]

This preserves each skipped frame's incoming state and interpolates only the update introduced by the current block.

## Repository layout

```text
coframe/
  config.py                 experiment contract
  selection.py              fixed and Rhyme-style selectors
  interpolation.py          temporal interpolation and LOO defect
  metrics.py                exact mesh oracle and action diagnostics
  controller.py             fixed-budget adaptive mesh
  wan/
    sparse_forward.py       sparse Wan block/model forward
    pipeline.py             shared dense/sparse denoising loop
  cli.py                     Wan2.1 runner
  compare.py                 latent fidelity and latency comparison
scripts/
  run_wan21_1_3b.py
  run_ablation_wan21.sh
  run_probe_wan21.sh
  compare_runs.py
tests/                       CPU tests for selection/interpolation/controller
docs/
  METHOD.md
  METRICS.md
  EXPERIMENT_PLAN.md
```

## Installation

Create a clean Python 3.10 or 3.11 environment. The Wan integration currently fails closed on any Diffusers version other than 0.34.0 because later releases changed the attention internals.

```bash
git clone https://github.com/sliortega295-ops/CADense.git CoFrame
cd CoFrame

pip install -r requirements-wan.txt
pip install -e .
```

Run the CPU tests before using a GPU:

```bash
pytest -q
```

## ODE-path-aware CoFrame

```bash
bash scripts/run_ode_coframe_wan21.sh \
  "A red toy car turns sharply around a blue cube on a wooden table." \
  outputs/ode_coframe \
  0
```

The default controller supports arbitrary integer frame budgets within its resolved range and exactly conserves the configured average frame budget across sparse steps. Within each sparse step, exact interior anchors are compared against leave-one-out residual interpolation and the measured defect updates the next block-group mesh. See `docs/ODE_PATH_COFRAME.md` for the causal and execution contract.

## Canonical Wan2.1-1.3B validation

The paper-quality mechanism validation uses the standard 480p setting, 81 decoded frames (21 latent frames), 50 denoising steps, identical prompt and seed, and latent-only timing.

```bash
python scripts/run_wan21_1_3b.py \
  --method coframe \
  --prompt "A red toy car turns sharply around a blue cube on a wooden table." \
  --seed 0 \
  --height 480 --width 832 --num-frames 81 --steps 50 \
  --warmup-steps 5 --num-anchors 9 \
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3 \
  --kv-mode anchor_only \
  --interpolation-target delta --defect-target delta \
  --output-dir outputs/wan21_main
```

Run all four methods with the same contract:

```bash
bash scripts/run_ablation_wan21.sh \
  "A red toy car turns sharply around a blue cube on a wooden table." \
  outputs/wan21_main \
  0
```

Then compare final latent fidelity and denoising latency:

```bash
python scripts/compare_runs.py \
  --dense outputs/wan21_main/dense_seed0/latents.pt \
  --candidate fixed outputs/wan21_main/fixed_seed0/latents.pt \
  --candidate rhyme outputs/wan21_main/rhyme_seed0/latents.pt \
  --candidate coframe outputs/wan21_main/coframe_seed0/latents.pt \
  --output-json outputs/wan21_main/comparison_seed0.json
```

Each run writes:

- `latents.pt`: final latent plus complete method metadata;
- `summary.json`: timing, memory, model contract, and anchor endpoints;
- `trace.json`: block-level meshes, defects, refresh decisions, and optional oracle probes;
- `video.mp4`: only when `--decode` is explicitly requested.

## The decisive oracle probe

A speed/quality table alone cannot show whether CoFrame's defect is the right signal. The probe recomputes selected dense blocks from the **same block input** and measures the actual per-frame sparse-block error.

```bash
bash scripts/run_probe_wan21.sh \
  "A gymnast performs a fast cartwheel while a yellow ball rolls behind her." \
  outputs/wan21_probe \
  0
```

For each probed `(denoising step, block)`, the trace now separates three levels:

- **mesh-only block-delta NMSE** for current CoFrame, the original Rhyme mesh, fixed selection, and an exact fixed-budget DP oracle;
- **Rhyme-to-oracle headroom recovery** and absolute oracle regret;
- **one-swap decision quality**: swap-gain Spearman, chosen-action gain recovery, regret, top-1 agreement, and no-op accuracy;
- realized sparse block-delta error, non-anchor CVaR-10, and exact-anchor context error;
- error after the same states pass through `+1` and `+3` subsequent dense blocks;
- the earlier frame-risk correlations, retained as secondary diagnostics.

Run the probe once with `anchor_only` and once with `full_kv`. If mesh-only error is good but realized/propagated error degrades only under `anchor_only`, the main obstacle is K/V context removal rather than frame placement.

## Minimal smoke test

The following is only an integration check, not a quality result:

```bash
python scripts/run_wan21_1_3b.py \
  --method coframe \
  --prompt "A dog runs from left to right." \
  --seed 0 \
  --height 256 --width 448 --num-frames 33 --steps 8 \
  --warmup-steps 2 --num-anchors 5 \
  --sparse-block-start 3 --sparse-block-end 27 \
  --output-dir outputs/smoke
```

The runner always leaves at least one sparse denoising step in few-step smoke tests.

## What counts as a positive result

The recommended first gate is not a VBench sweep. Use 4–8 prompts covering rigid motion, articulated motion, object interaction, camera motion, and short local events, with 2–4 seeds each.

Continue only when the paired evidence is consistent:

- CoFrame reduces **mesh-only NMSE** relative to the original Rhyme anchors at the same `K`, with positive Rhyme-to-oracle headroom recovery;
- the post-observation controller chooses swaps with lower oracle regret than the static/causal prior;
- realized block error and `+1/+3` dense-propagation error improve in the same direction;
- complete-run endpoint fidelity improves over static Rhyme without sacrificing the intended denoising-latency benefit.

Risk/error correlation is useful for explanation, but is no longer the primary pass/fail metric.

A negative result is also informative:

- high anchor context error under `anchor_only` means frame selection is not the dominant problem;
- low defect–oracle correlation under `full_kv` falsifies the proposed commutation/mesh signal;
- lower FLOPs without denoising latency improvement means the dynamic frame shapes are not kernel-efficient enough.

See [`docs/METRICS.md`](docs/METRICS.md) for exact definitions and [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for the staged protocol.

## Repository migration

The local repository and package are already named CoFrame. To rename the original empty `CADense` GitHub repository and push this initial commit from an authenticated machine, run:

```bash
bash scripts/publish_initial_repo.sh
```

See [`docs/REPOSITORY_MIGRATION.md`](docs/REPOSITORY_MIGRATION.md) for the exact API and Git commands.

## Current limitations

This is a validation implementation, not yet an optimized production kernel.

- The sparse path currently targets Wan T2V and Diffusers 0.34.0 only.
- Dynamic gather/scatter uses PyTorch operations; CUDA Graphs and shape-specific kernels are future latency work.
- Controller evidence comes from the conditional CFG branch; the unconditional branch replays the exact same schedule.
- Leave-one-out residual is a same-mask surrogate. The oracle probe is required to measure its bias.
- Wan2.1 base sampling is used to validate the mechanism first. A distilled 4–8-step checkpoint should be added only after the block-level signal passes the causal gate.

## Attribution

CoFrame builds on the public Wan2.1 and Hugging Face Diffusers interfaces and adopts the clean-latent sequential cosine selector as a RhymeFlow-compatible baseline/prior. See `NOTICE` for links and licenses.


## Stage-1b source-attribution experiment

The next experiment preserves RhymeFlow and FIS-DiT as strong baselines and tests whether CoFrame's improvement is specifically caused by correctly aligned block defects rather than generic regularization toward a uniform mesh.

```bash
bash scripts/run_stage1b_wan21.sh "<prompt>" outputs/stage1b 0
```

This runs `none`, `gap_only`, `shuffled`, and true `defect` refresh policies under `full_kv`.  At every oracle cell the probe also evaluates matched-input Rhyme-selector, fixed, and FIS-style interleaved meshes, including realized operator error and +1/+3 dense propagation.  For endpoint baselines use:

```bash
bash scripts/run_baselines_wan21.sh "<prompt>" outputs/baselines 0
```

`rhyme` here isolates the Rhyme keyframe selector under the shared sparse operator.  Full-system RhymeFlow and official FIS-DiT should still be reproduced separately for the final paper table.

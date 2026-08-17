# ODE-Path-Aware CoFrame

`--method coframe_ode` implements the current proposed two-level policy while preserving the earlier defect experiments under `--method coframe`.

## Execution contract

1. The first `warmup_steps` denoising steps are dense.
2. Later Wan steps keep dense blocks `[0,3)` and `[27,30)`, and sparsify the middle block groups.
3. The current guided flow output supplies three step-level signals without an extra DiT call:
   - cosine direction change between consecutive velocity fields;
   - relative change of the scheduler-consistent clean endpoint prediction;
   - normalized frame-wise second-difference energy of the velocity field.
4. EMA-normalized signals form a dimensionless difficulty. The current step causally assigns the *next* sparse step's exact-frame budget.
5. A remaining-budget multiplier and reachability check meet the configured total frame budget exactly. The default support is every integer between the automatic minimum and the latent-frame count; a smaller execution codebook can be supplied with `--ode-budget-values`.
6. Within a step, all sparse groups share the selected frame count. Frame positions minimize squared temporal gaps plus a frozen reuse penalty, producing near-uniform coverage while interleaving exact frames across block groups. Skipped frames retain their incoming state and receive a linearly interpolated block residual.
7. The clean-endpoint conversion fails closed unless the sampler exposes a compatible flow-prediction scheduler contract.

The controller is training-free and prompt-adaptive, but its scientific value is not assumed by the implementation. The next experiment must verify whether its online difficulty predicts a better budget than fixed-K and static-transfer baselines at matched total compute.

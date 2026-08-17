# ODE-Path-Aware + Self-Validating CoFrame

`--method coframe_ode` implements the current two-level policy.

## Execution contract

1. The first `warmup_steps` denoising steps are dense.
2. Later Wan steps keep dense blocks `[0,3)` and `[27,30)`, and sparsify the middle block groups.
3. The current guided flow output supplies ODE direction change, clean-endpoint change, and frame-wise temporal curvature without an extra DiT call. These signals determine the *next* sparse step's exact-frame budget K.
4. All sparse groups in one denoising step use the same K. When K changes between steps, the first group starts from a uniform boundary-preserving mesh while previously accumulated frame-wise risk is retained.
5. For each sparse block, selected anchors are computed exactly. At each interior exact anchor v with neighbors a<v<b, the controller predicts its block residual from the two neighbors and measures a normalized leave-one-out defect: `d_v = RMS(Delta h_v - Interp(Delta h_a, Delta h_b)) / (RMS(Delta h_v) + eps)`.
6. Defects from the blocks in one group are averaged, projected onto their neighboring temporal intervals, and EMA-updated into a frame-wise risk field. A fixed-K one-swap search removes one interior anchor and adds one non-anchor only when the risk-weighted interpolation cost decreases. The refreshed mesh is used by the next block group.
7. Skipped frames retain their incoming hidden state and receive a linearly interpolated block residual. This is residual interpolation inside a DiT block, not video frame synthesis.
8. The conditional CFG branch chooses and updates the sparse schedule; the unconditional branch replays exactly the same per-block anchors.

The method is training-free and never needs a dense-reference forward at runtime. The LOO defect signal is computed only from exact anchors that were already required by the sparse block.

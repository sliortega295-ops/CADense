# CoFrame metric protocol

This protocol is designed to answer three different questions without mixing them:

1. did CoFrame choose better frame positions than the static Rhyme mesh;
2. did those positions improve the actual sparse Wan operator;
3. did the local improvement survive downstream blocks and the full denoising trajectory.

All block-level comparisons must use the same block input, denoising timestep, block, exact-frame budget `K`, interpolation rule, and K/V mode.

## 1. Primary mechanism metric: mesh-only block-delta NMSE

Let the dense block update be

\[
\Delta h^{\mathrm{dense}}=B(h)-h.
\]

For an anchor mesh \(A\), take the **dense** values at the selected anchors and reconstruct every frame using the same piecewise-linear interpolation used by CoFrame:

\[
E_{\mathrm{mesh}}(A)=
\frac{\left\|\Delta h^{\mathrm{dense}}-
I_A\!\left(\Delta h^{\mathrm{dense}}_A\right)\right\|_2^2}
{\left\|\Delta h^{\mathrm{dense}}\right\|_2^2}.
\]

This is the cleanest matched-budget selector metric. It removes the confound that, under `anchor_only`, even an exact anchor can differ from dense because its K/V context was restricted.

Report `mesh_current_nmse`, `mesh_rhyme_nmse`, `mesh_fixed_nmse`, and `mesh_oracle_nmse`. Relative L2 values are also logged for readability, but normalized MSE is primary because the oracle minimizes squared interpolation error.

## 2. Exact fixed-budget interpolation oracle

Wan2.1 has only 21 latent frames in the canonical 81-frame setting. CoFrame computes every interval's exact linear-interpolation SSE from a frame Gram matrix, then solves

\[
A^*=\arg\min_{|A|=K} E_{\mathrm{mesh}}(A)
\]

with an \(O(KF^2)\) dynamic program, subject to boundary and minimum-gap constraints.

Two complementary oracle statistics are reported:

\[
\text{HeadroomRecovery}=
\frac{E_{\mathrm{Rhyme}}-E_{\mathrm{CoFrame}}}
{E_{\mathrm{Rhyme}}-E_{\mathrm{Oracle}}}.
\]

Rhyme is 0, oracle is 1, and a negative value means CoFrame is worse than Rhyme. When Rhyme is already numerically equal to oracle, the ratio is undefined and is reported as `null`. Always report the absolute oracle regret

\[
E_{\mathrm{CoFrame}}-E_{\mathrm{Oracle}}
\]

alongside the normalized ratio.

## 3. Controller-action metric: one-swap decision quality

Frame-error correlation is not sufficient because the controller does not independently classify frames; it chooses one legal remove/add swap under a fixed budget. At each probe, CoFrame enumerates every legal one-swap action and computes its true dense interpolation gain.

The probe reports:

- Spearman correlation between predicted and true swap gains;
- `gain_recovery`: true gain of the chosen action divided by the best available true gain;
- absolute and normalized regret relative to the best action;
- top-1 exact agreement;
- whether the controller correctly chose no-op.

This is the most direct metric for whether the defect field produces the right control action.

## 4. Realized sparse-operator error

The actual sparse block output is also compared with the dense block output:

\[
E_{\mathrm{operator}}=
\frac{\|\Delta h^{\mathrm{dense}}-
\Delta \widetilde h^{\mathrm{sparse}}\|_2^2}
{\|\Delta h^{\mathrm{dense}}\|_2^2}.
\]

Unlike mesh-only error, this includes interpolation, anchor-only K/V context restriction, and implementation effects. Report total NMSE/relative L2 plus non-anchor CVaR-10. With 21 latent frames and 9 anchors, CVaR-10 averages the two worst skipped-frame errors and is more stable than relying on a single maximum or P95 alone.

The exact-anchor delta error is logged separately. A large anchor error under `anchor_only` but near-zero anchor error under `full_kv` means context removal, not frame placement, is the dominant failure source.

## 5. Downstream propagation

A local block improvement matters only if later dynamics retain it. Starting from the dense and sparse outputs of a probed block, both states are passed through the same subsequent **dense** blocks. Relative error and tail error are logged after `+1` and `+3` blocks.

This distinguishes:

- local errors that later dense blocks correct;
- errors that are amplified by the network;
- a good selector paired with a harmful sparse operator.

## 6. End-to-end metrics

After the mechanism passes the block-level gate, compare complete runs against the same-seed dense reference using:

- final-latent normalized MSE / relative L2 and cosine;
- temporal-gradient relative L2, which emphasizes motion and transitions rather than global latent offsets;
- frame-error CVaR-10 and maximum;
- denoising latency and speedup measured with CUDA synchronization.

Decoded PSNR, SSIM, LPIPS and benchmark quality are stage-two confirmation metrics, not the first mechanism gate.

## 7. Aggregation

Treat every prompt–seed–step–block cell as paired. Report median paired improvement, win rate, and a prompt-clustered bootstrap confidence interval in addition to the mean. A convincing early result should show all of the following directions:

1. CoFrame lowers mesh-only NMSE relative to Rhyme at the same `K`;
2. headroom recovery is positive and one-swap regret decreases after observing the current defect;
3. realized operator and `+1/+3` propagation errors move in the same direction;
4. complete-run endpoint error improves without losing the intended latency advantage.

A high risk/error correlation without better swap decisions or mesh NMSE is not sufficient evidence for the method.

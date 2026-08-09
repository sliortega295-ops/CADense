# CoFrame method specification

## 1. Problem

For a video DiT block \(B_{l,t}\), let \(h\in\mathbb{R}^{F\times P\times D}\) be the hidden state with \(F\) latent frames and \(P\) spatial tokens per frame. A sparse frame mesh \(A\subset\{0,\ldots,F-1\}\) computes only exact anchor-frame outputs and reconstructs the rest.

The central question is not whether a frame has high semantic change in general. It is whether the current anchor mesh can accurately represent the update produced by this particular block and denoising time.

## 2. Rhyme prior

At the end of dense warmup, CoFrame forms the standard one-step clean-latent proxy

\[
\widehat x_0 = x_t - \sigma_t v_\theta(x_t,t,c).
\]

Per-frame representations are flattened from \(\widehat x_0\). Frames are scanned in time, opening a new anchor when cosine similarity to the most recent anchor falls below a threshold. Adjacent transition scores fill or trim the set to exactly \(K\) anchors. CoFrame and the `rhyme` baseline use the same initialization.

## 3. Sparse block operator

For each sparse block:

1. normalize all hidden tokens;
2. compute self-attention queries only for anchor tokens;
3. use either anchor-only K/V or full-frame K/V;
4. run cross-attention and FFN only for anchor tokens;
5. reconstruct non-anchor block outputs.

The default reconstruction interpolates block updates:

\[
\Delta h^l_a = B_{l,t}(h^l)_a-h^l_a,\qquad
\widetilde h^{l+1}_f=h^l_f+I_A(\Delta h^l_A)_f.
\]

`state` interpolation is retained as an ablation.

## 4. Leave-one-out block defect

For three neighboring exact anchors \(a<v<b\), pretend validator \(v\) was absent:

\[
\widetilde{\Delta h}^{(-v)}_v=
\frac{b-v}{b-a}\Delta h_a+
\frac{v-a}{b-a}\Delta h_b.
\]

The normalized residual is

\[
d_{v,l,t}=
\frac{\|S(\Delta h_v-\widetilde{\Delta h}^{(-v)}_v)\|_\mathrm{RMS}}
{\|S(\Delta h_v)\|_\mathrm{RMS}+\epsilon},
\]

where \(S\) is either the identity or a fixed random channel sketch. No extra DiT block call is needed because all three anchor outputs were already computed.

This is a same-mask surrogate: in `anchor_only` mode the exact validator itself was evaluated with sparse K/V context. The dense-block oracle probe measures the resulting bias.

## 5. Risk field and fixed-budget mesh update

Each validator defect is spread over the interval between its neighboring anchors. CoFrame maintains

\[
r_f = r_\mathrm{floor}+\lambda r_f^\mathrm{Rhyme}+r_f^\mathrm{dynamic},
\]

where dynamic risk is an EMA of past block-group observations.

For an interval \((a,b)\), only non-anchor frames contribute. Its interpolation envelope is

\[
e_f=4u_f(1-u_f),\quad u_f=\frac{f-a}{b-a}.
\]

The controller cost is proportional to the mean envelope-weighted risk multiplied by \((b-a)^p\). Because standard Wan generation has only 21 latent frames, CoFrame exhaustively searches all one-anchor swaps, keeps the anchor count fixed, and accepts a move only if cost reduction exceeds a threshold plus movement penalty.

Anchors are held fixed within a block group and may change only at group boundaries. This avoids per-block shape thrashing.

## 6. CFG consistency

The conditional branch chooses and updates the mesh. Its exact per-block anchor schedule is then replayed by the unconditional branch. This guarantees identical sparse shapes across CFG branches and prevents the controller from receiving two incompatible observations at one denoising step.

## 7. Causal oracle probe

At selected cells, CoFrame evaluates both dense and sparse versions of the same block from the same input. It reports per-frame relative RMS error and compares that error with:

- the static Rhyme prior;
- causal risk accumulated before the current block;
- the current block's newly measured defect field.

Correlations are computed on non-anchor frames. Anchor error is reported separately as a direct estimate of context-restriction error.

## 8. Claims this code can and cannot support

A successful experiment can support:

- block-conditional defects predict sparse interpolation failure better than a static semantic prior;
- fixed-budget online mesh adaptation improves quality over static Rhyme anchors;
- the measured controller overhead is small enough to preserve denoising speedup.

It cannot yet support:

- optimized end-to-end deployment across arbitrary dynamic frame counts;
- generalization beyond Wan T2V without a new integration;
- superiority on distilled few-step models before that checkpoint is explicitly tested.

# Trajectory Interaction Screen

## Question and claim boundary

The calibrated step-block experiment rejected one specific construction: costs
measured by changing one `(denoising step, block group)` on a uniform-K9
trajectory could not be added to predict a deployed multi-cell schedule. It did
**not** establish that dynamic K is useless.

This preregistered screen asks one deliberately narrower question:

> Do two within-step frame-budget interventions have a joint trajectory effect
> that is materially different from the sum of their isolated effects?

The screen measures causal pairwise interactions only. It does not implement a
planner, select a schedule, or make a quality/latency claim.

## Frozen model and runtime contract

- base source: `e3a713eb11421c4ea505ae77386ce478cd5b6f5c` plus the
  calibrated-budget implementation;
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, BF16 transformer;
- 480x832, 81 decoded / 21 latent frames, 50 denoising steps;
- five dense warm-up steps;
- sparse blocks 3-26, block-group size 3, `full_kv`, delta interpolation;
- candidate K values remain exactly `{6, 9, 12, 21}`;
- eight frozen prompts, seed 0;
- four independent GPU jobs, no DDP;
- the only deployed trajectory is Uniform K=9;
- counterfactual arms are discarded and never modify that main trajectory;
- video decode and endpoint evaluation are out of scope;
- latency is `NOT_REPORTED`.

Runtime step indices are zero-based. Step 5 is therefore the first sparse step
after the five-step dense warm-up. Group-to-block mapping is:

| Group | DiT blocks |
|---:|---:|
| 0 | 3-5 |
| 1 | 6-8 |
| 2 | 9-11 |
| 3 | 12-14 |
| 4 | 15-17 |
| 5 | 18-20 |
| 6 | 21-23 |
| 7 | 24-26 |

The immutable machine-readable plan is
`configs/trajectory_interaction_screen.json`.

## Frozen prompts

| ID | Prompt |
|---|---|
| `p0_s0` | A red toy car moves smoothly from left to right across a wooden table. |
| `p1_s0` | A red toy car makes a sharp U-turn around a blue cube. |
| `p2_s0` | A gymnast performs a fast cartwheel and lands upright. |
| `p3_s0` | Two dancers cross paths, exchange positions, and continue in opposite directions. |
| `p4_s0` | A brown dog jumps to catch a small yellow ball thrown through the air. |
| `p5_s0` | A hummingbird rapidly moves between two red flowers while the camera remains fixed. |
| `p6_s0` | The camera circles around a stationary bronze statue in a plaza. |
| `p7_s0` | A cup sits on a table while steam rises slowly; the rest of the scene remains still. |

## Frozen pairs and arms

Exactly six ordered pairs are probed:

| Pair ID | Step | `group_i -> group_j` | Distance stratum |
|---|---:|---:|---|
| `step05_g0_g1_adjacent` | 5 | 0 -> 1 | adjacent |
| `step05_g0_g7_long` | 5 | 0 -> 7 | long |
| `step20_g3_g4_adjacent` | 20 | 3 -> 4 | adjacent |
| `step20_g0_g7_long` | 20 | 0 -> 7 | long |
| `step40_g6_g7_adjacent` | 40 | 6 -> 7 | adjacent |
| `step40_g0_g7_long` | 40 | 0 -> 7 | long |

Each pair has seven unique arms. `K_i` and `K_j` are the budgets at the two
intervened groups; all groups strictly between them use K=9.

| Arm | `K_i` | `K_j` | Role |
|---|---:|---:|---|
| `k9_k9` | 9 | 9 | shared factorial baseline |
| `k6_k9` | 6 | 9 | first-only, 6-to-12 orientation |
| `k9_k12` | 9 | 12 | second-only, 6-to-12 orientation |
| `k6_k12` | 6 | 12 | joint, 6-to-12 orientation |
| `k12_k9` | 12 | 9 | first-only, 12-to-6 orientation |
| `k9_k6` | 9 | 6 | second-only, 12-to-6 orientation |
| `k12_k6` | 12 | 6 | joint, 12-to-6 orientation |

Thus the two factorial orientations are:

```text
6-to-12: H00=k9_k9, H10=k6_k9,  H01=k9_k12, H11=k6_k12
12-to-6: H00=k9_k9, H10=k12_k9, H01=k9_k6,  H11=k12_k6
```

Both joint arms have `K_i + K_j = 18`, exactly matching the two-cell K9
baseline. The single-intervention arms exist to identify the factorial main
effects; they are not standalone budget-matched deployment claims.

## Causal branch protocol

For a pair, first run blocks before `group_i` once on the Uniform-K9 main
trajectory. Snapshot the hidden state at the exact entry to `group_i`. All
seven arms and one all-dense reference must be cloned from that same tensor and
must use identical conditioning, timestep embeddings, rotary embeddings, CFG
batching, masks, dtype, and deterministic inputs.

From this common entry:

1. The dense reference executes every remaining block in the current
   transformer step densely.
2. Each arm runs `group_i` at `K_i`.
3. Any sparse groups between `group_i` and `group_j` run at K=9. No arm is
   reattached to the main trajectory or dense reference at `group_j`.
4. The arm runs `group_j` at `K_j` and records `after_j`.
5. Starting from its own changed state, every arm executes the next three DiT
   blocks densely and records `plus_3_dense`.
6. It continues through the remaining blocks densely, applies the shared Wan
   output head (`norm_out` and `proj_out`), and records the conditional
   transformer prediction as `step_end`, immediately before CFG composition
   and the scheduler update. For a group ending at block 26, the pre-head state
   of `plus_3_dense` and `step_end` is shared, but the latter remains a distinct
   post-head diagnostic.
7. The branch is discarded. The Uniform-K9 main trajectory resumes from its
   untouched state.

The common all-dense branch supplies the reference hidden state at all three
checkpoints. Every arm record must identify the common-entry/reference IDs so
the analyzer can reject mismatched inputs rather than silently compare them.

Required causal audit checks are:

- all arms in a pair have the same group-i entry fingerprint;
- all arms use one dense-reference fingerprint per checkpoint;
- the seven-arm execution order cannot affect results;
- intermediate group budgets are all 9;
- the deployed main-trajectory budget is 9 everywhere;
- probes do not modify the main trajectory (probe-on/off smoke outputs must be
  bitwise identical, or have maximum absolute latent difference exactly zero);
- probes execute only on the conditional branch; the unconditional CFG replay
  must not create a second probe, while both deployed CFG branches retain the
  same Uniform-K9 block schedule;
- all saved metrics are finite.

## Measurements and interaction definitions

At the state-space checkpoints `after_j` and `plus_3_dense`, let `X_i` be the
common group-i entry tensor, `H_a^c` be an arm's full hidden-state tensor at
checkpoint `c`, and `H_D^c` be the state produced at the same checkpoint by the
common all-dense tail from `X_i`. Define tail updates `T_a^c = H_a^c - X_i`
and `T_D^c = H_D^c - X_i`. The scalar response is the dense-referenced
tail-delta normalized mean-squared error:

```text
E_a^c = ||T_a^c - T_D^c||_2^2 / (||T_D^c||_2^2 + 1e-12)
```

At `step_end`, the same factorial definitions are applied to the conditional
transformer predictions, normalized by the common dense prediction energy.
For each orientation and checkpoint, define the isolated and joint scalar
effects relative to the shared K9/K9 arm:

```text
delta_i     = E10 - E00
delta_j     = E01 - E00
delta_joint = E11 - E00
I_scalar    = delta_joint - delta_i - delta_j
tau         = max(1e-12, 0.01 * |E00|)
rho_scalar  = |I_scalar| / (|delta_i| + |delta_j| + tau)
```

The vector factorial residual is computed directly on the full hidden states,
not on sketches or scalar errors:

```text
D_i        = H10 - H00
D_j        = H01 - H00
D_joint    = H11 - H00
R_vector   = D_joint - D_i - D_j
           = H11 - H10 - H01 + H00
rho_vector = ||R_vector||_2 / (||D_i||_2 + ||D_j||_2 + 1e-12)
```

The common entry and dense-reference terms cancel algebraically from
`R_vector`; they remain mandatory for the scalar tail-delta response and causal
audit. Accumulate norms in float64 over chunks. Do not change the
representation, denominator, `tau`, numerical floor, or checkpoint after
observing results.

Also report the additive scalar prediction `delta_i + delta_j`, the realized
`delta_joint`, signed `I_scalar`, `tau`, denominator magnitudes, and a meaningful
sign-flip indicator. A sign flip is true only when the additive prediction and
`delta_joint` have opposite signs **and** both magnitudes exceed `tau`.
Exact/near-zero effects are not flips. Sign-flip rate is descriptive and is
**not** a decision-gate requirement.

## Completeness contract

With eight prompts, six pairs, and seven arms, the trace must contain exactly:

```text
8 x 6 x 7 = 336 arm records
```

Each arm record contains all three checkpoint measurements. With two
orientations per prompt/pair, the analyzer must produce exactly:

```text
8 x 6 x 2 = 96 orientation records
```

Each orientation record contains scalar and vector interaction diagnostics for
all three checkpoints. Duplicate or missing prompt/pair/arm/orientation keys,
non-finite values, failed reference/entry fingerprints, non-K9 main/intermediate
budgets, or probe leakage make the scientific gate not evaluable. Such a run is
reported as `INCOMPLETE_TRAJECTORY_INTERACTION_SCREEN`, fixed only by rerunning
the identical preregistered protocol, and never counted as a negative result.

## Preregistered decision gate

For `after_j` and `plus_3_dense`, first take the median `rho_scalar` across the
12 orientation records belonging to each prompt. The "overall prompt median"
is the median of those eight prompt-level medians.

A complete, causally valid screen outputs
`SUPPORT_STRONG_WITHIN_STEP_TRAJECTORY_INTERACTION` only when **all** of the
following frozen conditions hold:

1. `after_j` scalar: the overall prompt median is at least 0.25 and at least
   6/8 prompt-level medians are at least 0.25.
2. `plus_3_dense` scalar: the overall prompt median is at least 0.25 and at
   least 6/8 prompt-level medians are at least 0.25.
3. The overall median `rho_vector` across all 96 `plus_3_dense` orientation
   records is at least 0.10.
4. At `plus_3_dense`, each of four marginal strata--orientation `6-to-12`,
   orientation `12-to-6`, distance `adjacent`, and distance `long`--has median
   `rho_scalar >= 0.25` and median `rho_vector >= 0.10`.
5. At `plus_3_dense`, at least two of the three step strata `{5, 20, 40}` each
   have median `rho_scalar >= 0.25` and median `rho_vector >= 0.10`.

If condition 1 passes but any required `plus_3_dense` or stratum condition
fails, output:

```text
LOCAL_STATE_DEPENDENCE_ONLY
```

Otherwise output:

```text
NO_STRONG_WITHIN_STEP_INTERACTION
```

`step_end`, signed interactions, sign-flip rates, per-pair results, and
near-zero denominator counts are mandatory diagnostics but cannot rescue or
reverse the gate.

## Execution and artifacts

Run each complete prompt independently on one GPU; do not split a prompt's
arms across GPUs:

```bash
bash scripts/run_trajectory_interaction_wan21.sh \
  "<prompt>" \
  outputs/trajectory_interaction/pX_s0 \
  0 pX_s0 \
  configs/trajectory_interaction_screen.json
```

Before the eight full jobs, CPU tests and one bounded probe-on/off smoke test
must pass. Preserve:

- raw traces and summaries for every prompt;
- the 336-arm table and 96-orientation table;
- checkpoint-level scalar effects and vector residual metrics;
- entry/reference fingerprints and causal-audit results;
- stdout/stderr logs, exact source commit, environment and GPU inventory;
- the immutable JSON plan, final summary, and human-readable report.

Publish the complete package as a GitHub Release or an explicitly named result
branch, including a file manifest and SHA-256 digest.

## Prohibited post-hoc changes and out-of-scope claims

After inspecting any GPU result, do not change the six pairs, seven arms,
directions, checkpoints, prompts, seed, K set, interpolation scheme, dense-tail
reference, interaction formulas, thresholds, aggregation, or decision labels.
Do not add favorable cells or drop unfavorable complete cells. Identical reruns
are allowed only for recorded operational failures.

This screen deliberately excludes cross-step interactions, scheduler-state
interventions, endpoint latents/videos, perceptual quality, an online selector,
Adaptive-K deployment, beam search, MPC, RL, and latency/speedup conclusions.
Passing it motivates a separately preregistered sequential or receding-horizon
planner experiment; failing it redirects attention to the local objective or
surrogate rather than proving that every form of dynamic K is ineffective.

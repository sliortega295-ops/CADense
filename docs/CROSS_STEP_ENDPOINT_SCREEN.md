# Cross-Step Endpoint Factorial Screen

## Question and claim boundary

The prior exact-budget LOPO schedule changed 170 `(step, group)` cells.  Its
local operator NMSE worsened on all eight held-out prompts, while endpoint
latent NMSE improved on all eight.  A later preregistered screen found no strong
pairwise non-additivity inside one denoising step.  This experiment asks two
narrow questions without implementing another scheduler:

1. Does an early budget change interact non-additively with a much later budget
   change after CFG and scheduler updates have carried its state forward?
2. Are the existing local operator and +3 propagation diagnostics aligned with
   the final dense-referenced endpoint error for isolated interventions?

It is a causal factorial screen, not an online selector, planner, video-quality,
or speed experiment.  Passing it motivates a later, separately preregistered
planner.  Failing it does not establish that every form of dynamic K is useless.

The immutable machine plan is
`configs/cross_step_endpoint_screen.json`.  Do not change this runbook or plan
after inspecting any full-screen endpoint result.

## Frozen workload and resources

- source parent: trajectory-interaction implementation commit
  `2a1dd8189e504fb129c500e89b18a42d00c95535`;
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, BF16 transformer;
- 480x832, 81 decoded / 21 latent frames, 50 denoising steps;
- five dense warm-up steps;
- sparse blocks 3-26, eight groups of three blocks;
- `full_kv`, delta interpolation;
- exact-frame candidates remain `{6,9,12,21}`;
- all sparse operators use `uniform_select(F=21,K,force_boundaries=True)`;
- the same eight frozen prompts, seed 0, guidance 5, flow shift 3;
- four independent prompt jobs, no DDP or model parallelism;
- physical GPUs 1-4 are assigned because GPU0 was occupied at launch.  This is
  an allocation-only deviation and does not change the scientific contract;
- latency, throughput, and speedup are `NOT_REPORTED`.

Runtime step and group indices are zero-based.  Group 0 is DiT blocks 3-5,
group 2 blocks 9-11, group 3 blocks 12-14, and group 5 blocks 18-20.

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

## Frozen pairs, arms, and budget fairness

The source intervention is shared by three cross-step pairs:

| Pair ID | Source `i` | Target `j` |
|---|---:|---:|
| `step22_g0_to_step44_g5` | `(22,0)` | `(44,5)` |
| `step22_g0_to_step47_g3` | `(22,0)` | `(47,3)` |
| `step22_g0_to_step49_g2` | `(22,0)` | `(49,2)` |

These cells are frozen from the old LOPO schedules before this screen: the
shared source received extra budget across all eight folds, while each target
received reduced budget across all eight folds.  This is a targeted test of the
old schedule's early-spend/late-save pattern, not a claim about every possible
cross-step pair.

Each logical pair has exactly seven arms.  Every unlisted one of the 360 sparse
`(step,group)` slots uses K=9.

| Arm | `K_i` | `K_j` | Role |
|---|---:|---:|---|
| `k9_k9` | 9 | 9 | shared Uniform-K9 baseline |
| `k6_k9` | 6 | 9 | reverse source-only |
| `k9_k12` | 9 | 12 | reverse target-only |
| `k6_k12` | 6 | 12 | reverse budget-neutral joint |
| `k12_k9` | 12 | 9 | primary source-only |
| `k9_k6` | 9 | 6 | primary target-only |
| `k12_k6` | 12 | 6 | primary budget-neutral joint |

The factorial mappings are fixed:

```text
primary 12-to-6: H00=k9_k9, H10=k12_k9, H01=k9_k6,  H11=k12_k6
reverse  6-to-12: H00=k9_k9, H10=k6_k9,  H01=k9_k12, H11=k6_k12
```

For both joint arms, `K_i+K_j=18`, exactly matching K9+K9; hence every joint
full trajectory has total K=3240 and exact average K=9.  The singleton arms are
required only to identify factorial effects and are not standalone
budget-matched deployment claims.

The common source permits physical reuse without changing the logical design.
Per prompt there are exactly 15 unique schedules:

```text
1 baseline + 2 shared source-only
           + 3 targets * (2 target-only + 2 joint) = 15
```

Every physical schedule must store its complete 360-slot schedule and SHA-256.
Logical records must point to a physical `schedule_id`; reused records must have
the same schedule SHA and endpoint latent SHA.

## Causal execution protocol

For each prompt, build all 15 full trajectories from an identical prompt,
seed, initial noise, text conditioning, timestep sequence, scheduler, and
source revision.  The runner may share immutable computation up to the first
branch, but semantic results must equal independently executing each complete
schedule.

1. Run one all-dense trajectory under the same prompt/seed/sampler contract and
   retain its final latent as the common endpoint reference.
2. Run the Uniform-K9 baseline and fourteen unique intervention schedules.
3. Conditional and unconditional CFG transformer calls in one trajectory must
   receive the identical 360-slot K schedule.  No branch may reattach to a
   Uniform-K9 or dense state after its first intervention.
4. Continue every schedule through step 49 and the final CFG/scheduler update;
   save its final latent.  Counterfactual schedules never select another
   schedule or affect one another.
5. Record the exact source commit/tree, plan and protocol SHA-256, model
   fingerprint, prompt ID/text, seed, schedule ID/SHA, full budget overrides and
   360-slot physical schedule, output latent SHA, shape/dtype/finiteness, and
   peak allocated/reserved memory.  Timing remains `NOT_REPORTED`.
6. A bounded preflight must first prove schema validity, exact hashes, model and
   dependency identity, 15 unique schedules, exact joint average K=9, CFG
   schedule equality, finite outputs, and deterministic baseline equivalence.
   Smoke outputs establish connectivity only and are not scientific results.

The dense reference and all 15 sparse schedules are complete endpoint runs.
Endpoint errors compare each saved latent to the prompt's dense latent.  The
factorial vector residual is computed directly on the four full endpoint
latents and therefore does not use the dense latent algebraically.

## Frozen metrics

For each arm `a`, let `L_a` be its final latent and `L_D` the all-dense final
latent.  Record:

- `endpoint_nmse = ||L_a-L_D||^2 / (||L_D||^2 + 1e-12)`;
- endpoint relative L2 and cosine as diagnostics;
- dense-referenced endpoint temporal-gradient relative L2, using the existing
  repository metric without modification;
- endpoint frame-error diagnostics already emitted by the existing metric code.

For each orientation and each scalar endpoint error `E`, use the existing
factorial definition unchanged:

```text
delta_i     = E10 - E00
delta_j     = E01 - E00
delta_joint = E11 - E00
I_scalar    = delta_joint - delta_i - delta_j
tau         = max(1e-12, 0.01 * abs(E00))
rho_scalar  = abs(I_scalar) / (abs(delta_i) + abs(delta_j) + tau)
```

A meaningful sign flip is true only when `delta_i+delta_j` and `delta_joint`
have opposite signs and both magnitudes exceed `tau`.  It is descriptive and
is not a gate.  On full final latents, accumulate in float64 chunks:

```text
R = L11 - L10 - L01 + L00
rho_vector = ||R|| / (||L10-L00|| + ||L01-L00|| + 1e-12)
```

### Objective-alignment diagnostic

Use the preserved calibrated Phase-A surface, never a newly measured or edited
surface:

```text
inputs/calibrated_step_block/budget_error_surface.csv
SHA256 de0c409905a0f77b341001559edb6bb10ee0750cf2fab66f12f25528a63819b5
```

It has exactly 11,520 unique rows and joins on
`(prompt_id,step,group,k)`.  Required fields are `operator_nmse` and
`propagation_h3_relative_l2`; the complete header is frozen in the JSON plan.
A missing/mismatched file, row, key, column, anchor set, or digest fails closed.

Only singleton effects are used, because their intervention slot has the same
Uniform-K9 prefix as Phase A.  Deduplicate the shared source singleton across
three pairs, yielding exactly:

```text
8 prompts * (2 source effects + 3 targets * 2 target effects) = 64 effects
```

For each effect, subtract its same-prompt K9 surface value and its Uniform-K9
endpoint value.  Evaluate the Cartesian product of both Phase-A predictors and
both endpoint outcomes, yielding four combinations:

- operator-NMSE delta versus endpoint-NMSE delta;
- operator-NMSE delta versus endpoint temporal-gradient-relative-L2 delta;
- +3-relative-L2 delta versus endpoint-NMSE delta;
- +3-relative-L2 delta versus endpoint temporal-gradient-relative-L2 delta.

For every combination report the eight within-prompt Spearman correlations,
their median, a 20,000-draw prompt-cluster bootstrap 95% interval using frozen
seed 20260813, prompt-balanced signed-direction agreement, each prompt's sign
agreement, Pearson diagnostics, and per-slot effects.  This diagnostic never
rescues the primary interaction gate.

## Completeness and causal-integrity gate

A complete screen contains exactly:

- 8 prompt jobs and 8 dense endpoint references;
- 120 unique physical sparse schedules (`8*15`);
- 168 logical arm records (`8*3*7`);
- 48 orientation records (`8*3*2`);
- 64 deduplicated singleton-alignment effects.

Duplicate/missing IDs, a non-finite metric or latent, a source/model/hash
mismatch, a changed prompt, a schedule outside K={6,9,12,21}, a non-K9 unlisted
slot, a non-budget-neutral joint, CFG schedule disagreement, incorrect reuse,
a surface mismatch, or missing dense reference yields only:

```text
INCOMPLETE_CROSS_STEP_ENDPOINT_SCREEN
```

Repair only the operational fault and rerun the identical frozen protocol.  An
incomplete run is never counted as a negative scientific result.

## Preregistered gates

All thresholds below are frozen before full endpoint inspection.  Smaller
error is better; relative improvement is `(baseline-method)/baseline`.

### Primary cross-step interaction

Only `12_to_6` is primary.  For each scalar metric--endpoint NMSE and endpoint
temporal-gradient relative L2--first take median `rho_scalar` across the three
pairs within each prompt.  Each metric independently passes only if:

1. the median of the eight prompt medians is at least 0.25; and
2. at least 6/8 prompt medians are at least 0.25.

In addition, all must hold:

3. at least 2/3 pair-specific median endpoint-NMSE `rho_scalar` values are at
   least 0.25; and
4. pooled median final-latent `rho_vector` across the 24 primary orientation
   records is at least 0.10.

Set the independent mechanism label
`SUPPORT_CROSS_STEP_INTERACTION` only if all four conditions pass.  The final
action label is chosen later by the frozen decision tree.  The reverse
`6_to_12` orientation is mandatory as a control but cannot rescue the primary
gate.  Sign flips, individual pairs, and alternate endpoint diagnostics
cannot rescue it either.

### Atomic budget-transfer diagnostic

For each prompt and each of endpoint NMSE and temporal-gradient error, take the
median relative improvement of `k12_k6` over `k9_k9` across the three pairs.
This diagnostic passes only if both metrics have strictly positive median
improvement across prompts and each metric wins at least 6/8 prompts.  It never
rescues the interaction gate.

### Objective alignment

Each predictor/outcome combination passes only if all conditions hold:

1. the median of the eight within-prompt Spearman correlations is at least 0.50;
2. its 20,000-draw prompt-cluster bootstrap 95% CI has lower bound strictly
   greater than zero;
3. prompt-balanced signed-direction agreement is at least 75%; and
4. at least 6/8 prompts individually have signed-direction agreement at least
   75%.

A predictor is `aligned` only if both endpoint outcomes pass for that predictor.
Thus +3 is aligned only if both +3 combinations pass.  Operator-only alignment
cannot rescue the old operator-optimized schedule, cannot trigger another LOPO
experiment, and never rescues the primary interaction gate.

## Frozen decision tree

Apply in this exact order after the completeness gate.  Always report the
independent `mechanism_label=SUPPORT_CROSS_STEP_INTERACTION` when the primary
interaction gate passes, even though the action label below is single-valued.

1. If both the primary interaction and atomic-transfer gates pass, emit
   `ADVANCE_SEQUENTIAL_PLANNER`.  The planner itself remains `NOT_RUN` here.
2. If primary interaction passes but atomic transfer fails, emit
   `SUPPORT_CROSS_STEP_INTERACTION_NO_PLANNER_ADVANCE`: the mechanism exists,
   but this frozen budget transfer provides no quality basis for planner work.
3. If primary interaction fails but atomic transfer passes, emit
   `SUPPORT_STATIC_BUDGET_TRANSFER_PRIOR`.  This supports only a fixed
   early-spend/late-save prior, not adaptive or sequential planning.
4. If both primary interaction and atomic transfer fail, but the +3 predictor is
   aligned with both endpoint outcomes, emit `TEST_PLUS3_OBJECTIVE_LOPO`.  Any
   such LOPO is a later preregistered experiment and is `NOT_RUN` here.
5. Otherwise emit `REJECT_CURRENT_CROSS_STEP_EXPLANATION`.  Operator-only
   alignment cannot change this label and cannot rescue the old schedule.

Always report primary interaction, atomic transfer, all four alignment
combinations, each predictor-level alignment result, and the final action label.

## Execution, artifacts, and prohibited changes

Run one complete prompt job per GPU; do not distribute schedules from one prompt
across GPUs.  GPUs 1-4 process four prompts concurrently in two waves.  Preserve
all dense and sparse latents, traces, summaries, full schedules and mappings,
source/model fingerprints, exact commands, logs, memory samples, the Phase-A
surface input, arm/orientation/alignment tables, summary, report, environment,
failures, file manifest, and internal SHA-256 manifest.  Publish a reproducible
GitHub Release or explicit result branch.

After any full endpoint result is visible, do not change prompts, pairs, arms,
orientation roles, K values, uniform meshes, budget constraint, metrics,
references, surface, formulas, thresholds, aggregation, or decision tree.  Do
not add favorable pairs or remove unfavorable complete records.  Identical
reruns are permitted only for documented operational failures.

Online defect, curvature, Proxy-DP, learned selection, Adaptive-K control,
sequential planning, beam search, MPC, RL, video decoding, VBench, perceptual
video evaluation, kernel optimization, and latency/speedup claims are all
`NOT_RUN` or `NOT_REPORTED` as specified by the JSON plan.

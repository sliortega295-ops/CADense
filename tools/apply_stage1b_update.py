from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"pattern not found in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    file = ROOT / path
    text = file.read_text(encoding="utf-8")
    if marker in text:
        return
    file.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# config.py: add FIS baseline and controlled refresh ablations.
# ---------------------------------------------------------------------------
replace_once(
    "coframe/config.py",
    'Method = Literal["dense", "fixed", "rhyme", "coframe"]\n',
    'Method = Literal["dense", "fixed", "fis", "rhyme", "coframe"]\n'
    'RefreshSignal = Literal["defect", "none", "gap_only", "shuffled"]\n',
)
replace_once(
    "coframe/config.py",
    '    defect_target: DefectTarget = "delta"\n\n    # RhymeFlow-style clean-latent prior.\n',
    '    defect_target: DefectTarget = "delta"\n\n'
    '    # CoFrame source-attribution ablations. "none" freezes the initial\n'
    '    # Rhyme mesh; "gap_only" remeshes with a uniform risk field;\n'
    '    # "shuffled" preserves defect magnitudes but breaks frame alignment.\n'
    '    refresh_signal: RefreshSignal = "defect"\n'
    '    shuffle_defect_seed: int = 20260810\n\n'
    '    # FIS-DiT-style interleaved baseline. stride=0 derives ceil(F/K).\n'
    '    # A dense tail is optional and explicitly counted in latency.\n'
    '    fis_anchor_stride: int = 0\n'
    '    fis_dense_tail_steps: int = 0\n\n'
    '    # RhymeFlow-style clean-latent prior.\n',
)
replace_once(
    "coframe/config.py",
    '    oracle_metric_chunk_size: int = 65_536\n\n    trace_path: str | None = None\n',
    '    oracle_metric_chunk_size: int = 65_536\n'
    '    probe_counterfactual_methods: Sequence[str] = field(default_factory=lambda: ("rhyme", "fis", "fixed"))\n\n'
    '    trace_path: str | None = None\n',
)
replace_once(
    "coframe/config.py",
    '        if self.method not in {"dense", "fixed", "rhyme", "coframe"}:\n',
    '        if self.method not in {"dense", "fixed", "fis", "rhyme", "coframe"}:\n',
)
replace_once(
    "coframe/config.py",
    '        if self.num_anchors < 1:\n',
    '        if self.refresh_signal not in {"defect", "none", "gap_only", "shuffled"}:\n'
    '            raise ValueError(f"Unsupported refresh_signal: {self.refresh_signal}")\n'
    '        if self.fis_anchor_stride < 0:\n'
    '            raise ValueError("fis_anchor_stride must be >= 0")\n'
    '        if self.fis_dense_tail_steps < 0:\n'
    '            raise ValueError("fis_dense_tail_steps must be >= 0")\n'
    '        invalid_counterfactuals = set(self.probe_counterfactual_methods) - {"rhyme", "fis", "fixed"}\n'
    '        if invalid_counterfactuals:\n'
    '            raise ValueError(f"Unsupported probe counterfactuals: {sorted(invalid_counterfactuals)}")\n'
    '        if self.num_anchors < 1:\n',
)
replace_once(
    "coframe/config.py",
    '    def should_probe(self, step_index: int, block_index: int) -> bool:\n        return step_index in set(self.oracle_probe_steps) and block_index in set(self.oracle_probe_blocks)\n\n    def to_dict(self) -> dict[str, Any]:\n',
    '    def should_probe(self, step_index: int, block_index: int) -> bool:\n'
    '        return step_index in set(self.oracle_probe_steps) and block_index in set(self.oracle_probe_blocks)\n\n'
    '    def is_sparse_step(self, step_index: int, total_steps: int) -> bool:\n'
    '        if self.method == "dense":\n'
    '            return False\n'
    '        if self.method == "fis" and self.fis_dense_tail_steps > 0:\n'
    '            return step_index < max(0, total_steps - self.fis_dense_tail_steps)\n'
    '        return True\n\n'
    '    def to_dict(self) -> dict[str, Any]:\n',
)
replace_once(
    "coframe/config.py",
    '        result["oracle_probe_horizons"] = list(self.oracle_probe_horizons)\n        return result\n',
    '        result["oracle_probe_horizons"] = list(self.oracle_probe_horizons)\n'
    '        result["probe_counterfactual_methods"] = list(self.probe_counterfactual_methods)\n'
    '        return result\n',
)


# ---------------------------------------------------------------------------
# selection.py: budget-matched FIS interleaved schedule.
# ---------------------------------------------------------------------------
replace_once("coframe/selection.py", "from collections.abc import Sequence\n\nimport torch\n", "from collections.abc import Sequence\nimport math\n\nimport torch\n")
append_once(
    "coframe/selection.py",
    "def fis_interleaved_select(",
    r'''
def _fit_interleaved_budget(
    selected: Sequence[int],
    *,
    num_frames: int,
    num_anchors: int,
    force_boundaries: bool,
) -> list[int]:
    """Keep the FIS residue pattern while matching an exact frame budget."""
    chosen = sorted({int(i) for i in selected if 0 <= int(i) < num_frames})
    boundaries = [0, num_frames - 1] if force_boundaries and num_frames > 1 else []
    chosen = sorted(set(chosen + boundaries))

    if len(chosen) > num_anchors:
        protected = set(boundaries)
        interior = [i for i in chosen if i not in protected]
        keep_n = max(0, num_anchors - len(protected))
        if keep_n < len(interior):
            slots = torch.linspace(0, len(interior) - 1, keep_n).round().long().tolist() if keep_n else []
            interior = [interior[i] for i in sorted(set(slots))]
        chosen = sorted(set(boundaries + interior))

    while len(chosen) < num_anchors:
        candidates = [i for i in range(num_frames) if i not in chosen]
        if not candidates:
            break
        candidate = max(
            candidates,
            key=lambda i: (
                min(abs(i - anchor) for anchor in chosen) if chosen else num_frames,
                -i,
            ),
        )
        chosen.append(candidate)
        chosen.sort()
    return chosen[:num_anchors]


def fis_interleaved_select(
    num_frames: int,
    num_anchors: int,
    block_index: int,
    first_sparse_block: int,
    *,
    force_boundaries: bool = True,
    anchor_stride: int = 0,
) -> list[int]:
    """FIS-DiT-style interleaved anchor schedule with an exact budget.

    The paper uses r_l=(l-l0) mod n and selects frames satisfying
    (f-r_l) mod n=0, always keeping temporal boundaries.  For fair matched-K
    experiments we preserve that rotating residue set and deterministically
    fill/trim only when boundary insertion changes the exact count.
    """
    if num_frames < 1 or num_anchors < 1:
        raise ValueError("num_frames and num_anchors must be positive")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    stride = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
    phase = (int(block_index) - int(first_sparse_block)) % stride
    selected = [frame for frame in range(num_frames) if (frame - phase) % stride == 0]
    return _fit_interleaved_budget(
        selected,
        num_frames=num_frames,
        num_anchors=num_anchors,
        force_boundaries=force_boundaries,
    )
''',
)


# ---------------------------------------------------------------------------
# pipeline.py: initialize FIS without semantic warmup and support controls.
# ---------------------------------------------------------------------------
replace_once(
    "coframe/wan/pipeline.py",
    '    frame_representations_from_clean_latents,\n    rhyme_select,\n',
    '    fis_interleaved_select,\n    frame_representations_from_clean_latents,\n    rhyme_select,\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''    if config.method == "fixed":
        anchors = uniform_select(num_frames, config.num_anchors, config.force_boundaries)
        prior = torch.zeros_like(prior)
    elif config.method in {"rhyme", "coframe"}:
        anchors = rhyme_select(
            frame_representations,
            config.num_anchors,
            similarity_threshold=config.rhyme_similarity_threshold,
            force_boundaries=config.force_boundaries,
            min_gap=config.min_anchor_gap,
        )
    else:
        raise ValueError(f"A sparse controller is not defined for method={config.method}")

    controller = AdaptiveMeshController(
''',
    '''    rhyme_anchors = rhyme_select(
        frame_representations,
        config.num_anchors,
        similarity_threshold=config.rhyme_similarity_threshold,
        force_boundaries=config.force_boundaries,
        min_gap=config.min_anchor_gap,
    )
    fixed_anchors = uniform_select(num_frames, config.num_anchors, config.force_boundaries)

    controller_prior = prior
    controller_prior_weight = config.rhyme_prior_weight
    if config.method == "fixed":
        anchors = fixed_anchors
        controller_prior = torch.zeros_like(prior)
        controller_prior_weight = 0.0
    elif config.method == "fis":
        anchors = fis_interleaved_select(
            num_frames,
            config.num_anchors,
            config.sparse_block_start,
            config.sparse_block_start,
            force_boundaries=config.force_boundaries,
            anchor_stride=config.fis_anchor_stride,
        )
        controller_prior = torch.zeros_like(prior)
        controller_prior_weight = 0.0
    elif config.method in {"rhyme", "coframe"}:
        anchors = rhyme_anchors
        if config.method == "coframe" and config.refresh_signal == "gap_only":
            controller_prior = torch.zeros_like(prior)
            controller_prior_weight = 0.0
    else:
        raise ValueError(f"A sparse controller is not defined for method={config.method}")

    controller = AdaptiveMeshController(
''',
)
replace_once("coframe/wan/pipeline.py", "        prior_scores=prior,\n", "        prior_scores=controller_prior,\n")
replace_once("coframe/wan/pipeline.py", "        prior_weight=config.rhyme_prior_weight,\n", "        prior_weight=controller_prior_weight,\n")
replace_once(
    "coframe/wan/pipeline.py",
    '        max_swaps_per_refresh=config.max_swaps_per_refresh if config.method == "coframe" else 0,\n',
    '        max_swaps_per_refresh=(\n'
    '            config.max_swaps_per_refresh\n'
    '            if config.method == "coframe" and config.refresh_signal != "none"\n'
    '            else 0\n'
    '        ),\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '    )\n    return controller, anchors, prior\n',
    '    )\n'
    '    controller.rhyme_reference_anchors = list(rhyme_anchors)\n'
    '    controller.fixed_reference_anchors = list(fixed_anchors)\n'
    '    return controller, anchors, prior\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '    # Fixed selection does not need semantic warmup, but we intentionally use\n'
    '    # the same dense warmup budget as Rhyme/CoFrame so the comparison isolates\n'
    '    # the selector and online refresh rather than early-step compute.\n'
    '    if config.method == "dense" or num_inference_steps <= 1:\n'
    '        effective_warmup = 0\n'
    '    else:\n',
    '    # FIS is prompt-agnostic and can start sparsity at the first denoising\n'
    '    # step.  Rhyme/CoFrame keep their semantic warmup.\n'
    '    if config.method == "fis":\n'
    '        controller, initial_anchors, prior_scores = _make_controller(config=config, clean_proxy=latents)\n'
    '        trace.add("mesh_initialization", step=-1, timestep=None, anchors=initial_anchors, prior_scores=prior_scores)\n'
    '    if config.method in {"dense", "fis"} or num_inference_steps <= 1:\n'
    '        effective_warmup = 0\n'
    '    else:\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '            use_sparse = config.method != "dense" and controller is not None and step_index >= effective_warmup\n',
    '            use_sparse = (\n'
    '                config.method != "dense"\n'
    '                and controller is not None\n'
    '                and step_index >= effective_warmup\n'
    '                and config.is_sparse_step(step_index, num_inference_steps)\n'
    '            )\n',
)


# ---------------------------------------------------------------------------
# sparse_forward.py: FIS schedule, signal controls, matched-input baselines.
# ---------------------------------------------------------------------------
replace_once(
    "coframe/wan/sparse_forward.py",
    'from ..selection import uniform_select\n',
    'from ..selection import fis_interleaved_select, uniform_select\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    projection: torch.Tensor | None,\n) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:\n',
    '    projection: torch.Tensor | None,\n'
    '    interpolation_target: str | None = None,\n'
    ') -> tuple[torch.Tensor, dict[int, torch.Tensor]]:\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        target=config.interpolation_target,\n',
    '        target=interpolation_target or config.interpolation_target,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    'def _propagation_diagnostics(\n',
    '''def _shuffle_defects(
    defects: dict[int, torch.Tensor],
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    if len(defects) <= 1:
        return dict(defects)
    keys = sorted(defects)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(keys), generator=generator).tolist()
    values = [defects[keys[index]] for index in order]
    return {key: value for key, value in zip(keys, values)}


def _propagation_diagnostics(
''',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    rhyme_anchors = list(controller.initial_anchors)\n',
    '    rhyme_anchors = list(getattr(controller, "rhyme_reference_anchors", controller.initial_anchors))\n'
    '    fis_anchors = fis_interleaved_select(\n'
    '        geometry.num_frames,\n'
    '        config.num_anchors,\n'
    '        block_index,\n'
    '        config.sparse_block_start,\n'
    '        force_boundaries=config.force_boundaries,\n'
    '        anchor_stride=config.fis_anchor_stride,\n'
    '    )\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "fixed": fixed_anchors,\n        "oracle": oracle.anchors,\n',
    '        "fixed": fixed_anchors,\n'
    '        "fis": fis_anchors,\n'
    '        "oracle": oracle.anchors,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    swap_diagnostics = {\n',
    '    shuffled_defects = _shuffle_defects(\n'
    '        defects,\n'
    '        seed=config.shuffle_defect_seed + step_index * 1009 + block_index * 9176,\n'
    '    )\n'
    '    shuffled_post_risk = _post_observation_risk(controller, shuffled_defects, anchors)\n'
    '    gap_only_risk = torch.ones(controller.num_frames, dtype=torch.float32)\n\n'
    '    swap_diagnostics = {\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "post_observation": one_swap_diagnostics(\n'
    '            anchors=anchors,\n'
    '            interval_costs=interval_costs,\n'
    '            predicted_risk=post_risk,\n'
    '            gap_power=controller.gap_power,\n'
    '            move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain,\n'
    '            min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        ),\n'
    '    }\n',
    '        "post_observation": one_swap_diagnostics(\n'
    '            anchors=anchors,\n'
    '            interval_costs=interval_costs,\n'
    '            predicted_risk=post_risk,\n'
    '            gap_power=controller.gap_power,\n'
    '            move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain,\n'
    '            min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        ),\n'
    '        "gap_only": one_swap_diagnostics(\n'
    '            anchors=anchors,\n'
    '            interval_costs=interval_costs,\n'
    '            predicted_risk=gap_only_risk,\n'
    '            gap_power=controller.gap_power,\n'
    '            move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain,\n'
    '            min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        ),\n'
    '        "shuffled_defect": one_swap_diagnostics(\n'
    '            anchors=anchors,\n'
    '            interval_costs=interval_costs,\n'
    '            predicted_risk=shuffled_post_risk,\n'
    '            gap_power=controller.gap_power,\n'
    '            move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain,\n'
    '            min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        ),\n'
    '    }\n',
)
# Extend _probe_entry arguments so counterfactual operators reuse exactly the same block input.
replace_once(
    "coframe/wan/sparse_forward.py",
    '    propagation: dict[str, Any],\n) -> dict[str, Any]:\n',
    '    propagation: dict[str, Any],\n'
    '    block: Any,\n'
    '    encoder_hidden_states: torch.Tensor,\n'
    '    timestep_projection: torch.Tensor,\n'
    '    rotary_emb: torch.Tensor,\n'
    '    projection: torch.Tensor | None,\n'
    ') -> dict[str, Any]:\n',
)
# Insert matched-input operator baselines immediately before anchor-context diagnostics.
replace_once(
    "coframe/wan/sparse_forward.py",
    '    anchor_index = torch.tensor(anchors, device=dense_delta.device, dtype=torch.long)\n',
    '''    counterfactual_operator: dict[str, Any] = {}
    reference_meshes = {"rhyme": rhyme_anchors, "fixed": fixed_anchors, "fis": fis_anchors}
    for name in config.probe_counterfactual_methods:
        mesh = list(reference_meshes[name])
        if mesh == anchors:
            candidate_output = sparse_output
            candidate_propagation = propagation
        else:
            candidate_output, _ = _sparse_block_forward(
                block,
                block_input,
                encoder_hidden_states,
                timestep_projection,
                rotary_emb,
                anchors=mesh,
                geometry=geometry,
                config=config,
                compute_defects=False,
                projection=projection,
                interpolation_target="state" if name == "fis" else config.interpolation_target,
            )
            candidate_propagation = _propagation_diagnostics(
                blocks=[block],
                block_index=0,
                dense_output=dense_output,
                sparse_output=candidate_output,
                encoder_hidden_states=encoder_hidden_states,
                timestep_projection=timestep_projection,
                rotary_emb=rotary_emb,
                horizons=(),
                geometry=geometry,
                chunk_size=config.oracle_metric_chunk_size,
            )
        candidate_frames = tokens_to_frames(candidate_output, geometry.num_frames, geometry.tokens_per_frame)
        candidate_delta = candidate_frames - input_frames
        candidate_realized = reconstruction_metrics(
            dense_delta,
            candidate_delta,
            anchors=mesh,
            chunk_size=config.oracle_metric_chunk_size,
        )
        counterfactual_operator[name] = {
            "anchors": mesh,
            "realized_block_delta": candidate_realized,
            "propagation": candidate_propagation,
        }

    anchor_index = torch.tensor(anchors, device=dense_delta.device, dtype=torch.long)
''',
)
# The temporary propagation above only records local operator; real +1/+3 counterfactuals need transformer.blocks.
# We replace it later at the call-site by passing the full block list into _probe_entry.
replace_once(
    "coframe/wan/sparse_forward.py",
    '    block: Any,\n    encoder_hidden_states: torch.Tensor,\n',
    '    block: Any,\n    blocks: Any,\n    encoder_hidden_states: torch.Tensor,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '                blocks=[block],\n                block_index=0,\n',
    '                blocks=blocks,\n                block_index=block_index,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '                horizons=(),\n',
    '                horizons=tuple(int(value) for value in config.oracle_probe_horizons),\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "swap_decision": swap_diagnostics,\n        "propagation": propagation,\n',
    '        "swap_decision": swap_diagnostics,\n'
    '        "counterfactual_operator": counterfactual_operator,\n'
    '        "propagation": propagation,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "mesh_fixed_relative_l2": float(mesh_metrics["fixed"]["relative_l2"]),\n',
    '        "mesh_fixed_relative_l2": float(mesh_metrics["fixed"]["relative_l2"]),\n'
    '        "mesh_fis_relative_l2": float(mesh_metrics["fis"]["relative_l2"]),\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "mesh_fixed_nmse": float(mesh_metrics["fixed"]["normalized_mse"]),\n',
    '        "mesh_fixed_nmse": float(mesh_metrics["fixed"]["normalized_mse"]),\n'
    '        "mesh_fis_nmse": float(mesh_metrics["fis"]["normalized_mse"]),\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "swap_post_observation_top1_exact": swap_diagnostics["post_observation"]["top1_exact"],\n',
    '        "swap_post_observation_top1_exact": swap_diagnostics["post_observation"]["top1_exact"],\n'
    '        "swap_gap_only_gain_recovery": swap_diagnostics["gap_only"]["gain_recovery"],\n'
    '        "swap_gap_only_regret": swap_diagnostics["gap_only"]["regret"],\n'
    '        "swap_shuffled_gain_recovery": swap_diagnostics["shuffled_defect"]["gain_recovery"],\n'
    '        "swap_shuffled_regret": swap_diagnostics["shuffled_defect"]["regret"],\n',
)
# Add flat matched-input improvements before propagation aliases.
replace_once(
    "coframe/wan/sparse_forward.py",
    '    for horizon, metrics in propagation.items():\n',
    '''    for name, payload in counterfactual_operator.items():
        baseline_nmse = float(payload["realized_block_delta"]["normalized_mse"])
        result[f"counterfactual_{name}_block_delta_nmse"] = baseline_nmse
        result[f"operator_nmse_relative_improvement_over_{name}"] = (
            (baseline_nmse - float(realized_delta["normalized_mse"])) / (baseline_nmse + 1.0e-12)
        )
        for horizon, metrics in payload["propagation"].items():
            result[f"counterfactual_{name}_propagated_relative_l2_h{horizon}"] = metrics["relative_l2"]
            current_metrics = propagation.get(horizon)
            if current_metrics is not None:
                baseline_error = float(metrics["relative_l2"])
                result[f"propagation_relative_improvement_over_{name}_h{horizon}"] = (
                    (baseline_error - float(current_metrics["relative_l2"])) / (baseline_error + 1.0e-12)
                )

    for horizon, metrics in propagation.items():
''',
)
# FIS per-block anchors and controlled defect collection.
replace_once(
    "coframe/wan/sparse_forward.py",
    '        else:\n            anchors = list(controller.anchors)\n        metadata.block_anchors[block_index] = anchors\n',
    '        elif config.method == "fis":\n'
    '            anchors = fis_interleaved_select(\n'
    '                geometry.num_frames,\n'
    '                config.num_anchors,\n'
    '                block_index,\n'
    '                config.sparse_block_start,\n'
    '                force_boundaries=config.force_boundaries,\n'
    '                anchor_stride=config.fis_anchor_stride,\n'
    '            )\n'
    '        else:\n'
    '            anchors = list(controller.anchors)\n'
    '        metadata.block_anchors[block_index] = anchors\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        compute_defects = config.method == "coframe" and replay_block_anchors is None and update_controller\n',
    '        compute_defects = (\n'
    '            config.method == "coframe"\n'
    '            and replay_block_anchors is None\n'
    '            and (\n'
    '                (update_controller and config.refresh_signal in {"defect", "shuffled"})\n'
    '                or dense_output is not None\n'
    '            )\n'
    '        )\n',
)
# FIS uses state interpolation for its own execution path.
replace_once(
    "coframe/wan/sparse_forward.py",
    '            projection=projection,\n        )\n\n        if defects:\n',
    '            projection=projection,\n'
    '            interpolation_target="state" if config.method == "fis" else config.interpolation_target,\n'
    '        )\n\n'
    '        if defects:\n',
)
# Pass full context into probes.
replace_once(
    "coframe/wan/sparse_forward.py",
    '                    defects=defects,\n                    propagation=propagation,\n                )\n',
    '                    defects=defects,\n'
    '                    propagation=propagation,\n'
    '                    block=block,\n'
    '                    blocks=transformer.blocks,\n'
    '                    encoder_hidden_states=encoder_hidden_states,\n'
    '                    timestep_projection=timestep_proj,\n'
    '                    rotary_emb=rotary_emb,\n'
    '                    projection=projection,\n'
    '                )\n',
)
# Replace group refresh logic with signal-controlled variants.
replace_once(
    "coframe/wan/sparse_forward.py",
    '''        if is_group_boundary and replay_block_anchors is None:
            sparse_group_index += 1
            if compute_defects and group_defects:
                aggregated = {
                    frame: sum(values) / len(values)
                    for frame, values in group_defects.items()
                    if values
                }
                controller.observe(aggregated, anchors=anchors)
                if sparse_group_index % config.refresh_every_groups == 0:
                    refreshes = controller.refresh()
                    for refresh in refreshes:
                        entry = {
                            "step": step_index,
                            "after_block": block_index,
                            "group": sparse_group_index,
                            **refresh.to_dict(),
                        }
                        metadata.refresh_events.append(entry)
                        if trace is not None:
                            trace.add("mesh_refresh", **entry)
                group_defects = {}
''',
    '''        if is_group_boundary and replay_block_anchors is None:
            sparse_group_index += 1
            if config.method == "coframe" and update_controller:
                aggregated = {
                    frame: sum(values) / len(values)
                    for frame, values in group_defects.items()
                    if values
                }
                if config.refresh_signal == "defect" and aggregated:
                    controller.observe(aggregated, anchors=anchors)
                elif config.refresh_signal == "shuffled" and aggregated:
                    shuffled = _shuffle_defects(
                        {frame: torch.tensor(value) for frame, value in aggregated.items()},
                        seed=config.shuffle_defect_seed + step_index * 1009 + sparse_group_index * 9176,
                    )
                    controller.observe(shuffled, anchors=anchors)
                # gap_only intentionally leaves a uniform risk field; none freezes the mesh.
                if (
                    config.refresh_signal != "none"
                    and sparse_group_index % config.refresh_every_groups == 0
                ):
                    refreshes = controller.refresh()
                    for refresh in refreshes:
                        entry = {
                            "step": step_index,
                            "after_block": block_index,
                            "group": sparse_group_index,
                            "refresh_signal": config.refresh_signal,
                            **refresh.to_dict(),
                        }
                        metadata.refresh_events.append(entry)
                        if trace is not None:
                            trace.add("mesh_refresh", **entry)
            group_defects = {}
''',
)


# ---------------------------------------------------------------------------
# cli.py: expose new controls.
# ---------------------------------------------------------------------------
replace_once(
    "coframe/cli.py",
    'def _csv_ints(value: str) -> tuple[int, ...]:\n',
    'def _csv_strings(value: str) -> tuple[str, ...]:\n'
    '    return tuple(item.strip() for item in value.split(",") if item.strip())\n\n\n'
    'def _csv_ints(value: str) -> tuple[int, ...]:\n',
)
replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--method", choices=["dense", "fixed", "rhyme", "coframe"], default="coframe")\n',
    '    parser.add_argument("--method", choices=["dense", "fixed", "fis", "rhyme", "coframe"], default="coframe")\n',
)
replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--defect-target", choices=["delta", "state"], default="delta")\n',
    '    parser.add_argument("--defect-target", choices=["delta", "state"], default="delta")\n'
    '    parser.add_argument("--refresh-signal", choices=["defect", "none", "gap_only", "shuffled"], default="defect")\n'
    '    parser.add_argument("--shuffle-defect-seed", type=int, default=20260810)\n'
    '    parser.add_argument("--fis-anchor-stride", type=int, default=0)\n'
    '    parser.add_argument("--fis-dense-tail-steps", type=int, default=0)\n',
)
replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--oracle-metric-chunk-size", type=int, default=65_536)\n',
    '    parser.add_argument("--oracle-metric-chunk-size", type=int, default=65_536)\n'
    '    parser.add_argument("--probe-counterfactual-methods", type=_csv_strings, default=("rhyme", "fis", "fixed"))\n',
)
replace_once(
    "coframe/cli.py",
    '        defect_target=args.defect_target,\n',
    '        defect_target=args.defect_target,\n'
    '        refresh_signal=args.refresh_signal,\n'
    '        shuffle_defect_seed=args.shuffle_defect_seed,\n'
    '        fis_anchor_stride=args.fis_anchor_stride,\n'
    '        fis_dense_tail_steps=args.fis_dense_tail_steps,\n',
)
replace_once(
    "coframe/cli.py",
    '        oracle_metric_chunk_size=args.oracle_metric_chunk_size,\n',
    '        oracle_metric_chunk_size=args.oracle_metric_chunk_size,\n'
    '        probe_counterfactual_methods=args.probe_counterfactual_methods,\n',
)


# ---------------------------------------------------------------------------
# New Stage-1b/Stage-2 scripts and summarizer.
# ---------------------------------------------------------------------------
(ROOT / "scripts/run_stage1b_wan21.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-A gymnast performs a fast cartwheel while a yellow ball rolls behind her.}"
OUTPUT_ROOT="${2:-outputs/stage1b}"
SEED="${3:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --prompt "$PROMPT"
  --seed "$SEED"
  --height 480 --width 832 --num-frames 81 --steps 50
  --guidance-scale 5.0 --flow-shift 3.0
  --warmup-steps 5 --num-anchors 9
  --sparse-block-start 3 --sparse-block-end 27 --block-group-size 3
  --kv-mode full_kv
  --interpolation-target delta --defect-target delta
  --oracle-probe-steps 5,20,40
  --oracle-probe-blocks 8,14,20
  --oracle-probe-horizons 1,3
  --probe-counterfactual-methods rhyme,fis,fixed
  --output-dir "$OUTPUT_ROOT"
)

# These four runs attribute improvement to the refresh signal itself.  Every
# oracle probe also evaluates matched-input Rhyme/FIS/fixed operators.
for SIGNAL in none gap_only shuffled defect; do
  python scripts/run_wan21_1_3b.py \
    --method coframe \
    --refresh-signal "$SIGNAL" \
    --run-name "coframe_${SIGNAL}_full_kv_seed${SEED}" \
    "${COMMON[@]}"
done

python scripts/summarize_stage1b.py --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/stage1b_summary.json"
''', encoding="utf-8")
(ROOT / "scripts/run_stage1b_wan21.sh").chmod(0o755)

(ROOT / "scripts/run_baselines_wan21.sh").write_text(r'''#!/usr/bin/env bash
set -euo pipefail
PROMPT="${1:-A red toy car turns sharply around a blue cube on a wooden table.}"
OUTPUT_ROOT="${2:-outputs/baselines}"
SEED="${3:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BASE=(--prompt "$PROMPT" --seed "$SEED" --height 480 --width 832 --num-frames 81 --steps 50 \
      --guidance-scale 5.0 --flow-shift 3.0 --num-anchors 9 --sparse-block-start 3 --sparse-block-end 27 \
      --block-group-size 3 --kv-mode anchor_only --output-dir "$OUTPUT_ROOT")

python scripts/run_wan21_1_3b.py --method dense --run-name "dense_seed${SEED}" "${BASE[@]}"
python scripts/run_wan21_1_3b.py --method fixed --warmup-steps 5 --interpolation-target delta --run-name "fixed_seed${SEED}" "${BASE[@]}"
python scripts/run_wan21_1_3b.py --method rhyme --warmup-steps 5 --interpolation-target delta --run-name "rhyme_selector_seed${SEED}" "${BASE[@]}"
# FIS uses block-interleaved anchors and state interpolation.  Dense tail is explicit; set FIS_DENSE_TAIL_STEPS if desired.
python scripts/run_wan21_1_3b.py --method fis --fis-dense-tail-steps "${FIS_DENSE_TAIL_STEPS:-0}" --interpolation-target state --run-name "fis_seed${SEED}" "${BASE[@]}"
python scripts/run_wan21_1_3b.py --method coframe --refresh-signal defect --warmup-steps 5 --interpolation-target delta --run-name "coframe_seed${SEED}" "${BASE[@]}"
''', encoding="utf-8")
(ROOT / "scripts/run_baselines_wan21.sh").chmod(0o755)

(ROOT / "scripts/summarize_stage1b.py").write_text(r'''from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def probes_from_trace(path: Path):
    payload = json.loads(path.read_text())
    signal = payload.get("run", {}).get("config", {}).get("refresh_signal", "unknown")
    for event in payload.get("events", []):
        if event.get("event") != "transformer_forward":
            continue
        for probe in event.get("probes", []):
            yield signal, probe


def safe_median(values):
    values = [float(v) for v in values if v is not None]
    return median(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    buckets = defaultdict(lambda: defaultdict(list))
    for trace in args.root.rglob("trace.json"):
        for signal, probe in probes_from_trace(trace):
            current = probe.get("mesh_current_nmse")
            for baseline in ("rhyme", "fis", "fixed"):
                base = probe.get(f"mesh_{baseline}_nmse")
                if current is not None and base not in (None, 0):
                    buckets[signal][f"mesh_improvement_over_{baseline}"].append((float(base) - float(current)) / float(base))
                op = probe.get(f"operator_nmse_relative_improvement_over_{baseline}")
                if op is not None:
                    buckets[signal][f"operator_improvement_over_{baseline}"].append(op)
                for horizon in (1, 3):
                    prop = probe.get(f"propagation_relative_improvement_over_{baseline}_h{horizon}")
                    if prop is not None:
                        buckets[signal][f"prop_h{horizon}_improvement_over_{baseline}"].append(prop)
            for policy, field in (
                ("defect", "swap_post_observation_gain_recovery"),
                ("gap_only", "swap_gap_only_gain_recovery"),
                ("shuffled", "swap_shuffled_gain_recovery"),
            ):
                value = probe.get(field)
                if value is not None:
                    buckets[signal][f"local_swap_gain_recovery_{policy}"].append(value)

    result = {}
    for signal, metrics in sorted(buckets.items()):
        result[signal] = {name: {"count": len(values), "median": safe_median(values), "win_rate": sum(float(v) > 0 for v in values) / len(values) if values else None} for name, values in sorted(metrics.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
''', encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
(ROOT / "tests/test_stage1b_policies.py").write_text(r'''import torch

from coframe.config import CoFrameConfig
from coframe.selection import fis_interleaved_select


def test_fis_interleaved_is_exact_budget_and_rotates():
    meshes = [fis_interleaved_select(21, 9, block, 3) for block in (3, 4, 5)]
    assert all(len(mesh) == 9 for mesh in meshes)
    assert all(mesh[0] == 0 and mesh[-1] == 20 for mesh in meshes)
    assert len({tuple(mesh) for mesh in meshes}) == 3
    covered = set().union(*map(set, meshes))
    assert set(range(21)).issubset(covered)


def test_refresh_signal_validation_and_fis_step_gate():
    config = CoFrameConfig(method="fis", fis_dense_tail_steps=2)
    config.validate(num_blocks=30, num_frames=21)
    assert config.is_sparse_step(47, 50)
    assert not config.is_sparse_step(48, 50)
    for signal in ("defect", "none", "gap_only", "shuffled"):
        CoFrameConfig(refresh_signal=signal).validate(num_blocks=30, num_frames=21)
''', encoding="utf-8")


# ---------------------------------------------------------------------------
# Documentation.
# ---------------------------------------------------------------------------
append_once(
    "docs/EXPERIMENT_PLAN.md",
    "## Stage B2 — source attribution after the 4xL40 pilot",
    r'''
## Stage B2 — source attribution after the 4xL40 pilot

The first 4xL40 Stage-1 release showed a strong adaptive-mesh signal but an under-calibrated one-swap controller.  Do **not** weaken RhymeFlow or FIS-DiT: both remain strong paper baselines.  The next experiment asks where CoFrame's gain comes from.

Use the same eight prompts and seeds.  Run `scripts/run_stage1b_wan21.sh` in `full_kv` first.  It evaluates four trajectories with the same Rhyme initialization:

- `refresh_signal=none`: no online remeshing;
- `gap_only`: remesh from interval geometry with no semantic/defect evidence;
- `shuffled`: preserve defect magnitudes while destroying frame alignment;
- `defect`: the real CoFrame signal.

Every probe additionally executes **matched-input** Rhyme-selector, fixed, and FIS-style sparse counterfactuals from the exact same block input and propagates all states through the same +1/+3 dense blocks.  This closes the missing comparison in Stage 1.

FIS uses the published interleaved residue rule `r_l=(l-l0) mod n`, boundary anchors, and state interpolation; an exact-K fill/trim is applied only so mechanism comparisons use the same exact-frame budget.  `fis_dense_tail_steps` is explicit rather than hidden.  For final paper tables, also reproduce the official FIS-DiT and full RhymeFlow systems with their authors' recommended settings; the in-repo `rhyme` method is a selector-controlled baseline under CoFrame's sparse operator, not a claim of reproducing the entire asynchronous RhymeFlow scheduler.

Primary Stage-B2 decisions:

1. true defect must beat `gap_only` and `shuffled` on matched-input mesh NMSE and harmful-swap rate;
2. CoFrame must beat Rhyme and FIS on realized block-delta NMSE from the same input;
3. the advantage should retain the same sign after +1/+3 dense propagation;
4. only then run `scripts/run_baselines_wan21.sh` for endpoint fidelity and warmed latency.
''',
)
append_once(
    "README.md",
    "## Stage-1b source-attribution experiment",
    r'''
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
''',
)

print("Stage-1b patch applied")

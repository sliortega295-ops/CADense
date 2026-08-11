from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern missing in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "coframe/config.py",
    '    probe_counterfactual_methods: Sequence[str] = field(default_factory=lambda: ("rhyme", "fis", "fixed"))\n\n    trace_path: str | None = None\n',
    '    probe_counterfactual_methods: Sequence[str] = field(default_factory=lambda: ("rhyme", "fis", "fixed"))\n\n'
    '    # Stage-1c signal screening. These diagnostics never change the deployed\n'
    '    # mesh; they only rank hypothetical one-swap actions against dense truth.\n'
    '    probe_curvature_signals: bool = False\n'
    '    curvature_shuffle_seed: int = 20260811\n\n'
    '    trace_path: str | None = None\n',
)

replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--probe-counterfactual-methods", type=_csv_strings, default=("rhyme", "fis", "fixed"))\n\n',
    '    parser.add_argument("--probe-counterfactual-methods", type=_csv_strings, default=("rhyme", "fis", "fixed"))\n'
    '    parser.add_argument("--probe-curvature-signals", action="store_true")\n'
    '    parser.add_argument("--curvature-shuffle-seed", type=int, default=20260811)\n\n',
)
replace_once(
    "coframe/cli.py",
    '        probe_counterfactual_methods=args.probe_counterfactual_methods,\n        trace_path=str(trace_path),\n',
    '        probe_counterfactual_methods=args.probe_counterfactual_methods,\n'
    '        probe_curvature_signals=args.probe_curvature_signals,\n'
    '        curvature_shuffle_seed=args.curvature_shuffle_seed,\n'
    '        trace_path=str(trace_path),\n',
)

helper_anchor = '''def _propagation_diagnostics(\n'''
helper_code = '''def _normalize_frame_signal(scores: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:\n    values = scores.detach().float().cpu().clamp_min(0.0)\n    if values.numel() <= 2:\n        return values\n    interior = values[1:-1]\n    scale = float(interior.mean().item())\n    if scale > eps:\n        values = (values / scale).clamp_max(10.0)\n    values[0] = 0.0\n    values[-1] = 0.0\n    return values\n\n\ndef _temporal_curvature_scores(\n    frame_values: torch.Tensor,\n    projection: torch.Tensor | None,\n    eps: float = 1.0e-8,\n) -> torch.Tensor:\n    \"\"\"Cheap frame-wise curvature from an already-available [B,F,P,D] state.\n\n    With the default random channel sketch this adds only a small projection and\n    elementwise temporal residual; it does not execute another DiT block.\n    \"\"\"\n    if frame_values.ndim != 4:\n        raise ValueError(\"frame_values must be [B,F,P,D]\")\n    frame_count = int(frame_values.shape[1])\n    scores = torch.zeros(frame_count, dtype=torch.float32)\n    if frame_count < 3:\n        return scores\n    if projection is not None:\n        reduced = torch.matmul(\n            frame_values,\n            projection.to(device=frame_values.device, dtype=frame_values.dtype),\n        ).float()\n    else:\n        # Deterministic channel subsampling keeps exact-diagnostic mode bounded.\n        stride = max(1, int(frame_values.shape[-1]) // 64)\n        reduced = frame_values[..., ::stride].float()\n    predicted = 0.5 * (reduced[:, :-2] + reduced[:, 2:])\n    center = reduced[:, 1:-1]\n    residual = (center - predicted).square().mean(dim=(0, 2, 3)).sqrt()\n    magnitude = center.square().mean(dim=(0, 2, 3)).sqrt()\n    scores[1:-1] = (residual / (magnitude + eps)).detach().cpu()\n    return scores\n\n\ndef _shuffle_frame_signal(scores: torch.Tensor, *, seed: int) -> torch.Tensor:\n    values = scores.detach().float().cpu().clone()\n    if values.numel() <= 3:\n        return values\n    generator = torch.Generator(device=\"cpu\").manual_seed(int(seed))\n    interior = values[1:-1].clone()\n    order = torch.randperm(interior.numel(), generator=generator)\n    values[1:-1] = interior.index_select(0, order)\n    return values\n\n\n'''
replace_once("coframe/wan/sparse_forward.py", helper_anchor, helper_code + helper_anchor)

replace_once(
    "coframe/wan/sparse_forward.py",
    '    projection: torch.Tensor | None,\n) -> dict[str, Any]:\n',
    '    projection: torch.Tensor | None,\n    previous_delta_curvature: torch.Tensor | None,\n) -> dict[str, Any]:\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '    gap_only_risk = torch.ones(controller.num_frames, dtype=torch.float32)\n\n    swap_diagnostics = {\n',
    '    gap_only_risk = torch.ones(controller.num_frames, dtype=torch.float32)\n'
    '    input_curvature = None\n'
    '    shuffled_input_curvature = None\n'
    '    normalized_previous_delta_curvature = None\n'
    '    shuffled_previous_delta_curvature = None\n'
    '    if config.probe_curvature_signals:\n'
    '        input_curvature = _normalize_frame_signal(_temporal_curvature_scores(input_frames, projection))\n'
    '        shuffled_input_curvature = _shuffle_frame_signal(\n'
    '            input_curvature,\n'
    '            seed=config.curvature_shuffle_seed + step_index * 1009 + block_index * 9176,\n'
    '        )\n'
    '        if previous_delta_curvature is not None:\n'
    '            normalized_previous_delta_curvature = _normalize_frame_signal(previous_delta_curvature)\n'
    '            shuffled_previous_delta_curvature = _shuffle_frame_signal(\n'
    '                normalized_previous_delta_curvature,\n'
    '                seed=config.curvature_shuffle_seed + 17 + step_index * 1009 + block_index * 9176,\n'
    '            )\n\n'
    '    swap_diagnostics = {\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '    }\n\n    counterfactual_operator: dict[str, Any] = {}\n',
    '    }\n'
    '    if input_curvature is not None:\n'
    '        swap_diagnostics["input_curvature"] = one_swap_diagnostics(\n'
    '            anchors=anchors, interval_costs=interval_costs, predicted_risk=input_curvature,\n'
    '            gap_power=controller.gap_power, move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        )\n'
    '        swap_diagnostics["shuffled_input_curvature"] = one_swap_diagnostics(\n'
    '            anchors=anchors, interval_costs=interval_costs, predicted_risk=shuffled_input_curvature,\n'
    '            gap_power=controller.gap_power, move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        )\n'
    '    if normalized_previous_delta_curvature is not None:\n'
    '        swap_diagnostics["previous_delta_curvature"] = one_swap_diagnostics(\n'
    '            anchors=anchors, interval_costs=interval_costs, predicted_risk=normalized_previous_delta_curvature,\n'
    '            gap_power=controller.gap_power, move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        )\n'
    '        swap_diagnostics["shuffled_previous_delta_curvature"] = one_swap_diagnostics(\n'
    '            anchors=anchors, interval_costs=interval_costs, predicted_risk=shuffled_previous_delta_curvature,\n'
    '            gap_power=controller.gap_power, move_penalty=controller.move_penalty,\n'
    '            min_gain=controller.min_refresh_gain, min_gap=controller.min_gap,\n'
    '            force_boundaries=controller.force_boundaries,\n'
    '        )\n\n'
    '    counterfactual_operator: dict[str, Any] = {}\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '        "swap_decision": swap_diagnostics,\n        "counterfactual_operator": counterfactual_operator,\n',
    '        "swap_decision": swap_diagnostics,\n'
    '        "input_curvature_scores": None if input_curvature is None else input_curvature.tolist(),\n'
    '        "previous_delta_curvature_scores": (\n'
    '            None if normalized_previous_delta_curvature is None else normalized_previous_delta_curvature.tolist()\n'
    '        ),\n'
    '        "counterfactual_operator": counterfactual_operator,\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '    group_defects: dict[int, list[float]] = {}\n    sparse_group_index = 0\n\n    for block_index, block in enumerate(transformer.blocks):\n',
    '    group_defects: dict[int, list[float]] = {}\n'
    '    sparse_group_index = 0\n'
    '    previous_delta_curvature: torch.Tensor | None = None\n\n'
    '    for block_index, block in enumerate(transformer.blocks):\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '        block_input = hidden_states\n        dense_output = None\n',
    '        block_input = hidden_states\n'
    '        curvature_from_previous_block = previous_delta_curvature\n'
    '        dense_output = None\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '            interpolation_target="state" if config.method == "fis" else config.interpolation_target,\n        )\n\n        if defects:\n',
    '            interpolation_target="state" if config.method == "fis" else config.interpolation_target,\n'
    '        )\n'
    '        if config.probe_curvature_signals:\n'
    '            if config.should_probe(step_index, block_index + 1):\n'
    '                before_frames = tokens_to_frames(block_input, geometry.num_frames, geometry.tokens_per_frame)\n'
    '                after_frames = tokens_to_frames(hidden_states, geometry.num_frames, geometry.tokens_per_frame)\n'
    '                previous_delta_curvature = _temporal_curvature_scores(after_frames - before_frames, projection)\n'
    '            else:\n'
    '                previous_delta_curvature = None\n\n'
    '        if defects:\n',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '                    rotary_emb=rotary_emb,\n                    projection=projection,\n                )\n',
    '                    rotary_emb=rotary_emb,\n'
    '                    projection=projection,\n'
    '                    previous_delta_curvature=curvature_from_previous_block,\n'
    '                )\n',
)

# Add focused CPU tests for the new signal, keeping tests independent of diffusers.
test_path = Path("tests/test_stage1c_signals.py")
test_path.write_text('''import torch\n\nfrom coframe.wan.sparse_forward import (\n    _normalize_frame_signal,\n    _shuffle_frame_signal,\n    _temporal_curvature_scores,\n)\n\n\ndef test_temporal_curvature_is_zero_for_linear_trajectory():\n    base = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)\n    scores = _temporal_curvature_scores(base, projection=None)\n    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-6)\n\n\ndef test_temporal_curvature_detects_local_kink_and_shuffle_preserves_values():\n    values = torch.arange(7, dtype=torch.float32)\n    values[3] += 4.0\n    frames = values.view(1, 7, 1, 1)\n    scores = _normalize_frame_signal(_temporal_curvature_scores(frames, projection=None))\n    assert int(torch.argmax(scores).item()) == 3\n    shuffled = _shuffle_frame_signal(scores, seed=123)\n    assert torch.allclose(torch.sort(scores[1:-1]).values, torch.sort(shuffled[1:-1]).values)\n    assert shuffled[0] == 0 and shuffled[-1] == 0\n''', encoding="utf-8")

# Append the staged plan without changing the earlier preregistered records.
doc = Path("docs/EXPERIMENT_PLAN.md")
text = doc.read_text(encoding="utf-8")
addition = '''\n\n## Stage C0 / Stage-1c — mechanism pivot after Stage-1b\n\nStage-1b falsified the current leave-one-out defect **localization** mechanism: true and shuffled defects produced nearly the same actions. Do not tune that signal further. First reuse the existing Stage-1b traces with no GPU:\n\n```bash\npython scripts/analyze_stage1c_offline.py \\\n  --root outputs/stage1b \\\n  --signal defect \\\n  --output outputs/stage1c/offline_analysis.json\n```\n\nThe analysis separates (a) prompt dependence of the exact DP oracle mesh from step/block dependence and (b) whether the global defect magnitude can still predict block-level operator or +3 propagation risk. The script emits triage flags, not paper claims.\n\nOnly if neither a calibrated step/block schedule nor block-risk gating is supported should GPU time be spent on per-frame signal discovery. The first bounded screen is input temporal curvature plus previous-block delta curvature. These are computed from states that already exist; the probe ranks hypothetical swaps but **does not deploy them**. Run static-Rhyme and gap-only trajectories:\n\n```bash\nbash scripts/run_stage1c_curvature_wan21.sh "<prompt>" outputs/stage1c_curvature/p0_s0 0\npython scripts/summarize_stage1c_curvature.py \\\n  --root outputs/stage1c_curvature \\\n  --output outputs/stage1c_curvature/summary.json\n```\n\nA curvature signal is worth implementing as a real controller only if it beats gap-only on actionable-cell gain recovery / regret and also beats its own shuffled control. A high same-action rate with shuffled curvature rejects the localization mechanism just as in Stage-1b.\n'''
if "## Stage C0 / Stage-1c" not in text:
    doc.write_text(text + addition, encoding="utf-8")

print("Stage-1c patch applied")

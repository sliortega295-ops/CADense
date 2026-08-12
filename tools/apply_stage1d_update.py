from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern missing in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------- config ----------------
replace_once(
    "coframe/config.py",
    'Method = Literal["dense", "fixed", "fis", "rhyme", "coframe"]\nRefreshSignal = Literal["defect", "none", "gap_only", "shuffled"]\n',
    'Method = Literal["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k"]\n'
    'RefreshSignal = Literal["defect", "none", "gap_only", "shuffled"]\n'
    'BudgetPolicy = Literal["none", "step_block", "mean_defect", "max_defect"]\n',
)
replace_once(
    "coframe/config.py",
    '    curvature_shuffle_seed: int = 20260811\n\n    trace_path: str | None = None\n',
    '    curvature_shuffle_seed: int = 20260811\n\n'
    '    # Stage-1d: causal exact-frame budget allocation. Frame placement is\n'
    '    # deliberately uniform so this experiment isolates "how much to compute"\n'
    '    # from the rejected defect-localization/remeshing mechanism.\n'
    '    adaptive_k_policy: BudgetPolicy = "none"\n'
    '    adaptive_k_values: Sequence[int] = field(default_factory=lambda: (6, 9, 12, 21))\n'
    '    adaptive_k_thresholds: Sequence[float] = field(default_factory=tuple)\n'
    '    adaptive_k_schedule: dict[str, int] = field(default_factory=dict)\n'
    '    adaptive_k_carry_across_steps: bool = True\n\n'
    '    trace_path: str | None = None\n',
)
replace_once(
    "coframe/config.py",
    '        if self.method not in {"dense", "fixed", "fis", "rhyme", "coframe"}:\n',
    '        if self.method not in {"dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k"}:\n',
)
replace_once(
    "coframe/config.py",
    '        if self.refresh_signal not in {"defect", "none", "gap_only", "shuffled"}:\n            raise ValueError(f"Unsupported refresh_signal: {self.refresh_signal}")\n',
    '        if self.refresh_signal not in {"defect", "none", "gap_only", "shuffled"}:\n'
    '            raise ValueError(f"Unsupported refresh_signal: {self.refresh_signal}")\n'
    '        if self.adaptive_k_policy not in {"none", "step_block", "mean_defect", "max_defect"}:\n'
    '            raise ValueError(f"Unsupported adaptive_k_policy: {self.adaptive_k_policy}")\n'
    '        if self.method == "adaptive_k" and self.adaptive_k_policy == "none":\n'
    '            raise ValueError("method=adaptive_k requires an adaptive_k_policy")\n'
    '        budget_values = [int(value) for value in self.adaptive_k_values]\n'
    '        if not budget_values or budget_values != sorted(set(budget_values)):\n'
    '            raise ValueError("adaptive_k_values must be strictly increasing")\n'
    '        if any(value < 1 for value in budget_values):\n'
    '            raise ValueError("adaptive_k_values must be positive")\n'
    '        thresholds = [float(value) for value in self.adaptive_k_thresholds]\n'
    '        if thresholds != sorted(thresholds):\n'
    '            raise ValueError("adaptive_k_thresholds must be sorted")\n'
    '        if self.adaptive_k_policy in {"mean_defect", "max_defect"} and len(thresholds) != len(budget_values) - 1:\n'
    '            raise ValueError("adaptive defect policies require len(values)-1 thresholds")\n'
    '        if self.adaptive_k_policy == "step_block":\n'
    '            invalid_schedule = [value for value in self.adaptive_k_schedule.values() if int(value) not in budget_values]\n'
    '            if invalid_schedule:\n'
    '                raise ValueError("adaptive_k_schedule contains a budget outside adaptive_k_values")\n',
)
replace_once(
    "coframe/config.py",
    '        if num_frames is not None and self.num_anchors > num_frames:\n            raise ValueError(f"num_anchors={self.num_anchors} exceeds latent frames={num_frames}")\n',
    '        if num_frames is not None and self.num_anchors > num_frames:\n'
    '            raise ValueError(f"num_anchors={self.num_anchors} exceeds latent frames={num_frames}")\n'
    '        if num_frames is not None and any(int(value) > num_frames for value in self.adaptive_k_values):\n'
    '            raise ValueError("adaptive_k_values exceed the latent frame count")\n'
    '        if self.force_boundaries and any(int(value) < 2 for value in self.adaptive_k_values) and (num_frames is None or num_frames > 1):\n'
    '            raise ValueError("adaptive_k_values must be >=2 when force_boundaries=True")\n',
)
replace_once(
    "coframe/config.py",
    '        result["probe_counterfactual_methods"] = list(self.probe_counterfactual_methods)\n        return result\n',
    '        result["probe_counterfactual_methods"] = list(self.probe_counterfactual_methods)\n'
    '        result["adaptive_k_values"] = list(self.adaptive_k_values)\n'
    '        result["adaptive_k_thresholds"] = list(self.adaptive_k_thresholds)\n'
    '        result["adaptive_k_schedule"] = dict(self.adaptive_k_schedule)\n'
    '        return result\n',
)

# ---------------- CLI ----------------
replace_once(
    "coframe/cli.py",
    'def _csv_ints(value: str) -> tuple[int, ...]:\n    if not value.strip():\n        return ()\n    return tuple(int(item.strip()) for item in value.split(",") if item.strip())\n\n\n',
    'def _csv_ints(value: str) -> tuple[int, ...]:\n'
    '    if not value.strip():\n'
    '        return ()\n'
    '    return tuple(int(item.strip()) for item in value.split(",") if item.strip())\n\n\n'
    'def _csv_floats(value: str) -> tuple[float, ...]:\n'
    '    if not value.strip():\n'
    '        return ()\n'
    '    return tuple(float(item.strip()) for item in value.split(",") if item.strip())\n\n\n',
)
replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--method", choices=["dense", "fixed", "fis", "rhyme", "coframe"], default="coframe")\n',
    '    parser.add_argument("--method", choices=["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k"], default="coframe")\n',
)
replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--curvature-shuffle-seed", type=int, default=20260811)\n\n',
    '    parser.add_argument("--curvature-shuffle-seed", type=int, default=20260811)\n'
    '    parser.add_argument("--adaptive-k-policy", choices=["none", "step_block", "mean_defect", "max_defect"], default="none")\n'
    '    parser.add_argument("--adaptive-k-values", type=_csv_ints, default=(6, 9, 12, 21))\n'
    '    parser.add_argument("--adaptive-k-thresholds", type=_csv_floats, default=())\n'
    '    parser.add_argument("--adaptive-k-schedule-json", type=Path, default=None)\n'
    '    parser.add_argument("--no-adaptive-k-carry", action="store_true")\n\n',
)
replace_once(
    "coframe/cli.py",
    '    video_path = run_dir / "video.mp4"\n\n    config = CoFrameConfig(\n',
    '    video_path = run_dir / "video.mp4"\n\n'
    '    adaptive_k_schedule = {}\n'
    '    if args.adaptive_k_schedule_json is not None:\n'
    '        adaptive_k_schedule = json.loads(args.adaptive_k_schedule_json.read_text(encoding="utf-8"))\n'
    '        if not isinstance(adaptive_k_schedule, dict):\n'
    '            raise ValueError("adaptive-k schedule JSON must contain an object mapping step:group to K")\n\n'
    '    config = CoFrameConfig(\n',
)
replace_once(
    "coframe/cli.py",
    '        curvature_shuffle_seed=args.curvature_shuffle_seed,\n        trace_path=str(trace_path),\n',
    '        curvature_shuffle_seed=args.curvature_shuffle_seed,\n'
    '        adaptive_k_policy=args.adaptive_k_policy,\n'
    '        adaptive_k_values=args.adaptive_k_values,\n'
    '        adaptive_k_thresholds=args.adaptive_k_thresholds,\n'
    '        adaptive_k_schedule={str(key): int(value) for key, value in adaptive_k_schedule.items()},\n'
    '        adaptive_k_carry_across_steps=not args.no_adaptive_k_carry,\n'
    '        trace_path=str(trace_path),\n',
)

# ---------------- controller state ----------------
replace_once(
    "coframe/controller.py",
    '        self.refresh_history: list[MeshRefresh] = []\n',
    '        self.refresh_history: list[MeshRefresh] = []\n'
    '        # Stage-1d budget state is intentionally separate from self.anchors,\n'
    '        # whose fixed length still serves the original remeshing controller.\n'
    '        self.current_budget = int(num_anchors)\n'
    '        self.budget_history: list[dict[str, Any]] = []\n',
)
replace_once(
    "coframe/controller.py",
    '            "refresh_history": [event.to_dict() for event in self.refresh_history],\n',
    '            "refresh_history": [event.to_dict() for event in self.refresh_history],\n'
    '            "current_budget": self.current_budget,\n'
    '            "budget_history": list(self.budget_history),\n',
)

# ---------------- pipeline initialization ----------------
replace_once(
    "coframe/wan/pipeline.py",
    '    if config.method == "fixed":\n        anchors = fixed_anchors\n        controller_prior = torch.zeros_like(prior)\n        controller_prior_weight = 0.0\n',
    '    if config.method in {"fixed", "adaptive_k"}:\n'
    '        anchors = fixed_anchors\n'
    '        controller_prior = torch.zeros_like(prior)\n'
    '        controller_prior_weight = 0.0\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '    elif config.method in {"rhyme", "coframe"}:\n',
    '    elif config.method in {"rhyme", "coframe"}:\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '        "final_anchors": None if controller is None else list(controller.anchors),\n',
    '        "final_anchors": None if controller is None else list(controller.anchors),\n'
    '        "final_budget": None if controller is None else int(controller.current_budget),\n',
)

# ---------------- sparse forward ----------------
replace_once(
    "coframe/wan/sparse_forward.py",
    'from ..config import CoFrameConfig\nfrom ..controller import AdaptiveMeshController\n',
    'from ..budget import defect_stat, lookup_scheduled_budget, select_budget\n'
    'from ..config import CoFrameConfig\n'
    'from ..controller import AdaptiveMeshController\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    refresh_events: list[dict[str, Any]]\n    defects: list[dict[str, Any]]\n    probes: list[dict[str, Any]]\n',
    '    refresh_events: list[dict[str, Any]]\n'
    '    budget_events: list[dict[str, Any]]\n'
    '    defects: list[dict[str, Any]]\n'
    '    probes: list[dict[str, Any]]\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '            "refresh_events": self.refresh_events,\n            "defects": self.defects,\n',
    '            "refresh_events": self.refresh_events,\n'
    '            "budget_events": self.budget_events,\n'
    '            "defects": self.defects,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    fixed_anchors = uniform_select(\n        geometry.num_frames,\n        config.num_anchors,\n        config.force_boundaries,\n    )\n    rhyme_anchors = list(getattr(controller, "rhyme_reference_anchors", controller.initial_anchors))\n    fis_anchors = fis_interleaved_select(\n        geometry.num_frames,\n        config.num_anchors,\n',
    '    probe_budget = len(anchors) if config.method == "adaptive_k" else config.num_anchors\n'
    '    fixed_anchors = uniform_select(\n'
    '        geometry.num_frames,\n'
    '        probe_budget,\n'
    '        config.force_boundaries,\n'
    '    )\n'
    '    rhyme_anchors = (\n'
    '        list(fixed_anchors)\n'
    '        if config.method == "adaptive_k"\n'
    '        else list(getattr(controller, "rhyme_reference_anchors", controller.initial_anchors))\n'
    '    )\n'
    '    fis_anchors = fis_interleaved_select(\n'
    '        geometry.num_frames,\n'
    '        probe_budget,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        num_anchors=config.num_anchors,\n        total_energy=total_energy,\n',
    '        num_anchors=probe_budget,\n'
    '        total_energy=total_energy,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        "step": step_index,\n        "block": block_index,\n        "anchors": list(anchors),\n',
    '        "step": step_index,\n'
    '        "block": block_index,\n'
    '        "anchors": list(anchors),\n'
    '        "anchor_budget": probe_budget,\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        refresh_events=[],\n        defects=[],\n',
    '        refresh_events=[],\n'
    '        budget_events=[],\n'
    '        defects=[],\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    group_defects: dict[int, list[float]] = {}\n    sparse_group_index = 0\n    previous_delta_curvature: torch.Tensor | None = None\n\n    for block_index, block in enumerate(transformer.blocks):\n',
    '    group_defects: dict[int, list[float]] = {}\n'
    '    sparse_group_index = 0\n'
    '    previous_delta_curvature: torch.Tensor | None = None\n'
    '    if config.method == "adaptive_k" and not config.adaptive_k_carry_across_steps:\n'
    '        controller.current_budget = int(config.num_anchors)\n\n'
    '    for block_index, block in enumerate(transformer.blocks):\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        if replay_block_anchors is not None:\n            if block_index not in replay_block_anchors:\n                raise KeyError(f"Missing replay anchors for sparse block {block_index}")\n            anchors = list(replay_block_anchors[block_index])\n        elif config.method == "fis":\n',
    '        relative_zero = block_index - config.sparse_block_start\n'
    '        adaptive_group_index = relative_zero // config.block_group_size\n'
    '        is_group_start = relative_zero % config.block_group_size == 0\n'
    '        if replay_block_anchors is not None:\n'
    '            if block_index not in replay_block_anchors:\n'
    '                raise KeyError(f"Missing replay anchors for sparse block {block_index}")\n'
    '            anchors = list(replay_block_anchors[block_index])\n'
    '        elif config.method == "adaptive_k":\n'
    '            if is_group_start and config.adaptive_k_policy == "step_block":\n'
    '                controller.current_budget = lookup_scheduled_budget(\n'
    '                    config.adaptive_k_schedule,\n'
    '                    step_index=step_index,\n'
    '                    group_index=adaptive_group_index,\n'
    '                    fallback=config.num_anchors,\n'
    '                )\n'
    '            anchors = uniform_select(\n'
    '                geometry.num_frames,\n'
    '                int(controller.current_budget),\n'
    '                config.force_boundaries,\n'
    '            )\n'
    '            if is_group_start:\n'
    '                assignment = {\n'
    '                    "step": step_index,\n'
    '                    "group": adaptive_group_index,\n'
    '                    "after_block": block_index - 1,\n'
    '                    "policy": config.adaptive_k_policy,\n'
    '                    "assigned_k": int(controller.current_budget),\n'
    '                    "source": "step_block_schedule" if config.adaptive_k_policy == "step_block" else "previous_group",\n'
    '                }\n'
    '                metadata.budget_events.append(assignment)\n'
    '                controller.budget_history.append(dict(assignment))\n'
    '        elif config.method == "fis":\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        compute_defects = (\n            config.method == "coframe"\n            and replay_block_anchors is None\n            and (\n                (update_controller and config.refresh_signal in {"defect", "shuffled"})\n                or dense_output is not None\n            )\n        )\n',
    '        compute_defects = (\n'
    '            replay_block_anchors is None\n'
    '            and (\n'
    '                (\n'
    '                    config.method == "coframe"\n'
    '                    and ((update_controller and config.refresh_signal in {"defect", "shuffled"}) or dense_output is not None)\n'
    '                )\n'
    '                or (\n'
    '                    config.method == "adaptive_k"\n'
    '                    and ((update_controller and config.adaptive_k_policy in {"mean_defect", "max_defect"}) or dense_output is not None)\n'
    '                )\n'
    '            )\n'
    '        )\n',
)
# Add adaptive budget update at group boundary, before group_defects reset.
replace_once(
    "coframe/wan/sparse_forward.py",
    '            group_defects = {}\n\n    shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)\n',
    '            if config.method == "adaptive_k" and update_controller and config.adaptive_k_policy in {"mean_defect", "max_defect"}:\n'
    '                samples = [value for values in group_defects.values() for value in values]\n'
    '                statistic = "mean" if config.adaptive_k_policy == "mean_defect" else "max"\n'
    '                risk_value = defect_stat(samples, statistic)\n'
    '                if risk_value is not None:\n'
    '                    before_k = int(controller.current_budget)\n'
    '                    next_k = select_budget(risk_value, config.adaptive_k_thresholds, config.adaptive_k_values)\n'
    '                    controller.current_budget = int(next_k)\n'
    '                    update = {\n'
    '                        "step": step_index,\n'
    '                        "source_group": adaptive_group_index,\n'
    '                        "after_block": block_index,\n'
    '                        "policy": config.adaptive_k_policy,\n'
    '                        "risk_statistic": statistic,\n'
    '                        "risk_value": float(risk_value),\n'
    '                        "before_k": before_k,\n'
    '                        "next_k": int(next_k),\n'
    '                        "causal": True,\n'
    '                    }\n'
    '                    metadata.budget_events.append(update)\n'
    '                    controller.budget_history.append(dict(update))\n'
    '                    if trace is not None:\n'
    '                        trace.add("budget_update", **update)\n'
    '            group_defects = {}\n\n'
    '    shift, scale = (transformer.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)\n',
)

# ---------------- docs ----------------
doc = Path("docs/EXPERIMENT_PLAN.md")
text = doc.read_text(encoding="utf-8")
addition = '''\n\n## Stage-1d — causal adaptive exact-frame budget\n\nStage-1c suggests defect magnitude may be useful as a scalar block-risk signal even though defect localization was rejected. Stage-1d therefore tests **how much exact computation** to allocate, not where to place defect-driven anchors.\n\nFirst run the zero-GPU lag test on the preserved Stage-1b `full_kv` traces:\n\n```bash\npython scripts/analyze_stage1d_lagged.py \\\n  --root <STAGE1B_ROOT> \\\n  --output outputs/stage1d/lagged_analysis.json \\\n  --plan-output outputs/stage1d/budget_plan.json\n```\n\nThe causal contract is previous completed block-group defect -> next block-group budget. Only `RUN_ADAPTIVE_K_SCREEN` permits GPU execution. LOPO folds calibrate thresholds from the other seven prompts. The default budgets are `{6,9,12,21}` with calibration quantiles `{0.35,0.80,0.95}`, whose intended mean exact-frame count is approximately 9 under `full_kv`.\n\nFor each held-out prompt run dense, static K=9, step/block-only schedule, previous-group mean-defect adaptive K, and max-defect ablation:\n\n```bash\nbash scripts/run_stage1d_adaptive_k_wan21.sh \\\n  "<prompt>" outputs/stage1d_gpu/p0_s0 0 \\\n  outputs/stage1d/budget_plan.json p0_s0\n```\n\nThen aggregate:\n\n```bash\npython scripts/summarize_stage1d.py \\\n  --root outputs/stage1d_gpu \\\n  --output outputs/stage1d_gpu/summary.json\n```\n\nPrimary support requires mean-defect adaptation to remain within 5% of the K=9 exact-frame budget, beat static K=9 and the prompt-independent step/block schedule on realized operator NMSE, retain the sign after +3 dense propagation and at the dense-referenced endpoint, and survive the max-defect ablation. Latency is not a primary claim unless GPUs are exclusive. This Stage uses `full_kv` to keep mechanism comparisons close to linear in query-frame count; `anchor_only` speed validation follows only after the mechanism gate passes.\n'''
if "## Stage-1d — causal adaptive exact-frame budget" not in text:
    doc.write_text(text + addition, encoding="utf-8")

# ---------------- tests ----------------
Path("tests/test_stage1d_budget.py").write_text('''import pytest\n\nfrom coframe.budget import defect_stat, lookup_scheduled_budget, select_budget\nfrom coframe.config import CoFrameConfig\n\n\ndef test_budget_mapping_is_monotonic():\n    thresholds = (0.2, 0.5, 0.9)\n    values = (6, 9, 12, 21)\n    assert [select_budget(x, thresholds, values) for x in (0.1, 0.3, 0.7, 1.2)] == [6, 9, 12, 21]\n\n\ndef test_defect_statistics_and_schedule_lookup():\n    assert defect_stat([1.0, 2.0, 3.0], "mean") == 2.0\n    assert defect_stat([1.0, 2.0, 3.0], "max") == 3.0\n    assert lookup_scheduled_budget({"5:2": 12}, step_index=5, group_index=2, fallback=9) == 12\n    assert lookup_scheduled_budget({}, step_index=5, group_index=2, fallback=9) == 9\n\n\ndef test_adaptive_k_config_requires_resolved_policy():\n    with pytest.raises(ValueError):\n        CoFrameConfig(method="adaptive_k").validate(num_frames=21, num_blocks=30)\n    config = CoFrameConfig(\n        method="adaptive_k",\n        adaptive_k_policy="mean_defect",\n        adaptive_k_thresholds=(0.1, 0.2, 0.3),\n    )\n    config.validate(num_frames=21, num_blocks=30)\n\n\ndef test_step_block_schedule_budget_validation():\n    config = CoFrameConfig(\n        method="adaptive_k",\n        adaptive_k_policy="step_block",\n        adaptive_k_schedule={"5:0": 12, "5:1": 9},\n    )\n    config.validate(num_frames=21, num_blocks=30)\n''', encoding="utf-8")

print("Stage-1d runtime patch applied")

from pathlib import Path


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing patch target: {name}")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, replacement: str, name: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"missing start marker: {name}")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"missing end marker: {name}")
    return text[:i] + replacement + text[j:]


# 1) Allow the mesh controller to resize when the ODE budget changes.
path = Path("coframe/controller.py")
text = path.read_text()
old = """        self.current_budget = int(num_anchors)\n        self.budget_history: list[dict[str, Any]] = []\n\n    @property\n"""
new = """        self.current_budget = int(num_anchors)\n        self.budget_history: list[dict[str, Any]] = []\n\n    def set_budget(self, num_anchors: int, *, reset_dynamic_risk: bool = False) -> list[int]:\n        \"\"\"Resize the exact-frame mesh while preserving frame-wise risk evidence.\n\n        ODE-path allocation can change K between denoising steps. When K changes,\n        start that step from a uniform boundary-preserving mesh; subsequent block\n        groups immediately adapt it using leave-one-out residual defects.\n        \"\"\"\n        target = int(num_anchors)\n        if not 1 <= target <= self.num_frames:\n            raise ValueError(\"num_anchors must lie in [1, num_frames]\")\n        if self.force_boundaries and self.num_frames > 1 and target < 2:\n            raise ValueError(\"at least two anchors are required when boundaries are forced\")\n        if target != self.num_anchors:\n            self.num_anchors = target\n            self.anchors = uniform_select(self.num_frames, target, self.force_boundaries)\n        self.current_budget = target\n        if reset_dynamic_risk:\n            self.dynamic_risk.zero_()\n            self.observation_count.zero_()\n        return list(self.anchors)\n\n    @property\n"""
text = replace_once(text, old, new, "controller.set_budget")
path.write_text(text)


# 2) Turn coframe_ode placement into LOO-defect remeshing.
path = Path("coframe/wan/sparse_forward.py")
text = path.read_text()
start = '        elif config.method == "coframe_ode":\n'
end = '        elif config.method == "fis":\n'
replacement = '''        elif config.method == "coframe_ode":\n            # K is fixed within this denoising step by the ODE-path controller.\n            # The exact frame positions are then updated between block groups\n            # using leave-one-out residual defects measured on already-computed\n            # interior anchors. No dense reference or extra DiT forward is used.\n            if is_group_start or current_group_anchors is None:\n                target_budget = int(controller.current_budget)\n                if controller.num_anchors != target_budget:\n                    controller.set_budget(target_budget)\n                current_group_anchors = list(controller.anchors)\n            anchors = list(current_group_anchors)\n            if is_group_start:\n                metadata.budget_events.append(\n                    {\n                        "step": step_index,\n                        "group": adaptive_group_index,\n                        "after_block": block_index - 1,\n                        "policy": "ode_path",\n                        "assigned_k": int(controller.current_budget),\n                        "source": "previous_step_trajectory",\n                        "placement": "loo_residual_defect",\n                    }\n                )\n'''
text = replace_between(text, start, end, replacement, "coframe_ode placement")
old = '''                or (\n                    config.method == "adaptive_k"\n                    and ((update_controller and config.adaptive_k_policy in {"mean_defect", "max_defect"}) or dense_output is not None)\n                )\n'''
new = '''                or (\n                    config.method == "adaptive_k"\n                    and ((update_controller and config.adaptive_k_policy in {"mean_defect", "max_defect"}) or dense_output is not None)\n                )\n                or (\n                    config.method == "coframe_ode"\n                    and (update_controller or dense_output is not None)\n                )\n'''
text = replace_once(text, old, new, "coframe_ode defect computation")
marker = '''            if config.method == "adaptive_k" and update_controller and config.adaptive_k_policy in {"mean_defect", "max_defect"}:\n'''
insert = '''            if config.method == "coframe_ode" and update_controller:\n                aggregated = {\n                    frame: sum(values) / len(values)\n                    for frame, values in group_defects.items()\n                    if values\n                }\n                if aggregated:\n                    controller.observe(aggregated, anchors=anchors)\n                    if sparse_group_index % config.refresh_every_groups == 0:\n                        refreshes = controller.refresh()\n                        for refresh in refreshes:\n                            entry = {\n                                "step": step_index,\n                                "after_block": block_index,\n                                "group": sparse_group_index,\n                                "refresh_signal": "loo_residual_defect",\n                                "defect_mean": sum(aggregated.values()) / len(aggregated),\n                                "defect_max": max(aggregated.values()),\n                                **refresh.to_dict(),\n                            }\n                            metadata.refresh_events.append(entry)\n                            if trace is not None:\n                                trace.add("mesh_refresh", **entry)\n                # The refreshed controller mesh becomes the next group's mesh.\n                current_group_anchors = list(controller.anchors)\n\n'''
if marker not in text:
    raise RuntimeError("missing adaptive-k boundary marker")
text = text.replace(marker, insert + marker, 1)
path.write_text(text)


# 3) Conditional CFG branch must update the LOO controller; unconditional replays it.
path = Path("coframe/wan/pipeline.py")
text = path.read_text()
text = replace_once(
    text,
    '                    update_controller=config.method in {"coframe", "adaptive_k"},\n',
    '                    update_controller=config.method in {"coframe", "adaptive_k", "coframe_ode"},\n',
    "pipeline update_controller",
)
old = '''        max_swaps_per_refresh=(\n            config.max_swaps_per_refresh\n            if config.method == "coframe" and config.refresh_signal != "none"\n            else 0\n        ),\n'''
new = '''        max_swaps_per_refresh=(\n            config.max_swaps_per_refresh\n            if (config.method == "coframe_ode" or (config.method == "coframe" and config.refresh_signal != "none"))\n            else 0\n        ),\n'''
text = replace_once(text, old, new, "enable coframe_ode swaps")
path.write_text(text)


# 4) Replace the old coverage/interleaving integration test with a LOO-defect test.
path = Path("tests/test_transformer_forward.py")
text = path.read_text()
start = 'def test_ode_coframe_uses_group_level_interleaving(monkeypatch):\n'
replacement = '''def test_ode_coframe_uses_group_level_loo_defects(monkeypatch):\n    monkeypatch.setattr(sparse_module, "require_diffusers_034", lambda strict=True: "0.34.0")\n    torch.manual_seed(17)\n    transformer = Transformer()\n    hidden = torch.randn(1, 2, 5, 1, 2)\n    timestep = torch.tensor([500.0])\n    context = torch.randn(1, 3, 4)\n    config = CoFrameConfig(\n        method="coframe_ode",\n        num_anchors=3,\n        sparse_block_start=0,\n        sparse_block_end=2,\n        block_group_size=1,\n        kv_mode="full_kv",\n        sketch_dim=0,\n        risk_ema=0.0,\n        move_penalty=0.0,\n        min_refresh_gain=0.0,\n        max_swaps_per_refresh=1,\n    )\n    controller = AdaptiveMeshController(\n        num_frames=5,\n        num_anchors=3,\n        initial_anchors=[0, 2, 4],\n        prior_scores=torch.zeros(5),\n        prior_weight=0.0,\n        risk_ema=0.0,\n        move_penalty=0.0,\n        min_refresh_gain=0.0,\n        max_swaps_per_refresh=1,\n    )\n    controller.current_budget = 3\n\n    conditional, cond_meta = sparse_module.coframe_transformer_forward(\n        transformer,\n        hidden,\n        timestep,\n        context,\n        config=config,\n        controller=controller,\n        step_index=1,\n        update_controller=True,\n    )\n    replayed, replay_meta = sparse_module.coframe_transformer_forward(\n        transformer,\n        hidden,\n        timestep,\n        context * 0.0,\n        config=config,\n        controller=controller,\n        step_index=1,\n        replay_block_anchors=cond_meta.block_anchors,\n        update_controller=False,\n    )\n\n    assert conditional.shape == replayed.shape == hidden.shape\n    assert cond_meta.block_anchors == replay_meta.block_anchors\n    assert all(len(mesh) == 3 for mesh in cond_meta.block_anchors.values())\n    assert len(cond_meta.defects) == 2\n    assert all(entry["values"] for entry in cond_meta.defects)\n    assert len(cond_meta.refresh_events) >= 1\n    assert all(event["refresh_signal"] == "loo_residual_defect" for event in cond_meta.refresh_events)\n    assert all(mesh[0] == 0 and mesh[-1] == 4 for mesh in cond_meta.block_anchors.values())\n'''
i = text.find(start)
if i < 0:
    raise RuntimeError("missing old coframe_ode integration test")
text = text[:i] + replacement
path.write_text(text)


# 5) Add a unit test for changing K between denoising steps.
path = Path("tests/test_controller.py")
text = path.read_text()
append = '''\n\ndef test_set_budget_resizes_mesh_and_preserves_risk():\n    controller = make_controller()\n    controller.dynamic_risk[6] = 3.0\n    anchors = controller.set_budget(6)\n    assert len(anchors) == 6\n    assert anchors[0] == 0 and anchors[-1] == 12\n    assert controller.current_budget == 6\n    assert controller.num_anchors == 6\n    assert controller.dynamic_risk[6] == 3.0\n'''
if "test_set_budget_resizes_mesh_and_preserves_risk" not in text:
    text += append
path.write_text(text)


# 6) Make the canonical runner emit an actual MP4 for qualitative inspection.
path = Path("scripts/run_ode_coframe_wan21.sh")
text = path.read_text()
text = replace_once(
    text,
    '  --interpolation-target delta \\\n  --output-dir "$OUT_ROOT" \\\n',
    '  --interpolation-target delta \\\n  --decode --vae-tiling \\\n  --output-dir "$OUT_ROOT" \\\n',
    "runner decode",
)
path.write_text(text)


# 7) Update docs so coframe_ode no longer claims coverage-aware placement.
path = Path("README.md")
text = path.read_text()
text = text.replace(
    "**ODE-Path-Aware Frame Budgets and Coverage-Aware Sparse Video Diffusion**",
    "**ODE-Path-Aware Frame Budgets with Self-Validating Sparse Video Diffusion**",
)
text = text.replace(
    "`--method coframe_ode` uses current trajectory signals to allocate the next step's frame budget, then applies deterministic coverage-aware interleaving across block groups. The earlier leave-one-out defect controller remains available as `--method coframe` so all negative and diagnostic experiments stay reproducible.",
    "`--method coframe_ode` uses current trajectory signals to allocate the next step's frame budget. Inside that step, already-computed exact anchors validate their own residual interpolation through a leave-one-out defect, and the resulting risk field remeshes the next block group. The earlier fixed-K variant remains available as `--method coframe` for reproducibility.",
)
text = text.replace(
    "| `coframe_ode` | coverage-aware interleaving | step-level ODE/path budget | current proposed path |",
    "| `coframe_ode` | LOO residual-defect remeshing | step-level ODE/path budget + group-level self-validation | current proposed path |",
)
text = text.replace(
    "The default controller supports arbitrary integer frame budgets within its resolved range and exactly conserves the configured average frame budget across sparse steps. See `docs/ODE_PATH_COFRAME.md` for the causal and execution contract.",
    "The default controller supports arbitrary integer frame budgets within its resolved range and exactly conserves the configured average frame budget across sparse steps. Within each sparse step, exact interior anchors are compared against leave-one-out residual interpolation and the measured defect updates the next block-group mesh. See `docs/ODE_PATH_COFRAME.md` for the causal and execution contract.",
)
path.write_text(text)

path = Path("docs/ODE_PATH_COFRAME.md")
path.write_text('''# ODE-Path-Aware + Self-Validating CoFrame\n\n`--method coframe_ode` implements the current two-level policy.\n\n## Execution contract\n\n1. The first `warmup_steps` denoising steps are dense.\n2. Later Wan steps keep dense blocks `[0,3)` and `[27,30)`, and sparsify the middle block groups.\n3. The current guided flow output supplies ODE direction change, clean-endpoint change, and frame-wise temporal curvature without an extra DiT call. These signals determine the *next* sparse step's exact-frame budget K.\n4. All sparse groups in one denoising step use the same K. When K changes between steps, the first group starts from a uniform boundary-preserving mesh while previously accumulated frame-wise risk is retained.\n5. For each sparse block, selected anchors are computed exactly. At each interior exact anchor v with neighbors a<v<b, the controller predicts its block residual from the two neighbors and measures a normalized leave-one-out defect: `d_v = RMS(Delta h_v - Interp(Delta h_a, Delta h_b)) / (RMS(Delta h_v) + eps)`.\n6. Defects from the blocks in one group are averaged, projected onto their neighboring temporal intervals, and EMA-updated into a frame-wise risk field. A fixed-K one-swap search removes one interior anchor and adds one non-anchor only when the risk-weighted interpolation cost decreases. The refreshed mesh is used by the next block group.\n7. Skipped frames retain their incoming hidden state and receive a linearly interpolated block residual. This is residual interpolation inside a DiT block, not video frame synthesis.\n8. The conditional CFG branch chooses and updates the sparse schedule; the unconditional branch replays exactly the same per-block anchors.\n\nThe method is training-free and never needs a dense-reference forward at runtime. The LOO defect signal is computed only from exact anchors that were already required by the sparse block.\n''')

print("LOO defect placement update applied")

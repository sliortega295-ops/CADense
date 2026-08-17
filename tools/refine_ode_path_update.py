from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + addition.lstrip(), encoding="utf-8")


old_selector = '''def coverage_interleaved_select(
    num_frames: int,
    num_anchors: int,
    phase_index: int,
    *,
    force_boundaries: bool = True,
    anchor_stride: int = 0,
) -> list[int]:
    """Coverage-aware group-level interleaving with an exact frame budget.

    A rotating residue class changes which frames receive exact computation.
    Boundary insertion and largest-gap filling preserve temporal coverage while
    returning exactly ``num_anchors`` indices for arbitrary integer budgets.
    """
    if num_frames < 1 or num_anchors < 1:
        raise ValueError("num_frames and num_anchors must be positive")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    stride = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
    phase = int(phase_index) % stride
    selected = [frame for frame in range(num_frames) if (frame - phase) % stride == 0]
    return _fit_interleaved_budget(
        selected,
        num_frames=num_frames,
        num_anchors=num_anchors,
        force_boundaries=force_boundaries,
    )
'''
new_selector = '''def _minimum_coverage_mesh(
    num_frames: int,
    num_anchors: int,
    usage: Sequence[int],
    *,
    reuse_penalty: float,
) -> list[int]:
    """Solve one tiny exact-budget coverage problem with dynamic programming."""
    if num_anchors == 1:
        return [0]
    # State maps (selected_count, last_anchor) to (cost, path). Boundaries are
    # fixed; the gap-squared term minimizes large interpolation intervals while
    # the usage term encourages successive groups to cover different frames.
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(1, 0): (0.0, (0,))}
    for selected_count in range(2, num_anchors + 1):
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        if selected_count == num_anchors:
            right_candidates = (num_frames - 1,)
        else:
            minimum_right = selected_count - 1
            maximum_right = num_frames - 1 - (num_anchors - selected_count)
            right_candidates = range(minimum_right, maximum_right + 1)
        for right in right_candidates:
            best: tuple[float, tuple[int, ...]] | None = None
            for (count, left), (cost, path) in states.items():
                if count != selected_count - 1 or left >= right:
                    continue
                candidate_cost = cost + float((right - left) ** 2)
                if right != num_frames - 1:
                    candidate_cost += float(reuse_penalty) * float(usage[right])
                candidate = (candidate_cost, path + (right,))
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                next_states[(selected_count, right)] = best
        states = next_states
    return list(states[(num_anchors, num_frames - 1)][1])


def coverage_interleaved_select(
    num_frames: int,
    num_anchors: int,
    phase_index: int,
    *,
    force_boundaries: bool = True,
    anchor_stride: int = 0,
    reuse_penalty: float = 2.0,
) -> list[int]:
    """Select an exact-budget, coverage-aware mesh for one block group.

    With temporal boundaries enabled, a small dynamic program minimizes the
    sum of squared anchor gaps plus a reuse penalty accumulated over earlier
    phases in the cycle. This retains near-uniform coverage while rotating
    exact computation across frames. ``anchor_stride`` optionally sets the
    interleaving period; zero derives a short period from ``ceil(F/K)``.
    """
    if num_frames < 1 or num_anchors < 1:
        raise ValueError("num_frames and num_anchors must be positive")
    if reuse_penalty < 0.0:
        raise ValueError("reuse_penalty must be non-negative")
    if num_anchors >= num_frames:
        return list(range(num_frames))
    if not force_boundaries:
        # The residual interpolator is normally used with boundary anchors.
        # Preserve the earlier deterministic rotating-residue behavior for the
        # uncommon boundary-free diagnostic mode.
        stride = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
        phase = int(phase_index) % stride
        selected = [frame for frame in range(num_frames) if (frame - phase) % stride == 0]
        return _fit_interleaved_budget(
            selected,
            num_frames=num_frames,
            num_anchors=num_anchors,
            force_boundaries=False,
        )

    period = int(anchor_stride) if int(anchor_stride) > 0 else max(1, math.ceil(num_frames / num_anchors))
    phase = int(phase_index) % period
    usage = [0 for _ in range(num_frames)]
    mesh: list[int] = []
    for _ in range(phase + 1):
        mesh = _minimum_coverage_mesh(
            num_frames,
            num_anchors,
            usage,
            reuse_penalty=float(reuse_penalty),
        )
        for frame in mesh[1:-1]:
            usage[frame] += 1
    return mesh
'''
replace_once("coframe/selection.py", old_selector, new_selector)

replace_once(
    "coframe/config.py",
    '''    ode_anchor_stride: int = 0
    ode_interleave_across_steps: bool = True
''',
    '''    ode_anchor_stride: int = 0
    ode_interleave_penalty: float = 2.0
    ode_interleave_across_steps: bool = True
''',
)
replace_once(
    "coframe/config.py",
    '''        if self.ode_anchor_stride < 0:
            raise ValueError("ode_anchor_stride must be >= 0")
''',
    '''        if self.ode_anchor_stride < 0:
            raise ValueError("ode_anchor_stride must be >= 0")
        if self.ode_interleave_penalty < 0.0:
            raise ValueError("ode_interleave_penalty must be non-negative")
''',
)

replace_once(
    "coframe/cli.py",
    '''    parser.add_argument("--ode-anchor-stride", type=int, default=0)
    parser.add_argument("--no-ode-interleave-across-steps", action="store_true")
''',
    '''    parser.add_argument("--ode-anchor-stride", type=int, default=0)
    parser.add_argument("--ode-interleave-penalty", type=float, default=2.0)
    parser.add_argument("--no-ode-interleave-across-steps", action="store_true")
''',
)
replace_once(
    "coframe/cli.py",
    '''        ode_anchor_stride=args.ode_anchor_stride,
        ode_interleave_across_steps=not args.no_ode_interleave_across_steps,
''',
    '''        ode_anchor_stride=args.ode_anchor_stride,
        ode_interleave_penalty=args.ode_interleave_penalty,
        ode_interleave_across_steps=not args.no_ode_interleave_across_steps,
''',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    '''    previous_delta_curvature: torch.Tensor | None = None
    if config.method == "adaptive_k" and not config.adaptive_k_carry_across_steps:
''',
    '''    previous_delta_curvature: torch.Tensor | None = None
    current_group_anchors: list[int] | None = None
    if config.method == "adaptive_k" and not config.adaptive_k_carry_across_steps:
''',
)
old_branch = '''        elif config.method == "coframe_ode":
            group_count = max(
                1,
                math.ceil((config.sparse_block_end - config.sparse_block_start) / config.block_group_size),
            )
            phase_index = adaptive_group_index
            if config.ode_interleave_across_steps:
                phase_index += step_index * group_count
            anchors = coverage_interleaved_select(
                geometry.num_frames,
                int(controller.current_budget),
                phase_index,
                force_boundaries=config.force_boundaries,
                anchor_stride=config.ode_anchor_stride,
            )
            if is_group_start:
                metadata.budget_events.append(
                    {
                        "step": step_index,
                        "group": adaptive_group_index,
                        "after_block": block_index - 1,
                        "policy": "ode_path",
                        "assigned_k": int(controller.current_budget),
                        "source": "previous_step_trajectory",
                        "phase_index": int(phase_index),
                    }
                )
'''
new_branch = '''        elif config.method == "coframe_ode":
            group_count = max(
                1,
                math.ceil((config.sparse_block_end - config.sparse_block_start) / config.block_group_size),
            )
            phase_index = adaptive_group_index
            if config.ode_interleave_across_steps:
                phase_index += step_index * group_count
            if is_group_start or current_group_anchors is None:
                current_group_anchors = coverage_interleaved_select(
                    geometry.num_frames,
                    int(controller.current_budget),
                    phase_index,
                    force_boundaries=config.force_boundaries,
                    anchor_stride=config.ode_anchor_stride,
                    reuse_penalty=config.ode_interleave_penalty,
                )
            anchors = list(current_group_anchors)
            if is_group_start:
                metadata.budget_events.append(
                    {
                        "step": step_index,
                        "group": adaptive_group_index,
                        "after_block": block_index - 1,
                        "policy": "ode_path",
                        "assigned_k": int(controller.current_budget),
                        "source": "previous_step_trajectory",
                        "phase_index": int(phase_index),
                    }
                )
'''
replace_once("coframe/wan/sparse_forward.py", old_branch, new_branch)

replace_once(
    "coframe/wan/pipeline.py",
    '''def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
''',
    '''def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_flow_prediction_scheduler(scheduler: Any) -> None:
    """Fail closed before applying the flow clean-endpoint conversion."""
    scheduler_name = type(scheduler).__name__
    scheduler_config = getattr(scheduler, "config", None)
    prediction_type = getattr(scheduler_config, "prediction_type", None)
    use_flow_sigmas = bool(getattr(scheduler_config, "use_flow_sigmas", False))
    native_flow_scheduler = scheduler_name.startswith("FlowMatch")
    if not native_flow_scheduler and not (prediction_type == "flow_prediction" and use_flow_sigmas):
        raise RuntimeError(
            "coframe_ode requires a flow-prediction scheduler: either a FlowMatch scheduler or "
            "prediction_type='flow_prediction' with use_flow_sigmas=True"
        )


@torch.no_grad()
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''    if not hasattr(pipe.scheduler, "sigmas"):
        raise RuntimeError("CoFrame requires a flow scheduler exposing sigmas")
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
''',
    '''    if not hasattr(pipe.scheduler, "sigmas"):
        raise RuntimeError("CoFrame requires a scheduler exposing sigmas")
    if config.method == "coframe_ode":
        _require_flow_prediction_scheduler(pipe.scheduler)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''    transformer_events: list[dict[str, Any]] = []
    ode_budget_controller: ODEPathBudgetController | None = None
''',
    '''    transformer_events: list[dict[str, Any]] = []
    last_sparse_anchors: list[int] | None = None
    ode_budget_controller: ODEPathBudgetController | None = None
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''                sparse_step_count += 1
                transformer_events.append(cond_metadata.to_dict())
''',
    '''                sparse_step_count += 1
                transformer_events.append(cond_metadata.to_dict())
                if cond_metadata.block_anchors:
                    final_block = max(cond_metadata.block_anchors)
                    last_sparse_anchors = list(cond_metadata.block_anchors[final_block])
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''        "final_anchors": None if controller is None else list(controller.anchors),
''',
    '''        "final_anchors": (
            last_sparse_anchors
            if last_sparse_anchors is not None
            else (None if controller is None else list(controller.anchors))
        ),
''',
)

replace_once(
    "tests/test_ode_budget.py",
    'from coframe.selection import coverage_interleaved_select\n',
    'from coframe.selection import coverage_interleaved_select\nfrom coframe.wan.pipeline import _require_flow_prediction_scheduler\n',
)
replace_once(
    "tests/test_ode_budget.py",
    '''def test_coverage_interleaving_keeps_budget_boundaries_and_changes_phase():
    first = coverage_interleaved_select(21, 9, 0)
    second = coverage_interleaved_select(21, 9, 1)
    assert len(first) == len(second) == 9
    assert first[0] == second[0] == 0
    assert first[-1] == second[-1] == 20
    assert first != second
''',
    '''def test_coverage_interleaving_keeps_budget_boundaries_and_changes_phase():
    meshes = [coverage_interleaved_select(21, 9, phase) for phase in range(3)]
    assert all(len(mesh) == 9 for mesh in meshes)
    assert all(mesh[0] == 0 and mesh[-1] == 20 for mesh in meshes)
    assert len({tuple(mesh) for mesh in meshes}) == 3
    assert all(max(right - left for left, right in zip(mesh[:-1], mesh[1:])) <= 3 for mesh in meshes)


def test_flow_scheduler_contract_fails_closed():
    class Scheduler:
        pass

    flow = Scheduler()
    flow.config = type("Config", (), {"prediction_type": "flow_prediction", "use_flow_sigmas": True})()
    _require_flow_prediction_scheduler(flow)

    epsilon = Scheduler()
    epsilon.config = type("Config", (), {"prediction_type": "epsilon", "use_flow_sigmas": False})()
    with pytest.raises(RuntimeError, match="flow-prediction"):
        _require_flow_prediction_scheduler(epsilon)
''',
)

replace_once(
    "docs/ODE_PATH_COFRAME.md",
    '''5. A remaining-budget multiplier and reachability check meet the configured total frame budget exactly. The default support is every integer between the automatic minimum and the latent-frame count; a smaller execution codebook can be supplied with `--ode-budget-values`.
6. Within a step, all sparse groups share the selected frame count, while a coverage-aware interleaved mesh changes the exact frame positions across block groups. Skipped frames retain their incoming state and receive a linearly interpolated block residual.
''',
    '''5. A remaining-budget multiplier and reachability check meet the configured total frame budget exactly. The default support is every integer between the automatic minimum and the latent-frame count; a smaller execution codebook can be supplied with `--ode-budget-values`.
6. Within a step, all sparse groups share the selected frame count. Frame positions minimize squared temporal gaps plus a frozen reuse penalty, producing near-uniform coverage while interleaving exact frames across block groups. Skipped frames retain their incoming state and receive a linearly interpolated block residual.
7. The clean-endpoint conversion fails closed unless the sampler exposes a compatible flow-prediction scheduler contract.
''',
)

print("ODE CoFrame refinements applied")

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


replace_once(
    "coframe/config.py",
    'Method = Literal["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k"]',
    'Method = Literal["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k", "coframe_ode"]',
)
replace_once(
    "coframe/config.py",
    '''    adaptive_k_carry_across_steps: bool = True

    trace_path: str | None = None
''',
    '''    adaptive_k_carry_across_steps: bool = True

    # Current proposed path: step-level ODE/path-aware budget plus deterministic
    # coverage-aware interleaved placement. Empty ode_budget_values means every
    # integer budget in [ode_min_anchors, ode_max_anchors] is supported.
    ode_target_average_k: float = 0.0  # 0 -> num_anchors
    ode_min_anchors: int = 0  # 0 -> round(2/3 * target), respecting boundaries
    ode_max_anchors: int = 0  # 0 -> all latent frames
    ode_budget_values: Sequence[int] = field(default_factory=tuple)
    ode_signal_ema: float = 0.9
    ode_signal_clip: float = 4.0
    ode_direction_weight: float = 0.5
    ode_endpoint_weight: float = 0.5
    ode_difficulty_power: float = 1.0 / 3.0
    ode_anchor_stride: int = 0
    ode_interleave_across_steps: bool = True

    trace_path: str | None = None
''',
)
replace_once(
    "coframe/config.py",
    '        if self.method not in {"dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k"}:\n',
    '        if self.method not in {"dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k", "coframe_ode"}:\n',
)
replace_once(
    "coframe/config.py",
    '''        if self.adaptive_k_policy == "step_block":
            invalid_schedule = [value for value in self.adaptive_k_schedule.values() if int(value) not in budget_values]
            if invalid_schedule:
                raise ValueError("adaptive_k_schedule contains a budget outside adaptive_k_values")
''',
    '''        if self.adaptive_k_policy == "step_block":
            invalid_schedule = [value for value in self.adaptive_k_schedule.values() if int(value) not in budget_values]
            if invalid_schedule:
                raise ValueError("adaptive_k_schedule contains a budget outside adaptive_k_values")
        if not 0.0 <= self.ode_signal_ema < 1.0:
            raise ValueError("ode_signal_ema must be in [0,1)")
        if self.ode_signal_clip < 1.0:
            raise ValueError("ode_signal_clip must be >= 1")
        if self.ode_direction_weight < 0.0 or self.ode_endpoint_weight < 0.0:
            raise ValueError("ODE difficulty weights must be non-negative")
        if self.ode_direction_weight + self.ode_endpoint_weight <= 0.0:
            raise ValueError("at least one ODE difficulty weight must be positive")
        if self.ode_difficulty_power <= 0.0:
            raise ValueError("ode_difficulty_power must be positive")
        if self.ode_min_anchors < 0 or self.ode_max_anchors < 0:
            raise ValueError("ODE anchor bounds must be non-negative; zero selects the automatic bound")
        if self.ode_anchor_stride < 0:
            raise ValueError("ode_anchor_stride must be >= 0")
        ode_values = [int(value) for value in self.ode_budget_values]
        if ode_values and ode_values != sorted(set(ode_values)):
            raise ValueError("ode_budget_values must be strictly increasing when provided")
        if any(value < 1 for value in ode_values):
            raise ValueError("ode_budget_values must be positive")
''',
)
replace_once(
    "coframe/config.py",
    '''        if (
            self.method == "adaptive_k"
            and self.force_boundaries
            and any(int(value) < 2 for value in self.adaptive_k_values)
            and (num_frames is None or num_frames > 1)
        ):
            raise ValueError("adaptive_k_values must be >=2 when force_boundaries=True")
''',
    '''        if (
            self.method == "adaptive_k"
            and self.force_boundaries
            and any(int(value) < 2 for value in self.adaptive_k_values)
            and (num_frames is None or num_frames > 1)
        ):
            raise ValueError("adaptive_k_values must be >=2 when force_boundaries=True")
        if self.method == "coframe_ode" and num_frames is not None:
            target = float(self.ode_target_average_k or self.num_anchors)
            boundary_floor = 2 if self.force_boundaries and num_frames > 1 else 1
            minimum = int(self.ode_min_anchors or max(boundary_floor, round(2.0 * target / 3.0)))
            maximum = int(self.ode_max_anchors or num_frames)
            if not boundary_floor <= minimum <= target <= maximum <= num_frames:
                raise ValueError("resolved ODE budgets must satisfy boundary_floor <= min <= target <= max <= frames")
            if self.ode_budget_values and any(
                int(value) < minimum or int(value) > maximum for value in self.ode_budget_values
            ):
                raise ValueError("ode_budget_values must lie inside the resolved ODE budget range")
''',
)
replace_once(
    "coframe/config.py",
    '''        result["adaptive_k_schedule"] = dict(self.adaptive_k_schedule)
        return result
''',
    '''        result["adaptive_k_schedule"] = dict(self.adaptive_k_schedule)
        result["ode_budget_values"] = list(self.ode_budget_values)
        return result
''',
)

replace_once(
    "coframe/selection.py",
    'def fis_interleaved_select(\n',
    '''def coverage_interleaved_select(
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


def fis_interleaved_select(
''',
)

replace_once(
    "coframe/wan/sparse_forward.py",
    'from ..selection import fis_interleaved_select, uniform_select',
    'from ..selection import coverage_interleaved_select, fis_interleaved_select, uniform_select',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        elif config.method == "fis":\n',
    '''        elif config.method == "coframe_ode":
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
        elif config.method == "fis":
''',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '    probe_budget = len(anchors) if config.method == "adaptive_k" else config.num_anchors\n',
    '    probe_budget = len(anchors) if config.method in {"adaptive_k", "coframe_ode"} else config.num_anchors\n',
)
replace_once(
    "coframe/wan/sparse_forward.py",
    '        if config.method == "adaptive_k"\n',
    '        if config.method in {"adaptive_k", "coframe_ode"}\n',
)

replace_once(
    "coframe/wan/pipeline.py",
    'from ..controller import AdaptiveMeshController\n',
    'from ..controller import AdaptiveMeshController\nfrom ..ode_budget import ODEPathBudgetController\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '    if config.method in {"fixed", "adaptive_k"}:\n',
    '    if config.method in {"fixed", "adaptive_k", "coframe_ode"}:\n',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''    transformer_events: list[dict[str, Any]] = []
    dense_step_count = 0
''',
    '''    transformer_events: list[dict[str, Any]] = []
    ode_budget_controller: ODEPathBudgetController | None = None
    dense_step_count = 0
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''    if config.method in {"dense", "fis"} or num_inference_steps <= 1:
        effective_warmup = 0
    else:
        # Always leave at least one sparse denoising step, including few-step
        # smoke tests. The canonical 50-step setting still uses warmup=5.
        effective_warmup = max(1, min(config.warmup_steps, num_inference_steps - 1))

    _cuda_sync()
''',
    '''    if config.method in {"dense", "fis"} or num_inference_steps <= 1:
        effective_warmup = 0
    else:
        # Always leave at least one sparse denoising step, including few-step
        # smoke tests. The canonical 50-step setting still uses warmup=5.
        effective_warmup = max(1, min(config.warmup_steps, num_inference_steps - 1))

    if config.method == "coframe_ode":
        controller, initial_anchors, prior_scores = _make_controller(config=config, clean_proxy=latents)
        total_sparse_steps = sum(
            1
            for index in range(num_inference_steps)
            if index >= effective_warmup and config.is_sparse_step(index, num_inference_steps)
        )
        ode_budget_controller = ODEPathBudgetController.from_config(
            config,
            num_frames=latent_frame_count,
            total_sparse_steps=total_sparse_steps,
        )
        trace.add(
            "mesh_initialization",
            step=-1,
            timestep=None,
            anchors=initial_anchors,
            prior_scores=prior_scores,
            placement_policy="coverage_interleaved",
        )
        trace.add("ode_budget_initialization", **ode_budget_controller.state_dict())

    _cuda_sync()
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '            latents = pipe.scheduler.step(noise_pred, timestep_scalar, latents, return_dict=False)[0]\n',
    '''            if ode_budget_controller is not None:
                sigma = sigmas[step_index].to(device=latents.device, dtype=torch.float32)
                signal = ode_budget_controller.observe(
                    step_index=step_index,
                    sample=latents,
                    velocity=noise_pred,
                    sigma=sigma,
                )
                trace.add("ode_path_signal", **signal.to_dict())
                next_step = step_index + 1
                if (
                    next_step < num_inference_steps
                    and next_step >= effective_warmup
                    and config.is_sparse_step(next_step, num_inference_steps)
                ):
                    budget_event = ode_budget_controller.allocate_next(
                        source_step=step_index,
                        target_step=next_step,
                        difficulty=signal.difficulty,
                    )
                    assert controller is not None
                    controller.current_budget = int(budget_event.assigned_budget)
                    controller.budget_history.append(budget_event.to_dict())
                    trace.add("budget_update", policy="ode_path", causal=True, **budget_event.to_dict())

            latents = pipe.scheduler.step(noise_pred, timestep_scalar, latents, return_dict=False)[0]
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '''        "controller": None if controller is None else controller.state_dict(),
        "config": config.to_dict(),
''',
    '''        "controller": None if controller is None else controller.state_dict(),
        "ode_budget_controller": None if ode_budget_controller is None else ode_budget_controller.state_dict(),
        "config": config.to_dict(),
''',
)
replace_once(
    "coframe/wan/pipeline.py",
    '    """Run dense/fixed/Rhyme/CoFrame under one Wan2.1 sampler contract."""\n',
    '    """Run dense and sparse CoFrame variants under one Wan2.1 sampler contract."""\n',
)

replace_once(
    "coframe/cli.py",
    '    parser.add_argument("--method", choices=["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k"], default="coframe")\n',
    '''    parser.add_argument(
        "--method",
        choices=["dense", "fixed", "fis", "rhyme", "coframe", "adaptive_k", "coframe_ode"],
        default="coframe",
    )
''',
)
replace_once(
    "coframe/cli.py",
    '''    parser.add_argument("--no-adaptive-k-carry", action="store_true")

    parser.add_argument("--output-dir", type=Path, required=True)
''',
    '''    parser.add_argument("--no-adaptive-k-carry", action="store_true")
    parser.add_argument("--ode-target-average-k", type=float, default=0.0)
    parser.add_argument("--ode-min-anchors", type=int, default=0)
    parser.add_argument("--ode-max-anchors", type=int, default=0)
    parser.add_argument("--ode-budget-values", type=_csv_ints, default=())
    parser.add_argument("--ode-signal-ema", type=float, default=0.9)
    parser.add_argument("--ode-signal-clip", type=float, default=4.0)
    parser.add_argument("--ode-direction-weight", type=float, default=0.5)
    parser.add_argument("--ode-endpoint-weight", type=float, default=0.5)
    parser.add_argument("--ode-difficulty-power", type=float, default=1.0 / 3.0)
    parser.add_argument("--ode-anchor-stride", type=int, default=0)
    parser.add_argument("--no-ode-interleave-across-steps", action="store_true")

    parser.add_argument("--output-dir", type=Path, required=True)
''',
)
replace_once(
    "coframe/cli.py",
    '''        adaptive_k_carry_across_steps=not args.no_adaptive_k_carry,
        trace_path=str(trace_path),
''',
    '''        adaptive_k_carry_across_steps=not args.no_adaptive_k_carry,
        ode_target_average_k=args.ode_target_average_k,
        ode_min_anchors=args.ode_min_anchors,
        ode_max_anchors=args.ode_max_anchors,
        ode_budget_values=args.ode_budget_values,
        ode_signal_ema=args.ode_signal_ema,
        ode_signal_clip=args.ode_signal_clip,
        ode_direction_weight=args.ode_direction_weight,
        ode_endpoint_weight=args.ode_endpoint_weight,
        ode_difficulty_power=args.ode_difficulty_power,
        ode_anchor_stride=args.ode_anchor_stride,
        ode_interleave_across_steps=not args.no_ode_interleave_across_steps,
        trace_path=str(trace_path),
''',
)

replace_once(
    "coframe/__init__.py",
    '''from .controller import AdaptiveMeshController
from .selection import rhyme_select, uniform_select

__all__ = ["AdaptiveMeshController", "CoFrameConfig", "rhyme_select", "uniform_select"]
''',
    '''from .controller import AdaptiveMeshController
from .ode_budget import ODEPathBudgetController
from .selection import coverage_interleaved_select, rhyme_select, uniform_select

__all__ = [
    "AdaptiveMeshController",
    "CoFrameConfig",
    "ODEPathBudgetController",
    "coverage_interleaved_select",
    "rhyme_select",
    "uniform_select",
]
''',
)

append_once(
    "tests/test_transformer_forward.py",
    "test_ode_coframe_uses_group_level_interleaving",
    '''
def test_ode_coframe_uses_group_level_interleaving(monkeypatch):
    monkeypatch.setattr(sparse_module, "require_diffusers_034", lambda strict=True: "0.34.0")
    torch.manual_seed(17)
    transformer = Transformer()
    hidden = torch.randn(1, 2, 5, 1, 2)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 4)
    config = CoFrameConfig(
        method="coframe_ode",
        num_anchors=3,
        sparse_block_start=0,
        sparse_block_end=2,
        block_group_size=1,
        kv_mode="full_kv",
        sketch_dim=0,
    )
    controller = AdaptiveMeshController(
        num_frames=5,
        num_anchors=3,
        initial_anchors=[0, 2, 4],
        prior_scores=torch.zeros(5),
        prior_weight=0.0,
    )
    controller.current_budget = 3

    conditional, cond_meta = sparse_module.coframe_transformer_forward(
        transformer,
        hidden,
        timestep,
        context,
        config=config,
        controller=controller,
        step_index=1,
        update_controller=False,
    )
    replayed, replay_meta = sparse_module.coframe_transformer_forward(
        transformer,
        hidden,
        timestep,
        context * 0.0,
        config=config,
        controller=controller,
        step_index=1,
        replay_block_anchors=cond_meta.block_anchors,
        update_controller=False,
    )

    assert conditional.shape == replayed.shape == hidden.shape
    assert cond_meta.block_anchors == replay_meta.block_anchors
    assert all(len(mesh) == 3 for mesh in cond_meta.block_anchors.values())
    assert cond_meta.block_anchors[0] != cond_meta.block_anchors[1]
    assert all(mesh[0] == 0 and mesh[-1] == 4 for mesh in cond_meta.block_anchors.values())
''',
)

replace_once(
    "README.md",
    '''**Self-Validating Adaptive Frame Meshes for Sparse Video Diffusion**

CoFrame is a research prototype for testing block-conditional sparse-frame computation in **Wan2.1-T2V-1.3B**. The repository is deliberately organized around a falsifiable question:

> Starting from a strong RhymeFlow-style clean-latent frame selector, can block-level leave-one-out interpolation defects improve the frame mesh under the same exact-frame budget?

The current code targets `diffusers==0.34.0`, the same Wan integration generation used by the public RhymeFlow implementation. It is training-free and does not modify model weights.

## Why this version of the idea
''',
    '''**ODE-Path-Aware Frame Budgets and Coverage-Aware Sparse Video Diffusion**

CoFrame is a training-free research prototype for block-conditional sparse-frame computation in **Wan2.1-T2V-1.3B**. The current proposed path separates two decisions:

> How many frames should receive exact computation at the next denoising step, and where should those exact frames be placed inside each sparse block group?

`--method coframe_ode` uses current trajectory signals to allocate the next step's frame budget, then applies deterministic coverage-aware interleaving across block groups. The earlier leave-one-out defect controller remains available as `--method coframe` so all negative and diagnostic experiments stay reproducible. The code targets `diffusers==0.34.0` and does not modify model weights.

## Historical mechanism experiments
''',
)
replace_once(
    "README.md",
    '| `coframe` | same Rhyme initialization | leave-one-out block defect | proposed method |\n',
    '| `coframe` | same Rhyme initialization | leave-one-out block defect | legacy mechanism ablation |\n| `coframe_ode` | coverage-aware interleaving | step-level ODE/path budget | current proposed path |\n',
)
replace_once(
    "README.md",
    '## Canonical Wan2.1-1.3B validation\n',
    '''## ODE-path-aware CoFrame

```bash
bash scripts/run_ode_coframe_wan21.sh \\
  "A red toy car turns sharply around a blue cube on a wooden table." \\
  outputs/ode_coframe \\
  0
```

The default controller supports arbitrary integer frame budgets within its resolved range and exactly conserves the configured average frame budget across sparse steps. See `docs/ODE_PATH_COFRAME.md` for the causal and execution contract.

## Canonical Wan2.1-1.3B validation
''',
)

print("existing files patched")

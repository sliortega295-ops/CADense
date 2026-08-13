from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from coframe.config import CoFrameConfig
from coframe.cross_step_endpoint import (
    BASE_K,
    BUDGETS,
    build_physical_runs,
    validate_runtime_manifest,
)
from coframe.wan.pipeline import coframe_wan_generate, prepare_initial_latents


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)
LATENT_SCHEMA = "coframe.cross-step-endpoint-latents.v1"
SUMMARY_SCHEMA = "coframe.cross-step-endpoint-run-summary.v1"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = _json_bytes({"shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return hashlib.sha256(header + tensor.numpy().tobytes(order="C")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_tree(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _scheduler(flow_shift: float) -> Any:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    return UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=float(flow_shift),
    )


def _model_fingerprint(model_id: str) -> str:
    path = Path(model_id)
    if not path.exists():
        return _json_sha256({"model_id": model_id})
    rows = []
    for value in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = value.stat()
        rows.append([str(value.relative_to(path)), int(stat.st_size)])
    return _json_sha256({"model_path": str(path.resolve()), "files": rows})


def _runtime_invariants(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "warmup_steps": args.warmup_steps,
        "num_anchors": BASE_K,
        "sparse_blocks": [3, 27],
        "block_group_size": 3,
        "kv_mode": "full_kv",
        "interpolation_target": "delta",
        "adaptive_k_values": list(BUDGETS),
        "calibrated_budget_probe_mode": "current",
        "calibrated_budget_probe_slots": ["22:0", "44:5", "47:3", "49:2"],
    }


def _validate_cfg_and_schedule(metadata: dict[str, Any], schedule: dict[str, int]) -> dict[str, Any]:
    events = metadata.get("transformer_events", [])
    expected_steps = set(range(5, 50))
    actual_steps = {int(event["step_index"]) for event in events}
    if actual_steps != expected_steps or len(events) != len(expected_steps):
        raise RuntimeError(f"sparse transformer events differ from steps 5..49: {sorted(actual_steps)}")
    observed: dict[str, int] = {}
    for event in events:
        step = int(event["step_index"])
        block_anchors = {int(key): list(value) for key, value in event["block_anchors"].items()}
        if set(block_anchors) != set(range(3, 27)):
            raise RuntimeError(f"step {step}: sparse block anchors are incomplete")
        for group in range(8):
            sizes = {len(block_anchors[block]) for block in range(3 + 3 * group, 6 + 3 * group)}
            if len(sizes) != 1:
                raise RuntimeError(f"step {step} group {group}: within-group budgets diverged")
            observed[f"{step}:{group}"] = sizes.pop()
    if observed != schedule:
        raise RuntimeError("deployed conditional block schedule differs from physical schedule")
    # The generation pipeline already raises if the unconditional replay differs.
    # Record that this runtime-level replay assertion completed successfully.
    return {
        "conditional_schedule_matches_manifest": True,
        "unconditional_replay_runtime_assertion_passed": True,
        "observed_schedule_sha256": _json_sha256(observed),
    }


def _save_run(
    *,
    pipe: Any,
    args: argparse.Namespace,
    root: Path,
    run_id: str,
    kind: str,
    initial_latents: torch.Tensor,
    schedule: dict[str, int] | None,
    schedule_sha256: str | None,
    source_commit: str,
    source_tree: str,
    model_fingerprint: str,
    plan_sha256: str,
    protocol_sha256: str,
    load_time_sec: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trace.json"
    latent_path = run_dir / "latents.pt"
    summary_path = run_dir / "summary.json"
    physical_schedule_path: Path | None = None
    if schedule is not None:
        physical_schedule_path = run_dir / "physical_schedule.json"
        _write_json(physical_schedule_path, schedule)

    if kind == "dense":
        config = CoFrameConfig(method="dense", trace_path=str(trace_path))
    else:
        if schedule is None:
            raise RuntimeError("sparse and parity runs require a physical schedule")
        config = CoFrameConfig(
            method="adaptive_k",
            warmup_steps=args.warmup_steps,
            num_anchors=BASE_K,
            sparse_block_start=3,
            sparse_block_end=27,
            block_group_size=3,
            kv_mode="full_kv",
            interpolation_target="delta",
            defect_target="delta",
            adaptive_k_policy="step_block",
            adaptive_k_values=BUDGETS,
            adaptive_k_schedule=dict(schedule),
            calibrated_budget_probe_mode="current",
            calibrated_budget_probe_slots=("22:0", "44:5", "47:3", "49:2"),
            trace_path=str(trace_path),
        )
    config.validate(num_blocks=len(pipe.transformer.blocks), num_frames=int(initial_latents.shape[2]))

    pipe.scheduler = _scheduler(args.flow_shift)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = coframe_wan_generate(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        latents=initial_latents.clone(),
        generator=None,
        config=config,
    )
    torch.cuda.synchronize()
    generation_time = time.perf_counter() - started

    final_latents = result.frames.detach().float().cpu()
    if not bool(torch.isfinite(final_latents).all().item()):
        raise RuntimeError(f"{run_id}: final latents contain NaN or Inf")
    initial_sha = _tensor_sha256(initial_latents.float())
    final_sha = _tensor_sha256(final_latents)
    runtime_invariants = _runtime_invariants(args)
    runtime_config_sha = _json_sha256(runtime_invariants)
    source_fingerprint = _json_sha256(
        {
            "initial_latent_sha256": initial_sha,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "model_fingerprint": model_fingerprint,
            "runtime_config_sha256": runtime_config_sha,
            "plan_sha256": plan_sha256,
            "protocol_sha256": protocol_sha256,
        }
    )
    cfg_audit = None if kind == "dense" else _validate_cfg_and_schedule(result.metadata, schedule)
    payload = {
        "schema_version": LATENT_SCHEMA,
        "latents": final_latents,
        "prompt": args.prompt,
        "prompt_id": args.prompt_id,
        "seed": args.seed,
        "run_id": run_id,
        "kind": kind,
        "initial_latent_sha256": initial_sha,
        "final_latent_sha256": final_sha,
        "physical_schedule_sha256": schedule_sha256,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "model_id": args.model_id,
        "model_fingerprint": model_fingerprint,
        "runtime_config_sha256": runtime_config_sha,
        "source_fingerprint": source_fingerprint,
        "plan_sha256": plan_sha256,
        "protocol_sha256": protocol_sha256,
    }
    torch.save(payload, latent_path)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "success",
        **{key: value for key, value in payload.items() if key != "latents"},
        "physical_schedule_path": None if physical_schedule_path is None else str(physical_schedule_path),
        "trace_path": str(trace_path),
        "latent_path": str(latent_path),
        "load_time_sec": load_time_sec,
        "generation_time_sec": generation_time,
        "denoise_time_sec": float(result.metadata["denoise_time_sec"]),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1.0e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1.0e9,
        "finite": True,
        "latent_shape": list(final_latents.shape),
        "latent_dtype": str(final_latents.dtype),
        "latent_frame_dim": 2,
        "cfg_schedule_audit": cfg_audit,
        "config": config.to_dict(),
    }
    _write_json(summary_path, summary)
    return final_latents, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one prompt's complete frozen cross-step endpoint screen.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("cross-step endpoint Wan run requires a CUDA GPU")
    if (args.height, args.width, args.num_frames, args.steps, args.warmup_steps, args.seed) != (480, 832, 81, 50, 5, 0):
        raise ValueError("scientific run requires 480x832, 81 frames, 50 steps, warmup=5, seed=0")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    prompt_contract = {str(item["prompt_id"]): str(item["text"]) for item in plan.get("prompts", [])}
    if set(prompt_contract) != {f"p{index}_s0" for index in range(8)}:
        raise ValueError("protocol must contain exactly p0_s0..p7_s0")
    if args.prompt_id not in prompt_contract or args.prompt != prompt_contract[args.prompt_id]:
        raise ValueError("prompt-id/text differ from the frozen protocol")
    if plan.get("seed") != args.seed:
        raise ValueError("seed differs from the frozen protocol")
    runtime_contract = plan.get("runtime_contract", {})
    expected_runtime = {
        "height": args.height,
        "width": args.width,
        "decoded_frames": args.num_frames,
        "denoising_steps": args.steps,
        "dense_warmup_steps": args.warmup_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
    }
    if any(runtime_contract.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("CLI sampler arguments differ from runtime_contract")
    model_path = Path(args.model_id)
    if not args.local_files_only or not model_path.is_dir():
        raise ValueError("scientific run requires --local-files-only and an existing local model directory")
    root = args.output_root / args.prompt_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = build_physical_runs(plan)
    protocol_manifest_path = Path(__file__).resolve().parents[1] / "configs" / "CROSS_STEP_ENDPOINT_PROTOCOL.sha256"
    manifest["plan_file_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    protocol_sha = hashlib.sha256(protocol_manifest_path.read_bytes()).hexdigest()
    manifest["protocol_manifest_sha256"] = protocol_sha
    manifest["protocol_files_sha256"] = {
        relative: digest
        for digest, relative in (
            line.split(maxsplit=1)
            for line in protocol_manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    validate_runtime_manifest(manifest)
    _write_json(root / "runtime_manifest.json", manifest)
    _write_json(root / "logical_arm_map.json", manifest["logical_arm_map"])
    status = {
        "schema_version": "coframe.cross-step-endpoint-run-status.v1",
        "status": "running",
        "prompt_id": args.prompt_id,
        "completed": [],
        "failed": [],
    }
    _write_json(root / "run_status.json", status)

    original_excepthook = sys.excepthook

    def record_unhandled(error_type: type[BaseException], error: BaseException, traceback: Any) -> None:
        if status.get("status") != "failed":
            status["status"] = "failed"
            status["failed"].append({"error_type": error_type.__name__, "message": str(error)})
            _write_json(root / "run_status.json", status)
        original_excepthook(error_type, error, traceback)

    sys.excepthook = record_unhandled

    from diffusers import AutoencoderKLWan, WanPipeline

    local_only = bool(args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1")
    load_started = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_only,
    )
    pipe = WanPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
    )
    pipe.scheduler = _scheduler(args.flow_shift)
    pipe.to("cuda")
    load_time = time.perf_counter() - load_started

    source_commit = _git_commit(Path(__file__).resolve().parents[1])
    source_tree = _git_tree(Path(__file__).resolve().parents[1])
    model_fp = _model_fingerprint(args.model_id)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    initial_latents = prepare_initial_latents(
        pipe,
        batch_size=1,
        num_videos_per_prompt=1,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        generator=generator,
    ).detach().float()
    if not bool(torch.isfinite(initial_latents).all().item()):
        raise RuntimeError("initial latent contains NaN or Inf")
    initial_sha = _tensor_sha256(initial_latents)
    _write_json(
        root / "source.json",
        {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "model_id": args.model_id,
            "model_fingerprint": model_fp,
            "initial_latent_sha256": initial_sha,
            "plan_sha256": manifest["plan_sha256"],
            "plan_file_sha256": manifest["plan_file_sha256"],
            "protocol_sha256": protocol_sha,
            "protocol_files_sha256": manifest["protocol_files_sha256"],
            "runtime_invariants": _runtime_invariants(args),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "assigned_physical_gpus": runtime_contract.get("assigned_physical_gpus"),
        },
    )

    outputs: dict[str, torch.Tensor] = {}
    summaries: dict[str, Any] = {}
    try:
        _, dense_summary = _save_run(
            pipe=pipe,
            args=args,
            root=root,
            run_id=manifest["dense_run_id"],
            kind="dense",
            initial_latents=initial_latents,
            schedule=None,
            schedule_sha256=None,
            source_commit=source_commit,
            source_tree=source_tree,
            model_fingerprint=model_fp,
            plan_sha256=manifest["plan_sha256"],
            protocol_sha256=protocol_sha,
            load_time_sec=load_time,
        )
        status["completed"].append(dense_summary["run_id"])
        _write_json(root / "run_status.json", status)
        for item in manifest["physical_runs"]:
            output, summary = _save_run(
                pipe=pipe,
                args=args,
                root=root,
                run_id=item["run_id"],
                kind="sparse",
                initial_latents=initial_latents,
                schedule=item["schedule"],
                schedule_sha256=item["schedule_sha256"],
                source_commit=source_commit,
                source_tree=source_tree,
                model_fingerprint=model_fp,
                plan_sha256=manifest["plan_sha256"],
                protocol_sha256=protocol_sha,
                load_time_sec=load_time,
            )
            outputs[item["run_id"]] = output
            summaries[item["run_id"]] = summary
            status["completed"].append(item["run_id"])
            _write_json(root / "run_status.json", status)

        baseline = manifest["baseline_run_id"]
        baseline_item = next(item for item in manifest["physical_runs"] if item["run_id"] == baseline)
        parity_output, parity_summary = _save_run(
            pipe=pipe,
            args=args,
            root=root,
            run_id=manifest["parity_repeat_run_id"],
            kind="parity_repeat",
            initial_latents=initial_latents,
            schedule=baseline_item["schedule"],
            schedule_sha256=baseline_item["schedule_sha256"],
            source_commit=source_commit,
            source_tree=source_tree,
            model_fingerprint=model_fp,
            plan_sha256=manifest["plan_sha256"],
            protocol_sha256=protocol_sha,
            load_time_sec=load_time,
        )
        status["completed"].append(parity_summary["run_id"])
        parity_equal = torch.equal(outputs[baseline], parity_output)
        parity_max_abs = float((outputs[baseline] - parity_output).abs().max().item())
        if not parity_equal or parity_max_abs != 0.0:
            raise RuntimeError(f"K9 parity repeat failed: equal={parity_equal} max_abs={parity_max_abs}")
        fingerprints = {summary["source_fingerprint"] for summary in summaries.values()}
        fingerprints.add(parity_summary["source_fingerprint"])
        fingerprints.add(dense_summary["source_fingerprint"])
        if len(fingerprints) != 1:
            raise RuntimeError("source fingerprints differ across trajectories")
        status.update(
            {
                "status": "success",
                "parity": {
                    "baseline_run_id": baseline,
                    "repeat_run_id": parity_summary["run_id"],
                    "torch_equal": True,
                    "max_abs_difference": 0.0,
                },
                "source_fingerprint": next(iter(fingerprints)),
                "physical_sparse_run_count": 15,
                "dense_run_count": 1,
                "parity_repeat_run_count": 1,
            }
        )
    except Exception as error:
        status["status"] = "failed"
        status["failed"].append({"error_type": type(error).__name__, "message": str(error)})
        _write_json(root / "run_status.json", status)
        raise
    _write_json(root / "run_status.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

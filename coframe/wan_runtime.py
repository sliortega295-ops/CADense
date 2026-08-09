"""Shared Wan2.1 runtime helpers with no import-time dependency on Wan."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_unipc_scheduler(
    config: Any,
    device: torch.device,
    *,
    steps: int,
    shift: float,
) -> Any:
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=device, shift=shift)
    return scheduler


def target_geometry(
    pipeline: Any,
    size: tuple[int, int],
    frame_num: int,
) -> tuple[tuple[int, ...], int, int, int]:
    latent_frames = (frame_num - 1) // pipeline.vae_stride[0] + 1
    latent_height = size[1] // pipeline.vae_stride[1]
    latent_width = size[0] // pipeline.vae_stride[2]
    shape = (pipeline.vae.model.z_dim, latent_frames, latent_height, latent_width)
    tokens_per_frame = math.ceil(
        latent_height
        * latent_width
        / (pipeline.patch_size[1] * pipeline.patch_size[2])
    )
    sequence_length = tokens_per_frame * latent_frames
    return shape, sequence_length, latent_frames, tokens_per_frame


def scheduler_step(
    scheduler: Any,
    prediction: torch.Tensor,
    timestep: torch.Tensor,
    latent: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return scheduler.step(
        prediction.unsqueeze(0),
        timestep,
        latent.unsqueeze(0),
        return_dict=False,
        generator=generator,
    )[0].squeeze(0)


def cfg_combine(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guide_scale: float,
) -> torch.Tensor:
    return unconditional + float(guide_scale) * (conditional - unconditional)


def parse_int_list(value: str) -> list[int]:
    output: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"invalid range: {part}")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 else 1
            output.extend(range(start, stop, step))
        else:
            output.append(int(part))
    return sorted(set(output))

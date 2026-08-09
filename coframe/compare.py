from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "latents" not in payload:
        raise ValueError(f"{path} is not a CoFrame latent payload")
    return payload


def _per_frame_error(reference: torch.Tensor, approximation: torch.Tensor) -> torch.Tensor:
    if reference.shape != approximation.shape or reference.ndim != 5:
        raise ValueError("Latents must share [B,C,F,H,W] shape")
    reference = reference.float().permute(0, 2, 1, 3, 4)
    approximation = approximation.float().permute(0, 2, 1, 3, 4)
    dims = (0, 2, 3, 4)
    numerator = (reference - approximation).square().mean(dim=dims).sqrt()
    denominator = reference.square().mean(dim=dims).sqrt()
    return numerator / (denominator + 1.0e-8)


def _probe_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for event in metadata.get("transformer_events", []):
        probes.extend(event.get("probes", []))
    if not probes:
        return {
            "probe_count": 0,
            "probe_spearman_mean": None,
            "probe_pearson_mean": None,
            "probe_prior_spearman_mean": None,
            "probe_defect_spearman_mean": None,
            "probe_spearman_gain_mean": None,
            "probe_anchor_context_error_mean": None,
        }

    def mean_present(key: str) -> float | None:
        values = [float(item[key]) for item in probes if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "probe_count": len(probes),
        "probe_spearman_mean": mean_present("spearman"),
        "probe_pearson_mean": mean_present("pearson"),
        "probe_prior_spearman_mean": mean_present("prior_spearman"),
        "probe_defect_spearman_mean": mean_present("defect_spearman"),
        "probe_spearman_gain_mean": mean_present("spearman_gain_over_rhyme_prior"),
        "probe_anchor_context_error_mean": mean_present("anchor_context_error_mean"),
    }


def compare_payload(reference: dict[str, Any], candidate: dict[str, Any], name: str) -> dict[str, Any]:
    dense = reference["latents"].float()
    sparse = candidate["latents"].float()
    if dense.shape != sparse.shape:
        raise ValueError(f"Shape mismatch for {name}: {tuple(dense.shape)} vs {tuple(sparse.shape)}")

    relative_l2 = torch.linalg.vector_norm(sparse - dense) / (torch.linalg.vector_norm(dense) + 1.0e-8)
    cosine = F.cosine_similarity(dense.flatten(), sparse.flatten(), dim=0)
    frame_error = _per_frame_error(dense, sparse)
    dense_time = float(reference.get("metadata", {}).get("denoise_time_sec", 0.0))
    sparse_time = float(candidate.get("metadata", {}).get("denoise_time_sec", 0.0))

    result = {
        "method": name,
        "relative_l2": float(relative_l2.item()),
        "cosine": float(cosine.item()),
        "frame_error_mean": float(frame_error.mean().item()),
        "frame_error_max": float(frame_error.max().item()),
        "frame_error": frame_error.tolist(),
        "denoise_time_sec": sparse_time or None,
        "speedup_vs_dense": dense_time / sparse_time if dense_time > 0 and sparse_time > 0 else None,
        "initial_anchors": candidate.get("metadata", {}).get("initial_anchors"),
        "final_anchors": candidate.get("metadata", {}).get("final_anchors"),
        **_probe_summary(candidate.get("metadata", {})),
    }
    controller = candidate.get("metadata", {}).get("controller")
    result["refresh_count"] = len(controller.get("refresh_history", [])) if controller else 0
    result["accepted_refresh_count"] = (
        sum(1 for event in controller.get("refresh_history", []) if event.get("gain", 0.0) > 0) if controller else 0
    )
    return result


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["method", "speedup", "rel-L2", "cosine", "frame-mean", "frame-max", "probe-rho", "refresh"]
    print("  ".join(f"{header:>12}" for header in headers))
    for row in rows:
        speedup = row["speedup_vs_dense"]
        probe = row["probe_spearman_mean"]
        values = [
            row["method"],
            "-" if speedup is None else f"{speedup:.3f}x",
            f"{row['relative_l2']:.5f}",
            f"{row['cosine']:.5f}",
            f"{row['frame_error_mean']:.5f}",
            f"{row['frame_error_max']:.5f}",
            "-" if probe is None else f"{probe:.3f}",
            str(row["accepted_refresh_count"]),
        ]
        print("  ".join(f"{value:>12}" for value in values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare CoFrame latent runs against a dense Wan reference")
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "LATENTS_PT"), required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = _load(args.dense)
    rows = [compare_payload(reference, _load(Path(path)), name) for name, path in args.candidate]
    _print_table(rows)
    payload = {"dense": str(args.dense), "results": rows}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

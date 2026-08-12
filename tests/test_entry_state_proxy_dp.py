import pytest
import torch

from coframe.config import CoFrameConfig
from coframe.metrics import frame_gram_matrix, interpolation_interval_costs, optimal_piecewise_linear_mesh
from coframe.wan.sparse_forward import _entry_state_proxy_dp_mesh


def test_entry_state_proxy_dp_reuses_sketch_and_exact_dp():
    torch.manual_seed(7)
    frames = torch.randn(1, 9, 3, 6)
    projection = torch.randn(6, 3)

    proxy = _entry_state_proxy_dp_mesh(
        frames,
        projection,
        num_anchors=4,
        sketch_dim=3,
        min_gap=1,
        force_boundaries=True,
        chunk_size=16,
    )

    sketch = frames @ projection
    gram = frame_gram_matrix(sketch, chunk_size=16)
    direct = optimal_piecewise_linear_mesh(
        interpolation_interval_costs(gram),
        num_anchors=4,
        total_energy=float(torch.diagonal(gram).sum().item()),
    )
    assert proxy.anchors == direct.anchors
    assert proxy.squared_error == pytest.approx(direct.squared_error)
    assert proxy.anchors[0] == 0 and proxy.anchors[-1] == 8


def test_entry_state_proxy_dp_is_exact_for_linear_two_anchor_state():
    time = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    frames = torch.cat((time, 2.0 * time + 1.0), dim=-1)
    proxy = _entry_state_proxy_dp_mesh(
        frames,
        projection=None,
        num_anchors=2,
        sketch_dim=2,
        min_gap=1,
        force_boundaries=True,
        chunk_size=16,
    )
    assert proxy.anchors == [0, 6]
    assert proxy.normalized_mse == pytest.approx(0.0, abs=1.0e-7)


def test_entry_state_proxy_dp_screen_contract_is_fixed():
    CoFrameConfig(probe_entry_state_proxy_dp=True, refresh_signal="none").validate(
        num_blocks=30,
        num_frames=21,
    )
    with pytest.raises(ValueError, match="sketch_dim=64"):
        CoFrameConfig(probe_entry_state_proxy_dp=True, refresh_signal="none", sketch_dim=32).validate(
            num_blocks=30,
            num_frames=21,
        )
    with pytest.raises(ValueError, match="blocks 0-2"):
        CoFrameConfig(probe_entry_state_proxy_dp=True, refresh_signal="none", sparse_block_start=2).validate(
            num_blocks=30,
            num_frames=21,
        )
    with pytest.raises(ValueError, match="counterfactual-only"):
        CoFrameConfig(probe_entry_state_proxy_dp=True, refresh_signal="defect").validate(
            num_blocks=30,
            num_frames=21,
        )

import numpy as np
import pytest
import torch

from avsr.fusion import entropy_weights, fuse_log_linear, normalised_entropy, select_alpha
from avsr.models import build_model
from avsr.visual.features import (
    PCAProjector,
    block_dct_features,
    dct_features,
    dct_reconstruct,
    energy_compaction,
    polygon_area,
    resample,
    zigzag_indices,
)


def test_zigzag_matches_jpeg_order():
    r, c = zigzag_indices(4, 4)
    expected = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]
    assert list(zip(r[:10], c[:10])) == expected


def test_full_dct_reconstruction_is_lossless_and_energy_compacts():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:32, 0:64]
    img = (127 + 100 * np.sin(xx / 10.0) * np.cos(yy / 8.0) + rng.normal(0, 3, (32, 64))).clip(0, 255).astype(np.uint8)
    assert np.allclose(dct_reconstruct(img, 32 * 64), img / 255.0, atol=1e-5)
    assert energy_compaction(img[None])[44] > 0.6


def test_dct_feature_shapes():
    rois = np.random.default_rng(0).integers(0, 255, (5, 32, 64), dtype=np.uint8)
    assert dct_features(rois, 44).shape == (5, 44)
    assert dct_features(rois, 44, drop_dc=True).shape == (5, 44)
    assert block_dct_features(rois, 8, 3).shape == (5, 4 * 8 * 3)


def test_pca_projector_reduces_dimension():
    rois = np.random.default_rng(0).integers(0, 255, (50, 32, 64), dtype=np.uint8)
    p = PCAProjector(10).fit(rois)
    assert p.transform(rois).shape == (50, 10)
    assert np.all(np.diff(p.explained_variance_ratio_) <= 1e-9)


def test_polygon_area_unit_square():
    sq = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
    assert polygon_area(sq)[0] == pytest.approx(1.0)


def test_resample_handles_irregular_timestamps():
    t_src = np.array([0.0, 0.03, 0.1, 0.13, 0.2])      # variable frame rate with a dropped frame
    feat = np.stack([2 * t_src, -t_src], 1)
    t_dst = np.linspace(0, 0.2, 21)
    out = resample(feat, t_src, t_dst)
    assert np.allclose(out[:, 0], 2 * t_dst, atol=1e-6)
    out_c = resample(feat, t_src, t_dst, "cubic")
    assert np.allclose(out_c[:, 1], -t_dst, atol=1e-6)


def test_log_linear_fusion_limits_and_normalisation():
    rng = np.random.default_rng(0)
    pa, pv = rng.dirichlet(np.ones(20), 10), rng.dirichlet(np.ones(20), 10)
    assert np.allclose(fuse_log_linear(pa, pv, 1.0), pa, atol=1e-5)
    assert np.allclose(fuse_log_linear(pa, pv, 0.0), pv, atol=1e-5)
    assert np.allclose(fuse_log_linear(pa, pv, 0.3).sum(1), 1.0)


def test_entropy_weights_favour_the_confident_stream():
    confident = np.eye(20)[[0]] * 0.95 + 0.05 / 20
    flat = np.full((1, 20), 1 / 20)
    assert normalised_entropy(flat)[0] == pytest.approx(1.0)
    assert entropy_weights(confident, flat, 10, 0)[0] > 0.99    # trust audio when vision is flat
    assert entropy_weights(flat, confident, 10, 0)[0] < 0.01


def test_alpha_selection_prefers_informative_stream():
    y = np.arange(20)
    good = np.eye(20) * 0.9 + 0.1 / 20
    bad = np.roll(good, 1, axis=1)
    assert select_alpha([(good, bad)], [y]) > 0.5
    assert select_alpha([(bad, good)], [y]) < 0.5


@pytest.mark.parametrize("kind,shape", [("tcn", (4, 39, 379)), ("cnn2d", (4, 39, 379)), ("mlp", (4, 88, 114))])
def test_model_forward_shapes(kind, shape):
    model = build_model(kind, shape[1], 20)
    assert model(torch.randn(*shape)).shape == (4, 20)


def test_roicnn_forward_shape():
    model = build_model("roicnn", 0, 20)
    assert model(torch.rand(2, 114, 32, 64)).shape == (2, 20)

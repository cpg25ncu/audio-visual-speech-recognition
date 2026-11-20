"""Keras backend: shapes, parameter parity with PyTorch, augmentation, schedule, training loop."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")
torch = pytest.importorskip("torch")

from avsr.backends import get_backend  # noqa: E402
from avsr.keras_models import TemporalAugment, build_keras_model, trainable_parameters  # noqa: E402
from avsr.keras_training import OneCycle  # noqa: E402
from avsr.models import build_model, count_parameters  # noqa: E402
from avsr.training import TrainConfig  # noqa: E402


@pytest.mark.parametrize("kind,in_dim", [("tcn", 39), ("tcn", 141), ("cnn2d", 39), ("mlp", 102)])
def test_keras_and_torch_models_have_identical_trainable_parameter_counts(kind, in_dim):
    assert trainable_parameters(build_keras_model(kind, in_dim, 20)) == count_parameters(build_model(kind, in_dim, 20))


@pytest.mark.parametrize("kind", ["tcn", "cnn2d", "mlp"])
def test_keras_forward_shapes_and_variable_length(kind):
    model = build_keras_model(kind, 39, 20)
    for T in (114, 379):
        assert model(np.random.randn(3, T, 39).astype("float32"), training=False).shape == (3, 20)


def test_augmentation_is_identity_at_inference_and_masks_in_training():
    layer = TemporalAugment(time_shift=0.0, time_mask=0.2, feat_mask=0.2)
    x = np.ones((8, 100, 40), "float32")
    assert np.array_equal(layer(x, training=False).numpy(), x)
    y = layer(x, training=True).numpy()
    assert y.shape == x.shape and (y == 0).any() and (y == 1).mean() > 0.5


def test_one_cycle_matches_pytorch_schedule():
    total = 200
    ours = OneCycle(1e-3, total, pct_start=0.15)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.15,
                                                cycle_momentum=False)
    ref = []
    for _ in range(total):
        ref.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    mine = [float(ours(i)) for i in range(total)]
    assert np.allclose(mine, ref, rtol=1e-4, atol=1e-9)


def test_keras_backend_learns_a_separable_toy_problem():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 20)
    X = rng.normal(0, 1, (80, 8, 30)).astype("float32")
    X[np.arange(80), 2 * y, :] += 3.0                      # class k lights up features 2k and 2k+1,
    X[np.arange(80), 2 * y + 1, :] += 3.0                  # so one masked feature cannot hide it
    backend = get_backend("keras")
    model = backend.build("tcn", 8, 4, channels=16)
    cfg = TrainConfig(epochs=25, batch_size=16, patience=25, label_smoothing=0.0)
    model, hist = backend.fit(model, X, y, X, y, cfg, seed=0, n_classes=4)
    assert hist["best_val_acc"] >= 0.9
    assert backend.predict_proba(model, X).shape == (80, 4)

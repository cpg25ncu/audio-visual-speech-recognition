"""Framework backends: one interface, two implementations (PyTorch and Keras/TensorFlow).

Everything upstream (features, noise, splits) and downstream (scoring, late fusion, reports)
works on NumPy arrays and posterior probabilities, so it is framework-agnostic. A backend only
has to build a model, train it with the shared protocol, return class posteriors and save it.
Imports are lazy so that TensorFlow is loaded only when the Keras backend is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TorchBackend:
    name = "torch"

    def build(self, kind: str, in_dim: int, n_classes: int, **kw):
        from .models import build_model
        return build_model(kind, in_dim, n_classes, **kw)

    def fit(self, model, X_train, y_train, X_val, y_val, cfg, seed: int, n_classes: int):
        from .training import fit
        return fit(model, X_train, y_train, X_val, y_val, cfg, seed)

    def predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        from .training import predict_proba
        return predict_proba(model, X)

    def n_params(self, model) -> int:
        from .models import count_parameters
        return count_parameters(model)

    def save(self, model, path: Path, meta: dict) -> Path:
        import torch
        path = Path(path).with_suffix(".pt")
        torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
        return path


class KerasBackend:
    name = "keras"

    def build(self, kind: str, in_dim: int, n_classes: int, **kw):
        from .keras_models import build_keras_model
        return build_keras_model(kind, in_dim, n_classes, **kw)

    def fit(self, model, X_train, y_train, X_val, y_val, cfg, seed: int, n_classes: int):
        from .keras_training import fit
        return fit(model, X_train, y_train, X_val, y_val, cfg, seed, n_classes)

    def predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        from .keras_training import predict_proba
        return predict_proba(model, X)

    def n_params(self, model) -> int:
        from .keras_models import trainable_parameters
        return trainable_parameters(model)

    def save(self, model, path: Path, meta: dict) -> Path:
        path = Path(path).with_suffix(".keras")
        model.save(path)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return path


_BACKENDS = {"torch": TorchBackend, "keras": KerasBackend}


def get_backend(name: str):
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend '{name}' (choose from {sorted(_BACKENDS)})")
    return _BACKENDS[name]()

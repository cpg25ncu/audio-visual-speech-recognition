"""Keras training loop with the same protocol as ``training.py`` (PyTorch).

Idiomatic Keras 3 pieces instead of a hand-written loop:

* ``ClipDataset`` (a ``keras.utils.PyDataset``) shuffles every epoch and, for multi-condition
  training, draws one of the V pre-computed noisy versions of each clip;
* ``OneCycle`` is a ``LearningRateSchedule`` reproducing ``torch.optim.lr_scheduler.OneCycleLR``
  (cosine warm-up from max_lr/25 to max_lr over 15% of the steps, cosine decay to max_lr/25e4);
* ``SelectBest`` is a callback for early stopping on validation accuracy with ties broken by
  validation log-likelihood, restoring the best weights at the end, exactly like the PyTorch loop;
* AdamW (epsilon 1e-8 as in PyTorch) with decoupled weight decay, *global* gradient-norm clipping
  at 5 (``clip_grad_norm_`` semantics) and label smoothing 0.05.

Inputs follow the project convention ``(N, D, T)`` (or ``(N, V, D, T)`` for a feature bank) and
are transposed to Keras' channels-last ``(N, T, D)`` here.
"""

from __future__ import annotations

import math
import time

import keras
import numpy as np
import tensorflow as tf

from .training import TrainConfig


def to_channels_last(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.swapaxes(x, -1, -2))


@keras.saving.register_keras_serializable(package="avsr")
class OneCycle(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr: float, total_steps: int, pct_start: float = 0.15, div_factor: float = 25.0,
                 final_div_factor: float = 1e4):
        self.max_lr, self.total_steps, self.pct_start = max_lr, total_steps, pct_start
        self.div_factor, self.final_div_factor = div_factor, final_div_factor

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        initial = self.max_lr / self.div_factor
        final = initial / self.final_div_factor
        up_end = float(self.pct_start * self.total_steps) - 1.0
        down_end = float(self.total_steps) - 1.0

        def cos_anneal(start, end, frac):
            return end + (start - end) / 2.0 * (1.0 + tf.cos(math.pi * tf.clip_by_value(frac, 0.0, 1.0)))

        warm = cos_anneal(initial, self.max_lr, step / max(up_end, 1.0))
        decay = cos_anneal(self.max_lr, final, (step - up_end) / max(down_end - up_end, 1.0))
        return tf.where(step <= up_end, warm, decay)

    def get_config(self):
        return {"max_lr": self.max_lr, "total_steps": self.total_steps, "pct_start": self.pct_start,
                "div_factor": self.div_factor, "final_div_factor": self.final_div_factor}


class ClipDataset(keras.utils.PyDataset):
    """Mini-batches of (features, one-hot labels); re-shuffles and re-draws bank versions each epoch."""

    def __init__(self, X: np.ndarray, y: np.ndarray, n_classes: int, batch_size: int, bank: bool, seed: int, **kw):
        super().__init__(**kw)
        self.X, self.y, self.bank, self.batch_size = X, y, bank, batch_size
        self.onehot = np.eye(n_classes, dtype=np.float32)[y]
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.y))
        self.version = self.rng.integers(0, self.X.shape[1], len(self.y)) if self.bank else None

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        xb = self.X[idx, self.version[idx]] if self.bank else self.X[idx]
        return to_channels_last(xb.astype(np.float32)), self.onehot[idx]


class SelectBest(keras.callbacks.Callback):
    """Early stopping on validation accuracy, ties broken by lower validation NLL."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, patience: int):
        super().__init__()
        self.X_val, self.y_val, self.patience = X_val, y_val, patience
        self.history = {"val_acc": [], "val_nll": []}

    def on_train_begin(self, logs=None):
        self.best_acc, self.best_nll, self.best_epoch, self.waited, self.best_weights = -1.0, np.inf, 0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        p = predict_proba(self.model, self.X_val, channels_last=True)
        acc = float((p.argmax(1) == self.y_val).mean())
        nll = float(-np.log(p[np.arange(len(self.y_val)), self.y_val] + 1e-8).mean())
        self.history["val_acc"].append(acc)
        self.history["val_nll"].append(nll)
        better = acc > self.best_acc + 1e-9
        tie = abs(acc - self.best_acc) <= 1e-9 and nll < self.best_nll
        significant = better or (tie and nll < self.best_nll - 1e-3)
        if better or tie:
            self.best_acc, self.best_nll, self.best_epoch = acc, nll, epoch
            self.best_weights = self.model.get_weights()
        self.waited = 0 if significant else self.waited + 1
        if self.waited >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def predict_proba(model: keras.Model, X: np.ndarray, batch_size: int = 128, channels_last: bool = False) -> np.ndarray:
    Xc = X if channels_last else to_channels_last(X)
    logits = model.predict(Xc.astype(np.float32), batch_size=batch_size, verbose=0)
    return tf.nn.softmax(logits, axis=1).numpy()


def fit(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        cfg: TrainConfig, seed: int = 0, n_classes: int = 20) -> tuple[keras.Model, dict]:
    keras.utils.set_random_seed(seed)
    bank = bool(cfg.extra.get("bank", False))
    data = ClipDataset(X_train, y_train, n_classes, cfg.batch_size, bank, seed)
    schedule = OneCycle(cfg.lr, total_steps=cfg.epochs * len(data))
    optimizer = keras.optimizers.AdamW(schedule, weight_decay=cfg.weight_decay, epsilon=1e-8, global_clipnorm=5.0)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=cfg.label_smoothing),
                  metrics=["categorical_accuracy"])
    select = SelectBest(to_channels_last(X_val), y_val, cfg.patience)
    t0 = time.time()
    hist = model.fit(data, epochs=cfg.epochs, callbacks=[select], verbose=0)
    history = {"train_loss": [float(v) for v in hist.history["loss"]],
               "train_acc": [float(v) for v in hist.history["categorical_accuracy"]],
               **select.history, "best_val_acc": select.best_acc, "best_epoch": select.best_epoch,
               "seconds": round(time.time() - t0, 1)}
    return model, history

"""Keras 3 (TensorFlow backend) implementations of the recognisers in ``models.py``.

The architectures mirror the PyTorch ones layer for layer, so trainable-parameter counts are
identical (asserted in the tests). Differences between the frameworks are handled explicitly:

* layout - PyTorch convolves ``(batch, channels, time)``; Keras is channels-last, so the
  backend transposes inputs to ``(batch, time, channels)`` before they reach these models;
* batch norm - Keras' ``momentum`` is the weight of the *old* running statistic, PyTorch's
  ``momentum`` the weight of the new one: PyTorch 0.1 == Keras 0.9, and epsilon is set to 1e-5;
* initialisation - Keras defaults to Glorot weights and zero biases, PyTorch to
  U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for both; ``torch_init`` reproduces PyTorch's scheme;
* outputs - all models return logits; the loss uses ``from_logits=True``.

Data augmentation is a custom layer (``TemporalAugment``) that is active only when Keras calls
the model with ``training=True``, so the same model object is used for training and inference.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

BN = {"momentum": 0.9, "epsilon": 1e-5}


def torch_init(fan_in: int) -> dict:
    """Kernel and bias initialisers matching PyTorch's default for Linear/Conv layers."""
    bound = 1.0 / float(fan_in) ** 0.5
    return {"kernel_initializer": keras.initializers.VarianceScaling(1.0 / 3.0, "fan_in", "uniform"),
            "bias_initializer": keras.initializers.RandomUniform(-bound, bound)}


def conv1d(x, filters: int, kernel: int, name: str, dilation: int = 1):
    return layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation, name=name,
                         **torch_init(kernel * x.shape[-1]))(x)


def dense(x, units: int, name: str, activation=None):
    return layers.Dense(units, activation=activation, name=name, **torch_init(x.shape[-1]))(x)


@keras.saving.register_keras_serializable(package="avsr")
class TemporalAugment(layers.Layer):
    """SpecAugment-style augmentation for ``(batch, time, features)`` sequences.

    One random time shift (zero-filled) shared by the batch, then one time mask and one feature
    mask per example. Identity at inference time.
    """

    def __init__(self, time_shift: float = 0.1, time_mask: float = 0.1, feat_mask: float = 0.1, **kw):
        super().__init__(**kw)
        self.time_shift, self.time_mask, self.feat_mask = time_shift, time_mask, feat_mask

    def call(self, x, training=None):
        if not training:
            return x
        T, D = tf.shape(x)[1], tf.shape(x)[2]
        B = tf.shape(x)[0]
        if self.time_shift > 0:
            max_shift = tf.cast(tf.cast(T, tf.float32) * self.time_shift, tf.int32)
            s = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
            x = tf.roll(x, s, axis=1)
            t = tf.range(T)
            keep = tf.logical_or(tf.logical_and(s > 0, t >= s), tf.logical_and(s <= 0, t < T + s))
            x = x * tf.cast(keep, x.dtype)[None, :, None]
        x = self._mask(x, B, T, self.time_mask, axis=1)
        if self.feat_mask > 0:
            x = tf.cond(D > 4, lambda: self._mask(x, B, D, self.feat_mask, axis=2), lambda: x)
        return x

    @staticmethod
    def _mask(x, B, n, frac, axis):
        if frac <= 0:
            return x
        max_w = tf.maximum(1, tf.cast(tf.cast(n, tf.float32) * frac, tf.int32))
        w = tf.random.uniform([B], 0, max_w + 1, dtype=tf.int32)
        start = tf.cast(tf.random.uniform([B]) * tf.cast(tf.maximum(n - w, 1), tf.float32), tf.int32)
        pos = tf.range(n)[None, :]
        inside = tf.logical_and(pos >= start[:, None], pos < (start + w)[:, None])
        keep = tf.cast(tf.logical_not(inside), x.dtype)
        return x * (keep[:, :, None] if axis == 1 else keep[:, None, :])

    def get_config(self):
        return {**super().get_config(), "time_shift": self.time_shift, "time_mask": self.time_mask,
                "feat_mask": self.feat_mask}


def _temporal_block(x, c_out: int, kernel: int, dilation: int, dropout: float, name: str):
    skip = x if x.shape[-1] == c_out else conv1d(x, c_out, 1, f"{name}_skip")
    h = conv1d(x, c_out, kernel, f"{name}_conv", dilation)
    h = layers.BatchNormalization(**BN, name=f"{name}_bn")(h)
    h = layers.Dropout(dropout, name=f"{name}_drop")(layers.ReLU(name=f"{name}_relu")(h))
    return layers.Add(name=f"{name}_add")([h, skip])


def temporal_cnn(in_dim: int, n_classes: int, channels: int = 128, kernel: int = 5,
                 dilations=(1, 2, 4, 8), dropout: float = 0.2, input_dropout: float = 0.1,
                 augment: bool = True) -> keras.Model:
    """``(batch, time, in_dim)`` -> logits. Same topology as ``models.TemporalCNN``."""
    inp = keras.Input((None, in_dim), name="features")
    x = TemporalAugment(name="augment")(inp) if augment else inp
    x = layers.BatchNormalization(**BN, name="input_norm")(x)
    x = layers.Dropout(input_dropout, name="input_drop")(x)
    for i, d in enumerate(dilations):
        x = _temporal_block(x, channels, kernel, d, dropout, f"block{i}")
    pooled = layers.Concatenate(name="mean_max")([layers.GlobalAveragePooling1D(name="mean")(x),
                                                  layers.GlobalMaxPooling1D(name="max")(x)])
    h = layers.Dropout(0.3, name="head_drop")(dense(pooled, channels, "head_fc", "relu"))
    return keras.Model(inp, dense(h, n_classes, "logits"), name="temporal_cnn")


def spectro_temporal_cnn(in_dim: int, n_classes: int, dropout: float = 0.3, augment: bool = True) -> keras.Model:
    """2-D CNN over the (time x feature) 'image' - the coursework-style audio model."""
    inp = keras.Input((None, in_dim), name="features")
    x = TemporalAugment(name="augment")(inp) if augment else inp
    x = layers.Reshape((-1, in_dim, 1), name="as_image")(x)
    for i, c in enumerate((32, 64, 128)):
        x = layers.Conv2D(c, 3, padding="same", name=f"conv{i}", **torch_init(9 * x.shape[-1]))(x)
        x = layers.ReLU(name=f"relu{i}")(layers.BatchNormalization(**BN, name=f"bn{i}")(x))
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(layers.Dropout(dropout, name="drop")(x))
    h = layers.Dropout(dropout, name="head_drop")(dense(x, 128, "head_fc", "relu"))
    return keras.Model(inp, dense(h, n_classes, "logits"), name="spectro_temporal_cnn")


def mlp(in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.3, augment: bool = True) -> keras.Model:
    """Time-averaged features -> MLP (the original visual baseline)."""
    inp = keras.Input((None, in_dim), name="features")
    x = TemporalAugment(name="augment")(inp) if augment else inp
    x = layers.GlobalAveragePooling1D(name="time_mean")(x)
    x = layers.BatchNormalization(**BN, name="input_norm")(x)
    h = layers.Dropout(dropout, name="drop")(dense(x, hidden, "fc", "relu"))
    return keras.Model(inp, dense(h, n_classes, "logits"), name="mlp")


def build_keras_model(kind: str, in_dim: int, n_classes: int, **kw) -> keras.Model:
    builders = {"tcn": temporal_cnn, "cnn2d": spectro_temporal_cnn, "mlp": mlp}
    if kind not in builders:
        raise ValueError(f"model '{kind}' has no Keras implementation")
    return builders[kind](in_dim, n_classes, **kw)


def trainable_parameters(model: keras.Model) -> int:
    return int(sum(int(tf.size(w)) for w in model.trainable_weights))

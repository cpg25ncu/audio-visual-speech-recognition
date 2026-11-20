"""Model export and deployment benchmark for the multi-condition audio recogniser.

* PyTorch -> ONNX (opset 17), executed with ONNX Runtime; outputs are checked against PyTorch.
* Keras   -> TFLite in four flavours: float32, float16 weights, dynamic-range int8 (int8 weights,
  float activations) and full-integer int8 (weights and activations, calibrated on 100
  training clips); outputs are checked against Keras.

For every format: file size, accuracy on the clean and the 0 dB test sets (mean of the four
noise types), the largest logit difference to the source framework, and the median single-clip
CPU latency (batch 1, after warm-up).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import work_path


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _latency_ms(fn, x: np.ndarray, warmup: int = 5, runs: int = 30) -> float:
    for i in range(warmup):
        fn(x[i % len(x)][None])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(x[i % len(x)][None])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _tflite_runner(model_bytes: bytes):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def run_one(x1: np.ndarray) -> np.ndarray:            # x1: (1, T, D)
        interp.set_tensor(inp["index"], x1.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out["index"]).astype(np.float32)

    def run(X: np.ndarray) -> np.ndarray:
        return np.concatenate([run_one(X[i:i + 1]) for i in range(len(X))])

    return run_one, run


def benchmark_exports(runner) -> pd.DataFrame:
    import keras
    import onnxruntime as ort
    import tensorflow as tf
    import torch

    from . import keras_models  # noqa: F401  (registers the custom Keras layer for deserialisation)
    from .keras_training import to_channels_last
    from .models import build_model

    c, cfg = runner.corpus, runner.cfg
    mf = c.mfcc_config()
    yt = c.y_of("test")
    X_clean = c.audio_eval("test", "clean", "clean", mf)                       # (N, D, T)
    X_noisy = np.concatenate([c.audio_eval("test", k, 0, mf) for k in cfg["noise"]["types"]])
    y_noisy = np.tile(yt, len(cfg["noise"]["types"]))
    out_dir = work_path(cfg, "export", "_").parent
    models_dir = runner.models_dir
    rows = []

    def record(framework, fmt, path_or_bytes, logits_fn, one_fn, ref_logits=None, layout="ncw"):
        xc, xn = (X_clean, X_noisy) if layout == "ncw" else (to_channels_last(X_clean), to_channels_last(X_noisy))
        lc, ln = logits_fn(xc), logits_fn(xn)
        size = len(path_or_bytes) if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).stat().st_size
        rows.append({"framework": framework, "format": fmt, "size_kb": round(size / 1024, 1),
                     "accuracy_clean": float((lc.argmax(1) == yt).mean()),
                     "accuracy_0db": float((ln.argmax(1) == y_noisy).mean()),
                     "max_abs_logit_diff": None if ref_logits is None else float(np.abs(lc - ref_logits).max()),
                     "latency_ms": round(_latency_ms(one_fn, xc), 3)})
        return lc

    # ---------------- PyTorch eager and ONNX
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ckpt = torch.load(models_dir / "audio-tcn-multi.pt", map_location="cpu", weights_only=False)
    tm = build_model("tcn", ckpt["meta"]["in_dim"], len(c.labels))
    tm.load_state_dict(ckpt["state_dict"])
    tm.eval()

    @torch.no_grad()
    def torch_logits(X):
        return np.concatenate([tm(torch.as_tensor(X[i:i + 64])).numpy() for i in range(0, len(X), 64)])

    with torch.no_grad():
        ref_torch = record("PyTorch", "eager (CPU)", models_dir / "audio-tcn-multi.pt", torch_logits,
                           lambda x: tm(torch.as_tensor(x)))
    onnx_path = out_dir / "audio-tcn-multi.onnx"
    torch.onnx.export(tm, (torch.randn(1, *X_clean.shape[1:]),), str(onnx_path), input_names=["features"],
                      output_names=["logits"], dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def onnx_logits(X):
        return np.concatenate([sess.run(None, {"features": X[i:i + 64].astype(np.float32)})[0]
                               for i in range(0, len(X), 64)])

    record("PyTorch", "ONNX Runtime", onnx_path, onnx_logits,
           lambda x: sess.run(None, {"features": x.astype(np.float32)}), ref_torch)

    # ---------------- Keras and TFLite
    km = keras.saving.load_model(models_dir / "keras-audio-tcn-multi.keras", compile=False)

    def keras_logits(Xc):
        return km.predict(Xc.astype(np.float32), batch_size=64, verbose=0)

    serve = tf.function(lambda x: km(x, training=False),
                        input_signature=[tf.TensorSpec([1, X_clean.shape[2], X_clean.shape[1]], tf.float32)])
    ref_keras = record("Keras", "Keras, tf.function (CPU)", models_dir / "keras-audio-tcn-multi.keras", keras_logits,
                       lambda x: serve(tf.constant(x, tf.float32)), layout="nwc")
    T, D = X_clean.shape[2], X_clean.shape[1]
    # SavedModel with a fixed (1, T, D) serving signature; converting from it freezes the weights
    saved_dir = out_dir / "keras-audio-tcn-multi.savedmodel"
    archive = keras.export.ExportArchive()
    archive.track(km)
    archive.add_endpoint(name="serve", fn=lambda x: km(x, training=False),
                         input_signature=[tf.TensorSpec([1, T, D], tf.float32)])
    archive.write_out(str(saved_dir), verbose=False)
    calib = to_channels_last(c.audio_features([c.waves[i] for i in c.ids_of("train")[:100]], mf))

    def representative():
        for x in calib:
            yield [x[None].astype(np.float32)]

    variants = {
        "TFLite float32": {},
        "TFLite float16": {"opt": True, "types": [tf.float16]},
        "TFLite int8 (dynamic range)": {"opt": True},
        "TFLite int8 (full integer)": {"opt": True, "repr": True},
    }
    for name, v in variants.items():
        conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
        if v.get("opt"):
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if v.get("types"):
            conv.target_spec.supported_types = v["types"]
        if v.get("repr"):
            conv.representative_dataset = representative
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        blob = conv.convert()
        (out_dir / f"audio-tcn-multi-{name.split()[1]}{'-full' if v.get('repr') else ''}.tflite").write_bytes(blob)
        one, run_all = _tflite_runner(blob)
        record("Keras", name, blob, run_all, one, ref_keras, layout="nwc")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

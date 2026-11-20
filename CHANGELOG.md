# Changelog

## v1.0.0 — 2026-11-20

First public release.

**Pipeline**
- Time-aligned audio/video decoding: per-frame timestamps for variable-frame-rate video, AAC start offset (mean 272 ms) re-inserted.
- Hand-built MFCC front-end (pre-emphasis, framing, Hamming window, power spectrum, mel/linear × triangular/rectangular filterbanks, DCT, log-energy, Δ/ΔΔ, CMVN), unit-tested against librosa.
- Noise at active-speech-level SNR (white, pink, babble, ambient) with disjoint training/evaluation noise; spectral subtraction; multi-condition training.
- MediaPipe face-mesh mouth tracking (Haar baseline), 2D-DCT / block-DCT / PCA / lip-shape features, ROI-CNN; controlled video degradation.
- Temporal-CNN family for audio, visual and early-fusion inputs; log-linear late fusion with global, oracle and entropy-adaptive weights.

**Two frameworks**
- PyTorch (default) and Keras 3 / TensorFlow backends behind one interface (`--backend torch|keras`), with identical architectures, trainable-parameter counts, schedule, augmentation and model selection.
- Framework-parity experiment and cross-framework late fusion.
- Export to ONNX (PyTorch) and TFLite float32 / float16 / int8 (Keras) with accuracy, size and latency benchmark.

**Evaluation**
- Fixed 18/6/6 split, 5 seeds, deterministic corrupted test inputs.
- Audio front-end ablation, visual front-end comparison, DCT-coefficient sweep, recording-session confound probes, fusion under acoustic noise × unseen video degradation.

**Engineering**
- `avsr prepare | run | report | predict` CLI, YAML configuration, generated figures and tables, unit tests, GitHub Actions CI (PyTorch and Keras jobs).

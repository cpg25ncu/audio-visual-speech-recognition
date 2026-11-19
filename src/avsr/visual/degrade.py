"""Controlled degradation of the visual stream.

On clean, close-up, frontal video the lip reader is near-perfect for this small vocabulary, so a
fusion study would only ever say "trust the video". Real video is often worse: low resolution
(the speaker is far from the camera), sensor noise in poor light and less precise landmarks.
We simulate one "visual quality" axis with three coupled effects, applied at test time to the
tracked mouth data:

* resolution - the ROI is down-sampled by ``factor`` and up-sampled back (lost detail);
* sensor noise - additive Gaussian pixel noise with std ``noise`` (fraction of full scale);
* tracking error - i.i.d. Gaussian jitter on every lip landmark, ``jitter`` ROI-pixels std
  (one ROI pixel = crop width / 64 original pixels), as a landmark detector would produce on a
  degraded frame.
"""

from __future__ import annotations

import cv2
import numpy as np

LEVELS: dict[int, dict | None] = {
    0: None,
    1: {"factor": 4, "noise": 0.06, "jitter": 1.5},
    2: {"factor": 8, "noise": 0.12, "jitter": 3.0},
    3: {"factor": 16, "noise": 0.20, "jitter": 6.0},
}
LEVEL_NAMES = {0: "clean video", 1: "mild", 2: "moderate", 3: "severe"}


def degrade_track(track: dict, level: int, rng: np.random.Generator, roi_scale: float = 1.8) -> dict:
    spec = LEVELS[level]
    if spec is None:
        return track
    out = dict(track)
    rois = track["roi"]
    T, H, W = rois.shape
    small = (max(1, W // spec["factor"]), max(1, H // spec["factor"]))
    deg = np.empty_like(rois)
    for i in range(T):
        lo = cv2.resize(rois[i], small, interpolation=cv2.INTER_AREA)
        deg[i] = cv2.resize(lo, (W, H), interpolation=cv2.INTER_LINEAR)
    noisy = deg.astype(np.float32) + rng.normal(0, spec["noise"] * 255, deg.shape)
    out["roi"] = np.clip(noisy, 0, 255).astype(np.uint8)
    if "lips" in track:
        px_per_roi_px = roi_scale * float(track["mouth_width_px"]) / W
        out["lips"] = track["lips"] + rng.normal(0, spec["jitter"] * px_per_roi_px, track["lips"].shape).astype(np.float32)
    return out

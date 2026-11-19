"""Energy-based speech endpoint detection on *clean* audio.

Used only for analysis (locating the word to build speech-removed control inputs for the
confound study, and plotting); recognition itself never uses endpoints, because an endpoint
detector run on noisy audio would fail exactly where it matters.
"""

from __future__ import annotations

import numpy as np


def speech_endpoints(x: np.ndarray, sample_rate: int = 16000, frame_ms: float = 10.0,
                     threshold_db: float = 25.0) -> tuple[float, float]:
    """(onset, offset) in seconds: first/last frame within ``threshold_db`` of the loudest frame."""
    n = int(sample_rate * frame_ms / 1000)
    usable = len(x) // n * n
    e = 10 * np.log10(np.mean(x[:usable].reshape(-1, n).astype(np.float64) ** 2, axis=1) + 1e-12)
    active = np.flatnonzero(e > e.max() - threshold_db)
    return active[0] * frame_ms / 1000, (active[-1] + 1) * frame_ms / 1000

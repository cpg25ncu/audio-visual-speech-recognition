"""Power spectral subtraction (Berouti et al., 1979) applied in the feature domain.

The noise power spectrum is estimated without any knowledge of where the word is: we average
the spectra of the lowest-energy frames of the utterance (a crude minimum-statistics estimate).
Subtraction uses an SNR-dependent over-subtraction factor and a spectral floor to limit
"musical noise". Because the recogniser only needs filterbank energies, the enhanced power
spectrum is passed straight to the filterbank - no waveform resynthesis is required.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-10


def estimate_noise_psd(pspec: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    energy = pspec.sum(axis=1)
    k = max(1, int(round(quantile * len(energy))))
    quietest = np.argsort(energy)[:k]
    return pspec[quietest].mean(axis=0)


def spectral_subtraction(pspec: np.ndarray, noise_psd: np.ndarray | None = None,
                         floor: float = 0.02, alpha_min: float = 1.0, alpha_max: float = 4.0) -> np.ndarray:
    """Return the enhanced power spectrum ``(n_frames, n_bins)``.

    alpha (over-subtraction) falls linearly from ``alpha_max`` at -5 dB frame SNR to
    ``alpha_min`` at +20 dB: subtract aggressively where the frame is noise-dominated.
    """
    noise = estimate_noise_psd(pspec) if noise_psd is None else noise_psd
    frame_snr = 10 * np.log10(pspec.sum(1) / (noise.sum() + EPS) + EPS)
    alpha = np.interp(frame_snr, [-5.0, 20.0], [alpha_max, alpha_min])[:, None]
    cleaned = pspec - alpha * noise[None, :]
    return np.maximum(cleaned, floor * noise[None, :]).astype(np.float32)

"""Filterbank-derived cepstral features implemented from first principles with NumPy.

Pipeline (each step is a separate function so it can be tested and visualised on its own):

    waveform -> pre-emphasis -> framing (25 ms / 10 ms) -> Hamming window -> |FFT|^2
             -> filterbank (mel|linear, triangular|rectangular) -> log -> DCT-II -> truncate
             -> [replace c0 by log-energy] -> [+ delta, + delta-delta] -> [utterance CMVN]

Why each step exists is discussed in docs/technical_report.md; the short version:
* speech is only quasi-stationary, hence short overlapping frames;
* the window trades main-lobe width for side-lobe leakage;
* the mel filterbank mimics the ear's coarser resolution at high frequencies and smooths away
  pitch harmonics, keeping the vocal-tract envelope (formants);
* the log turns the source-filter product into a sum and compresses dynamic range;
* the DCT decorrelates neighbouring (overlapping) channels and compacts the envelope into the
  first few coefficients, so truncation discards fine spectral detail that varies with pitch;
* deltas add the trajectory information that a frame-wise representation lacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dct

EPS = 1e-10


# --------------------------------------------------------------------------- framing & spectra
def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """First-order high-pass y[n] = x[n] - a x[n-1]; boosts the ~-6 dB/octave spectral tilt of voiced speech."""
    if coeff <= 0:
        return x.astype(np.float32)
    return np.append(x[0], x[1:] - coeff * x[:-1]).astype(np.float32)


def num_frames(n_samples: int, frame_len: int, hop: int) -> int:
    if n_samples <= frame_len:
        return 1
    return 1 + int(np.ceil((n_samples - frame_len) / hop))


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split into overlapping frames ``(n_frames, frame_len)``; the tail is zero-padded."""
    n = num_frames(len(x), frame_len, hop)
    total = (n - 1) * hop + frame_len
    x = np.pad(x, (0, max(0, total - len(x))))
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def frame_times(n_frames: int, frame_len: int, hop: int, sample_rate: int) -> np.ndarray:
    """Centre time (s) of each analysis frame - used to align video features to audio frames."""
    return (np.arange(n_frames) * hop + frame_len / 2) / sample_rate


def power_spectrum(frames: np.ndarray, n_fft: int, window: str = "hamming") -> np.ndarray:
    win = {"hamming": np.hamming, "hann": np.hanning, "rect": np.ones}[window](frames.shape[1])
    spec = np.fft.rfft(frames * win[None, :], n=n_fft, axis=1)
    return (np.abs(spec) ** 2).astype(np.float32)


# --------------------------------------------------------------------------- filterbanks
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def filter_edges(n_filters: int, f_min: float, f_max: float, scale: str = "mel") -> np.ndarray:
    """``n_filters + 2`` equally spaced points on the chosen scale, returned in Hz.

    Filter m spans edges[m] .. edges[m+2] and peaks at edges[m+1].
    """
    if scale == "mel":
        return mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_filters + 2))
    if scale == "linear":
        return np.linspace(f_min, f_max, n_filters + 2)
    raise ValueError(f"unknown filterbank scale '{scale}'")


def filterbank(sample_rate: int, n_fft: int, n_filters: int = 40, f_min: float = 0.0,
               f_max: float | None = None, scale: str = "mel", shape: str = "triangular") -> np.ndarray:
    """Filterbank matrix ``(n_filters, n_fft // 2 + 1)`` applied to a power spectrum.

    * triangular: overlapping triangles, weight 1 at the centre frequency (HTK convention,
      matches ``librosa.filters.mel(htk=True, norm=None)``);
    * rectangular: contiguous, non-overlapping bands whose boundaries lie half-way between
      neighbouring centres - the simplest filterbank, useful as an ablation baseline.
    """
    f_max = sample_rate / 2 if f_max is None else f_max
    edges = filter_edges(n_filters, f_min, f_max, scale)
    freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    fb = np.zeros((n_filters, len(freqs)), dtype=np.float32)
    for m in range(n_filters):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        if shape == "triangular":
            rising = (freqs - lo) / max(c - lo, EPS)
            falling = (hi - freqs) / max(hi - c, EPS)
            fb[m] = np.maximum(0.0, np.minimum(rising, falling))
        elif shape == "rectangular":
            band_lo, band_hi = (lo + c) / 2, (c + hi) / 2
            fb[m] = (freqs >= band_lo) & (freqs < band_hi)
            if not fb[m].any():  # very narrow low-frequency band: take the nearest bin
                fb[m, np.argmin(np.abs(freqs - c))] = 1.0
        else:
            raise ValueError(f"unknown filter shape '{shape}'")
    return fb


# --------------------------------------------------------------------------- cepstra & dynamics
def log_energies(pspec: np.ndarray, fb: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(pspec @ fb.T, EPS)).astype(np.float32)


def cepstra(log_fbank: np.ndarray, n_ceps: int) -> np.ndarray:
    """DCT-II (orthonormal) of log filterbank energies, truncated to ``n_ceps`` coefficients."""
    return dct(log_fbank, type=2, axis=1, norm="ortho")[:, :n_ceps].astype(np.float32)


def deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression-based time derivative (HTK formula) with edge replication.

    d_t = sum_{n=1..N} n (c_{t+n} - c_{t-n}) / (2 sum n^2)
    """
    if width < 1:
        raise ValueError("delta width must be >= 1")
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    T = feat.shape[0]
    num = sum(n * (padded[width + n:width + n + T] - padded[width - n:width - n + T])
              for n in range(1, width + 1))
    return (num / (2 * sum(n * n for n in range(1, width + 1)))).astype(np.float32)


def cmvn(feat: np.ndarray) -> np.ndarray:
    """Per-utterance cepstral mean and variance normalisation.

    Subtracting the mean removes stationary convolutional effects (microphone, room); this is
    what makes the model transfer to the earlier microphone session.
    """
    return ((feat - feat.mean(0, keepdims=True)) / (feat.std(0, keepdims=True) + 1e-5)).astype(np.float32)


# --------------------------------------------------------------------------- extractor
@dataclass
class MFCCConfig:
    sample_rate: int = 16000
    pre_emphasis: float = 0.97
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    scale: str = "mel"
    shape: str = "triangular"
    n_filters: int = 40
    f_min: float = 0.0
    f_max: float = 8000.0
    n_ceps: int = 13
    use_energy: bool = True
    deltas: int = 2
    delta_window: int = 2
    cmvn: str = "utterance"
    enhance: str = "none"          # none | spectral_subtraction
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, audio_cfg: dict, **overrides) -> MFCCConfig:
        fbc = audio_cfg.get("filterbank", {})
        kw = dict(sample_rate=audio_cfg["sample_rate"], pre_emphasis=audio_cfg["pre_emphasis"],
                  frame_ms=audio_cfg["frame_ms"], hop_ms=audio_cfg["hop_ms"], n_fft=audio_cfg["n_fft"],
                  scale=fbc.get("scale", "mel"), shape=fbc.get("shape", "triangular"),
                  n_filters=fbc.get("n_filters", 40), f_min=fbc.get("f_min", 0),
                  f_max=fbc.get("f_max", audio_cfg["sample_rate"] / 2), n_ceps=audio_cfg["n_ceps"],
                  use_energy=audio_cfg["use_energy"], deltas=audio_cfg["deltas"],
                  delta_window=audio_cfg["delta_window"], cmvn=audio_cfg["cmvn"],
                  enhance=audio_cfg.get("enhance", "none"))
        kw.update(overrides)
        return cls(**kw)

    @property
    def frame_len(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000))

    @property
    def hop(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000))

    @property
    def dim(self) -> int:
        return self.n_ceps * (1 + self.deltas)


class MFCCExtractor:
    """Callable feature extractor; the filterbank matrix is built once."""

    def __init__(self, cfg: MFCCConfig):
        self.cfg = cfg
        self.fb = filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_filters, cfg.f_min, cfg.f_max,
                             cfg.scale, cfg.shape)

    def frame_times(self, n_frames: int) -> np.ndarray:
        return frame_times(n_frames, self.cfg.frame_len, self.cfg.hop, self.cfg.sample_rate)

    def spectrum(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = frame_signal(pre_emphasis(x, self.cfg.pre_emphasis), self.cfg.frame_len, self.cfg.hop)
        return frames, power_spectrum(frames, self.cfg.n_fft)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c = self.cfg
        frames, pspec = self.spectrum(x)
        if c.enhance == "spectral_subtraction":
            from .enhance import spectral_subtraction
            pspec = spectral_subtraction(pspec)
        elif c.enhance != "none":
            raise ValueError(f"unknown enhancement '{c.enhance}'")
        feat = cepstra(log_energies(pspec, self.fb), c.n_ceps)
        if c.use_energy:
            if c.enhance == "none":
                energy = np.sum(frames.astype(np.float64) ** 2, axis=1)
            else:  # energy of the enhanced spectrum (Parseval, up to a constant)
                energy = pspec.sum(axis=1) / c.n_fft
            feat[:, 0] = np.log(np.maximum(energy, EPS))
        parts = [feat]
        for _ in range(c.deltas):
            parts.append(deltas(parts[-1], c.delta_window))
        out = np.concatenate(parts, axis=1)
        return cmvn(out) if c.cmvn == "utterance" else out.astype(np.float32)

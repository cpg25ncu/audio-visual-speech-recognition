"""Noise sources and SNR-controlled mixing.

SNR definition. Each clip is ~3.8 s long but the name occupies well under a second of it, so an
SNR computed over the whole clip would overstate the noise level the listener experiences by
~5 dB. We therefore use the *active speech level*: the power of the clean signal averaged only
over frames within 30 dB of the loudest frame (in the spirit of ITU-T P.56). Noise power is
averaged over the whole segment that is added.

Noise types
* ``white``  - flat spectrum (synthetic).
* ``pink``   - 1/f spectrum, more low-frequency energy like most real noise (synthetic).
* ``babble`` - sum of several other utterances of the corpus, i.e. competing speech, the
               hardest case for an audio-only recogniser because it is speech-shaped. With a
               single speaker this is "same-talker babble"; built from disjoint utterance pools for
               training and evaluation so test babble is never seen during training.
* ``ambient``- a 22 s room recording made with the original microphone; disjoint time ranges
               are used for training and evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-12


def active_power(x: np.ndarray, sample_rate: int = 16000, frame_ms: float = 20.0,
                 threshold_db: float = 30.0) -> float:
    n = max(1, int(sample_rate * frame_ms / 1000))
    usable = len(x) // n * n
    if usable == 0:
        return float(np.mean(x ** 2) + EPS)
    p = np.mean(x[:usable].reshape(-1, n).astype(np.float64) ** 2, axis=1)
    p_db = 10 * np.log10(p + EPS)
    active = p[p_db >= p_db.max() - threshold_db]
    return float(active.mean() + EPS)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, sample_rate: int = 16000) -> np.ndarray:
    """Scale ``noise`` so that active-speech-level SNR equals ``snr_db`` and add it."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)].astype(np.float64)
    gain = np.sqrt(active_power(clean, sample_rate) / ((np.mean(noise ** 2) + EPS) * 10 ** (snr_db / 10)))
    return (clean + gain * noise).astype(np.float32)


def measured_snr(clean: np.ndarray, noisy: np.ndarray, sample_rate: int = 16000) -> float:
    noise = noisy.astype(np.float64) - clean
    return 10 * np.log10(active_power(clean, sample_rate) / (np.mean(noise ** 2) + EPS))


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Shape white noise by 1/sqrt(f) in the frequency domain (power ~ 1/f)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n=n)
    return (x / (x.std() + EPS)).astype(np.float32)


def babble_noise(n: int, rng: np.random.Generator, pool: Sequence[np.ndarray], talkers: int = 6,
                 exclude: Sequence[int] = ()) -> np.ndarray:
    """Overlap ``talkers`` utterances, each circularly shifted and normalised to equal active level."""
    candidates = np.setdiff1d(np.arange(len(pool)), np.asarray(exclude, dtype=int))
    if len(candidates) < talkers:
        raise ValueError("babble pool is smaller than the number of talkers")
    out = np.zeros(n, dtype=np.float64)
    for i in rng.choice(candidates, size=talkers, replace=False):
        u = np.resize(pool[i], n).astype(np.float64)
        u = np.roll(u, rng.integers(n))
        out += u / np.sqrt(active_power(u.astype(np.float32)))
    return (out / (out.std() + EPS)).astype(np.float32)


class NoiseBank:
    """Draws noise segments for a role (``train`` or ``eval``) with disjoint material per role."""

    def __init__(self, ambient: np.ndarray | None, babble_pools: dict[str, dict[str, np.ndarray]],
                 ambient_train_fraction: float = 0.6, talkers: int = 6):
        """``babble_pools[role]`` maps clip id -> waveform; a clip is never part of its own babble."""
        self.babble_pools = {role: (list(p.keys()), list(p.values())) for role, p in babble_pools.items()}
        self.talkers = talkers
        self.ambient = {}
        if ambient is not None and len(ambient):
            cut = int(len(ambient) * ambient_train_fraction)
            self.ambient = {"train": ambient[:cut], "eval": ambient[cut:]}

    @property
    def available(self) -> list[str]:
        types = ["white", "pink"]
        if self.babble_pools:
            types.append("babble")
        if self.ambient:
            types.append("ambient")
        return types

    def sample(self, kind: str, n: int, rng: np.random.Generator, role: str = "eval",
               exclude_id: str | None = None) -> np.ndarray:
        if kind == "white":
            return white_noise(n, rng)
        if kind == "pink":
            return pink_noise(n, rng)
        if kind == "babble":
            ids, waves = self.babble_pools[role]
            exclude = [ids.index(exclude_id)] if exclude_id in ids else []
            return babble_noise(n, rng, waves, self.talkers, exclude)
        if kind == "ambient":
            src = self.ambient[role]
            if len(src) <= n:
                return np.resize(src, n).astype(np.float32)
            start = rng.integers(0, len(src) - n)
            return src[start:start + n].astype(np.float32)
        raise ValueError(f"unknown noise type '{kind}'")

    def corrupt(self, clean: np.ndarray, kind: str, snr_db, rng: np.random.Generator,
                role: str = "eval", sample_rate: int = 16000, exclude_id: str | None = None) -> np.ndarray:
        if snr_db in (None, "clean") or kind in (None, "clean"):
            return clean
        noise = self.sample(kind, len(clean), rng, role, exclude_id)
        return mix_at_snr(clean, noise, float(snr_db), sample_rate)

"""In-memory view of the prepared dataset and feature builders for every stream.

Time bases
* audio features: 100 Hz, frame centres from ``MFCCExtractor.frame_times`` (379 frames / 3.8 s)
* visual-only models: a uniform 30 Hz grid (114 frames / 3.8 s), resampled from the
  variable-rate frame timestamps
* early fusion: visual features (with their deltas, computed at 30 Hz) are interpolated onto the
  audio frame centres so both streams share one 100 Hz time axis
"""

from __future__ import annotations

import zlib
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .audio.features import MFCCConfig, MFCCExtractor, cmvn, deltas
from .audio.noise import NoiseBank
from .config import Config, work_path
from .data.manifest import load_manifest
from .prepare import audio_cache_path, track_dir
from .visual.degrade import degrade_track
from .visual.features import (
    PCAProjector,
    block_dct_features,
    dct_features,
    resample,
    shape_features,
    uniform_grid,
)

CLEAN = ("clean", "clean")


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def eval_conditions(cfg: Config) -> list[tuple[str, object]]:
    conds = [CLEAN]
    for kind in cfg["noise"]["types"]:
        for snr in cfg["noise"]["eval_snrs_db"]:
            if snr != "clean":
                conds.append((kind, snr))
    return conds


def cond_key(split: str, kind: str, snr) -> str:
    return f"{split}|{kind}|{snr}"


def frame_features(track: dict, kind: str, vcfg: dict, pca: PCAProjector | None = None,
                   n_dct: int | None = None) -> np.ndarray:
    """Per-video-frame features ``(T_frames, D)``; ``kind`` combines parts with '+', e.g. 'dct+shape'."""
    n_dct = n_dct or vcfg["dct_coeffs"]
    parts = []
    for k in kind.split("+"):
        if k == "dct":
            parts.append(dct_features(track["roi"], n_dct, vcfg["dct_drop_dc"]))
        elif k == "blockdct":
            parts.append(block_dct_features(track["roi"], 8, 3))
        elif k == "pca":
            parts.append(pca.transform(track["roi"]))
        elif k == "shape":
            if "lips" not in track:
                raise ValueError("shape features need the landmark ROI method")
            parts.append(shape_features(track["lips"]))
        else:
            raise ValueError(f"unknown visual feature '{k}'")
    return np.concatenate(parts, axis=1)


def visual_sequence(track: dict, kind: str, vcfg: dict, clip_seconds: float, pca: PCAProjector | None = None,
                    n_dct: int | None = None, n_deltas: int | None = None, target_times: np.ndarray | None = None,
                    interp: str = "linear", normalise: bool | None = None) -> np.ndarray:
    """One clip's visual feature matrix ``(D, T)``.

    Frame features (at the irregular video timestamps) -> uniform 30 Hz grid -> deltas (which
    assume uniform spacing) -> per-utterance mean/variance normalisation -> optionally
    interpolated onto the audio frame centres for early fusion.

    ``normalise=False`` keeps raw values; needed by models that average over time, because the
    time-average of a mean-normalised sequence is identically zero.
    """
    normalise = (vcfg["cmvn"] == "utterance") if normalise is None else normalise
    n_deltas = vcfg["deltas"] if n_deltas is None else n_deltas
    grid = uniform_grid(clip_seconds, vcfg["frame_rate"])
    f = resample(frame_features(track, kind, vcfg, pca, n_dct), track["t"], grid)
    parts = [f]
    for _ in range(n_deltas):
        parts.append(deltas(parts[-1], 2))
    f = np.concatenate(parts, axis=1)
    if normalise:
        f = cmvn(f)
    if target_times is not None:
        f = resample(f, grid, target_times, interp)
    return f.T.astype(np.float32)


class Corpus:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = load_manifest(cfg)
        self.labels: list[str] = list(cfg["labels"])
        self.ids: list[str] = self.df.clip_id.tolist()
        self.y = self.df.label_idx.to_numpy()
        self.split = self.df.split.to_numpy()
        self.sr = cfg["audio"]["sample_rate"]
        self.clip_seconds = cfg["audio"]["clip_seconds"]
        self._tracks: dict[str, dict] = {}
        self._audio_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ indexing
    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def ids_of(self, split: str) -> list[str]:
        return [self.ids[i] for i in self.idx(split)]

    def y_of(self, split: str) -> np.ndarray:
        return self.y[self.idx(split)]

    # ------------------------------------------------------------------ audio
    @cached_property
    def waves(self) -> dict[str, np.ndarray]:
        with np.load(audio_cache_path(self.cfg)) as z:
            return {k: z[k] for k in z.files}

    @cached_property
    def session2(self) -> pd.DataFrame:
        p = work_path(self.cfg, "session2_manifest.csv")
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["clip_id", "label_idx"])

    @cached_property
    def noise_bank(self) -> NoiseBank:
        ambient = None
        p = Path(self.cfg["paths"]["ambient_noise"])
        if p.exists():
            ambient, fs = sf.read(str(p), dtype="float32")
            if fs != self.sr:
                raise ValueError("ambient noise must be sampled at the audio sample rate")
        pools = {"train": {i: self.waves[i] for i in self.ids_of("train")},
                 "eval": {i: self.waves[i] for i in self.ids_of("val") + self.ids_of("test")}}
        n = self.cfg["noise"]
        return NoiseBank(ambient, pools, n["ambient_train_fraction"], n["babble_talkers"])

    def mfcc_config(self, **overrides) -> MFCCConfig:
        return MFCCConfig.from_config(self.cfg["audio"], **overrides)

    def audio_features(self, waves: list[np.ndarray], mfcc: MFCCConfig) -> np.ndarray:
        ext = MFCCExtractor(mfcc)
        return np.stack([ext(w).T for w in waves]).astype(np.float32)        # (N, D, T)

    def audio_frame_times(self, mfcc: MFCCConfig) -> np.ndarray:
        ext = MFCCExtractor(mfcc)
        n = int(round(self.clip_seconds * self.sr))
        from .audio.features import num_frames
        return ext.frame_times(num_frames(n, mfcc.frame_len, mfcc.hop))

    def noisy_waves(self, ids: list[str], kind: str, snr, role: str = "eval") -> list[np.ndarray]:
        """Deterministic corruption: the same clip/noise/SNR always yields the same signal, so
        every model is evaluated on identical inputs (paired comparisons, late fusion)."""
        out = []
        for cid in ids:
            rng = np.random.default_rng(stable_seed(cid, kind, snr, role))
            out.append(self.noise_bank.corrupt(self.waves[cid], kind, snr, rng, role, self.sr, exclude_id=cid))
        return out

    def audio_eval(self, split: str, kind: str, snr, mfcc: MFCCConfig) -> np.ndarray:
        key = (split, kind, str(snr), repr(mfcc))
        if key not in self._audio_cache:
            ids = self.ids_of(split)
            self._audio_cache[key] = self.audio_features(self.noisy_waves(ids, kind, snr), mfcc)
        return self._audio_cache[key]

    def audio_bank(self, split: str, mfcc: MFCCConfig, n_versions: int, seed: int) -> np.ndarray:
        """Multi-condition training data ``(N, V, D, T)``: version 0 is clean, the rest are corrupted
        with *training* noise material at random types and SNRs."""
        ids = self.ids_of(split)
        types = [t for t in self.cfg["noise"]["types"] if t in self.noise_bank.available]
        snrs = [s for s in self.cfg["noise"]["train_snrs_db"] if s != "clean"]
        rng = np.random.default_rng(seed)
        versions = [self.audio_features([self.waves[i] for i in ids], mfcc)]
        for _ in range(n_versions - 1):
            waves = [self.noise_bank.corrupt(self.waves[cid], rng.choice(types), float(rng.choice(snrs)), rng,
                                             "train", self.sr, exclude_id=cid) for cid in ids]
            versions.append(self.audio_features(waves, mfcc))
        return np.stack(versions, axis=1)

    def session2_audio(self, mfcc: MFCCConfig) -> tuple[np.ndarray, np.ndarray]:
        ids = self.session2.clip_id.tolist()
        return self.audio_features([self.waves[i] for i in ids], mfcc), self.session2.label_idx.to_numpy()

    # ------------------------------------------------------------------ visual
    def tracks(self, method: str) -> dict:
        if method not in self._tracks:
            d = track_dir(self.cfg, method)
            tr = {}
            for cid in self.ids:
                with np.load(d / f"{cid}.npz") as z:
                    tr[cid] = {k: z[k] for k in z.files}
            self._tracks[method] = tr
        return self._tracks[method]

    def fit_pca(self, method: str, n_components: int, stride: int = 3) -> PCAProjector:
        tr = self.tracks(method)
        frames = np.concatenate([tr[c]["roi"][::stride] for c in self.ids_of("train")])
        return PCAProjector(n_components).fit(frames)

    def frame_features(self, track: dict, kind: str, pca: PCAProjector | None = None,
                       n_dct: int | None = None) -> np.ndarray:
        return frame_features(track, kind, self.cfg["visual"], pca, n_dct)

    def track(self, cid: str, method: str, degrade_level: int = 0) -> dict:
        """A clip's mouth track, optionally degraded (deterministically per clip and level)."""
        tr = self.tracks(method)[cid]
        if not degrade_level:
            return tr
        rng = np.random.default_rng(stable_seed(cid, "visual", degrade_level))
        return degrade_track(tr, degrade_level, rng, self.cfg["visual"]["roi_scale"])

    def visual_features(self, ids: list[str], method: str, kind: str, target_times: np.ndarray | None = None,
                        pca: PCAProjector | None = None, n_dct: int | None = None,
                        n_deltas: int | None = None, interp: str = "linear", degrade_level: int = 0,
                        normalise: bool | None = None) -> np.ndarray:
        """``(N, D, T)`` visual features on the 30 Hz grid, or interpolated to ``target_times``."""
        out = [visual_sequence(self.track(cid, method, degrade_level), kind, self.cfg["visual"], self.clip_seconds,
                               pca, n_dct, n_deltas, target_times, interp, normalise) for cid in ids]
        return np.stack(out).astype(np.float32)

    def roi_sequences(self, ids: list[str], method: str, degrade_level: int = 0) -> np.ndarray:
        """Raw mouth crops on the 30 Hz grid ``(N, T, H, W)`` in [0, 1] (nearest earlier frame)."""
        grid = uniform_grid(self.clip_seconds, self.cfg["visual"]["frame_rate"])
        out = []
        for cid in ids:
            tr = self.track(cid, method, degrade_level)
            t, roi = tr["t"], tr["roi"]
            j = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
            out.append(roi[j].astype(np.float32) / 255.0)
        return np.stack(out)

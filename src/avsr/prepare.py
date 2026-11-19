"""Offline preparation: manifest, time-aligned audio cache and mouth tracks for every clip.

Everything here is deterministic and cached under ``work_dir`` so experiments start in seconds.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .config import Config, work_path
from .data.manifest import build_manifest, build_session2_manifest, manifest_path
from .data.media import fit_length, load_aligned_audio, probe


def prepare_manifest(cfg: Config, force: bool = False) -> pd.DataFrame:
    path = manifest_path(cfg)
    if path.exists() and not force:
        return pd.read_csv(path)
    df = build_manifest(cfg)
    df.to_csv(path, index=False)
    print(f"manifest: {len(df)} clips, {df.label.nunique()} labels -> {path}")
    print(df.groupby("split").size().to_string())
    print(f"audio/video offset: {df.av_offset.mean() * 1000:.0f} ms on average "
          f"({df.av_offset.min() * 1000:.0f}-{df.av_offset.max() * 1000:.0f} ms)")
    return df


def audio_cache_path(cfg: Config) -> Path:
    return work_path(cfg, "cache", "audio.npz")


def prepare_audio(cfg: Config, df: pd.DataFrame, force: bool = False) -> None:
    path = audio_cache_path(cfg)
    if path.exists() and not force:
        return
    sr, secs = cfg["audio"]["sample_rate"], cfg["audio"]["clip_seconds"]
    waves = {}
    for row in tqdm(df.itertuples(), total=len(df), desc="audio"):
        waves[row.clip_id] = load_aligned_audio(row.path, sr, secs, probe(row.path))
    s2 = build_session2_manifest(cfg)
    for row in s2.itertuples():
        x, fs = sf.read(row.path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if fs != sr:
            raise ValueError(f"{row.path}: expected {sr} Hz, got {fs}")
        waves[row.clip_id] = fit_length(x, int(round(sr * secs)))
    np.savez_compressed(path, **waves)
    s2.to_csv(work_path(cfg, "session2_manifest.csv"), index=False)
    print(f"audio cache: {len(waves)} waveforms ({len(s2)} from the second session) -> {path}")


def track_dir(cfg: Config, method: str) -> Path:
    d = Path(cfg["paths"]["work_dir"]) / "tracks" / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def _track_one(args):
    clip_id, video_path, out_path, method, roi_size, roi_scale, clahe = args
    os.environ.setdefault("GLOG_minloglevel", "2")
    from .visual.roi import extract_mouth_track
    track = extract_mouth_track(video_path, method=method, roi_size=tuple(roi_size),
                                roi_scale=roi_scale, clahe=clahe)
    np.savez_compressed(out_path, **track)
    return clip_id, float(track["ok"].mean())


def prepare_tracks(cfg: Config, df: pd.DataFrame, method: str | None = None, workers: int | None = None,
                   force: bool = False) -> pd.DataFrame:
    vcfg = cfg["visual"]
    method = method or vcfg["roi_method"]
    out_dir = track_dir(cfg, method)
    jobs = []
    for row in df.itertuples():
        out = out_dir / f"{row.clip_id}.npz"
        if force or not out.exists():
            jobs.append((row.clip_id, row.path, str(out), method, vcfg["roi_size"], vcfg["roi_scale"],
                         vcfg["clahe"]))
    stats = []
    if jobs:
        workers = workers or max(1, min(6, (os.cpu_count() or 2) - 2))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_track_one, j) for j in jobs]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"tracks[{method}]"):
                stats.append(f.result())
    report = pd.DataFrame(stats, columns=["clip_id", "detection_rate"])
    if len(report):
        report.to_csv(out_dir / "detection_report.csv", index=False)
        print(f"{method}: mean detection rate {report.detection_rate.mean():.4f}, "
              f"worst clip {report.detection_rate.min():.3f}")
    return report


def prepare_all(cfg: Config, methods=("landmarks", "haar"), force: bool = False) -> None:
    df = prepare_manifest(cfg, force)
    prepare_audio(cfg, df, force)
    for m in methods:
        prepare_tracks(cfg, df, m, force=force)

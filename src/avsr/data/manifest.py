"""Dataset manifest and the fixed, stratified speaker-dependent split.

Layout on disk: ``<video_root>/<Name>/<NameNNN>.mp4`` (20 names x 30 takes). One folder uses a
truncated prefix (``Ngoz``); ``label_aliases`` in the config maps it to the canonical label so it
can never become a 21st class (a bug in an earlier version of this project).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, work_path
from .media import probe

_TAKE_RE = re.compile(r"^(?P<prefix>[A-Za-z]+?)_?(?P<take>\d+)$")

MANIFEST_COLUMNS = ["clip_id", "label", "label_idx", "take", "split", "path",
                    "av_offset", "video_duration", "n_video_frames"]


def canonical_label(prefix: str, cfg: Config) -> str:
    aliases = {k.lower(): v for k, v in cfg["label_aliases"].items()}
    labels = {l.lower(): l for l in cfg["labels"]}
    key = prefix.lower()
    if key in aliases:
        return aliases[key]
    if key in labels:
        return labels[key]
    raise ValueError(f"unknown label prefix '{prefix}'")


def parse_name(stem: str, cfg: Config) -> tuple[str, int]:
    m = _TAKE_RE.match(stem)
    if not m:
        raise ValueError(f"cannot parse file name '{stem}'")
    return canonical_label(m["prefix"], cfg), int(m["take"])


def assign_splits(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Deterministic stratified split: the same takes go to train/val/test for every experiment."""
    counts = cfg["split"]["per_class"]
    rng = np.random.default_rng(cfg["split"]["seed"])
    split = pd.Series(index=df.index, dtype=object)
    for _, group in df.groupby("label", sort=True):
        idx = rng.permutation(group.index.to_numpy())
        n_tr, n_va, n_te = counts["train"], counts["val"], counts["test"]
        if len(idx) < n_tr + n_va + n_te:
            raise ValueError(f"class {group.label.iloc[0]} has only {len(idx)} clips")
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:n_tr + n_va + n_te]] = "test"
    return split


def build_manifest(cfg: Config, with_probe: bool = True) -> pd.DataFrame:
    root = Path(cfg["paths"]["video_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"video_root not found: {root} (set AVSR_VIDEO_ROOT)")
    label_to_idx = {l: i for i, l in enumerate(cfg["labels"])}
    rows = []
    for path in sorted(root.glob("*/*.mp4")):
        label, take = parse_name(path.stem, cfg)
        row = {"clip_id": f"{label}_{take:03d}", "label": label, "label_idx": label_to_idx[label],
               "take": take, "path": str(path)}
        if with_probe:
            info = probe(path)
            row.update(av_offset=round(info.av_offset, 6), video_duration=info.video_duration,
                       n_video_frames=info.n_video_frames)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["label_idx", "take"]).reset_index(drop=True)
    if df.clip_id.duplicated().any():
        raise ValueError("duplicate clip ids in dataset")
    df["split"] = assign_splits(df, cfg)
    return df[[c for c in MANIFEST_COLUMNS if c in df.columns]]


def manifest_path(cfg: Config) -> Path:
    return work_path(cfg, "manifest.csv")


def load_manifest(cfg: Config) -> pd.DataFrame:
    p = manifest_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `avsr prepare` first")
    return pd.read_csv(p)


def build_session2_manifest(cfg: Config) -> pd.DataFrame:
    """Audio-only recordings of the same vocabulary made weeks earlier with a different microphone.

    Used purely as an out-of-session test set (never for training or model selection).
    """
    root = Path(cfg["paths"]["session2_audio_root"])
    if not root.is_dir():
        return pd.DataFrame(columns=["clip_id", "label", "label_idx", "take", "path"])
    label_to_idx = {l: i for i, l in enumerate(cfg["labels"])}
    rows = []
    for path in sorted(root.glob("*/*.wav")):
        label, take = parse_name(path.stem, cfg)
        rows.append({"clip_id": f"s2_{label}_{take:03d}", "label": label,
                     "label_idx": label_to_idx[label], "take": take, "path": str(path)})
    return pd.DataFrame(rows)

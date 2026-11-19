"""Explanatory figures built from one real clip (not from experiment results).

* ``av_alignment.png``   - waveform, spectrogram, MFCCs and lip-opening on one time axis, with the
                           naive (offset-ignoring) audio extraction overlaid to show the 0.27 s error;
* ``roi_comparison.png`` - landmark-tracked vs. Haar fixed-proportion mouth crops over the word;
* ``dct_reconstruction.png`` - the mouth ROI rebuilt from the first k zig-zag DCT coefficients;
* ``visual_degradation.png`` - the four video-quality levels used in the fusion study;
* ``landmarks_frame.png`` - the face-mesh lip landmarks and crop box on a video frame.
"""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audio.features import MFCCExtractor  # noqa: E402
from .audio.vad import speech_endpoints  # noqa: E402
from .config import Config  # noqa: E402
from .corpus import Corpus  # noqa: E402
from .data.media import decode_audio, iter_frames  # noqa: E402
from .report import INK, INK2, MUTED, SYSTEMS, _figdir, _save, _style  # noqa: E402
from .visual.degrade import LEVEL_NAMES  # noqa: E402
from .visual.features import dct_reconstruct, shape_features  # noqa: E402

EXAMPLE = "Sivaprasath_007"


def _example(c: Corpus) -> str:
    return EXAMPLE if EXAMPLE in c.ids else c.ids_of("test")[0]


def fig_alignment(cfg: Config, c: Corpus, cid: str) -> None:
    row = c.df.set_index("clip_id").loc[cid]
    x = c.waves[cid]
    sr = c.sr
    naive = decode_audio(row.path, sr)
    mf = c.mfcc_config(cmvn="none", deltas=0)
    ext = MFCCExtractor(mf)
    feats = ext(x)
    t_f = ext.frame_times(len(feats))
    _, pspec = ext.spectrum(x)
    tr = c.tracks("landmarks")[cid]
    opening = shape_features(tr["lips"])[:, 3]
    on, off = speech_endpoints(x, sr)
    lo, hi = max(0, on - 0.5), min(c.clip_seconds, off + 0.5)

    fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3, 1.1, 0.9]})
    t = np.arange(len(x)) / sr
    axes[0].plot(np.arange(len(naive)) / sr, naive, color=MUTED, lw=0.7, label="naive extraction (offset ignored)")
    axes[0].plot(t, x, color=SYSTEMS["audio-tcn-multi"][1], lw=0.8, label="aligned to the video clock")
    axes[0].set_ylabel("waveform")
    axes[0].legend(loc="upper right", fontsize=7.5)
    axes[0].set_title(f"One clip ('{row.label}') on a single time axis — audio stream starts "
                      f"{row.av_offset * 1000:.0f} ms after the first video frame", loc="left")
    axes[1].imshow(10 * np.log10(pspec.T + 1e-10), aspect="auto", origin="lower", cmap="magma",
                   extent=[t_f[0], t_f[-1], 0, sr / 2000])
    axes[1].set_ylabel("frequency (kHz)")
    axes[1].grid(False)
    axes[2].imshow(feats[:, 1:].T, aspect="auto", origin="lower", cmap="RdBu_r",
                   extent=[t_f[0], t_f[-1], 1, feats.shape[1] - 1])
    axes[2].set_ylabel("MFCC c1–c12")
    axes[2].grid(False)
    axes[3].plot(tr["t"], opening, color=SYSTEMS["visual-clean"][1], marker=".", ms=3)
    axes[3].set_ylabel("lip opening")
    axes[3].set_xlabel("time since first video frame (s)  ·  dots = video frames (variable frame rate)")
    for ax in axes:
        ax.axvspan(on, off, color=INK, alpha=0.05, lw=0)
        ax.set_xlim(lo, hi)
    fig.tight_layout()
    _save(fig, cfg, "av_alignment.png")


def _speech_frames(tr, on, off, n=10):
    t = tr["t"]
    idx = np.flatnonzero((t >= on - 0.05) & (t <= off + 0.05))
    return idx[np.linspace(0, len(idx) - 1, n).astype(int)]


def fig_roi_comparison(cfg: Config, c: Corpus, cid: str) -> None:
    on, off = speech_endpoints(c.waves[cid], c.sr)
    rows = []
    for method in ("haar", "landmarks"):
        tr = c.tracks(method)[cid]
        rows.append(np.concatenate([tr["roi"][i] for i in _speech_frames(tr, on, off)], axis=1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 2.6))
    for ax, img, title in zip(axes, rows, ["Haar face box + fixed proportions (original approach)",
                                           "Face-mesh lip landmarks, size fixed per clip (this project)"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, cfg, "roi_comparison.png")


def fig_dct_reconstruction(cfg: Config, c: Corpus, cid: str) -> None:
    tr = c.tracks("landmarks")[cid]
    img = tr["roi"][int(np.argmax(shape_features(tr["lips"])[:, 3]))]       # most open frame
    ks = [1, 6, 15, 44, 150, img.size]
    fig, axes = plt.subplots(1, len(ks), figsize=(12, 1.9))
    for ax, k in zip(axes, ks):
        ax.imshow(dct_reconstruct(img, k), cmap="gray", vmin=0, vmax=1)
        ax.set_title("original" if k == img.size else f"k = {k}", fontsize=9, color=INK2)
        ax.axis("off")
    fig.suptitle("Mouth ROI rebuilt from the first k zig-zag DCT coefficients (the feature uses k = 44)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, cfg, "dct_reconstruction.png")


def fig_degradation(cfg: Config, c: Corpus, cid: str) -> None:
    tr0 = c.tracks("landmarks")[cid]
    i = int(np.argmax(shape_features(tr0["lips"])[:, 3]))
    fig, axes = plt.subplots(1, len(LEVEL_NAMES), figsize=(10, 1.8))
    for ax, L in zip(axes, sorted(LEVEL_NAMES)):
        ax.imshow(c.track(cid, "landmarks", L)["roi"][i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(LEVEL_NAMES[L], fontsize=9, color=INK2)
        ax.axis("off")
    fig.suptitle("Video-quality levels used to test fusion (resolution loss, sensor noise, landmark jitter)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, cfg, "visual_degradation.png")


def fig_landmarks(cfg: Config, c: Corpus, cid: str) -> None:
    row = c.df.set_index("clip_id").loc[cid]
    tr = c.tracks("landmarks")[cid]
    i = int(np.argmax(shape_features(tr["lips"])[:, 3]))
    frame = next(f for k, (_, f) in enumerate(iter_frames(row.path)) if k == i)
    lips = tr["lips"][i]
    cx, cy = lips.mean(0)
    w = cfg["visual"]["roi_scale"] * float(tr["mouth_width_px"])
    h = w * cfg["visual"]["roi_size"][0] / cfg["visual"]["roi_size"][1]
    x0, y0 = int(cx - 2.2 * w), int(cy - 2.6 * w)
    crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[max(0, y0):int(cy + 1.2 * w), max(0, x0):int(cx + 2.2 * w)]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(crop)
    ox, oy = max(0, x0), max(0, y0)
    n_outer = 20
    ax.plot(*np.vstack([lips[:n_outer], lips[:1]]).T - np.array([[ox], [oy]]), color="#1baf7a", lw=1.2)
    ax.plot(*np.vstack([lips[n_outer:], lips[n_outer:n_outer + 1]]).T - np.array([[ox], [oy]]), color="#eda100", lw=1.2)
    ax.add_patch(plt.Rectangle((cx - w / 2 - ox, cy - h / 2 - oy), w, h, fill=False, ec="#ffffff", lw=1.2, ls="--"))
    ax.set_title("Lip landmarks (outer / inner) and the mouth crop", loc="left", fontsize=9.5)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, cfg, "landmarks_frame.png")


def build_illustrations(cfg: Config) -> None:
    _style()
    c = Corpus(cfg)
    cid = _example(c)
    for fn in (fig_alignment, fig_roi_comparison, fig_dct_reconstruction, fig_degradation, fig_landmarks):
        fn(cfg, c, cid)
    print(f"illustrations written to {_figdir(cfg)}")

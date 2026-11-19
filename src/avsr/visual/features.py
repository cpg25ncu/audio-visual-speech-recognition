"""Visual speech features computed from mouth-ROI sequences and lip landmarks.

Appearance (implicit) features
* 2D-DCT: the orthonormal DCT-II of the grayscale ROI; coefficients are taken in zig-zag
  order (low spatial frequencies first). Most of the image energy sits in a handful of low-order
  coefficients, so the first K form a compact code of the overall mouth configuration while
  discarding fine texture, skin detail and noise.
* Block DCT: the same idea applied to 8x8 tiles, keeping the lowest few coefficients per tile;
  it preserves *where* in the mouth a change happens at the cost of more dimensions.
* PCA ("eigenlips"): a basis learnt from training frames instead of a fixed cosine basis.

Geometric (explicit) features
* Shape: widths, heights, opening and areas of the inner/outer lip contours from face-mesh
  landmarks, normalised by the clip's median mouth width (removes camera distance).

All per-frame features are timestamped and later resampled onto a uniform time grid.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn

# MediaPipe face-mesh indices
LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIP_POINTS = LIP_OUTER + LIP_INNER
SHAPE_NAMES = ["outer_width", "outer_height", "inner_width", "inner_height",
               "outer_area", "inner_area", "opening_ratio"]


# --------------------------------------------------------------------------- DCT
@lru_cache(maxsize=16)
def zigzag_indices(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """JPEG-style zig-zag scan of an ``h x w`` grid: anti-diagonals of increasing frequency,
    alternating direction. Returns (rows, cols)."""
    order = sorted(((i, j) for i in range(h) for j in range(w)),
                   key=lambda ij: (ij[0] + ij[1], ij[0] if (ij[0] + ij[1]) % 2 else ij[1]))
    rows, cols = zip(*order)
    return np.array(rows), np.array(cols)


def roi_to_float(roi: np.ndarray) -> np.ndarray:
    return roi.astype(np.float32) / 255.0


def dct2(img: np.ndarray) -> np.ndarray:
    return dctn(img, type=2, norm="ortho")


def dct_features(rois: np.ndarray, n_coeffs: int, drop_dc: bool = False) -> np.ndarray:
    """``(T, H, W)`` uint8 -> ``(T, n_coeffs)`` zig-zag-ordered 2D-DCT coefficients."""
    T, H, W = rois.shape
    coeffs = dctn(roi_to_float(rois), type=2, norm="ortho", axes=(1, 2))
    r, c = zigzag_indices(H, W)
    start = 1 if drop_dc else 0
    return coeffs[:, r[start:start + n_coeffs], c[start:start + n_coeffs]].astype(np.float32)


def dct_reconstruct(img: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Keep only the first ``n_coeffs`` zig-zag coefficients and invert - shows what the feature 'sees'."""
    H, W = img.shape
    coeffs = dct2(roi_to_float(img))
    mask = np.zeros_like(coeffs)
    r, c = zigzag_indices(H, W)
    mask[r[:n_coeffs], c[:n_coeffs]] = 1.0
    return np.clip(idctn(coeffs * mask, type=2, norm="ortho"), 0, 1)


def energy_compaction(rois: np.ndarray) -> np.ndarray:
    """Cumulative fraction of AC energy captured by the first k zig-zag coefficients (mean over frames)."""
    T, H, W = rois.shape
    coeffs = dctn(roi_to_float(rois), type=2, norm="ortho", axes=(1, 2))
    r, c = zigzag_indices(H, W)
    e = (coeffs[:, r, c] ** 2)[:, 1:]          # drop DC: brightness, not shape
    return np.cumsum(e.mean(0)) / e.mean(0).sum()


def block_dct_features(rois: np.ndarray, block: int = 8, per_block: int = 3) -> np.ndarray:
    T, H, W = rois.shape
    if H % block or W % block:
        raise ValueError("ROI size must be divisible by the block size")
    x = roi_to_float(rois).reshape(T, H // block, block, W // block, block).transpose(0, 1, 3, 2, 4)
    coeffs = dctn(x, type=2, norm="ortho", axes=(3, 4))
    r, c = zigzag_indices(block, block)
    return coeffs[..., r[:per_block], c[:per_block]].reshape(T, -1).astype(np.float32)


# --------------------------------------------------------------------------- shape
def polygon_area(pts: np.ndarray) -> np.ndarray:
    """Shoelace formula; ``pts`` is ``(T, K, 2)``."""
    x, y = pts[..., 0], pts[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1))


def shape_features(lips: np.ndarray) -> np.ndarray:
    """``lips``: ``(T, 40, 2)`` points in ``LIP_POINTS`` order -> ``(T, 7)`` geometric features."""
    idx = {p: i for i, p in enumerate(LIP_POINTS)}

    def dist(a, b):
        return np.linalg.norm(lips[:, idx[a]] - lips[:, idx[b]], axis=-1)

    outer_w, outer_h = dist(61, 291), dist(0, 17)
    inner_w, inner_h = dist(78, 308), dist(13, 14)
    scale = np.nanmedian(outer_w) + 1e-6
    outer_a = polygon_area(lips[:, :len(LIP_OUTER)])
    inner_a = polygon_area(lips[:, len(LIP_OUTER):])
    feats = np.stack([outer_w / scale, outer_h / scale, inner_w / scale, inner_h / scale,
                      outer_a / scale ** 2, inner_a / scale ** 2, inner_h / (inner_w + 1e-6)], axis=1)
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- PCA
class PCAProjector:
    """Eigen-lips: PCA basis fitted on (a subsample of) training ROI frames."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, frames: np.ndarray) -> PCAProjector:
        X = roi_to_float(frames).reshape(len(frames), -1)
        self.mean_ = X.mean(0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components].astype(np.float32)
        var = s ** 2
        self.explained_variance_ratio_ = (var / var.sum())[: self.n_components]
        return self

    def transform(self, rois: np.ndarray) -> np.ndarray:
        X = roi_to_float(rois).reshape(len(rois), -1)
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)


# --------------------------------------------------------------------------- temporal resampling
def resample(feat: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray, method: str = "linear") -> np.ndarray:
    """Resample a timestamped feature sequence ``(T_src, D)`` onto times ``t_dst``.

    Works for irregular ``t_src`` (variable-frame-rate video). Outside the observed range the
    first/last value is held. ``cubic`` uses a natural cubic spline per dimension.
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    order = np.argsort(t_src)
    t_src, feat = t_src[order], feat[order]
    keep = np.concatenate([[True], np.diff(t_src) > 1e-6])
    t_src, feat = t_src[keep], feat[keep]
    if method == "linear" or len(t_src) < 4:
        out = np.stack([np.interp(t_dst, t_src, feat[:, d]) for d in range(feat.shape[1])], axis=1)
    elif method == "cubic":
        from scipy.interpolate import CubicSpline
        tq = np.clip(t_dst, t_src[0], t_src[-1])
        out = CubicSpline(t_src, feat, axis=0, bc_type="natural")(tq)
    else:
        raise ValueError(f"unknown interpolation '{method}'")
    return out.astype(np.float32)


def uniform_grid(duration: float, rate: float) -> np.ndarray:
    return np.arange(int(round(duration * rate))) / rate

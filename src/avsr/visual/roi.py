"""Mouth region-of-interest tracking.

Two front-ends are provided so their effect on recognition can be measured:

``landmarks`` (default)
    MediaPipe FaceLandmarker (478-point face mesh, VIDEO mode with temporal tracking). The ROI is
    centred on the lip landmarks of every frame and its size is fixed per clip to
    ``roi_scale x`` the median mouth-corner distance, so the crop follows the mouth but does not
    "breathe" when the mouth opens. Lip landmarks are kept for geometric shape features.

``haar``
    The approach of the original coursework: Viola-Jones face box, mouth assumed to lie at fixed
    proportions of the box. Cheap, but the box jitters from frame to frame and the proportions
    are only right for a frontal, neutral face.

Common post-processing: missing detections are interpolated, centres are smoothed with a short
moving average, the crop is taken with sub-pixel accuracy, resized to ``roi_size`` and contrast
normalised with CLAHE (removes slow illumination differences between recordings).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data.media import iter_frames
from .features import LIP_POINTS

DEFAULT_LANDMARKER = Path(__file__).resolve().parents[3] / "data" / "models" / "face_landmarker.task"
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


class LandmarkDetector:
    """Thin wrapper around MediaPipe's FaceLandmarker (Tasks API); one instance per clip."""

    def __init__(self, model_path: str | Path = DEFAULT_LANDMARKER):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}. Download it from {LANDMARKER_URL}")
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def __call__(self, bgr: np.ndarray, t: float) -> np.ndarray | None:
        """Return the 478 landmarks in pixel coordinates ``(478, 2)`` or None."""
        h, w = bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        ms = max(int(round(t * 1000)), self._last_ms + 1)   # VIDEO mode needs strictly increasing time
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self._landmarker.close()


class HaarDetector:
    """Largest frontal face via Viola-Jones; mouth at fixed proportions of the face box."""

    def __init__(self, downscale: float = 0.5):
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(path))
        self.downscale = downscale

    def __call__(self, bgr: np.ndarray, t: float):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)
        faces = self.cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = (np.array(max(faces, key=lambda b: b[2] * b[3])) / self.downscale)
        return np.array([x + 0.5 * w, y + 0.82 * h, 0.40 * w], dtype=np.float32)   # cx, cy, mouth width

    def close(self):
        pass


def _fill_missing(values: np.ndarray, ok: np.ndarray, t: np.ndarray) -> np.ndarray:
    if ok.all():
        return values
    if not ok.any():
        raise RuntimeError("no face detected in any frame")
    flat = values.reshape(len(values), -1).copy()
    for d in range(flat.shape[1]):
        flat[~ok, d] = np.interp(t[~ok], t[ok], flat[ok, d])
    return flat.reshape(values.shape)


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    return np.stack([np.convolve(xp[:, d], kernel, mode="valid") for d in range(x.shape[1])], axis=1)


def extract_mouth_track(video_path: str | Path, method: str = "landmarks", roi_size=(32, 64),
                        roi_scale: float = 1.7, clahe: bool = True,
                        model_path: str | Path = DEFAULT_LANDMARKER) -> dict[str, np.ndarray]:
    """Track the mouth through a clip.

    Returns ``t`` (s), ``roi`` (T, H, W) uint8, ``ok`` (T,) detection mask and, for the landmark
    method, ``lips`` (T, 40, 2) lip landmarks in pixels.
    """
    detector = LandmarkDetector(model_path) if method == "landmarks" else HaarDetector()
    times, grays, centers, widths, lips, ok = [], [], [], [], [], []
    try:
        for t, bgr in iter_frames(video_path):
            times.append(t)
            grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            det = detector(bgr, t)
            ok.append(det is not None)
            if method == "landmarks":
                pts = det[LIP_POINTS] if det is not None else np.zeros((len(LIP_POINTS), 2), np.float32)
                lips.append(pts)
                centers.append(pts.mean(0))
                widths.append(np.linalg.norm(pts[0] - pts[10]))     # corners 61 and 291
            else:
                d = det if det is not None else np.zeros(3, np.float32)
                centers.append(d[:2])
                widths.append(d[2])
    finally:
        detector.close()

    t = np.asarray(times, dtype=np.float64)
    ok = np.asarray(ok)
    centers = _smooth(_fill_missing(np.asarray(centers, np.float32), ok, t))
    width = float(np.median(np.asarray(widths)[ok]))
    H, W = roi_size
    crop_w = roi_scale * width
    crop_h = crop_w * H / W
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 4)) if clahe else None
    rois = np.empty((len(t), H, W), dtype=np.uint8)
    for i, (gray, (cx, cy)) in enumerate(zip(grays, centers)):
        patch = cv2.getRectSubPix(gray, (int(round(crop_w)), int(round(crop_h))), (float(cx), float(cy)))
        patch = cv2.resize(patch, (W, H), interpolation=cv2.INTER_AREA)
        rois[i] = clahe_op.apply(patch) if clahe_op is not None else patch
    out = {"t": t, "roi": rois, "ok": ok, "mouth_width_px": np.float32(width)}
    if method == "landmarks":
        out["lips"] = _fill_missing(np.asarray(lips, np.float32), ok, t)
    return out

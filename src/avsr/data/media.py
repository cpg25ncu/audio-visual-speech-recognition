"""Media I/O: probing containers, decoding time-aligned audio, iterating video frames with timestamps.

Two properties of the recordings drive the design here (both found by inspecting the files with
ffprobe, see docs/technical_report.md):

* The phone recorded **variable frame rate** video (nominally 60 fps, ~29-31 fps on average with
  irregular gaps). Frame index is therefore not a clock; every frame keeps its presentation time.
* The AAC audio stream starts ~0.25-0.30 s **after** the video stream. A plain
  ``ffmpeg -i clip.mp4 out.wav`` drops that offset and shifts the audio ~9 video frames early.
  We re-insert it so audio sample 0 sits at video time 0.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"'{tool}' was not found on PATH; install FFmpeg first.")
    return exe


@dataclass(frozen=True)
class StreamInfo:
    video_start: float
    audio_start: float
    video_duration: float
    audio_duration: float
    n_video_frames: int
    width: int
    height: int

    @property
    def av_offset(self) -> float:
        """Seconds by which audio starts after video (positive = audio late)."""
        return self.audio_start - self.video_start


def probe(path: str | Path) -> StreamInfo:
    cmd = [_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", str(path)]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    return StreamInfo(
        video_start=float(v.get("start_time", 0.0)),
        audio_start=float(a.get("start_time", 0.0)) if a else 0.0,
        video_duration=float(v.get("duration", 0.0)),
        audio_duration=float(a.get("duration", 0.0)) if a else 0.0,
        n_video_frames=int(v.get("nb_frames", 0) or 0),
        width=int(v["width"]),
        height=int(v["height"]),
    )


def decode_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode the audio stream to mono float32 at ``sample_rate`` (no offset handling)."""
    cmd = [_require("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or crop the first axis of ``x`` to length ``n``."""
    if len(x) >= n:
        return x[:n]
    pad = [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad)


def load_aligned_audio(path: str | Path, sample_rate: int, clip_seconds: float,
                       info: StreamInfo | None = None) -> np.ndarray:
    """Audio on the *video* timeline: sample k is at time k / sample_rate after the first video frame."""
    info = info or probe(path)
    x = decode_audio(path, sample_rate)
    offset = int(round(info.av_offset * sample_rate))
    if offset > 0:
        x = np.concatenate([np.zeros(offset, dtype=np.float32), x])
    elif offset < 0:
        x = x[-offset:]
    return fit_length(x, int(round(clip_seconds * sample_rate))).astype(np.float32)


def iter_frames(path: str | Path) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_seconds, bgr_frame)`` using each frame's presentation time."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame
    finally:
        cap.release()


def frame_timestamps_ffprobe(path: str | Path) -> np.ndarray:
    """Reference frame times from the container (used to validate OpenCV's timestamps)."""
    cmd = [_require("ffprobe"), "-v", "error", "-select_streams", "v", "-show_entries",
           "frame=pts_time", "-of", "csv=p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()
    return np.array([float(t) for t in out if t not in ("", "N/A")])

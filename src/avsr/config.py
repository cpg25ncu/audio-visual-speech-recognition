"""Configuration loading.

The YAML file is turned into a nested dict with attribute access. Relative paths are resolved
against the repository root so scripts work from any working directory, and selected paths can
be overridden with environment variables (useful when the dataset lives elsewhere).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

_ENV_OVERRIDES = {
    "video_root": "AVSR_VIDEO_ROOT",
    "session2_audio_root": "AVSR_SESSION2_ROOT",
    "ambient_noise": "AVSR_AMBIENT_NOISE",
    "work_dir": "AVSR_WORK_DIR",
    "results_dir": "AVSR_RESULTS_DIR",
}


class Config(dict):
    """A dict whose keys are also readable as attributes (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def copy(self) -> Config:
        return Config(copy.deepcopy(dict(self)))


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = deep_update(raw, overrides)
    paths = raw.setdefault("paths", {})
    for key, env in _ENV_OVERRIDES.items():
        if os.environ.get(env):
            paths[key] = os.environ[env]
    raw["paths"] = {k: str(_resolve(v)) for k, v in paths.items()}
    return Config(raw)


def deep_update(base: dict, update: dict) -> dict:
    """Return a copy of ``base`` with ``update`` merged in recursively."""
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def work_path(cfg: Config, *parts: str) -> Path:
    p = Path(cfg["paths"]["work_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def results_path(cfg: Config, *parts: str) -> Path:
    p = Path(cfg["paths"]["results_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

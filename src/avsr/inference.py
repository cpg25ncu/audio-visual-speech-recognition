"""End-to-end recognition of a single video clip with the trained models.

    avsr predict clip.mp4 --noise pink --snr 0

runs the same front-ends as training (aligned audio -> MFCC; mouth tracking -> visual features)
and reports the top candidates of the audio, visual, early-fusion and adaptive late-fusion
recognisers. Optional noise is added to the audio to see fusion at work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .audio.features import MFCCConfig, MFCCExtractor, num_frames
from .audio.noise import NoiseBank
from .config import Config, results_path, work_path
from .corpus import visual_sequence
from .data.media import load_aligned_audio, probe
from .fusion import entropy_weights, fuse_log_linear
from .models import build_model
from .visual.roi import extract_mouth_track


class Predictor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.labels = list(cfg["labels"])
        self.models, self.meta = {}, {}
        for tag in ("audio-tcn-multi", "visual-multi", "av-early-multi"):
            p = work_path(cfg, "models_trained", f"{tag}.pt")
            if not p.exists():
                raise FileNotFoundError(f"{p} not found - run `avsr run audio visual fusion` first")
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            meta = ckpt["meta"]
            model = build_model(meta["arch"], meta["in_dim"], len(self.labels))
            model.load_state_dict(ckpt["state_dict"])
            self.models[tag], self.meta[tag] = model.eval(), meta
        self.mfcc = MFCCExtractor(MFCCConfig.from_config(cfg["audio"]))
        alpha = results_path(cfg, "tables", "fusion_alpha.csv")
        self.adaptive = (8.0, 0.0)
        if alpha.exists():
            a = pd.read_csv(alpha).iloc[0]
            self.adaptive = (float(a.adaptive_a), float(a.adaptive_b))

    @torch.no_grad()
    def _probs(self, tag: str, x: np.ndarray) -> np.ndarray:
        return torch.softmax(self.models[tag](torch.as_tensor(x[None], dtype=torch.float32)), 1).numpy()

    def predict(self, video: str | Path, noise: str | None = None, snr: float | None = None, top: int = 3):
        a, v = self.cfg["audio"], self.cfg["visual"]
        x = load_aligned_audio(video, a["sample_rate"], a["clip_seconds"], probe(video))
        if noise and snr is not None:
            ambient = None
            if noise == "ambient":
                ambient, _ = sf.read(self.cfg["paths"]["ambient_noise"], dtype="float32")
            bank = NoiseBank(ambient, {}, self.cfg["noise"]["ambient_train_fraction"])
            x = bank.corrupt(x, noise, snr, np.random.default_rng(0), "eval", a["sample_rate"])
        fa = self.mfcc(x).T
        n = num_frames(len(x), self.mfcc.cfg.frame_len, self.mfcc.cfg.hop)
        t_audio = self.mfcc.frame_times(n)
        meta = self.meta["visual-multi"]
        track = extract_mouth_track(video, method=meta["roi_method"], roi_size=tuple(v["roi_size"]),
                                    roi_scale=v["roi_scale"], clahe=v["clahe"])
        fv30 = visual_sequence(track, meta["kind"], v, a["clip_seconds"], n_deltas=meta["deltas"])
        fv100 = visual_sequence(track, meta["kind"], v, a["clip_seconds"], n_deltas=meta["deltas"], target_times=t_audio)
        pa = self._probs("audio-tcn-multi", fa)
        pv = self._probs("visual-multi", fv30)
        pe = self._probs("av-early-multi", np.concatenate([fa, fv100], 0))
        pl = fuse_log_linear(pa, pv, entropy_weights(pa, pv, *self.adaptive))
        out = {}
        for name, p in [("audio", pa), ("visual", pv), ("AV early", pe), ("AV late", pl)]:
            order = np.argsort(p[0])[::-1][:top]
            out[name] = [(self.labels[i], float(p[0, i])) for i in order]
        return out

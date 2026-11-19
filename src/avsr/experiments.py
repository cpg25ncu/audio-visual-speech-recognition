"""Experiment suite. Each experiment writes a long-format CSV to ``results/tables`` and stores
posterior probabilities under ``work_dir/probs`` (needed for late fusion and the report).

Protocol (identical for every system)
* fixed stratified split: 18 / 6 / 6 takes per name for train / validation / test;
* model selection (early stopping, fusion weights, best visual front-end) on validation only;
* every configuration is trained with several seeds and reported as mean +/- std on the test split;
* corrupted test inputs (acoustic noise, visual degradation) are generated deterministically, so
  all systems are scored on exactly the same signals.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .audio.features import cmvn, deltas
from .audio.vad import speech_endpoints
from .config import Config, results_path, work_path
from .corpus import CLEAN, Corpus, cond_key, eval_conditions, stable_seed
from .fusion import entropy_weights, fuse_log_linear, select_adaptive_params, select_alpha
from .models import build_model, count_parameters
from .training import TrainConfig, accuracy, fit, predict_proba
from .visual.degrade import LEVELS
from .visual.features import resample

AUDIO_BANK_VERSIONS = 10
VIS_LEVELS = sorted(LEVELS)


def vkey(split: str, level: int) -> str:
    return f"{split}|video{level}"


def avkey(split: str, kind: str, snr, level: int) -> str:
    return f"{split}|{kind}|{snr}|video{level}"


class Runner:
    def __init__(self, cfg: Config, seeds: list[int] | None = None, epochs: int | None = None):
        self.cfg = cfg
        self.corpus = Corpus(cfg)
        self.seeds = list(seeds if seeds is not None else cfg["training"]["seeds"])
        overrides = {"epochs": epochs} if epochs else {}
        self.tcfg = TrainConfig.from_config(cfg["training"], **overrides)
        self.conds = eval_conditions(cfg)
        self.n_classes = len(self.corpus.labels)
        self.probs_dir = work_path(self.cfg, "probs", "_").parent
        self.models_dir = work_path(self.cfg, "models_trained", "_").parent

    # ------------------------------------------------------------------ helpers
    def train(self, kind: str, Xtr, ytr, Xva, yva, seed: int, bank: bool = False, **model_kw):
        in_dim = Xtr.shape[-2] if kind != "roicnn" else 0
        model = build_model(kind, in_dim, self.n_classes, **model_kw)
        tc = replace(self.tcfg, extra={"bank": bank})
        t0 = time.time()
        model, hist = fit(model, Xtr, ytr, Xva, yva, tc, seed)
        hist.update(seconds=round(time.time() - t0, 1), parameters=count_parameters(model))
        return model, hist

    def save_probs(self, tag: str, seed: int, probs: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.probs_dir / f"{tag}__seed{seed}.npz", **probs)

    def load_probs(self, tag: str, seed: int) -> dict[str, np.ndarray]:
        with np.load(self.probs_dir / f"{tag}__seed{seed}.npz") as z:
            return {k: z[k] for k in z.files}

    def save_history(self, tag: str, seed: int, hist: dict) -> None:
        (self.probs_dir / f"{tag}__seed{seed}.history.json").write_text(json.dumps(hist))

    def save_model(self, tag: str, seed: int, model, meta: dict) -> None:
        """Keep the first seed's weights of the systems used by the inference CLI."""
        if seed != self.seeds[0]:
            return
        import torch
        torch.save({"state_dict": model.state_dict(), "meta": meta}, self.models_dir / f"{tag}.pt")

    @staticmethod
    def write(rows: list[dict], name: str) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.to_csv(results_path_for(name), index=False)
        return df

    def y(self, split: str) -> np.ndarray:
        return self.corpus.y_of(split)

    def _audio_eval_sets(self, mfcc, splits=("val", "test"), conds=None) -> dict[str, np.ndarray]:
        conds = conds or self.conds
        return {cond_key(s, k, snr): self.corpus.audio_eval(s, k, snr, mfcc) for s in splits for k, snr in conds}

    def _rows_from_probs(self, probs, experiment, system, seed, extra=None, split="test"):
        rows = []
        for key, p in probs.items():
            parts = key.split("|")
            if parts[0] != split:
                continue
            row = {"experiment": experiment, "system": system, "seed": seed,
                   "accuracy": accuracy(p, self.y(parts[0]))}
            if len(parts) >= 3:
                row.update(noise=parts[1], snr=parts[2])
            if parts[-1].startswith("video"):
                row["vis_level"] = int(parts[-1][5:])
            rows.append({**row, **(extra or {})})
        return rows

    def _multicondition_val(self, mfcc, seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
        """Validation set for multi-condition training: clean + one noisy copy (training noise)."""
        bank = self.corpus.audio_bank("val", mfcc, 2, seed)
        yv = self.y("val")
        return np.concatenate([bank[:, 0], bank[:, 1]]), np.concatenate([yv, yv])

    # ------------------------------------------------------------------ E1 audio-only
    def audio(self) -> pd.DataFrame:
        c = self.corpus
        ytr, yva = self.y("train"), self.y("val")
        rows = []
        mf = c.mfcc_config()
        mf_ss = c.mfcc_config(enhance="spectral_subtraction")
        train_waves = [c.waves[i] for i in c.ids_of("train")]
        Xtr, Xtr_ss = c.audio_features(train_waves, mf), c.audio_features(train_waves, mf_ss)
        bank = c.audio_bank("train", mf, AUDIO_BANK_VERSIONS, seed=7)
        Xva_mc, yva_mc = self._multicondition_val(mf)
        evals, evals_ss = self._audio_eval_sets(mf), self._audio_eval_sets(mf_ss)
        s2X, s2y = c.session2_audio(mf)
        clean_val = cond_key("val", *CLEAN)
        setups = [
            ("audio-tcn-clean", "tcn", Xtr, False, evals[clean_val], yva, evals),
            ("audio-tcn-multi", "tcn", bank, True, Xva_mc, yva_mc, evals),
            ("audio-tcn-clean-ss", "tcn", Xtr_ss, False, evals_ss[clean_val], yva, evals_ss),
            ("audio-cnn2d-clean", "cnn2d", Xtr, False, evals[clean_val], yva, evals),
            ("audio-cnn2d-multi", "cnn2d", bank, True, Xva_mc, yva_mc, evals),
        ]
        for tag, arch, X, is_bank, Xv, yv, ev in setups:
            for seed in self.seeds:
                model, hist = self.train(arch, X, ytr, Xv, yv, seed, bank=is_bank)
                probs = {k: predict_proba(model, x) for k, x in ev.items()}
                if len(s2y) and not tag.endswith("-ss"):
                    p2 = predict_proba(model, s2X)
                    rows.append({"experiment": "audio", "system": tag, "noise": "session2", "snr": "clean",
                                 "seed": seed, "accuracy": accuracy(p2, s2y)})
                rows += self._rows_from_probs(probs, "audio", tag, seed,
                                              {"best_epoch": hist["best_epoch"], "seconds": hist["seconds"]})
                self.save_probs(tag, seed, probs)
                self.save_history(tag, seed, hist)
                self.save_model(tag, seed, model, {"arch": arch, "in_dim": int(X.shape[-2]),
                                                   "enhance": "spectral_subtraction" if tag.endswith("-ss") else "none"})
                print(f"{tag} seed {seed}: clean {accuracy(probs[cond_key('test', *CLEAN)], self.y('test')):.3f} "
                      f"babble 0 dB {accuracy(probs[cond_key('test', 'babble', 0)], self.y('test')):.3f} "
                      f"({hist['seconds']}s, {hist['parameters']} params)")
        return self.write(rows, "audio")

    # ------------------------------------------------------------------ E2 audio front-end ablation
    FRONTEND_VARIANTS = {
        "baseline (mel, tri, 40ch, 13c, E+D+DD, CMVN)": {},
        "linear + rectangular": {"scale": "linear", "shape": "rectangular"},
        "linear + triangular": {"scale": "linear", "shape": "triangular"},
        "mel + rectangular": {"scale": "mel", "shape": "rectangular"},
        "20 channels": {"n_filters": 20},
        "26 channels": {"n_filters": 26},
        "64 channels": {"n_filters": 64},
        "8 cepstra": {"n_ceps": 8},
        "20 cepstra": {"n_ceps": 20},
        "no log-energy": {"use_energy": False},
        "static only": {"deltas": 0},
        "static + delta": {"deltas": 1},
        "no CMVN": {"cmvn": "none"},
    }

    def audio_frontend(self, n_seeds: int = 3) -> pd.DataFrame:
        c = self.corpus
        conds = [CLEAN] + [(k, s) for k in self.cfg["noise"]["types"] for s in (10, 0)]
        rows = []
        for name, ov in self.FRONTEND_VARIANTS.items():
            mf = c.mfcc_config(**ov)
            Xtr = c.audio_features([c.waves[i] for i in c.ids_of("train")], mf)
            evals = self._audio_eval_sets(mf, conds=conds)
            s2X, s2y = c.session2_audio(mf)
            for seed in self.seeds[:n_seeds]:
                model, _ = self.train("tcn", Xtr, self.y("train"), evals[cond_key("val", *CLEAN)], self.y("val"), seed)
                probs = {k: predict_proba(model, x) for k, x in evals.items()}
                rows += self._rows_from_probs(probs, "audio_frontend", name, seed, {"dim": mf.dim})
                if len(s2y):
                    rows.append({"experiment": "audio_frontend", "system": name, "noise": "session2", "snr": "clean",
                                 "seed": seed, "accuracy": accuracy(predict_proba(model, s2X), s2y), "dim": mf.dim})
            print(f"frontend '{name}' done")
        return self.write(rows, "audio_frontend")

    # ------------------------------------------------------------------ E3 visual-only
    VISUAL_VARIANTS = [
        # tag, roi method, feature kind, model, n_deltas
        ("haar / mean DCT / MLP (original approach)", "haar", "dct", "mlp", 0),
        ("landmarks / mean DCT / MLP", "landmarks", "dct", "mlp", 0),
        ("haar / DCT / TCN", "haar", "dct", "tcn", 1),
        ("landmarks / DCT / TCN", "landmarks", "dct", "tcn", 1),
        ("landmarks / block-DCT / TCN", "landmarks", "blockdct", "tcn", 1),
        ("landmarks / PCA / TCN", "landmarks", "pca", "tcn", 1),
        ("landmarks / shape / TCN", "landmarks", "shape", "tcn", 1),
        ("landmarks / DCT+shape / TCN", "landmarks", "dct+shape", "tcn", 1),
        ("landmarks / pixels / ROI-CNN", "landmarks", "pixels", "roicnn", 0),
    ]

    def visual_inputs(self, method: str, kind: str, n_deltas: int, ids: list[str], target_times=None,
                      n_dct: int | None = None, level: int = 0, pca=None, normalise: bool | None = None):
        c = self.corpus
        if kind == "pixels":
            return c.roi_sequences(ids, method, level)
        if "pca" in kind and pca is None:
            pca = self._pca(method)
        return c.visual_features(ids, method, kind, target_times, pca, n_dct, n_deltas, degrade_level=level,
                                 normalise=normalise)

    def _pca(self, method: str):
        if not hasattr(self, "_pca_cache"):
            self._pca_cache = {}
        if method not in self._pca_cache:
            self._pca_cache[method] = self.corpus.fit_pca(method, self.cfg["visual"]["pca_components"])
        return self._pca_cache[method]

    def visual(self) -> pd.DataFrame:
        """Each front-end is trained on clean video and tested on clean and degraded video."""
        c = self.corpus
        rows = []
        for tag, method, kind, arch, nd in self.VISUAL_VARIANTS:
            norm = False if arch == "mlp" else None      # time-averaged features must not be mean-normalised
            Xtr = self.visual_inputs(method, kind, nd, c.ids_of("train"), normalise=norm)
            ev = {vkey(s, L): self.visual_inputs(method, kind, nd, c.ids_of(s), level=L, normalise=norm)
                  for s in ("val", "test") for L in VIS_LEVELS}
            for seed in self.seeds:
                model, hist = self.train(arch, Xtr, self.y("train"), ev[vkey("val", 0)], self.y("val"), seed)
                probs = {k: predict_proba(model, x) for k, x in ev.items()}
                for L in VIS_LEVELS:
                    rows.append({"experiment": "visual", "system": tag, "seed": seed, "vis_level": L,
                                 "accuracy": accuracy(probs[vkey("test", L)], self.y("test")),
                                 "val_accuracy": accuracy(probs[vkey("val", L)], self.y("val")),
                                 "dim": int(Xtr.shape[1]) if arch != "roicnn" else 0, "seconds": hist["seconds"]})
                self.save_probs(_slug(tag), seed, probs)
                self.save_history(_slug(tag), seed, hist)
            r = pd.DataFrame([x for x in rows if x["system"] == tag]).groupby("vis_level").accuracy.mean()
            print(f"visual '{tag}': test by video level " + ", ".join(f"{a:.3f}" for a in r))
        return self.write(rows, "visual")

    def dct_sweep(self, ks=(3, 6, 10, 15, 21, 28, 44, 66, 100, 150), n_seeds: int = 3, level: int = 2) -> pd.DataFrame:
        """Number of DCT coefficients. Clean video saturates, so we also score moderately degraded video."""
        c = self.corpus
        rows = []
        for k in ks:
            Xtr = self.visual_inputs("landmarks", "dct", 1, c.ids_of("train"), n_dct=k)
            ev = {vkey(s, L): self.visual_inputs("landmarks", "dct", 1, c.ids_of(s), n_dct=k, level=L)
                  for s in ("val", "test") for L in (0, level)}
            for seed in self.seeds[:n_seeds]:
                model, _ = self.train("tcn", Xtr, self.y("train"), ev[vkey("val", 0)], self.y("val"), seed)
                for L in (0, level):
                    rows.append({"experiment": "dct_sweep", "system": f"k={k}", "k": k, "seed": seed, "vis_level": L,
                                 "val_accuracy": accuracy(predict_proba(model, ev[vkey("val", L)]), self.y("val")),
                                 "accuracy": accuracy(predict_proba(model, ev[vkey("test", L)]), self.y("test"))})
            print(f"dct k={k} done")
        return self.write(rows, "dct_sweep")

    # ------------------------------------------------------------------ E4 confound analysis
    def _windowed(self, cid: str, method: str, kind: str, lo: float, hi: float, T: int = 60) -> np.ndarray:
        """Features of the clip restricted to [lo, hi) s, normalised within the window, zero-padded."""
        c = self.corpus
        tr = c.track(cid, method)
        grid = np.arange(lo, hi, 1 / self.cfg["visual"]["frame_rate"])
        if len(grid) < 4:
            grid = np.linspace(lo, lo + 0.1, 4)
        f = resample(c.frame_features(tr, kind), tr["t"], grid)
        f = cmvn(np.concatenate([f, deltas(f, 2)], 1))
        out = np.zeros((T, f.shape[1]), np.float32)
        out[:min(T, len(f))] = f[:T]
        return out.T

    def confound(self, n_seeds: int = 3) -> pd.DataFrame:
        """How much of the visual accuracy could come from the recording session rather than speech?

        Every name was recorded as one block of 30 consecutive takes, so anything that drifts
        between blocks (lighting, head pose, camera placement) is perfectly correlated with the label.
        Probes:
        * rest->rest   : train and test on the lips *at rest* before the word starts. Any accuracy
                         above chance (5%) is information about the session, not the word.
        * speech->speech, speech->rest : train on the word segment; test on the word, and on rest frames.
                         If the word model also recognises rest frames it has learnt a session shortcut.
        * full clip, word masked : the deployed full-clip model tested with the word segment zeroed out.
        """
        c = self.corpus
        endpoints = {cid: speech_endpoints(c.waves[cid], c.sr) for cid in c.ids}
        rows = []
        for method, kind in [("landmarks", "dct"), ("landmarks", "shape"), ("landmarks", "dct+shape"), ("haar", "dct")]:
            speech = {s: np.stack([self._windowed(i, method, kind, endpoints[i][0] - 0.2, endpoints[i][1] + 0.2)
                                   for i in c.ids_of(s)]) for s in ("train", "val", "test")}
            rest = {s: np.stack([self._windowed(i, method, kind, 0.0, endpoints[i][0] - 0.15)
                                 for i in c.ids_of(s)]) for s in ("train", "val", "test")}
            full = {s: self.visual_inputs(method, kind, 1, c.ids_of(s)) for s in ("train", "val", "test")}
            grid = np.arange(full["test"].shape[-1]) / self.cfg["visual"]["frame_rate"]
            masked = full["test"].copy()
            for j, cid in enumerate(c.ids_of("test")):
                on, off = endpoints[cid]
                masked[j][:, (grid >= on - 0.2) & (grid < off + 0.2)] = 0.0
            yt = self.y("test")
            for seed in self.seeds[:n_seeds]:
                m_s, _ = self.train("tcn", speech["train"], self.y("train"), speech["val"], self.y("val"), seed)
                m_r, _ = self.train("tcn", rest["train"], self.y("train"), rest["val"], self.y("val"), seed)
                m_f, _ = self.train("tcn", full["train"], self.y("train"), full["val"], self.y("val"), seed)
                res = {"rest -> rest": accuracy(predict_proba(m_r, rest["test"]), yt),
                       "word -> word": accuracy(predict_proba(m_s, speech["test"]), yt),
                       "word -> rest": accuracy(predict_proba(m_s, rest["test"]), yt),
                       "full clip": accuracy(predict_proba(m_f, full["test"]), yt),
                       "full clip, word masked": accuracy(predict_proba(m_f, masked), yt)}
                rows += [{"experiment": "confound", "system": f"{method} / {kind}", "probe": k, "seed": seed,
                          "accuracy": v} for k, v in res.items()]
            print(f"confound {method}/{kind} done")
        df = self.write(rows, "confound")
        on = np.array([endpoints[i][0] for i in c.ids])
        off = np.array([endpoints[i][1] for i in c.ids])
        pd.DataFrame({"clip_id": c.ids, "onset": on, "offset": off}).to_csv(results_path_for("endpoints"), index=False)
        return df

    # ------------------------------------------------------------------ E5 fusion
    def best_visual(self) -> tuple[str, str, str]:
        """Feature-based TCN front-end with the best validation accuracy averaged over video levels."""
        df = pd.read_csv(results_path_for("visual"))
        df = df[df.system.str.endswith("/ TCN")]
        best = df.groupby("system").val_accuracy.mean().idxmax()
        _, method, kind, _, _ = next(v for v in self.VISUAL_VARIANTS if v[0] == best)
        return best, method, kind

    def fusion(self) -> pd.DataFrame:
        c = self.corpus
        ytr, yva, yte = self.y("train"), self.y("val"), self.y("test")
        vis_tag, method, kind = self.best_visual()
        print(f"visual front-end for fusion: {vis_tag}")
        mf = c.mfcc_config()
        t_audio = c.audio_frame_times(mf)
        splits = ("train", "val", "test")
        V30 = {(s, L): self.visual_inputs(method, kind, 1, c.ids_of(s), level=L) for s in splits for L in VIS_LEVELS}
        V100 = {(s, L): self.visual_inputs(method, kind, 1, c.ids_of(s), target_times=t_audio, level=L)
                for s in splits for L in VIS_LEVELS}
        rng = np.random.default_rng(stable_seed("fusion-val-levels"))

        # ---- visual stream, multi-condition over video quality
        vbank = np.stack([V30[("train", L)] for L in VIS_LEVELS], axis=1)
        vval_levels = rng.integers(0, len(VIS_LEVELS), len(yva))
        vval = np.stack([V30[("val", L)][j] for j, L in enumerate(vval_levels)])
        for seed in self.seeds:
            model, hist = self.train("tcn", vbank, ytr, vval, yva, seed, bank=True)
            probs = {vkey(s, L): predict_proba(model, V30[(s, L)]) for s in ("val", "test") for L in VIS_LEVELS}
            self.save_probs("visual-multi", seed, probs)
            self.save_history("visual-multi", seed, hist)
            self.save_model("visual-multi", seed, model, {"arch": "tcn", "in_dim": int(vbank.shape[2]),
                                                          "roi_method": method, "kind": kind, "deltas": 1})
            print(f"visual-multi seed {seed}: " + ", ".join(f"{accuracy(probs[vkey('test', L)], yte):.3f}"
                                                            for L in VIS_LEVELS))

        # ---- early fusion: concatenate MFCC and (interpolated) visual features at 100 Hz
        abank = c.audio_bank("train", mf, AUDIO_BANK_VERSIONS, seed=7)
        Xva_a, yva_mc = self._multicondition_val(mf)
        val_levels = np.concatenate([vval_levels, rng.integers(0, len(VIS_LEVELS), len(yva))])
        Xva_v = np.stack([V100[("val", L)][j % len(yva)] for j, L in enumerate(val_levels)])
        audio_eval = self._audio_eval_sets(mf)
        variants = {   # video level paired with each of the 10 audio versions
            "av-early-multi": [v % len(VIS_LEVELS) for v in range(AUDIO_BANK_VERSIONS)],
            "av-early-multi-cleanvideo": [0] * AUDIO_BANK_VERSIONS,
        }
        for tag, levels in variants.items():
            bank = np.concatenate([abank, np.stack([V100[("train", L)] for L in levels], axis=1)], axis=2)
            Xva = np.concatenate([Xva_a, Xva_v if tag == "av-early-multi" else
                                  np.concatenate([V100[("val", 0)]] * 2)], axis=1)
            for seed in self.seeds:
                model, hist = self.train("tcn", bank, ytr, Xva, yva_mc, seed, bank=True)
                probs = {}
                for s in ("val", "test"):
                    for k, snr in self.conds:
                        xa = audio_eval[cond_key(s, k, snr)]
                        for L in VIS_LEVELS:
                            probs[avkey(s, k, snr, L)] = predict_proba(model, np.concatenate([xa, V100[(s, L)]], 1))
                self.save_probs(tag, seed, probs)
                self.save_history(tag, seed, hist)
                self.save_model(tag, seed, model, {"arch": "tcn", "in_dim": int(bank.shape[2]),
                                                   "roi_method": method, "kind": kind, "deltas": 1})
                print(f"{tag} seed {seed}: clean {accuracy(probs[avkey('test', 'clean', 'clean', 0)], yte):.3f}, "
                      f"babble 0 dB {accuracy(probs[avkey('test', 'babble', '0', 0)], yte):.3f}")

        results_path(self.cfg, "tables", "fusion_visual_stream.txt").write_text(vis_tag + "\n")
        return self.fusion_late()

    def fusion_late(self) -> pd.DataFrame:
        """Assemble every fusion result from saved posteriors and compute the late-fusion systems.

        Two scenarios:
        * unseen video degradation (main): the visual recogniser and every fusion parameter are
          trained / fitted on *clean* video only; degraded video appears only at test time;
        * matched: the visual recogniser was trained with video-quality augmentation and fusion
          parameters are fitted on validation data at every video level.
        """
        yva = self.y("val")
        p = results_path(self.cfg, "tables", "fusion_visual_stream.txt")
        vis_tag = p.read_text().strip() if p.exists() else self.best_visual()[0]
        rows, alpha_rows = [], []
        for seed in self.seeds:
            pa = self.load_probs("audio-tcn-multi", seed)
            pv_clean, pv_multi = self.load_probs(_slug(vis_tag), seed), self.load_probs("visual-multi", seed)
            rows += self._rows_from_probs(pv_clean, "fusion", "visual-clean", seed)
            rows += self._rows_from_probs(pv_multi, "fusion", "visual-multi", seed)
            for tag in ("av-early-multi", "av-early-multi-cleanvideo"):
                rows += self._rows_from_probs(self.load_probs(tag, seed), "fusion", tag, seed)
            for scenario, pv, fit_levels, suffix in [("unseen", pv_clean, [0], ""),
                                                     ("matched", pv_multi, VIS_LEVELS, "-matched")]:
                val_pairs = [(pa[cond_key("val", k, s)], pv[vkey("val", L)]) for k, s in self.conds for L in fit_levels]
                ys = [yva] * len(val_pairs)
                a_global = select_alpha(val_pairs, ys, "nll")
                ab = select_adaptive_params(val_pairs, ys)
                fused = {f"late-global{suffix}": {}, f"late-oracle{suffix}": {}, f"late-adaptive{suffix}": {}}
                for k, s in self.conds:
                    pa_t = pa[cond_key("test", k, s)]
                    for L in VIS_LEVELS:
                        pv_t = pv[vkey("test", L)]
                        a_or = select_alpha([(pa[cond_key("val", k, s)], pv[vkey("val", L)])], [yva])
                        w = entropy_weights(pa_t, pv_t, *ab)
                        a_nll = select_alpha([(pa[cond_key("val", k, s)], pv[vkey("val", L)])], [yva], "nll")
                        alpha_rows.append({"scenario": scenario, "seed": seed, "noise": k, "snr": s, "vis_level": L,
                                           "alpha_oracle": a_or, "alpha_nll": a_nll, "alpha_global": a_global,
                                           "alpha_adaptive_mean": float(w.mean()), "adaptive_a": ab[0],
                                           "adaptive_b": ab[1]})
                        key = avkey("test", k, s, L)
                        fused[f"late-global{suffix}"][key] = fuse_log_linear(pa_t, pv_t, a_global)
                        fused[f"late-oracle{suffix}"][key] = fuse_log_linear(pa_t, pv_t, a_or)
                        fused[f"late-adaptive{suffix}"][key] = fuse_log_linear(pa_t, pv_t, w)
                for tag, probs in fused.items():
                    rows += self._rows_from_probs(probs, "fusion", tag, seed)
                    self.save_probs(tag, seed, probs)
                print(f"late fusion ({scenario}) seed {seed}: global alpha {a_global}, adaptive (a, b) = {ab}")
        pd.DataFrame(alpha_rows).to_csv(results_path_for("fusion_alpha"), index=False)
        return self.write(rows, "fusion")


_RESULTS_DIR: list[Path] = []


def results_path_for(name: str) -> Path:
    return _RESULTS_DIR[0] / f"{name}.csv"


def _slug(tag: str) -> str:
    s = "".join(ch if ch.isalnum() else "-" for ch in tag.lower()).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return "visual-" + s


EXPERIMENTS = ["audio", "audio_frontend", "visual", "dct_sweep", "confound", "fusion", "fusion_late"]


def run(cfg: Config, experiments: list[str], seeds=None, epochs=None) -> None:
    _RESULTS_DIR[:] = [results_path(cfg, "tables", "_").parent]
    r = Runner(cfg, seeds, epochs)
    for name in EXPERIMENTS:
        if name == "fusion_late" and ("all" in experiments or "fusion" in experiments):
            continue                      # already called at the end of `fusion`
        if name in experiments or "all" in experiments:
            t0 = time.time()
            getattr(r, name)()
            print(f"== {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)

"""Report section for the PyTorch vs. Keras parity study and the deployment benchmark."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .report import INK, SURFACE, TABLE_NAMES, _save, fmt  # noqa: E402

FRAMEWORK_STYLE = {"torch": ("PyTorch", "#6250d6", "o"), "keras": ("Keras / TensorFlow", "#e34948", "s"),
                   "keras audio + torch video": ("Keras audio + PyTorch video", "#e34948", "D")}
PARITY_NAMES = {**TABLE_NAMES, "audio-tcn-clean": "Audio, clean training", "audio-cnn2d-clean": "Audio 2-D CNN, clean",
                "audio-tcn-multi": "Audio, multi-condition", "av-early-multi-cleanvideo": "AV early fusion",
                "late-adaptive": "AV late fusion, adaptive α", "visual-landmarks-dct-shape-tcn": "Visual (DCT+shape)"}
VIDEO = {0: "clean video", 2: "moderate video"}


def parity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds per system, condition and framework (noise types averaged per seed)."""
    d = df.copy()
    d["snr"] = d.snr.astype(str) if "snr" in d else "nan"
    has_noise = d.noise.notna() if "noise" in d else pd.Series(False, index=d.index)
    has_video = d.vis_level.notna() if "vis_level" in d else pd.Series(False, index=d.index)

    def condition(r):
        audio = "clean audio" if r.noise == "clean" else f"{r.snr} dB"
        if r.hasnoise and r.hasvideo:
            return f"{VIDEO.get(int(r.vis_level), r.vis_level)}, {audio}"
        if r.hasvideo:
            return VIDEO.get(int(r.vis_level), str(r.vis_level))
        return audio

    d = d.assign(hasnoise=has_noise, hasvideo=has_video)
    d["condition"] = [condition(r) for r in d.itertuples()]
    per_seed = d.groupby(["system", "condition", "framework", "seed"]).accuracy.mean()
    out = per_seed.groupby(["system", "condition", "framework"]).agg(["mean", "std"]).reset_index()
    sys_order = {s: i for i, s in enumerate(dict.fromkeys(d.system))}

    def cond_rank(c: str) -> tuple:
        video = 0 if c.startswith("clean video") else 1 if c.startswith("moderate video") else -1
        audio = c.split(", ")[-1]
        a = 0 if audio in ("clean audio", "clean video", "moderate video") else -float(audio.replace(" dB", "")) + 100
        return video, a

    out = out.assign(_s=out.system.map(sys_order), _c=out.condition.map(cond_rank))
    return out.sort_values(["_s", "_c"]).drop(columns=["_s", "_c"]).reset_index(drop=True)


def fig_parity(cfg, summ: pd.DataFrame) -> None:
    systems = list(dict.fromkeys(summ.system))
    n = len(systems)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sys_ in zip(axes, systems):
        s = summ[summ.system == sys_]
        conds = list(dict.fromkeys(s.condition))
        x = np.arange(len(conds))
        fws = sorted(dict.fromkeys(s.framework), key=lambda f: f != "torch")      # PyTorch first
        for j, fw in enumerate(fws):
            lab, col, mk = FRAMEWORK_STYLE[fw]
            f = s[s.framework == fw].set_index("condition").reindex(conds)
            off = (j - (len(fws) - 1) / 2) * 0.18
            ax.errorbar(x + off, f["mean"], yerr=f["std"], fmt=mk, color=col, ms=7, capsize=3, lw=1.5, label=lab,
                        markeredgecolor=SURFACE)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(", ", "\n") for c in conds], fontsize=7.5)
        ax.set_title(PARITY_NAMES.get(sys_, sys_), loc="left")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.legend(loc="lower left", fontsize=7.5)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Same models, same data, two frameworks: PyTorch vs. Keras (mean ± std over 5 seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, cfg, "framework_parity.png")


def markdown(cfg, parity: pd.DataFrame | None, meta: pd.DataFrame | None, deploy: pd.DataFrame | None) -> list[str]:
    md: list[str] = []
    if parity is not None:
        summ = parity_summary(parity)
        fig_parity(cfg, summ)
        md += ["## PyTorch vs. Keras", "",
               "Every system below was trained twice from identical features, splits, seeds, augmentation, learning-rate "
               "schedule and model-selection rule: once with PyTorch and once with Keras 3 (TensorFlow backend). The "
               "last block fuses Keras audio posteriors with PyTorch video posteriors.", "",
               "| System | Condition | PyTorch | Keras | Δ (Keras − PyTorch) |", "|---|---|---:|---:|---:|"]
        for sys_ in dict.fromkeys(summ.system):
            s = summ[summ.system == sys_]
            for cond in dict.fromkeys(s.condition):
                t = s[(s.condition == cond) & (s.framework == "torch")]
                k = s[(s.condition == cond) & (s.framework != "torch")]
                if t.empty or k.empty:
                    continue
                tm, ts, km_, ks = t["mean"].iloc[0], t["std"].iloc[0], k["mean"].iloc[0], k["std"].iloc[0]
                md.append(f"| {PARITY_NAMES.get(sys_, sys_)} | {cond} | {fmt(tm, ts)} | {fmt(km_, ks)} | "
                          f"{(km_ - tm) * 100:+.1f} |")
        md.append("")
    if meta is not None:
        g = meta.groupby(["system", "framework"]).agg(parameters=("parameters", "first"), seconds=("seconds", "mean"),
                                                      epoch=("best_epoch", "mean"), device=("device", "first"))
        md += ["Trainable parameters and mean training time per model:", "",
               "| System | PyTorch params | Keras params | PyTorch time | Keras time |", "|---|---:|---:|---:|---:|"]
        for sys_ in dict.fromkeys(meta.system):
            t, k = g.loc[(sys_, "torch")], g.loc[(sys_, "keras")]
            md.append(f"| {PARITY_NAMES.get(sys_, sys_)} | {int(t.parameters):,} | {int(k.parameters):,} | "
                      f"{t.seconds:.1f} s ({t.device}) | {k.seconds:.1f} s ({k.device}) |")
        md.append("")
    if deploy is not None:
        md += ["## Deployment formats (multi-condition audio model)", "",
               "Single-clip CPU latency (batch 1, median of 30 runs); accuracy on the 120-clip clean test set and on the "
               "same clips at 0 dB (mean of four noise types); largest logit difference to the source framework. The "
               "`.keras` file also stores the optimiser state, hence its size.", "",
               "| Framework | Format | Size (KB) | Clean | 0 dB | Max logit diff | Latency (ms) |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for r in deploy.itertuples():
            diff = "–" if pd.isna(r.max_abs_logit_diff) else f"{r.max_abs_logit_diff:.2g}"
            md.append(f"| {r.framework} | {r.format} | {r.size_kb:,.0f} | {r.accuracy_clean * 100:.1f} | "
                      f"{r.accuracy_0db * 100:.1f} | {diff} | {r.latency_ms:.2f} |")
        md.append("")
    return md

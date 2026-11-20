"""Figures and markdown tables generated from the experiment CSVs (nothing here is hand-entered).

Visual conventions: one colour per *system* across every figure (colour follows the entity), a
palette order validated for colour-vision deficiency, a marker shape per series as secondary
encoding, hairline grids and no dual axes. Ordered quantities (video quality level) use a single
blue ramp from light to dark.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, results_path, work_path  # noqa: E402
from .visual.degrade import LEVEL_NAMES  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LEVEL_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal blue ramp (steps 250-700)

# system -> (label, colour, marker). Fixed for the whole report.
SYSTEMS = {
    "audio-tcn-multi": ("Audio only", "#2a78d6", "o"),
    "visual-clean": ("Visual only", "#eb6834", "s"),
    "av-early-multi-cleanvideo": ("AV early fusion", "#1baf7a", "^"),
    "late-oracle": ("AV late fusion, oracle α", "#eda100", "D"),
    "late-adaptive": ("AV late fusion, adaptive α", "#e87ba4", "v"),
    "audio-tcn-clean": ("Audio, clean training", "#008300", "P"),
    "audio-cnn2d-multi": ("Audio 2-D CNN, multi-condition", "#6250d6", "P"),
    "audio-tcn-clean-ss": ("Audio, clean training + spectral subtraction", "#e34948", "X"),
}
TABLE_NAMES = {
    "audio-tcn-multi": "Audio only (multi-condition training)",
    "audio-tcn-clean": "Audio only (clean training)",
    "audio-tcn-clean-ss": "Audio only (clean training + spectral subtraction)",
    "audio-cnn2d-clean": "Audio 2-D CNN (clean training)",
    "audio-cnn2d-multi": "Audio 2-D CNN (multi-condition training)",
    "visual-clean": "Visual only",
    "visual-multi": "Visual only (trained with degraded video)",
    "av-early-multi-cleanvideo": "AV early fusion",
    "av-early-multi": "AV early fusion (trained with degraded video)",
    "late-global": "AV late fusion, one global α",
    "late-oracle": "AV late fusion, oracle α per condition",
    "late-adaptive": "AV late fusion, adaptive (entropy) α",
    "late-global-matched": "AV late fusion, global α (matched)",
    "late-oracle-matched": "AV late fusion, oracle α (matched)",
    "late-adaptive-matched": "AV late fusion, adaptive α (matched)",
}
NOISE_LABELS = {"white": "White noise", "pink": "Pink noise", "babble": "Babble (competing speech)",
                "ambient": "Ambient room noise"}
SNR_ORDER = ["-10", "-5", "0", "5", "10", "20", "clean"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.markersize": 6,
    })


def _tables(cfg: Config) -> Path:
    return results_path(cfg, "tables", "_").parent


def _figdir(cfg: Config) -> Path:
    return results_path(cfg, "figures", "_").parent


def _save(fig, cfg: Config, name: str) -> Path:
    p = _figdir(cfg) / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _snr_x(s) -> float:
    return 25.0 if str(s) == "clean" else float(s)


def _xaxis(ax):
    ax.set_xticks([-10, -5, 0, 5, 10, 15, 20, 25])
    ax.set_xticklabels(["−10", "−5", "0", "5", "10", "15", "20", "clean"])
    ax.set_xlabel("SNR (dB, active speech level)")


def _pct(ax, lo=0.0):
    ticks = np.linspace(0, 1, 6)
    ax.set_ylim(lo, 1.02)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(v * 100)}%" for v in ticks])


def per_condition(df: pd.DataFrame, system: str, noise: str | None, vis_level: int | None = None) -> pd.DataFrame:
    """Mean/std over seeds for each SNR. ``noise=None`` averages the noise types within each seed."""
    s = df[df.system == system]
    if vis_level is not None and "vis_level" in s:
        s = s[s.vis_level == vis_level]
    rows = []
    for snr in SNR_ORDER:
        if snr == "clean":
            sel = s[s.noise == "clean"]
        elif noise is None:
            sel = s[(~s.noise.isin(["clean", "session2"])) & (s.snr.astype(str) == snr)]
        else:
            sel = s[(s.noise == noise) & (s.snr.astype(str) == snr)]
        per_seed = sel.groupby("seed").accuracy.mean()
        if len(per_seed):
            rows.append({"snr": snr, "mean": per_seed.mean(), "std": per_seed.std(ddof=1) if len(per_seed) > 1 else 0})
    return pd.DataFrame(rows)


def _curve(ax, s: pd.DataFrame, system: str, label: str | None = None):
    lab, col, mk = SYSTEMS[system]
    lab = label or lab
    x = s.snr.map(_snr_x)
    ax.plot(x, s["mean"], color=col, marker=mk, label=lab, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.1)
    ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.12, lw=0)


def _hline(ax, mean, std, system):
    lab, col, _ = SYSTEMS[system]
    ax.axhspan(mean - std, mean + std, color=col, alpha=0.12, lw=0)
    ax.axhline(mean, color=col, lw=2, ls="--", label=f"{lab} (unaffected by acoustic noise)")


# ----------------------------------------------------------------------------- figures
def fig_audio(cfg: Config, df: pd.DataFrame) -> Path:
    systems = ["audio-tcn-clean", "audio-tcn-clean-ss", "audio-tcn-multi"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, noise in zip(axes, cfg["noise"]["types"]):
        for sys_ in systems:
            _curve(ax, per_condition(df, sys_, noise), sys_,
                   "Audio, multi-condition training" if sys_ == "audio-tcn-multi" else None)
        ax.set_title(NOISE_LABELS[noise], loc="left")
        _xaxis(ax)
        _pct(ax)
    axes[0].set_ylabel("Test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Audio-only recognition under noise (temporal CNN unless stated; mean ± std over 5 seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return _save(fig, cfg, "audio_snr_curves.png")


def fig_av_grid(cfg: Config, both: pd.DataFrame, noise: str | None, name: str, title: str) -> Path:
    systems = ["audio-tcn-multi", "av-early-multi-cleanvideo", "late-oracle", "late-adaptive"]
    vis = both[both.system == "visual-clean"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, L in zip(axes, sorted(LEVEL_NAMES)):
        v = vis[vis.vis_level == L].groupby("seed").accuracy.mean()
        _hline(ax, v.mean(), v.std(), "visual-clean")
        for sys_ in systems:
            lvl = None if sys_.startswith("audio") else L
            _curve(ax, per_condition(both, sys_, noise, lvl), sys_)
        ax.set_title(f"Video: {LEVEL_NAMES[L]}", loc="left")
        _xaxis(ax)
        _pct(ax)
    axes[0].set_ylabel("Test accuracy" + (" (mean of 4 noise types)" if noise is None else ""))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return _save(fig, cfg, name)


def fig_visual(cfg: Config, df: pd.DataFrame, selected: str | None = None) -> Path:
    order = list(dict.fromkeys(df.system))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(order) + 1.3), sharey=True)
    for ax, L in zip(axes, (0, 2)):
        s = df[df.vis_level == L].groupby("system").accuracy.agg(["mean", "std"]).reindex(order)
        y = np.arange(len(order))
        colors = [SYSTEMS["visual-clean"][1] if n == selected else "#f2a27f" for n in order]
        ax.barh(y, s["mean"], xerr=s["std"], color=colors, height=0.62, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        for yi, val, sd in zip(y, s["mean"], s["std"].fillna(0)):
            ax.text(val + sd + 0.02, yi, f"{val:.1%}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.axvline(0.05, color=MUTED, lw=1, ls=":")
        ax.set_title(f"Test on {LEVEL_NAMES[L]}" + (" (trained on clean video)" if L else ""), loc="left")
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    fig.suptitle("Lip reading: ROI tracking, features and temporal modelling (dark bar = front-end used for fusion; "
                 "dotted = chance)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, cfg, "visual_frontends.png")


def fig_frontend(cfg: Config, df: pd.DataFrame) -> Path:
    order = list(dict.fromkeys(df.system))
    noisy = df[~df.noise.isin(["clean", "session2"])]
    panels = [
        (df[df.noise == "clean"].groupby("system").accuracy.agg(["mean", "std"]), "Clean test"),
        (noisy[noisy.snr.astype(str) == "0"].groupby(["system", "seed"]).accuracy.mean().groupby("system")
         .agg(["mean", "std"]), "0 dB, mean of 4 noise types"),
        (df[df.noise == "session2"].groupby("system").accuracy.agg(["mean", "std"]),
         "Other recording session (clean)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.4 * len(order) + 1.4), sharey=True)
    for ax, (data, title) in zip(axes, panels):
        vals = data.reindex(order)
        y = np.arange(len(order))
        colors = [INK2 if i == 0 else SYSTEMS["audio-tcn-multi"][1] for i in range(len(order))]
        ax.barh(y, vals["mean"], xerr=vals["std"], color=colors, height=0.62, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        for yi, val, sd in zip(y, vals["mean"], vals["std"].fillna(0)):
            ax.text(val + sd + 0.02, yi, f"{val:.0%}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    fig.suptitle("Audio front-end ablation (clean training, 3 seeds; dark bar = baseline)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, cfg, "audio_frontend_ablation.png")


def fig_dct_sweep(cfg: Config, df: pd.DataFrame, compaction: np.ndarray | None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for L, col, mk in [(0, LEVEL_RAMP[0], "o"), (2, LEVEL_RAMP[2], "s")]:
        s = df[df.vis_level == L].groupby("k").accuracy.agg(["mean", "std"])
        ax.plot(s.index, s["mean"], marker=mk, color=col, label=f"test, {LEVEL_NAMES[L]}")
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.15, lw=0)
    ks = sorted(df.k.unique())
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    _pct(ax)
    ax.set_xlabel("Zig-zag DCT coefficients per frame (log scale)")
    ax.set_ylabel("Visual-only accuracy")
    ax.set_title("How many DCT coefficients does lip reading need?", loc="left")
    ax.legend(loc="lower right")
    ax2 = axes[1]
    if compaction is not None:
        k = np.arange(1, len(compaction) + 1)
        ax2.plot(k, compaction, color=SYSTEMS["audio-tcn-multi"][1])
        for kk in (6, 44, 150):
            ax2.plot([kk], [compaction[kk - 1]], marker="o", color=SYSTEMS["audio-tcn-multi"][1])
            ax2.annotate(f"{kk} coeffs: {compaction[kk - 1]:.0%}", (kk, compaction[kk - 1]), xytext=(6, -14),
                         textcoords="offset points", fontsize=8, color=INK2)
        ax2.set_xscale("log")
        ax2.set_xlabel("AC coefficients kept, zig-zag order (log scale)")
        ax2.set_ylabel("Share of AC image energy")
        _pct(ax2)
        ax2.set_title("Energy compaction of the 32×64 mouth ROI", loc="left")
    fig.tight_layout()
    return _save(fig, cfg, "dct_coefficients.png")


def fig_alpha(cfg: Config, alpha: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.9))
    for L in sorted(LEVEL_NAMES):
        a = alpha[(alpha.scenario == "unseen") & (alpha.vis_level == L)]
        a = a.assign(x=a.snr.map(_snr_x)).groupby("x").alpha_nll.mean()
        ax.plot(a.index, a.values, marker="o", color=LEVEL_RAMP[L], label=f"video: {LEVEL_NAMES[L]}")
    _xaxis(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Audio weight α (min. validation log-loss)")
    ax.set_title("Who to trust: audio weight vs. acoustic noise and video quality", loc="left")
    ax.legend(loc="lower right", title=None)
    fig.tight_layout()
    return _save(fig, cfg, "fusion_alpha.png")


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def fig_confusions(cfg: Config, labels, y, panels: list[tuple[str, list[np.ndarray]]], name: str) -> Path:
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6))
    for ax, (title, plist) in zip(np.atleast_1d(axes), panels):
        cm = sum(confusion_matrix(y, p.argmax(1), labels=range(len(labels))) for p in plist).astype(float)
        cm /= cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap=_seq_cmap(), vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.grid(False)
        ax.set_title(f"{title}  (accuracy {np.trace(cm) / len(labels):.0%})", loc="left", fontsize=10)
        ax.set_xlabel("Predicted")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] >= 0.15:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6,
                            color="#ffffff" if cm[i, j] > 0.6 else INK)
    np.atleast_1d(axes)[0].set_ylabel("True name")
    fig.tight_layout()
    return _save(fig, cfg, name)


def fig_training_curves(cfg: Config, histories: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, len(histories), figsize=(4.6 * len(histories), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, h) in zip(axes, histories.items()):
        ep = np.arange(1, len(h["train_acc"]) + 1)
        ax.plot(ep, h["train_acc"], color=SYSTEMS["audio-tcn-multi"][1], label="train (augmented)")
        ax.plot(ep, h["val_acc"], color=SYSTEMS["visual-clean"][1], label="validation")
        ax.axvline(h["best_epoch"] + 1, color=MUTED, lw=1, ls=":")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Epoch")
        _pct(ax)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, cfg, "training_curves.png")


# ----------------------------------------------------------------------------- markdown
def fmt(mean: float, std: float | None = None) -> str:
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean * 100:.1f}"
    return f"{mean * 100:.1f} ± {std * 100:.1f}"


def snr_table(df: pd.DataFrame, systems: list[str], noise: str | None = None, vis_level: int | None = None,
              visual_row: tuple[float, float] | None = None, visual_label: str = "Visual only") -> str:
    cols = SNR_ORDER[::-1]
    lines = ["| System | " + " | ".join("clean" if s == "clean" else f"{s} dB" for s in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for sys_ in systems:
        lvl = None if sys_.startswith("audio") else vis_level
        pc = per_condition(df, sys_, noise, lvl).set_index("snr")
        cells = [fmt(pc.loc[s, "mean"], pc.loc[s, "std"]) if s in pc.index else "–" for s in cols]
        lines.append(f"| {TABLE_NAMES.get(sys_, sys_)} | " + " | ".join(cells) + " |")
    if visual_row is not None:
        lines.append(f"| {visual_label} | " + " | ".join([fmt(*visual_row)] * len(cols)) + " |")
    return "\n".join(lines)


def effective_snr_gain(both: pd.DataFrame, other: str, vis_level: int, levels=(0.5, 0.8)) -> dict:
    """Extra dB of noise the fused system tolerates at equal accuracy (mean of noise types)."""
    def snr_at(system, lvl, acc):
        pc = per_condition(both, system, None, lvl)
        pc = pc[pc.snr != "clean"].assign(x=lambda d: d.snr.astype(float)).sort_values("x")
        x, y = pc.x.to_numpy(), np.maximum.accumulate(pc["mean"].to_numpy())
        if acc < y.min():
            return -np.inf   # better than the target everywhere we measured
        if acc > y.max():
            return None
        return float(np.interp(acc, y, x))
    out = {}
    for a in levels:
        base, fused = snr_at("audio-tcn-multi", None, a), snr_at(other, vis_level, a)
        if base is None or fused is None:
            out[a] = None
        elif fused == -np.inf:
            out[a] = f"> {base - (-10):.0f}"
        else:
            out[a] = f"{base - fused:+.1f}"
    return out


def build_report(cfg: Config) -> None:
    _style()
    t = _tables(cfg)

    def load(name):
        p = t / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None

    audio, visual, fusion = load("audio"), load("visual"), load("fusion")
    frontend, sweep, confound, alpha = load("audio_frontend"), load("dct_sweep"), load("confound"), load("fusion_alpha")
    vis_tag = (t / "fusion_visual_stream.txt").read_text().strip() if (t / "fusion_visual_stream.txt").exists() else None
    md = ["# Results", "", "Auto-generated by `avsr report` from `results/tables/*.csv`. "
          "Accuracy in %, mean ± std over training seeds, 120 test clips per condition.", ""]

    if audio is not None:
        fig_audio(cfg, audio)
        md += ["## Audio only", "", "Mean over the four noise types:", "",
               snr_table(audio, ["audio-tcn-clean", "audio-tcn-clean-ss", "audio-cnn2d-clean", "audio-cnn2d-multi",
                                 "audio-tcn-multi"]), ""]
        for noise in cfg["noise"]["types"]:
            md += [f"**{NOISE_LABELS[noise]}**", "", snr_table(audio, ["audio-tcn-clean", "audio-tcn-clean-ss",
                                                                       "audio-tcn-multi"], noise), ""]
        s2 = audio[audio.noise == "session2"].groupby("system", sort=False).accuracy.agg(["mean", "std"])
        if len(s2):
            md += ["Second recording session (audio recorded weeks earlier with a different microphone, clean):", "",
                   "| System | accuracy |", "|---|---:|"]
            md += [f"| {TABLE_NAMES.get(k, k)} | {fmt(r['mean'], r['std'])} |" for k, r in s2.iterrows()]
            md.append("")
    if frontend is not None:
        fig_frontend(cfg, frontend)
        noisy = frontend[~frontend.noise.isin(["clean", "session2"])]

        def agg(sel):
            return sel.groupby(["system", "seed"]).accuracy.mean().groupby("system", sort=False).agg(["mean", "std"])
        clean = agg(frontend[frontend.noise == "clean"])
        n10, n0 = agg(noisy[noisy.snr.astype(str) == "10"]), agg(noisy[noisy.snr.astype(str) == "0"])
        s2 = agg(frontend[frontend.noise == "session2"])
        dims = frontend.groupby("system", sort=False).dim.first()
        order = list(dict.fromkeys(frontend.system))
        clean = clean.reindex(order)
        md += ["## Audio front-end ablation", "", "Clean training, temporal CNN, 3 seeds. Noisy columns average the four "
               "noise types.", "", "| Front-end | dim | clean | 10 dB | 0 dB | other session |", "|---|---:|---:|---:|---:|---:|"]
        for name in clean.index:
            md.append(f"| {name} | {dims[name]} | {fmt(*clean.loc[name])} | {fmt(*n10.loc[name])} | {fmt(*n0.loc[name])} | "
                      f"{fmt(*s2.loc[name]) if name in s2.index else '–'} |")
        md.append("")
    if visual is not None:
        fig_visual(cfg, visual, vis_tag)
        order = list(dict.fromkeys(visual.system))
        md += ["## Visual only (lip reading)", "", "Trained on clean video, tested on clean and degraded video.", "",
               "| ROI / features / model | dim | " + " | ".join(LEVEL_NAMES[L] for L in sorted(LEVEL_NAMES)) + " |",
               "|---|---:|" + "---:|" * len(LEVEL_NAMES)]
        for name in order:
            s = visual[visual.system == name]
            cells = [fmt(*s[s.vis_level == L].accuracy.agg(["mean", "std"])) for L in sorted(LEVEL_NAMES)]
            dim = int(s.dim.iloc[0])
            md.append(f"| {name} | {dim if dim else '–'} | " + " | ".join(cells) + " |")
        md.append("")
    if sweep is not None:
        fig_dct_sweep(cfg, sweep, _energy_compaction(cfg))
        md += ["## Number of DCT coefficients", "", "| k | " + " | ".join(str(k) for k in sorted(sweep.k.unique())) + " |",
               "|---|" + "---:|" * sweep.k.nunique()]
        for L in sorted(sweep.vis_level.unique()):
            s = sweep[sweep.vis_level == L].groupby("k").accuracy.mean()
            md.append(f"| {LEVEL_NAMES[L]} | " + " | ".join(f"{v * 100:.1f}" for v in s) + " |")
        md.append("")
    if confound is not None:
        probes = list(dict.fromkeys(confound.probe))
        md += ["## Recording-session confound", "", "Each name was recorded as one block of 30 takes. Chance is 5%.", "",
               "| Visual front-end | " + " | ".join(probes) + " |", "|---|" + "---:|" * len(probes)]
        for name, g in confound.groupby("system", sort=False):
            md.append(f"| {name} | " + " | ".join(fmt(*g[g.probe == p].accuracy.agg(["mean", "std"])) for p in probes) + " |")
        md.append("")
    if fusion is not None and audio is not None:
        both = pd.concat([audio, fusion], ignore_index=True)
        fig_av_grid(cfg, both, None, "av_snr_curves.png",
                    "Fusion when the video degrades in ways never seen in training (mean of 4 noise types)")
        fig_av_grid(cfg, both, "babble", "av_snr_curves_babble.png",
                    "Same, babble noise only: competing speech makes the audio recogniser confidently wrong")
        systems = ["audio-tcn-multi", "av-early-multi-cleanvideo", "late-global", "late-oracle", "late-adaptive"]
        vis = fusion[fusion.system == "visual-clean"]
        md += ["## Audio-visual fusion", "",
               f"Visual front-end: **{vis_tag}** (best mean validation accuracy). Audio: temporal CNN with "
               "multi-condition training. Rows: systems; columns: SNR, averaged over the four noise types.", "",
               "### Scenario A — video degradation unseen in training (main)", "",
               "The visual recogniser, the early-fusion network and every fusion parameter are trained or fitted on clean "
               "video only; degraded video appears only at test time.", ""]
        gains = []
        for L in sorted(LEVEL_NAMES):
            v = vis[vis.vis_level == L].groupby("seed").accuracy.mean()
            md += [f"**Video: {LEVEL_NAMES[L]}**", "", snr_table(both, systems, None, L, (v.mean(), v.std())), ""]
            for o in ["av-early-multi-cleanvideo", "late-adaptive"]:
                g = effective_snr_gain(both, o, L)
                gains.append(f"| {LEVEL_NAMES[L]} | {TABLE_NAMES[o]} | " +
                             " | ".join("n/a" if x is None else f"{x} dB" for x in g.values()) + " |")
        vb = vis[vis.vis_level == 2].groupby("seed").accuracy.mean()
        md += ["**Babble noise, moderate video degradation**", "",
               snr_table(both, systems, "babble", 2, (vb.mean(), vb.std())), ""]
        md += ["Effective SNR gain over audio only (extra dB of noise tolerated at the same accuracy; '> x' means the "
               "fused system stays above the target down to the lowest SNR tested, −10 dB):", "",
               "| Video | System | at 50% accuracy | at 80% accuracy |", "|---|---|---:|---:|"] + gains + [""]
        md += ["### Scenario B — video degradation seen in training", "",
               "The visual recogniser and early fusion are trained with video-quality augmentation; fusion parameters are "
               "fitted at every video level. Accuracy at 0 dB and −10 dB SNR (mean of the four noise types):", "",
               "| System | " + " | ".join(f"{LEVEL_NAMES[L]}, 0 dB | {LEVEL_NAMES[L]}, −10 dB" for L in (2, 3)) + " |",
               "|---|---:|---:|---:|---:|"]
        for sys_ in ["visual-multi", "av-early-multi", "late-global-matched", "late-oracle-matched",
                     "late-adaptive-matched"]:
            cells = []
            for L in (2, 3):
                pc = per_condition(both, sys_, None, L).set_index("snr") if sys_ != "visual-multi" else None
                for snr in ("0", "-10"):
                    if pc is None:
                        v = both[(both.system == sys_) & (both.vis_level == L)].groupby("seed").accuracy.mean()
                        cells.append(fmt(v.mean(), v.std()))
                    else:
                        cells.append(fmt(pc.loc[snr, "mean"], pc.loc[snr, "std"]))
            md.append(f"| {TABLE_NAMES[sys_]} | " + " | ".join(cells) + " |")
        md.append("")
        if alpha is not None:
            fig_alpha(cfg, alpha)
        _confusion_figure(cfg)
    _training_figure(cfg, vis_tag)
    parity, meta, deploy = load("framework_parity"), load("framework_parity_meta"), load("deployment")
    if parity is not None or deploy is not None:
        from .report_frameworks import markdown as framework_markdown
        md += framework_markdown(cfg, parity, meta, deploy)
    (t / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    try:
        from .illustrate import build_illustrations
        build_illustrations(cfg)
    except FileNotFoundError as exc:      # dataset not available (e.g. CI)
        print(f"illustrations skipped: {exc}")
    print(f"report written to {t / 'RESULTS.md'} and {_figdir(cfg)}")


def _energy_compaction(cfg: Config) -> np.ndarray | None:
    from .corpus import Corpus
    from .visual.features import energy_compaction
    try:
        c = Corpus(cfg)
        tr = c.tracks("landmarks")
        frames = np.concatenate([tr[i]["roi"][::5] for i in c.ids_of("train")])
        return energy_compaction(frames)[:300]
    except FileNotFoundError:
        return None


def _load_probs(cfg: Config, tag: str, seed: int) -> dict:
    p = work_path(cfg, "probs", f"{tag}__seed{seed}.npz")
    if not p.exists():
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _confusion_figure(cfg: Config) -> None:
    from .corpus import Corpus
    c = Corpus(cfg)
    seeds = cfg["training"]["seeds"]
    pa = [_load_probs(cfg, "audio-tcn-multi", s).get("test|babble|0") for s in seeds]
    pav = [_load_probs(cfg, "av-early-multi-cleanvideo", s).get("test|babble|0|video2") for s in seeds]
    pa, pav = [p for p in pa if p is not None], [p for p in pav if p is not None]
    if pa and pav:
        fig_confusions(cfg, c.labels, c.y_of("test"), [("Audio only, babble 0 dB", pa),
                                                        ("AV early fusion, babble 0 dB, moderate video", pav)],
                       "confusion_babble_0db.png")


def _training_figure(cfg: Config, vis_tag: str | None) -> None:
    seed = cfg["training"]["seeds"][0]
    from .experiments import _slug
    tags = {"Audio (multi-condition)": "audio-tcn-multi", "AV early fusion": "av-early-multi-cleanvideo"}
    if vis_tag:
        tags = {"Audio (multi-condition)": "audio-tcn-multi", "Visual": _slug(vis_tag),
                "AV early fusion": "av-early-multi-cleanvideo"}
    hist = {}
    for title, tag in tags.items():
        p = work_path(cfg, "probs", f"{tag}__seed{seed}.history.json")
        if p.exists():
            hist[title] = json.loads(p.read_text())
    if hist:
        fig_training_curves(cfg, hist)

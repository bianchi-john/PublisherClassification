"""Every figure in the paper, one function each.

**Style contract: the output is meant to be visually indistinguishable from the
figures in the rejected TCSS paper.** The old notebooks set matplotlib's PGF
backend with `text.usetex=True` and then re-declared slightly different rcParams
before each figure (font sizes 11, 14 or 16 depending on the plot), so the per
figure settings below are not tidy-ups -- they are transcriptions. Changing one
changes how that figure looks next to the others in the PDF.

Each figure's geometry (figsize, bar width, offsets, hatch, colour, y-limits,
tick values, legend title and placement) is likewise copied from the code that
produced the published version, cross-checked against the text extracted from
the published PDFs themselves in `Paper/NewIRForGood/img/`.

Two conventions the old code did *not* keep and that are fixed here:

* colour and hatch per model come from ``config`` alone. Notebook 5 used two
  contradictory conventions in one file, so a model changed colour between
  figures of the same paper.
* the model name is never re-bound to data; it travels with it from
  ``config.FINETUNED_MODELS``.

Default output names are the paper's own filenames, so `PUBLISH = True` in the
notebook replaces the figures in `Paper/NewIRForGood/img/` in place and the
LaTeX source needs no edit.
"""
from __future__ import annotations

import shutil
import warnings
from contextlib import contextmanager, nullcontext as _nullcontext

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as C
from .metrics import baselines

# Baseline line styles. Both baselines are drawn across the FULL width of every
# metric group -- drawing them over half each would suggest they apply only to
# some of the models. They share one dash period and are offset by half of it,
# so where the two values coincide (Macro Recall, where both are 1/K) the line
# alternates red and violet instead of the two dash patterns interleaving into
# an illegible mess.
BASELINE_STYLES = {
    "majority": dict(color="red", linestyle=(0, (7, 7))),
    "random": dict(color="darkviolet", linestyle=(7, (7, 7))),
    # TF-IDF + logistic regression is an informative, trained baseline, not a
    # constant one, so it gets a solid line to set it apart from the two dashed
    # non-informative baselines above.
    "tfidf": dict(color="#117733", linestyle=(0, (1, 0))),
}
BASELINE_LABELS = {
    "majority": "Majority Class Baseline",
    "random": "Uniform Random Baseline",
    "tfidf": "TF-IDF + Log. Reg.",
}
# The legend key is a short segment, so the half-period offset that makes the
# two lines alternate on the axes would start the violet key inside a gap and
# render it as a stub. The keys therefore use the same dash pattern with no
# offset, and differ by colour alone.
BASELINE_LEGEND_STYLES = {
    k: dict(color=v["color"], linestyle=(0, (4, 3)))
    for k, v in BASELINE_STYLES.items()
}

METRIC_ORDER = [
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Tolerant_0.5",
    "Tolerant_1.0",
]

# The two metric-label sets differ between the two figures in the paper: the
# box plot prints "F1 Macro", the publisher bar chart prints "F1 (Macro)".
BOXPLOT_METRIC_LABELS = [
    "Accuracy",
    "Precision\n(Macro)",
    "Recall\n(Macro)",
    "F1\nMacro",
    "Tolerant\nAcc (0.5)",
    "Tolerant\nAcc (1.0)",
]
BARCHART_METRIC_LABELS = [
    "Accuracy",
    "Precision\n(Macro)",
    "Recall\n(Macro)",
    "F1\n(Macro)",
    "Tolerant\nAcc (0.5)",
    "Tolerant\nAcc (1.0)",
]

# Reliability tick labels, exactly as they appear in the published figures.
RELIABILITY_TICKS = [
    r"\textit{l1} (0-39)",
    r"\textit{l2} (40-59)",
    r"\textit{l3} (60-74)",
    r"\textit{l4} (75-99)",
    r"\textit{l5} (100)",
]
RELIABILITY_TICKS_PLAIN = ["l1 (0-39)", "l2 (40-59)", "l3 (60-74)", "l4 (75-99)", "l5 (100)"]

# Colours/hatches for the five reliability classes in the distribution figure.
CLASS_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
CLASS_HATCHES = ["///", "\\\\", "xxx", "---", "+++"]

_USETEX = False


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def set_style(use_latex: bool = True) -> None:
    """Enable the paper's PGF/LaTeX rendering.

    The published figures were produced with the PGF backend and
    ``text.usetex=True``; that is what gives them their Computer Modern serif
    type. Reproducing them therefore needs a working ``pdflatex`` on PATH.

    If ``pdflatex`` is missing we fall back to mathtext with a Computer Modern
    look-alike and warn: the layout stays right but the glyphs will not be
    pixel-identical, so do the final render on a machine with TeX.
    """
    global _USETEX
    have_tex = shutil.which("pdflatex") is not None

    if use_latex and have_tex:
        mpl.use("pgf", force=True)
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": [],
                "pgf.rcfonts": False,
            }
        )
        _USETEX = True
    else:
        if use_latex and not have_tex:
            warnings.warn(
                "pdflatex not found: falling back to mathtext. Layout will match "
                "the paper but the fonts will not. Install TeX (e.g. TinyTeX) and "
                "re-run to reproduce the published figures exactly.",
                stacklevel=2,
            )
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "cm",
            }
        )
        _USETEX = False

    plt.rcParams.update({"savefig.dpi": C.FIG_DPI, "figure.dpi": 110})


@contextmanager
def _plain_pdf_backend():
    """Render outside PGF, the way the published publisher-level figure was.

    `6 - Domain inpact analisys.ipynb` set ``"text.usetex": False`` and never
    switched to the PGF backend, so `domain_metrics_comparison.pdf` in the paper
    is the one figure NOT typeset by LaTeX: its text is matplotlib's fallback
    serif, not Computer Modern. The published figure set is therefore not
    internally consistent.

    This context reproduces that, so the regenerated figure matches the paper.
    Pass ``match_paper=False`` to `fig_publisher_metrics` to render it like the
    other six instead, which is what you want if you ever decide to make the
    whole set uniform.
    """
    prev_backend = mpl.get_backend()
    prev = {k: plt.rcParams[k] for k in ("text.usetex", "font.family", "font.serif")}
    mpl.use("pdf", force=True)
    # font.serif must be restored to matplotlib's default list: the PGF settings
    # leave it empty (LaTeX picks the font there), which outside PGF makes the
    # serif family unresolvable and silently falls back to the sans default.
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": mpl.rcParamsDefault["font.serif"],
    })
    try:
        yield
    finally:
        mpl.use(prev_backend, force=True)
        plt.rcParams.update(prev)


@contextmanager
def _figure_style(**overrides):
    """Apply one figure's rcParams, then restore. See the module docstring."""
    saved = {k: plt.rcParams[k] for k in overrides}
    plt.rcParams.update(overrides)
    try:
        yield
    finally:
        plt.rcParams.update(saved)


def _ticks(latex: list[str], plain: list[str]) -> list[str]:
    return latex if _USETEX else plain


def _save(fig: plt.Figure, name: str, tight_bbox: bool = False) -> plt.Figure:
    C.ensure_output_dirs()
    kwargs = {"bbox_inches": "tight"} if tight_bbox else {}
    fig.savefig(C.FIGURES_DIR / f"{name}.pdf", **kwargs)
    return fig


def _models_in(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].astype(str))
    return [m for m in C.MODEL_ORDER_WITH_GPT if m in present]


def _bar_kwargs(model: str) -> dict:
    return {
        "color": C.MODEL_COLORS.get(model, "gray"),
        "hatch": C.MODEL_HATCHES.get(model, ""),
        "edgecolor": "black",
    }


def _model_handles(models: list[str], linewidth: float | None = None) -> list[Patch]:
    kw = {} if linewidth is None else {"linewidth": linewidth}
    return [
        Patch(facecolor=C.MODEL_COLORS[m], edgecolor="black",
              hatch=C.MODEL_HATCHES[m], label=m, **kw)
        for m in models
    ]


# --------------------------------------------------------------------------
# 1. Article-level metrics across folds -> results_boxplot_baselines
# --------------------------------------------------------------------------
def fig_article_metrics_boxplot(
    per_fold: pd.DataFrame,
    y_true_pool,
    name: str = "results_boxplot_baselines",
    include_gpt: bool = False,
    tfidf: dict | None = None,
) -> plt.Figure:
    """Per-fold distribution of the six metrics, with baselines.

    This is the figure the RoBERTa/Llama swap corrupted: the old code read
    `llama.csv` into `robertaDf` and then stamped the wrong label on it.
    """
    order = C.MODEL_ORDER_WITH_GPT if include_gpt else C.MODEL_ORDER
    models = [m for m in order if m in set(per_fold["model"].astype(str))]
    base = baselines(y_true_pool)

    with _figure_style(**{
        "axes.labelsize": 16, "font.size": 14, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    }):
        x = np.arange(len(METRIC_ORDER))
        width = 0.15
        fig, ax = plt.subplots(figsize=(14, 6))

        for i, model in enumerate(models):
            sub = per_fold[per_fold["model"].astype(str) == model]
            data = [sub[m].to_numpy() for m in METRIC_ORDER]
            bp = ax.boxplot(
                data, positions=x + i * width, widths=width, patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(edgecolor="black", linewidth=1),
            )
            for box in bp["boxes"]:
                box.set(facecolor=C.MODEL_COLORS[model],
                        hatch=C.MODEL_HATCHES[model], alpha=0.9)

        # Both baselines on every metric. The published figure drew only the
        # one each metric "naturally challenges", which hid that the majority
        # baseline has a non-zero macro F1 and that the random baseline is not
        # at a flat 1/K on macro F1.
        offset = width * (len(models) - 1) / 2
        centers = x + offset
        half = width * len(models) / 2 * 0.95
        for i, metric in enumerate(METRIC_ORDER):
            for which in ("majority", "random"):
                ax.hlines(base[which][metric], centers[i] - half, centers[i] + half,
                          linewidth=1.8, zorder=5, **BASELINE_STYLES[which])
            if tfidf is not None:
                ax.hlines(tfidf[metric], centers[i] - half, centers[i] + half,
                          linewidth=2.2, zorder=6, **BASELINE_STYLES["tfidf"])

        ax.set_xticks(centers)
        ax.set_xticklabels(BOXPLOT_METRIC_LABELS)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Classification performance")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        handles = _model_handles(models)
        if tfidf is not None:
            handles.append(
                Line2D([0], [0], label=BASELINE_LABELS["tfidf"], linewidth=2.2,
                       **BASELINE_STYLES["tfidf"]))
        handles += [
            Line2D([0], [0], label=BASELINE_LABELS["majority"],
                   **BASELINE_LEGEND_STYLES["majority"]),
            Line2D([0], [0], label=BASELINE_LABELS["random"],
                   **BASELINE_LEGEND_STYLES["random"]),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True)
        fig.tight_layout()
        _save(fig, name)
    return fig


# --------------------------------------------------------------------------
# 2. Publisher-level metrics -> domain_metrics_comparison
# --------------------------------------------------------------------------
def fig_publisher_metrics(
    metrics: pd.DataFrame, y_true_pub, name: str = "domain_metrics_comparison",
    match_paper: bool = True, tfidf: dict | None = None,
) -> plt.Figure:
    """Publisher-level metrics with baselines.

    ``match_paper=True`` reproduces the published figure exactly, including the
    fact that it alone was rendered without LaTeX (see ``_plain_pdf_backend``).
    Set it to False to typeset it like the other six figures.
    """
    models = [m for m in C.MODEL_ORDER if m in set(metrics["model"].astype(str))]
    base = baselines(y_true_pub)

    stack = _plain_pdf_backend() if (match_paper and _USETEX) else _nullcontext()
    with stack, _figure_style(**{
        "axes.labelsize": 11, "font.size": 11, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    }):
        x = np.arange(len(METRIC_ORDER))
        width = 0.18
        offsets = [(i - (len(models) - 1) / 2) * width for i in range(len(models))]
        fig, ax = plt.subplots(figsize=(11, 6))

        for i, model in enumerate(models):
            row = metrics[metrics["model"].astype(str) == model].iloc[0]
            ax.bar(x + offsets[i], [row[m] for m in METRIC_ORDER], width,
                   label=model, linewidth=0.7, **_bar_kwargs(model))

        half = (len(models) * width) / 2 + 0.02
        for i, metric in enumerate(METRIC_ORDER):
            for which in ("majority", "random"):
                st = BASELINE_STYLES[which]
                line = ax.hlines(
                    base[which][metric], x[i] - half, x[i] + half,
                    colors=st["color"], linestyles=st["linestyle"],
                    linewidth=2.2, zorder=10,
                )
                line.set_path_effects(
                    [pe.Stroke(linewidth=4, foreground="black", alpha=0.7), pe.Normal()]
                )
            if tfidf is not None:
                st = BASELINE_STYLES["tfidf"]
                line = ax.hlines(
                    tfidf[metric], x[i] - half, x[i] + half,
                    colors=st["color"], linestyles=st["linestyle"],
                    linewidth=2.6, zorder=11,
                )
                line.set_path_effects(
                    [pe.Stroke(linewidth=4.4, foreground="black", alpha=0.7), pe.Normal()]
                )

        ax.set_xticks(x)
        ax.set_xticklabels(BARCHART_METRIC_LABELS, rotation=0, fontsize=10)
        ax.set_ylim(0, 1.05)
        # Pinned explicitly: with ylim=1.05 newer matplotlib adds a tick at 1.2,
        # outside the axes, which the published figure does not have.
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel("Score", fontsize=12)

        handles = _model_handles(models)
        if tfidf is not None:
            handles.append(
                Line2D([], [], linewidth=2.2, label=BASELINE_LABELS["tfidf"],
                       **BASELINE_STYLES["tfidf"]))
        handles += [
            Line2D([], [], linewidth=2.0, label=BASELINE_LABELS["majority"],
                   **BASELINE_LEGEND_STYLES["majority"]),
            Line2D([], [], linewidth=2.0, label=BASELINE_LABELS["random"],
                   **BASELINE_LEGEND_STYLES["random"]),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  ncol=3, frameon=True, fancybox=True, framealpha=0.95)
        fig.tight_layout()
        _save(fig, name, tight_bbox=True)
    return fig


# --------------------------------------------------------------------------
# 3. Error rate per reliability class -> error_rate_analysis
# --------------------------------------------------------------------------
def fig_error_rate_by_class(err: pd.DataFrame, name: str = "error_rate_analysis") -> plt.Figure:
    models = [m for m in C.MODEL_ORDER if m in set(err["model"].astype(str))]

    # These are the rcParams actually in force when the published version was
    # drawn (set several cells earlier in the old notebook, not next to the
    # plotting code): legend at 11pt and ticks at 12pt, which is why the
    # published PDF embeds CMR10 alongside CMR12.
    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    }):
        x = np.arange(C.N_CLASSES)
        width = 0.19
        offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            sub = err[err["model"].astype(str) == model].sort_values("y_true")
            ax.bar(x + offsets[i], sub["error_rate"].to_numpy(), width,
                   label=model, zorder=3, **_bar_kwargs(model))

        ax.set_ylabel("Error Rate")
        ax.set_xlabel("True Label")
        ax.set_xticks(x)
        ax.set_xticklabels(_ticks(RELIABILITY_TICKS, RELIABILITY_TICKS_PLAIN))
        ax.set_ylim(0, 0.9)
        ax.grid(axis="y", linestyle="-", alpha=0.5, color="gray", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", title="Model", framealpha=1.0, edgecolor="black")
        fig.tight_layout()
        _save(fig, name)
    return fig


# --------------------------------------------------------------------------
# 4. Mean error direction per reliability class -> mean_error_direction
# --------------------------------------------------------------------------
def fig_med_by_class(med: pd.DataFrame, name: str = "mean_error_direction") -> plt.Figure:
    models = [m for m in C.MODEL_ORDER if m in set(med["model"].astype(str))]

    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 12,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
    }):
        x = np.arange(C.N_CLASSES)
        width = 0.19
        offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.axhline(0, color="black", linewidth=1.2, zorder=2)
        for i, model in enumerate(models):
            sub = med[med["model"].astype(str) == model].sort_values("y_true")
            ax.bar(x + offsets[i], sub["med"].to_numpy(), width,
                   label=model, zorder=3, **_bar_kwargs(model))

        ax.set_ylabel(r"Mean Error Direction $(\hat{y} - y)$")
        ax.set_xlabel("True Label (Reliability)")
        ax.set_xticks(x)
        ax.set_xticklabels(_ticks(RELIABILITY_TICKS, RELIABILITY_TICKS_PLAIN))
        ax.grid(axis="y", linestyle="-", alpha=0.3, color="gray", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", title="Model", framealpha=1.0, edgecolor="black")
        fig.tight_layout()
        _save(fig, name)
    return fig


# --------------------------------------------------------------------------
# 5. Reliability distribution across leanings -> orientationDistributionInverted
# --------------------------------------------------------------------------
def fig_leaning_distribution(
    tab: pd.DataFrame, name: str = "orientationDistributionInverted"
) -> plt.Figure:
    """Article counts per political leaning, grouped by reliability class."""
    leanings = [l for l in C.LEANING_ORDER if l in list(tab.index)]
    tab = tab.loc[leanings]

    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 12,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
    }):
        n_labels = len(tab.columns)
        group_width = 0.8
        bar_width = group_width / n_labels
        x = np.arange(len(tab.index))
        fig, ax = plt.subplots(figsize=(10, 6))

        tick_labels = _ticks(RELIABILITY_TICKS, RELIABILITY_TICKS_PLAIN)
        for i, col in enumerate(tab.columns):
            offset = (i - n_labels / 2 + 0.5) * bar_width
            ax.bar(x + offset, tab[col].to_numpy(), width=bar_width,
                   color=CLASS_COLORS[i % len(CLASS_COLORS)],
                   hatch=CLASS_HATCHES[i % len(CLASS_HATCHES)],
                   edgecolor="black", label=tick_labels[i])

        ax.set_xlabel("Leaning")
        ax.set_ylabel(r"N\textdegree{} of articles" if _USETEX else "N° of articles")
        ax.set_xticks(x)
        ax.set_xticklabels(list(tab.index), rotation=0, ha="center")
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(title="Reliability label", frameon=True, edgecolor="black")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        _save(fig, name, tight_bbox=True)
    return fig


# --------------------------------------------------------------------------
# 6. Error rate by leaning -> politicalErrorRates
# --------------------------------------------------------------------------
def fig_error_rate_by_leaning(er: pd.DataFrame, name: str = "politicalErrorRates") -> plt.Figure:
    """Grouped bars, one group per leaning.

    The published version went through ``seaborn.barplot(..., errorbar=None)``.
    The bars are drawn directly here with seaborn's own dodge geometry -- total
    group width 0.8 split evenly across the hue levels -- so the result matches
    without depending on which seaborn version is installed.
    """
    models = [m for m in C.MODEL_ORDER if m in set(er["model"].astype(str))]
    leanings = [l for l in C.LEANING_ORDER if l in set(er["leaning"].astype(str))]

    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 14,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.linewidth": 1.5, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    }):
        x = np.arange(len(leanings))
        width = 0.8 / len(models)
        offsets = [(i - (len(models) - 1) / 2) * width for i in range(len(models))]
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            sub = er[er["model"].astype(str) == model]
            vals = [float(sub[sub["leaning"].astype(str) == l]["error_rate"].iloc[0])
                    for l in leanings]
            ax.bar(x + offsets[i], vals, width, label=model, linewidth=1.2,
                   **_bar_kwargs(model))

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(leanings)
        ax.set_ylabel("Error Rate (Mean)")
        ax.set_xlabel("Leaning")
        ax.legend(handles=_model_handles(models), title="Model", loc="upper center")
        fig.tight_layout()
        _save(fig, name, tight_bbox=True)
    return fig


# --------------------------------------------------------------------------
# 7. Mean error direction by leaning -> politicalErrorsDirection
# --------------------------------------------------------------------------
def fig_med_by_leaning(med: pd.DataFrame, name: str = "politicalErrorsDirection") -> plt.Figure:
    models = [m for m in C.MODEL_ORDER if m in set(med["model"].astype(str))]
    leanings = [l for l in C.LEANING_ORDER if l in set(med["leaning"].astype(str))]

    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    }):
        x = np.arange(len(leanings))
        width = 0.18
        offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            sub = med[med["model"].astype(str) == model]
            vals = [float(sub[sub["leaning"].astype(str) == l]["med"].iloc[0]) for l in leanings]
            ax.bar(x + offsets[i], vals, width, label=model, linewidth=0.5,
                   zorder=3, **_bar_kwargs(model))

        ax.axhline(0, color="black", linewidth=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [l.replace("Non labelled", "Non labelled\nLeaning") for l in leanings]
        )
        ax.set_ylabel(r"Mean Error Direction $(\hat{y} - y)$")
        ax.set_ylim(-1.1, 1.2)
        ax.set_yticks(np.arange(-1.0, 1.25, 0.25))
        ax.grid(axis="y", linestyle="-", alpha=0.5, color="gray", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles=_model_handles(models), title="Model", loc="upper left",
                  framealpha=1.0, edgecolor="black")
        fig.tight_layout()
        _save(fig, name)
    return fig


# --------------------------------------------------------------------------
# 8. Aggregation learning curve (new; no published counterpart)
# --------------------------------------------------------------------------
def fig_aggregation_curve(curve: pd.DataFrame, name: str = "aggregation_curve") -> plt.Figure:
    """Publisher accuracy vs number of articles voted.

    k=1 is the article-level baseline measured on the publisher unit, so the
    rise from k=1 is what aggregation contributes -- the comparison the paper
    previously conflated with a change of both unit and weighting.

    Styled to sit beside the published figures rather than to copy one.
    """
    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    }):
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = {"BERT": "o", "RoBERTa": "s", "Llama": "^", "Mistral": "D"}
        for model in _models_in(curve):
            sub = curve[curve["model"].astype(str) == model].sort_values("k")
            ax.errorbar(sub["k"], sub["accuracy_mean"], yerr=sub["accuracy_std"],
                        marker=markers.get(model, "o"), capsize=3, linewidth=1.8,
                        label=model, color=C.MODEL_COLORS.get(model, "gray"))
        n = int(curve["n_publishers"].iloc[0])
        ax.set_xlabel(r"Articles aggregated per publisher ($k$)")
        ax.set_ylabel("Publisher-level Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Model", framealpha=1.0, edgecolor="black")
        ax.set_title(f"Fixed population of {n} publishers at every $k$", fontsize=12)
        fig.tight_layout()
        _save(fig, name)
    return fig


# --------------------------------------------------------------------------
# 9. Zero-shot comparison helpers (kept for include_gpt=True; unused by default)
# --------------------------------------------------------------------------
def fig_metrics_with_gpt(
    metrics: pd.DataFrame, level: str, name: str | None = None
) -> plt.Figure:
    name = name or f"{level}_metrics_with_gpt"
    models = _models_in(metrics)

    with _figure_style(**{
        "axes.labelsize": 11, "font.size": 11, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    }):
        x = np.arange(len(METRIC_ORDER))
        width = 0.8 / len(models)
        offsets = [(i - (len(models) - 1) / 2) * width for i in range(len(models))]
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, model in enumerate(models):
            row = metrics[metrics["model"].astype(str) == model].iloc[0]
            ax.bar(x + offsets[i], [row[m] for m in METRIC_ORDER], width,
                   label=model, linewidth=0.7, **_bar_kwargs(model))
        ax.set_xticks(x)
        ax.set_xticklabels(BARCHART_METRIC_LABELS, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(handles=_model_handles(models), loc="upper left", ncol=3,
                  framealpha=0.95)
        fig.tight_layout()
        _save(fig, name, tight_bbox=True)
    return fig


def fig_prediction_distribution(
    preds: pd.DataFrame, name: str = "prediction_distribution"
) -> plt.Figure:
    models = _models_in(preds)

    with _figure_style(**{
        "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 12,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
    }):
        x = np.arange(C.N_CLASSES)
        width = 0.8 / (len(models) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        truth = preds[preds["model"].astype(str) == models[0]]["y_true"]
        ax.bar(x - 0.4 + width / 2, [float((truth == c).mean()) for c in C.CLASS_CODES],
               width, label="Ground truth", facecolor="white", edgecolor="black", hatch="..")
        for i, model in enumerate(models, start=1):
            sub = preds[preds["model"].astype(str) == model]
            ax.bar(x + i * width - 0.4 + width / 2,
                   [float((sub["y_pred"] == c).mean()) for c in C.CLASS_CODES],
                   width, label=model, **_bar_kwargs(model))

        ax.set_xticks(x)
        ax.set_xticklabels(_ticks(RELIABILITY_TICKS, RELIABILITY_TICKS_PLAIN))
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Share of articles")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Model", framealpha=1.0, edgecolor="black")
        fig.tight_layout()
        _save(fig, name)
    return fig

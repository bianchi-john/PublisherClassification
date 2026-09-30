"""Every figure in the paper, one function each.

Conventions fixed here once, because the old notebooks drifted:

* colour and hatch per model come from ``config``. Notebook 5 used two
  contradictory colour conventions in the same file (Llama green in some
  figures, orange in others), so the same model changed colour between
  figures of one paper.
* every function returns the Figure and writes a PDF whose stem is the
  figure's canonical name. No function writes a second file behind your back.
* nothing calls ``plt.show()``; the notebook decides what to display.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as C
from .metrics import METRIC_LABELS, baselines

METRIC_ORDER = [
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Tolerant_0.5",
    "Tolerant_1.0",
]


def set_style(use_latex: bool = False) -> None:
    """Call once at the top of the notebook.

    ``use_latex=True`` reproduces the original PGF/LaTeX rendering, which needs
    a working TeX installation. It is off by default so the pipeline runs
    anywhere; turn it on for the camera-ready to match the paper's fonts.
    """
    if use_latex:
        mpl.use("pgf")
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "serif",
                "pgf.rcfonts": False,
            }
        )
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": C.FIG_DPI,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    C.ensure_output_dirs()
    fig.savefig(C.FIGURES_DIR / f"{name}.pdf")
    return fig


def _models_in(df: pd.DataFrame) -> list[str]:
    present = [m for m in C.MODEL_ORDER_WITH_GPT if m in set(df["model"].astype(str))]
    return present


def _style(model: str) -> dict:
    return {
        "facecolor": C.MODEL_COLORS.get(model, "#888888"),
        "hatch": C.MODEL_HATCHES.get(model, ""),
        "edgecolor": "black",
    }


# --------------------------------------------------------------------------
# 1. Article-level metrics across folds  (paper: fig:results)
# --------------------------------------------------------------------------
def fig_article_metrics_boxplot(
    per_fold: pd.DataFrame,
    y_true_pool,
    name: str = "article_metrics_boxplot",
    include_gpt: bool = False,
) -> plt.Figure:
    """Per-fold distribution of the six metrics, with baselines.

    This is the figure the RoBERTa/Llama swap corrupted. Here the model name
    travels with the data from ``config.FINETUNED_MODELS`` and is never
    re-assigned, so the legend cannot disagree with the numbers.

    With ``include_gpt=True`` the zero-shot baseline is drawn alongside the
    fine-tuned models. It can be boxed like the others because all four
    fine-tuned models share one article -> fold map, so GPT's predictions can be
    scored on the same five publisher-disjoint partitions. What the box *means*
    differs, though: for the fine-tuned models the spread reflects a different
    training set as well as a different test set, while GPT is never trained.
    The note printed under the axis says so, because a reader would otherwise
    read the five boxes as five comparable things.
    """
    base = baselines(y_true_pool)
    order = C.MODEL_ORDER_WITH_GPT if include_gpt else C.MODEL_ORDER
    models = [m for m in order if m in set(per_fold["model"].astype(str))]
    x = np.arange(len(METRIC_ORDER))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        sub = per_fold[per_fold["model"] == model]
        data = [sub[m].to_numpy() for m in METRIC_ORDER]
        pos = x + i * width - 0.4 + width / 2
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
        )
        for box in bp["boxes"]:
            box.set(**_style(model), alpha=0.9)

    # Baselines: majority for accuracy-like metrics, 1/K for the macro ones.
    for j, metric in enumerate(METRIC_ORDER):
        is_macro = "Macro" in metric
        color = C.BASELINE_RANDOM_COLOR if is_macro else C.BASELINE_MAJORITY_COLOR
        style = "-." if is_macro else "--"
        ax.hlines(base[metric], j - 0.45, j + 0.45, color=color, linestyle=style, lw=1.8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Article-level performance")
    handles = [Patch(label=m, **_style(m)) for m in models]
    handles += [
        Line2D([0], [0], color=C.BASELINE_MAJORITY_COLOR, ls="--", label="Majority-class baseline"),
        Line2D([0], [0], color=C.BASELINE_RANDOM_COLOR, ls="-.", label="Random baseline (1/K)"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3, framealpha=0.95)
    if include_gpt:
        ax.set_xlabel(
            "Boxes span the five publisher-disjoint folds. For the fine-tuned models that "
            "spread reflects\na different training set as well as a different test set; "
            "the zero-shot baseline is never trained.",
            fontsize=8,
            labelpad=8,
        )
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 2. Model comparison including the zero-shot baseline
# --------------------------------------------------------------------------
def fig_metrics_with_gpt(
    metrics: pd.DataFrame, level: str, name: str | None = None
) -> plt.Figure:
    """Grouped bars of the six metrics, fine-tuned models plus GPT-5-mini.

    ``metrics`` is the output of ``aggregate.article_metrics`` or
    ``aggregate.publisher_metrics``: one row per model.
    """
    name = name or f"{level}_metrics_with_gpt"
    models = _models_in(metrics)
    x = np.arange(len(METRIC_ORDER))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        row = metrics[metrics["model"].astype(str) == model].iloc[0]
        vals = [row[m] for m in METRIC_ORDER]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9, label=model, **_style(model))

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{level.capitalize()}-level performance")
    ax.legend(ncol=3, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 3. Publisher-level metrics  (paper: fig:metrics)
# --------------------------------------------------------------------------
def fig_publisher_metrics(
    metrics: pd.DataFrame, y_true_pub, name: str = "publisher_metrics"
) -> plt.Figure:
    base = baselines(y_true_pub)
    models = [m for m in C.MODEL_ORDER if m in set(metrics["model"].astype(str))]
    x = np.arange(len(METRIC_ORDER))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        row = metrics[metrics["model"].astype(str) == model].iloc[0]
        vals = [row[m] for m in METRIC_ORDER]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9, label=model, **_style(model))
    for j, metric in enumerate(METRIC_ORDER):
        is_macro = "Macro" in metric
        ax.hlines(
            base[metric],
            j - 0.45,
            j + 0.45,
            color=C.BASELINE_RANDOM_COLOR if is_macro else C.BASELINE_MAJORITY_COLOR,
            linestyle="-." if is_macro else "--",
            lw=1.8,
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Publisher-level performance")
    handles = [Patch(label=m, **_style(m)) for m in models]
    handles += [
        Line2D([0], [0], color=C.BASELINE_MAJORITY_COLOR, ls="--", label="Majority-class baseline"),
        Line2D([0], [0], color=C.BASELINE_RANDOM_COLOR, ls="-.", label="Random baseline (1/K)"),
    ]
    ax.legend(handles=handles, ncol=3, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 4. Error rate per reliability class  (paper: fig:error_rate)
# --------------------------------------------------------------------------
def fig_error_rate_by_class(err: pd.DataFrame, name: str = "error_rate_by_class") -> plt.Figure:
    models = _models_in(err)
    x = np.arange(C.N_CLASSES)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(models):
        sub = err[err["model"].astype(str) == model].sort_values("y_true")
        ax.bar(x + i * width - 0.4 + width / 2, sub["error_rate"], width * 0.9, label=model, **_style(model))
    ax.set_xticks(x)
    ax.set_xticklabels(C.CLASS_LABELS)
    ax.set_xlabel("True reliability level")
    ax.set_ylabel(r"Error rate  $(1-\mathrm{recall})$")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 5. Mean error direction per reliability class  (paper: fig:error_direction)
# --------------------------------------------------------------------------
def fig_med_by_class(med: pd.DataFrame, name: str = "med_by_class") -> plt.Figure:
    models = _models_in(med)
    x = np.arange(C.N_CLASSES)
    width = 0.8 / len(models)
    denom = med["med_denominator"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(models):
        sub = med[med["model"].astype(str) == model].sort_values("y_true")
        ax.bar(x + i * width - 0.4 + width / 2, sub["med"], width * 0.9, label=model, **_style(model))
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(C.CLASS_LABELS)
    ax.set_xlabel("True reliability level")
    ax.set_ylabel(r"Mean error direction $(\hat{y}-y)$")
    ax.set_title(
        f"averaged over {'misclassified articles only' if denom == 'errors' else 'all articles'}",
        fontsize=10,
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 6. Reliability distribution across leanings  (paper: fig:orientationDistributionInverted)
# --------------------------------------------------------------------------
def fig_leaning_distribution(tab: pd.DataFrame, name: str = "leaning_distribution") -> plt.Figure:
    """Stacked-by-class article counts per political leaning."""
    x = np.arange(len(tab.index))
    width = 0.8 / len(tab.columns)
    cmap = plt.get_cmap("viridis", len(tab.columns))
    hatches = ["//", "\\\\", "xx", "||", "--"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, col in enumerate(tab.columns):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            tab[col].to_numpy(),
            width * 0.9,
            label=col,
            facecolor=cmap(i),
            hatch=hatches[i % len(hatches)],
            edgecolor="black",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i).replace(" ", "\n") for i in tab.index])
    ax.set_ylabel("Number of articles")
    ax.set_xlabel("Political leaning of the publisher")
    ax.legend(title="Reliability", ncol=5)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 7. Error rate by leaning  (paper: fig:politicalErrorRates)
# --------------------------------------------------------------------------
def fig_error_rate_by_leaning(er: pd.DataFrame, name: str = "error_rate_by_leaning") -> plt.Figure:
    models = _models_in(er)
    leanings = [l for l in C.LEANING_ORDER if l in set(er["leaning"].astype(str))]
    x = np.arange(len(leanings))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, model in enumerate(models):
        sub = er[er["model"].astype(str) == model]
        vals = [float(sub[sub["leaning"].astype(str) == l]["error_rate"].iloc[0]) for l in leanings]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9, label=model, **_style(model))
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in leanings])
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 8. Mean error direction by leaning  (paper: fig:politicalErrorsDirection)
# --------------------------------------------------------------------------
def fig_med_by_leaning(med: pd.DataFrame, name: str = "med_by_leaning") -> plt.Figure:
    models = _models_in(med)
    leanings = [l for l in C.LEANING_ORDER if l in set(med["leaning"].astype(str))]
    x = np.arange(len(leanings))
    width = 0.8 / len(models)
    denom = med["med_denominator"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, model in enumerate(models):
        sub = med[med["model"].astype(str) == model]
        vals = [float(sub[sub["leaning"].astype(str) == l]["med"].iloc[0]) for l in leanings]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9, label=model, **_style(model))
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in leanings])
    ax.set_ylabel(r"Mean error direction $(\hat{y}-y)$")
    ax.set_title(
        f"averaged over {'misclassified articles only' if denom == 'errors' else 'all articles'}",
        fontsize=10,
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 9. Aggregation learning curve  (new)
# --------------------------------------------------------------------------
def fig_aggregation_curve(curve: pd.DataFrame, name: str = "aggregation_curve") -> plt.Figure:
    """Publisher accuracy vs number of articles voted.

    k=1 is the article-level baseline measured on the publisher unit, so the
    rise from k=1 is exactly what aggregation contributes -- the comparison the
    paper previously conflated with a change of both unit and weighting.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in _models_in(curve):
        sub = curve[curve["model"].astype(str) == model].sort_values("k")
        ax.errorbar(
            sub["k"],
            sub["accuracy_mean"],
            yerr=sub["accuracy_std"],
            marker="o",
            capsize=3,
            label=model,
            color=C.MODEL_COLORS.get(model, "#888888"),
        )
    n = int(curve["n_publishers"].iloc[0])
    ax.set_xlabel("Number of articles aggregated per publisher ($k$)")
    ax.set_ylabel("Publisher-level accuracy")
    ax.set_title(f"Fixed population of {n} publishers at every $k$", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# 10. Prediction distribution  (new -- shows GPT's collapse onto the centre)
# --------------------------------------------------------------------------
def fig_prediction_distribution(
    preds: pd.DataFrame, name: str = "prediction_distribution"
) -> plt.Figure:
    models = _models_in(preds)
    x = np.arange(C.N_CLASSES)
    width = 0.8 / (len(models) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    truth = preds[preds["model"].astype(str) == models[0]]["y_true"]
    share = [float((truth == c).mean()) for c in C.CLASS_CODES]
    ax.bar(x - 0.4 + width / 2, share, width * 0.9, label="Ground truth",
           facecolor="white", edgecolor="black", hatch="..")

    for i, model in enumerate(models, start=1):
        sub = preds[preds["model"].astype(str) == model]
        share = [float((sub["y_pred"] == c).mean()) for c in C.CLASS_CODES]
        ax.bar(x + i * width - 0.4 + width / 2, share, width * 0.9, label=model, **_style(model))

    ax.set_xticks(x)
    ax.set_xticklabels(C.CLASS_LABELS)
    ax.set_xlabel("Predicted reliability level")
    ax.set_ylabel("Share of articles")
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, name)

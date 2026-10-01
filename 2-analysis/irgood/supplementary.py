"""Figures that live in the supplementary material, not in the paper body.

Same contract as ``plots``: the output is meant to be indistinguishable from the
figures in the old supplementary PDF, so each function transcribes the rcParams
and geometry that were actually in force when the published version was drawn.

Note that three of these five families were drawn *without* LaTeX in the old
code (`text.usetex=False`), so they are rendered outside the PGF backend here
too. The published supplementary is, like the paper, not typeset uniformly.

Families
--------
cm_{model}.pdf                       per-fold + average article-level matrices
domain_confusion_matrix_{model}.pdf  publisher-level matrices
cmOnPoliticalLeaning{Model}.pdf      article-level matrices split by leaning
loss_plot.pdf                        Llama 3 convergence over 20 epochs
training_loss_mistral.pdf            Mistral training loss, 5 folds
"""
from __future__ import annotations

import ast
import math
import re

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from . import config as C
from . import plots as _plots
from .plots import _figure_style, _plain_pdf_backend


def _usetex() -> bool:
    """Read the flag live.

    Importing ``_USETEX`` by value would snapshot it at import time, i.e. before
    ``plots.set_style()`` runs, so every check here would see the initial False.
    That silently drew the heaviest figure through the PGF backend (blowing
    TeX's main memory) and picked the non-italic class labels.
    """
    return _plots._USETEX

# The old filenames are not uniformly cased -- note the double L in "LLama".
# They are kept verbatim so the supplementary .tex needs no edit.
SUPP_MODEL_SLUG = {
    "BERT": "Bert",
    "RoBERTa": "Roberta",
    "Llama": "LLama",
    "Mistral": "Mistral",
}


def _supp_dir():
    d = C.FIGURES_DIR / "supplementary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: plt.Figure, name: str, tight: bool = True) -> None:
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(_supp_dir() / f"{name}.pdf", **kwargs)
    plt.close(fig)


def _parse_cm(cell) -> np.ndarray:
    """Confusion matrices are stored as a stringified nested list."""
    if isinstance(cell, str):
        try:
            return np.asarray(ast.literal_eval(cell), dtype=int)
        except (ValueError, SyntaxError):
            vals = [int(x) for x in re.findall(r"\d+", cell)]
            n = int(round(math.sqrt(len(vals))))
            return np.asarray(vals).reshape(n, n)
    return np.asarray(cell, dtype=int)


# --------------------------------------------------------------------------
# 1. Article-level confusion matrices: five folds plus the average
# --------------------------------------------------------------------------
def _draw_article_cm(ax, cm, title, cmap, normalized):
    label_map = (
        [r"\textit{l1}", r"\textit{l2}", r"\textit{l3}", r"\textit{l4}", r"\textit{l5}"]
        if _usetex()
        else ["l1", "l2", "l3", "l4", "l5"]
    )
    vmax = 1.0 if normalized else cm.max()
    ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=18, pad=20)

    thresh = vmax / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:.2f}" if normalized else str(int(val)),
                ha="center", va="center",
                color="white" if val > thresh else "black", fontsize=14,
            )

    ax.set_xlabel("Predicted Label", labelpad=15, fontsize=16)
    ax.set_ylabel("True Label", labelpad=15, fontsize=16)
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(label_map, fontsize=12, rotation=0, ha="right")
    ax.set_yticklabels(label_map, fontsize=12)
    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def fig_article_confusion_grid(model: str) -> None:
    """`cm_{model}.pdf` -- one panel per fold plus a normalised average."""
    df = pd.read_csv(C.FINETUNED_MODELS[model]["enriched_csv"])
    mats = [_parse_cm(c) for c in df["Confusion_Matrix"]]
    avg = np.mean(mats, axis=0)
    avg = avg / np.maximum(1, avg.sum(axis=1, keepdims=True))

    colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
              "#4292c6", "#2171b5", "#08519c", "#08306b"]
    cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

    total = len(mats) + 1
    cols = int(np.ceil(np.sqrt(total)))
    rows = int(np.ceil(total / cols))

    # The published set is not uniform: cm_bert/roberta/llama are 15x12 inches,
    # cm_mistral is 15x10 -- it was evidently not regenerated when the others
    # were resized. Reproduced verbatim rather than silently harmonised.
    figsize = (15, 10) if model == "Mistral" else (15, 12)

    with _figure_style(**{
        "axes.labelsize": 16, "font.size": 14, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    }):
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        axes = axes.ravel()
        for i, cm in enumerate(mats):
            _draw_article_cm(axes[i], cm, f"Fold {i + 1}", cmap, False)
        _draw_article_cm(axes[len(mats)], avg, "Average (Normalized)", cmap, True)
        for ax in axes[total:]:
            ax.axis("off")
        fig.tight_layout()
        _save(fig, f"cm_{model.lower()}", tight=False)


# --------------------------------------------------------------------------
# 2. Publisher-level confusion matrices
# --------------------------------------------------------------------------
def fig_publisher_confusion(model: str, pub: pd.DataFrame) -> None:
    """`domain_confusion_matrix_{model}.pdf`.

    Drawn outside PGF: the old notebook set ``text.usetex=False`` here, so the
    class labels are mathtext, not LaTeX.
    """
    from sklearn.metrics import confusion_matrix

    g = pub[pub["model"].astype(str) == model]
    cm = confusion_matrix(g["y_true"], g["y_pred"], labels=C.CLASS_CODES)
    labels = [r"$l_1$", r"$l_2$", r"$l_3$", r"$l_4$", r"$l_5$"]

    stack = _plain_pdf_backend() if _usetex() else _nullctx()
    with stack, _figure_style(**{
        "axes.labelsize": 11, "font.size": 11, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "mathtext.fontset": "cm",
    }):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0, vmax=cm.max())
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=24, fontfamily="serif")
        ax.set_xlabel("Predicted Label", fontsize=26, labelpad=14)
        ax.set_ylabel("True Label", fontsize=26, labelpad=14)
        ax.set_xticks(np.arange(5)); ax.set_yticks(np.arange(5))
        ax.set_xticklabels(labels, fontsize=22)
        ax.set_yticklabels(labels, fontsize=22)
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_xticks(np.arange(6) - 0.5, minor=True)
        ax.set_yticks(np.arange(6) - 0.5, minor=True)
        ax.grid(which="minor", color="dimgray", linestyle="-", linewidth=1.0, alpha=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_color("dimgray"); spine.set_linewidth(1.0)
        fig.tight_layout()
        _save(fig, f"domain_confusion_matrix_{model.lower()}")


# --------------------------------------------------------------------------
# 3. Confusion matrices split by political leaning
# --------------------------------------------------------------------------
def fig_leaning_confusion(model: str, preds: pd.DataFrame) -> None:
    """`cmOnPoliticalLeaning{Model}.pdf`.

    One 5x5 panel per leaning. Each cell carries four numbers: share of the true
    class, raw count, share of that leaning's articles, and how many distinct
    publishers contribute -- the last is what shows that the extreme-leaning
    errors come from a handful of outlets.
    """
    df = preds[preds["model"].astype(str) == model].copy()
    df["leaning"] = df["leaning"].astype(str)
    all_labels = C.CLASS_CODES
    orientations = sorted(df["leaning"].unique())
    n_classes = len(all_labels)
    label_map = {i: rf"$\mathit{{l{i + 1}}}$" for i in all_labels}

    # Shared colour scale across panels, so they are comparable.
    max_pct = 0.0
    for o in orientations:
        d = df[df["leaning"] == o]
        cm = pd.crosstab(d["y_true"], d["y_pred"]).reindex(
            index=all_labels, columns=all_labels, fill_value=0)
        rs = cm.sum(axis=1)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and rs.iloc[i] > 0:
                    max_pct = max(max_pct, cm.iloc[i, j] / rs.iloc[i] * 100)
    vmax = max(max_pct, 10)

    n_cols = 3
    n_rows = math.ceil(len(orientations) / n_cols)

    # No backend switch: the published figures are Computer Modern, i.e. they
    # were typeset by LaTeX. The old code set text.usetex=False here, but that
    # is inert under the PGF backend, which always goes through LaTeX.
    with _figure_style(**{
        "mathtext.fontset": "cm", "axes.labelsize": 14, "font.size": 14,
        "legend.fontsize": 12, "xtick.labelsize": 14, "ytick.labelsize": 14,
    }):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 9 * n_rows))
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, orientation in enumerate(orientations):
            ax = axes_flat[idx]
            d = df[df["leaning"] == orientation]
            total_samples = len(d)
            cm = pd.crosstab(d["y_true"], d["y_pred"]).reindex(
                index=all_labels, columns=all_labels, fill_value=0)
            row_sums = cm.sum(axis=1)

            color_m = pd.DataFrame(index=all_labels, columns=all_labels, dtype=float)
            annot_m = pd.DataFrame(index=all_labels, columns=all_labels, dtype=object)
            empty_rows = []

            for i, true_label in enumerate(all_labels):
                total_class = row_sums[true_label]
                row_subset = d[d["y_true"] == true_label]
                dom_in_row = row_subset["domain"].nunique()
                if total_class == 0:
                    # A reliability level with no articles for this leaning
                    # (e.g. l5 for Far Right). The published figure prints a
                    # plain "0" across such a row rather than leaving it blank.
                    empty_rows.append(i)
                    for j in range(len(all_labels)):
                        color_m.iloc[i, j] = np.nan
                        annot_m.iloc[i, j] = "0"
                    continue
                for j, pred_label in enumerate(all_labels):
                    count = cm.iloc[i, j]
                    pct = count / total_class * 100
                    n_dom = d[(d["y_true"] == true_label) &
                              (d["y_pred"] == pred_label)]["domain"].nunique()
                    text = (f"{pct:.1f}%\n{count}/{total_class}\n"
                            f"{count / total_samples:.1%}\nDom: {n_dom}/{dom_in_row}")
                    if i == j:
                        color_m.iloc[i, j] = np.nan
                        annot_m.iloc[i, j] = text
                    else:
                        color_m.iloc[i, j] = pct if count > 0 else np.nan
                        annot_m.iloc[i, j] = text if count > 0 else f"0\nDom: 0/{dom_in_row}"

            mask = np.isnan(color_m.values.astype(float))
            sns.heatmap(color_m.astype(float), cmap="plasma", mask=mask, vmin=0,
                        vmax=vmax, linewidths=1, linecolor="gray", square=True,
                        ax=ax, cbar=False)
            for i in range(n_classes):
                for j in range(n_classes):
                    bg = color_m.iloc[i, j]
                    color = "white" if (not pd.isna(bg) and bg < vmax * 0.5) else "black"
                    if i == j:
                        color = "black"
                    # Bold only on the diagonal and on non-empty off-diagonal
                    # cells; empty cells stay regular weight. That conditional
                    # is why the published figure embeds CMR10 next to CMBX10.
                    weight = "bold" if (i == j or not pd.isna(bg)) else "normal"
                    ax.text(j + 0.5, i + 0.5, annot_m.iloc[i, j], ha="center",
                            va="center", color=color, fontsize=11, fontweight=weight)

            ax.set_title(f"Orientation: {orientation}\n(Diag: Accuracy \u2014 Off-Diag: % Error)",
                         fontsize=20, fontweight="bold", pad=15)
            ax.set_xlabel("Predicted Class", fontsize=16, fontweight="bold")
            ax.set_ylabel("True Class", fontsize=16, fontweight="bold")
            ax.set_xticks(np.arange(n_classes) + 0.5)
            ax.set_yticks(np.arange(n_classes) + 0.5)
            ax.set_xticklabels([label_map[i] for i in all_labels], fontsize=14)
            ax.set_yticklabels([label_map[i] for i in all_labels], fontsize=14)

            total_errors = int((d["y_true"] != d["y_pred"]).sum())
            acc = (1 - total_errors / total_samples) * 100 if total_samples else 0.0
            ax.text(0.5, -0.16,
                    f"Total Samples: {total_samples} \u2014 Global Acc: {acc:.1f}% "
                    f"\u2014 Total Errors: {total_errors}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=14, fontweight="bold")

        for i in range(len(orientations), len(axes_flat)):
            axes_flat[i].axis("off")

        fig.tight_layout()
        fig.subplots_adjust(top=0.94, bottom=0.1, hspace=0.4, wspace=0.2, right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap="plasma")
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax).set_label(
            "Global % Error Rate (on True Class)", fontsize=16, fontweight="bold")
        _save(fig, f"cmOnPoliticalLeaning{SUPP_MODEL_SLUG[model]}")


# --------------------------------------------------------------------------
# 4. Llama 3 convergence over 20 epochs
# --------------------------------------------------------------------------
def fig_llama_convergence() -> None:
    """`loss_plot.pdf` -- the run that justified stopping at 5 epochs."""
    text = C.FINETUNING_LOG.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"RISULTATI EPOCA (\d+): Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)", text
    )
    if not matches:
        raise ValueError(f"no epoch results parsed from {C.FINETUNING_LOG}")
    epochs = [int(m[0]) for m in matches]
    train = [float(m[1]) for m in matches]
    val = [float(m[2]) for m in matches]
    best = int(np.argmin(val))

    with _figure_style(**{
        "axes.labelsize": 16, "font.size": 14, "legend.fontsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    }):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train, "b-o", label="Training Loss", linewidth=2, markersize=6)
        ax.plot(epochs, val, "r-s", label="Validation Loss", linewidth=2, markersize=6)
        ax.axvline(x=epochs[best], color="g", linestyle="--", alpha=0.7)
        ax.plot(epochs[best], val[best], "g*", markersize=18,
                label=f"Best Model (Epoch {epochs[best]})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        ax.text(epochs[best], val[best] + max(val) * 0.02,
                f"  Epoch {epochs[best]}\n  Loss: {val[best]:.4f}",
                fontsize=10, verticalalignment="bottom", color="green")
        fig.tight_layout()
        _save(fig, "loss_plot")


# --------------------------------------------------------------------------
# 5. Mistral training loss across the five folds
# --------------------------------------------------------------------------
def fig_mistral_training_loss(window: int = 20) -> None:
    """`training_loss_mistral.pdf` -- one panel per fold, epochs shaded."""
    pattern = re.compile(
        r"STATUS\s+F(\d+)\s+E(\d+)\s+\|.*?(\d+\.\d+)%.*?\|\s+Loss\s+([\d.]+)"
    )
    records = []
    with open(C.MISTRAL_LOSS_LOG, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                records.append({"fold": int(m.group(1)), "epoch": int(m.group(2)),
                                "pct": float(m.group(3)), "loss": float(m.group(4))})
    if not records:
        raise ValueError(f"no STATUS lines parsed from {C.MISTRAL_LOSS_LOG}")

    df = pd.DataFrame(records)
    df["global_pct"] = (df["epoch"] - 1) * 100 + df["pct"]
    df = df.sort_values(["fold", "global_pct"])
    df["loss_smooth"] = df.groupby("fold")["loss"].transform(
        lambda s: s.rolling(window, min_periods=1, center=True).mean()
    )

    ep_colors = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868"}
    ep_hatch = {1: "///", 2: "...", 3: "xxx"}
    folds = sorted(df["fold"].unique())
    epochs = sorted(df["epoch"].unique())

    stack = _plain_pdf_backend() if _usetex() else _nullctx()
    with stack, _figure_style(**{
        "font.family": "DejaVu Serif", "axes.labelsize": 10, "font.size": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    }):
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, axes = plt.subplots(len(folds), 1, figsize=(14, 4.0 * len(folds)))
        axes = np.atleast_1d(axes)

        for ax, fold in zip(axes, folds):
            sub = df[df["fold"] == fold]
            y_max = sub["loss"].quantile(0.97)
            for ep in epochs:
                es = sub[sub["epoch"] == ep]
                if es.empty:
                    continue
                ax.axvspan(es["global_pct"].min(), es["global_pct"].max(), alpha=0.07,
                           color=ep_colors[ep], hatch=ep_hatch[ep],
                           edgecolor=ep_colors[ep], linewidth=0.0)
            ax.plot(sub["global_pct"], sub["loss"], color="#555555", alpha=0.55,
                    linewidth=0.9, label="Raw loss" if fold == folds[0] else "")
            ax.plot(sub["global_pct"], sub["loss_smooth"],
                    color=cycle[(fold - 1) % len(cycle)], linewidth=1.8,
                    label=f"Smoothed (w={window})" if fold == folds[0] else "")
            for boundary in (100, 200):
                if boundary < sub["global_pct"].max():
                    ax.axvline(boundary, color="black", linewidth=0.8,
                               linestyle="--", alpha=0.5)
            ax.set_ylim(bottom=0, top=y_max * 1.22)
            for ep in epochs:
                es = sub[sub["epoch"] == ep]
                if es.empty:
                    continue
                ax.text(es["global_pct"].mean(), y_max * 1.05,
                        f"mu={es['loss'].mean():.3f}", ha="center", va="bottom",
                        fontsize=10, color=ep_colors[ep], fontweight="bold")
            ax.set_title(f"Fold {fold}", fontweight="bold", fontsize=13)
            ax.set_ylabel("Loss")
            ax.set_xlabel("Global progress (% across epochs)")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"E{int(x // 100) + 1} {int(x % 100)}%")
            )
            ax.grid(True, linestyle=":", alpha=0.4)

        line_handles, _ = axes[0].get_legend_handles_labels()
        epoch_handles = [
            mpatches.Patch(facecolor=ep_colors[ep], hatch=ep_hatch[ep],
                           edgecolor=ep_colors[ep], alpha=0.35, label=f"Epoch {ep}")
            for ep in epochs
        ]
        handles = epoch_handles + line_handles
        fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                   loc="upper center", ncol=6, fontsize=11,
                   bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
        fig.tight_layout()
        _save(fig, "training_loss_mistral")


# --------------------------------------------------------------------------
def _nullctx():
    from contextlib import nullcontext
    return nullcontext()


def generate_all(preds: pd.DataFrame, pub: pd.DataFrame) -> list[str]:
    """Every supplementary figure. Returns the filenames written."""
    for model in C.MODEL_ORDER:
        fig_article_confusion_grid(model)
        fig_publisher_confusion(model, pub)
        fig_leaning_confusion(model, preds)
    fig_llama_convergence()
    fig_mistral_training_loss()
    return sorted(p.name for p in _supp_dir().glob("*.pdf"))

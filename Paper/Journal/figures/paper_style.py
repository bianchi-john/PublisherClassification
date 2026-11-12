"""The style of the paper's figures: that of the first (rejected) version of the paper (Code/IrForGood/2-analysis/irgood
/plots.py), made readable without colour.

From the first version: LaTeX text (Computer Modern), bars with black edges and one hatch per series, boxed legends
with a title, light grey grid behind the bars, baselines as dashed lines (red: no-text / majority, violet: second
baseline).

Readable without colour: every series has a second channel that is not colour. Bars and patches have a hatch per
series, lines have their own marker and dash pattern, points their own marker shape, and the categories are named on
the axes or in a legend. The colours only add to these channels:
- models: the four categorical slots of the validated palette (blue, orange, aqua, yellow), in a fixed order; adjacent
  pairs are separable for colour-blind readers (CVD Delta E >= 9.1), and each model also has its hatch and marker;
- NewsGuard levels: a diverging red - grey - blue scale (unreliable - middle - reliable), with the hatches of the
  first version;
- the other categorical figures use the same first slots, with hatches or marker shapes.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ models
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]            # keys of the pipeline outputs
MODEL_NAMES = {"TF-IDF": "TF-IDF", "Qwen3-8B": "Qwen3-Embedding-8B", "Mistral-24B": "Mistral-Small-24B",
               "ModernBERT": "ModernBERT"}
MODEL_COLORS = {"TF-IDF": "#2a78d6", "Qwen3-8B": "#eb6834", "Mistral-24B": "#1baf7a", "ModernBERT": "#eda100"}
MODEL_HATCHES = {"TF-IDF": "//", "Qwen3-8B": "\\\\", "Mistral-24B": "xx", "ModernBERT": "||"}
MODEL_MARKERS = {"TF-IDF": "o", "Qwen3-8B": "s", "Mistral-24B": "^", "ModernBERT": "D"}
MODEL_DASHES = {"TF-IDF": "-", "Qwen3-8B": "--", "Mistral-24B": "-.", "ModernBERT": ":"}

# ------------------------------------------------------------------------------------------------ NewsGuard levels
LEVELS = ["l1", "l2", "l3", "l4", "l5"]
LEVEL_BANDS = ["0--39", "40--59", "60--74", "75--99", "100"]
LEVEL_COLORS = ["#a9302f", "#ec9a99", "#e4e2dc", "#86b6ef", "#1c5cab"]   # diverging: red - grey - blue
LEVEL_HATCHES = ["///", "\\\\\\", "xxx", "---", "+++"]                    # the first version's class hatches
LEVEL_MARKERS = ["v", "<", "o", ">", "^"]

# ------------------------------------------------------------------------------------------------ other categories
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]          # validated categorical order
HATCHES = ["//", "\\\\", "xx", "..", "oo"]
BASELINE_RED, BASELINE_VIOLET = "#B22222", "#6A3D9A"                   # the first version's baseline colours
INK, MUTED, GRID = "black", "#55544f", "#c8c8c8"

_USETEX = False


def set_style(use_latex: bool = True) -> None:
    """LaTeX text (Computer Modern) as in the first version; mathtext with Computer Modern if pdflatex is missing."""
    global _USETEX
    _USETEX = use_latex and shutil.which("pdflatex") is not None
    plt.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        "text.usetex": _USETEX, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if _USETEX else ["DejaVu Serif"], "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8, "legend.title_fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": INK, "legend.fancybox": False,
        "hatch.linewidth": 0.6, "hatch.color": INK, "patch.edgecolor": INK,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def tex(s: str) -> str:
    """Text safe for both renderers: en dashes and percent signs written the LaTeX way only under LaTeX."""
    if _USETEX:
        return s.replace("%", r"\%")
    return s.replace("--", "–")


def save(fig: plt.Figure, name: str) -> Path:
    """The paper's figure `name`.pdf next to this module (the file main.tex includes)."""
    path = HERE / f"{name}.pdf"
    fig.savefig(path)
    return path


# ------------------------------------------------------------------------------------------------ helpers
def model_bar(model: str) -> dict:
    return {"color": MODEL_COLORS[model], "hatch": MODEL_HATCHES[model], "edgecolor": INK, "linewidth": 0.7}


def model_handles(models=MODELS) -> list[Patch]:
    return [Patch(facecolor=MODEL_COLORS[m], edgecolor=INK, hatch=MODEL_HATCHES[m], label=MODEL_NAMES[m])
            for m in models]


def model_line(model: str) -> dict:
    return {"color": MODEL_COLORS[model], "marker": MODEL_MARKERS[model], "linestyle": MODEL_DASHES[model],
            "markeredgecolor": INK, "markeredgewidth": 0.6, "markersize": 5, "linewidth": 1.4}


def level_handles() -> list[Patch]:
    return [Patch(facecolor=c, edgecolor=INK, hatch=h, label=tex(rf"$l_{i + 1}$ ({b})"))
            for i, (c, h, b) in enumerate(zip(LEVEL_COLORS, LEVEL_HATCHES, LEVEL_BANDS))]


def level_ticks() -> list[str]:
    return [tex(rf"$l_{i + 1}$ ({b})") for i, b in enumerate(LEVEL_BANDS)]


def grouped_bars(ax, groups, series, values, styles, width=0.8, errors=None, zero_line=False):
    """Bars of `series` (one colour+hatch each) inside every group; values[s][g]; errors[s][g] = (low, high)."""
    n = len(series)
    w = width / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        pos = x - width / 2 + (i + 0.5) * w
        v = np.array([values[s][g] for g in groups], dtype=float)
        ax.bar(pos, v, w, zorder=3, **styles[s])
        if errors is not None:
            lo = np.array([errors[s][g][0] for g in groups]); hi = np.array([errors[s][g][1] for g in groups])
            ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
    if zero_line:
        ax.axhline(0, color=INK, linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    return x


def baseline(ax, x0, x1, y, kind="red", label=None):
    """A baseline drawn across one group of bars (red dashed or violet dash-dot, as in the first version)."""
    style = {"red": dict(color=BASELINE_RED, linestyle=(0, (5, 3))),
             "violet": dict(color=BASELINE_VIOLET, linestyle=(0, (6, 2, 1, 2)))}[kind]
    ax.plot([x0, x1], [y, y], linewidth=1.6, zorder=5, label=label, **style)


def baseline_handle(kind, label):
    style = {"red": dict(color=BASELINE_RED, linestyle=(0, (5, 3))),
             "violet": dict(color=BASELINE_VIOLET, linestyle=(0, (6, 2, 1, 2)))}[kind]
    return Line2D([], [], linewidth=1.6, label=label, **style)


def packed_layout(edges, seed=20261112):
    """Positions of the outlets that share articles: every connected component laid out on its own (spring layout,
    radius ~ sqrt of its size), the largest on the left, the others packed in rows to its right."""
    import networkx as nx
    G = nx.Graph(); G.add_edges_from(zip(edges.a, edges.b))
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    radius = lambda c: 0.55 * np.sqrt(len(c)) + 0.35
    pos, r0 = {}, radius(comps[0])
    x0, y0, row_h, width = 2 * r0 + 0.8, r0, 0.0, 2.6 * r0
    for i, c in enumerate(comps):
        r = radius(c)
        p = nx.spring_layout(G.subgraph(c), seed=seed, iterations=200) if len(c) > 2 else \
            {v: np.array([(-1) ** j * 0.6, 0.0]) for j, v in enumerate(sorted(c))}
        xy = np.array(list(p.values())); xy = xy - xy.mean(0); xy = xy / max(np.abs(xy).max(), 1e-9)
        if i == 0:
            centre = np.array([r0, 0.0])
        else:
            if x0 + 2 * r > 2 * r0 + 0.8 + width:
                x0, y0 = 2 * r0 + 0.8, y0 - row_h - 0.5; row_h = 0.0
            centre = np.array([x0 + r, y0 - r]); x0 += 2 * r + 0.4; row_h = max(row_h, 2 * r)
        for v, q in zip(p, xy):
            pos[v] = centre + q * (r - 0.3)
    return pos


def greyscale_check(path: Path, out: Path) -> Path:
    """A greyscale PNG of a figure, to check that it reads without colour."""
    import subprocess
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pnggray", "-r110", f"-sOutputFile={out}", str(path)],
                   check=True)
    return out

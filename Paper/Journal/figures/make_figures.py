"""Figures of the journal draft (Paper/Journal). Run from this folder:  python make_figures.py

- dataset.pdf: the 226 outlets: NewsGuard score (with the five NewsGuard levels), articles per outlet, score per fold.
  Final: the dataset and the split do not depend on the models.
- curves.pdf: Spearman and MAE per outlet as more articles per outlet are read (k = 1, 2, 3, 5, 10, 20, 50, all;
  20 random draws per k, the same for every model), for the regressors of step 7 and the no-text baseline. Drawn only
  when the step-7 predictions of the current run are in Dataset/tentativi/20_FromScratch/predictions/regression/.
Only aggregate statistics are plotted (NewsGuard labels are not redistributable)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "Dataset" / "tentativi" / "20_FromScratch"
sys.path.insert(0, str(REPO / "Code" / "Pipeline"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
BOUNDS = [39.5, 59.5, 74.5, 99.5]
LEVEL_COLOURS = ["#b02a2a", "#e58a8a", "#bdbdbd", "#7fb0e0", "#1d4f91"]
MODELS = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834"}


def dataset_figure():
    o = pd.read_csv(DATA / "outlets.csv", index_col="domain")
    o["fold"] = pd.read_csv(DATA / "split.csv", index_col="domain").fold
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    edges = [0, *BOUNDS, 100.5]
    for k in range(5):
        a.axvspan(edges[k], edges[k + 1], color=LEVEL_COLOURS[k], alpha=0.15, lw=0)
    a.hist(o.score, bins=np.arange(0, 102.5, 5), color="#444444", rwidth=0.9)
    a.set_xlabel("NewsGuard score"); a.set_ylabel("outlets"); a.set_title("(a) score of the 226 outlets", fontsize=9)
    b.hist(o.articles, bins=np.arange(0, 211, 15), color="#444444", rwidth=0.9)
    b.set_xlabel("articles per outlet"); b.set_title("(b) articles per outlet", fontsize=9)
    c.boxplot([o.score[o.fold == f] for f in range(1, 6)], widths=0.5, medianprops={"color": "#b02a2a"})
    c.set_xlabel("fold"); c.set_title("(c) score per fold", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "dataset.pdf", bbox_inches="tight"); plt.close()
    print("dataset.pdf")


def curves_figure():
    folder = DATA / "predictions" / "regression"
    done = [m for m in MODELS if len(list((folder / m).glob("fold*.csv.gz"))) == 5]
    if not done or not (DATA / "articles_anonymous.csv.gz").exists():
        print("curves.pdf: predictions of the current run not here yet, skipped"); return
    from scipy.stats import spearmanr

    import boost as BO
    art, outlets, rows = BO.load()
    y = outlets.score.to_numpy()
    K = [1, 2, 3, 5, 10, 20, 50, "all"]; X = [1, 2, 3, 5, 10, 20, 50, 160]
    draws = [d for f in range(1, 6) for d in BO.test_draws(outlets, rows, f)]
    base = np.array([outlets.score[outlets.fold != f].mean() for f in outlets.fold])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in done:
        p = pd.concat([pd.read_csv(f) for f in sorted((folder / m).glob("fold*.csv.gz"))]).set_index("article_id").prediction
        p = p.reindex(art.article_id).to_numpy()
        rho, mae = [], []
        for k in K:
            t = pd.DataFrame([(d, j, p[r].mean()) for d, kk, j, r in draws if kk == str(k)], columns=["domain", "draw", "p"])
            M = t.pivot_table(index="draw", columns="domain", values="p").reindex(columns=outlets.index).to_numpy()
            rho.append(np.mean([spearmanr(y, row).correlation for row in M])); mae.append(np.mean(np.abs(M - y).mean(1)))
        a.plot(X, rho, "o-", ms=3, lw=1.4, color=MODELS[m], label=m); b.plot(X, mae, "o-", ms=3, lw=1.4, color=MODELS[m])
    b.axhline(np.abs(base - y).mean(), color="#444444", ls=":", lw=1); b.text(1, np.abs(base - y).mean(), " no text", va="bottom", fontsize=8)
    for ax, t in [(a, "Spearman (outlets)"), (b, "MAE (points)")]:
        ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
        ax.set_xlabel("articles read per outlet"); ax.set_title(t, fontsize=9)
    a.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(HERE / "curves.pdf", bbox_inches="tight"); plt.close()
    print("curves.pdf")


if __name__ == "__main__":
    dataset_figure()
    curves_figure()

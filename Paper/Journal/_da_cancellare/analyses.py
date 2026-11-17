"""Further analyses of the saved out-of-fold predictions (no new model is trained). Run from this folder:
    python analyses.py
Reads Dataset/tentativi/20_FromScratch/{outlets.csv, split.csv, predictions/regression/<model>/fold*.csv.gz} and
writes Dataset/tentativi/20_FromScratch/paper_figures/analyses.json (the numbers quoted in the paper; aggregates of
the NewsGuard scores, so outside git). The figures are drawn by paper_figures.ipynb.
Analyses: article level vs outlet level; errors per level; errors per orientation (Kruskal-Wallis, also on the
residual errors after removing the linear effect of the true score, i.e. at equal NewsGuard score; left-right gap with
a bootstrap interval); ranking within each orientation; orientation alone as a predictor. Intervals: 95% bootstrap over outlets (1,000 resamples,
seeded)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "Dataset" / "tentativi" / "20_FromScratch"
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
LEVELS = ["l1", "l2", "l3", "l4", "l5"]
B, SEED = 1000, 20261112


def load():
    o = pd.read_csv(DATA / "outlets.csv", index_col="domain")
    art = {}
    for m in MODELS:
        p = pd.concat(pd.read_csv(f) for f in sorted((DATA / "predictions" / "regression" / m).glob("fold*.csv.gz")))
        art[m] = p.assign(score=p.domain.map(o.score))
    return o, art


def boot(f, n, rng):
    """95% interval of f(index) over bootstrap resamples of n outlets."""
    v = np.array([f(rng.integers(0, n, n)) for _ in range(B)])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    o, art = load()
    rng = np.random.default_rng(SEED)
    out = {"article_level": {}, "outlet_level": {}, "levels": {}, "orientation": {}, "within_orientation": {}}
    pred = {}
    for m, a in art.items():
        pred[m] = a.groupby("domain").prediction.mean().reindex(o.index)
        out["article_level"][m] = {"spearman": float(spearmanr(a.prediction, a.score).correlation),
                                   "mae": float((a.prediction - a.score).abs().mean()),
                                   "within_20": float(((a.prediction - a.score).abs() <= 20).mean() * 100)}
        out["outlet_level"][m] = {"spearman": float(spearmanr(pred[m], o.score).correlation),
                                  "mae": float((pred[m] - o.score).abs().mean()),
                                  "range": [float(pred[m].min()), float(pred[m].max())]}
        err = pred[m] - o.score
        out["levels"][m] = {lv: {"n": int((o.level == lv).sum()), "signed": float(err[o.level == lv].mean()),
                                 "mae": float(err[o.level == lv].abs().mean())} for lv in LEVELS}
        # orientation: signed errors, raw and at equal score (residuals of error ~ score, which removes the
        # compression of the predictions towards the centre)
        X = np.c_[np.ones(len(o)), o.score]
        beta = np.linalg.lstsq(X, err.to_numpy(), rcond=None)[0]
        adj = err - X @ beta
        L, R = adj[o.orientation == "Left"].to_numpy(), adj[o.orientation == "Right"].to_numpy()
        gaps = [rng.choice(L, len(L)).mean() - rng.choice(R, len(R)).mean() for _ in range(B)]
        out["orientation"][m] = {
            "signed": {k: float(v) for k, v in err.groupby(o.orientation).mean().items()},
            "mae": {k: float(v) for k, v in err.abs().groupby(o.orientation).mean().items()},
            "error_slope_on_score": float(beta[1]),
            "signed_at_equal_score": {k: float(v) for k, v in adj.groupby(o.orientation).mean().items()},
            "left_minus_right_at_equal_score": float(L.mean() - R.mean()),
            "left_minus_right_ci": [float(x) for x in np.percentile(gaps, [2.5, 97.5])],
            "kruskal_p": float(kruskal(*[g for _, g in err.groupby(o.orientation)]).pvalue),
            "kruskal_p_at_equal_score": float(kruskal(*[g for _, g in adj.groupby(o.orientation)]).pvalue)}
        w = {}
        for side in ["Left", "Right", "none"]:
            s = o.orientation == side
            y, p = o.score[s].to_numpy(), pred[m][s].to_numpy()
            w[side] = {"n": int(s.sum()), "spearman": float(spearmanr(p, y).correlation),
                       "ci": boot(lambda i: spearmanr(p[i], y[i]).correlation, len(y), rng)}
        out["within_orientation"][m] = w
    # orientation alone as a predictor: the mean training score of the outlet's orientation (outlet-disjoint)
    split = pd.read_csv(DATA / "split.csv", index_col="domain").fold.reindex(o.index)
    base = pd.Series(index=o.index, dtype=float)
    for f in range(1, 6):
        tr, te = split != f, split == f
        means = o.score[tr].groupby(o.orientation[tr]).mean()
        base[te] = o.orientation[te].map(means)
    out["orientation_only"] = {"spearman": float(spearmanr(base, o.score).correlation),
                               "mae": float((base - o.score).abs().mean())}
    out["counts"] = {"outlets": len(o), "orientation": o.orientation.value_counts().to_dict(),
                     "levels_by_orientation": pd.crosstab(o.orientation, o.level).to_dict()}
    (DATA / "paper_figures" / "analyses.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ["article_level", "outlet_level", "orientation_only"]}, indent=1))
    for m in MODELS:
        print(m, {lv: round(v["signed"], 1) for lv, v in out["levels"][m].items()},
              "| equal score", {k: round(v, 1) for k, v in out["orientation"][m]["signed_at_equal_score"].items()},
              "L-R %.1f" % out["orientation"][m]["left_minus_right_at_equal_score"],
              "p %.1g" % out["orientation"][m]["kruskal_p_at_equal_score"],
              "| within", {k: round(v["spearman"], 2) for k, v in out["within_orientation"][m].items()})
    return out


if __name__ == "__main__":
    main()

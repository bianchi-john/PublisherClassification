"""Are article errors clustered by publisher beyond chance?

For each model: the intra-class correlation (ICC, one-way ANOVA estimator) of
the 0/1 error indicator within publishers, and a permutation test that
shuffles errors across articles OF THE SAME RELIABILITY CLASS (so class
difficulty is preserved and only the publisher grouping is broken).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irgood import config as C

def icc(err, groups):
    d = pd.DataFrame({"e": err, "g": groups})
    k = d.groupby("g").size()
    n, a = len(d), len(k)
    gm = d.e.mean()
    msb = (k * (d.groupby("g").e.mean() - gm) ** 2).sum() / (a - 1)
    msw = ((d.e - d.groupby("g").e.transform("mean")) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)

pl = pd.read_csv(C.DERIVED_DIR / "predictions_long.csv")
tf = pd.read_csv(C.DERIVED_DIR / "tfidf_lr_article_predictions.csv").assign(model="TF-IDF")
df = pd.concat([pl[["model","article_id","domain","y_true","y_pred"]],
                tf[["model","article_id","domain","y_true","y_pred"]]])
rng = np.random.default_rng(C.RANDOM_SEED)
rows = []
for m, g in df.groupby("model"):
    err = (g.y_true != g.y_pred).astype(float).to_numpy()
    obs = icc(err, g.domain.to_numpy())
    null = []
    for _ in range(200):
        perm = err.copy()
        for c in range(C.N_CLASSES):
            idx = np.flatnonzero(g.y_true.to_numpy() == c)
            perm[idx] = rng.permutation(err[idx])
        null.append(icc(perm, g.domain.to_numpy()))
    pr = g.assign(e=err).groupby("domain").e.mean()
    rows.append({"model": m, "ICC_errors": round(obs, 3),
                 "ICC_null_mean": round(float(np.mean(null)), 3),
                 "ICC_null_max": round(float(np.max(null)), 3),
                 "publishers_error_rate_over_80pct": round(float((pr > 0.8).mean()), 3),
                 "publishers_error_rate_under_20pct": round(float((pr < 0.2).mean()), 3)})
out = pd.DataFrame(rows)
out.to_csv(C.DERIVED_DIR / "explain" / "error_clustering.csv", index=False)
print(out.to_string(index=False))

"""Which publishers are misread, and does the text track the continuous score?

Publisher-level analysis over the five models (four transformers + TF-IDF):
  1. error rate vs distance of the NewsGuard score from the nearest class
     threshold of the publisher's own class;
  2. Spearman correlation between the publisher's mean predicted level and its
     continuous score, overall and within each class;
  3. profile of publishers misclassified by all five models;
  4. error rate vs number of articles per publisher.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irgood import config as C
import run_tfidf_lr_baseline as B

e = pd.read_csv(B.ENRICHED, usecols=["article_id", "domain", "score", "label"])
# One publisher (labourlist.org) carries two scores, 80 and 92.5, both in l4:
# the label is unique per publisher, and the prevailing score is used.
assert e.groupby("domain").label.nunique().max() == 1
pubinfo = e.groupby("domain").agg(score=("score", lambda s: s.mode().iloc[0]), y_true=("label", "first"),
                                  n_articles=("article_id", "size"))
pl = pd.read_csv(C.DERIVED_DIR / "predictions_long.csv")
lean = pl.drop_duplicates("domain").set_index("domain").leaning
pubinfo["leaning"] = lean
tf = pd.read_csv(C.DERIVED_DIR / "tfidf_lr_article_predictions.csv").assign(model="TF-IDF")
art = pd.concat([pl[["model", "domain", "y_true", "y_pred"]], tf[["model", "domain", "y_true", "y_pred"]]])
models = C.MODEL_ORDER + ["TF-IDF"]

# Aggregated prediction per publisher (mode, ties -> lowest) and mean level.
def mode_low(s):
    c = s.value_counts(); return c[c == c.max()].index.min()
agg = art.groupby(["model", "domain"]).y_pred.agg(pred=mode_low, mean_pred="mean").reset_index()
P = agg.pivot(index="domain", columns="model", values="pred")[models]
MEANP = agg.pivot(index="domain", columns="model", values="mean_pred")[models]
correct = P.eq(pubinfo.y_true, axis=0)

# 1. distance from the thresholds of the publisher's own class
bounds = {0: [39.5], 1: [39.5, 59.5], 2: [59.5, 74.5], 3: [74.5, 99.5], 4: [99.5]}
pubinfo["dist"] = [min(abs(s - b) for b in bounds[c]) for s, c in zip(pubinfo.score, pubinfo.y_true)]
res = {}
bins = pd.cut(pubinfo.dist, [0, 3, 6, 11, 100], labels=["<=2.5", "3-5.5", "6-10.5", ">10.5"])
tab = (1 - correct).groupby(bins, observed=True).mean().round(3)
tab["n_publishers"] = bins.value_counts()
res["1_error_rate_by_distance_from_threshold"] = tab.to_dict(orient="index")
# l5 is a single point (score 100); report the classes l1-l4 separately too
sub = pubinfo.y_true < 4
tab2 = (1 - correct[sub]).groupby(bins[sub], observed=True).mean().round(3)
tab2["n_publishers"] = bins[sub].value_counts()
res["1b_same_excluding_l5"] = tab2.to_dict(orient="index")

# 2. does the mean predicted level track the continuous score?
res["2_spearman_overall"] = {m: round(spearmanr(MEANP[m], pubinfo.score)[0], 3) for m in models}
within = {}
for c in range(4):  # l5 has no score variance
    idx = pubinfo.y_true == c
    within[C.CLASS_LABELS[c]] = {m: round(spearmanr(MEANP.loc[idx, m], pubinfo.score[idx])[0], 3) for m in models}
res["2_spearman_within_class"] = within

# 3. publishers misclassified by all five
allw = (~correct).all(axis=1)
allr = correct.all(axis=1)
res["3_counts"] = {"wrong_for_all_5": int(allw.sum()), "right_for_all_5": int(allr.sum()),
                   "mixed": int((~allw & ~allr).sum())}
prof = lambda m: {
    "true_class": pubinfo[m].y_true.map(lambda c: C.CLASS_LABELS[c]).value_counts().to_dict(),
    "leaning": pubinfo[m].leaning.value_counts().to_dict(),
    "median_articles": float(pubinfo[m].n_articles.median()),
    "median_dist_threshold": float(pubinfo[m].dist.median()),
}
res["3_profile_wrong_all5"] = prof(allw)
res["3_profile_right_all5"] = prof(allr)
sign = np.sign(P[allw].median(axis=1) - pubinfo[allw].y_true)
res["3_direction_wrong_all5"] = {"over": int((sign > 0).sum()), "under": int((sign < 0).sum())}
gap = (P[allw].median(axis=1) - pubinfo[allw].y_true).abs()
res["3_levels_off_wrong_all5"] = gap.value_counts().sort_index().to_dict()
pairs = pd.crosstab(pubinfo[allw].y_true.map(lambda c: C.CLASS_LABELS[c]),
                    P[allw].median(axis=1).round().astype(int).map(lambda c: C.CLASS_LABELS[c]))
res["3_true_vs_predicted_wrong_all5"] = pairs.to_dict(orient="index")

# 4. evidence: error rate vs number of articles
nb = pd.cut(pubinfo.n_articles, [0, 5, 15, 40, 1000], labels=["1-5", "6-15", "16-40", ">40"])
t4 = (1 - correct).groupby(nb, observed=True).mean().round(3)
t4["n_publishers"] = nb.value_counts()
res["4_error_rate_by_n_articles"] = t4.to_dict(orient="index")

out = C.DERIVED_DIR / "explain"
pubinfo.assign(**{f"correct_{m}": correct[m] for m in models}).to_csv(out / "publisher_profile.csv")
(out / "publisher_results.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))

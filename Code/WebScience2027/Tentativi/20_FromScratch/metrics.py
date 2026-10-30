"""Step 7: evaluation, recomputed from the article outputs (logits or predicted scores), per outlet.

1. from articles to outlet: the classifier takes the majority level, the regressor the mean score;
2. conversions, to compare the two approaches on the same ground:
   - regressor -> levels: NewsGuard bounds, training quotas (attempt 16) or article vote (attempt 17);
     regressor -> score re-mapped to the training score quantiles;
   - classifier -> score: expected score = probabilities × mean training score of each level;
3. metrics on levels (F1, tolerant, kappa, ...) and on scores (MAE, Spearman, "within 10/20 points", ...);
4. the views of a model: every (true, predicted) pair it gives, with its metrics;
5. 95% intervals: bootstrap over outlets, also for the difference between two views on the same outlets.
Every conversion is computed once on the real data; the bootstrap only resamples outlets.
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn import metrics as M

from common import BOUNDS, LEVELS, SEED, majority
from models import PROBS

WITHIN = (10, 20)                     # "within N points": the tolerant accuracy of the score


# ------------------------------------------------------------------------- 1. from articles to outlet
def outlets_classifier(pa, outlets):
    """One row per outlet: true level and score, fold, majority level of its articles (ties: the lowest) and mean
    probabilities of the 5 levels."""
    g = pa.groupby("domain")
    T = pd.DataFrame({"true": outlets.label, "true score": outlets.score, "fold": outlets.fold})
    T["predicted"] = g.prediction.agg(majority)
    T[PROBS] = g[PROBS].mean()
    return T


def outlets_regressor(pa, outlets):
    """One row per outlet: true level and score, fold, mean predicted score of its articles."""
    T = pd.DataFrame({"true": outlets.label, "true score": outlets.score, "fold": outlets.fold})
    T["predicted"] = pa.groupby("domain").prediction.mean()
    return T


# ------------------------------------------------------------------------- 2. conversions
def levels_bounds(score):
    """Score -> level with the NewsGuard bounds, no adaptation."""
    return np.digitize(np.asarray(score, float), BOUNDS)


def levels_quotas(predicted, fold, true_level):
    """Score -> level with the training quotas (attempt 16): in every fold the test predictions are ranked and cut at
    the level proportions of the training rows (the other folds)."""
    predicted, fold, true_level = (np.asarray(x) for x in (predicted, fold, true_level))
    out = np.zeros(len(predicted), int)
    for k in np.unique(fold):
        te = fold == k
        quotas = np.cumsum(np.bincount(true_level[~te].astype(int), minlength=5))[:-1] / (~te).sum()
        out[te] = np.searchsorted(np.quantile(predicted[te], quotas), predicted[te])
    return out


def levels_article_vote(pa, art):
    """Score -> level with the article vote (attempt 17): every article gets a level with the quotas of the training
    articles of its fold, then the outlet takes the most voted level (ties: the lowest)."""
    a = pa.merge(art[["article_id", "label"]], on="article_id")
    a["level"] = levels_quotas(a.prediction, a.fold, a.label)
    return a.groupby("domain").level.agg(majority)


def expected_score(T, outlets):
    """Classifier -> score: mean probabilities × mean score of each level among the training outlets of the fold."""
    out = pd.Series(np.nan, index=T.index)
    for k in T.fold.unique():
        means = outlets[outlets.fold != k].groupby("label").score.mean().reindex(range(5)).to_numpy()
        te = T.fold == k
        out[te] = T.loc[te, PROBS].to_numpy() @ means
    return out


def remapped_score(predicted, fold, true_score):
    """Regressor score re-mapped to the training quantiles (like histogram equalisation): in every fold, the outlet
    at q% of the test predictions gets the score at q% of the training true scores."""
    predicted, fold, true_score = (np.asarray(x, float) for x in (predicted, fold, true_score))
    out = np.zeros(len(predicted))
    for k in np.unique(fold):
        te = fold == k
        q = (pd.Series(predicted[te]).rank().to_numpy() - 1) / max(te.sum() - 1, 1)
        out[te] = np.quantile(true_score[~te], q)
    return out


# ------------------------------------------------------------------------- 3. metrics
def _safe(f):
    try:
        return float(f())
    except ValueError:
        return np.nan


def level_metrics(y, p):
    """Metrics between two series of levels 0-4 (true, predicted)."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    f1 = M.f1_score(y, p, average=None, labels=range(5), zero_division=0)
    out = {"F1 macro": f1.mean(),
           "accuracy": M.accuracy_score(y, p),
           "balanced accuracy": M.balanced_accuracy_score(y, p),
           "tolerant (±1 level)": np.mean(np.abs(y - p) <= 1),
           "mean level error": np.mean(np.abs(y - p)),
           "kappa": M.cohen_kappa_score(y, p),
           "quadratic kappa": M.cohen_kappa_score(y, p, weights="quadratic"),
           "Spearman": _safe(lambda: spearmanr(y, p).correlation),
           "Kendall": _safe(lambda: kendalltau(y, p).correlation),
           "F1 weighted": M.f1_score(y, p, average="weighted", labels=range(5), zero_division=0)}
    out.update({f"F1 {LEVELS[i]}": f1[i] for i in range(5)})
    a, b = y >= 2, p >= 2                                           # reliable = l3 or above (score >= 60)
    out.update({"reliable: accuracy": M.accuracy_score(a, b),
                "reliable: F1 macro": M.f1_score(a, b, average="macro", zero_division=0),
                "reliable: MCC": M.matthews_corrcoef(a, b)})
    return out


def score_metrics(y, p):
    """Metrics between two series of scores 0-100 (true, predicted)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    out = {"MAE": np.mean(np.abs(e)),
           "RMSE": np.sqrt(np.mean(e ** 2)),
           "median error": np.median(np.abs(e)),
           "R²": M.r2_score(y, p),
           "Spearman": _safe(lambda: spearmanr(y, p).correlation),
           "Pearson": _safe(lambda: pearsonr(y, p)[0]),
           "Kendall": _safe(lambda: kendalltau(y, p).correlation),
           "Lin's CCC": 2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2),
           "bias (predicted − true)": np.mean(e)}
    out.update({f"within {n} points": np.mean(np.abs(e) <= n) for n in WITHIN})
    a, b = y >= 60, p >= 60
    out.update({"reliable: accuracy": M.accuracy_score(a, b),
                "reliable: F1 macro": M.f1_score(a, b, average="macro", zero_division=0),
                "reliable: AUC": _safe(lambda: M.roc_auc_score(a, p))})
    return out


def probability_metrics(T):
    """Metrics that use the classifier probabilities (mean per outlet)."""
    y = T.true.to_numpy(int)
    P = T[PROBS].to_numpy(); P = P / P.sum(1, keepdims=True)
    return {"AUC macro (5 levels)": _safe(lambda: M.roc_auc_score(y, P, multi_class="ovr", average="macro", labels=range(5))),
            "log loss": M.log_loss(y, np.clip(P, 1e-12, 1), labels=range(5)),
            "Brier": np.mean(((np.eye(5)[y] - P) ** 2).sum(1)),
            "reliable: AUC (probabilities)": _safe(lambda: M.roc_auc_score(y >= 2, P[:, 2:].sum(1)))}


METRICS = {"levels": level_metrics, "score": score_metrics}


# ------------------------------------------------------------------------- 4. the views of a model
def classifier_views(T, outlets):
    """{view: (space, true, predicted)} of a classifier."""
    return {"majority level": ("levels", T.true, T.predicted),
            "expected score": ("score", T["true score"], expected_score(T, outlets))}


def regressor_views(T, pa, art):
    """{view: (space, true, predicted)} of a regressor."""
    return {"mean score": ("score", T["true score"], T.predicted),
            "remapped score": ("score", T["true score"], remapped_score(T.predicted, T.fold, T["true score"])),
            "levels: NewsGuard bounds": ("levels", T.true, levels_bounds(T.predicted)),
            "levels: training quotas": ("levels", T.true, levels_quotas(T.predicted, T.fold, T.true)),
            "levels: article vote": ("levels", T.true, levels_article_vote(pa, art).reindex(T.index))}


def baseline_views(outlets):
    """Without reading the text: the most frequent training level, and the mean training score."""
    level, score = pd.Series(0, index=outlets.index), pd.Series(0.0, index=outlets.index)
    for k in outlets.fold.unique():
        tr, te = outlets.fold != k, outlets.fold == k
        level[te] = outlets.label[tr].mode().min(); score[te] = outlets.score[tr].mean()
    return {"most frequent level": ("levels", outlets.label, level), "mean training score": ("score", outlets.score, score)}


# ------------------------------------------------------------------------- 5. intervals
def bootstrap(y, p, space, keys, B=2000, seed=SEED):
    """95% intervals resampling outlets with replacement (the true-predicted pairs stay fixed)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    values = {c: [] for c in keys}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        m = METRICS[space](y[i], p[i])
        for c in keys:
            values[c].append(m[c])
    return {c: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for c, v in values.items()}


def bootstrap_difference(view1, view2, key, B=2000, seed=SEED):
    """view1 − view2 for the same metric, on the same resampled outlets. Returns the difference and its 95% interval:
    if the interval does not contain 0, the difference is not due to chance."""
    (s1, y1, p1), (s2, y2, p2) = view1, view2
    y1, p1, y2, p2 = (np.asarray(x) for x in (y1, p1, y2, p2))
    rng = np.random.default_rng(seed)
    diff = []
    for _ in range(B):
        i = rng.integers(0, len(y1), len(y1))
        diff.append(METRICS[s1](y1[i], p1[i])[key] - METRICS[s2](y2[i], p2[i])[key])
    d = METRICS[s1](y1, p1)[key] - METRICS[s2](y2, p2)[key]
    return d, (np.nanpercentile(diff, 2.5), np.nanpercentile(diff, 97.5))

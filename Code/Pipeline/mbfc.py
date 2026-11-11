"""Comparison with Media Bias/Fact Check (MBFC), a second expert. MBFC never enters training: it is only a reference.

MBFC does not give a score but a class (Factual Reporting). With Very Low and Very High almost empty among our outlets
(1 each), we use 4 classes: Low (+ Very Low), Mixed, Mostly Factual, High (+ Very High).

To count how often a score (the true NewsGuard score, or the score predicted by a model) falls in the right MBFC class,
the score must become a class. Three conversions, each LEARNED on the outlets of the other folds and APPLIED to the
outlets of the test fold, exactly the same way for NewsGuard and for the models:
1. quotas:     rank the outlets by score and cut them at the MBFC class proportions of the other folds;
2. medians:    every class has a typical score (the median of its outlets in the other folds); an outlet goes to the
               class whose typical score is closest;
3. thresholds: the 3 cuts of the score that make the fewest errors on the other folds (sum of the class distances).
"""
import numpy as np
import pandas as pd
from sklearn import metrics as M

from common import DATA

MBFC = DATA / "tentativi" / "10_EtichetteMBFC" / "mbfc_pulito.csv"
FACTUAL = {"Very Low": 0, "Low": 0, "Mixed": 1, "Mostly Factual": 2, "High": 3, "Very High": 3}
CLASSES = ["Low", "Mixed", "Mostly Factual", "High"]


def load(outlets):
    """MBFC class (0-3) of our outlets that MBFC rates (same domain, site still alive)."""
    mb = pd.read_csv(MBFC)
    mb["domain"] = mb.dominio.str.lower().str.replace(r"^www\.", "", regex=True)
    mb = mb[~mb.morto].drop_duplicates("domain").set_index("domain")
    common = outlets.index[outlets.index.str.lower().isin(mb.index)]
    return pd.Series(mb.factual.reindex(common.str.lower()).map(FACTUAL).to_numpy(), index=common).dropna().astype(int)


# ------------------------------------------------------------------------- score -> MBFC class
def quotas(train_score, train_class, test_score):
    """Rank the test outlets by score and cut them at the class proportions of the training outlets."""
    cuts = np.cumsum(np.bincount(train_class, minlength=len(CLASSES)))[:-1] / len(train_class)
    return np.searchsorted(np.quantile(test_score, cuts), test_score)


def medians(train_score, train_class, test_score):
    """Every class has the median score of its training outlets; a test outlet goes to the closest one."""
    typical = pd.Series(train_score).groupby(np.asarray(train_class)).median()
    distance = np.abs(np.asarray(test_score)[:, None] - typical.to_numpy()[None, :])
    return typical.index.to_numpy()[distance.argmin(1)]


def thresholds(train_score, train_class, test_score):
    """The cuts of the score that make the fewest errors (sum of |true class - assigned class|) on the training
    outlets, found exactly by dynamic programming; a cut falls halfway between two training scores."""
    o = np.argsort(train_score, kind="stable")
    s, c, n, K = np.asarray(train_score, float)[o], np.asarray(train_class)[o], len(o), len(CLASSES)
    ends = [0] + [i for i in range(1, n) if s[i] > s[i - 1]] + [n]                 # where a cut may fall
    errors = np.array([np.concatenate([[0], np.cumsum(np.abs(c - k))]) for k in range(K)])   # errors[k, j]: class k on s[:j]
    best = {e: (errors[0, e], []) for e in ends}                                   # class 0 on s[:e]
    for k in range(1, K):
        best = {e: min((best[a][0] + errors[k, e] - errors[k, a], best[a][1] + [a]) for a in ends if a <= e) for e in ends}
    cuts = [-np.inf if a == 0 else np.inf if a == n else (s[a - 1] + s[a]) / 2 for a in best[n][1]]
    return np.digitize(test_score, cuts)


CONVERSIONS = {"quotas": quotas, "medians": medians, "thresholds": thresholds}


def by_fold(conversion, score, mbfc_class, fold):
    """Every outlet's MBFC class from its score, with the conversion learned on the outlets of the other folds."""
    score, mbfc_class, fold = (np.asarray(x) for x in (score, mbfc_class, fold))
    out = np.zeros(len(score), int)
    for k in np.unique(fold):
        te = fold == k
        out[te] = CONVERSIONS[conversion](score[~te], mbfc_class[~te], score[te])
    return out


# ------------------------------------------------------------------------- agreement with MBFC
def agreement(y, p):
    """Exact class, class within ±1, quadratic kappa (1 = perfect, 0 = chance), macro F1."""
    y, p = np.asarray(y), np.asarray(p)
    return {"exact": np.mean(y == p), "within ±1": np.mean(np.abs(y - p) <= 1),
            "quadratic kappa": M.cohen_kappa_score(y, p, weights="quadratic"),
            "F1 macro": M.f1_score(y, p, average="macro", labels=range(len(CLASSES)), zero_division=0)}

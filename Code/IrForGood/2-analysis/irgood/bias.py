"""Political-leaning bias analysis.

Includes both the original (article-level chi-square) test and the
cluster-aware versions the TCSS reviewers asked for. The two are kept side by
side deliberately: the point is to show how much of the original effect
survives once the clustering is respected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from . import config as cfg


def error_rate_by_leaning(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["model", "leaning"], observed=True)
        .agg(error_rate=("correct", lambda s: 1 - s.mean()), n=("correct", "size"))
        .reset_index()
    )
    return out


def med_by_leaning(df: pd.DataFrame, over: str = "errors") -> pd.DataFrame:
    """Mean error direction per leaning. See metrics.mean_error_direction."""
    data = df[~df["correct"]] if over == "errors" else df
    out = (
        data.groupby(["model", "leaning"], observed=True)["signed_error"]
        .agg(med="mean", sd="std", n="size")
        .reset_index()
    )
    out["med_denominator"] = over
    return out


# --------------------------------------------------------------------------
# Independence tests
# --------------------------------------------------------------------------
def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1))))


def chi_square_article_level(df: pd.DataFrame) -> pd.DataFrame:
    """The test as originally published: one row per article.

    This is the pseudo-replication the reviewers objected to. Reliability and
    leaning are properties of the *publisher*, so the ~19.5k articles are not
    19.5k independent observations -- they are 439 clusters. The test is kept
    here only so the corrected versions can be compared against it.
    """
    rows = []
    for model, g in df.groupby("model", observed=True):
        table = pd.crosstab(g["leaning"], ~g["correct"])
        chi2, p, dof, _ = chi2_contingency(table)
        rows.append(
            {
                "model": model,
                "unit": "article (uncorrected)",
                "n": len(g),
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "cramers_v": cramers_v(chi2, len(g), *table.shape),
            }
        )
    return pd.DataFrame(rows)


def chi_square_publisher_level(pub: pd.DataFrame) -> pd.DataFrame:
    """The same question asked at the unit the labels actually live on.

    Each publisher contributes one observation (was its aggregated prediction
    correct?), so the independence assumption holds.
    """
    rows = []
    for model, g in pub.groupby("model", observed=True):
        table = pd.crosstab(g["leaning"], ~g["correct"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, expected = chi2_contingency(table)
        rows.append(
            {
                "model": model,
                "unit": "publisher (clustered)",
                "n": len(g),
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "cramers_v": cramers_v(chi2, len(g), *table.shape),
                "min_expected": float(expected.min()),
                "pct_expected_lt5": float(100 * (expected < 5).mean()),
            }
        )
    return pd.DataFrame(rows)


def mixed_effects_by_leaning(
    df: pd.DataFrame, control_for_reliability: bool = True
) -> pd.DataFrame:
    """Logistic regression of error on leaning, with publisher-clustered SEs.

    Two reviewer points are addressed at once:

    * clustering -- standard errors are clustered on publisher, so the model
      cannot treat 200 articles from one outlet as 200 independent facts;
    * confounding -- with ``control_for_reliability`` the true reliability
      level enters the model, so the leaning coefficients are no longer
      absorbing "far-left outlets happen to be unreliable".

    Returns one row per coefficient per model.
    """
    import statsmodels.formula.api as smf

    rows = []
    for model, g in df.groupby("model", observed=True):
        d = g.copy()
        d["error"] = (~d["correct"]).astype(int)
        d["leaning"] = d["leaning"].astype(str)
        d["reliability"] = d["y_true"].astype(str)
        # Reference category: the large, politically unmarked bucket.
        ref = cfg.LEANING_UNLABELLED
        d["leaning"] = pd.Categorical(
            d["leaning"],
            categories=[ref] + [x for x in cfg.LEANING_ORDER if x != ref],
        )
        formula = "error ~ leaning"
        if control_for_reliability:
            formula += " + reliability"
        try:
            fit = smf.logit(formula, data=d).fit(
                disp=False, cov_type="cluster", cov_kwds={"groups": d["domain"]}
            )
        except Exception as exc:  # singular fits are possible on small strata
            rows.append({"model": model, "term": "FIT FAILED", "note": str(exc)[:120]})
            continue
        for term in fit.params.index:
            rows.append(
                {
                    "model": model,
                    "term": term,
                    "coef": float(fit.params[term]),
                    "odds_ratio": float(np.exp(fit.params[term])),
                    "std_err": float(fit.bse[term]),
                    "p_value": float(fit.pvalues[term]),
                    "ci_low": float(fit.conf_int().loc[term, 0]),
                    "ci_high": float(fit.conf_int().loc[term, 1]),
                    "controls_reliability": control_for_reliability,
                }
            )
    return pd.DataFrame(rows)


def leaning_by_reliability_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Article counts, leaning x reliability.

    This is the confound in one table: if leaning and reliability are strongly
    associated in the corpus, an apparent "political bias" in the errors can be
    a reliability effect wearing a political coat.
    """
    one = df[df["model"] == cfg.MODEL_ORDER[0]]
    tab = pd.crosstab(one["leaning"], one["y_true"])
    tab.columns = [cfg.CLASS_LABELS[c] for c in tab.columns]
    return tab

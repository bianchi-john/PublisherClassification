"""Metric definitions.

Each metric is defined exactly once here. The old notebooks reimplemented the
weighted-confusion formula three times and used one *name* for two different
things; both are called out in the docstrings below so the ambiguity cannot
silently come back.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config as C


# --------------------------------------------------------------------------
# Tolerant accuracy
# --------------------------------------------------------------------------
def tolerance_weights(alpha: float, n: int = C.N_CLASSES) -> np.ndarray:
    """W = I + alpha * (superdiagonal + subdiagonal)."""
    return np.eye(n) + alpha * (np.eye(n, k=1) + np.eye(n, k=-1))


def tolerant_accuracy(y_true, y_pred, alpha: float) -> float:
    """Partial-credit accuracy on an ordinal scale.

    An exact hit scores 1, an adjacent miss scores ``alpha``, anything further
    scores 0.

    Note on naming: at alpha=1.0 this coincides with "off-by-one accuracy"
    (the share of predictions with |y_hat - y| <= 1), but at alpha=0.5 it is a
    partial-credit *score*, not the accuracy of anything. The old notebooks
    used the label "tolerant accuracy" for both this and a hard |err| <= 1
    boolean, which are different quantities.
    """
    cm = confusion_matrix(y_true, y_pred, labels=C.CLASS_CODES)
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float((cm * tolerance_weights(alpha)).sum() / total)


def tolerant_accuracy_from_cm(cm_repr, alpha: float) -> float:
    """Same metric, from a confusion matrix stored as a string in a CSV cell."""
    if isinstance(cm_repr, str):
        values = [int(x) for x in re.findall(r"\d+", cm_repr)]
        n = int(round(np.sqrt(len(values))))
        cm = np.asarray(values).reshape(n, n)
    else:
        cm = np.asarray(cm_repr)
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float((cm * tolerance_weights(alpha, cm.shape[0])).sum() / total)


# --------------------------------------------------------------------------
# Metric bundle
# --------------------------------------------------------------------------
def metric_bundle(y_true, y_pred) -> dict[str, float]:
    """The six metrics reported everywhere in the paper, always macro-averaged.

    Macro (not weighted) is used consistently: notebook 6 reported macro F1 in
    one section and weighted F1 in another under near-identical labels, which
    made those numbers non-comparable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "Accuracy": float((y_true == y_pred).mean()),
        "Precision_Macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall_Macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "F1_Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Tolerant_0.5": tolerant_accuracy(y_true, y_pred, 0.5),
        "Tolerant_1.0": tolerant_accuracy(y_true, y_pred, 1.0),
    }


METRIC_LABELS = {
    "Accuracy": "Accuracy",
    "Precision_Macro": "Precision (macro)",
    "Recall_Macro": "Recall (macro)",
    "F1_Macro": "F1 (macro)",
    "Tolerant_0.5": r"Tolerant Acc. ($\alpha$=0.5)",
    "Tolerant_1.0": r"Tolerant Acc. ($\alpha$=1.0)",
}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def baselines(y_true) -> dict[str, float]:
    """Reference values for each metric.

    - Accuracy and both tolerant accuracies: a constant majority-class
      predictor, evaluated on the same ground truth.
    - Macro P/R/F1: the analytic uniform-random floor, 1/K.

    The 1/K figure is exact for macro *recall* only. Macro precision and macro
    F1 of a uniform random classifier on an imbalanced problem are not exactly
    1/K, so 1/K is an approximation. It is kept because the paper's figures use
    it, but it should be described as a nominal floor, not an exact
    expectation.
    """
    y_true = np.asarray(y_true)
    majority = int(pd.Series(y_true).mode().min())
    y_major = np.full_like(y_true, majority)
    random_floor = 1.0 / C.N_CLASSES
    return {
        "Accuracy": float((y_true == y_major).mean()),
        "Precision_Macro": random_floor,
        "Recall_Macro": random_floor,
        "F1_Macro": random_floor,
        "Tolerant_0.5": tolerant_accuracy(y_true, y_major, 0.5),
        "Tolerant_1.0": tolerant_accuracy(y_true, y_major, 1.0),
    }


# --------------------------------------------------------------------------
# Error structure
# --------------------------------------------------------------------------
def error_rate_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """1 - recall, per true reliability class, per model."""
    out = (
        df.groupby(["model", "y_true"], observed=True)["correct"]
        .mean()
        .rsub(1.0)
        .rename("error_rate")
        .reset_index()
    )
    out["class_label"] = out["y_true"].map(lambda c: C.CLASS_LABELS[c])
    return out


def mean_error_direction(
    df: pd.DataFrame, group_cols: list[str], over: str = "errors"
) -> pd.DataFrame:
    """Mean signed error (y_hat - y). Positive = reliability overestimated.

    ``over`` selects the denominator, and this is the detail that matters:

    - ``"errors"``  : mean over misclassified items only, i.e. 1/|E| * sum.
                      This is the formula written in the paper.
    - ``"all"``     : mean over every item in the group, correct ones included
                      (they contribute 0). Smaller in magnitude.

    The published figures are inconsistent on this point: the per-reliability
    MED numbers match ``over="all"`` while the per-leaning MED numbers match
    ``over="errors"``. Both were labelled "MED" and the paper states the
    ``"errors"`` formula for both. Pick one deliberately and say which.
    """
    if over not in {"errors", "all"}:
        raise ValueError("over must be 'errors' or 'all'")
    data = df[~df["correct"]] if over == "errors" else df
    out = (
        data.groupby(group_cols, observed=True)["signed_error"]
        .mean()
        .rename("med")
        .reset_index()
    )
    out["med_denominator"] = over
    return out


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "Accuracy",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = C.RANDOM_SEED,
    cluster: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI.

    ``cluster`` enables a *cluster* bootstrap: pass the publisher of each item
    and whole publishers are resampled instead of individual articles. That is
    the correct unit here, because articles from one publisher share a label
    and are not independent. Ignoring this is what made the original
    chi-square test a pseudo-replication.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    point = metric_bundle(y_true, y_pred)[metric]

    # Accuracy and the tolerant accuracies are elementwise scores, so a
    # resample is just a mean over sampled positions -- no need to rebuild a
    # confusion matrix or call sklearn 2000 times. Only the macro metrics,
    # which are not decomposable per item, take the slow path.
    diff = np.abs(y_true - y_pred)
    fast_scores = {
        "Accuracy": (diff == 0).astype(float),
        "Tolerant_0.5": np.where(diff == 0, 1.0, np.where(diff == 1, 0.5, 0.0)),
        "Tolerant_1.0": (diff <= 1).astype(float),
    }
    per_item = fast_scores.get(metric)

    if cluster is None:
        n = len(y_true)
        idx_pool = [rng.integers(0, n, n) for _ in range(n_boot)]
    else:
        cluster = np.asarray(cluster)
        order = np.argsort(cluster, kind="stable")
        sorted_c = cluster[order]
        starts = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1]])
        members = np.split(order, starts[1:])
        n_groups = len(members)
        idx_pool = [
            np.concatenate([members[k] for k in rng.integers(0, n_groups, n_groups)])
            for _ in range(n_boot)
        ]

    if per_item is not None:
        stats = np.fromiter((per_item[i].mean() for i in idx_pool), float, n_boot)
    else:
        stats = np.fromiter(
            (metric_bundle(y_true[i], y_pred[i])[metric] for i in idx_pool),
            float,
            n_boot,
        )

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)

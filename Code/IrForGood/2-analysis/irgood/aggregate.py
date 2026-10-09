"""Stage 2: article-level predictions -> publisher-level reliability.

Also hosts the aggregation learning curve (accuracy as a function of how many
articles are voted), which is the clean way to isolate what aggregation buys:
k=1 is the article-level baseline measured on the same unit as k=10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .metrics import metric_bundle


def _mode_lowest(values: pd.Series | np.ndarray) -> int:
    """Majority vote; ties go to the LOWEST class code.

    scipy.stats.mode behaves this way, which is what the published
    publisher-level numbers were computed with. It is not neutral: on an
    ordinal reliability scale it always breaks ties toward *less* reliable,
    and it decides 3-5% of publishers (BERT 13, RoBERTa 20, Llama 15 of 439).
    Kept as the default for continuity, but stated explicitly so it can be
    reported -- or changed deliberately.
    """
    counts = np.bincount(np.asarray(values, dtype=int), minlength=C.N_CLASSES)
    return int(np.flatnonzero(counts == counts.max())[0])


def aggregate_to_publisher(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, publisher): true label, voted label, support.

    The publisher's true label is taken as the unique label of its articles;
    a publisher whose articles disagree would be a data error, so that is
    checked rather than assumed.
    """
    bad = (
        preds.groupby(["model", "domain"], observed=True)["y_true"].nunique().gt(1).sum()
    )
    if bad:
        raise ValueError(f"{bad} publisher(s) have inconsistent ground-truth labels")

    out = (
        preds.groupby(["model", "domain"], observed=True)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", _mode_lowest),
            n_articles=("y_pred", "size"),
        )
        .reset_index()
    )
    out["correct"] = out["y_true"] == out["y_pred"]
    out["signed_error"] = out["y_pred"] - out["y_true"]
    if "leaning" in preds.columns:
        leaning = preds.groupby("domain", observed=True)["leaning"].first()
        out["leaning"] = out["domain"].map(leaning)
    return out


def publisher_metrics(pub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in pub.groupby("model", observed=True):
        rows.append({"model": model, "n_publishers": len(g), **metric_bundle(g["y_true"], g["y_pred"])})
    return pd.DataFrame(rows)


def article_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in preds.groupby("model", observed=True):
        rows.append({"model": model, "n_articles": len(g), **metric_bundle(g["y_true"], g["y_pred"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Aggregation learning curve
# --------------------------------------------------------------------------
def aggregation_curve(
    preds: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20, 30),
    n_repeats: int = 20,
    seed: int = C.RANDOM_SEED,
    fixed_population: bool = True,
) -> pd.DataFrame:
    """Publisher accuracy as a function of k, the number of articles voted.

    For each publisher with at least k articles, k predictions are sampled
    without replacement and aggregated; this is repeated ``n_repeats`` times
    and averaged.

    ``fixed_population=True`` restricts every k to the SAME publishers -- those
    with at least max(k_values) articles. This matters more than it looks. The
    original notebook let the population shrink as k grew, so accuracy appeared
    to *fall* for large k; that was survivorship, not a real effect: at k=100
    only the ~38 largest (and more ambiguous) publishers survived. With the
    population fixed, the curve answers the intended question -- how many
    articles do you need to profile a source -- rather than silently comparing
    different sets of sources.
    """
    rng = np.random.default_rng(seed)
    k_values = tuple(sorted(k_values))
    rows = []

    for model, g in preds.groupby("model", observed=True):
        by_domain = {d: sub["y_pred"].to_numpy() for d, sub in g.groupby("domain", observed=True)}
        truth = g.groupby("domain", observed=True)["y_true"].first().to_dict()

        if fixed_population:
            k_max = max(k_values)
            domains = [d for d, v in by_domain.items() if len(v) >= k_max]
        else:
            domains = list(by_domain)

        for k in k_values:
            accs, tol1 = [], []
            for _ in range(n_repeats):
                yt, yp = [], []
                for d in domains:
                    v = by_domain[d]
                    if len(v) < k:
                        continue
                    pick = v if len(v) == k else rng.choice(v, size=k, replace=False)
                    yt.append(truth[d])
                    yp.append(_mode_lowest(pick))
                if not yt:
                    continue
                yt_a, yp_a = np.array(yt), np.array(yp)
                accs.append((yt_a == yp_a).mean())
                tol1.append((np.abs(yt_a - yp_a) <= 1).mean())
            if accs:
                rows.append(
                    {
                        "model": model,
                        "k": k,
                        "accuracy_mean": float(np.mean(accs)),
                        "accuracy_std": float(np.std(accs)),
                        "tolerant_1.0_mean": float(np.mean(tol1)),
                        "n_publishers": len(domains),
                        "fixed_population": fixed_population,
                    }
                )

    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(
        out["model"], categories=C.MODEL_ORDER_WITH_GPT, ordered=True
    )
    return out.sort_values(["model", "k"]).reset_index(drop=True)


def tie_report(preds: pd.DataFrame) -> pd.DataFrame:
    """How many publishers are decided by the tie-break rule, per model."""
    rows = []
    for model, g in preds.groupby("model", observed=True):
        n_tied = 0
        for _, sub in g.groupby("domain", observed=True):
            counts = np.bincount(sub["y_pred"].to_numpy(dtype=int), minlength=C.N_CLASSES)
            if (counts == counts.max()).sum() > 1:
                n_tied += 1
        n_dom = g["domain"].nunique()
        rows.append(
            {
                "model": model,
                "n_publishers": n_dom,
                "n_tied": n_tied,
                "pct_tied": 100 * n_tied / n_dom if n_dom else float("nan"),
            }
        )
    return pd.DataFrame(rows)

"""Loading layer: turns the heterogeneous result files into one tidy table.

The fine-tuned result files disagree on column names (BERT uses
``True_Labels``/``Predicted_Labels``, the others ``Val_True_Labels``/
``Val_Predictions``) and store one row per fold with Python-literal lists in the
cells. Everything downstream works on a single long DataFrame instead:

    model | article_id | url | domain | fold | y_true | y_pred

Loading is the only place that knows about those file quirks.
"""
from __future__ import annotations

import ast

import pandas as pd

from . import config as C

# Column-name candidates, in priority order.
_TRUE_COLS = ["True_Labels", "Val_True_Labels"]
_PRED_COLS = ["Predicted_Labels", "Val_Predictions"]
_ID_COLS = ["Article_IDs", "Val_Article_IDs"]

TIDY_COLUMNS = ["model", "article_id", "url", "domain", "fold", "y_true", "y_pred"]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of {candidates} found. Available columns: {df.columns.tolist()}"
    )


def _parse_list(cell) -> list:
    """Cells hold Python literals such as '[0, 1, 2]'."""
    return ast.literal_eval(cell) if isinstance(cell, str) else list(cell)


def load_finetuned_predictions(model_name: str) -> pd.DataFrame:
    """Per-article predictions for one fine-tuned model, pooled over the 5 folds.

    Because the evaluation is 5-fold and publisher-disjoint, pooling the five
    test folds yields exactly one out-of-sample prediction per article.
    """
    path = C.FINETUNED_MODELS[model_name]["enriched_csv"]
    raw = pd.read_csv(path)

    true_col = _first_present(raw, _TRUE_COLS)
    pred_col = _first_present(raw, _PRED_COLS)
    id_col = _first_present(raw, _ID_COLS)

    frames = []
    for _, row in raw.iterrows():
        frames.append(
            pd.DataFrame(
                {
                    "article_id": _parse_list(row[id_col]),
                    "url": _parse_list(row["Article_URLs"]),
                    "domain": _parse_list(row["Article_Domains"]),
                    "y_true": _parse_list(row[true_col]),
                    "y_pred": _parse_list(row[pred_col]),
                    "fold": int(row["Fold"]),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "model", model_name)
    return out[TIDY_COLUMNS]


def load_gpt_predictions() -> pd.DataFrame:
    """Zero-shot GPT-5-mini baseline, mapped onto the same schema.

    The raw file stores predictions on a 1..5 scale; labels are 0..4, so the
    prediction is shifted. A handful of articles (7 of 19,476 -- the longest
    ones) have no prediction and are dropped.
    """
    raw = pd.read_csv(C.GPT_PRED_CSV)
    n_missing = int(raw["pred_1_5"].isna().sum())
    raw = raw.dropna(subset=["pred_1_5"]).copy()

    out = pd.DataFrame(
        {
            "model": C.GPT_MODEL_NAME,
            "article_id": raw["idx"].astype(int),
            "url": pd.NA,  # not carried in this file; join on article_id if needed
            "domain": raw["domain"],
            "fold": pd.NA,  # zero-shot: no folds
            "y_true": raw["label"].astype(int),
            "y_pred": raw["pred_1_5"].astype(int) - 1,
        }
    )
    out.attrs["n_missing"] = n_missing
    return out[TIDY_COLUMNS]


def load_publisher_leaning() -> pd.DataFrame:
    """domain -> political leaning, from the NewsGuard master table.

    Publishers NewsGuard did not assign an orientation to become
    'Non labelled', which is how the paper treats them.
    """
    ng = pd.read_csv(C.NEWSGUARD_CSV, low_memory=False)
    out = (
        ng[["Domain", "Orientation"]]
        .drop_duplicates("Domain")
        .rename(columns={"Domain": "domain", "Orientation": "leaning"})
    )
    out["leaning"] = out["leaning"].fillna(C.LEANING_UNLABELLED)
    return out


def load_all_predictions(include_gpt: bool = True, with_leaning: bool = True) -> pd.DataFrame:
    """The tidy table every analysis starts from."""
    frames = [load_finetuned_predictions(m) for m in C.MODEL_ORDER]
    if include_gpt:
        frames.append(load_gpt_predictions())
    preds = pd.concat(frames, ignore_index=True)

    # The zero-shot baseline has no folds of its own, but the five
    # publisher-disjoint partitions are a property of the *data*, not of any
    # model (verified: all four fine-tuned models share an identical
    # article -> fold map). Assigning GPT's predictions to the same partitions
    # lets it be evaluated fold by fold alongside the others.
    fold_map = (
        preds.loc[preds["model"].isin(C.MODEL_ORDER), ["article_id", "fold"]]
        .dropna()
        .drop_duplicates("article_id")
        .set_index("article_id")["fold"]
    )
    preds["fold"] = preds["fold"].fillna(preds["article_id"].map(fold_map))

    if with_leaning:
        leaning = load_publisher_leaning()
        preds = preds.merge(leaning, on="domain", how="left")
        preds["leaning"] = preds["leaning"].fillna(C.LEANING_UNLABELLED)
        preds["leaning"] = pd.Categorical(
            preds["leaning"], categories=C.LEANING_ORDER, ordered=True
        )

    preds["correct"] = preds["y_true"] == preds["y_pred"]
    preds["signed_error"] = preds["y_pred"] - preds["y_true"]
    preds["model"] = pd.Categorical(
        preds["model"], categories=C.MODEL_ORDER_WITH_GPT, ordered=True
    )
    return preds


def load_per_fold_metrics() -> pd.DataFrame:
    """The per-fold metric table used for the article-level box plot.

    Tolerant accuracies are not stored in the files, so they are recomputed
    here from each fold's confusion matrix.
    """
    from .metrics import tolerant_accuracy_from_cm  # local import avoids a cycle

    rows = []
    for model_name, spec in C.FINETUNED_MODELS.items():
        df = pd.read_csv(spec["per_fold_csv"])
        for _, r in df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "fold": int(r["Fold"]),
                    "Accuracy": r["Accuracy"],
                    "Precision_Macro": r["Precision_Macro"],
                    "Recall_Macro": r["Recall_Macro"],
                    "F1_Macro": r["F1_Macro"],
                    "Tolerant_0.5": tolerant_accuracy_from_cm(r["Confusion_Matrix"], 0.5),
                    "Tolerant_1.0": tolerant_accuracy_from_cm(r["Confusion_Matrix"], 1.0),
                }
            )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=C.MODEL_ORDER, ordered=True)
    return out.sort_values(["model", "fold"]).reset_index(drop=True)


def validate(preds: pd.DataFrame) -> pd.DataFrame:
    """Sanity checks that would have caught the known historical mistakes.

    Returns a small report rather than raising, so the notebook can print it.
    """
    rows = []
    for model, g in preds.groupby("model", observed=True):
        rows.append(
            {
                "model": model,
                "n_articles": len(g),
                "n_publishers": g["domain"].nunique(),
                "duplicate_article_ids": int(g["article_id"].duplicated().sum()),
                "labels_out_of_range": int((~g["y_true"].isin(C.CLASS_CODES)).sum()),
                "preds_out_of_range": int((~g["y_pred"].isin(C.CLASS_CODES)).sum()),
                "accuracy": g["correct"].mean(),
            }
        )
    report = pd.DataFrame(rows)

    # Every model must see the same ground truth for the same article.
    truth = preds.pivot_table(
        index="article_id", columns="model", values="y_true", observed=True
    )
    report.attrs["ground_truth_consistent"] = bool(truth.nunique(axis=1).max() == 1)
    return report


def per_fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics recomputed per (model, fold) from the predictions themselves.

    Unlike :func:`load_per_fold_metrics`, which reads the numbers the training
    runs wrote out, this works for any model in the tidy table -- including the
    zero-shot baseline, which has no training run and therefore no stored
    per-fold CSV.

    A caveat worth repeating wherever this is plotted: the spread means
    different things per model. For the fine-tuned models it mixes two sources
    of variation, a different training set and a different test set. GPT is not
    trained, so its spread is test-partition variability alone -- yet it is
    empirically the *widest* of the five (accuracy range 0.082 across folds,
    against 0.038-0.069 for the fine-tuned models), which says the zero-shot
    scorer is the least stable across publisher subsets, not the most.
    """
    from .metrics import metric_bundle

    rows = []
    for (model, fold), g in preds.dropna(subset=["fold"]).groupby(
        ["model", "fold"], observed=True
    ):
        rows.append(
            {"model": model, "fold": int(fold), **metric_bundle(g["y_true"], g["y_pred"])}
        )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(
        out["model"], categories=C.MODEL_ORDER_WITH_GPT, ordered=True
    )
    return out.sort_values(["model", "fold"]).reset_index(drop=True)

"""TF-IDF + Logistic Regression baseline.

A lexical reference point for the transformer results: it answers whether the
task is solvable by surface word statistics alone. It runs under the *same*
publisher-disjoint folds as the fine-tuned models (corpus_folds.csv), fitting
the TF-IDF vocabulary on each fold's training articles only, so no test-fold
vocabulary leaks in. Predictions are pooled across the five test folds and then
aggregated to the publisher level by the same majority vote (ties -> lowest
class) used everywhere else in the pipeline.

Outputs:
  Results/derived/tfidf_lr_article_predictions.csv
  Results/derived/tfidf_lr_publisher_predictions.csv
  Results/derived/tfidf_lr_metrics.csv   (article- and publisher-level bundles)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2-analysis"))
from irgood import config as C  # noqa: E402
from irgood import metrics as M  # noqa: E402

ENRICHED = (
    ROOT
    / "Dataset"
    / "CleannedDataset"
    / "newDataframeWithOldCleanedPoliticsVersion_enriched.csv"
)
# Publisher-disjoint fold assignment shared with the fine-tuned models
# (article_id, url, domain, fold). Released as data/corpus_folds.csv.
CORPUS_FOLDS = C.DERIVED_DIR / "corpus_folds.csv"


def load() -> pd.DataFrame:
    e = pd.read_csv(ENRICHED)
    cf = pd.read_csv(CORPUS_FOLDS)[["article_id", "fold"]]
    df = e.merge(cf, on="article_id", how="inner")
    # Leaning from the long predictions table (already joined to NewsGuard).
    pl = pd.read_csv(C.DERIVED_DIR / "predictions_long.csv")
    leaning = pl[pl.model == "BERT"][["article_id", "leaning"]].drop_duplicates()
    df = df.merge(leaning, on="article_id", how="left")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def run_cv(df: pd.DataFrame) -> pd.DataFrame:
    preds = np.empty(len(df), dtype=int)
    preds[:] = -1
    idx = df.reset_index(drop=True)
    for fold in sorted(idx.fold.unique()):
        tr = idx[idx.fold != fold]
        te = idx[idx.fold == fold]
        vec = TfidfVectorizer(
            sublinear_tf=True,
            min_df=5,
            max_df=0.9,
            ngram_range=(1, 2),
            max_features=100_000,
            strip_accents="unicode",
        )
        Xtr = vec.fit_transform(tr.text)
        Xte = vec.transform(te.text)
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=C.RANDOM_SEED,
        )
        clf.fit(Xtr, tr.label.to_numpy())
        preds[te.index.to_numpy()] = clf.predict(Xte)
        print(f"  fold {fold}: train {len(tr)}, test {len(te)} done", flush=True)
    idx["y_pred"] = preds
    idx = idx.rename(columns={"label": "y_true"})
    assert (idx.y_pred >= 0).all()
    return idx[
        ["article_id", "url", "domain", "fold", "leaning", "y_true", "y_pred"]
    ]


def aggregate_publisher(article: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom, g in article.groupby("domain"):
        counts = g.y_pred.value_counts()
        top = counts[counts == counts.max()].index.min()  # tie -> lowest
        rows.append(
            {
                "domain": dom,
                "y_true": int(g.y_true.iloc[0]),
                "y_pred": int(top),
                "n_articles": len(g),
                "leaning": g.leaning.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def bundle(y_true, y_pred) -> dict:
    return M.metric_bundle(np.asarray(y_true), np.asarray(y_pred))


def main() -> None:
    C.ensure_output_dirs()
    df = load()
    print(f"Loaded {len(df)} articles, {df.domain.nunique()} publishers")

    art = run_cv(df)
    art.to_csv(C.DERIVED_DIR / "tfidf_lr_article_predictions.csv", index=False)

    pub = aggregate_publisher(art)
    pub.to_csv(C.DERIVED_DIR / "tfidf_lr_publisher_predictions.csv", index=False)

    art_b = bundle(art.y_true, art.y_pred)
    pub_b = bundle(pub.y_true, pub.y_pred)
    out = pd.DataFrame([{"level": "article", **art_b}, {"level": "publisher", **pub_b}])
    out.to_csv(C.DERIVED_DIR / "tfidf_lr_metrics.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== TF-IDF + LR ===")
    print(out.round(4).to_string(index=False))
    print("\nArticle-level accuracy:", round(art_b["Accuracy"], 4))
    print("Publisher-level accuracy:", round(pub_b["Accuracy"], 4))


if __name__ == "__main__":
    main()

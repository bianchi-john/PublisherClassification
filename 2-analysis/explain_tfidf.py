"""Why do fine-tuned transformers barely beat a TF-IDF linear model?

Runs the explainability and ablation analyses on the TF-IDF + logistic
regression baseline (run_tfidf_lr_baseline.py), under the same
publisher-disjoint folds, and compares its errors with the transformers'.

  A  interpretation: top n-grams per class (model fit on the full corpus, for
     reading only, never for scoring) and how many publishers carry each term
  B  truncation: the baseline restricted to the words a transformer reads
  C  style vs content: function words only / function words removed
  D  ablation: drop the K most predictive n-grams of each fold, refit
  E  boilerplate: drop paragraphs with newsletter/donation/ad notices, refit
  F  agreement: prediction agreement and error overlap across all five models
  G  label-noise proxy: article errors inside correctly classified publishers
  H  per-class recall, and the l4/l5 boundary

Outputs go to Results/derived/explain/.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from irgood import config as C  # noqa: E402
from irgood import metrics as M  # noqa: E402
import run_tfidf_lr_baseline as B  # noqa: E402

OUT = C.DERIVED_DIR / "explain"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


VEC_KW = dict(sublinear_tf=True, min_df=5, max_df=0.9, ngram_range=(1, 2),
              max_features=100_000, strip_accents="unicode")


def lr() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                              random_state=C.RANDOM_SEED)


def scores(art: pd.DataFrame) -> dict:
    pub = B.aggregate_publisher(art)
    a = M.metric_bundle(art.y_true.to_numpy(), art.y_pred.to_numpy())
    p = M.metric_bundle(pub.y_true.to_numpy(), pub.y_pred.to_numpy())
    return {"art_acc": a["Accuracy"], "art_f1": a["F1_Macro"], "art_tol1": a["Tolerant_1.0"],
            "pub_acc": p["Accuracy"], "pub_f1": p["F1_Macro"], "pub_tol1": p["Tolerant_1.0"]}


def _fold(fold, folds, texts, y, vec_kw, drop_top_k):
    tr = folds != fold
    te = ~tr
    vec = TfidfVectorizer(**vec_kw)
    Xtr = vec.fit_transform(texts[tr])
    Xte = vec.transform(texts[te])
    clf = lr().fit(Xtr, y[tr])
    if drop_top_k:
        # Selected on the TRAINING fold only: no test information is used.
        imp = np.abs(clf.coef_).max(axis=0)
        keep = np.argsort(imp)[:-drop_top_k]
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        clf = lr().fit(Xtr, y[tr])
    return np.flatnonzero(te), clf.predict(Xte)


def cv(df: pd.DataFrame, texts: pd.Series, vec_kw: dict, drop_top_k: int = 0) -> pd.DataFrame:
    """Publisher-disjoint CV with an optional per-fold top-K feature ablation."""
    folds = df.fold.to_numpy()
    y = df.y_true.to_numpy()
    t = texts.to_numpy()
    parts = Parallel(n_jobs=3)(
        delayed(_fold)(f, folds, t, y, vec_kw, drop_top_k) for f in sorted(set(folds)))
    pred = np.full(len(df), -1)
    for idx, p in parts:
        pred[idx] = p
    assert (pred >= 0).all()
    out = df[["article_id", "domain", "fold", "leaning", "y_true"]].copy()
    out["y_pred"] = pred
    return out


def first_words(s: pd.Series, n: int) -> pd.Series:
    return s.str.split().str[:n].str.join(" ")


BOILER = re.compile(
    r"subscri|newsletter|donat|advertis|sign up|click here|read more|follow us|"
    r"all rights reserved|cookie|privacy policy|terms of (use|service)|support (us|our)",
    re.I)


def strip_boilerplate(t: str) -> str:
    return "\n\n".join(p for p in t.split("\n\n") if not BOILER.search(p))


def main() -> None:
    df = B.load().rename(columns={"label": "y_true"})
    texts = df.text
    results = {}

    if os.environ.get("ONLY_FH"):
        return main_fh(df, results)

    # ---------------- A. interpretation ----------------
    log("A: fitting the reading model on the full corpus")
    vec = TfidfVectorizer(**VEC_KW)
    X = vec.fit_transform(texts)
    clf = lr().fit(X, df.y_true)
    terms = np.array(vec.get_feature_names_out())
    Xb = (X > 0).tocsc()
    dom_codes, dom_index = pd.factorize(df.domain)
    rows = []
    for k in range(C.N_CLASSES):
        for j in np.argsort(clf.coef_[k])[::-1][:40]:
            docs = Xb[:, j].nonzero()[0]
            pubs = pd.Series(dom_codes[docs]).value_counts()
            rows.append({
                "class": C.CLASS_LABELS[k], "term": terms[j], "coef": clf.coef_[k, j],
                "n_articles": len(docs), "n_publishers": len(pubs),
                "top_publisher": dom_index[pubs.index[0]],
                "top_publisher_share": pubs.iloc[0] / len(docs),
            })
    top = pd.DataFrame(rows)
    top.to_csv(OUT / "top_terms_per_class.csv", index=False)

    # How concentrated is the predictive mass? For the top-200 terms per class,
    # the share of a term's articles that come from its single largest publisher.
    conc = []
    for k in range(C.N_CLASSES):
        for j in np.argsort(clf.coef_[k])[::-1][:200]:
            docs = Xb[:, j].nonzero()[0]
            pubs = pd.Series(dom_codes[docs]).value_counts()
            conc.append({"class": C.CLASS_LABELS[k], "n_publishers": len(pubs),
                         "top_share": pubs.iloc[0] / len(docs)})
    conc = pd.DataFrame(conc)
    results["A_concentration"] = conc.groupby("class").agg(
        median_publishers=("n_publishers", "median"),
        share_terms_dominated=("top_share", lambda s: float((s > 0.5).mean())),
    ).round(3).to_dict()
    log("A done")

    # ---------------- B. truncation ----------------
    for n in (190, 750):
        log(f"B: first {n} words")
        art = cv(df, first_words(texts, n), VEC_KW)
        results[f"B_first_{n}_words"] = scores(art)
        art.to_csv(OUT / f"pred_first_{n}_words.csv", index=False)

    # ---------------- C. style vs content ----------------
    log("C: function words only")
    fw = dict(VEC_KW, ngram_range=(1, 1), vocabulary=sorted(ENGLISH_STOP_WORDS),
              min_df=1, max_df=1.0, max_features=None)
    results["C_function_words_only"] = scores(cv(df, texts, fw))
    log("C: function words removed")
    cw = dict(VEC_KW, stop_words="english")
    results["C_function_words_removed"] = scores(cv(df, texts, cw))

    # ---------------- D. top-K ablation ----------------
    for k in (100, 1000, 10000):
        log(f"D: dropping the top {k} n-grams per fold")
        results[f"D_drop_top_{k}"] = scores(cv(df, texts, VEC_KW, drop_top_k=k))

    # ---------------- E. boilerplate ----------------
    log("E: boilerplate paragraphs removed")
    clean = texts.map(strip_boilerplate)
    results["E_boilerplate_share_articles"] = float((clean != texts).mean())
    results["E_boilerplate_removed"] = scores(cv(df, clean, VEC_KW))

    main_fh(df, results)


def main_fh(df, results):
    # Reference: the full baseline, recomputed from its saved predictions.
    base = pd.read_csv(C.DERIVED_DIR / "tfidf_lr_article_predictions.csv")
    results["REF_full_text"] = scores(base)

    # ---------------- F, G, H. agreement, noise, classes ----------------
    log("F-H: agreement and error structure")
    pl = pd.read_csv(C.DERIVED_DIR / "predictions_long.csv")
    wide = pl.pivot(index="article_id", columns="model", values="y_pred")
    wide["TF-IDF"] = base.set_index("article_id").y_pred
    y = pl.drop_duplicates("article_id").set_index("article_id").y_true.loc[wide.index]
    dom = pl.drop_duplicates("article_id").set_index("article_id").domain.loc[wide.index]
    models = C.MODEL_ORDER + ["TF-IDF"]

    kappa = {f"{a}|{b}": cohen_kappa_score(wide[a], wide[b])
             for a, b in combinations(models, 2)}
    results["F_prediction_kappa"] = {k: round(v, 3) for k, v in kappa.items()}

    correct = wide[models].eq(y, axis=0)
    results["F_all5_wrong"] = float((~correct).all(axis=1).mean())
    results["F_all5_right"] = float(correct.all(axis=1).mean())
    results["F_oracle_any_right"] = float(correct.any(axis=1).mean())
    for m in C.MODEL_ORDER:
        both = (~correct[m] & ~correct["TF-IDF"]).sum()
        results.setdefault("F_tfidf_errors_shared_with", {})[m] = round(
            float(both / (~correct["TF-IDF"]).sum()), 3)
    # Majority vote across models, ties -> lowest class.
    def vote(cols):
        v = wide[cols].apply(lambda r: r.value_counts().pipe(
            lambda c: c[c == c.max()].index.min()), axis=1)
        return v
    for name, cols in (("4_transformers", C.MODEL_ORDER), ("all_5", models)):
        v = vote(cols)
        art = pd.DataFrame({"article_id": wide.index, "domain": dom.values,
                            "y_true": y.values, "y_pred": v.values, "leaning": None})
        results[f"F_ensemble_{name}"] = scores(art)

    # G. label-noise proxy: how many article errors sit inside publishers that
    # the same model classifies correctly once aggregated?
    g = {}
    for m in models:
        art = pd.DataFrame({"domain": dom.values, "y_true": y.values,
                            "y_pred": wide[m].values, "leaning": None})
        pub = B.aggregate_publisher(art).set_index("domain")
        ok_pub = art.domain.map(pub.y_pred.eq(pub.y_true))
        err = art.y_pred != art.y_true
        g[m] = {"article_errors_in_correct_publishers": round(float((err & ok_pub).sum() / err.sum()), 3)}
    # Articles every model gets wrong while every model gets their publisher right.
    pub_ok_all = pd.Series(True, index=dom.unique())
    for m in models:
        art = pd.DataFrame({"domain": dom.values, "y_true": y.values,
                            "y_pred": wide[m].values, "leaning": None})
        pub = B.aggregate_publisher(art).set_index("domain")
        pub_ok_all &= pub.y_pred.eq(pub.y_true).reindex(pub_ok_all.index)
    atyp = (~correct).all(axis=1) & dom.map(pub_ok_all).values
    g["atypical_articles_share"] = round(float(atyp.mean()), 3)
    g["publishers_right_for_all_5"] = int(pub_ok_all.sum())
    results["G_noise"] = g

    # H. per-class recall and the l4/l5 boundary.
    rec = {m: dict(zip(C.CLASS_LABELS, recall_score(y, wide[m], labels=C.CLASS_CODES,
                                                    average=None).round(3)))
           for m in models}
    results["H_recall_per_class"] = rec
    l45 = {}
    for m in models:
        err = wide[m] != y
        pair = err & y.isin([3, 4]) & wide[m].isin([3, 4])
        l45[m] = round(float(pair.sum() / err.sum()), 3)
    results["H_share_of_errors_l4_l5"] = l45

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log("ALL DONE")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()

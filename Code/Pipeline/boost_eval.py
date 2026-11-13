"""Step 10, evaluation (used by 10_boost.ipynb), from the saved outlet predictions only.

Sources compared, all on the same 226 outlets and the same random draws of k articles:
- TF-IDF: the ridge of step 7, unchanged (mean of its article predictions on the same draws);
- every (representation, method) of boost.py;
- "Qwen+Mistral avg · <method>": mean of the Qwen3-8B and Mistral-24B predictions of the same method;
- "avg methods · <representation>": mean of the predictions of all the methods of a representation (no choice);
- "auto · <representation>": in every outer fold, the method with the best inner Spearman (chosen without the test
  outlets); "auto · all": the same over every representation and method. These are the honest summary numbers: picking
  the best row of the table by its test result would be optimistic.

Measures per outlet, as in the deck: Spearman, Kendall, MAE, % within 20 points; on the predicted score ("normal") and
on the stretched score (training quantiles, per fold). For k < all: mean over the 20 draws.
Differences with TF-IDF and with the ridge of step 7 (same representation): paired bootstrap over outlets, 1,000
times (seed 7), the same resamples and draws for both; "significant" = 95% interval entirely on the better side."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import boost as BO
import metrics as ME

MEASURES, LOWER = ME.MEASURES, ME.LOWER       # the measures (in metrics.py, shared with step 15)
B = 1000


def inner_choices():
    """Per (representation, method, fold): the best inner Spearman (the value used by auto)."""
    rows = []
    for f in BO.PRED.glob("*/*/inner_fold*.csv"):
        d = pd.read_csv(f)
        rows.append({"embedding": f.parent.parent.name, "method": f.parent.name, "fold": int(d.fold.iloc[0]),
                     "inner Spearman": d["inner Spearman"].max(),
                     "chosen": d.loc[d.chosen].drop(columns=["inner Spearman", "fold", "chosen"]).dropna(axis=1).iloc[0].to_dict()})
    return pd.DataFrame(rows)


def sources(outlets, art, rows):
    """Every source as a long table: source, domain, k, draw, prediction (only complete sources: 5 folds)."""
    P = BO.collect()
    n = P.groupby(["embedding", "method"]).domain.nunique()
    complete = n[n == len(outlets)].index
    P = P.set_index(["embedding", "method"]).loc[list(complete)].reset_index()
    P["source"] = P.embedding + " · " + P.method
    parts = [P[["source", "domain", "k", "draw", "prediction"]],
             BO.tfidf_draws(art, outlets, rows).assign(source="TF-IDF")]
    q, m = P[P.embedding == "Qwen3-8B"], P[P.embedding == "Mistral-24B"]
    both = q.merge(m, on=["method", "domain", "k", "draw"], suffixes=("_q", "_m"))
    both["prediction"] = (both.prediction_q + both.prediction_m) / 2
    parts.append(both.assign(source="Qwen+Mistral avg · " + both.method)[["source", "domain", "k", "draw", "prediction"]])
    for e in P.embedding.unique():                        # the mean of all the methods of a representation
        a = P[P.embedding == e].groupby(["domain", "k", "draw"]).prediction.mean().reset_index()
        parts.append(a.assign(source=f"avg methods · {e}"))
    inner = inner_choices()
    inner = inner.set_index(["embedding", "method"]).loc[[c for c in complete if c in inner.set_index(["embedding", "method"]).index]].reset_index()
    fold = outlets.fold
    for name, group in [*[(f"auto · {e}", inner[inner.embedding == e]) for e in inner.embedding.unique()], ("auto · all", inner)]:
        best = group.loc[group.groupby("fold")["inner Spearman"].idxmax()]
        if len(best) < 5:
            continue
        pick = P.merge(best[["embedding", "method", "fold"]], on=["embedding", "method"])
        pick = pick[pick.domain.map(fold) == pick.fold]
        parts.append(pick.assign(source=name)[["source", "domain", "k", "draw", "prediction"]])
    return pd.concat(parts, ignore_index=True)




def _one(M, src, k, outlets, idx):
    """M: matrix draws × outlets of the predictions of a source at k."""
    y, fold = outlets.score.to_numpy(), outlets.fold.to_numpy()
    out = {}
    for draw in M:
        for score, p in [("normal", draw), ("stretched", ME.remapped_score(draw, fold, y))]:
            for mes, v in ME.resampled_measures(p, y, idx).items():
                out.setdefault((score, mes), []).append(v)
    return src, k, {key: np.mean(v, axis=0) for key, v in out.items()}


def evaluate(S, outlets, n_jobs=32):
    """Values and paired differences (with intervals) for every source, k, score and measure."""
    idx = np.vstack([np.arange(len(outlets))[None, :], np.random.default_rng(7).integers(0, len(outlets), (B, len(outlets)))])
    jobs = [(s, k, g.pivot_table(index="draw", columns="domain", values="prediction").reindex(columns=outlets.index).to_numpy())
            for (s, k), g in S.groupby(["source", "k"])]
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(M, s, k, outlets, idx) for s, k, M in jobs)
    V = {(s, k, *key): v for s, k, d in res for key, v in d.items()}
    rows = []
    for (s, k, score, mes), v in V.items():
        r = {"source": s, "k": k, "score": score, "measure": mes, "value": v[0]}
        refs = {"TF-IDF": ("TF-IDF", k, score, mes)}
        emb = s.split(" · ")[0]
        if " · " in s and not s.startswith(("auto", "Qwen+Mistral avg", "avg methods")):
            refs["ridge"] = (f"{emb} · ridge", k, score, mes)
        elif s.startswith(("auto", "Qwen+Mistral avg", "avg methods")):
            refs["ridge"] = ("Mistral-24B · ridge", k, score, mes)
        for name, ref in refs.items():
            if ref in V and ref[0] != s:
                d = v - V[ref]
                lo, hi = np.percentile(d[1:], [2.5, 97.5])
                r.update({f"vs {name}": d[0], f"vs {name} low": lo, f"vs {name} high": hi,
                          f"beats {name}": ME.better(lo, hi, mes)})
        rows.append(r)
    return pd.DataFrame(rows)

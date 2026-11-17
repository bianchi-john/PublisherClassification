"""Step 17: four checks asked by the reviews of the paper (CPU, except the name-only probe on the GPU).

1. Recalibration. The predictions are compressed towards the centre, and the MAE pays for it. Per outer fold, a straight
   line (score ~ a + b * predicted outlet score) is fitted on the training outlets only, with predictions that do not
   see them: three inner folds of the training outlets (as in step 7), the penalty and layer chosen in step 7. The line
   is then applied to the outer predictions of step 7. For the zero-shot model nothing is trained, so the line is
   fitted on its answers for the training outlets directly.
2. Another population. The MAE per NewsGuard level, re-weighted to the levels of the political outlets that could not
   be collected (step 2): the error to expect on outlets like the unrated ones, rather than like the 226.
3. Leftover names. The comparison with TF-IDF without the outlets whose brand name (NewsGuard "Brand Name", two words
   or more) is still in the anonymous text of at least one of their articles.
4. Name only. The frozen vector of the outlet's domain name alone (no article), then a ridge over the 226 outlets with
   the same folds: how much of the score the frozen models represent from the name of an outlet, without reading it.

Outputs in robustness/ in the data folder; the numbers of the paper in paper_robustness.csv (step 17 notebook).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cleaning as CL
import common as C
import models as MO

DIR = C.OUT / "robustness"
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B"]


# ------------------------------------------------------------------------- 1. recalibration
def chosen(name, k):
    """The penalty (and layer) chosen on the inner folds of outer fold k in step 7."""
    t = pd.read_csv(C.PREDICTIONS / "regression" / name / f"penalty_fold{k}.csv")
    row = t[t.chosen].iloc[0]
    return row.penalty, (int(row.layer) if "layer" in t else 0)


def inner_predictions(name, art, k, X):
    """Out-of-fold predictions for the training articles of outer fold k, from three inner folds of the training outlets
    (split by score as in step 7) and the setting chosen in step 7."""
    fold = art.fold.to_numpy()
    tr = np.where(fold != k)[0]
    domains = art.domain.to_numpy()
    inner = MO.score_split(art.score.iloc[tr].groupby(domains[tr]).first(), C.SEED + k, k=3)
    fi = pd.Series(domains[tr]).map(inner).to_numpy()
    penalty, layer = chosen(name, k)
    texts = art.text.fillna("").to_numpy() if name == "TF-IDF" else None
    Xk = X[layer] if isinstance(X, dict) else X
    out = pd.Series(np.nan, index=art.article_id.to_numpy()[tr])
    for j in (1, 2, 3):
        a, b = tr[fi != j], tr[fi == j]
        A, B, _ = MO._matrices(name, a, b, texts, Xk)
        out.iloc[np.where(fi == j)[0]] = MO._fit("regression", penalty, A, art.score.to_numpy()[a]).predict(B)
    return out


def recalibrated(name, art, outlets, logfile=None):
    """Outlet scores of step 7 re-mapped by a line fitted inside the training folds (cached in robustness/)."""
    f = DIR / f"recalibrated_{name}.csv"
    if f.exists():
        return pd.read_csv(f, index_col="domain").prediction.reindex(outlets.index)
    DIR.mkdir(parents=True, exist_ok=True)
    outer = MO.predictions("regression", name).groupby("domain").prediction.mean().reindex(outlets.index)
    X = None if name in ("TF-IDF", "LLM-zero-shot") else MO.load_embeddings(name, art)
    domain_of = art.set_index("article_id").domain
    res = pd.Series(np.nan, index=outlets.index)
    for k in range(1, 6):
        train = outlets.index[outlets.fold != k]
        if name == "LLM-zero-shot":                     # nothing is trained: its answers for the training outlets
            p_train = outer[train]
        else:
            p = inner_predictions(name, art, k, X)
            p_train = p.groupby(domain_of.reindex(p.index).to_numpy()).mean().reindex(train)
        b, a = np.polyfit(p_train.to_numpy(), outlets.score[train].to_numpy(), 1)
        test = outlets.index[outlets.fold == k]
        res[test] = a + b * outer[test]
        C.log(f"recalibration, {name}, fold {k}: score = {a:.1f} + {b:.2f} x prediction", logfile)
    res.rename("prediction").rename_axis("domain").to_csv(f)
    return res


# ------------------------------------------------------------------------- 2. another population
def level_weights(selection):
    """Share of each NewsGuard level among the outlets that could not be collected and among all the political outlets
    (from paper_selection.csv of step 15)."""
    levels = [f"% {lv}" for lv in C.LEVELS]
    s = selection.set_index("last step")
    never = s.loc["sent to the scraper", levels].to_numpy(float) / 100
    everyone = (s[levels].mul(s.outlets, axis=0).sum() / s.outlets.sum()).to_numpy(float) / 100
    return {"not collected": never, "all political outlets": everyone}


def reweighted_mae(pred, outlets, weights, idx):
    """MAE per level, re-weighted; on every row of idx (row 0 = the real outlets, then bootstrap resamples)."""
    level = np.array(C.level_from_score(outlets.score.to_numpy()))
    err = np.abs(pred - outlets.score.to_numpy())
    out = []
    for row in idx:
        e = np.array([err[row][level[row] == i].mean() for i in range(len(C.LEVELS))])
        out.append(np.nansum(e * weights) / weights[~np.isnan(e)].sum())
    return np.array(out)


# ------------------------------------------------------------------------- 3. leftover names
def outlets_with_brand(art, min_words=2):
    """Outlets whose NewsGuard brand name (at least two words) is still in the anonymous text of one of their articles."""
    ng = CL.read_newsguard(C.NEWSGUARD)
    brand = art.domain.map(ng["Brand Name"]).fillna("")
    text = art.text.fillna("")
    hit = [len(b.split()) >= min_words and b in t for b, t in zip(brand, text)]
    return sorted(set(art.domain[np.array(hit)]))


# ------------------------------------------------------------------------- 4. name only
def name_vectors(outlets, logfile=None, gpu_gb=19):
    """Frozen vectors of the domain name alone (e.g. "example.com"), with the settings of step 5; cached."""
    f = {m: DIR / f"names_{m}.npy" for m in ["Qwen3-8B", "Mistral-24B"]}
    names = list(outlets.index)
    if not f["Qwen3-8B"].exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        np.save(f["Qwen3-8B"], MO.qwen_embeddings(names, "Qwen/Qwen3-Embedding-8B", 16384, logfile))
    if not f["Mistral-24B"].exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        E = MO.mistral_embeddings(names, "mistralai/Mistral-Small-24B-Base-2501", [27], 8192, logfile,
                                  DIR / "names_mistral_partial.npy")
        np.save(f["Mistral-24B"], E[27])
    return {m: np.load(x) for m, x in f.items()}


def name_only(X, outlets):
    """Ridge over the outlets, the folds of the paper; penalty chosen on three inner folds of the training outlets by
    Spearman, as in step 7. Returns the predicted score of every outlet (out of fold)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    y, fold = outlets.score.to_numpy(), outlets.fold.to_numpy()
    pred = np.full(len(y), np.nan)
    for k in range(1, 6):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        inner = MO.score_split(outlets.score.iloc[tr], C.SEED + k, k=3).to_numpy()
        best, best_score = None, -np.inf
        for alpha in MO.GRIDS[("regression", "embedding")]:
            scores = []
            for j in (1, 2, 3):
                a, b = tr[inner != j], tr[inner == j]
                s = StandardScaler().fit(X[a])
                p = Ridge(alpha=alpha).fit(s.transform(X[a]), y[a]).predict(s.transform(X[b]))
                scores.append(spearmanr(p, y[b])[0])
            if np.mean(scores) > best_score:
                best, best_score = alpha, np.mean(scores)
        s = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=best).fit(s.transform(X[tr]), y[tr]).predict(s.transform(X[te]))
    return pred

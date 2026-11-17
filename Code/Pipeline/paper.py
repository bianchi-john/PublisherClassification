"""Step 15: every number of the paper, from the saved predictions only (no model is trained here).

- Main table (all articles): Spearman, Kendall, MAE, % within 20 points per outlet, for every regressor of steps 7
  and 13 and three baselines without a model of the text: the training mean, the training median (the best constant
  for the MAE) and the training mean of the outlet's political orientation. 95% percentile intervals, B = 10,000
  bootstrap resamples of the outlets. For the appendix: R², RMSE and AUC of reliable (>= 60) against the others.
- Curves: Spearman and MAE from k = 1, 2, 3, 5, 10, 20, 50 random articles per outlet and from all (the draws of step
  10, the same for every model; measure computed on every draw, then averaged), B = 2,000.
- Comparisons: paired bootstrap on the same resampled outlets; two-sided bootstrap p-value; Holm correction inside each
  family (main table against TF-IDF; curves against TF-IDF, per model; the claims of the text).
- Where the models err: signed error and MAE per NewsGuard level and per political orientation (also at equal score,
  and the correlation inside each orientation), and every measure per article against per outlet.
- Robustness: political orientation (the gap at equal score, the gap that the compression alone would produce,
  calibration by group), who is left out by the cleaning, the leakage ladder of step 14 (with the similarity of test
  and training articles, and how much the NewsGuard score changes inside an ownership group).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

import boost as BO
import cleaning as CL
import common as C
import metrics as ME
import models as MO

MODELS = ["TF-IDF", "Qwen3-0.6B", "Qwen3-8B", "Mistral-24B", "ModernBERT-frozen", "ModernBERT", "LLM-zero-shot"]
BASELINES = ["no text (mean)", "no text (median)", "orientation only"]
ORIENTATIONS = ["Left", "Right", "none"]
K = [str(k) for k in BO.K]
SEED = 7
MEASURES = ME.MEASURES

load = BO.load                       # articles, outlets (score, fold, orientation) and the draws of step 10


def available(models=MODELS):
    """The models whose five folds of regression predictions are saved."""
    return [m for m in models if len(list((C.PREDICTIONS / "regression" / m).glob("fold*.csv.gz"))) == 5]


def article_vector(name, art):
    """The prediction of every article (rows of `art`), NaN where the model gave none (zero-shot)."""
    p = MO.predictions("regression", name).set_index("article_id").prediction
    return p.reindex(art.article_id).to_numpy(dtype=float)


def draw_index(outlets, rows):
    """{k: list of (outlet position, draw, article rows)} from the draws of step 10 (the same for every model)."""
    pos = {d: i for i, d in enumerate(outlets.index)}
    out = {k: [] for k in K}
    for f in range(1, 6):
        for d, k, j, r in BO.test_draws(outlets, rows, f):
            out[k].append((pos[d], j, r))
    return out


def matrix(p, draws, k, n):
    """Draws × outlets: the mean prediction of the outlet's articles of each draw (NaN predictions ignored)."""
    nd = 1 if k == "all" else BO.DRAWS
    M = np.full((nd, n), np.nan)
    for i, j, r in draws[k]:
        v = p[r]
        M[j, i] = np.nanmean(v) if np.isfinite(v).any() else np.nan
    return M


def baselines(outlets):
    """The predictions of the three baselines (one per outlet; the statistics come from the training outlets)."""
    s, fold, o = outlets.score, outlets.fold, outlets.orientation
    out = {b: pd.Series(np.nan, index=outlets.index) for b in BASELINES}
    for f in range(1, 6):
        tr, te = fold != f, fold == f
        out["no text (mean)"][te] = s[tr].mean()
        out["no text (median)"][te] = s[tr].median()
        out["orientation only"][te] = o[te].map(s[tr].groupby(o[tr]).mean())
    return {b: v.to_numpy() for b, v in out.items()}


def resamples(n, B, seed=SEED):
    """Row 0 = the real outlets, rows 1..B = bootstrap resamples (the same for every model: paired comparisons)."""
    return np.vstack([np.arange(n)[None, :], np.random.default_rng(seed).integers(0, n, (B, n))])


def over_draws(M, y, idx, which):
    """A measure computed on every draw of M, then averaged over the draws."""
    acc = {}
    for draw in M:
        for m, v in ME.resampled_measures(draw, y, idx, which).items():
            acc.setdefault(m, []).append(v)
    return {m: np.mean(v, axis=0) for m, v in acc.items()}


def compare(a, b, measure):
    """a - b with its 95% interval and two-sided bootstrap p-value."""
    d = a - b
    if np.isnan(d).any():                          # a measure that cannot be computed: no comparison
        return {"diff": np.nan, "dlo": np.nan, "dhi": np.nan, "p": np.nan}
    lo, hi = np.percentile(d[1:], [2.5, 97.5])
    p = min(1.0, 2 * min((d[1:] <= 0).mean(), (d[1:] >= 0).mean()))
    return {"diff": d[0], "dlo": lo, "dhi": hi, "p": p}


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def with_holm(T, flag, by=None):
    """Holm-adjusted p-values (inside each group of rows `by`, or all together) and `flag` = significantly better after
    the correction (NaN where there is no comparison)."""
    for c in ["diff", "dlo", "dhi", "p"]:
        if c not in T:
            T[c] = np.nan
    T["p_holm"] = np.nan
    for _, g in (T.groupby(by) if by else [(None, T)]):
        g = g[g.p.notna()]
        if len(g):
            T.loc[g.index, "p_holm"] = holm(g.p)
    T[flag] = [ME.better(lo, hi, m) and q < 0.05 if q == q else np.nan
               for lo, hi, m, q in zip(T.dlo, T.dhi, T.measure, T.p_holm)]
    return T


# ------------------------------------------------------------------------- main table and curves
def main_table(art, outlets, rows, B=10_000, models=None):
    models = models or available()
    y, n = outlets.score.to_numpy(), len(outlets)
    idx, draws = resamples(n, B), draw_index(outlets, rows)
    V = {m: ME.resampled_measures(matrix(article_vector(m, art), draws, "all", n)[0], y, idx) for m in models}
    V.update({b: ME.resampled_measures(p, y, idx) for b, p in baselines(outlets).items()})
    out = []
    for src, res in V.items():
        for mes, v in res.items():
            lo, hi = np.percentile(v[1:], [2.5, 97.5])
            r = {"model": src, "measure": mes, "value": v[0], "low": lo, "high": hi}
            if src != "TF-IDF" and src in models:
                r.update(compare(v, V["TF-IDF"][mes], mes))
            out.append(r)
    return with_holm(pd.DataFrame(out), "sig"), V



def more_measures(art, outlets, rows, B=10_000, models=None):
    """For the appendix: R², RMSE and AUC of reliable (score >= 60) against the others (metrics.more_measures), for the
    models and the baselines, with 95% intervals over B resamples of the outlets (the same as the main table)."""
    models = models or available()
    y, n = outlets.score.to_numpy(), len(outlets)
    idx, draws = resamples(n, B), draw_index(outlets, rows)
    P = {m: matrix(article_vector(m, art), draws, "all", n)[0] for m in models} | baselines(outlets)
    out = []
    for src, p in P.items():
        for mes, v in ME.more_measures(p, y, idx).items():
            out.append({"model": src, "measure": mes, "value": v[0], "low": np.nanpercentile(v[1:], 2.5),
                        "high": np.nanpercentile(v[1:], 97.5)})
    return pd.DataFrame(out)


def curves(art, outlets, rows, B=2_000, models=None):
    models = models or available()
    y, n = outlets.score.to_numpy(), len(outlets)
    idx, draws = resamples(n, B), draw_index(outlets, rows)
    which = ["Spearman", "MAE"]
    V = {(m, k): over_draws(matrix(article_vector(m, art), draws, k, n), y, idx, which) for m in models for k in K}
    out = []
    for (m, k), res in V.items():
        for mes, v in res.items():
            r = {"model": m, "k": k, "measure": mes, "value": v[0]}
            if m != "TF-IDF":
                r.update(compare(v, V[("TF-IDF", k)][mes], mes))
            out.append(r)
    return with_holm(pd.DataFrame(out), "beats_tfidf", by="model"), V      # Holm per model, over k and the measures


def claims(V_main, V_curves, models):
    """The comparisons the text makes, in one Holm family."""
    pairs = [("Mistral-24B", "5", "TF-IDF", "all", "Mistral with 5 articles vs TF-IDF with all", ["Spearman", "MAE"]),
             ("Mistral-24B", "all", "Qwen3-8B", "all", "Mistral vs Qwen3-8B", ["Spearman", "MAE"]),
             ("Qwen3-8B", "all", "Qwen3-0.6B", "all", "Qwen3-8B vs Qwen3-0.6B (size)", ["Spearman", "MAE"]),
             ("ModernBERT-frozen", "all", "ModernBERT", "all", "ModernBERT frozen vs trained", ["Spearman", "MAE"]),
             ("Mistral-24B", "all", "LLM-zero-shot", "all", "Mistral frozen + ridge vs zero-shot", ["Spearman", "MAE"])]
    pairs += [(m, "all", m, "1", f"{m}: all articles vs one (aggregation)", ["Spearman", "MAE"]) for m in models]
    out = []
    for a, ka, b, kb, label, mes_list in pairs:
        if (a, ka) not in V_curves or (b, kb) not in V_curves:
            continue
        first, second = V_curves[(a, ka)], V_curves[(b, kb)]          # "diff" = first − second
        for mes in mes_list:
            out.append({"claim": label, "measure": mes, "first": first[mes][0], "second": second[mes][0],
                        **compare(first[mes], second[mes], mes)})
    return with_holm(pd.DataFrame(out), "first better")


# ------------------------------------------------------------------------- where the models err
def article_vs_outlet(art, outlets, rows, models=None):
    """Per model: the measures per article (each article against the score of its outlet) and per outlet (the mean of
    its articles), with the lowest and highest prediction (how much the predictions are compressed)."""
    models = models or available()
    y, n, draws = outlets.score.to_numpy(), len(outlets), draw_index(outlets, rows)
    ya = art.domain.map(outlets.score).to_numpy(dtype=float)
    out = []
    for m in models:
        pa = article_vector(m, art)
        ok = np.isfinite(pa)
        for unit, p, t in [("article", pa[ok], ya[ok]), ("outlet", matrix(pa, draws, "all", n)[0], y)]:
            v = ME.resampled_measures(p, t, np.arange(len(t))[None, :])
            out.append({"model": m, "unit": unit, "n": len(t), **{k: x[0] for k, x in v.items()},
                        "lowest prediction": p.min(), "highest prediction": p.max()})
    return pd.DataFrame(out)


def errors(art, outlets, rows, models=None, B=2_000):
    """Where the models err, per outlet (signed error = predicted − true score): mean signed error and MAE per NewsGuard
    level and per political orientation; per orientation also the signed error at equal true score (residuals of a
    linear regression of the signed error on the true score, fitted on all the outlets) and the Spearman correlation
    inside the group; on all the outlets the slope of the signed error on the true score and the Kruskal-Wallis p-value
    of the differences between orientations (raw and at equal score). 95% intervals: the same B bootstrap resamples of
    the outlets for every model; p-values and group sizes on the real outlets only."""
    models = models or available()
    y, o, n = outlets.score.to_numpy(), outlets.orientation.to_numpy(), len(outlets)
    level = np.array(C.LEVELS)[C.level_from_score(y)]
    groups = [("level", g, level == g) for g in C.LEVELS] + [("orientation", g, o == g) for g in ORIENTATIONS]
    draws, idx = draw_index(outlets, rows), resamples(n, B)

    def one(s, p, mask_of):
        """Every quantity on one set of outlets (s true scores, p predictions); mask_of(group mask) -> rows."""
        e = p - s
        X = np.c_[np.ones_like(s), s]
        beta = np.linalg.lstsq(X, e, rcond=None)[0]
        res = e - X @ beta
        v = {("all", "all", "slope of the signed error on the score"): beta[1]}
        for by, g, mask in groups:
            k = mask_of(mask)
            v[(by, g, "signed error")] = e[k].mean() if k.any() else np.nan
            v[(by, g, "MAE")] = np.abs(e[k]).mean() if k.any() else np.nan
            if by == "orientation":
                v[(by, g, "signed error at equal score")] = res[k].mean()
                v[(by, g, "Spearman inside the group")] = spearmanr(s[k], p[k]).correlation
        return v, e, res

    out = []
    for m in models:
        p = matrix(article_vector(m, art), draws, "all", n)[0]
        vals = {}
        for r in idx:
            for key, x in one(y[r], p[r], lambda mask: mask[r])[0].items():
                vals.setdefault(key, []).append(x)
        for (by, g, q), v in vals.items():
            v = np.array(v, dtype=float)
            out.append({"model": m, "by": by, "group": g, "quantity": q, "value": v[0],
                        "low": np.nanpercentile(v[1:], 2.5), "high": np.nanpercentile(v[1:], 97.5)})
        _, e, res = one(y, p, lambda mask: mask)
        for by, g, mask in groups:
            out.append({"model": m, "by": by, "group": g, "quantity": "outlets", "value": mask.sum()})
        for q, x in [("Kruskal-Wallis p, raw", e), ("Kruskal-Wallis p, at equal score", res)]:
            out.append({"model": m, "by": "orientation", "group": "all", "quantity": q,
                        "value": kruskal(*[x[o == g] for g in ORIENTATIONS]).pvalue})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- robustness
def _residual_gap(err, s, o, how):
    """Left − right mean of the signed error at equal true score: residuals of err ~ f(s)."""
    if how == "linear":
        X = np.c_[np.ones_like(s), s]
    elif how == "spline":
        knots = np.percentile(s, [25, 50, 75])
        X = np.c_[np.ones_like(s), s, *[np.clip(s - k, 0, None) ** 3 for k in knots], s ** 2, s ** 3]
    else:                                                    # NewsGuard level fixed effects
        X = pd.get_dummies(C.level_from_score(s)).to_numpy(float)
    res = err - X @ np.linalg.lstsq(X, err, rcond=None)[0]
    return res[o == "Left"].mean() - res[o == "Right"].mean()


def orientation(art, outlets, rows, models=None, B=2_000):
    """Per model: the left − right gap of the signed error at equal NewsGuard score (linear, spline, level fixed
    effects), the gap that a predictor with the same compression and nothing but the orientation means would show
    (λ s + (1 − λ) mean of the orientation, λ = the model's slope), and calibration by group (left − right true score
    at equal prediction: 0 = the same prediction means the same score for both)."""
    models = models or available()
    y, o, n = outlets.score.to_numpy(), outlets.orientation.to_numpy(), len(outlets)
    draws = draw_index(outlets, rows)
    P = {m: matrix(article_vector(m, art), draws, "all", n)[0] for m in models}
    P["orientation only"] = baselines(outlets)["orientation only"]
    idx = resamples(n, B)
    out = []
    for m, p in P.items():
        vals = {k: [] for k in ["gap linear", "gap spline", "gap levels", "gap from compression", "calibration gap"]}
        for r in idx:
            s, pr, oo = y[r], p[r], o[r]
            err = pr - s
            vals["gap linear"].append(_residual_gap(err, s, oo, "linear"))
            vals["gap spline"].append(_residual_gap(err, s, oo, "spline"))
            vals["gap levels"].append(_residual_gap(err, s, oo, "levels"))
            lam = np.polyfit(s, pr, 1)[0]
            mean_o = pd.Series(s).groupby(oo).transform("mean").to_numpy()
            shrunk = lam * s + (1 - lam) * mean_o
            vals["gap from compression"].append(_residual_gap(shrunk - s, s, oo, "linear"))
            if m == "orientation only":                     # its prediction IS the orientation: not defined
                vals["calibration gap"].append(np.nan); continue
            X = np.c_[np.ones_like(pr), pr, oo == "Left", oo == "none"]
            beta = np.linalg.lstsq(X.astype(float), s, rcond=None)[0]
            vals["calibration gap"].append(beta[2])          # true score of left minus right at equal prediction
        for k, v in vals.items():
            v = np.array(v, dtype=float)
            if np.isnan(v).all():
                continue
            out.append({"model": m, "quantity": k, "value": v[0], "low": np.percentile(v[1:], 2.5),
                        "high": np.percentile(v[1:], 97.5)})
    return pd.DataFrame(out)


def selection(outlets_by_step):
    """Who is left out: score, level and orientation of the outlets at every step of the cleaning."""
    o = outlets_by_step.copy()
    assert o.score.notna().all(), "an outlet without a NewsGuard score"
    o["level"] = np.array(C.LEVELS)[C.level_from_score(o.score)]
    rows = []
    for st in [s for s in CL.STEPS if s in set(o["last step"])]:
        g = o[o["last step"] == st]
        rows.append({"last step": st, "outlets": len(g), "mean score": g.score.mean(),
                     **{f"% {lv}": 100 * (g.level == lv).mean() for lv in C.LEVELS},
                     **{f"% {x}": 100 * (g.orientation == x).mean() for x in ORIENTATIONS},
                     "% US": 100 * (g.country == "US").mean()})
    T = pd.DataFrame(rows)
    paper = o["last step"] == CL.STEPS[-1]
    test = {"score (Mann-Whitney p)": mannwhitneyu(o.score[paper].dropna(), o.score[~paper].dropna()).pvalue,
            "orientation (chi-square p)": chi2_contingency(pd.crosstab(paper, o.orientation))[1]}
    return T, test


def ladder_results(art, outlets, B=10_000):
    """Per rung and model: Spearman and MAE on the paper's outlets and test articles, and the difference with the rung
    of reference (ladder.REFERENCE: the same text, one protection more). The sisters rung and its reference also on the
    outlets with a sister and on those without."""
    import ladder as LA
    articles, y = set(art.article_id), outlets.score.to_numpy()
    has = outlets.index.isin(LA.sisters(LA.stages()[0])[1])
    P = {(r, m): LA.outlet_predictions(r, m, outlets, articles) for r in LA.RUNGS for m in LA.MODELS}
    out = []
    for group, mask in [("all", np.full(len(y), True)), ("with a sister", has), ("without a sister", ~has)]:
        ii = np.where(mask)[0]
        idx = ii[resamples(len(ii), B)]
        rungs = LA.RUNGS if group == "all" else ["sisters", "L2"]
        V = {(r, m): ME.resampled_measures(P[r, m].to_numpy(), y, idx, ["Spearman", "MAE"])
             for r in rungs for m in LA.MODELS if P[r, m] is not None}
        for (r, m), res in V.items():
            ref = LA.REFERENCE.get(r)
            for mes, v in res.items():
                row = {"outlets": group, "n": len(ii), "rung": r, "model": m, "measure": mes, "value": v[0],
                       "low": np.percentile(v[1:], 2.5), "high": np.percentile(v[1:], 97.5)}
                if (ref, m) in V:
                    row.update(compare(v, V[ref, m][mes], mes))
                out.append(row)
    return pd.DataFrame(out)


def sister_similarity(art, outlets, B=10_000):
    """Being a sister is only a hint: what counts is how similar the texts are.
    1. Per group of outlets (with a sister, without): how close the test articles get to the training ones without the
       sisters (L2) and with them (mean over the outlets of the mean highest cosine, ladder.similarity), % of articles
       whose closest training article is a sister's, % with a near copy among the sisters' articles.
    2. Per model: Spearman, over the outlets, between how much closer the training gets with the sisters and how much
       the absolute error of the outlet drops (L2 - sisters); on all the outlets and on those with a sister; 95%
       interval over B resamples of the outlets."""
    import ladder as LA
    A = LA.stages()[0]
    S = LA.similarity(A)
    S = S[S.article_id.isin(art.article_id)]
    S = S.assign(closest=S[["paper", "sisters"]].max(axis=1), sister_closest=S.sisters > S.paper)
    o = S.groupby("domain").agg(without=("paper", "mean"), closest=("closest", "mean"),
                                sister_closest=("sister_closest", "mean"), near_copy=("near copy", "mean"))
    o = o.reindex(outlets.index)
    has = outlets.index.isin(LA.sisters(A)[1])
    T = pd.DataFrame([{"outlets": group, "n": int(mask.sum()), "similarity without the sisters": o.without[mask].mean(),
                       "with the sisters": o.closest[mask].mean(),
                       "% closest training article is a sister's": 100 * o.sister_closest[mask].mean(),
                       "% near copy in a sister": 100 * o.near_copy[mask].mean()}
                      for group, mask in [("with a sister", has), ("without a sister", ~has)]])
    closer = (o.closest - o.without).to_numpy()
    articles, y = set(art.article_id), outlets.score
    R = []
    for m in LA.MODELS:
        drop = ((LA.outlet_predictions("L2", m, outlets, articles) - y).abs()
                - (LA.outlet_predictions("sisters", m, outlets, articles) - y).abs()).to_numpy()
        for group, mask in [("all", np.full(len(y), True)), ("with a sister", has)]:
            ii = np.where(mask)[0]
            idx = ii[resamples(len(ii), B)]
            r = ME.spearman_rows(closer[idx], drop[idx])
            R.append({"model": m, "outlets": group, "n": len(ii), "Spearman": r[0],
                      "low": np.nanpercentile(r[1:], 2.5), "high": np.nanpercentile(r[1:], 97.5)})
    return T, pd.DataFrame(R)


def owner_scores():
    """How much the NewsGuard score changes inside an ownership group: for every paper outlet with sisters, the mean
    absolute difference from its sisters' scores; against any two outlets among the same ones."""
    import ladder as LA
    o = pd.read_csv(LA.DIR / "outlets_by_step.csv", index_col="domain")
    sis, withs = LA.sisters(LA.stages()[0])
    diff = pd.Series({p: (o.score[[x for x in sis if o.group[x] == o.group[p]]] - o.score[p]).abs().mean() for p in withs})
    sc = o.score[sis + withs].to_numpy()
    anyp = np.abs(sc[:, None] - sc[None, :])[np.triu_indices(len(sc), 1)]
    return pd.DataFrame([
        {"pairs": "the same owner (paper outlet and its sisters)", "outlets": len(withs), "mean |difference|": diff.mean(),
         "median |difference|": diff.median(), "% the same score": 100 * (diff == 0).mean()},
        {"pairs": "any two of these outlets", "outlets": len(sc), "mean |difference|": anyp.mean(),
         "median |difference|": np.median(anyp), "% the same score": 100 * (anyp == 0).mean()}])

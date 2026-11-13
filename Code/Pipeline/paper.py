"""Step 15: every number of the paper, from the saved predictions only (no model is trained here).

- Main table (all articles): Spearman, Kendall, MAE, % within 20 points per outlet, for every regressor of steps 7
  and 13 and three baselines without a model of the text: the training mean, the training median (the best constant
  for the MAE) and the training mean of the outlet's political orientation. 95% percentile intervals, B = 10,000
  bootstrap resamples of the outlets.
- Curves: Spearman and MAE from k = 1, 2, 3, 5, 10, 20, 50 random articles per outlet and from all (the draws of step
  10, the same for every model; measure computed on every draw, then averaged), B = 2,000.
- Comparisons: paired bootstrap on the same resampled outlets; two-sided bootstrap p-value; Holm correction inside each
  family (main table against TF-IDF; curves against TF-IDF, per model; the claims of the text).
- Where the models err: signed error and MAE per NewsGuard level and per political orientation (also at equal score,
  and the correlation inside each orientation), and every measure per article against per outlet.
- Robustness: notoriety of the outlets (Tranco rank), political orientation (the gap at equal score, the gap that the
  compression alone would produce, calibration by group), who is left out by the cleaning, the leakage ladder (14).
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
def tranco_ranks(outlets, path=None):
    """Tranco rank of every outlet's domain (exact match; NaN = not in the top million)."""
    path = path or C.DATA / "risorse" / "tranco" / "top-1m.csv"
    t = pd.read_csv(path, header=None, names=["rank", "domain"])
    return outlets.index.map(t.drop_duplicates("domain").set_index("domain")["rank"]).astype(float)


def notoriety(art, outlets, rows, models=("TF-IDF", "Qwen3-8B", "Mistral-24B"), B=10_000):
    """Results in the more and in the less known half of the outlets (median Tranco rank; outside the top million =
    less known), and the gain of the language models over TF-IDF in each half."""
    rank = tranco_ranks(outlets).fillna(np.inf).to_numpy()
    cut = np.median(rank)
    assert np.isfinite(cut), "more than half of the outlets are outside the Tranco top million: split another way"
    known = rank <= cut
    y, n, draws = outlets.score.to_numpy(), len(outlets), draw_index(outlets, rows)
    P = {m: matrix(article_vector(m, art), draws, "all", n)[0] for m in models}
    out = []
    for half, mask in [("more known", known), ("less known", ~known)]:
        ii = np.where(mask)[0]
        idx = ii[resamples(len(ii), B)]
        V = {m: ME.resampled_measures(P[m], y, idx, ["Spearman", "MAE"]) for m in models}
        for m in models:
            for mes, v in V[m].items():
                r = {"half": half, "outlets": len(ii), "median rank": float(np.median(rank[mask])), "model": m,
                     "measure": mes, "value": v[0]}
                if m != "TF-IDF":
                    r.update(compare(v, V["TF-IDF"][mes], mes))
                out.append(r)
    return pd.DataFrame(out)


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
    """Per rung and model: Spearman and MAE on the paper's outlets and test articles, and the difference with L4."""
    import ladder as LA
    articles = set(art.article_id)
    y, idx = outlets.score.to_numpy(), resamples(len(outlets), B)
    out, V = [], {}
    for rung in LA.RUNGS:
        for m in LA.MODELS:
            p = LA.outlet_predictions(rung, m, outlets, articles)
            if p is not None:
                V[(rung, m)] = ME.resampled_measures(p.to_numpy(), y, idx, ["Spearman", "MAE"])
    for (rung, m), res in V.items():
        for mes, v in res.items():
            r = {"rung": rung, "model": m, "measure": mes, "value": v[0],
                 "low": np.percentile(v[1:], 2.5), "high": np.percentile(v[1:], 97.5)}
            if rung != "L4" and ("L4", m) in V:
                r.update(compare(v, V[("L4", m)][mes], mes))
            out.append(r)
    return pd.DataFrame(out)

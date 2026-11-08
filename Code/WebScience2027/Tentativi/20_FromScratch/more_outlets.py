"""Step 12 (added 8/11): more outlets in TRAINING only. The test is unchanged (the 226 political outlets, the same 5
folds and draws); the non-political NewsGuard outlets we scraped are added to the training of every fold.

Why: the limit of steps 10 and 11 is the number of independent outlets (about 180 per fold), not the way articles are
presented. Here the training gets several hundred more outlets.

Rules (fixed before the results; the political dataset of steps 2-3 is not touched):
1. candidates: scraped outlets with a NewsGuard 2026 score that are NOT political-only (steps 2-11 use those);
2. the same cleaning as step 2: text from the HTML (trafilatura, precision), English, 100-1,500 words, from 2025;
3. duplicates: an extra article that is a copy (exact or Jaccard of word 5-grams >= 0.5) of ANY article of the political
   dataset is dropped, whatever the dates (no test text can enter the training); among extra articles the oldest copy
   is kept; then at least 10 articles per outlet;
4. ownership groups (NewsGuard Owner / Parent Domain, as in step 2) computed on political + extra outlets together: an
   extra outlet in the group of one of the 226 political outlets is dropped (the groups stay isolated); among the
   others one outlet per group (the one with most articles);
5. at most `MAX_EXTRA` articles per extra outlet (random, seeded): every outlet counts about the same and the frozen
   vectors stay affordable;
6. anonymization with the same six rules as step 3.
Models: the best configurations found so far on the same footing as steps 7 and 10: TF-IDF + ridge; Qwen3-8B and
Mistral-24B frozen + ridge, ridge with outlets weighted the same, small neural network (`boost.py`). Settings chosen on
3 inner folds of the POLITICAL training outlets (the extra outlets are always in the inner training); test = political
outlets of the fold, all articles and k random ones (the same draws as steps 10-11).
ModernBERT (meta-articles, as step 11) is prepared but runs only if `RUN_BERT` is set.
Outputs in Dataset/tentativi/20_FromScratch/more_outlets/."""
import hashlib
import time

import numpy as np
import pandas as pd

import boost as BO
import cleaning as CL
import models as MO20
from common import DATA, NEWSGUARD, SCRAPING, SEED, level_from_score, log
from common import OUT as SOURCE

OUT = SOURCE / "more_outlets"
PRED = OUT / "predictions"
LOG = OUT / "log.txt"
MAX_EXTRA = 30
MIN_WORDS, MAX_WORDS, MIN_ARTICLES, FROM_DATE = 100, 1500, 10, "2025-01-01"
NEAR_DUPLICATE, CANDIDATES = 0.5, 0.3
EXTRA_ID = 10_000_000                                   # extra article ids start here (no clash with step 2)
EMB_FILES = {"Qwen3-8B": "extra_qwen3_embedding_8b.npy", "Mistral-24B": "extra_mistral_small_24b_layer27.npy"}
QWEN, MISTRAL = "Qwen/Qwen3-Embedding-8B", "mistralai/Mistral-Small-24B-Base-2501"
MODELS = ["TF-IDF", "Qwen3-8B · ridge", "Qwen3-8B · ridge_outlet", "Qwen3-8B · mlp",
          "Mistral-24B · ridge", "Mistral-24B · ridge_outlet", "Mistral-24B · mlp"]


# ------------------------------------------------------------------------- 1-6. the extra dataset
def candidates():
    ng = CL.read_newsguard(NEWSGUARD)
    status = CL.read_status(SCRAPING)
    political = set(CL.political_outlets(status, ng))
    scored = status.dominio.map(ng.Score).notna()
    return ng, sorted(status.dominio[scored & ~status.dominio.isin(political)])


def extract(domains, n_jobs):
    from joblib import Parallel, delayed
    cache = OUT / "extracted.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    rows = Parallel(n_jobs=n_jobs)(delayed(CL.extract_outlet)(d, SCRAPING) for d in domains)
    raw = pd.DataFrame([r for rr in rows for r in rr]); raw.to_csv(cache, index=False)
    return raw


def filters(raw, n_jobs, step):
    """Text, English, length, date (the rules of step 2)."""
    from joblib import Parallel, delayed
    art = raw[raw.text.notna()].reset_index(drop=True); step("with extracted text", art)
    art["language"] = Parallel(n_jobs=n_jobs, batch_size=200)(delayed(CL.language)(t) for t in art.text)
    art = art[art.language == "en"]; step("English", art)
    art = art.assign(words=art.text.str.split().str.len())
    art = art[art.words.between(MIN_WORDS, MAX_WORDS)]; step(f"{MIN_WORDS}-{MAX_WORDS:,} words", art)
    art = art.assign(date=pd.to_datetime(art.date, errors="coerce", utc=True).dt.tz_localize(None))
    art = art[art.date >= FROM_DATE].reset_index(drop=True); step(f"published from {FROM_DATE[:4]}", art)
    art["article_id"] = EXTRA_ID + np.arange(len(art))
    return art


def deduplicate(art, political, n_jobs, step):
    """Drop extra articles that copy any political article; among extra articles keep the oldest copy."""
    art = art.assign(norm=art.text.map(CL.normalize))
    pol = political.assign(norm=political.text.map(CL.normalize))
    fp = lambda s: hashlib.md5(s.encode()).hexdigest()
    art["fingerprint"] = art.norm.map(fp)
    art = art[~art.fingerprint.isin(set(pol.norm.map(fp)))]
    art = art.sort_values(["date", "article_id"]).drop_duplicates("fingerprint").sort_values("article_id")
    step("no exact copies (of political or extra articles)", art)
    both = pd.concat([pol[["article_id", "domain", "norm"]], art[["article_id", "domain", "norm"]]], ignore_index=True)
    pairs = CL.similar_pairs(both, CANDIDATES, n_jobs)
    pairs = pairs[pairs.jaccard >= NEAR_DUPLICATE]
    is_pol = lambda x: x < EXTRA_ID
    copies_of_political = set(pairs.article_b[is_pol(pairs.article_a) & ~is_pol(pairs.article_b)]) | \
        set(pairs.article_a[is_pol(pairs.article_b) & ~is_pol(pairs.article_a)])
    extra_pairs = pairs[~is_pol(pairs.article_a) & ~is_pol(pairs.article_b)]
    drop = copies_of_political | CL.copies_to_drop(extra_pairs, art.set_index("article_id").date)
    art = art[~art.article_id.isin(drop)]
    step(f"no near copies (Jaccard >= {NEAR_DUPLICATE}; {len(copies_of_political):,} were copies of political articles)", art)
    return art.drop(columns=["norm", "fingerprint"])


def groups_and_cap(art, political_outlets, ng, step):
    n = art.groupby("domain").size()
    art = art[art.domain.isin(n.index[n >= MIN_ARTICLES])]; step(f"outlets with >= {MIN_ARTICLES} articles", art)
    groups = CL.ownership_groups(sorted(set(art.domain) | set(political_outlets)), ng)
    political_groups = {groups[d] for d in political_outlets}
    art = art.assign(group=art.domain.map(groups))
    art = art[~art.group.isin(political_groups)]; step("not in the ownership group of a political outlet", art)
    kept = (art.groupby("domain").agg(group=("group", "first"), n=("article_id", "size")).reset_index()
            .sort_values(["n", "domain"], ascending=[False, True]).groupby("group").head(1).domain)
    art = art[art.domain.isin(kept)]; step("one outlet per ownership group", art)
    art = art.sample(frac=1, random_state=SEED).groupby("domain").head(MAX_EXTRA).sort_values("article_id")
    step(f"at most {MAX_EXTRA} articles per outlet", art)
    art = art.assign(score=art.domain.map(ng.Score).astype(float))
    return art.assign(label=level_from_score(art.score))


# ------------------------------------------------------------------------- union with the political dataset
def union():
    """Political articles first (same order as steps 5-10: same row numbers, same test draws), then extra ones."""
    art, outlets, _ = BO.load()
    extra = pd.read_csv(OUT / "articles_extra_anonymous.csv.gz")
    extra["fold"] = 0                                    # 0 = always training
    U = pd.concat([art[["article_id", "domain", "text", "score", "label", "fold"]],
                   extra[["article_id", "domain", "text", "score", "label", "fold"]]], ignore_index=True)
    eo = extra.groupby("domain").agg(score=("score", "first"), label=("label", "first")).assign(fold=0)
    O = pd.concat([outlets[["score", "label", "fold"]], eo])
    rows = pd.Series(np.arange(len(U))).groupby(U.domain.to_numpy()).apply(np.array)
    return U, O, rows, outlets, list(eo.index)


def embed(name, U):
    """Frozen vectors of the extra articles (the political ones exist from step 5); returns the union matrix."""
    f = OUT / "embeddings" / EMB_FILES[name]
    extra = U.article_id.to_numpy() >= EXTRA_ID
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        MO20.wait_for_gpu(22, LOG)
        texts = U.text[extra].fillna("").tolist()
        if name == "Qwen3-8B":
            np.save(f, MO20.qwen_embeddings(texts, QWEN, 16384, LOG))
        else:
            E = MO20.mistral_embeddings(texts, MISTRAL, [27], 4096, LOG, OUT / "embeddings" / "mistral_partial.npy")
            np.save(f, E[27]); (OUT / "embeddings" / "mistral_partial.npy").unlink(missing_ok=True)
        log(f"extra embeddings {name}: done", LOG)
    pol = np.load(SOURCE / "embeddings" / BO.EMBEDDINGS[name][0])
    return np.vstack([pol, np.load(f)]).astype(np.float32)


# ------------------------------------------------------------------------- models
def _save(name, fold, draws, p, scores=None, best=None):
    folder = PRED / name; folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"domain": [d for d, *_ in draws], "k": [k for _, k, _, _ in draws], "draw": [j for _, _, j, _ in draws],
                  "prediction": p}).to_csv(folder / f"fold{fold}.csv.gz", index=False)
    if scores is not None:
        scores.assign(fold=fold, chosen=scores.index == scores["inner Spearman"].idxmax()).to_csv(folder / f"inner_fold{fold}.csv", index=False)
    log(f"more outlets, {name}, fold {fold}: {best}", LOG)


def inner_split(outlets, fold):
    train = outlets.index[outlets.fold != fold]
    return train, MO20.score_split(outlets.score.reindex(train), SEED + fold, k=3)


def fit_embedding(name, method, fold, X, U, O, rows, outlets, extra):
    """boost.py method with the extra outlets always in training; settings chosen on the political inner folds."""
    import torch
    from scipy.stats import spearmanr
    label = f"{name} · {method}"
    if (PRED / label / f"fold{fold}.csv.gz").exists():
        return
    start = time.time()
    train, inner = inner_split(outlets, fold)
    settings = BO.grid(method)
    res = {i: [] for i in range(len(settings))}
    for j in (1, 2, 3):
        F = BO.Fold(X, U, O, rows, list(inner.index[inner != j]) + extra)
        val = inner.index[inner == j]
        bags, y = [rows[d] for d in val], outlets.score.reindex(val).to_numpy()
        rho = lambda p: spearmanr(y, p).correlation
        seed = SEED + 10 * fold + j
        if method in BO.RIDGES:
            w = BO.article_weights(settings[0]["weights"], F.dom_art[F.tr], F.y_art[F.tr], F.train_scores)
            for i, (beta, b0) in enumerate(BO.ridge_path(F.X[F.tr], F.y_art[F.tr], w, BO.ALPHAS)):
                res[i].append(rho(BO.predictor_articles(F, lambda r: (F.X[r] @ beta + b0).cpu().numpy())(bags)))
        else:                                             # mlp: one training per setting, every epoch checkpoint
            groups = {}
            for i, s in enumerate(settings):
                groups.setdefault(BO._key(s, ("epoch",)), []).append(i)
            for ids in groups.values():
                net, st = BO.train_mlp(F, settings[ids[0]], seed, sorted({settings[i]["epoch"] for i in ids}))
                for i in ids:
                    res[i].append(rho(BO.mlp_predictor(F, net, [st[settings[i]["epoch"]]])(bags)))
        del F; torch.cuda.empty_cache()
    scores = pd.DataFrame([{**s, "inner Spearman": np.mean(res[i])} for i, s in enumerate(settings)])
    best = {k: v for k, v in scores.loc[scores["inner Spearman"].idxmax()].drop("inner Spearman").to_dict().items()
            if not (isinstance(v, float) and np.isnan(v))}
    for k in ("epoch", "hidden"):
        if k in best:
            best[k] = int(best[k])
    F = BO.Fold(X, U, O, rows, list(train) + extra)
    predict = BO.final_predictor(method, best, F, fold)
    draws = BO.test_draws(outlets, rows, fold)
    _save(label, fold, draws, predict([b for *_, b in draws]), scores, f"{best} inner {scores['inner Spearman'].max():.3f}, {time.time() - start:.0f} s")
    del F; torch.cuda.empty_cache()


def fit_tfidf(fold, U, rows, outlets, extra, n_jobs=8):
    """TF-IDF of step 7 + ridge, the extra outlets always in training; alpha on the political inner folds."""
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    if (PRED / "TF-IDF" / f"fold{fold}.csv.gz").exists():
        return
    start = time.time()
    texts, y = U.text.fillna("").to_numpy(), U.score.to_numpy()
    rows_of = lambda ds: np.concatenate([rows[d] for d in ds])
    vec = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000,
                                  strip_accents="unicode")
    train, inner = inner_split(outlets, fold)
    grid = MO20.GRIDS[("regression", "TF-IDF")]

    def one(j):
        val = inner.index[inner == j]
        tr = rows_of(list(inner.index[inner != j]) + extra)
        v = vec(); A = v.fit_transform(texts[tr]); out = {}
        for a in grid:
            m = Ridge(alpha=a, random_state=SEED).fit(A, y[tr])
            p = [m.predict(v.transform(texts[rows[d]])).mean() for d in val]
            out[a] = spearmanr(outlets.score.reindex(val), p).correlation
        return out
    res = Parallel(n_jobs=min(3, n_jobs))(delayed(one)(j) for j in (1, 2, 3))
    means = pd.Series({a: np.mean([r[a] for r in res]) for a in grid})
    best = means.idxmax()
    tr = rows_of(list(train) + extra)
    v = vec(); m = Ridge(alpha=best, random_state=SEED).fit(v.fit_transform(texts[tr]), y[tr])
    te = rows_of(outlets.index[outlets.fold == fold])
    P = np.full(len(U), np.nan); P[te] = m.predict(v.transform(texts[te]))
    draws = BO.test_draws(outlets, rows, fold)
    scores = pd.DataFrame({"alpha": means.index, "inner Spearman": means.values})
    _save("TF-IDF", fold, draws, np.array([P[b].mean() for *_, b in draws]), scores,
          f"alpha {best} inner {means.max():.3f}, {time.time() - start:.0f} s")


# ------------------------------------------------------------------------- evaluation
def matrix(source, outlets, k, with_extra):
    """Draws × outlets of a model, with the extra outlets (this step) or without (steps 7 and 10, same draws)."""
    if with_extra:
        P = pd.concat([pd.read_csv(f, dtype={"k": str}) for f in sorted((PRED / source).glob("fold*.csv.gz"))])
    elif source == "TF-IDF":
        art, o, rows = BO.load()
        P = BO.tfidf_draws(art, o, rows)
    else:
        emb, method = source.split(" · ")
        P = pd.concat([pd.read_csv(f, dtype={"k": str}) for f in sorted((BO.PRED / emb / method).glob("fold*.csv.gz"))])
    return P[P.k == str(k)].pivot_table(index="draw", columns="domain", values="prediction").reindex(columns=outlets.index).to_numpy()

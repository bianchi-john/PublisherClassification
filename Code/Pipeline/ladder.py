"""Step 14: how much each protection of the protocol matters (the "leakage ladder").

The same models (TF-IDF, Qwen3-Embedding-8B and Mistral-Small-24B with the ridge of step 7) are trained under weaker
protocols and always tested on the SAME 25,524 articles of the 226 outlets of the paper, in the same folds; only the
training data and the text change:

| rung | training outlets and articles | text | compared with |
|---|---|---|---|
| article split | the 226 outlets, folds of ARTICLES (an outlet is on both sides) | anonymous | L4 |
| sisters | the 226 outlets + ALWAYS the sisters (see below), with all their articles, copies included | original | L2 |
| L2 | the 226 outlets (one per ownership group) | original | L4 |
| L3 | the 226 outlets | + links, e-mails, the outlet's own name (step 3, passes 1-2) | L4 |
| L4 | the 226 outlets | + GLiNER: the paper (step 7) | |

Sisters: the outlets that step 2 removed only because another outlet of their ownership group was kept (>= 10
articles published from 2025). They are put in training on purpose, in every fold, so every paper outlet of their group
is tested with a sister in training (a split that ignores ownership would do it by chance, most of the times). Being
sisters is only a hint that the texts are similar: `similarity` measures how close the test articles get to the
training ones, with and without the sisters.

The sisters come back from the step-2 cache of the extracted texts (`extracted.csv.gz`), with the chain of step 2
(`cleaning.chain`: same functions and rules; checked: the last step gives exactly the 25,524 articles of the paper).
Inner folds: by outlet (as step 7), by article for the article split. Outputs: `ladder/` in the data folder.

The same reconstruction gives `ladder/outlets_by_step.csv`: every political outlet sent to the scraper and the last
step it survived, for the analysis of who is left out (step 15).
"""
import functools

import numpy as np
import pandas as pd

import anonymization_ner as AX
import cleaning as CL
import common as C
import models as MO

DIR = C.OUT / "ladder"
EMB = DIR / "embeddings"                   # raw/ and rules/: frozen vectors of the original and of the rules text
PRED = DIR / "predictions"
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B"]
RUNGS = ["article split", "sisters", "L2", "L3", "L4"]
REFERENCE = {"article split": "L4", "sisters": "L2", "L2": "L4", "L3": "L4"}    # the same text, one protection more


# ------------------------------------------------------------------------- the stages of step 2, kept this time
def stages(n_jobs=16, logfile=None):
    """Articles of the step "from 2025" (article_id as in step 2) with the stage they reach, and every political outlet
    with the last step it survives. The chain of step 2 again (cleaning.chain); cached in ladder/."""
    DIR.mkdir(parents=True, exist_ok=True)
    fa, fo = DIR / "articles_by_stage.csv.gz", DIR / "outlets_by_step.csv"
    if fa.exists() and fo.exists():
        return pd.read_csv(fa), pd.read_csv(fo, index_col="domain")
    ng = CL.read_newsguard(C.NEWSGUARD)
    steps = {CL.STEPS[0]: set(CL.political_outlets(CL.read_status(C.SCRAPING), ng))}
    for name, df in CL.chain(pd.read_csv(C.OUT / "extracted.csv.gz"), ng, n_jobs):
        steps[name] = set(df.domain)
        if name == "from 2025":
            art = df.copy()
        elif name == CL.STEPS[-1]:
            s2 = set(df.article_id)
    n = art.groupby("domain").size()
    art["s0"] = art.domain.isin(n.index[n >= CL.MIN_ARTICLES])              # the paper's outlets and their sisters
    art["s2"] = art.article_id.isin(s2)                                     # the paper's articles
    final = pd.read_csv(C.OUT / "articles.csv.gz", usecols=["article_id", "domain"])
    assert s2 == set(final.article_id), "not the paper's articles"
    assert (art.set_index("article_id").domain.reindex(final.article_id).to_numpy() == final.domain.to_numpy()).all()
    art["score"] = art.domain.map(ng.Score).astype(float)
    art[art.s0][["article_id", "domain", "date", "score", "text", "s0", "s2"]].to_csv(fa, index=False)
    # outlets: score, orientation, country, and how far each one got
    o = pd.DataFrame(index=pd.Index(sorted(steps[CL.STEPS[0]]), name="domain"))
    o["score"] = o.index.map(ng.Score).astype(float)
    o["orientation"] = o.index.map(ng.Orientation).fillna("none")
    o["country"] = o.index.map(ng.Country).fillna("")
    o["last step"] = [next((s for s in reversed(CL.STEPS) if d in steps[s]), CL.STEPS[0]) for d in o.index]
    o["group"] = o.index.map(CL.ownership_groups(list(o.index), ng))
    o.to_csv(fo)
    C.log(f"ladder: stages saved ({art.s0.sum():,} articles of outlets with >= 10)", logfile)
    return pd.read_csv(fa), o


# ------------------------------------------------------------------------- texts, folds, frozen vectors
@functools.lru_cache(maxsize=None)
def _paper_text(kind):
    """The text of the 226 outlets by article_id: "rules" (passes 1-2 of step 3) or "anonymous" (step 3). Computed once
    per session: step 14 asks for the rules text three times."""
    if kind == "rules":
        full = pd.read_csv(C.OUT / "articles.csv.gz")
        full["title"] = full.title.fillna(""); full["text"] = full.text.fillna("")
        return AX.rules(full)[0].set_index("article_id").text
    return pd.read_csv(C.OUT / "articles_anonymous.csv.gz").set_index("article_id").text


def texts(kind, a):
    """The text of the articles `a` (article_id, domain, text = original): "original", "rules" or "anonymous"."""
    if kind == "original":
        return a.text.fillna("").to_numpy()
    return _paper_text(kind).reindex(a.article_id).fillna("").to_numpy()


def sisters(A):
    """(the sisters: outlets with >= 10 articles published from 2025 that are not in the paper but share the
    ownership group of a paper outlet; the paper outlets that have at least one sister)."""
    group = pd.read_csv(DIR / "outlets_by_step.csv", index_col="domain").group
    paper = pd.read_csv(C.OUT / "split.csv", index_col="domain").index
    others = pd.Index(A.domain.unique()).difference(paper)
    sis = others[others.map(group).isin(group[paper])]
    return list(sis), list(paper[paper.map(group).isin(group[sis])])


def rung_data(rung, A):
    """(articles with text, score and fold; inner folds or None; kind of text) of a rung. A = articles_by_stage.
    Outer folds by outlet are the paper's; the sisters get fold 0: never tested, always in training."""
    if rung == "article split":
        a = A[A.s2].reset_index(drop=True)
        rng = np.random.default_rng(C.SEED)
        a["fold"] = rng.permutation(np.arange(len(a)) % 5) + 1
        inner = rng.permutation(np.arange(len(a)) % 3) + 1
        return a.assign(text=texts("anonymous", a)), inner, "anonymous"
    keep = (A.s2 | A.domain.isin(sisters(A)[0])) if rung == "sisters" else A.s2
    a = A[keep].reset_index(drop=True)
    kind = "rules" if rung == "L3" else "original"
    fold = pd.read_csv(C.OUT / "split.csv", index_col="domain").fold
    return a.assign(text=texts(kind, a), fold=a.domain.map(fold).fillna(0).astype(int)), None, kind


def embed(kind, A, logfile, gpu_gb=19, qwen_budget=16384, mistral_budget=8192):
    """Frozen vectors of Qwen3-Embedding-8B and Mistral-Small-24B (layers 27 and 40) of the "original" text (all the
    articles of articles_by_stage: the paper's and the sisters') or of the "rules" text (the 226 outlets). Skips what
    is already saved."""
    a = A if kind == "original" else A[A.s2]
    folder = EMB / kind; folder.mkdir(parents=True, exist_ok=True)
    t = list(texts(kind, a))
    f = folder / MO.EMBEDDING_FILES["Qwen3-8B"]
    if not f.exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        np.save(f, MO.qwen_embeddings(t, "Qwen/Qwen3-Embedding-8B", qwen_budget, logfile,
                                      partial_file=folder / "qwen_partial.npy"))
        (folder / "qwen_partial.npy").unlink(missing_ok=True)
    files = MO.EMBEDDING_FILES["Mistral-24B"]
    if not all((folder / x).exists() for x in files.values()):
        MO.wait_for_gpu(gpu_gb, logfile)
        E = MO.mistral_embeddings(t, "mistralai/Mistral-Small-24B-Base-2501", list(files), mistral_budget, logfile,
                                  folder / "mistral_partial.npy")
        for layer, x in files.items():
            np.save(folder / x, E[layer])
        (folder / "mistral_partial.npy").unlink(missing_ok=True)
    a[["article_id"]].to_csv(folder / "article_order.csv", index=False)


def vectors(name, a, kind):
    """The frozen vectors of `name` for the rows of `a`, from the step-5 file (anonymous text) or from ladder/."""
    folder = C.EMBEDDINGS if kind == "anonymous" else EMB / kind
    order = pd.read_csv(folder / "article_order.csv").article_id
    pos = pd.Series(np.arange(len(order)), index=order).reindex(a.article_id).to_numpy()
    assert not np.isnan(pos).any(), f"missing vectors for {name}, {kind}"
    files = MO.EMBEDDING_FILES[name]
    load = lambda x: np.load(folder / x, mmap_mode="r")[pos.astype(int)]
    return {l: load(x) for l, x in files.items()} if isinstance(files, dict) else load(files)


# ------------------------------------------------------------------------- models and evaluation
def run(rung, A, models=MODELS, n_jobs=16, logfile=None):
    """The ridge models of step 7 under one rung (L4 is step 7 itself)."""
    if rung == "L4":
        return
    a, inner, kind = rung_data(rung, A)
    C.log(f"ladder {rung}: {a.domain.nunique()} outlets, {len(a):,} articles, {kind} text", logfile)
    for name in models:
        X = None if name == "TF-IDF" else vectors(name, a, kind)
        for k in range(1, 6):
            MO.linear_fold("regression", name, a, k, X=X, n_jobs=n_jobs, logfile=logfile,
                           folder=PRED / rung / name, models_dest=False, inner=inner)


def outlet_predictions(rung, name, outlets, articles):
    """Mean prediction of every paper outlet over its paper articles (the same test articles for every rung)."""
    folder = C.PREDICTIONS / "regression" / name if rung == "L4" else PRED / rung / name
    p = MO.predictions("regression", name, folder)
    if p is None:
        return None
    p = p[p.article_id.isin(articles)]
    return p.groupby("domain").prediction.mean().reindex(outlets.index)


# ------------------------------------------------------------------------- how similar test and training articles are
def _max_cosine(X, rows, cols, chunk=500):
    """For every row of X in `rows`: the highest dot product (= cosine: the rows have length 1) with a row in `cols`."""
    Xc = X[cols].T.tocsc()
    return np.concatenate([(X[r] @ Xc).max(axis=1).toarray().ravel() for r in np.array_split(rows, len(rows) // chunk + 1)])


def similarity(A, n_jobs=16, logfile=None):
    """For every test article of the paper, how close the training articles of the "sisters" rung get:
    - "paper": the highest cosine with an article of the paper's training outlets of its fold (what L2 trains on);
    - "sisters": the highest cosine with an article of the sisters (added in training by the "sisters" rung);
    - "near copy": a sister has a near copy of it (Jaccard of word 5-grams >= 0.5, the rule of step 2).
    Cosine of the TF-IDF vectors (the settings of the TF-IDF model, fitted on all these articles). Cached in ladder/."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    f = DIR / "similarity.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    a, _, _ = rung_data("sisters", A)
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000,
                        strip_accents="unicode").fit_transform(a.text).tocsr()
    fold = a.fold.to_numpy()
    out = []
    for k in range(1, 6):
        te = np.where(fold == k)[0]
        out.append(a.iloc[te][["article_id", "domain", "fold"]].assign(
            paper=_max_cosine(X, te, np.where((fold != k) & (fold != 0))[0]),
            sisters=_max_cosine(X, te, np.where(fold == 0)[0])))
        C.log(f"similarity: fold {k} done", logfile)
    out = pd.concat(out)
    pairs = CL.similar_pairs(a.assign(norm=a.text.map(CL.normalize)), CL.CANDIDATES, n_jobs)
    pairs = pairs[pairs.jaccard >= CL.NEAR_DUPLICATE]
    sis = set(sisters(A)[0])
    copied = set(pairs.article_a[pairs.outlet_b.isin(sis)]) | set(pairs.article_b[pairs.outlet_a.isin(sis)])
    out["near copy"] = out.article_id.isin(copied)
    out.to_csv(f, index=False)
    return out

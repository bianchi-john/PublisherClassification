"""Step 19: replication on held-out outlets, pre-registered in PREREGISTRATION_19.md (read it first).

A. non-political NewsGuard outlets (main test); B. political outlets with fewer than 10 articles (in-domain check).
The saved fold models of step 7 are applied without any change: the prediction of an article is the mean of the five
fold models, the score of an outlet the mean over its articles. A few articles of the 226 outlets go through the same
code as a check (their fold model must give the saved prediction).

Outputs in heldout/ in the data folder (every part cached, so a failed run goes on from there); the numbers in
paper_heldout.csv (step 19 notebook).
"""
import hashlib

import joblib
import numpy as np
import pandas as pd

import anonymization_ner as AX
import cleaning as CL
import common as C
import models as MO

DIR = C.OUT / "heldout"
MAX_ARTICLES = 30                   # per outlet of set A
FIRST_ID = 20_000_000               # article ids of set A (no clash with step 2)
CONTROL = 30                        # articles of the 226 outlets used as a check
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
QWEN, MISTRAL = "Qwen/Qwen3-Embedding-8B", "mistralai/Mistral-Small-24B-Base-2501"


def _steps(rows, name, art):
    rows.append({"step": name, "articles": len(art), "outlets": art.domain.nunique()})


def one_per_group(art, paper_domains, ng):
    """Drop the outlets in the ownership group of one of the 226, then keep one outlet per group (most articles)."""
    groups = CL.ownership_groups(sorted(set(art.domain) | set(paper_domains)), ng)
    paper_groups = {groups[d] for d in paper_domains}
    art = art.assign(group=art.domain.map(groups))
    art = art[~art.group.isin(paper_groups)]
    kept = (art.groupby("domain").agg(group=("group", "first"), n=("article_id", "size")).reset_index()
            .sort_values(["n", "domain"], ascending=[False, True]).groupby("group").head(1).domain)
    return art[art.domain.isin(kept)]


# ------------------------------------------------------------------------- A. non-political outlets
def set_a(paper_art, n_jobs=32, logfile=None):
    """The non-political outlets, cleaned as in step 2 (cached in heldout/)."""
    from joblib import Parallel, delayed
    f, fs = DIR / "articles_A.csv.gz", DIR / "steps_A.csv"
    if f.exists():
        return pd.read_csv(f), pd.read_csv(fs)
    DIR.mkdir(parents=True, exist_ok=True)
    ng = CL.read_newsguard(C.NEWSGUARD)
    status = CL.read_status(C.SCRAPING)
    political = set(CL.political_outlets(status, ng))
    scored = status.dominio.map(ng.Score).notna()
    domains = sorted(set(status.dominio[scored & ~status.dominio.isin(political)]))
    fx = DIR / "extracted_A.csv.gz"
    if not fx.exists():
        rows = Parallel(n_jobs=n_jobs)(delayed(CL.extract_outlet)(d, C.SCRAPING) for d in domains)
        pd.DataFrame([r for rr in rows for r in rr]).to_csv(fx, index=False)
        C.log(f"held-out A: text extracted for {len(domains):,} candidate outlets", logfile)
    raw = pd.read_csv(fx)
    steps = []
    _steps(steps, "scraped articles of the candidate outlets", raw)
    art = raw[raw.text.notna()].reset_index(drop=True); _steps(steps, "text extracted", art)
    art = art.assign(language=Parallel(n_jobs=n_jobs, batch_size=200)(delayed(CL.language)(t) for t in art.text))
    art = art[art.language == "en"]; _steps(steps, "English", art)
    art = art.assign(words=art.text.str.split().str.len())
    art = art[art.words.between(CL.MIN_WORDS, CL.MAX_WORDS)]; _steps(steps, "100-1,500 words", art)
    art = art.assign(date=pd.to_datetime(art.date, errors="coerce", utc=True).dt.tz_localize(None))   # mixed time zones
    art = art[art.date >= CL.FROM_DATE].reset_index(drop=True); _steps(steps, "from 2025", art)
    art["article_id"] = FIRST_ID + np.arange(len(art))
    # duplicates: copies of the 226 outlets' articles are dropped; among the new articles the oldest copy is kept
    fp = lambda s: hashlib.md5(s.encode()).hexdigest()
    pol = paper_art.assign(norm=paper_art.text.fillna("").map(CL.normalize))
    art = art.assign(norm=art.text.map(CL.normalize))
    art["fingerprint"] = art.norm.map(fp)
    art = art[~art.fingerprint.isin(set(pol.norm.map(fp)))]
    art = art.sort_values(["date", "article_id"]).drop_duplicates("fingerprint").sort_values("article_id")
    _steps(steps, "no exact duplicates", art)
    both = pd.concat([pol[["article_id", "domain", "norm"]], art[["article_id", "domain", "norm"]]], ignore_index=True)
    pairs = CL.similar_pairs(both, CL.CANDIDATES, n_jobs)
    pairs = pairs[pairs.jaccard >= CL.NEAR_DUPLICATE]
    new_a, new_b = pairs.article_a >= FIRST_ID, pairs.article_b >= FIRST_ID
    copies = set(pairs.article_b[~new_a & new_b]) | set(pairs.article_a[new_a & ~new_b])
    drop = copies | CL.copies_to_drop(pairs[new_a & new_b], art.set_index("article_id").date)
    art = art[~art.article_id.isin(drop)]; _steps(steps, "no near duplicates", art)
    n = art.groupby("domain").size()
    art = art[art.domain.isin(n.index[n >= CL.MIN_ARTICLES])]; _steps(steps, ">= 10 articles", art)
    art = one_per_group(art, set(paper_art.domain), ng); _steps(steps, "not in a group of the 226, one per group", art)
    art = art.sample(frac=1, random_state=C.SEED).groupby("domain").head(MAX_ARTICLES).sort_values("article_id")
    _steps(steps, f"at most {MAX_ARTICLES} articles per outlet", art)
    art = art.assign(set="A")[["article_id", "domain", "title", "date", "text", "set"]]
    art.to_csv(f, index=False); pd.DataFrame(steps).to_csv(fs, index=False)
    C.log(f"held-out A: {art.domain.nunique()} outlets, {len(art):,} articles", logfile)
    return art, pd.DataFrame(steps)


# ------------------------------------------------------------------------- B. political outlets with < 10 articles
def set_b(paper_art, n_jobs=32, logfile=None):
    """Political outlets that passed every step of step 2 except ">= 10 articles" (cached in heldout/)."""
    f = DIR / "articles_B.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    DIR.mkdir(parents=True, exist_ok=True)
    ng = CL.read_newsguard(C.NEWSGUARD)
    for name, df in CL.chain(pd.read_csv(C.OUT / "extracted.csv.gz"), ng, n_jobs):
        if name == "no near duplicates":
            art = df.copy()
            break
    n = art.groupby("domain").size()
    art = art[art.domain.isin(n.index[n < CL.MIN_ARTICLES])]
    art = one_per_group(art, set(paper_art.domain), ng)
    art = art.assign(set="B")[["article_id", "domain", "title", "date", "text", "set"]]
    art.to_csv(f, index=False)
    C.log(f"held-out B: {art.domain.nunique()} outlets, {len(art):,} articles", logfile)
    return art


# ------------------------------------------------------------------------- anonymization (step 3 unchanged)
def anonymize(art, logfile=None):
    """Passes 1-2 and GLiNER, as in step 3; the GLiNER spans are cached in heldout/anonymization (not step 3's)."""
    f = DIR / "articles_anonymous.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    AX.WORK = DIR / "anonymization"                   # never the cache of step 3
    a = art.assign(title=art.title.fillna(""), text=art.text.fillna(""))
    ruled, _ = AX.rules(a)
    spans = AX.run(ruled, logfile)
    anon = AX.apply(ruled, spans)
    anon.to_csv(f, index=False)
    C.log(f"held-out: {len(anon):,} articles anonymized", logfile)
    return anon


def control_articles(paper_anon, folds):
    """CONTROL articles of test fold 1 of the 226 outlets (seeded), already anonymized by step 3."""
    a = paper_anon.assign(fold=paper_anon.domain.map(folds))
    a = a[a.fold == 1].sample(CONTROL, random_state=C.SEED)
    return a.assign(set="control")[["article_id", "domain", "title", "date", "text", "set"]]


# ------------------------------------------------------------------------- the five fold models of every model
def _heads(name, X):
    """Predictions of the five saved fold models (one column per fold)."""
    out = np.zeros((len(X), 5))
    for k in range(1, 6):
        b = joblib.load(C.MODELS_DIR / "regression" / name / f"fold{k}.joblib")
        if name == "Mistral-24B":
            assert int(b["layer"]) == 27, "the fold model does not use layer 27"
        out[:, k - 1] = b["model"].predict(b["transformer"].transform(X))
    return out


def _modernbert(texts, logfile=None, gpu_gb=8, batch=4):
    """The five fine-tuned ModernBERT fold models (score = output x 100), as in step 7."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    MO.wait_for_gpu(gpu_gb, logfile)
    out = np.zeros((len(texts), 5), dtype=np.float32)
    for k in range(1, 6):
        d = C.MODELS_DIR / "regression" / "ModernBERT" / f"fold{k}"
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForSequenceClassification.from_pretrained(d, reference_compile=False).to("cuda").eval()
        ids = tok(texts, truncation=True, max_length=MO.MAX_TOKENS)["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        with torch.no_grad():
            for s in range(0, len(order), batch):
                g = order[s:s + batch]
                L = max(len(ids[i]) for i in g)
                x = np.full((len(g), L), tok.pad_token_id, dtype=np.int64); mask = np.zeros_like(x)
                for r, i in enumerate(g):
                    x[r, :len(ids[i])] = ids[i]; mask[r, :len(ids[i])] = 1
                with torch.autocast("cuda", dtype=torch.float16):
                    z = model(input_ids=torch.from_numpy(x).cuda(), attention_mask=torch.from_numpy(mask).cuda()).logits
                out[g, k - 1] = z.float().cpu().numpy()[:, 0] * 100
        del model; torch.cuda.empty_cache()
        C.log(f"held-out: ModernBERT fold {k} done", logfile)
    return out


def fold_predictions(art, gpu_gb=19, logfile=None):
    """Per article and model, the prediction of each of the five fold models (cached in heldout/)."""
    f = DIR / "fold_predictions.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    texts = art.text.fillna("").tolist()
    P = {"TF-IDF": _heads("TF-IDF", art.text.fillna("").to_numpy())}
    q = DIR / "qwen3_8b.npy"
    if not q.exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        np.save(q, MO.qwen_embeddings(texts, QWEN, 16384, logfile))
    P["Qwen3-8B"] = _heads("Qwen3-8B", np.load(q))
    m = DIR / "mistral_24b_layer27.npy"
    if not m.exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        np.save(m, MO.mistral_embeddings(texts, MISTRAL, [27], 8192, logfile, DIR / "mistral_partial.npy")[27])
    P["Mistral-24B"] = _heads("Mistral-24B", np.load(m))
    b = DIR / "modernbert.npy"
    if not b.exists():
        np.save(b, _modernbert(texts, logfile))
    P["ModernBERT"] = np.load(b)
    out = art[["article_id", "domain", "set"]].copy()
    for name, M in P.items():
        for k in range(5):
            out[f"{name} fold{k + 1}"] = M[:, k]
    out.to_csv(f, index=False)
    return out


def check_control(F, paper_predictions):
    """Largest difference between the fold-1 prediction of the control articles and the saved one of step 7."""
    c = F[F.set == "control"].set_index("article_id")
    return {name: float(np.abs(c[f"{name} fold1"] - paper_predictions[name].reindex(c.index)).max()) for name in MODELS}


def outlet_scores(F, held):
    """Outlet score of every model: mean over the articles of the mean of the five fold models."""
    F = F[F.set != "control"]
    out = pd.DataFrame({name: F[[f"{name} fold{k}" for k in range(1, 6)]].mean(1) for name in MODELS})
    out["domain"] = F.domain.to_numpy()
    return out.groupby("domain").mean().reindex(held.index)

"""Step 18: boilerplate, the lines that an outlet repeats in many of its articles (newsletter invitations, donation
appeals, disclaimers, captions), which the anonymization leaves in the text and which could let a model recognize the
outlet. Suggested by the explanation methods (Kozik et al., 2024): look at which parts of the text drive a prediction.

1. Boilerplate: a line that appears, with the digits ignored, in at least 3 articles and in at least 5% of the articles
   of the same outlet.
2. TF-IDF retrained without boilerplate (and without boilerplate and dates), same folds and inner choice of the
   penalty as step 7 (CPU).
3. A sample of 1,000 articles with boilerplate: the predictions of the saved fold models (the model that did not see
   the outlet) on the original text and on the text without boilerplate; TF-IDF, Qwen3-8B and Mistral-24B, whose frozen
   vectors are computed again for the cleaned texts (GPU).
4. Mistral-24B token by token on 300 of them: the ridge reads the standardized mean of the token states of layer 27, so
   the prediction is const + sum_t h_t . (w / sd) / T, exactly one contribution per token (GPU).

Outputs in boilerplate/ in the data folder (every part is cached, so a run that fails goes on from there); the numbers
of the paper in paper_boilerplate.csv (step 18 notebook).
"""
import re

import joblib
import numpy as np
import pandas as pd

import common as C
import models as MO

DIR = C.OUT / "boilerplate"
MISTRAL = "mistralai/Mistral-Small-24B-Base-2501"
MIN_ARTICLES, MIN_SHARE = 3, 0.05
MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE = re.compile(rf"\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b|\b\d{{1,2}}\s+{MONTHS}\b|\b20[12]\d\b|"
                  rf"\b\d{{1,2}}:\d{{2}}\b|\butc\b|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b", re.I)


# ------------------------------------------------------------------------- 1. boilerplate
def norm(line):
    """A line in lower case, digits as 0, spaces collapsed: the same line with another date or number is the same."""
    return re.sub(r"\s+", " ", re.sub(r"\d", "0", line.lower())).strip()


def find(art):
    """The boilerplate: the set of (outlet, normalized line)."""
    bp = set()
    for d, g in art.groupby("domain"):
        n, count = len(g), {}
        for text in g.text:
            for line in {norm(x) for x in text.split("\n") if x.strip()}:
                count[line] = count.get(line, 0) + 1
        bp |= {(d, line) for line, k in count.items() if k >= MIN_ARTICLES and k / n >= MIN_SHARE}
    return bp


def is_boilerplate(domain, line, bp):
    return bool(line.strip()) and (domain, norm(line)) in bp


def strip(art, bp):
    """The texts without the boilerplate lines (and without empty lines)."""
    return ["\n".join(x for x in t.split("\n") if x.strip() and not is_boilerplate(d, x, bp))
            for d, t in zip(art.domain, art.text)]


def summary(art, bp, clean):
    """How much boilerplate there is."""
    words = art.text.str.split().str.len()
    left = pd.Series(clean, index=art.index).str.split().str.len()
    return {"lines": len(bp), "outlets": len({d for d, _ in bp}), "outlets in all": art.domain.nunique(),
            "share of words": 1 - left.sum() / words.sum(), "share of articles": (left < words).mean()}


# ------------------------------------------------------------------------- 2. TF-IDF retrained
def retrain_tfidf(art, texts, version, n_jobs=16, logfile=None):
    """TF-IDF on other texts, the folds and the inner choice of step 7; returns the score of every outlet (mean of its
    articles). The models are not saved."""
    a = art.copy()
    a["text"] = texts
    folder = DIR / f"tfidf_{version}"
    for k in range(1, 6):
        MO.linear_fold("regression", "TF-IDF", a, k, n_jobs=n_jobs, logfile=logfile, folder=folder, models_dest=False)
    return MO.predictions("regression", "TF-IDF", folder).groupby("domain").prediction.mean()


# ------------------------------------------------------------------------- 3. saved models on a sample
def sample(art, clean, n=1000):
    """n articles drawn at random among those with boilerplate, with their cleaned text and share of boilerplate."""
    words = art.text.str.split().str.len().to_numpy()
    share = 1 - np.array([len(t.split()) for t in clean]) / words
    pick = np.sort(np.random.default_rng(C.SEED).choice(np.where(share > 0)[0], n, replace=False))
    S = art.iloc[pick][["article_id", "domain", "fold", "score", "text"]].reset_index(drop=True)
    S["clean"] = [clean[i] for i in pick]
    S["share"] = share[pick]
    return S


def fold_model(name, k):
    return joblib.load(C.MODELS_DIR / "regression" / name / f"fold{k}.joblib")


def predict_saved(name, S, X):
    """The prediction of the saved model of the article's fold. X: the texts (TF-IDF) or the frozen vectors."""
    p = np.full(len(S), np.nan)
    for k in range(1, 6):
        i = np.where(S.fold.to_numpy() == k)[0]
        if len(i):
            b = fold_model(name, k)
            p[i] = b["model"].predict(b["transformer"].transform([X[j] for j in i] if isinstance(X, list) else X[i]))
    return p


def sample_predictions(S, gpu_gb=19, logfile=None):
    """Original and cleaned predictions of the three models on the sample (frozen vectors cached in boilerplate/)."""
    f = DIR / "sample_predictions.csv"
    if f.exists():
        return pd.read_csv(f)
    out = S[["article_id", "domain", "fold", "score", "share"]].copy()
    for name in ["TF-IDF", "Qwen3-8B", "Mistral-24B"]:
        p = MO.predictions("regression", name).set_index("article_id").prediction
        out[f"{name} original"] = p.reindex(S.article_id).to_numpy()
    out["TF-IDF clean"] = predict_saved("TF-IDF", S, list(S.clean))
    q = DIR / "sample_qwen.npy"
    if not q.exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        np.save(q, MO.qwen_embeddings(list(S.clean), "Qwen/Qwen3-Embedding-8B", 16384, logfile))
    out["Qwen3-8B clean"] = predict_saved("Qwen3-8B", S, np.load(q))
    m = DIR / "sample_mistral.npy"
    if not m.exists():
        MO.wait_for_gpu(gpu_gb, logfile)
        np.save(m, MO.mistral_embeddings(list(S.clean), MISTRAL, [27], 8192, logfile, DIR / "sample_mistral_partial.npy")[27])
    out["Mistral-24B clean"] = predict_saved("Mistral-24B", S, np.load(m))
    out.to_csv(f, index=False)
    return out


# ------------------------------------------------------------------------- 4. Mistral token by token
def mistral_tokens(S, bp, n=300, gpu_gb=19, logfile=None):
    """Per article: the prediction rebuilt from the token contributions, the part of the boilerplate tokens and their
    share of the tokens (cached in boilerplate/)."""
    f = DIR / "mistral_tokens.csv"
    if f.exists():
        return pd.read_csv(f)
    import torch
    MO.wait_for_gpu(gpu_gb, logfile)
    tok, m = MO.load_4bit(MISTRAL, "right")
    saved = MO.predictions("regression", "Mistral-24B").set_index("article_id").prediction
    rows = []
    with torch.no_grad():
        for r in S.iloc[:n].itertuples():
            b = fold_model("Mistral-24B", r.fold)
            w = b["model"].coef_ / b["transformer"].scale_
            const = float(b["model"].intercept_ - (b["transformer"].mean_ * w).sum())
            enc = tok(r.text, return_offsets_mapping=True, truncation=True, max_length=MO.MAX_TOKENS, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].numpy()
            out = m.model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
                          output_hidden_states=True)
            H = out.hidden_states[27][0].float().cpu().numpy()
            c = H @ w / len(H)                                  # one contribution per token; prediction = const + sum
            lines = r.text.split("\n")
            line_start = np.cumsum([0] + [len(x) + 1 for x in lines])[:-1]
            line_of = np.searchsorted(line_start, offsets[:, 0], side="right") - 1
            bp_token = np.array([is_boilerplate(r.domain, lines[j], bp) for j in line_of])
            bp_token[0] = False                                 # the BOS token
            rows.append({"article_id": r.article_id, "saved": saved[r.article_id], "prediction": const + c.sum(),
                         "boilerplate part": c[bp_token].sum(), "boilerplate tokens": bp_token.mean()})
            del out
    del m; torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(f, index=False)
    C.log(f"boilerplate: Mistral token by token on {len(out)} articles", logfile)
    return out

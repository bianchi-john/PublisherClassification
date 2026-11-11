"""Step 11 (added 8/11): meta-articles. Does a model learn the outlet better if every training example is a text that
gathers many articles of ONE outlet, every outlet with the same number of examples?

Idea (John): with one example per article, outlets with many articles are seen many times and the model learns them by
heart; a new outlet should teach more than one already seen. A meta-article puts many articles of the same outlet in one
text, and every outlet gets the same number of meta-articles, so the training is balanced over outlets and the model reads
the articles of an outlet together.

**Meta-article** (fixed before the results): the lead (first `LEAD_WORDS` words) of articles of one outlet taken in random
order, separated by a blank line, until `META_WORDS` words (about 3,200 tokens, below the 4,096 limit of every model).
`N_META` meta-articles per outlet, each with its own random order; they use only the outlet's own articles and never
look at a label. Training outlets and test outlets are built the same way.

**Models** (the main ones of steps 6-10), same outer folds, inner folds and inner criterion (Spearman per outlet):
- TF-IDF + ridge on the meta-articles (the vectorizer of step 7, fitted on the training meta-articles);
- Qwen3-8B and Mistral-24B (layer 27), frozen vectors of the meta-articles, + ridge and + the small neural network of
  step 10 (`boost.py`);
- ModernBERT fine-tuned on meta-articles, set up for this case:
  - every epoch NEW meta-articles (`BERT_PER_EPOCH` per training outlet, freshly sampled): balanced and never the
    same text twice;
  - mean pooling (better for long texts than the first token), Huber loss on (score − 50) / 25;
  - lr 3e-5 with 10% warm-up and linear decay, weight decay 0.01, effective batch 16, at most `BERT_EPOCHS` epochs;
  - early stopping: 20% of the training outlets (one of 5 score-stratified groups) are kept apart and the epoch with the
    best Spearman on them is kept; the test fold is never used.
The outlet score is the mean over its meta-articles (all 10; also 1, 2, 3, 5 random ones for the curves).
Outputs in Dataset/tentativi/20_FromScratch/meta/: meta_texts.csv.gz, embeddings/, predictions/<model>/fold<k>.csv.gz
(one row per test meta-article) and inner/early-stopping tables. Run: run_meta.sh."""
import time

import numpy as np
import pandas as pd

import boost as BO
import models as MO20
from common import OUT as SOURCE
from common import SEED, log

OUT = SOURCE / "meta"
PRED = OUT / "predictions"
LOG = OUT / "log.txt"
LEAD_WORDS, META_WORDS, N_META = 120, 2400, 10
QWEN, MISTRAL, BERT = "Qwen/Qwen3-Embedding-8B", "mistralai/Mistral-Small-24B-Base-2501", "answerdotai/ModernBERT-base"
EMB_FILES = {"Qwen3-8B": "meta_qwen3_embedding_8b.npy", "Mistral-24B": "meta_mistral_small_24b_layer27.npy"}
BERT_PER_EPOCH, BERT_EPOCHS, BERT_LR, BERT_BATCH, BERT_ACCUMULATE = 6, 10, 3e-5, 2, 8
K_META = [1, 2, 3, 5, "all"]


# ------------------------------------------------------------------------- meta-articles
def leads(art):
    """The lead of every article, grouped by outlet."""
    lead = art.text.fillna("").map(lambda t: " ".join(t.split()[:LEAD_WORDS]))
    return lead.groupby(art.domain.to_numpy()).apply(list)


def one_meta(pieces, rng):
    """Leads of one outlet in random order until META_WORDS words (all of them if fewer)."""
    out, words = [], 0
    for i in rng.permutation(len(pieces)):
        out.append(pieces[i]); words += len(pieces[i].split())
        if words >= META_WORDS:
            break
    return "\n\n".join(out), len(out), words


def build(art, outlets):
    """N_META meta-articles per outlet (seeded per outlet), with score and fold of the outlet."""
    L = leads(art)
    rows = []
    for i, d in enumerate(sorted(outlets.index)):
        rng = np.random.default_rng([SEED, i])
        for j in range(N_META):
            text, n, w = one_meta(L[d], rng)
            rows.append({"domain": d, "meta": j, "text": text, "articles": n, "words": w})
    M = pd.DataFrame(rows)
    M["score"] = M.domain.map(outlets.score).astype(float)
    M["label"] = M.domain.map(outlets.label)
    M["fold"] = M.domain.map(outlets.fold)
    M["article_id"] = np.arange(len(M))                   # boost.py treats a meta-article as an "article"
    return M


def load_meta():
    M = pd.read_csv(OUT / "meta_texts.csv.gz")
    rows = pd.Series(np.arange(len(M))).groupby(M.domain.to_numpy()).apply(np.array)
    return M, rows


def save_predictions(name, fold, M, te, p, pred_dir=None):
    folder = (pred_dir or PRED) / name; folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"domain": M.domain.to_numpy()[te], "meta": M.meta.to_numpy()[te], "fold": fold, "prediction": p}) \
        .to_csv(folder / f"fold{fold}.csv.gz", index=False)


# ------------------------------------------------------------------------- frozen embeddings + boost methods
def embed(name, M):
    f = OUT / "embeddings" / EMB_FILES[name]
    if f.exists():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    MO20.wait_for_gpu(22, LOG)
    texts = M.text.tolist()
    if name == "Qwen3-8B":
        np.save(f, MO20.qwen_embeddings(texts, QWEN, 8192, LOG))          # long texts: small batches
    else:
        E = MO20.mistral_embeddings(texts, MISTRAL, [27], 4096, LOG, OUT / "embeddings" / "mistral_partial.npy")
        np.save(f, E[27]); (OUT / "embeddings" / "mistral_partial.npy").unlink(missing_ok=True)
    log(f"meta embeddings {name}: done", LOG)


def fit_embedding(name, method, fold, M, rows, outlets):
    """boost.py method (ridge or mlp) on the meta-article vectors: setting chosen on the inner folds."""
    folder = PRED / f"{name} · {method}"
    if (folder / f"fold{fold}.csv.gz").exists():
        return
    start = time.time()
    X = np.load(OUT / "embeddings" / EMB_FILES[name]).astype(np.float32)
    train_outlets = outlets.index[outlets.fold != fold]
    scores = BO.inner_scores(method, X, M, outlets, rows, train_outlets, fold)
    i = scores["inner Spearman"].idxmax()
    best = {k: v for k, v in scores.loc[i].drop("inner Spearman").to_dict().items() if not (isinstance(v, float) and np.isnan(v))}
    for k in ("epoch", "hidden"):
        if k in best:
            best[k] = int(best[k])
    F = BO.Fold(X, M, outlets, rows, train_outlets)
    predict = BO.final_predictor(method, best, F, fold)
    te = np.where(M.fold.to_numpy() == fold)[0]
    p = predict([np.array([r]) for r in te])               # one prediction per test meta-article
    save_predictions(f"{name} · {method}", fold, M, te, p)
    scores.assign(fold=fold, chosen=scores.index == i).to_csv(folder / f"inner_fold{fold}.csv", index=False)
    log(f"meta {name} · {method}, fold {fold}: {best} inner {scores['inner Spearman'].max():.3f}, {time.time() - start:.0f} s", LOG)


# ------------------------------------------------------------------------- TF-IDF
def fit_tfidf(fold, M, outlets):
    """TF-IDF of step 7 + ridge on the meta-articles; alpha chosen on the 3 inner folds (Spearman per outlet)."""
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    folder = PRED / "TF-IDF"
    if (folder / f"fold{fold}.csv.gz").exists():
        return
    start = time.time()
    texts, y, dom = M.text.to_numpy(), M.score.to_numpy(), M.domain.to_numpy()
    vec = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000,
                                  strip_accents="unicode")
    train_outlets = outlets.index[outlets.fold != fold]
    inner = MO20.score_split(outlets.score.reindex(train_outlets), SEED + fold, k=3)
    grid = (0.0001, 0.0003) + MO20.GRIDS[("regression", "TF-IDF")] + (3, 10)
    res = {a: [] for a in grid}
    for j in (1, 2, 3):
        a_tr = np.where(pd.Series(dom).map(inner).to_numpy() != j)[0]
        a_tr = a_tr[pd.Series(dom[a_tr]).isin(train_outlets).to_numpy()]
        a_va = np.where(pd.Series(dom).map(inner).to_numpy() == j)[0]
        v = vec(); A, B = v.fit_transform(texts[a_tr]), v.transform(texts[a_va])
        for a in grid:
            p = pd.Series(Ridge(alpha=a, random_state=SEED).fit(A, y[a_tr]).predict(B)).groupby(dom[a_va]).mean()
            res[a].append(spearmanr(outlets.score.reindex(p.index), p).correlation)
    means = pd.Series({a: np.mean(s) for a, s in res.items()})
    best = means.idxmax()
    tr, te = np.where(M.fold.to_numpy() != fold)[0], np.where(M.fold.to_numpy() == fold)[0]
    v = vec(); A, B = v.fit_transform(texts[tr]), v.transform(texts[te])
    save_predictions("TF-IDF", fold, M, te, Ridge(alpha=best, random_state=SEED).fit(A, y[tr]).predict(B))
    pd.DataFrame({"fold": fold, "alpha": means.index, "inner Spearman": means.values, "chosen": means.index == best}) \
        .to_csv(folder / f"inner_fold{fold}.csv", index=False)
    log(f"meta TF-IDF, fold {fold}: alpha {best} inner {means.max():.3f}, {time.time() - start:.0f} s", LOG)


# ------------------------------------------------------------------------- ModernBERT
def fit_bert(fold, M, art, outlets, extra_art=None, extra_scores=None, pred_dir=None):
    """ModernBERT fine-tuned on meta-articles (see the module docstring for every choice). Optional extra training
    outlets (`extra_art`, `extra_scores`) are always in training, never in the validation or the test."""
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    folder = (pred_dir or PRED) / "ModernBERT"; folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"fold{fold}.csv.gz").exists():
        return
    MO20.wait_for_gpu(16, LOG if pred_dir is None else pred_dir.parent / "log.txt")
    MO20.reproducible(SEED + fold)
    start = time.time()
    rng = np.random.default_rng(SEED + 100 * fold)
    train_outlets = outlets.index[outlets.fold != fold]
    groups = MO20.score_split(outlets.score.reindex(train_outlets), SEED + fold, k=5)
    val_outlets, fit_outlets = groups.index[groups == 1], list(groups.index[groups != 1])
    scores = outlets.score
    if extra_art is not None:
        fit_outlets += list(extra_scores.index); scores = pd.concat([outlets.score, extra_scores])
        art = pd.concat([art[["domain", "text"]], extra_art[["domain", "text"]]], ignore_index=True)
    L = leads(art)
    tok = AutoTokenizer.from_pretrained(BERT)
    dev = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=1, reference_compile=False,
                                                               classifier_pooling="mean").to(dev)
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": 0.01},
                             {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}], lr=BERT_LR)
    per_epoch = len(fit_outlets) * BERT_PER_EPOCH
    steps = BERT_EPOCHS * int(np.ceil(per_epoch / (BERT_BATCH * BERT_ACCUMULATE)))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")

    def encode(texts):
        enc = tok(list(texts), truncation=True, max_length=MO20.MAX_TOKENS, padding=True, return_tensors="pt")
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev)

    def predict(texts):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(texts), 4):
                x, m = encode(texts[s:s + 4])
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(model(input_ids=x, attention_mask=m).logits.float()[:, 0].cpu().numpy())
        return np.concatenate(out) * 25 + 50

    va = np.where(M.domain.isin(val_outlets).to_numpy())[0]
    curve, best, best_state = [], -np.inf, None
    for e in range(1, BERT_EPOCHS + 1):
        model.train(); opt.zero_grad()
        texts, ys = [], []
        for d in fit_outlets:                                    # fresh meta-articles, the same number per outlet
            for _ in range(BERT_PER_EPOCH):
                texts.append(one_meta(L[d], rng)[0]); ys.append(scores[d])
        order = rng.permutation(len(texts)); total = 0.0
        for b, s in enumerate(range(0, len(order), BERT_BATCH), 1):
            idx = order[s:s + BERT_BATCH]
            x, m = encode([texts[i] for i in idx])
            t = torch.tensor((np.array(ys)[idx] - 50) / 25, dtype=torch.float32, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=x, attention_mask=m).logits.float()[:, 0]
            loss = torch.nn.functional.smooth_l1_loss(z, t)
            scaler.scale(loss / BERT_ACCUMULATE).backward(); total += loss.item() * len(idx)
            if b % BERT_ACCUMULATE == 0 or s + BERT_BATCH >= len(order):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        pv = pd.Series(predict(M.text.to_numpy()[va])).groupby(M.domain.to_numpy()[va]).mean()
        rho = spearmanr(outlets.score.reindex(pv.index), pv).correlation
        curve.append({"epoch": e, "loss": total / len(order), "validation Spearman": rho})
        if rho > best:
            best, best_state = rho, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"meta ModernBERT, fold {fold}, epoch {e}/{BERT_EPOCHS}: loss {total / len(order):.4f}, validation Spearman "
            f"{rho:.3f} ({time.time() - start:.0f} s)", LOG)
    model.load_state_dict(best_state)
    te = np.where(M.fold.to_numpy() == fold)[0]
    save_predictions("ModernBERT", fold, M, te, predict(M.text.to_numpy()[te]), pred_dir)
    pd.DataFrame(curve).assign(fold=fold, chosen=lambda d: d["validation Spearman"] == best) \
        .to_csv(folder / f"curve_fold{fold}.csv", index=False)
    log(f"meta ModernBERT, fold {fold}: best epoch {int(np.argmax([c['validation Spearman'] for c in curve])) + 1}, "
        f"done in {time.time() - start:.0f} s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", LOG)
    del model, opt; torch.cuda.empty_cache()


# ------------------------------------------------------------------------- evaluation
MODELS = ["TF-IDF", "Qwen3-8B · ridge", "Qwen3-8B · mlp", "Mistral-24B · ridge", "Mistral-24B · mlp", "ModernBERT"]
ARTICLE_REFERENCE = {"TF-IDF": "TF-IDF", "Qwen3-8B · ridge": "Qwen3-8B · ridge", "Qwen3-8B · mlp": "Qwen3-8B · mlp",
                     "Mistral-24B · ridge": "Mistral-24B · ridge", "Mistral-24B · mlp": "Mistral-24B · mlp",
                     "ModernBERT": "ModernBERT"}


def meta_matrix(name, outlets, k):
    """Draws × outlets: outlet score = mean of all its meta-articles (k = "all") or of k random ones (20 draws)."""
    P = pd.concat([pd.read_csv(f) for f in sorted((PRED / name).glob("fold*.csv.gz"))])
    g = P.groupby("domain").prediction.apply(np.array).reindex(outlets.index)
    if k == "all":
        return np.array([[p.mean() for p in g]])
    rng = np.random.default_rng(SEED + 7)
    return np.array([[rng.choice(p, size=k, replace=False).mean() for p in g] for _ in range(BO.DRAWS)])


def article_matrix(name, outlets, art, rows):
    """The same model on whole articles (steps 7 and 10), all articles."""
    if name == "ModernBERT":
        return MO20.predictions("regression", "ModernBERT").groupby("domain").prediction.mean().reindex(outlets.index).to_numpy()[None, :]
    if name == "TF-IDF":
        return MO20.predictions("regression", "TF-IDF").groupby("domain").prediction.mean().reindex(outlets.index).to_numpy()[None, :]
    emb, method = name.split(" · ")
    P = pd.concat([pd.read_csv(f, dtype={"k": str}) for f in sorted((BO.PRED / emb / method).glob("fold*.csv.gz"))])
    return P[P.k == "all"].set_index("domain").prediction.reindex(outlets.index).to_numpy()[None, :]

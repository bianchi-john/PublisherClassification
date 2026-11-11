"""Steps 4-6: the outlet split, the frozen text representations, and the models for classification (5 levels) and
regression (exact NewsGuard score 0-100). Both tasks use the same split and the same models, each able to read every
article in full:
- TF-IDF + linear model (no length limit);
- Qwen3-Embedding-8B, frozen, + linear model;
- Mistral-Small-24B (Base), frozen, + linear model;
- ModernBERT, fully trained (up to 8,192 tokens).
For every article of every fold we save the logits (classification) or the predicted score (regression), so every
metric can be recomputed; the model of every fold is saved too.
"""
import time

import numpy as np
import pandas as pd

from common import EMBEDDINGS, LEVELS, MODELS_DIR, OUT, PREDICTIONS, SEED, majority

LOGITS = [f"logit_{l}" for l in LEVELS]
PROBS = [f"p_{l}" for l in LEVELS]
MAX_TOKENS = 4096                     # above the longest article for every tokenizer (checked in step 5)
QWEN_INSTRUCTION = ("Represent this news article to judge the reliability and journalistic standards of the outlet "
                    "that published it")
GRIDS = {                             # penalties tried with inner cross-validation
    ("classification", "TF-IDF"): (10, 100, 1_000, 10_000, 100_000),          # C of the logistic regression
    ("classification", "embedding"): (0.0001, 0.001, 0.01, 0.1, 1, 10),
    ("regression", "TF-IDF"): (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1),        # alpha of the ridge
    ("regression", "embedding"): (10, 100, 1_000, 10_000, 100_000, 1_000_000),
}
EMBEDDING_FILES = {"Qwen3-8B": "qwen3_embedding_8b.npy", "Mistral-24B": "mistral_small_24b_layer27.npy"}


# ------------------------------------------------------------------------- data and split
def load():
    """The anonymous articles with the fold of their outlet, and the outlets."""
    art = pd.read_csv(OUT / "articles_anonymous.csv.gz")
    outlets = pd.read_csv(OUT / "outlets.csv", index_col="domain")
    outlets["fold"] = pd.read_csv(OUT / "split.csv", index_col="domain").fold.reindex(outlets.index)
    art["fold"] = art.domain.map(outlets.fold)
    return art, outlets


def target(art, task):
    return art.label.to_numpy() if task == "classification" else art.score.to_numpy().astype(float)


def score_split(scores, seed, k=5):
    """Folds stratified on the exact score: outlets sorted by score (ties in random order) and, in blocks of k
    consecutive outlets, each one goes to a different fold. Every fold gets the same score distribution."""
    rng = np.random.default_rng(seed)
    order = scores.index[np.lexsort((rng.random(len(scores)), scores.to_numpy()))]
    fold = pd.Series(0, index=scores.index)
    for i in range(0, len(order), k):
        block = order[i:i + k]
        fold[block] = rng.permutation(np.arange(1, k + 1))[:len(block)]
    return fold


# ------------------------------------------------------------------------- reproducibility on the GPU
def reproducible(seed):
    """Fix the seeds of random, numpy and torch, and ask torch for deterministic algorithms."""
    import random

    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def wait_for_gpu(gb, logfile):
    """Wait until the shared GPU has at least `gb` GB free."""
    import torch
    from common import log
    while torch.cuda.mem_get_info()[0] / 2**30 < gb:
        log(f"GPU: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free, {gb} needed: waiting 5 minutes", logfile)
        time.sleep(300)


# ------------------------------------------------------------------------- frozen text representations
def count_tokens(texts, model, prefix=""):
    """Number of tokens of every article for the model's tokenizer (special tokens and instruction included)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return np.array([len(x) for x in tok([prefix + t for t in texts], add_special_tokens=True)["input_ids"]])


def batches_by_tokens(lengths, budget, max_size):
    """Batches from the shortest to the longest article, each with at most `budget` tokens (longest × size) and at
    most `max_size` articles."""
    batches, current = [], []
    for i in np.argsort(lengths, kind="stable"):
        if (current and (len(current) + 1) * lengths[i] > budget) or len(current) >= max_size:
            batches.append(current); current = []
        current.append(i)
    return batches + ([current] if current else [])


def qwen_embeddings(texts, model, budget, logfile, max_size=16):
    """Qwen3-Embedding: instruction + full article; vector = last token (left padding), L2-normalised."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from common import log
    reproducible(SEED)
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    m = AutoModel.from_pretrained(model, torch_dtype=torch.float16).cuda().eval()
    texts = [f"Instruct: {QWEN_INSTRUCTION}\nQuery:{x}" for x in texts]
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
    E = None
    with torch.no_grad():
        for b, idx in enumerate(batches_by_tokens(lengths, budget, max_size)):
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS + 64,
                      return_tensors="pt").to("cuda")
            h = F.normalize(m(**enc).last_hidden_state[:, -1].float(), p=2, dim=1).cpu().numpy()
            if E is None:
                E = np.zeros((len(texts), h.shape[1]), dtype=np.float32)
            E[idx] = h
            if b % 200 == 0:
                log(f"Qwen: batch {b}", logfile)
    del m; torch.cuda.empty_cache()
    return E


def mistral_embeddings(texts, model, layers, budget, logfile, partial_file, max_size=8, save_every=100):
    """Mistral-Small-24B Base in 4 bit (NF4, fp16 compute): per layer, the mean of the hidden states of the article
    tokens (no padding). Resumes from the saved partial work if interrupted."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from common import log
    reproducible(SEED)
    tok = AutoTokenizer.from_pretrained(model)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant, torch_dtype=torch.float16,
                                             device_map={"": 0}).eval()
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
    E, done = None, np.zeros(len(texts), bool)
    if partial_file.exists():
        p = np.load(partial_file, allow_pickle=True).item()
        if p["n"] == len(texts):
            E, done = p["E"], p["done"]; log(f"Mistral: resuming, {done.sum():,} articles already done", logfile)
    with torch.no_grad():
        for b, idx in enumerate(batches_by_tokens(lengths, budget, max_size)):
            if done[idx].all():
                continue
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt").to("cuda")
            out = m.model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            if E is None:
                E = {l: np.zeros((len(texts), out.hidden_states[l].shape[-1]), dtype=np.float32) for l in layers}
            for l in layers:
                E[l][idx] = ((out.hidden_states[l].float() * mask).sum(1) / mask.sum(1)).cpu().numpy()
            done[idx] = True
            del out
            if b % save_every == 0:
                log(f"Mistral: {done.sum():,}/{len(texts):,} articles", logfile)
                np.save(partial_file, {"n": len(texts), "E": E, "done": done}, allow_pickle=True)
    del m; torch.cuda.empty_cache()
    return E


def load_embeddings(name, art):
    """The frozen vectors of a model, checking that their rows follow the article order."""
    order = pd.read_csv(EMBEDDINGS / "article_order.csv").article_id.to_numpy()
    assert (order == art.article_id.to_numpy()).all()
    return np.load(EMBEDDINGS / EMBEDDING_FILES[name])


# ------------------------------------------------------------------------- linear models
def _fit(task, penalty, X, y):
    """Multinomial logistic regression with balanced classes (classification) or ridge (regression)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    if task == "classification":
        return LogisticRegression(C=penalty, max_iter=3000, class_weight="balanced", random_state=SEED).fit(X, y)
    return Ridge(alpha=penalty, random_state=SEED).fit(X, y)


def _outputs(task, m, X):
    """Classification: the 5 logits; regression: the predicted score."""
    return m.decision_function(X) if task == "classification" else m.predict(X)


def _inner_score(task, outputs, domains, y):
    """How a penalty is judged, per outlet: macro F1 of the majority level (classification) or Spearman between
    true score and mean predicted score (regression)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    if task == "classification":
        g = pd.DataFrame({"d": domains, "y": y, "p": outputs.argmax(1)}).groupby("d").agg(y=("y", "first"), p=("p", majority))
        return f1_score(g.y, g.p, average="macro", labels=range(5), zero_division=0)
    g = pd.DataFrame({"d": domains, "y": y, "p": outputs}).groupby("d").mean()
    return spearmanr(g.y, g.p).correlation


def _matrices(name, tr, te, texts, X):
    """Rows tr (training) and te (prediction) ready for the linear model. TF-IDF: vocabulary and idf from the training
    texts only. Embeddings: mean 0 and variance 1 from the training rows only. Returns the fitted transformer too."""
    if name == "TF-IDF":
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000,
                            strip_accents="unicode")
        return v.fit_transform(texts[tr]), v.transform(texts[te]), v
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(X[tr])
    return s.transform(X[tr]), s.transform(X[te]), s


def _try_penalty(task, name, penalty, tr, va, texts, X, y, domains):
    A, B, _ = _matrices(name, tr, va, texts, X)
    return _inner_score(task, _outputs(task, _fit(task, penalty, A, y[tr]), B), domains[va], y[va])


def _save_outputs(task, art, te, k, U):
    """One row per test article: logits and probabilities (classification) or predicted score (regression)."""
    out = pd.DataFrame({"article_id": art.article_id.to_numpy()[te], "domain": art.domain.to_numpy()[te], "fold": k})
    if task == "classification":
        out[LOGITS] = U
        P = np.exp(U - U.max(1, keepdims=True)); out[PROBS] = P / P.sum(1, keepdims=True)
        out["prediction"] = U.argmax(1)
    else:
        out["prediction"] = U
    return out


def linear_fold(task, name, art, k, X=None, n_jobs=16, logfile=None):
    """One outer fold of a linear model."""
    import joblib
    from joblib import Parallel, delayed

    from common import log
    folder = PREDICTIONS / task / name; folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"fold{k}.csv.gz").exists():
        log(f"{task}, {name}, fold {k}: already done", logfile); return
    start = time.time()
    y, domains = target(art, task), art.domain.to_numpy()
    texts = art.text.fillna("").to_numpy() if name == "TF-IDF" else None
    tr, te = np.where(art.fold.to_numpy() != k)[0], np.where(art.fold.to_numpy() == k)[0]
    # 1. three inner folds of the training outlets, split by score like the outer split
    inner = score_split(art.score.iloc[tr].groupby(domains[tr]).first(), SEED + k, k=3)
    fi = pd.Series(domains[tr]).map(inner).to_numpy()
    # 2. every penalty on every inner fold; keep the best mean
    grid = GRIDS[(task, "TF-IDF" if name == "TF-IDF" else "embedding")]
    jobs = [(p, tr[fi != j], tr[fi == j]) for p in grid for j in (1, 2, 3)]
    scores = Parallel(n_jobs=n_jobs)(delayed(_try_penalty)(task, name, p, a, b, texts, X, y, domains) for p, a, b in jobs)
    means = pd.DataFrame({"penalty": [p for p, _, _ in jobs], "score": scores}).groupby("penalty").score.mean()
    best = means.idxmax()
    # 3. the model with the chosen penalty on all the training outlets; outputs on the test fold
    Atr, Ate, prep = _matrices(name, tr, te, texts, X)
    m = _fit(task, best, Atr, y[tr])
    _save_outputs(task, art, te, k, _outputs(task, m, Ate)).to_csv(folder / f"fold{k}.csv.gz", index=False)
    pd.DataFrame({"fold": k, "penalty": means.index, "inner score": means.values, "chosen": means.index == best}) \
        .to_csv(folder / f"penalty_fold{k}.csv", index=False)
    dest = MODELS_DIR / task / name; dest.mkdir(parents=True, exist_ok=True)
    joblib.dump({"transformer": prep, "model": m, "penalty": best}, dest / f"fold{k}.joblib", compress=3)
    edge = "on the grid edge" if best in (grid[0], grid[-1]) else "inside the grid"
    log(f"{task}, {name}, fold {k}: penalty {best} ({edge}), {time.time() - start:.0f} s", logfile)


# ------------------------------------------------------------------------- ModernBERT, fully trained
def bert_fold(task, art, k, model_name, name, logfile, max_tokens=MAX_TOKENS, epochs=5, lr=3e-5, batch=4,
              effective_batch=32, gpu_gb=16):
    """One fold: the transformer is fully trained for 5 epochs, no validation (the last epoch is kept).
    Classification: cross-entropy with balanced classes and label smoothing 0.1. Regression: squared error on
    score/100, the output is multiplied back by 100."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from common import log
    folder = PREDICTIONS / task / name; folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"fold{k}.csv.gz").exists():
        log(f"{task}, {name}, fold {k}: already done", logfile); return
    wait_for_gpu(gpu_gb, logfile)
    reproducible(SEED + k)
    tok = AutoTokenizer.from_pretrained(model_name)
    ids = tok(art.text.fillna("").tolist(), truncation=True, max_length=max_tokens)["input_ids"]
    assert all(len(x) < max_tokens for x in ids), "some article reaches the token limit"
    cls = task == "classification"
    y = target(art, task) if cls else target(art, task) / 100
    tr, te = np.where(art.fold.to_numpy() != k)[0], np.where(art.fold.to_numpy() == k)[0]
    device = torch.device("cuda")

    def batches(rows, shuffle):
        """Batches of similar length (less padding), in random order when training."""
        order = np.array(sorted(rows, key=lambda i: len(ids[i])))
        groups = [order[i:i + batch] for i in range(0, len(order), batch)]
        if shuffle:
            np.random.shuffle(groups)
        return groups

    def tensors(group):
        L = max(len(ids[i]) for i in group)
        x = np.full((len(group), L), tok.pad_token_id, dtype=np.int64); mask = np.zeros_like(x)
        for r, i in enumerate(group):
            x[r, :len(ids[i])] = ids[i]; mask[r, :len(ids[i])] = 1
        return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5 if cls else 1,
                                                               reference_compile=False).to(device)
    if cls:
        counts = np.bincount(y[tr], minlength=5).astype(float)
        weights = torch.tensor(counts.sum() / (5 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": 0.01},
                             {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}], lr=lr)
    accumulate = max(1, effective_batch // batch)
    steps = epochs * int(np.ceil(len(tr) / (batch * accumulate)))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    curve, start = [], time.time()
    for e in range(1, epochs + 1):
        model.train(); opt.zero_grad(); total, n = 0.0, 0
        groups = batches(tr, True)
        for b, g in enumerate(groups, 1):
            x, mask = tensors(g)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=x, attention_mask=mask).logits.float()
            if cls:
                loss = loss_fn(z, torch.tensor(y[g], device=device))
            else:
                loss = ((z[:, 0] - torch.tensor(y[g], dtype=torch.float32, device=device)) ** 2).mean()
            scaler.scale(loss / accumulate).backward(); total += loss.item() * len(g); n += len(g)
            if b % accumulate == 0 or b == len(groups):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        curve.append({"epoch": e, "loss": total / n})
        log(f"{task}, {name}, fold {k}, epoch {e}/{epochs}: loss {total / n:.4f} ({time.time() - start:.0f} s)", logfile)
    model.eval(); U = np.zeros((len(te), 5 if cls else 1), dtype=np.float32); pos = {i: j for j, i in enumerate(te)}
    with torch.no_grad():
        for g in batches(te, False):
            x, mask = tensors(g)
            with torch.autocast("cuda", dtype=torch.float16):
                U[[pos[i] for i in g]] = model(input_ids=x, attention_mask=mask).logits.float().cpu().numpy()
    dest = MODELS_DIR / task / name / f"fold{k}"; dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest); tok.save_pretrained(dest)
    _save_outputs(task, art, te, k, U if cls else U[:, 0] * 100).to_csv(folder / f"fold{k}.csv.gz", index=False)
    pd.DataFrame(curve).to_csv(folder / f"loss_fold{k}.csv", index=False)
    log(f"{task}, {name}, fold {k}: done in {time.time() - start:.0f} s, GPU peak "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", logfile)
    del model, opt; torch.cuda.empty_cache()


def predictions(task, name):
    """All the article outputs of a model (5 folds), or None if some fold is missing."""
    files = sorted((PREDICTIONS / task / name).glob("fold*.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if len(files) == 5 else None

"""Tentativo 19: classificazione (5 livelli) e regressione (voto esatto 0-100) sul dataset anonimo, con lo stesso split.

Un solo split per i due compiti, stratificato sul voto esatto (il primo split del tentativo 18): è bilanciato anche per
livello (al più una testata di differenza tra i fold) e permette il confronto a coppie tra classificatore e regressore.

Gli stessi modelli per i due compiti, ognuno con un contesto abbastanza lungo da leggere ogni articolo intero:
- TF-IDF + modello lineare (nessun limite di lunghezza);
- Qwen3-Embedding-8B congelato + modello lineare;
- Mistral-Small-24B (Base) congelato + modello lineare;
- ModernBERT addestrato tutto (fino a 8.192 token).
Per ogni articolo di ogni fold si salvano i logit (classificazione) o il voto previsto (regressione), così ogni misura si
può ricalcolare; si salvano anche i modelli di ogni fold. Il codice riprende quello dei tentativi 16 e 18.

Riproducibilità: un seme per tutto (SEME), lo split generato una volta sola (notebook 2), `random_state` in ogni modello,
e sulla GPU algoritmi deterministici (`riproducibile`).
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from anonimizza import DATASET, DATI

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # serve a cuBLAS per i calcoli deterministici sulla GPU

MODELLI = DATI.parent / "Modelli" / "19_DatasetAnonimo"     # i modelli addestrati di ogni fold (fuori da git)
PREVISIONI = DATASET / "previsioni"                          # logit e voti previsti per articolo (fuori da git)
EMBEDDING = DATASET / "embedding"                            # i vettori dei modelli congelati
SPLIT = "split.csv"                                          # un fold (1-5) per testata, uguale per i due compiti
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
LOGIT = [f"logit_{l}" for l in LIVELLI]
PROB = [f"p_{l}" for l in LIVELLI]
SEME = 42                            # il seme di tutto: modelli, fold interni, lotti, bootstrap
SEME_SPLIT = 1                       # il seme dello split (lo stesso del primo split del tentativo 18)
MAX_TOKEN = 4096                     # più dell'articolo più lungo per ogni tokenizer (controllato nel notebook 3)
ISTRUZIONE_QWEN = ("Represent this news article to judge the reliability and journalistic standards of the outlet that "
                   "published it")                           # la stessa del tentativo 16
GRIGLIE = {                          # le penalità provate; quelle del tentativo 18 cadevano sul bordo, qui c'è un ordine in più
    ("classificazione", "TF-IDF"): (10, 100, 1_000, 10_000, 100_000),                 # C della logistica
    ("classificazione", "embedding"): (0.0001, 0.001, 0.01, 0.1, 1, 10),
    ("regressione", "TF-IDF"): (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1),               # alpha della ridge
    ("regressione", "embedding"): (10, 100, 1_000, 10_000, 100_000, 1_000_000),
}


# ===================================================================================== 1. i dati
def leggi():
    """Gli articoli anonimi con il fold della loro testata, e le testate (lo split è lo stesso per i due compiti)."""
    art = pd.read_csv(DATASET / "articoli.csv.gz")
    t = pd.read_csv(DATASET / "testate.csv", index_col="domain")
    t["fold"] = pd.read_csv(DATASET / SPLIT, index_col="domain").fold.reindex(t.index)
    art["fold"] = art.domain.map(t.fold)
    return art, t


def bersaglio(art, compito):
    return art.label.to_numpy() if compito == "classificazione" else art.score.to_numpy().astype(float)


def crea_split(punteggi, seme, k=5):
    """Fold stratificati sul voto continuo (tentativo 18): le testate in ordine di voto (a parità a caso) e, a blocchi
    di k consecutive, ognuna in un fold diverso."""
    rng = np.random.default_rng(seme)
    ordine = punteggi.index[np.lexsort((rng.random(len(punteggi)), punteggi.to_numpy()))]
    fold = pd.Series(0, index=punteggi.index)
    for i in range(0, len(ordine), k):
        blocco = ordine[i:i + k]
        fold[blocco] = rng.permutation(np.arange(1, k + 1))[:len(blocco)]
    return fold


def riproducibile(seme):
    """Fissa il seme di random, numpy e torch, e chiede a torch algoritmi deterministici sulla GPU."""
    import random

    import torch
    random.seed(seme); np.random.seed(seme); torch.manual_seed(seme); torch.cuda.manual_seed_all(seme)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def voto(valori):
    """Voto di maggioranza; a parità vince il livello più basso (come nei tentativi 16 e 17)."""
    c = pd.Series(valori).value_counts()
    return int(c[c == c.max()].index.min())


# ===================================================================================== 2. lunghezze e embedding
def conta_token(testi, modello, prefisso=""):
    """Quanti token ha ogni articolo per il tokenizer del modello (con i token speciali e l'eventuale istruzione)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modello)
    return np.array([len(x) for x in tok([prefisso + t for t in testi], add_special_tokens=True)["input_ids"]])


def lotti_per_token(lunghezze, budget, massimo):
    """Lotti di articoli dal più corto al più lungo, con al più `budget` token (lunghezza massima × articoli) e al più
    `massimo` articoli ciascuno: gli articoli lunghi vanno da soli o quasi, quelli corti a gruppi."""
    ordine = np.argsort(lunghezze, kind="stable")
    lotti, cur = [], []
    for i in ordine:
        if cur and (len(cur) + 1) * lunghezze[i] > budget or len(cur) >= massimo:
            lotti.append(cur); cur = []
        cur.append(i)
    return lotti + ([cur] if cur else [])


def aspetta_gpu(gb, registra):
    """Aspetta che sulla GPU condivisa ci siano almeno `gb` GB liberi."""
    import torch
    while torch.cuda.mem_get_info()[0] / 2**30 < gb:
        registra(f"GPU: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB liberi, ne servono {gb}: aspetto 5 minuti")
        time.sleep(300)


def embedding_qwen(testi, modello, budget, registra, massimo=16):
    """Qwen3-Embedding: l'istruzione del tentativo 16, poi l'articolo intero; vettore = ultimo token (padding a
    sinistra), normalizzato. Nessun troncamento: il limite è MAX_TOKEN più l'istruzione."""
    import torch
    import torch.nn.functional as Fn
    from transformers import AutoModel, AutoTokenizer
    riproducibile(SEME)
    tok = AutoTokenizer.from_pretrained(modello, padding_side="left")
    m = AutoModel.from_pretrained(modello, torch_dtype=torch.float16).cuda().eval()
    testi = [f"Instruct: {ISTRUZIONE_QWEN}\nQuery:{x}" for x in testi]
    ids = tok(testi, add_special_tokens=True)["input_ids"]
    E, t0 = None, time.time()
    with torch.no_grad():
        for b, idx in enumerate(lotti_per_token(np.array([len(x) for x in ids]), budget, massimo)):
            enc = tok([testi[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKEN + 64,
                      return_tensors="pt").to("cuda")
            h = Fn.normalize(m(**enc).last_hidden_state[:, -1].float(), p=2, dim=1).cpu().numpy()
            if E is None:
                E = np.zeros((len(testi), h.shape[1]), dtype=np.float32)
            E[idx] = h
            if b % 200 == 0:
                registra(f"Qwen: lotto {b}, {time.time() - t0:.0f} s")
    del m; torch.cuda.empty_cache()
    return E


def embedding_mistral(testi, modello, strati, budget, registra, file_parziale, massimo=8, ogni=100):
    """Mistral-Small-24B Base in 4 bit (NF4, calcoli in fp16): per ogni strato, la media degli stati nascosti dei token
    dell'articolo intero (senza padding), come nel tentativo 16. Riparte dal lavoro salvato se si interrompe."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    riproducibile(SEME)
    tok = AutoTokenizer.from_pretrained(modello)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(modello, quantization_config=quant, torch_dtype=torch.float16,
                                             device_map={"": 0}).eval()
    lung = np.array([len(x) for x in tok(testi, add_special_tokens=True)["input_ids"]])
    E, fatti = None, np.zeros(len(testi), bool)
    if Path(file_parziale).exists():
        p = np.load(file_parziale, allow_pickle=True).item()
        if p["n"] == len(testi):
            E, fatti = p["E"], p["fatti"]; registra(f"Mistral: riprendo, {fatti.sum():,} articoli già fatti")
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(lotti_per_token(lung, budget, massimo)):
            if fatti[idx].all():
                continue
            enc = tok([testi[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKEN,
                      return_tensors="pt").to("cuda")
            out = m.model(**enc, output_hidden_states=True)
            maschera = enc["attention_mask"].unsqueeze(-1).float()
            if E is None:
                E = {s: np.zeros((len(testi), out.hidden_states[s].shape[-1]), dtype=np.float32) for s in strati}
            for s in strati:
                E[s][idx] = ((out.hidden_states[s].float() * maschera).sum(1) / maschera.sum(1)).cpu().numpy()
            fatti[idx] = True
            del out
            if b % ogni == 0:
                registra(f"Mistral: {fatti.sum():,}/{len(testi):,} articoli ({time.time() - t0:.0f} s)")
                np.save(file_parziale, {"n": len(testi), "E": E, "fatti": fatti}, allow_pickle=True)
    del m; torch.cuda.empty_cache()
    return E


# ===================================================================================== 3. i modelli lineari
def _adatta(compito, valore, X, y):
    """Logistica multinomiale con le classi bilanciate (classificazione) o ridge (regressione)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    if compito == "classificazione":
        return LogisticRegression(C=valore, max_iter=3000, class_weight="balanced", random_state=SEME).fit(X, y)
    return Ridge(alpha=valore, random_state=SEME).fit(X, y)


def _uscite(compito, m, X):
    """Classificazione: logit (decision_function) per i 5 livelli; regressione: il voto previsto."""
    return m.decision_function(X) if compito == "classificazione" else m.predict(X)


def _criterio(compito, uscite, dom, y):
    """La misura con cui si sceglie la penalità, per testata: F1 macro sul voto degli articoli (classificazione) o
    Spearman tra voto vero e media prevista (regressione)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    if compito == "classificazione":
        g = pd.DataFrame({"d": dom, "y": y, "p": uscite.argmax(1)}).groupby("d").agg(y=("y", "first"), p=("p", voto))
        return f1_score(g.y, g.p, average="macro", labels=range(5), zero_division=0)
    g = pd.DataFrame({"d": dom, "y": y, "p": uscite}).groupby("d").mean()
    return spearmanr(g.y, g.p).correlation


def _matrici(nome, tr, te, testi, X):
    """Le righe tr (addestramento) e te (previsione) pronte per il modello lineare. TF-IDF: vocabolario e idf dai soli
    testi di tr. Embedding: media 0 e varianza 1 dalle sole righe di tr. Ritorna anche la preparazione, da salvare."""
    if nome == "TF-IDF":
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000,
                            strip_accents="unicode")
        return v.fit_transform(testi[tr]), v.transform(testi[te]), v
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(X[tr])
    return s.transform(X[tr]), s.transform(X[te]), s


def _prova_penalita(compito, nome, valore, tr, va, testi, X, y, dom):
    """Un fold interno: addestra con la penalità `valore` sulle righe tr e misura sulle righe va."""
    A, B, _ = _matrici(nome, tr, va, testi, X)
    return _criterio(compito, _uscite(compito, _adatta(compito, valore, A, y[tr]), B), dom[va], y[va])


def lineare_fold(compito, nome, art, k, X=None, n_jobs=16, registra=print):
    """Un fold esterno: la penalità si sceglie con 3 fold interni sulle sole testate di training (divise per voto,
    come lo split esterno), poi si riaddestra su tutto il training e si prevede il fold di test. Salva le uscite per
    articolo, la scelta della penalità e il modello (preparazione + modello lineare)."""
    import joblib
    from joblib import Parallel, delayed
    cartella = PREVISIONI / compito / nome; cartella.mkdir(parents=True, exist_ok=True)
    file = cartella / f"fold{k}.csv.gz"
    if file.exists():
        registra(f"{compito}, {nome}, fold {k}: già fatto"); return pd.read_csv(file)
    t0 = time.time()
    y, dom = bersaglio(art, compito), art.domain.to_numpy()
    testi = art.text.fillna("").to_numpy() if nome == "TF-IDF" else None
    tr, te = np.where(art.fold.to_numpy() != k)[0], np.where(art.fold.to_numpy() == k)[0]
    # 1. i 3 fold interni: le testate di training divise per voto, con un seme fisso per fold
    interne = crea_split(art.score.iloc[tr].groupby(dom[tr]).first(), SEME + k, k=3)
    fi = pd.Series(dom[tr]).map(interne).to_numpy()
    # 2. ogni penalità della griglia su ogni fold interno; si tiene quella con la misura media migliore
    griglia = GRIGLIE[(compito, "TF-IDF" if nome == "TF-IDF" else "embedding")]
    lavori = [(v, tr[fi != j], tr[fi == j]) for v in griglia for j in range(1, 4)]
    misure = Parallel(n_jobs=n_jobs)(delayed(_prova_penalita)(compito, nome, v, a, b, testi, X, y, dom) for v, a, b in lavori)
    medie = pd.DataFrame({"valore": [v for v, _, _ in lavori], "misura": misure}).groupby("valore").misura.mean()
    scelto = medie.idxmax()
    # 3. il modello con la penalità scelta, su tutto il training; le uscite sul fold di test
    Atr, Ate, prep = _matrici(nome, tr, te, testi, X)
    m = _adatta(compito, scelto, Atr, y[tr])
    U = _uscite(compito, m, Ate)
    out = pd.DataFrame({"article_id": art.article_id.to_numpy()[te], "domain": dom[te], "fold": k})
    if compito == "classificazione":
        out[LOGIT] = U
        P = np.exp(U - U.max(1, keepdims=True)); out[PROB] = P / P.sum(1, keepdims=True)
        out["previsione"] = U.argmax(1)
    else:
        out["previsione"] = U
    out.to_csv(file, index=False)
    pd.DataFrame({"fold": k, "valore": medie.index, "misura interna": medie.values, "scelto": medie.index == scelto}) \
        .to_csv(cartella / f"penalita_fold{k}.csv", index=False)
    dest = MODELLI / compito / nome; dest.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preparazione": prep, "modello": m, "penalità": scelto}, dest / f"fold{k}.joblib", compress=3)
    registra(f"{compito}, {nome}, fold {k}: penalità {scelto} ({'bordo della griglia' if scelto in (griglia[0], griglia[-1]) else 'interna'}), "
             f"{time.time() - t0:.0f} s")
    return out


# ===================================================================================== 4. ModernBERT addestrato
def bert_fold(compito, art, k, modello, nome, registra, max_token=MAX_TOKEN, epoche=5, lr=3e-5, batch=4,
              batch_effettivo=32, gb_gpu=16):
    """Un fold: il transformer si addestra tutto, 5 epoche, nessuna validazione (si tiene l'ultima epoca), come nei
    tentativi 16 e 18. Classificazione: entropia incrociata con le classi bilanciate e label smoothing 0,1 (tentativo 16).
    Regressione: errore quadratico sul voto/100, l'uscita si rimoltiplica per 100 (tentativo 18). Salva le uscite per
    articolo (logit o voto), la curva della loss e il modello del fold."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    cartella = PREVISIONI / compito / nome; cartella.mkdir(parents=True, exist_ok=True)
    file = cartella / f"fold{k}.csv.gz"
    if file.exists():
        registra(f"{compito}, {nome}, fold {k}: già fatto"); return pd.read_csv(file)
    aspetta_gpu(gb_gpu, registra)
    riproducibile(SEME + k)
    tok = AutoTokenizer.from_pretrained(modello)
    ids = tok(art.text.fillna("").tolist(), truncation=True, max_length=max_token)["input_ids"]
    troncati = sum(len(x) >= max_token for x in ids)
    if troncati:
        registra(f"ATTENZIONE: {troncati} articoli arrivano al limite di {max_token} token")
    cls = compito == "classificazione"
    y = bersaglio(art, compito) if cls else bersaglio(art, compito) / 100
    dom = art.domain.to_numpy()
    fit, te = np.where(art.fold.to_numpy() != k)[0], np.where(art.fold.to_numpy() == k)[0]
    device = torch.device("cuda")

    def lotti(indici, mescola):
        ordine = np.array(sorted(indici, key=lambda i: len(ids[i])))
        gruppi = [ordine[i:i + batch] for i in range(0, len(ordine), batch)]
        if mescola:
            np.random.shuffle(gruppi)
        return gruppi

    def tensori(gruppo):
        L = max(len(ids[i]) for i in gruppo)
        x = np.full((len(gruppo), L), tok.pad_token_id, dtype=np.int64); msk = np.zeros_like(x)
        for r, i in enumerate(gruppo):
            x[r, :len(ids[i])] = ids[i]; msk[r, :len(ids[i])] = 1
        return torch.from_numpy(x).to(device), torch.from_numpy(msk).to(device)

    extra = {"reference_compile": False} if "ModernBERT" in modello else {}
    model = AutoModelForSequenceClassification.from_pretrained(modello, num_labels=5 if cls else 1, **extra).to(device)
    if cls:
        conta = np.bincount(y[fit], minlength=5).astype(float)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(conta.sum() / (5 * np.maximum(conta, 1)), dtype=torch.float32,
                                                                device=device), label_smoothing=0.1)
    nodecay = ["bias", "LayerNorm.weight", "norm.weight"]
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nodecay)], "weight_decay": 0.01},
                             {"params": [p for n, p in model.named_parameters() if any(x in n for x in nodecay)], "weight_decay": 0.0}], lr=lr)
    accumulo = max(1, batch_effettivo // batch)
    passi = epoche * int(np.ceil(len(fit) / (batch * accumulo)))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * passi), passi)
    scaler = torch.amp.GradScaler("cuda")
    curva, t0 = [], time.time()
    for e in range(1, epoche + 1):
        model.train(); opt.zero_grad(); tot, n = 0.0, 0
        gruppi = lotti(fit, True)
        for b, g in enumerate(gruppi, 1):
            x, msk = tensori(g)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=x, attention_mask=msk).logits.float()
            if cls:
                loss = loss_fn(z, torch.tensor(y[g], device=device))
            else:
                loss = ((z[:, 0] - torch.tensor(y[g], dtype=torch.float32, device=device)) ** 2).mean()
            scaler.scale(loss / accumulo).backward(); tot += loss.item() * len(g); n += len(g)
            if b % accumulo == 0 or b == len(gruppi):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        curva.append({"epoca": e, "loss": tot / n})
        registra(f"{compito}, {nome}, fold {k}, epoca {e}/{epoche}: loss {tot / n:.4f} ({time.time() - t0:.0f} s)")
    model.eval(); U = np.zeros((len(te), 5 if cls else 1), dtype=np.float32); pos = {i: j for j, i in enumerate(te)}
    with torch.no_grad():
        for g in lotti(te, False):
            x, msk = tensori(g)
            with torch.autocast("cuda", dtype=torch.float16):
                U[[pos[i] for i in g]] = model(input_ids=x, attention_mask=msk).logits.float().cpu().numpy()
    out = pd.DataFrame({"article_id": art.article_id.to_numpy()[te], "domain": dom[te], "fold": k})
    if cls:
        out[LOGIT] = U
        P = np.exp(U - U.max(1, keepdims=True)); out[PROB] = P / P.sum(1, keepdims=True)
        out["previsione"] = U.argmax(1)
    else:
        out["previsione"] = U[:, 0] * 100
    dest = MODELLI / compito / nome / f"fold{k}"; dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest); tok.save_pretrained(dest)
    out.to_csv(file, index=False)
    pd.DataFrame(curva).to_csv(cartella / f"curva_fold{k}.csv", index=False)
    registra(f"{compito}, {nome}, fold {k}: finito in {time.time() - t0:.0f} s, memoria GPU massima "
             f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB; modello in {dest}")
    del model, opt; torch.cuda.empty_cache()
    return out


def ambiente():
    """Le versioni di Python e delle librerie usate, e i semi, per ripetere il calcolo."""
    import importlib
    import platform
    v = {"python": platform.python_version(), "seme": SEME, "seme dello split": SEME_SPLIT}
    for lib in ["numpy", "pandas", "sklearn", "scipy", "torch", "transformers", "bitsandbytes"]:
        try:
            v[lib] = importlib.import_module(lib).__version__
        except ImportError:
            pass
    return v


def previsioni(compito, nome):
    """Tutte le uscite per articolo di un modello (i 5 fold), se ci sono."""
    fs = sorted((PREVISIONI / compito / nome).glob("fold*.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None

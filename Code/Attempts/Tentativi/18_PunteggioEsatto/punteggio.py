"""Tentativo 18: il punteggio esatto NewsGuard (0-100), solo come numero, senza classi.

Split stratificati sul punteggio continuo; cinque condizioni dei dati (quanto pesa ogni testata in addestramento);
regressori (e classificatori trasformati in punteggio, per confronto) sopra TF-IDF e sopra Mistral-Small 24B congelato;
ModernBERT e DeBERTa addestrati come regressori. Valutazione solo continua (errore in punti, Spearman, Pearson), per
articolo singolo e per testata.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONDIZIONI = {"A": "tutti gli articoli",
              "B": "tutti gli articoli, ogni testata pesa uguale",
              "C": "al più 10 articoli per testata",
              "D": "al più 30 articoli per testata",
              "E": "un vettore per testata (media degli articoli)"}
TETTO = {"C": 10, "D": 30}
GRIGLIA = {("TF-IDF", "regressore"): (0.03, 0.1, 0.3, 1, 3), ("TF-IDF", "classificatore"): (10, 100, 1_000),
           ("Mistral 24B", "regressore"): (1_000, 10_000, 100_000), ("Mistral 24B", "classificatore"): (0.001, 0.01, 0.1, 1)}


def livello(score):
    """I 5 livelli NewsGuard (0 = l1 … 4 = l5) dal punteggio: 0-39, 40-59, 60-74, 75-99, 100."""
    return np.digitize(np.asarray(score, dtype=float), [39.5, 59.5, 74.5, 99.5])


def crea_split(punteggi, seme, k=5):
    """Fold stratificati sul punteggio continuo: le testate si ordinano per punteggio (a parità a caso) e, a blocchi di
    k consecutive, ognuna va in un fold diverso. Ogni fold ha così la stessa distribuzione del punteggio."""
    rng = np.random.default_rng(seme)
    ordine = punteggi.index[np.lexsort((rng.random(len(punteggi)), punteggi.to_numpy()))]
    fold = pd.Series(0, index=punteggi.index)
    for i in range(0, len(ordine), k):
        blocco = ordine[i:i + k]
        fold[blocco] = rng.permutation(np.arange(1, k + 1))[:len(blocco)]
    return fold


def righe_e_pesi(domini, condizione, seme):
    """Quali articoli di training si usano e con che peso. A: tutti, peso 1. B: tutti, peso 1/(articoli della testata),
    così ogni testata pesa uguale. C, D: al più 10 o 30 articoli per testata, presi a caso, peso 1."""
    domini = np.asarray(domini)
    if condizione in TETTO:
        rng = np.random.default_rng(seme)
        s = pd.Series(rng.random(len(domini)))
        rango = s.groupby(domini).rank(method="first").to_numpy()
        idx = np.where(rango <= TETTO[condizione])[0]
        return idx, np.ones(len(idx))
    idx = np.arange(len(domini))
    if condizione == "B":
        n = pd.Series(domini).map(pd.Series(domini).value_counts()).to_numpy()
        w = 1.0 / n
        return idx, w * len(w) / w.sum()
    return idx, np.ones(len(idx))


def matrice_media(domini):
    """Matrice sparsa testate × articoli che fa la media dei vettori degli articoli di ogni testata."""
    from scipy import sparse
    domini = np.asarray(domini)
    nomi, inv = np.unique(domini, return_inverse=True)
    n = np.bincount(inv)
    P = sparse.csr_matrix((1.0 / n[inv], (inv, np.arange(len(domini)))), shape=(len(nomi), len(domini)))
    return nomi, P


def adatta(tipo, X, y, w, valore):
    """Regressore: ridge sul punteggio. Classificatore: logistica sui 5 livelli; il punteggio previsto è la media dei
    punteggi medi di ogni livello nel training, pesata con le probabilità. Ritorna la funzione che prevede il punteggio."""
    from sklearn.linear_model import LogisticRegression, Ridge
    if tipo == "regressore":
        m = Ridge(alpha=valore).fit(X, y, sample_weight=w)
        return m.predict
    lv = livello(y)
    medie = pd.Series(y).groupby(lv).apply(lambda s: np.average(s, weights=w[s.index])).reindex(range(5)).to_numpy()
    m = LogisticRegression(C=valore, max_iter=3000).fit(X, lv, sample_weight=w)
    return lambda B: m.predict_proba(B) @ medie[m.classes_]


def _testate_mae(prev, y, domini):
    d = pd.DataFrame({"d": domini, "p": prev, "y": y}).groupby("d").mean()
    return float(np.mean(np.abs(d.p - d.y)))


def un_lavoro(modello, X, art, fold, k, condizione, tipo, seme, interni=3):
    """Un fold, una condizione, un tipo di modello. La penalità si sceglie con una validazione interna sulle testate
    di training (split stratificato sul punteggio, errore medio assoluto per testata); poi si riaddestra su tutto il
    training. Ritorna le previsioni per gli articoli di test e, per la condizione E, anche quelle per testata."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    dom, y = art.domain.to_numpy(), art.score.to_numpy().astype(float)
    tr_t = fold.index[fold != k]
    tr, te = np.where(np.isin(dom, tr_t))[0], np.where(~np.isin(dom, tr_t))[0]
    if modello == "TF-IDF":                        # vocabolario dai soli testi di training (senza etichette)
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000, strip_accents="unicode")
        Xtr, Xte = v.fit_transform(art.text.iloc[tr].fillna("")), v.transform(art.text.iloc[te].fillna(""))
    else:
        Xtr, Xte = X[tr], X[te]

    def prepara(righe_tr, righe_val=None):
        """Righe di addestramento (dentro il training) secondo la condizione; restituisce matrici pronte e pesi."""
        d_tr = dom[tr][righe_tr]
        if condizione == "E":
            nomi, P = matrice_media(d_tr)
            A = P @ Xtr[righe_tr]
            yy = pd.Series(y[tr][righe_tr]).groupby(d_tr).first().reindex(nomi).to_numpy()
            ww = np.ones(len(nomi))
        else:
            idx, ww = righe_e_pesi(d_tr, condizione, seme * 100 + k)
            A, yy = Xtr[righe_tr][idx], y[tr][righe_tr][idx]
        sc = None
        if modello != "TF-IDF":
            A = np.asarray(A); sc = StandardScaler().fit(A); A = sc.transform(A)
        return A, yy, ww, sc

    def applica(f, sc, B):
        B = np.asarray(B) if modello != "TF-IDF" else B
        return f(sc.transform(B) if sc is not None else B)

    # scelta della penalità: validazione interna sulle testate di training
    griglia = GRIGLIA[(modello, tipo)]
    interne = crea_split(pd.Series(y[tr]).groupby(dom[tr]).first(), seme * 1000 + k, interni)
    errori = {g: [] for g in griglia}
    for j in range(1, interni + 1):
        val_t = interne.index[interne == j]
        r_fit, r_val = np.where(~np.isin(dom[tr], val_t))[0], np.where(np.isin(dom[tr], val_t))[0]
        A, yy, ww, sc = prepara(r_fit)
        for g in griglia:
            f = adatta(tipo, A, yy, ww, g)
            errori[g].append(_testate_mae(applica(f, sc, Xtr[r_val]), y[tr][r_val], dom[tr][r_val]))
    migliore = min(griglia, key=lambda g: np.mean(errori[g]))
    A, yy, ww, sc = prepara(np.arange(len(tr)))
    f = adatta(tipo, A, yy, ww, migliore)
    out = pd.DataFrame({"article_id": art.article_id.to_numpy()[te], "domain": dom[te], "previsione": applica(f, sc, Xte)})
    testate = None
    if condizione == "E":
        nomi, P = matrice_media(dom[te])
        testate = pd.DataFrame({"domain": nomi, "previsione": applica(f, sc, P @ Xte)})
    info = {"modello": modello, "fold": k, "condizione": condizione, "tipo": tipo, "split": seme, "penalità": migliore,
            **{f"errore interno {g}": np.mean(e) for g, e in errori.items()}}
    return out, testate, info


# ------------------------------------------------------------------------------------------------ BERT addestrati
def bert_fold(art, fold, k, condizione, modello, cartella, registra, max_token, epoche=5, lr=3e-5, batch=4,
              batch_effettivo=32, seme=42, gb_gpu=14):
    """Un fold: il transformer (ModernBERT, DeBERTa…) si addestra tutto come regressore del punteggio/100, con errore
    quadratico pesato secondo la condizione (A, B, C, D), 5 epoche, nessuna validazione (si tiene l'ultima epoca).
    Il test sono tutti gli articoli delle testate del fold. Salva fold{k}.csv (previsioni ×100) e curva_fold{k}.csv."""
    import random

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    file = Path(cartella) / f"fold{k}.csv"
    if file.exists():
        fatto = pd.read_csv(file)
        if set(fatto.article_id) == set(art.article_id[~art.domain.isin(fold.index[fold != k])]):
            registra(f"fold {k}: già fatto"); return fatto
    while True:                                                    # aspetta che la GPU abbia abbastanza memoria libera
        libera = torch.cuda.mem_get_info()[0] / 2**30
        if libera >= gb_gpu:
            break
        registra(f"GPU: {libera:.1f} GB liberi, ne servono {gb_gpu}: aspetto 5 minuti"); time.sleep(300)
    random.seed(seme + k); np.random.seed(seme + k); torch.manual_seed(seme + k); torch.cuda.manual_seed_all(seme + k)
    tok = AutoTokenizer.from_pretrained(modello)
    ids = tok(art.text.fillna("").tolist(), truncation=True, max_length=max_token)["input_ids"]
    dom, score = art.domain.to_numpy(), art.score.to_numpy() / 100
    tr_t = fold.index[fold != k]
    tr, te = np.where(np.isin(dom, tr_t))[0], np.where(~np.isin(dom, tr_t))[0]
    idx, w = righe_e_pesi(dom[tr], condizione, seme * 100 + k)
    fit, peso_di = tr[idx], dict(zip(tr[idx], w))
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
    model = AutoModelForSequenceClassification.from_pretrained(modello, num_labels=1, **extra).to(device)
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
                z = model(input_ids=x, attention_mask=msk).logits.float()[:, 0]
            pw = torch.tensor([peso_di[i] for i in g], dtype=torch.float32, device=device)
            loss = (pw * (z - torch.tensor(score[g], dtype=torch.float32, device=device)) ** 2).sum() / pw.sum()
            scaler.scale(loss / accumulo).backward(); tot += loss.item() * len(g); n += len(g)
            if b % accumulo == 0 or b == len(gruppi):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        curva.append({"epoca": e, "loss": tot / n})
        registra(f"fold {k}, epoca {e}/{epoche}: loss {tot / n:.4f} ({time.time() - t0:.0f} s)")
    model.eval(); P = np.zeros(len(te), dtype=np.float32); pos = {i: j for j, i in enumerate(te)}
    with torch.no_grad():
        for g in lotti(te, False):
            x, msk = tensori(g)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=x, attention_mask=msk).logits.float()[:, 0]
            P[[pos[i] for i in g]] = z.cpu().numpy()
    out = pd.DataFrame({"article_id": art.article_id.to_numpy()[te], "domain": dom[te], "previsione": P * 100})
    out.to_csv(file, index=False)
    pd.DataFrame(curva).to_csv(Path(cartella) / f"curva_fold{k}.csv", index=False)
    registra(f"fold {k}: finito, memoria GPU massima {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    del model, opt; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------ valutazione continua
def misure(y, p):
    """Errore medio assoluto e quadratico (in punti), R², Spearman e Pearson."""
    from scipy.stats import pearsonr, spearmanr
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R²": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "Spearman": float(spearmanr(y, p).correlation), "Pearson": float(pearsonr(y, p)[0])}

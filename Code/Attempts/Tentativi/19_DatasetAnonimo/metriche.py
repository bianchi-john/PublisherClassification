"""Tentativo 19: le misure, ricalcolate dalle uscite per articolo (logit o voto previsto). Passo per passo:

1. dagli articoli alla testata: il classificatore vota il livello, il regressore fa la media del voto;
2. le trasformazioni, per mettere le due filosofie sullo stesso terreno:
   - il regressore passa ai livelli con i confini NewsGuard, con le quote del training (tentativo 16) o con il voto
     degli articoli (tentativo 17), e si può riadattare ai quantili del voto del training;
   - il classificatore passa al voto con il voto atteso: probabilità × voto medio di ogni livello nel training;
3. le misure: sui livelli (F1, tolerant, kappa…) e sul voto (MAE, Spearman, «entro 10 e 20 punti»…);
4. le viste di un modello: ogni coppia (vero, previsto) che ne viene, con le sue misure;
5. gli intervalli al 95%: bootstrap sulle testate, anche per la differenza tra due viste sulle stesse testate.
Ogni trasformazione si calcola una volta sui dati veri; il bootstrap ricampiona solo le testate.
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn import metrics as M

from modelli import LIVELLI, PROB, SEME, voto

CONFINI = [39.5, 59.5, 74.5, 99.5]          # dal voto NewsGuard ai livelli: 0-39, 40-59, 60-74, 75-99, 100
ENTRO = (10, 20)                             # «entro N punti»: la tolerant dello spazio del voto


# ===================================================================================== 1. dagli articoli alla testata
def testate_classificatore(pa, t):
    """Una riga per testata: livello e voto veri, fold, livello votato dai suoi articoli (a parità il più basso) e
    probabilità medie dei 5 livelli."""
    g = pa.groupby("domain")
    T = pd.DataFrame({"vero": t.label, "voto vero": t.score, "fold": t.fold})
    T["previsto"] = g.previsione.agg(voto)
    T[PROB] = g[PROB].mean()
    return T


def testate_regressore(pa, t):
    """Una riga per testata: livello e voto veri, fold, media dei voti previsti dei suoi articoli."""
    T = pd.DataFrame({"vero": t.label, "voto vero": t.score, "fold": t.fold})
    T["previsto"] = pa.groupby("domain").previsione.mean()
    return T


# ===================================================================================== 2. le trasformazioni
def livelli_confini(v):
    """Dal voto al livello con i confini NewsGuard, senza adattamenti."""
    return np.digitize(np.asarray(v, float), CONFINI)


def livelli_quote(previsto, fold, livello_vero):
    """Dal voto al livello con le quote del training (tentativo 16): in ogni fold le previsioni di test si ordinano e
    si tagliano alle proporzioni dei livelli veri nel training dello stesso fold (le righe degli altri fold)."""
    previsto, fold, livello_vero = (np.asarray(x) for x in (previsto, fold, livello_vero))
    out = np.zeros(len(previsto), int)
    for k in np.unique(fold):
        te = fold == k
        quote = np.cumsum(np.bincount(livello_vero[~te].astype(int), minlength=5))[:-1] / (~te).sum()
        out[te] = np.searchsorted(np.quantile(previsto[te], quote), previsto[te])
    return out


def livelli_voto_articoli(pa, art):
    """Dal voto al livello con il voto degli articoli (tentativo 17): ogni articolo prende un livello con le quote
    degli articoli di training del suo fold, poi la testata prende il livello più votato (a parità il più basso)."""
    a = pa.merge(art[["article_id", "label"]], on="article_id")
    a["livello"] = livelli_quote(a.previsione, a.fold, a.label)
    return a.groupby("domain").livello.agg(voto)


def voto_atteso(T, t):
    """Dal classificatore al voto: le probabilità medie dei 5 livelli per il voto medio di ogni livello tra le testate
    di training dello stesso fold (circa 21, 50, 67, 85, 100)."""
    out = pd.Series(np.nan, index=T.index)
    for k in T.fold.unique():
        tr = t[t.fold != k]
        medie = tr.groupby("label").score.mean().reindex(range(5)).to_numpy()
        te = T.fold == k
        out[te] = T.loc[te, PROB].to_numpy() @ medie
    return out


def voto_riadattato(previsto, fold, voto_vero):
    """Il voto del regressore riadattato ai quantili del training (come l'equalizzazione dell'istogramma): in ogni fold
    la testata che sta al q% delle previsioni di test prende il voto che sta al q% dei voti veri del training."""
    previsto, fold, voto_vero = (np.asarray(x, float) for x in (previsto, fold, voto_vero))
    out = np.zeros(len(previsto))
    for k in np.unique(fold):
        te = fold == k
        q = (pd.Series(previsto[te]).rank().to_numpy() - 1) / max(te.sum() - 1, 1)
        out[te] = np.quantile(voto_vero[~te], q)
    return out


# ===================================================================================== 3. le misure
def _sicuro(f):
    try:
        return float(f())
    except ValueError:
        return np.nan


def misure_livelli(y, p):
    """Le misure tra due serie di livelli 0-4 (vero, previsto)."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    f1 = M.f1_score(y, p, average=None, labels=range(5), zero_division=0)
    out = {"F1 macro": f1.mean(),
           "accuratezza": M.accuracy_score(y, p),
           "accuratezza bilanciata": M.balanced_accuracy_score(y, p),
           "tolerant (±1 livello)": np.mean(np.abs(y - p) <= 1),
           "errore medio in livelli": np.mean(np.abs(y - p)),
           "kappa": M.cohen_kappa_score(y, p),
           "kappa quadratica": M.cohen_kappa_score(y, p, weights="quadratic"),
           "Spearman": _sicuro(lambda: spearmanr(y, p).correlation),
           "Kendall": _sicuro(lambda: kendalltau(y, p).correlation),
           "F1 pesata": M.f1_score(y, p, average="weighted", labels=range(5), zero_division=0)}
    out.update({f"F1 {LIVELLI[i]}": f1[i] for i in range(5)})
    a, b = y >= 2, p >= 2                                            # affidabile = da l3 in su (voto ≥ 60)
    out.update({"affidabile: accuratezza": M.accuracy_score(a, b),
                "affidabile: F1 macro": M.f1_score(a, b, average="macro", zero_division=0),
                "affidabile: MCC": M.matthews_corrcoef(a, b)})
    return out


def misure_voto(y, p):
    """Le misure tra due serie di voti 0-100 (vero, previsto)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    out = {"MAE": np.mean(np.abs(e)),
           "RMSE": np.sqrt(np.mean(e ** 2)),
           "errore mediano": np.median(np.abs(e)),
           "R²": M.r2_score(y, p),
           "Spearman": _sicuro(lambda: spearmanr(y, p).correlation),
           "Pearson": _sicuro(lambda: pearsonr(y, p)[0]),
           "Kendall": _sicuro(lambda: kendalltau(y, p).correlation),
           "CCC di Lin": 2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2),
           "distorsione (previsto − vero)": np.mean(e)}
    out.update({f"entro {n} punti": np.mean(np.abs(e) <= n) for n in ENTRO})
    a, b = y >= 60, p >= 60
    out.update({"affidabile: accuratezza": M.accuracy_score(a, b),
                "affidabile: F1 macro": M.f1_score(a, b, average="macro", zero_division=0),
                "affidabile: AUC": _sicuro(lambda: M.roc_auc_score(a, p))})
    return out


def misure_probabilita(T):
    """Le misure che usano le probabilità del classificatore (medie per testata)."""
    y = T.vero.to_numpy(int)
    P = T[PROB].to_numpy(); P = P / P.sum(1, keepdims=True)
    return {"AUC macro (5 livelli)": _sicuro(lambda: M.roc_auc_score(y, P, multi_class="ovr", average="macro", labels=range(5))),
            "log loss": M.log_loss(y, np.clip(P, 1e-12, 1), labels=range(5)),
            "Brier": np.mean(((np.eye(5)[y] - P) ** 2).sum(1)),
            "affidabile: AUC (probabilità)": _sicuro(lambda: M.roc_auc_score(y >= 2, P[:, 2:].sum(1)))}


MISURA = {"livelli": misure_livelli, "voto": misure_voto}


# ===================================================================================== 4. le viste di un modello
def viste_classificatore(T, t):
    """Le coppie (vero, previsto) del classificatore, nei due spazi: {vista: (spazio, vero, previsto)}."""
    return {"livello votato dagli articoli": ("livelli", T.vero, T.previsto),
            "voto atteso dalle probabilità": ("voto", T["voto vero"], voto_atteso(T, t))}


def viste_regressore(T, pa, art):
    """Le coppie (vero, previsto) del regressore, nei due spazi: {vista: (spazio, vero, previsto)}."""
    return {"media prevista": ("voto", T["voto vero"], T.previsto),
            "media riadattata ai quantili del training": ("voto", T["voto vero"], voto_riadattato(T.previsto, T.fold, T["voto vero"])),
            "livelli: confini NewsGuard": ("livelli", T.vero, livelli_confini(T.previsto)),
            "livelli: quote del training": ("livelli", T.vero, livelli_quote(T.previsto, T.fold, T.vero)),
            "livelli: voto degli articoli": ("livelli", T.vero, livelli_voto_articoli(pa, art).reindex(T.index))}


def viste_riferimento(t):
    """Chi non legge il testo: la classe più frequente nel training (livelli) e la media del training (voto)."""
    cls = pd.Series(0, index=t.index); media = pd.Series(0.0, index=t.index)
    for k in t.fold.unique():
        tr, te = t.fold != k, t.fold == k
        cls[te] = t.label[tr].mode().min(); media[te] = t.score[tr].mean()
    return {"classe più frequente": ("livelli", t.label, cls), "media del training": ("voto", t.score, media)}


def misure_viste(viste):
    """Una riga per vista con tutte le misure del suo spazio."""
    return pd.DataFrame([{"vista": v, "spazio": s, **MISURA[s](y, p)} for v, (s, y, p) in viste.items()])


# ===================================================================================== 5. gli intervalli
def bootstrap(y, p, spazio, chiavi, B=2000, seme=SEME):
    """Intervalli al 95% ricampionando le testate con reinserimento (le coppie vero-previsto restano fisse)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seme)
    valori = {c: [] for c in chiavi}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        m = MISURA[spazio](y[i], p[i])
        for c in chiavi:
            valori[c].append(m[c])
    return {c: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for c, v in valori.items()}


def bootstrap_differenza(vista1, vista2, chiave, B=2000, seme=SEME):
    """Differenza vista1 − vista2 della stessa misura, sulle stesse testate ricampionate insieme (lo split è uno
    solo). Ritorna la differenza e l'intervallo al 95%: se non contiene lo 0, la differenza non è caso."""
    (s1, y1, p1), (s2, y2, p2) = vista1, vista2
    y1, p1, y2, p2 = (np.asarray(x) for x in (y1, p1, y2, p2))
    rng = np.random.default_rng(seme)
    diff = []
    for _ in range(B):
        i = rng.integers(0, len(y1), len(y1))
        diff.append(MISURA[s1](y1[i], p1[i])[chiave] - MISURA[s2](y2[i], p2[i])[chiave])
    d = MISURA[s1](y1, p1)[chiave] - MISURA[s2](y2, p2)[chiave]
    return d, (np.nanpercentile(diff, 2.5), np.nanpercentile(diff, 97.5))

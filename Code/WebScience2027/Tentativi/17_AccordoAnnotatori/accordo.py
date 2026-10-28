"""Tentativo 17: l'accordo tra annotatori. NewsGuard e Media Bias/Fact Check (MBFC) sono due annotatori esperti; i
modelli del tentativo 16, che leggono solo gli articoli, sono il terzo. Ogni giudizio è un livello ordinato da 0 a 4.

Le previsioni dei modelli vengono dal tentativo 16: ogni testata è stata prevista quando era nel fold di test, quindi da
un modello che non l'aveva mai vista; MBFC non è mai stato usato per addestrare.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

CLASSI = ["p_l1", "p_l2", "p_l3", "p_l4", "p_l5"]
# fattualità MBFC ("factual reporting") su 5 livelli, dal peggiore al migliore come i livelli NewsGuard
FATTUALITA = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
CREDIBILITA = {"Low": 0, "Medium": 1, "High": 2}
# affidabile sì/no, fissato prima dei risultati: NewsGuard almeno l3 (punteggio ≥ 60), MBFC almeno "Mostly Factual",
# credibilità MBFC "High", modelli almeno l3
SOGLIA = {"NewsGuard": 2, "MBFC": 3, "MBFC credibilità": 2, "modello": 2}


def voto(valori):
    """Voto di maggioranza; a parità vince il livello più basso (come nel tentativo 16)."""
    c = pd.Series(valori).value_counts()
    return int(c[c == c.max()].index.min())


def leggi_mbfc(file, testate):
    """Le valutazioni MBFC delle nostre testate (dominio identico, schede non chiuse): fattualità e credibilità."""
    m = pd.read_csv(file)
    m["d"] = m.dominio.str.lower().str.replace(r"^www\.", "", regex=True)
    m = m[~m.morto].drop_duplicates("d").set_index("d")
    chiave = pd.Series(testate.index.str.lower().str.replace(r"^www\.", "", regex=True), index=testate.index)
    chiave = chiave[chiave.isin(m.index)]
    out = m.loc[chiave.to_numpy(), ["factual", "credibilita", "orientamento"]].set_axis(chiave.index)
    out["fattualita"] = out.factual.map(FATTUALITA)
    out["credibilita_livello"] = out.credibilita.map(CREDIBILITA)
    return out


def livelli_articoli(pa, t):
    """Un livello (0-4) per ogni articolo e per ogni impostazione. Classificatore: il livello più probabile.
    Regressore: in ogni fold i punteggi previsti degli articoli di test si ordinano e si tagliano con le quote di livelli
    degli articoli di training (come `testate_voto` del tentativo 16). Ritorna domain, classificatore, regressore."""
    fold = t.fold.reindex(pa.domain).to_numpy()
    etichetta = t.label.reindex(pa.domain).to_numpy()
    out = pd.DataFrame({"domain": pa.domain.to_numpy()})
    out["classificatore"] = pa[CLASSI].to_numpy().argmax(1)
    lv = np.zeros(len(pa), dtype=int)
    for k in np.unique(fold):
        te = fold == k
        quote = np.cumsum(np.bincount(etichetta[~te], minlength=5))[:-1] / (~te).sum()
        sc = pa.s_regressione.to_numpy()[te]
        lv[te] = np.searchsorted(np.quantile(sc, quote), sc)
    out["regressore"] = lv
    return out


def gruppi(livelli, testate):
    """I livelli degli articoli di ogni testata (Series indicizzata dal dominio), nell'ordine di `testate`."""
    g = {d: v.to_numpy() for d, v in livelli.groupby(level=0)}
    return [g[d] for d in testate]


def giudizio(gruppi_testate, k=None, rng=None):
    """Il giudizio del modello per ogni testata: il livello votato dai suoi articoli (tutti, o `k` presi a caso) e
    affidabile = la maggioranza degli articoli votanti ha un livello da l3 in su (a parità: no)."""
    livello, affidabile = [], []
    for v in gruppi_testate:
        if k is not None and k < len(v):
            v = rng.choice(v, k, replace=False)
        livello.append(voto(v)); affidabile.append(int(np.mean(v >= SOGLIA["modello"]) > 0.5))
    return np.array(livello), np.array(affidabile)


def misure(a, b, aff_a, aff_b):
    """Accordo tra due annotatori sulle stesse testate: Spearman tra i livelli, kappa di Cohen e accordo su
    affidabile sì/no."""
    return {"Spearman": spearmanr(a, b).correlation, "kappa": cohen_kappa_score(aff_a, aff_b), "accordo": float(np.mean(aff_a == aff_b))}


def bootstrap(coppie, n, B=2000, seme=0):
    """Intervalli al 95% ricampionando le testate (le stesse per tutte le coppie, così anche le differenze sono
    appaiate). `coppie`: nome → (a, b, aff_a, aff_b) come array allineati. Ritorna valori, campioni bootstrap."""
    rng = np.random.default_rng(seme)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    valori = {c: misure(*x) for c, x in coppie.items()}
    campioni = {c: pd.DataFrame([misure(*(v[i] for v in x)) for i in idx]) for c, x in coppie.items()}
    return valori, campioni

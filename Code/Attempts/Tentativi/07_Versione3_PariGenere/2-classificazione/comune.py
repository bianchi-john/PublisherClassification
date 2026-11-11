"""Versione 3, a parità di genere: dati, pesi e metriche comuni ai notebook.

Il compito è binario (affidabile = punteggio NewsGuard >= 60, cioè livelli l3-l5, label >= 2 nel codice) e si
valuta DENTRO ogni orientamento politico NewsGuard: così sapere l'orientamento (il "genere" della testata) non
serve a indovinare. La metrica principale, fissata prima dei risultati, è la media sugli orientamenti
dell'accuratezza bilanciata per testata.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

VERSIONI = Path("../../../../../Dataset/tentativi/07_Versione3_PariGenere/versioni")        # costruito da ../1-data-cleaning (la stessa pulizia della Versione 2)
NEWSGUARD = Path("../../../../../Dataset/etichette/newsguard_2022.csv")
RESULTS = Path("../Results/2-classificazione")
SEED = 42
ORIENTAMENTI = ["(nessuno)", "Far Left", "Far Right", "Slightly Left", "Slightly Right"]


def registratore(log):
    """Stampa e accoda al log con data e ora."""
    def registra(m):
        riga = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
        print(riga, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(riga + "\n")
    return registra


def carica(prova=False):
    """Tutti gli articoli della Versione 2 (testate valutate e di solo training), con fold, peso, orientamento e y binaria."""
    art = pd.read_csv(VERSIONI / "dataset_finale.csv")
    art["text"] = art["text"].fillna("")
    fold = pd.read_csv(VERSIONI / "fold_finale.csv")[["domain", "peso", "fold_gruppo"]]
    art = art.merge(fold, on="domain", how="left")
    ng = pd.read_csv(NEWSGUARD, low_memory=False)
    ng = ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")
    art["orientamento"] = art.domain.map(ng.Orientation).fillna("(nessuno)")
    art["y"] = (art.label >= 2).astype(int)                 # 1 = affidabile (punteggio >= 60)
    if prova:
        t = art.drop_duplicates("domain")
        scelte = set(t[t.valutazione].groupby(["orientamento", "y"]).head(3).domain) | set(t[~t.valutazione].head(20).domain)
        art = art[art.domain.isin(scelte)]
    return art.reset_index(drop=True)


def fattori_genere(df, peso="peso"):
    """Per ogni (orientamento, classe) il fattore che rende affidabili e inaffidabili ugualmente pesanti DENTRO ogni
    orientamento (calcolato sulle righe di df con il loro peso). Così, nel training, l'orientamento non dice niente
    sulla classe."""
    w = df.groupby(["orientamento", "y"])[peso].sum()
    tot = w.groupby(level=0).sum()
    return {(o, c): float(tot[o] / (2 * w[(o, c)])) for o, c in w.index}


def pesi_righe(df, fattori, peso="peso"):
    return df[peso].to_numpy() * np.array([fattori[(o, c)] for o, c in zip(df.orientamento, df.y)])


def bilanciata(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    return 0.5 * ((y & yhat).sum() / max(y.sum(), 1) + (~y & ~yhat).sum() / max((~y).sum(), 1))


def per_testata(pred):
    """pred: una riga per articolo con domain, y, p (probabilità di 'affidabile'), orientamento."""
    g = pred.groupby("domain")
    return pd.DataFrame({"y": g.y.first(), "p": g.p.mean(), "orientamento": g.orientamento.first()})


def metriche(t, soglia=0.5):
    """Le metriche per testata. Principale: 'media sugli orientamenti' (accuratezza bilanciata dentro ogni
    orientamento che ha tutte e due le classi, poi la media)."""
    yhat = t.p >= soglia
    m, bil, auc = {"testate": len(t)}, {}, {}
    for o, g in t.groupby("orientamento"):
        if g.y.nunique() == 2:
            bil[o] = bilanciata(g.y, g.p >= soglia)
            auc[o] = roc_auc_score(g.y, g.p)
    m["media sugli orientamenti"] = float(np.mean(list(bil.values()))) if bil else np.nan
    m["AUC media sugli orientamenti"] = float(np.mean(list(auc.values()))) if auc else np.nan
    for o in ORIENTAMENTI:
        m[f"bilanciata {o}"] = bil.get(o, np.nan)
    m["bilanciata su tutto"] = bilanciata(t.y, yhat)
    m["accuratezza"] = accuracy_score(t.y, yhat)
    m["F1 macro"] = f1_score(t.y, yhat, average="macro", zero_division=0)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}

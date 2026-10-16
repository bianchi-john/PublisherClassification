"""Dati, percorsi e letture comuni ai notebook di classificazione."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

VERS = Path("../Dataset/versioni")
RESULTS = Path("../Results/2-classificazione")
SEED = 42
CLASSI = ["p_l1", "p_l2", "p_l3", "p_l4", "p_l5"]
SCHEMI = ["fold_gruppo", "fold_testata"]
NOME_SCHEMA = {"fold_gruppo": "per gruppo", "fold_testata": "per testata"}


def cartella_risultati(prova=False):
    """In prova si scrive in una sottocartella, per non toccare i risultati veri."""
    d = RESULTS / "prova" if prova else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    return d


def registratore(log):
    """Stampa e accoda al log con data e ora (utile quando il notebook gira in screen sul server)."""
    def registra(m):
        riga = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
        print(riga, flush=True)
        with open(log, "a") as f:
            f.write(riga + "\n")
    return registra


def taglia(testo, n):
    """Le prime n parole."""
    return " ".join(str(testo).split()[:n])


def carica(prova=False):
    """Articoli del dataset definitivo, con i fold e il testo delle pagine della loro testata.

    In prova tiene 8 testate per livello (40 in tutto), per controllare che il codice giri.
    """
    art = pd.read_csv(VERS / "dataset_finale.csv")
    art["testo"] = art["testo"].fillna("")
    fold = pd.read_csv(VERS / "fold_finale.csv")[["domain", "fold_gruppo", "fold_testata"]]
    art = art.merge(fold, on="domain", how="left")
    pag = pd.read_csv(VERS / "pagine_finale.csv")
    art["pagine"] = art.domain.map(dict(zip(pag.domain, pag.testo_pagine.fillna("")))).fillna("")
    if prova:
        testate = art.drop_duplicates("domain").groupby("label").sample(8, random_state=SEED).domain
        art = art[art.domain.isin(testate)]
    return art.reset_index(drop=True)


def tolerant(y, p, a):
    """Accuratezza che dà credito a per gli errori di un solo livello."""
    d = np.abs(np.asarray(y) - np.asarray(p))
    return float(np.mean(np.where(d == 0, 1.0, np.where(d == 1, a, 0.0))))


def voto(s):
    """Voto di maggioranza; a parità vince il livello più basso."""
    c = s.value_counts()
    return c[c == c.max()].index.min()


def per_testata(pred):
    return pred.groupby("domain").agg(y_true=("y_true", "first"), y_pred=("y_pred", voto))


def letture(pred):
    """Per articolo e per testata (voto): 5 classi, F1 macro, binario e tolerant (alfa 0.5 e 1).

    Binario: soglia tra l2 e l3 (punteggio NewsGuard 60), cioè affidabile se livello >= 2.
    """
    out = {}
    for livello, d in [("articolo", pred), ("testata", per_testata(pred))]:
        out[f"{livello} 5cl"] = accuracy_score(d.y_true, d.y_pred)
        out[f"{livello} F1"] = f1_score(d.y_true, d.y_pred, average="macro")
        out[f"{livello} bin"] = accuracy_score(d.y_true >= 2, d.y_pred >= 2)
        out[f"{livello} tol0.5"] = tolerant(d.y_true, d.y_pred, 0.5)
        out[f"{livello} tol1"] = tolerant(d.y_true, d.y_pred, 1.0)
    return {k: round(v, 3) for k, v in out.items()}

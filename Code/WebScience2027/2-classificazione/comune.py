"""Funzioni usate dai notebook di classificazione: dati, fold, metriche, voto di maggioranza, classificatore."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PULITO = "../Dataset/DatasetPulito/"     # versioni pulite del dataset (notebook 1-data-cleaning/02)
RESULTS = "../Results/2-classificazione/"
SEED = 42
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
VERSIONI = {"nessun minimo": "min0", "almeno 10 articoli": "min10", "almeno 20 articoli": "min20"}


def carica_versione(nome_file):
    """Una versione pulita del dataset, con i suoi fold (file fold_<versione>.csv)."""
    df = pd.read_csv(PULITO + f"dataset_{nome_file}.csv")
    df["testo"] = df["testo"].fillna("")
    fold = pd.read_csv(PULITO + f"fold_{nome_file}.csv")
    return df.merge(fold, on="domain", how="left")


def tolerant(y_true, y_pred, alpha):
    distanza = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.mean(np.where(distanza == 0, 1.0, np.where(distanza == 1, alpha, 0.0)))


def metriche(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "F1 macro": f1_score(y_true, y_pred, average="macro"),
            "Tolerant α=0.5": tolerant(y_true, y_pred, 0.5),
            "Tolerant α=1.0": tolerant(y_true, y_pred, 1.0)}


def voto(predizioni):
    """Voto di maggioranza; in caso di pareggio vince il livello più basso (come nel paper)."""
    conteggi = predizioni.value_counts()
    return conteggi[conteggi == conteggi.max()].index.min()


def per_testata(articoli):
    return articoli.groupby(["fold", "domain"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", voto), articoli=("y_pred", "size"))


def tabella_per_fold(dati):
    t = pd.DataFrame([{"fold": f, **metriche(g.y_true, g.y_pred)} for f, g in dati.groupby("fold")]).set_index("fold")
    media, std = t.mean(), t.std()
    t.loc["media"], t.loc["dev. std"] = media, std
    return t.round(3)


AVANZAMENTO = RESULTS + "avanzamento.log"


def registra(messaggio):
    """Aggiunge una riga al file di avanzamento, da seguire con: tail -f Results/2-classificazione/avanzamento.log"""
    with open(AVANZAMENTO, "a") as f:
        f.write(f"{datetime.now():%H:%M:%S}  {messaggio}\n")


def classificatore_cv(df, colonna_fold, colonna_testo="testo", etichetta=""):
    """TF-IDF + regressione logistica, fold per fold. Restituisce le predizioni per articolo."""
    pred = pd.Series(-1, index=df.index)
    for fold in sorted(df[colonna_fold].unique()):
        registra(f"{etichetta} fold {fold}: inizio")
        train = df[df[colonna_fold] != fold]
        test = df[df[colonna_fold] == fold]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
                                max_features=100_000, strip_accents="unicode")
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(tfidf.fit_transform(train[colonna_testo]), train.label)
        pred[test.index] = clf.predict(tfidf.transform(test[colonna_testo]))
        registra(f"{etichetta} fold {fold}: fatto ({len(train)} articoli di training, {len(test)} di test)")
    return pred

"""Classificatore a feature (TF-IDF + LR) sul dataset definitivo: due input, due letture.

Input (addestramenti):
  A  — solo testo degli articoli
  C  — testo degli articoli + testo delle pagine del sito della testata, attaccato in coda

Letture della stessa predizione per articolo:
  per articolo        — accuratezza su tutti gli articoli
  per testata (voto)  — voto di maggioranza degli articoli della testata

Fold: per gruppo (principale) e per testata (confronto). Salva TUTTO:
  Results/2-classificazione/feature_<A|C>_<schema>_articoli.csv
    con p_l1..p_l5 (probabilità per classe), y_pred, y_true, domain, fold
  Results/2-classificazione/feature_metriche.csv  (le 4 letture, 5 classi e binario)
Log passo-passo: Results/2-classificazione/avanzamento_feature.log
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

QUI = Path(__file__).resolve().parent
VERS = QUI.parent / "Dataset" / "versioni"
RESULTS = QUI.parent / "Results" / "2-classificazione"
LOG = RESULTS / "avanzamento_feature.log"
SEED = 42
CLASSI = ["p_l1", "p_l2", "p_l3", "p_l4", "p_l5"]
SCHEMI = ["fold_gruppo", "fold_testata"]


def registra(m):
    riga = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
    print(riga, flush=True)
    with open(LOG, "a") as f:
        f.write(riga + "\n")


def dati():
    art = pd.read_csv(VERS / "dataset_finale.csv")
    art["testo"] = art["testo"].fillna("")
    fold = pd.read_csv(VERS / "fold_finale.csv")[["domain", "fold_gruppo", "fold_testata"]]
    art = art.merge(fold, on="domain", how="left")
    pagine = pd.read_csv(VERS / "pagine_finale.csv")[["domain", "testo_pagine"]]
    testo_pag = dict(zip(pagine.domain, pagine.testo_pagine.fillna("")))
    # C = articolo + pagine della sua testata (le stesse pagine per ogni articolo della testata)
    art["testo_AC"] = art.testo + "\n\n" + art.domain.map(testo_pag).fillna("")
    return art


def tolerant(y, p, a):
    d = np.abs(np.asarray(y) - np.asarray(p))
    return float(np.mean(np.where(d == 0, 1.0, np.where(d == 1, a, 0.0))))


def quattro_letture(art):
    """5-classi e binario, per articolo e per testata (voto di maggioranza)."""
    out = {}
    # per articolo
    out["articolo 5cl"] = accuracy_score(art.y_true, art.y_pred)
    out["articolo F1"] = f1_score(art.y_true, art.y_pred, average="macro")
    out["articolo bin"] = accuracy_score(art.y_true >= 2, art.y_pred >= 2)
    out["articolo tol1"] = tolerant(art.y_true, art.y_pred, 1.0)
    # per testata (voto)
    def voto(s):
        c = s.value_counts()
        return c[c == c.max()].index.min()
    pub = art.groupby("domain").agg(y_true=("y_true", "first"), y_pred=("y_pred", voto))
    out["testata 5cl"] = accuracy_score(pub.y_true, pub.y_pred)
    out["testata F1"] = f1_score(pub.y_true, pub.y_pred, average="macro")
    out["testata bin"] = accuracy_score(pub.y_true >= 2, pub.y_pred >= 2)
    out["testata tol1"] = tolerant(pub.y_true, pub.y_pred, 1.0)
    return {k: round(v, 3) for k, v in out.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["A", "C"])
    ap.add_argument("--schemi", nargs="+", default=SCHEMI)
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="quali fold (default tutti)")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    art = dati()
    registra(f"=== dataset definitivo: {len(art)} articoli, {art.domain.nunique()} testate ===")
    righe = []
    colonne = {"A": "testo", "C": "testo_AC"}
    for config in args.config:
        colonna = colonne[config]
        for schema in args.schemi:
            file_out = RESULTS / f"feature_{config}_{schema}_articoli.csv"
            if file_out.exists():
                registra(f"{config} {schema}: già fatto, salto")
                pred = pd.read_csv(file_out)
            else:
                proba = np.zeros((len(art), 5))
                folds = args.folds if args.folds else sorted(art[schema].unique())
                for fold in folds:
                    t0 = time.time()
                    tr, te = art[art[schema] != fold], art[art[schema] == fold]
                    registra(f"{config} {schema} fold {fold}: inizio ({len(tr)} training, {len(te)} test)")
                    v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
                                        max_features=100_000, strip_accents="unicode")
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
                    clf.fit(v.fit_transform(tr[colonna]), tr.label)
                    p = clf.predict_proba(v.transform(te[colonna]))
                    proba[te.index.to_numpy()[:, None], clf.classes_[None, :]] = p
                    acc = accuracy_score(te.label, p.argmax(1))
                    registra(f"{config} {schema} fold {fold}: fatto in {time.time()-t0:.0f} s, acc articoli {acc:.3f}")
                pred = pd.DataFrame(proba, columns=CLASSI)
                pred.insert(0, "domain", art.domain.values)
                pred.insert(0, "article_id", art.article_id.values)
                pred["fold"] = art[schema].values
                pred["y_true"] = art.label.values
                pred["y_pred"] = proba.argmax(1)
                pred = pred[pred.fold.isin(folds)].reset_index(drop=True)   # solo i fold davvero eseguiti
                pred.to_csv(file_out, index=False)
                registra(f"=== {config} {schema}: salvato ({len(pred)} righe con probabilità) ===")
            etich = "per gruppo" if schema == "fold_gruppo" else "per testata"
            righe.append({"config": config, "fold": etich, **quattro_letture(pred)})
    ris = pd.DataFrame(righe).set_index(["config", "fold"])
    ris.to_csv(RESULTS / "feature_metriche.csv")
    registra("=== FINITO ===\n" + ris.to_string())


if __name__ == "__main__":
    main()

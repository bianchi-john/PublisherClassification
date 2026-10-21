"""Dati, percorsi e letture comuni ai notebook di classificazione."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

VERSIONI = Path("../Dataset/versioni")
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
        with open(log, "a", encoding="utf-8") as f:
            f.write(riga + "\n")
    return registra


def carica(prova=False, solo_valutazione=False):
    """Gli articoli del dataset finale, con i fold e il peso della loro testata.

    Versione 2: ci sono anche le testate di solo training (`valutazione` = False), che non vanno mai nel test.
    Con `solo_valutazione` si tengono solo le testate valutate, come nella prima versione (per il confronto).
    In prova tiene 8 testate valutate per livello (40 in tutto) e 20 di solo training, per controllare che il codice giri.
    """
    art = pd.read_csv(VERSIONI / "dataset_finale.csv")
    art["text"] = art["text"].fillna("")
    fold = pd.read_csv(VERSIONI / "fold_finale.csv")[["domain", "peso"] + SCHEMI]
    art = art.merge(fold, on="domain", how="left")
    if solo_valutazione:
        art = art[art.valutazione]
        art["peso"] = 1.0
    if prova:
        t = art.drop_duplicates("domain")
        scelte = set(t[t.valutazione].groupby("label").sample(8, random_state=SEED).domain)
        extra = t[~t.valutazione]
        scelte |= set(extra.sample(min(20, len(extra)), random_state=SEED).domain)
        art = art[art.domain.isin(scelte)]
    return art.reset_index(drop=True)


def pesi_bilanciati(label, peso):
    """Peso di ogni riga: il peso della sua testata (un gruppo vale al più 5 testate) per il bilanciamento delle classi,
    calcolato sui pesi (come class_weight="balanced", che però conterebbe le righe senza peso)."""
    label, peso = np.asarray(label), np.asarray(peso, dtype=float)
    per_classe = pd.Series(peso).groupby(label).sum()
    return peso * peso.sum() / (len(per_classe) * per_classe.reindex(label).to_numpy())


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


def per_testata_media(pred):
    """Testata = livello con la probabilità media più alta sui suoi articoli."""
    m = pred.groupby("domain")[CLASSI].mean()
    return pd.DataFrame({"y_true": pred.groupby("domain").y_true.first(), "y_pred": m.to_numpy().argmax(1)}, index=m.index)


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


def metriche_di(y, p):
    """Tutte le metriche di un vettore di livelli veri y e predetti p (0..4)."""
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    m = {"n": len(y), "accuratezza": accuracy_score(y, p),
         "f1_macro": f1_score(y, p, average="macro", labels=range(5), zero_division=0),
         "f1_pesata": f1_score(y, p, average="weighted", labels=range(5), zero_division=0),
         "tolerant_0.5": tolerant(y, p, 0.5), "tolerant_1": tolerant(y, p, 1.0),
         "errore_medio_livelli": float(np.abs(y - p).mean()),
         "confusione": confusion_matrix(y, p, labels=range(5)).tolist()}
    pr, rc, f1, su = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    for k in range(5):
        m.update({f"l{k + 1}_precisione": pr[k], f"l{k + 1}_recall": rc[k], f"l{k + 1}_f1": f1[k], f"l{k + 1}_supporto": int(su[k])})
    yb, pb = y >= 2, p >= 2                     # binario: affidabile = livello >= 2 (punteggio NewsGuard >= 60)
    tp, tn = int((yb & pb).sum()), int((~yb & ~pb).sum())
    fp, fn = int((~yb & pb).sum()), int((yb & ~pb).sum())
    m.update({"bin_accuratezza": (tp + tn) / len(y), "bin_tp": tp, "bin_tn": tn, "bin_fp": fp, "bin_fn": fn,
              "bin_precisione_affidabile": tp / max(tp + fp, 1), "bin_recall_affidabile": tp / max(tp + fn, 1),
              "bin_f1_affidabile": f1_score(yb, pb, zero_division=0),
              "bin_f1_inaffidabile": f1_score(~yb, ~pb, zero_division=0),
              "bin_f1_macro": f1_score(yb, pb, average="macro", zero_division=0)})
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def metriche(pred):
    """Le metriche per articolo, per testata col voto e per testata con la media delle probabilità."""
    viste = {"articolo": pred, "testata_voto": per_testata(pred)}
    if all(c in pred for c in CLASSI):
        viste["testata_media"] = per_testata_media(pred)
    return {nome: metriche_di(d.y_true, d.y_pred) for nome, d in viste.items()}

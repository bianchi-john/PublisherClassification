"""Versione 4 (titolo + articolo, criteri NewsGuard): dati, percorsi, criteri, divisione di sviluppo e metriche.

Il dataset è quello della Versione 2 (stessa pulizia, stesse 304 testate valutate, stessi fold per gruppo, testate di
solo training con il loro peso). Le metriche sono quelle della Versione 2 (5 livelli, binario, tolerant).
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

VERSIONI = Path("../../../../../Dataset/tentativi/08_Versione4_TitoloCriteri/versioni")
NEWSGUARD = Path("../../../../../Dataset/etichette/newsguard_2022.csv")
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


# i 5 criteri NewsGuard che si vedono negli articoli (gli altri 4 riguardano il sito: proprietà, pubblicità,
# responsabili, autori). 1 = rispettato, 0 = no, NaN = non valutato (escluso dalla loss)
CRITERI = ["Does not repeatedly publish false content", "Gathers and presents information responsibly",
           "Regularly corrects or clarifies errors", "Handles the difference between news and opinion responsibly",
           "Avoids deceptive headlines"]


def criteri(domini):
    """Matrice (n, 5) dei criteri per una lista di domini: 1.0, 0.0 o NaN."""
    ng = pd.read_csv(NEWSGUARD, low_memory=False)
    ng = ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")
    return np.column_stack([pd.Series(list(domini)).map(ng[c]).map({"Yes": 1.0, "No": 0.0}).to_numpy(dtype=float) for c in CRITERI])


def divisione_sviluppo(art, seed=42):
    """Per le prove preliminari: le testate di training del fold 1 (cioè i fold 2-5, il fold 1 non si guarda mai),
    divise per gruppi in training e validazione (1/7 dei gruppi, stratificato per livello). È la stessa divisione che
    02_transformer usa per la validazione del fold 1. Ritorna (indici di training, indici di validazione) di art;
    la validazione tiene solo le testate valutate."""
    from sklearn.model_selection import StratifiedGroupKFold
    tr_tutti = art[art.fold_gruppo != 1]
    t = tr_tutti.drop_duplicates("domain").reset_index(drop=True)
    _, iv = next(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(t, t.label, t.gruppo))
    in_val = set(t.domain.iloc[iv])
    tr = tr_tutti.index[~tr_tutti.domain.isin(in_val)].to_numpy()
    va = tr_tutti.index[tr_tutti.domain.isin(in_val) & tr_tutti.valutazione].to_numpy()
    return tr, va


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

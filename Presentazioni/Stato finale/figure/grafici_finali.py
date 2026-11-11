"""Le ultime slide di FinalState.odp come curve al crescere degli articoli letti per testata (revisione del 5 novembre,
sera 6), con il voto del regressore STIRATO:
- curve_mbfc.png: Spearman del modello con NewsGuard e con MBFC (170 testate in comune); linea = accordo tra i due
  esperti (NewsGuard ↔ MBFC). Cerchiati i punti in cui il modello batte il TF-IDF in modo significativo (bootstrap
  sulle testate, 1.000 volte, stesse estrazioni di articoli, intervallo al 95% tutto sopra 0).
- curve_cls_reg.png: classificatore (pieno) e regressore stirato (tratteggiato, come lo stirato nelle altre slide), stessi articoli estratti per entrambi:
  Spearman e errore sul voto (classificatore: voto atteso), entro 20 punti, F1 macro sui livelli (classificatore:
  livello più votato; regressore: quote del training).
- curve_cls_reg_tol.png: le stesse curve per la tolleranza ½ e ±1 sui livelli (slide in più).
Per ogni k: k articoli a caso per testata, 20 volte, poi la media; «tutti» = tutti gli articoli.
Uso, da questa cartella:  python grafici_finali.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import f1_score

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
D20 = REPO / "Dataset/tentativi/20_FromScratch"
sys.path.insert(0, str(REPO / "Code/Pipeline"))
import mbfc as MB      # noqa: E402
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402
from common import majority   # noqa: E402

OUT = QUI / "valutazione"
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834"}
GRIGIO = "#52514e"
PROBS = [f"p_l{i}" for i in range(1, 6)]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
K = [1, 2, 3, 5, 10, 20, 50, "tutti"]
X = [1, 2, 3, 5, 10, 20, 50, 160]
B = 1000

art, testate = MO.load()
dominio = testate.index
fold, vero, livello = testate.fold.to_numpy(), testate.score.to_numpy(), testate.label.to_numpy()
medie_livello = {f: testate[testate.fold != f].groupby("label").score.mean().reindex(range(5)).to_numpy() for f in np.unique(fold)}
y_mb = MB.load(testate)                                          # classe MBFC delle 170 testate in comune
comuni = dominio.get_indexer(y_mb.index)
articoli = art.groupby("domain").article_id.apply(np.array).reindex(dominio)


def tabella(m):
    r = MO.predictions("regression", m).set_index("article_id").prediction
    c = MO.predictions("classification", m).set_index("article_id")
    return r, c[PROBS], c.prediction


def tolleranza_mezzo(y, p):
    """Tolleranza ½ (come nei tentativi 14 e 16): livello giusto = 1, sbagliato di uno = ½, altrimenti 0."""
    d = np.abs(np.asarray(y) - np.asarray(p))
    return np.where(d == 0, 1, np.where(d == 1, 0.5, 0)).mean()


rng = np.random.default_rng(42)
righe, mbfc_vettori = [], {}
for m in MODELLI:
    reg, probs, voto_art = tabella(m)
    for k in K:
        for r in range(1 if k == "tutti" else 20):
            scelti = [a if k == "tutti" else rng.choice(a, size=min(k, len(a)), replace=False) for a in articoli]
            reg_m = np.array([reg.loc[a].mean() for a in scelti])
            stir = ME.remapped_score(reg_m, fold, vero)
            P = np.vstack([probs.loc[a].to_numpy().mean(0) for a in scelti])
            atteso = np.array([P[i] @ medie_livello[f] for i, f in enumerate(fold)])
            liv_cls = np.array([majority(voto_art.loc[a].to_numpy()) for a in scelti])
            liv_reg = ME.levels_quotas(reg_m, fold, livello)
            righe.append({"modello": m, "k": str(k), "r": r,
                          "reg Spearman": spearmanr(vero, stir).correlation, "cls Spearman": spearmanr(vero, atteso).correlation,
                          "reg MAE": np.mean(np.abs(stir - vero)), "cls MAE": np.mean(np.abs(atteso - vero)),
                          "reg entro 20": 100 * np.mean(np.abs(stir - vero) <= 20), "cls entro 20": 100 * np.mean(np.abs(atteso - vero) <= 20),
                          "reg F1": f1_score(livello, liv_reg, average="macro", labels=range(5), zero_division=0),
                          "cls F1": f1_score(livello, liv_cls, average="macro", labels=range(5), zero_division=0),
                          "reg tol½": tolleranza_mezzo(livello, liv_reg), "cls tol½": tolleranza_mezzo(livello, liv_cls),
                          "reg tol±1": np.mean(np.abs(livello - liv_reg) <= 1), "cls tol±1": np.mean(np.abs(livello - liv_cls) <= 1)})
            mbfc_vettori.setdefault((m, str(k)), []).append(stir[comuni])
curve = pd.DataFrame(righe).drop(columns="r").groupby(["modello", "k"], sort=False).mean().reset_index()
curve.to_csv(OUT / "curve_cls_reg.csv", index=False)

# MBFC: Spearman del modello (stirato) con NewsGuard e con MBFC, bootstrap sulle 170 testate
ng, mb = vero[comuni], y_mb.to_numpy()
idx = np.vstack([np.arange(len(comuni))[None, :], np.random.default_rng(7).integers(0, len(comuni), (B, len(comuni)))])


def spearman_righe(Y, P):
    ry, rp = rankdata(Y, axis=1), rankdata(P, axis=1)
    ry -= ry.mean(1, keepdims=True); rp -= rp.mean(1, keepdims=True)
    return (ry * rp).sum(1) / np.sqrt((ry ** 2).sum(1) * (rp ** 2).sum(1))


def boot(p):
    return {"NewsGuard": spearman_righe(ng[idx], p[idx]), "MBFC": spearman_righe(mb[idx], p[idx])}


chiavi = [(c, r) for c, v in mbfc_vettori.items() for r in range(len(v))]
ris = Parallel(n_jobs=8)(delayed(boot)(mbfc_vettori[c][r]) for c, r in chiavi)
S = {}
for (c, r), x in zip(chiavi, ris):
    for chi, v in x.items():
        S.setdefault((*c, chi), []).append(v)
S = {c: np.mean(v, axis=0) for c, v in S.items()}
esperti = spearmanr(ng, mb).correlation
righe_mb = []
for m in MODELLI:
    for k in K:
        for chi in ["NewsGuard", "MBFC"]:
            d = S[(m, str(k), chi)] - S[("TF-IDF", str(k), chi)]
            lo, hi = np.percentile(d[1:], [2.5, 97.5])
            righe_mb.append({"modello": m, "k": str(k), "con": chi, "Spearman": S[(m, str(k), chi)][0],
                             "meno TF-IDF": d[0], "da": lo, "a": hi, "batte TF-IDF": m != "TF-IDF" and lo > 0})
cmb = pd.DataFrame(righe_mb); cmb.to_csv(OUT / "curve_mbfc.csv", index=False)

fig, assi = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, chi, titolo in [(assi[0], "NewsGuard", "Il modello con NewsGuard"), (assi[1], "MBFC", "Il modello con MBFC")]:
    ax.axhline(esperti, color="#111111", ls="--", lw=1.5)
    ax.text(160, esperti - 0.006, f"i due esperti tra loro\n(NewsGuard ↔ MBFC): {esperti:.2f} ".replace(".", ","), fontsize=10, va="top", ha="right")
    for m in MODELLI:
        d = cmb[(cmb.modello == m) & (cmb.con == chi)].set_index("k").reindex([str(k) for k in K])
        ax.plot(X, d.Spearman, "-", lw=1.8, color=COLORE[m], label=m); ax.scatter(X, d.Spearman, s=14, color=COLORE[m], zorder=3)
        v = d["batte TF-IDF"].to_numpy(bool)
        ax.scatter(np.array(X)[v], d.Spearman.to_numpy()[v], s=90, color=COLORE[m], edgecolor="#111111", linewidth=1.6, zorder=4)
    ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
    ax.set_xlabel("articoli letti per testata"); ax.set_title(f"{titolo}: Spearman sulle {len(comuni)} testate in comune")
assi[0].set_ylabel("Spearman")
assi[1].scatter([], [], s=90, color="white", edgecolor="#111111", linewidth=1.6, label="batte il TF-IDF\n(in modo significativo)")
assi[1].legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "curve_mbfc.png", bbox_inches="tight"); plt.close()

tutte = pd.read_csv(D20 / "all_metrics.csv")
base = tutte[tutte.approach == "baseline"]
senza = {"MAE": base[base.space == "score"].MAE.iloc[0], "entro 20": 100 * base[base.space == "score"]["within 20 points"].iloc[0],
         "F1": base[base.space == "levels"]["F1 macro"].iloc[0]}
fig, assi = plt.subplots(1, 4, figsize=(18, 5.2))
for ax, (mis, titolo) in zip(assi, [("Spearman", "Spearman sul voto"), ("MAE", "Errore medio sul voto (punti)"),
                                    ("entro 20", "% testate con errore ≤ 20"), ("F1", "Livelli l1-l5: F1 macro")]):
    for m in MODELLI:
        d = curve[curve.modello == m].set_index("k").reindex([str(k) for k in K])
        ax.plot(X, d[f"cls {mis}"], "o-", ms=4, lw=1.8, color=COLORE[m], label=m)
        ax.plot(X, d[f"reg {mis}"], "o--", ms=4, lw=1.5, color=COLORE[m])
    if mis in senza:
        ax.axhline(senza[mis], color=GRIGIO, ls=":", lw=1.2); ax.text(1, senza[mis], " baseline: voto medio", va="bottom", fontsize=9, color=GRIGIO)
    ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
    ax.set_xlabel("articoli letti per testata"); ax.set_title(titolo)
assi[0].legend(frameon=False, fontsize=9, loc="lower right")
from matplotlib.lines import Line2D   # noqa: E402
assi[3].legend(handles=[Line2D([], [], color="k", ls="-", label="classificatore"), Line2D([], [], color="k", ls="--", label="regressore (stirato)")],
               frameon=False, fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "curve_cls_reg.png", bbox_inches="tight"); plt.close()

# le due tolleranze sui livelli, in una figura a parte (slide in più)
frequente = np.array([np.bincount(livello[fold != f], minlength=5).argmax() for f in fold])   # baseline: livello più frequente del training
base_tol = {"tol½": tolleranza_mezzo(livello, frequente), "tol±1": np.mean(np.abs(livello - frequente) <= 1)}
fig, assi = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, (mis, titolo) in zip(assi, [("tol½", "Livelli: tolleranza ½ (giusto = 1, di un livello = ½)"),
                                    ("tol±1", "Livelli: tolleranza ±1 (giusto se sbaglia al massimo di uno)")]):
    for m in MODELLI:
        d = curve[curve.modello == m].set_index("k").reindex([str(k) for k in K])
        ax.plot(X, d[f"cls {mis}"], "o-", ms=4, lw=1.8, color=COLORE[m], label=m)
        ax.plot(X, d[f"reg {mis}"], "o--", ms=4, lw=1.5, color=COLORE[m])
    ax.axhline(base_tol[mis], color=GRIGIO, ls=":", lw=1.2)
    ax.text(1, base_tol[mis], " baseline: livello più frequente", va="bottom", fontsize=9, color=GRIGIO)
    ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
    ax.set_xlabel("articoli letti per testata"); ax.set_title(titolo)
assi[0].legend(frameon=False, fontsize=9, loc="lower right")
assi[1].legend(handles=[Line2D([], [], color="k", ls="-", label="classificatore"), Line2D([], [], color="k", ls="--", label="regressore (quote del training)")],
               frameon=False, fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "curve_cls_reg_tol.png", bbox_inches="tight"); plt.close()
print("baseline tolleranze:", {k: round(v, 3) for k, v in base_tol.items()})

pd.set_option("display.width", 220)
print("esperti:", round(esperti, 3))
print(cmb[cmb.k.isin(["1", "5", "tutti"])].round(3).to_string(index=False))
print(curve[curve.k.isin(["1", "5", "tutti"])].round(3).to_string(index=False))

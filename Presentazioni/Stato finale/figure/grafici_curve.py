"""I risultati dei regressori come curve: come cambiano le misure all'aumentare degli articoli letti per testata
(revisione del 5 novembre, sera 4-5). Per ogni k si prendono a caso k articoli di ogni testata (20 volte, poi la media);
«tutti» = tutti gli articoli, cioè il risultato finale.
Evidenziati (cerchio grande contornato) i punti in cui Qwen, Mistral o ModernBERT battono il TF-IDF in modo
significativo: differenza modello − TF-IDF sulle stesse testate ricampionate (bootstrap, 1.000 volte) e sulle stesse
estrazioni di articoli, intervallo al 95% tutto dalla parte «meglio» (errore più basso, le altre misure più alte).
Il voto stirato si confronta con il TF-IDF stirato.
- curve_normale.png, curve_confronto.png; curve.csv (valori), curve_significative.csv (differenze e intervalli).
Uso, da questa cartella:  python grafici_curve.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, rankdata

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/Pipeline"))
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834"}
GRIGIO = "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
K = [1, 2, 3, 5, 10, 20, 50, "tutti"]
X = [1, 2, 3, 5, 10, 20, 50, 160]                        # «tutti» disegnato a destra di 50
MISURE = [("Spearman", "Spearman (ordine)"), ("Kendall", "Kendall (ordine, a coppie)"),
          ("MAE", "Errore medio (punti)"), ("entro 20", "% testate con errore ≤ 20")]
B = 1000

art, testate = MO.load()
fold = testate.fold
vero = testate.score.to_numpy()


def spearman_righe(Y, P):
    """Spearman di ogni riga (Y, P matrici: una riga per campione bootstrap)."""
    ry, rp = rankdata(Y, axis=1), rankdata(P, axis=1)
    ry -= ry.mean(1, keepdims=True); rp -= rp.mean(1, keepdims=True)
    return (ry * rp).sum(1) / np.sqrt((ry ** 2).sum(1) * (rp ** 2).sum(1))


def misure_boot(p, idx):
    """Le quattro misure sui dati veri (riga 0) e su ogni campione bootstrap (righe 1..B)."""
    tutti = np.vstack([np.arange(len(vero))[None, :], idx])
    Y, P = vero[tutti], np.asarray(p)[tutti]
    return {"Spearman": spearman_righe(Y, P), "Kendall": np.array([kendalltau(y, q).correlation for y, q in zip(Y, P)]),
            "MAE": np.abs(P - Y).mean(1), "entro 20": 100 * (np.abs(P - Y) <= 20).mean(1)}


# 1. le estrazioni di articoli (stesso ordine e stesso seme di prima: stessi valori)
rng = np.random.default_rng(42)
previsioni = {}                                           # (modello, k, voto) -> lista di vettori (una per estrazione)
for m in MODELLI:
    gruppi = {d: g.prediction.to_numpy() for d, g in MO.predictions("regression", m).groupby("domain")}
    for k in K:
        for r in range(1 if k == "tutti" else 20):
            prev = pd.Series({d: (v if k == "tutti" else rng.choice(v, size=min(k, len(v)), replace=False)).mean()
                              for d, v in gruppi.items()}).reindex(testate.index)
            previsioni.setdefault((m, str(k), "normale"), []).append(prev.to_numpy())
            previsioni.setdefault((m, str(k), "stirato"), []).append(ME.remapped_score(prev, fold, testate.score))

# 2. le misure su dati veri e campioni bootstrap (gli stessi campioni per tutti)
idx = np.random.default_rng(7).integers(0, len(vero), (B, len(vero)))
chiavi = [(c, r) for c, lista in previsioni.items() for r in range(len(lista))]
risultati = Parallel(n_jobs=8)(delayed(misure_boot)(previsioni[c][r], idx) for c, r in chiavi)
M = {}
for (c, r), ris in zip(chiavi, risultati):
    for mis, v in ris.items():
        M.setdefault((*c, mis), []).append(v)
M = {c: np.mean(v, axis=0) for c, v in M.items()}         # media sulle estrazioni: riga 0 = valore, righe 1.. = bootstrap

righe, sig = [], []
for m in MODELLI:
    for k in K:
        for voto in ["normale", "stirato"]:
            righe.append({"modello": m, "k": str(k), "voto": voto, **{mis: M[(m, str(k), voto, mis)][0] for mis, _ in MISURE}})
            if m == "TF-IDF":
                continue
            for mis, _ in MISURE:
                d = M[(m, str(k), voto, mis)] - M[("TF-IDF", str(k), voto, mis)]
                lo, hi = np.percentile(d[1:], [2.5, 97.5])
                meglio = hi < 0 if mis == "MAE" else lo > 0
                sig.append({"modello": m, "k": str(k), "voto": voto, "misura": mis, "differenza": d[0], "da": lo, "a": hi, "batte TF-IDF": meglio})
curve, sig = pd.DataFrame(righe), pd.DataFrame(sig)
curve.to_csv(OUT / "curve.csv", index=False); sig.to_csv(OUT / "curve_significative.csv", index=False)
senza_p = np.array([testate.score[fold != f].mean() for f in fold])          # senza testo: a tutte la media del training
senza = {"MAE": np.mean(np.abs(senza_p - vero)), "entro 20": 100 * np.mean(np.abs(senza_p - vero) <= 20)}


def disegna(voti, nome):
    fig, assi = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax, (misura, titolo) in zip(assi, MISURE):
        for m in MODELLI:
            for voto, stile in [("normale", "-"), ("stirato", "--")]:
                if voto not in voti:
                    continue
                d = curve[(curve.modello == m) & (curve.voto == voto)].set_index("k").reindex([str(k) for k in K])
                ax.plot(X, d[misura], stile, lw=1.8, color=COLORE[m], label=m if voto == voti[0] else None)
                ax.scatter(X, d[misura], s=14, color=COLORE[m], zorder=3)
                if m != "TF-IDF":
                    s = sig[(sig.modello == m) & (sig.voto == voto) & (sig.misura == misura)].set_index("k").reindex([str(k) for k in K])
                    vince = s["batte TF-IDF"].to_numpy(bool)
                    ax.scatter(np.array(X)[vince], d[misura].to_numpy()[vince], s=90, color=COLORE[m], edgecolor="#111111",
                               linewidth=1.6, zorder=4)
        if misura in senza:
            ax.axhline(senza[misura], color=GRIGIO, ls=":", lw=1.2)
            ax.text(1, senza[misura], " baseline: voto medio", va="bottom", fontsize=9, color=GRIGIO)
        ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
        ax.set_xlabel("articoli letti per testata"); ax.set_title(titolo)
    assi[0].legend(frameon=False, fontsize=9, loc="lower right")
    assi[1].scatter([], [], s=90, color="white", edgecolor="#111111", linewidth=1.6, label="batte il TF-IDF\n(in modo significativo)")
    if len(voti) == 2:
        assi[1].plot([], [], "k-", label="normale"); assi[1].plot([], [], "k--", label="stirato")
    assi[1].legend(frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


disegna(["normale"], "curve_normale.png")
disegna(["normale", "stirato"], "curve_confronto.png")
disegna(["stirato"], "curve_stirato.png")
pd.set_option("display.width", 200)
print(curve[curve.k.isin(["1", "5", "tutti"])].round(3).to_string(index=False))
t = sig.pivot_table(index=["voto", "modello", "misura"], columns="k", values="batte TF-IDF", aggfunc="first")[[str(k) for k in K]]
print(t.replace({True: "SÌ", False: "."}).to_string())

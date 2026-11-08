"""Le altre strategie sugli embedding congelati (tentativo 20, passo 10: boost.py), per FinalState.odp (revisione dell'8
novembre). Solo dai risultati già calcolati (Dataset/tentativi/20_FromScratch/boost/results.csv, scritto da 10_boost.ipynb):
- boost_strategie.png: Mistral, tutti gli articoli, voto stirato: Spearman ed errore medio di ognuno dei 14 metodi, con
  il TF-IDF e il ridge di prima come riferimento;
- boost_curve.png: le curve al crescere degli articoli (voto stirato) del TF-IDF, del ridge di Mistral, della rete
  neurale su Mistral e di «auto» (il metodo scelto, fold per fold, sui fold interni: il numero onesto); cerchiati i punti
  in cui battono il TF-IDF in modo significativo (bootstrap appaiato sulle testate).
Uso, da questa cartella:  python grafici_boost.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
R = pd.read_csv(REPO / "Dataset/tentativi/20_FromScratch/boost/results.csv", dtype={"k": str})
OUT = QUI / "valutazione"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
NOMI = {"ridge": "ridge (quello di prima)", "ridge_outlet": "ridge, testate a pari peso", "ridge_lds": "ridge, più peso ai voti rari",
        "ridge_median": "ridge, mediana degli articoli", "ridge_quantile": "ridge sulla posizione del voto",
        "outlet_mean": "ridge sul vettore medio della testata", "outlet_bags": "vettore medio + gruppi casuali",
        "outlet_mixup": "vettore medio + testate sintetiche", "outlet_krr": "kernel sulla testata", "outlet_knn": "testate più simili",
        "krr": "kernel sugli articoli", "mlp": "rete neurale sugli articoli", "mil": "attenzione sugli articoli",
        "mil_rank": "attenzione, addestrata sull'ordine"}
TF, BLU, VERDE, VIOLA, GRIGIO = "#8a8983", "#1d4f91", "#1baf7a", "#7b5ea7", "#52514e"
v = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")


def valore(fonte, misura, k="all", voto="stretched"):
    d = R[(R.source == fonte) & (R.k == k) & (R.score == voto) & (R.measure == misura)]
    return d.value.iloc[0]


# ------------------------------------------------------------------------------------------------ 1. le 14 strategie
fig, assi = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
ordine = sorted(NOMI, key=lambda m: valore(f"Mistral-24B · {m}", "Spearman"))
y = np.arange(len(ordine))
for ax, misura, titolo in [(assi[0], "Spearman", "Spearman (più alto è meglio)"), (assi[1], "MAE", "Errore medio, punti (più basso è meglio)")]:
    x = [valore(f"Mistral-24B · {m}", misura) for m in ordine]
    colori = [BLU if m == "ridge" else VERDE if m == "mlp" else "#9fb8d6" for m in ordine]
    ax.hlines(y, [min(x) if misura == "Spearman" else max(x)] * len(y), x, color="#e1e0d9", lw=1)
    ax.scatter(x, y, s=60, c=colori, zorder=3)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.28, v(xi, 3 if misura == "Spearman" else 1), ha="center", fontsize=8, color=GRIGIO)
    tf = valore("TF-IDF", misura)
    ax.axvline(tf, color=TF, ls="--", lw=1.5); ax.text(tf, len(y) - 0.4, " TF-IDF", color=GRIGIO, fontsize=10, va="bottom")
    ax.set_title(titolo); ax.grid(axis="y", visible=False)
assi[0].set_yticks(y, [NOMI[m] for m in ordine])
assi[0].set_xlim(0.6, 0.77)
fig.suptitle("Mistral, tutti gli articoli, voto stirato: i 14 metodi", x=0.01, ha="left", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.savefig(OUT / "boost_strategie.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------------------------------------ 2. le curve
K = [1, 2, 3, 5, 10, 20, 50, "all"]
X = [1, 2, 3, 5, 10, 20, 50, 160]
FONTI = [("TF-IDF", "TF-IDF", TF), ("Mistral-24B · ridge", "Mistral, ridge di prima", BLU),
         ("Mistral-24B · mlp", "Mistral, rete neurale", VERDE), ("auto · all", "auto (metodo scelto sui fold interni)", VIOLA)]
fig, assi = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, misura, titolo in [(assi[0], "Spearman", "Spearman (ordine)"), (assi[1], "MAE", "Errore medio (punti)")]:
    for fonte, nome, colore in FONTI:
        d = R[(R.source == fonte) & (R.score == "stretched") & (R.measure == misura)].set_index("k").reindex([str(k) for k in K])
        ax.plot(X, d.value, "-", lw=1.8, color=colore, label=nome); ax.scatter(X, d.value, s=14, color=colore, zorder=3)
        if fonte != "TF-IDF":
            vince = d["beats TF-IDF"].fillna(False).to_numpy(bool)
            ax.scatter(np.array(X)[vince], d.value.to_numpy()[vince], s=90, color=colore, edgecolor="#111111", linewidth=1.6, zorder=4)
    ax.set_xscale("log"); ax.set_xticks(X, ["1", "2", "3", "5", "10", "20", "50", "tutti"]); ax.minorticks_off()
    ax.set_xlabel("articoli letti per testata"); ax.set_title(f"{titolo}, voto stirato")
assi[0].scatter([], [], s=90, color="white", edgecolor="#111111", linewidth=1.6, label="batte il TF-IDF\n(in modo significativo)")
assi[0].legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "boost_curve.png", bbox_inches="tight"); plt.close()
for fonte, nome, _ in FONTI:
    print(nome, {m: round(valore(fonte, m), 3) for m in ["Spearman", "Kendall", "MAE", "within 20"]})

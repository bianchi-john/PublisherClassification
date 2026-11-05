"""Lo split vecchio (fold per livello, tentativo 16) e quello nuovo (fold per voto, tentativo 20) a confronto, in un solo
grafico: per ogni fold quante testate di ogni livello e tutti i singoli voti, con il voto medio delle l1 e delle l2.
Uso, da questa cartella:  python grafico_split.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
D = QUI.parents[2] / "Dataset/tentativi"
testate = pd.read_csv(D / "20_FromScratch/outlets.csv", index_col="domain")
SPLIT = {"Vecchio: fold per livello": pd.read_csv(D / "16_DataPreparation/testate.csv", index_col="domain").fold.reindex(testate.index),
         "Nuovo: fold per voto": pd.read_csv(D / "20_FromScratch/split.csv", index_col="domain").fold.reindex(testate.index)}
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
COLORI = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})

fig, assi = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 2.3]})
for riga, (nome, fold) in enumerate(SPLIT.items()):
    # a sinistra: testate per livello in ogni fold
    a = assi[riga, 0]
    conta = pd.crosstab(fold, testate.label).reindex(index=range(1, 6), columns=range(5), fill_value=0)
    sinistra = np.zeros(5)
    for k in range(5):
        a.barh(conta.index, conta[k], left=sinistra, color=COLORI[k], edgecolor="white", label=LIVELLI[k])
        for f, v, s in zip(conta.index, conta[k], sinistra):
            a.text(s + v / 2, f, str(v), ha="center", va="center", fontsize=10, color="white" if k in (0, 4) else "black")
        sinistra += conta[k].to_numpy()
    a.set_yticks(conta.index, [f"fold {f}" for f in conta.index]); a.invert_yaxis(); a.grid(False)
    a.set_title(f"{nome}\ntestate per livello"); a.set_xticks([])
    # a destra: i singoli voti di ogni fold, con il voto medio di l1 e l2
    b = assi[riga, 1]
    r = np.random.default_rng(0)
    for f in range(1, 6):
        t = testate[fold == f]
        b.scatter(f + r.uniform(-0.3, 0.3, len(t)), t.score, s=16, color=[COLORI[k] for k in t.label], alpha=0.85, zorder=2)
        for k in (0, 1):
            m = t.score[t.label == k].mean()
            b.plot([f - 0.38, f + 0.38], [m, m], color=COLORI[k] if k == 0 else "#c45a5a", lw=3, zorder=3)
            b.text(f + 0.41, m, f"{m:.0f}", va="center", fontsize=11, fontweight="bold", color=COLORI[0] if k == 0 else "#c45a5a")
    medie = {k: testate.score[testate.label == k].groupby(fold[testate.label == k]).mean() for k in (0, 1)}
    b.set_title(f"singoli voti; linee: voto medio delle l1 (da {medie[0].min():.0f} a {medie[0].max():.0f}) "
                f"e delle l2 (da {medie[1].min():.0f} a {medie[1].max():.0f})")
    b.set_xticks(range(1, 6), [f"fold {f}" for f in range(1, 6)]); b.set_xlim(0.5, 5.75); b.set_ylim(-3, 103)
    b.set_ylabel("voto NewsGuard"); b.grid(axis="x", visible=False)
assi[1, 0].legend(ncol=5, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(); plt.savefig(QUI / "split_fold.png", bbox_inches="tight"); plt.close()

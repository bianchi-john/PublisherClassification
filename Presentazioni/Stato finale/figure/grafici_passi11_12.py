"""L'ultima prova del tentativo 20 per FinalState.odp (revisione del 9 novembre), dai risultati già calcolati:
- meta_confronto.png (passo 11): Spearman ed errore medio, tutti gli articoli, voto stirato, di ogni modello sugli
  articoli singoli e sui meta-articoli (Dataset/tentativi/20_FromScratch/meta/results.csv).
Contornati: differenze significative (bootstrap appaiato sulle testate, intervallo al 95%).
Uso, da questa cartella:  python grafici_passi11_12.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
D = QUI.parents[2] / "Dataset/tentativi/20_FromScratch"
OUT = QUI / "valutazione"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
NOMI = {"TF-IDF": "TF-IDF", "Qwen3-8B · ridge": "Qwen, ridge", "Qwen3-8B · ridge_outlet": "Qwen, ridge a pari peso",
        "Qwen3-8B · mlp": "Qwen, rete neurale", "Mistral-24B · ridge": "Mistral, ridge",
        "Mistral-24B · ridge_outlet": "Mistral, ridge a pari peso", "Mistral-24B · mlp": "Mistral, rete neurale",
        "ModernBERT": "ModernBERT"}
PRIMA, DOPO = "#8a8983", "#1d4f91"
v = lambda x, m: f"{x:.2f}".replace(".", ",") if m == "Spearman" else f"{x:.1f}".replace(".", ",")


def coppie(ax, modelli, a, b, segnato, misura, etichette):
    y = np.arange(len(modelli))
    for i, (x0, x1, s) in enumerate(zip(a, b, segnato)):
        ax.plot([x0, x1], [i, i], color="#c3c2b7", lw=2, zorder=1)
        ax.scatter(x1, i, s=110 if s else 60, color=DOPO, edgecolor="#111111" if s else DOPO, linewidth=1.6, zorder=3)
        ax.text(x1, i + 0.28, v(x1, misura), ha="center", fontsize=8, color=DOPO)
    ax.scatter(a, y, s=60, color=PRIMA, zorder=2)
    ax.set_yticks(y, [NOMI[m] for m in modelli]); ax.set_ylim(len(y) - 0.5, -0.8); ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=60, color=PRIMA, label=etichette[0]); ax.scatter([], [], s=60, color=DOPO, label=etichette[1])
    ax.scatter([], [], s=110, color="white", edgecolor="#111111", linewidth=1.6, label="differenza significativa")


# ------------------------------------------------------------------------------------------------ passo 11
T = pd.read_csv(D / "meta/results.csv")
T = T[T.score == "stretched"]
modelli = ["TF-IDF", "Qwen3-8B · ridge", "Qwen3-8B · mlp", "Mistral-24B · ridge", "Mistral-24B · mlp", "ModernBERT"]
fig, assi = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
for ax, misura, titolo in [(assi[0], "Spearman", "Spearman (più alto è meglio)"), (assi[1], "MAE", "Errore medio, punti (più basso è meglio)")]:
    d = T[T.measure == misura]
    art = d[d.input == "articles"].set_index("model").value.reindex(modelli)
    met = d[d.input == "meta"].set_index("model").reindex(modelli)
    coppie(ax, modelli, art.to_numpy(), met.value.to_numpy(), (met.better | met.worse).fillna(False).to_numpy(bool),
           misura, ["articoli singoli", "meta-articoli"])
    ax.set_title(titolo)
fig.legend(*assi[1].get_legend_handles_labels(), frameon=False, fontsize=10, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Tutti gli articoli, voto stirato: articoli singoli e meta-articoli", x=0.01, ha="left", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.savefig(OUT / "meta_confronto.png", bbox_inches="tight"); plt.close()

print("ok")

"""Grafici per le risposte ai commenti di John (5 novembre, sera):
- modelli/penalita_conta.png: quanto conta la penalità (Spearman interno, media dei 5 giri): la peggiore della griglia,
  la migliore penalità fissa (uguale in tutti i giri) e quella scelta giro per giro;
- modelli/bert_comporta.png: ModernBERT contro gli altri sul test: errore per articolo e Spearman per testata.
Uso, da questa cartella:  python grafici_commenti.py"""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
D20 = REPO / "Dataset/tentativi/20_FromScratch"
sys.path.insert(0, str(REPO / "Code/Pipeline"))
import models as MO    # noqa: E402

OUT = QUI / "modelli"
PEGGIORE, FISSA, SCELTA, BERT, ALTRI = "#b02a2a", "#a8a69c", "#1d4f91", "#eb6834", "#1d4f91"
plt.rcParams.update({"figure.dpi": 150, "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left"})
virgola = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")

# 1. quanto conta la penalità
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B"]
fig, ax = plt.subplots(figsize=(12, 3.6))
for i, m in enumerate(MODELLI):
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(D20 / f"predictions/regression/{m}/penalty_fold*.csv"))))
    t = d.pivot(index="penalty", columns="fold", values="inner score")
    valori = [(t.min().mean(), "penalità peggiore della griglia", PEGGIORE),
              (t.mean(axis=1).max(), "migliore penalità fissa, uguale in tutti i giri", FISSA),
              (d[d.chosen]["inner score"].mean(), "scelta giro per giro (quella usata)", SCELTA)]
    ax.plot([min(v for v, _, _ in valori), max(v for v, _, _ in valori)], [i, i], color="#c3c2b7", lw=2, zorder=1)
    for v, nome, colore in valori:
        ax.scatter(v, i, s=110, color=colore, zorder=3, label=nome if i == 0 else None)
    ax.text(valori[0][0] - 0.004, i, virgola(valori[0][0]), ha="right", va="center", color=PEGGIORE, fontsize=12)
    ax.text(valori[2][0] + 0.004, i, virgola(valori[2][0]), ha="left", va="center", color=SCELTA, fontsize=12, fontweight="bold")
ax.set_yticks(range(3), MODELLI); ax.invert_yaxis(); ax.set_ylim(2.6, -0.6); ax.grid(axis="y", visible=False)
ax.set_xlim(0.52, 0.75); ax.set_xlabel("Spearman medio delle prove interne (solo testate di training), media dei 5 giri")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=11)
ax.set_title("Una penalità sbagliata costa; scegliere giro per giro dà quanto la migliore penalità fissa")
plt.tight_layout(); plt.savefig(OUT / "penalita_conta.png", bbox_inches="tight"); plt.close()

# 2. ModernBERT: che cosa comporta impararsi le testate
art, testate = MO.load()
ic = pd.read_csv(D20 / "intervals.csv")
NOMI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
errore = {}
for m in NOMI:
    p = MO.predictions("regression", m).merge(art[["article_id", "score"]], on="article_id")
    errore[m] = np.sqrt(np.mean((p.prediction - p.score) ** 2))
bert_train = np.mean([np.sqrt(pd.read_csv(D20 / f"predictions/regression/ModernBERT/loss_fold{k}.csv").loss.iloc[-1]) * 100
                      for k in range(1, 6)])
fig, (a, b) = plt.subplots(1, 2, figsize=(17, 3.9))
etichette = ["ModernBERT\nsul training"] + [f"{m}\nsul test" for m in NOMI]
valori = [bert_train] + [errore[m] for m in NOMI]
colori = [BERT] + [ALTRI] * 3 + [BERT]
a.bar(range(5), valori, color=colori, alpha=0.9)
a.patches[0].set_hatch("//"); a.patches[0].set_alpha(0.45)
for k, v in enumerate(valori):
    a.text(k, v + 0.6, virgola(v, 1), ha="center", fontsize=12, fontweight="bold")
a.set_xticks(range(5), etichette, fontsize=10.5); a.set_ylim(0, 30); a.grid(axis="x", visible=False)
a.set_title("Errore per articolo (punti, scarto quadratico medio)")
for k, m in enumerate(NOMI):
    r = ic[(ic.approach == "regressor") & (ic.view == "mean score") & (ic.metric == "Spearman") & (ic.model == m)].iloc[0]
    colore = BERT if m == "ModernBERT" else ALTRI
    b.plot([r.low, r.high], [k, k], color=colore, lw=2, alpha=0.5)
    b.scatter(r.value, k, s=90, color=colore, zorder=3)
    b.text(r.high + 0.01, k, virgola(r.value), va="center", fontsize=12, color=colore, fontweight="bold")
b.set_yticks(range(4), NOMI); b.invert_yaxis(); b.set_xlim(0.45, 0.85); b.grid(axis="y", visible=False)
b.set_title("Ordine delle testate di test (Spearman, intervallo 95%)")
plt.tight_layout(); plt.savefig(OUT / "bert_comporta.png", bbox_inches="tight"); plt.close()
print("errore per articolo:", {m: round(v, 1) for m, v in errore.items()}, "ModernBERT training:", round(bert_train, 1))

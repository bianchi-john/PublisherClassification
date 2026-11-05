"""Come si confrontano classificatore e regressore (Mistral, giro 1), per la slide prima di «Classificatore o regressore?»:
- a sinistra, classificatore → voto: le probabilità dei 5 livelli di una testata (media dei suoi articoli) per il voto
  medio di ogni livello nel training = voto atteso (metrics.expected_score);
- a destra, regressore → livello: le testate del test in fila per voto previsto, tagliate con le quote dei livelli nel
  training (metrics.levels_quotas).
Uso, da questa cartella:  python grafico_cls_reg_come.py   ->  valutazione/cls_reg_come.png"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

COLORI = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
LIVELLI = ["l1\n0-39", "l2\n40-59", "l3\n60-74", "l4\n75-99", "l5\n100"]
GRIGIO, BLU = "#52514e", "#1d4f91"
plt.rcParams.update({"figure.dpi": 150, "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def virgola(x, d=1):
    x = round(float(x), d)
    return f"{x:.0f}" if x.is_integer() else f"{x:.{d}f}".replace(".", ",")


art, testate = MO.load()
GIRO = 1
train = testate[testate.fold != GIRO]
medie = train.groupby("label").score.mean().reindex(range(5)).to_numpy()          # voto medio di ogni livello nel training
quote = np.bincount(train.label.astype(int), minlength=5) / len(train)

# classificatore: probabilità medie della testata (media dei suoi articoli)
pc = MO.predictions("classification", "Mistral-24B")
P = pc.groupby("domain")[[f"p_l{i}" for i in range(1, 6)]].mean().reindex(testate.index)
atteso = P.to_numpy() @ medie
test = testate.fold == GIRO
# una testata del giro con probabilità sparse su più livelli e voto atteso a metà scala
cand = P[test.to_numpy()].assign(atteso=atteso[test.to_numpy()], massimo=P[test.to_numpy()].max(axis=1))
cand = cand[(cand.atteso > 55) & (cand.atteso < 75)].sort_values("massimo")
d = cand.index[0]
p = P.loc[d].to_numpy()

# regressore: le testate del giro in fila, tagliate con le quote del training
T = ME.outlets_regressor(MO.predictions("regression", "Mistral-24B"), testate)
prev = T.predicted[test.to_numpy()].sort_values()
livello = ME.levels_quotas(T.predicted.to_numpy(), T.fold.to_numpy(), T.true.to_numpy())[test.to_numpy()]
livello = pd.Series(livello, index=T.index[test.to_numpy()]).reindex(prev.index)

fig, (a, b) = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
a.bar(range(5), p * 100, color=COLORI, width=0.65)
for k in range(5):
    a.text(k, p[k] * 100 + 1.2, f"{p[k] * 100:.0f}%", ha="center", fontsize=14, fontweight="bold")
a.set_xticks(range(5), [f"{LIVELLI[k]}\nmedia {virgola(medie[k])}" for k in range(5)], fontsize=12)
a.set_ylim(0, max(p) * 100 + 12); a.set_yticks([]); a.spines["left"].set_visible(False)
conto = " + ".join(f"{p[k]:.2f}".replace(".", ",") + f"×{virgola(medie[k], 0)}" for k in range(5))
a.set_title("Classificatore → voto: una testata del giro 1")
a.text(0.5, -0.36, f"voto atteso = {conto}\n= {virgola(atteso[testate.index.get_loc(d)])}   (voto NewsGuard vero: {virgola(testate.score[d])})",
       transform=a.transAxes, ha="center", va="top", fontsize=13, color=BLU, fontweight="bold")

x = np.arange(len(prev))
b.scatter(x, prev, s=30, color=[COLORI[k] for k in livello], zorder=3)
tagli = np.cumsum(np.bincount(livello, minlength=5))[:-1]
for t in tagli:
    b.axvline(t - 0.5, color=GRIGIO, lw=1.2, ls="--")
inizio = np.concatenate([[0], tagli]); fine = np.concatenate([tagli, [len(prev)]])
for k in range(5):
    b.text((inizio[k] + fine[k] - 1) / 2, 97, f"{LIVELLI[k].split()[0]}\n{quote[k] * 100:.0f}%", ha="center", va="top", fontsize=13,
           color=COLORI[k], fontweight="bold")
b.set_ylim(35, 100); b.set_xlim(-1, len(prev)); b.set_xticks([])
b.set_xlabel(f"le {len(prev)} testate del giro 1, in fila per voto previsto"); b.set_ylabel("voto previsto (Mistral)")
b.set_title("Regressore → livello: tagli alle quote dei livelli nel training")
b.grid(axis="y", color="#e1e0d9")
plt.tight_layout(w_pad=4); plt.savefig(QUI / "valutazione" / "cls_reg_come.png", bbox_inches="tight"); plt.close()
print("testata:", d, "p", p.round(3), "medie", medie.round(1), "atteso", round(atteso[testate.index.get_loc(d)], 1), "vero", testate.score[d],
      "| quote", (quote * 100).round(0), "tagli", tagli)

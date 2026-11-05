"""Come si stira, su un giro vero (Mistral, giro 1), per la slide «Come si stira» di FinalState.odp:
- in ascissa la posizione in fila (%), in ordinata il voto;
- grigio chiaro: i voti NewsGuard veri delle 181 testate del training, in fila dal più basso al più alto;
- grigio scuro: le 45 testate del test con il loro voto previsto, nella loro posizione in fila;
- blu: il voto stirato = il voto del training alla stessa posizione (metrics.remapped_score).
Uso, da questa cartella:  python grafico_come_stirare.py   ->  valutazione/come_stirare.png"""
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

PREVISTO, STIRATO, TRAINING, GRIGIO = "#8a8983", "#1d4f91", "#c3c2b7", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
virgola = lambda x: f"{x:.1f}".replace(".", ",")

art, testate = MO.load()
T = ME.outlets_regressor(MO.predictions("regression", "Mistral-24B"), testate)
T["stirato"] = ME.remapped_score(T.predicted, T.fold, T["true score"])
GIRO = T.fold.min()
test = T[T.fold == GIRO].sort_values("predicted")
training = np.sort(T.loc[T.fold != GIRO, "true score"].to_numpy())
n = len(test)
test["posizione"] = (test.predicted.rank().to_numpy() - 1) / (n - 1) * 100
assert np.allclose(test.stirato, np.quantile(training, test.posizione / 100))

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(np.linspace(0, 100, len(training)), training, color=TRAINING, lw=3, zorder=1,
        label=f"voti veri del training ({len(training)} testate), in fila")
ax.vlines(test.posizione, test.predicted, test.stirato, color=STIRATO, lw=0.8, alpha=0.35, zorder=2)
ax.scatter(test.posizione, test.predicted, s=34, color=PREVISTO, zorder=3, label=f"voto previsto ({n} testate del test)")
ax.scatter(test.posizione, test.stirato, s=34, color=STIRATO, zorder=4, label="voto stirato")

# l'esempio: l'8ª testata in fila
e = test.iloc[7]
ax.scatter([e.posizione] * 2, [e.predicted, e.stirato], s=110, facecolors="none", edgecolors="#c9211e", lw=1.8, zorder=5)
ax.annotate(f"8ª di {n} → posizione {e.posizione:.0f}%\nprevisto {virgola(e.predicted)} → stirato {virgola(e.stirato)}\n"
            f"(il voto del training al {e.posizione:.0f}%)", xy=(e.posizione, e.stirato), xytext=(22, 2),
            fontsize=15, color="#c9211e", arrowprops=dict(arrowstyle="->", color="#c9211e", lw=1.2))

ax.set_xlim(-2, 102); ax.set_ylim(-4, 104)
ax.set_xlabel("posizione in fila (%): 0 = la testata più bassa, 100 = la più alta")
ax.set_ylabel("voto")
ax.set_title(f"Mistral, giro {GIRO}: i previsti (grigi) stanno tra {test.predicted.min():.0f} e {test.predicted.max():.0f}, "
             "gli stirati (blu) tra 0 e 100", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=14)
plt.tight_layout()
plt.savefig(QUI / "valutazione" / "come_stirare.png", bbox_inches="tight")
print(test[["predicted", "posizione", "stirato", "true score"]].round(1).iloc[[0, 7, 22, 44]])

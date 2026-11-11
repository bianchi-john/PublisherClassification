"""Grafici delle slide sui modelli: un esempio giocattolo del TF-IDF e le parole più pesanti del regressore TF-IDF vero
(primo fold del tentativo 20, file tfidf_top_terms.csv preso dal server). Uso, da questa cartella: python grafici_modelli.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

QUI = Path(__file__).resolve().parent
D20 = QUI.parents[2] / "Dataset/tentativi/20_FromScratch"
OUT = QUI / "modelli"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})

# 1. esempio giocattolo: tre frasi, i pesi TF-IDF delle parole
frasi = ["senator told reporters the bill passed",
         "shocking truth media hides about the bill",
         "senator said the bill vote was delayed"]
v = TfidfVectorizer(sublinear_tf=True)
X = v.fit_transform(frasi).toarray()
nomi = v.get_feature_names_out()
fig, ax = plt.subplots(figsize=(13, 2.6))
ax.imshow(X, cmap="Blues", aspect="auto", vmin=0, vmax=X.max())
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i, j] > 0:
            ax.text(j, i, f"{X[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9,
                    color="white" if X[i, j] > 0.45 else "black")
ax.set_xticks(range(len(nomi)), nomi, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(3), [f"articolo {i + 1}: «{f}»" for i, f in enumerate(frasi)], fontsize=10)
ax.set_title("TF-IDF: un peso per parola; le parole presenti ovunque («the», «bill») pesano poco, quelle rare di più")
plt.tight_layout(); plt.savefig(OUT / "tfidf_esempio.png", bbox_inches="tight"); plt.close()

# 2. le parole che spingono il voto in su e in giù (ridge sul voto, primo fold)
t = pd.read_csv(D20 / "tfidf_top_terms.csv")
t = pd.concat([t.head(12), t.tail(12)]).sort_values("peso")
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(t.termine, t.peso, color=np.where(t.peso > 0, "#1d4f91", "#b02a2a"))
ax.axvline(0, color="#52514e", lw=0.8)
ax.set_xlabel("peso nella regressione (punti di voto per unità di TF-IDF)")
ax.set_title("Regressore TF-IDF: le parole che alzano (blu) e abbassano (rosso) il voto")
ax.grid(axis="x", color="#e1e0d9")
plt.tight_layout(); plt.savefig(OUT / "tfidf_parole.png", bbox_inches="tight"); plt.close()
print(sorted(p.name for p in OUT.glob("*.png")))

# ------------------------------------------------------------------------------------------------ l'addestramento
import glob                                         # noqa: E402
import sys                                          # noqa: E402
from matplotlib.patches import FancyBboxPatch       # noqa: E402

BLU, ARANCIO, VERDE, CHIARO = "#2a78d6", "#eb6834", "#1baf7a", "#a9c9ef"

# 3. la validazione interna: come si sceglie la penalità in un giro
fig, ax = plt.subplots(figsize=(13, 4.6)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)


def blocco(x, y, w, colore, testo, h=0.62, bianco=True):
    ax.add_patch(FancyBboxPatch((x, y), w - 0.06, h, boxstyle="round,pad=0,rounding_size=0.08", color=colore, lw=0))
    ax.text(x + w / 2, y + h / 2, testo, ha="center", va="center", fontsize=10, color="white" if bianco else "black")


ax.text(0, 5.75, "Un giro (fold esterno 1)", fontsize=12, fontweight="bold")
for j in range(4):
    blocco(0.0 + j * 1.6, 4.9, 1.6, BLU, f"fold {j + 2}")
blocco(6.4, 4.9, 1.6, ARANCIO, "fold 1: test")
ax.text(8.2, 5.2, "← il test non si tocca fino alla fine", va="center", fontsize=10, color=ARANCIO)
ax.text(0, 4.35, "Le sole testate di training (~180), divise per voto in 3 gruppi da ~60:", fontsize=11)
for r in range(3):
    y = 3.45 - r * 0.8
    ax.text(-0.05, y + 0.31, f"prova {r + 1}", ha="right", va="center", fontsize=10)
    for g in range(3):
        blocco(0.15 + g * 2.1, y, 2.1, VERDE if g == r else CHIARO, "valuta" if g == r else "addestra", bianco=g == r)
ax.text(6.6, 3.0, "Per ogni penalità della griglia:\n• 3 prove, ognuna valutata con lo Spearman\n  sulle ~60 testate tenute fuori\n• si fa la media delle 3 prove\n→ si tiene la penalità con la media più alta",
        fontsize=10.5, va="center")
ax.annotate("", xy=(3.3, 0.85), xytext=(3.3, 1.75), arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.5))
blocco(0.0, 0.1, 6.4, BLU, "con la penalità scelta: si riaddestra su tutti i 4 fold di training", h=0.65)
blocco(6.5, 0.1, 3.4, ARANCIO, "→ previsione del fold 1", h=0.65)
plt.tight_layout(); plt.savefig(OUT / "cv_interna.png", bbox_inches="tight"); plt.close()

# 4. lo Spearman interno di ogni penalità, per fold
P = D20 / "predictions/regression"
fig, assi = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, m in zip(assi, ["TF-IDF", "Qwen3-8B", "Mistral-24B"]):
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(P / m / "penalty_fold*.csv"))))
    for k, g in d.groupby("fold"):
        ax.plot(g.penalty, g["inner score"], "o-", ms=4, lw=1.2, color=plt.cm.tab10(k - 1), label=f"giro {k}")
        s = g[g.chosen].iloc[0]
        ax.scatter(s.penalty, s["inner score"], s=150, facecolor="none", edgecolor="#111111", lw=1.6, zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("penalità (alpha del ridge)"); ax.set_title(m)
assi[0].set_ylabel("Spearman medio\nsulle 3 prove interne"); assi[2].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Per ogni giro la curva della griglia; cerchiata la penalità scelta", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "penalita.png", bbox_inches="tight"); plt.close()

# 5. ModernBERT: l'errore sul training scende fino a ~4 punti, sul test resta ~18
sys.path.insert(0, str(QUI.parents[2] / "Code/Pipeline"))
import models as MO    # noqa: E402
art, _ = MO.load()
pb = MO.predictions("regression", "ModernBERT").merge(art[["article_id", "score"]], on="article_id")
fig, ax = plt.subplots(figsize=(10, 3.8))
test = []
for k in range(1, 6):
    c = pd.read_csv(P / "ModernBERT" / f"loss_fold{k}.csv")
    ax.plot(c.epoch, np.sqrt(c.loss) * 100, "o-", color=BLU, alpha=0.5, lw=1.2, label="training (durante l'epoca)" if k == 1 else None)
    g = pb[pb.fold == k]
    test.append(np.sqrt(np.mean((g.prediction - g.score) ** 2)))
ax.axhspan(min(test), max(test), color=ARANCIO, alpha=0.25, label="test, a fine addestramento (5 giri)")
ax.axhline(np.mean(test), color=ARANCIO, lw=2)
ax.text(5.05, np.mean(test), f" test {np.mean(test):.0f}", va="center", color=ARANCIO, fontweight="bold")
ax.text(5.05, np.sqrt(0.0018) * 100, " training ~4", va="center", color=BLU, fontweight="bold")
ax.set_xticks(range(1, 6)); ax.set_xlim(0.8, 5.6); ax.set_ylim(0, 30); ax.set_xlabel("epoca")
ax.set_ylabel("errore quadratico medio\nper articolo (punti)"); ax.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1, 0.6))
ax.set_title("ModernBERT regressore: sul training l'errore crolla, sul test no")
plt.tight_layout(); plt.savefig(OUT / "bert_curva.png", bbox_inches="tight"); plt.close()
print("errore di test per fold:", np.round(test, 1))

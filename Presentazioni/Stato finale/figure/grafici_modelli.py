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

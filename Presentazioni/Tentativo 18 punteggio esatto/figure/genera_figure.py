"""Figure delle slide del tentativo 18 (il punteggio esatto): il dataset, gli split vecchi e nuovi, le condizioni dei dati.
Legge Dataset/ nella radice del repo. Uso, da questa cartella:  python genera_figure.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
D = QUI.parents[2] / "Dataset" / "tentativi"
BLU, ARANCIO, GRIGIO, CHIARO, ROSSO = "#2a78d6", "#eb6834", "#52514e", "#b5b3ab", "#c0392b"
COLORI_FOLD = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#b02a2a"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
t = pd.read_csv(D / "16_DataPreparation" / "testate.csv", index_col="domain")
split = pd.read_csv(D / "18_PunteggioEsatto" / "split.csv", index_col=0).reindex(t.index)

# 1. il punteggio delle testate e gli articoli per testata
fig, assi = plt.subplots(1, 2, figsize=(11, 3.4))
assi[0].hist(t.score, bins=np.arange(0, 102.5, 2.5), color=BLU)
for x in [39.5, 59.5, 74.5, 99.5]:
    assi[0].axvline(x, color=GRIGIO, ls=":", lw=1)
assi[0].set_title(f"Il punteggio NewsGuard delle {len(t)} testate"); assi[0].set_xlabel("punteggio (le linee: i confini dei 5 livelli)")
assi[0].set_ylabel("testate")
assi[1].hist(t.articoli, bins=20, color=ARANCIO)
assi[1].set_title(f"Articoli per testata: da {t.articoli.min()} a {t.articoli.max()}, mediana {int(t.articoli.median())}")
assi[1].set_xlabel("articoli"); assi[1].set_ylabel("testate")
plt.tight_layout(); plt.savefig(QUI / "dataset.png", bbox_inches="tight"); plt.close()
print("valori distinti:", t.score.nunique(), "| testate a 100:", (t.score == 100).sum(), "| a 0:", (t.score == 0).sum())

# 2. split vecchi (per livello) e nuovi (per punteggio): media per fold dentro i livelli e media per articolo
righe = []
for nome, f in [("vecchi (per livello)", t.fold), ("nuovi (per punteggio)", split.split1)]:
    for k in range(1, 6):
        te = t[f == k]
        righe.append({"split": nome, "fold": k, "l1": te.score[te.label == 0].mean(), "l2": te.score[te.label == 1].mean(),
                      "l4": te.score[te.label == 3].mean(), "per articolo": np.repeat(te.score, te.articoli).mean()})
eq = pd.DataFrame(righe); eq.round(1).to_csv(QUI / "equilibrio_split.csv", index=False)
print(eq.round(1).to_string(index=False))
fig, assi = plt.subplots(1, 3, figsize=(13, 3.4), sharex=True)
for ax, col, titolo in zip(assi, ["l1", "l2", "per articolo"], ["punteggio medio delle testate l1", "punteggio medio delle testate l2",
                                                                "punteggio medio per articolo"]):
    for nome, colore in [("vecchi (per livello)", CHIARO), ("nuovi (per punteggio)", BLU)]:
        d = eq[eq.split == nome]
        ax.plot(d.fold, d[col], "o-", color=colore, lw=2, label=nome)
    ax.set_title(titolo, fontsize=10); ax.set_xticks(range(1, 6), [f"fold {k}" for k in range(1, 6)])
assi[0].legend(fontsize=8.5)
plt.tight_layout(); plt.savefig(QUI / "split_vecchi_nuovi.png", bbox_inches="tight"); plt.close()

# 3. come si costruisce lo split: testate in ordine di punteggio, a blocchi di 5, ognuna in un fold diverso
o = t.sort_values("score").head(30)
fig, ax = plt.subplots(figsize=(11, 2.4))
for i, (d, r) in enumerate(o.iterrows()):
    ax.scatter(i, r.score, s=60, color=COLORI_FOLD[int(split.split1[d]) - 1], zorder=3)
for i in range(0, 30, 5):
    ax.axvspan(i - 0.45, i + 4.45, color="#f0efe9" if (i // 5) % 2 == 0 else "white", zorder=0)
ax.set_xticks([]); ax.set_ylabel("punteggio"); ax.grid(False)
ax.set_title("Le 30 testate con il punteggio più basso, in ordine: in ogni blocco di 5 un colore per fold, ognuno una volta")
plt.tight_layout(); plt.savefig(QUI / "come_si_fa_lo_split.png", bbox_inches="tight"); plt.close()

# 4. le condizioni dei dati: quanti articoli vede il modello per ogni testata (un fold di training)
tr = t[split.split1 != 1].sort_values("articoli", ascending=False)
fig, assi = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)
x = np.arange(len(tr))
for ax, (titolo, peso) in zip(assi, [("A: tutti gli articoli", tr.articoli), ("B: ogni testata pesa uguale", np.full(len(tr), tr.articoli.mean())),
                                     ("C: al più 10 articoli", np.minimum(tr.articoli, 10)), ("D: al più 30 articoli", np.minimum(tr.articoli, 30))]):
    ax.bar(x, peso, width=1.0, color=BLU); ax.set_title(titolo, fontsize=10); ax.set_xticks([]); ax.grid(axis="x", visible=False)
    ax.set_xlabel("testate di training, dalla più prolifica")
assi[0].set_ylabel("peso in addestramento\n(in articoli)")
plt.tight_layout(); plt.savefig(QUI / "condizioni.png", bbox_inches="tight"); plt.close()
print("figure scritte")

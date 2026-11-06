"""La parte MBFC di FinalState.odp, rifatta più semplice (revisione del 6 novembre, 2):
- mbfc_distribuzioni.png: le etichette delle 170 testate in comune, NewsGuard (5 livelli) e MBFC (Factual Reporting, 6 classi);
- mbfc_adattamento.png: MBFC da 6 classi a 4, NewsGuard da 5 livelli a 4 fasce, e l'abbinamento Low ↔ l1 … High ↔ l4-l5;
- mbfc_incrocio.png: classe MBFC contro fascia NewsGuard (diagonale = stessa fascia);
- mbfc_confronto.png: Spearman e Kendall con il voto vero NewsGuard, sulle stesse 170 testate: MBFC (classe 1-4) e i
  regressori stirati ridotti alle stesse 4 fasce (tagli 39,5 / 59,5 / 74,5), così hanno la stessa grana di MBFC.
Differenze modello − MBFC: bootstrap sulle testate (1.000 volte), intervallo al 95% (stampato, non disegnato).
Uso, da questa cartella:  python grafici_mbfc.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import kendalltau, spearmanr

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import mbfc as MB      # noqa: E402
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
COLORI_LIV = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
COLORI_4 = ["#b02a2a", "#e0a43a", "#5b9be0", "#1d4f91"]
COLORE = {"MBFC": "#e0a43a", "TF-IDF": "#8a8983", "Mistral-24B": "#1d4f91"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
v = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
TAGLI = [39.5, 59.5, 74.5]
LIVELLI = ["l1\n0-39", "l2\n40-59", "l3\n60-74", "l4\n75-99", "l5\n100"]
FASCE = ["l1 · 0-39", "l2 · 40-59", "l3 · 60-74", "l4-l5 · 75-100"]
SEI = ["Very Low", "Low", "Mixed", "Mostly Factual", "High", "Very High"]


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


art, testate = MO.load()
mb = MB.load(testate)                                             # classe MBFC 0-3 delle 170 testate in comune
c = testate.index.get_indexer(mb.index)
voto = testate.score.to_numpy()[c]
fascia = np.digitize(voto, TAGLI)
livello = testate.label.to_numpy()[c]
classe = mb.to_numpy()
grezze = pd.read_csv(MB.MBFC)
grezze["d"] = grezze.dominio.str.lower().str.replace(r"^www\.", "", regex=True)
sei = grezze[~grezze.morto].drop_duplicates("d").set_index("d").factual.reindex(mb.index.str.lower()).value_counts().reindex(SEI, fill_value=0)
n = len(mb)

# ------------------------------------------------------------------------------------------------ 1. le due distribuzioni
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4.6))
q = np.bincount(livello, minlength=5)
a.bar(range(5), q, color=COLORI_LIV, edgecolor="white")
for i, x in enumerate(q):
    a.text(i, x, f"{x} ({x / n:.0%})", ha="center", va="bottom", fontsize=11)
a.set_xticks(range(5), LIVELLI); a.grid(axis="x", visible=False)
a.set_title(f"NewsGuard: livello (dal voto 0-100), {n} testate")
colori6 = [COLORI_4[MB.FACTUAL[s]] for s in SEI]
b.bar(range(6), sei, color=colori6, edgecolor="white")
for i, x in enumerate(sei):
    b.text(i, x, f"{x} ({x / n:.0%})", ha="center", va="bottom", fontsize=11)
b.set_xticks(range(6), [s.replace(" ", "\n") for s in SEI]); b.grid(axis="x", visible=False)
b.set_title(f"MBFC: Factual Reporting (una classe), le stesse {n} testate")
alto = max(q.max(), sei.max()) * 1.15
a.set_ylim(0, alto); b.set_ylim(0, alto); a.set_ylabel("testate")
salva("mbfc_distribuzioni.png")

# ------------------------------------------------------------------------------------------------ 2. l'adattamento
fig, ax = plt.subplots(figsize=(16, 6.2))
ax.axis("off"); ax.set_xlim(-0.2, 13.2); ax.set_ylim(-0.9, 6.3)
Y4 = [0, 1.5, 3, 4.5]
L, H = 2.4, 0.55                                                     # larghezza e altezza di una casella


def casella(x, y, testo, colore, chiaro=False):
    ax.add_patch(FancyBboxPatch((x - L / 2, y - H / 2), L, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=colore, edgecolor="white", lw=1.5, alpha=0.35 if chiaro else 1))
    scuro = colore in (COLORI_4[0], COLORI_4[3], COLORI_LIV[4])
    ax.text(x, y, testo, ha="center", va="center", fontsize=11, color="white" if scuro and not chiaro else "black")


def freccia(x0, y0, x1, y1, **k):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#8a8983", lw=1.2, shrinkA=0, shrinkB=0, **k))


X0, X1, X2, X3 = 1.2, 4.6, 8.4, 11.8
y6 = [-0.33, 0.33, 1.5, 3, 4.17, 4.83]
for s, y in zip(SEI, y6):
    casella(X0, y, f"{s} ({sei[s]})", COLORI_4[MB.FACTUAL[s]], chiaro=s in ("Very Low", "Very High"))
    freccia(X0 + L / 2, y, X1 - L / 2, Y4[MB.FACTUAL[s]])
cl = np.bincount(classe, minlength=4)
for k, (nome, y) in enumerate(zip(MB.CLASSES, Y4)):
    casella(X1, y, f"{nome} ({cl[k]})", COLORI_4[k])
fa = np.bincount(fascia, minlength=4)
for k, (nome, y) in enumerate(zip(FASCE, Y4)):
    casella(X2, y, f"{nome} ({fa[k]})", COLORI_4[k])
    ax.annotate("", (X2 - L / 2 - 0.05, y), (X1 + L / 2 + 0.05, y),
                arrowprops=dict(arrowstyle="<|-|>", color="#111111", lw=1.8, ls="--"))
y5 = [0, 1.5, 3, 4.17, 4.83]
for k, y in enumerate(y5):
    nome = ["l1 · 0-39", "l2 · 40-59", "l3 · 60-74", "l4 · 75-99", "l5 · 100"][k]
    casella(X3, y, f"{nome} ({np.sum(livello == k)})", COLORI_LIV[k])
    freccia(X3 - L / 2, y, X2 + L / 2, Y4[min(k, 3)])
for x, t in [(X0, "MBFC\nFactual Reporting: 6 classi"), (X1, "MBFC: 4 classi"), (X2, "NewsGuard: 4 fasce"),
             (X3, "NewsGuard: 5 livelli")]:
    ax.text(x, 5.75, t, ha="center", va="center", fontsize=12, fontweight="bold")
ax.text((X1 + X2) / 2, -0.75, "abbinamento: ogni classe MBFC con una fascia NewsGuard", ha="center", fontsize=11)
salva("mbfc_adattamento.png")

# ------------------------------------------------------------------------------------------------ 3. l'incrocio
C = pd.crosstab(classe, fascia).reindex(index=range(4), columns=range(4), fill_value=0).to_numpy()
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())
for i in range(4):
    for j in range(4):
        ax.text(j, i, C[i, j], ha="center", va="center", fontsize=14,
                fontweight="bold" if i == j else "normal", color="white" if C[i, j] > C.max() / 2 else "black")
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=2))
ax.set_xticks(range(4), [f.replace(" · ", "\n") for f in FASCE]); ax.set_yticks(range(4), MB.CLASSES); ax.grid(False)
ax.set_xlabel("fascia NewsGuard (dal voto vero)"); ax.set_ylabel("classe MBFC")
ax.set_title(f"Le {n} testate in comune: contornata la stessa fascia ({np.trace(C)} su {n})")
salva("mbfc_incrocio.png")

# ------------------------------------------------------------------------------------------------ 4. il confronto con i regressori
annotatori = {"MBFC": classe}
for m in ["TF-IDF", "Mistral-24B"]:
    p = MO.predictions("regression", m).groupby("domain").prediction.mean().reindex(testate.index)
    stirato = np.asarray(ME.remapped_score(p, testate.fold, testate.score))
    annotatori[m] = np.digitize(stirato, TAGLI)[c]                  # le stesse 4 fasce: stessa grana di MBFC
MISURE = {"Spearman": lambda y, p: spearmanr(y, p).correlation, "Kendall": lambda y, p: kendalltau(y, p).correlation}
valori = {(a, mis): f(voto, p) for a, p in annotatori.items() for mis, f in MISURE.items()}
idx = np.random.default_rng(7).integers(0, n, (1000, n))
for m in ["TF-IDF", "Mistral-24B"]:
    for mis, f in MISURE.items():
        d = np.array([f(voto[i], annotatori[m][i]) - f(voto[i], classe[i]) for i in idx])
        print(f"{m} − MBFC, {mis}: {valori[(m, mis)] - valori[('MBFC', mis)]:+.3f}, 95% {np.percentile(d, [2.5, 97.5]).round(3)}")
fig, ax = plt.subplots(figsize=(10, 4.6))
larghezza = 0.26
nomi = {"MBFC": "MBFC (4 classi)", "TF-IDF": "TF-IDF stirato, in 4 fasce", "Mistral-24B": "Mistral stirato, in 4 fasce"}
for j, a in enumerate(annotatori):
    for i, mis in enumerate(MISURE):
        x = i + (j - 1) * larghezza
        ax.bar(x, valori[(a, mis)], larghezza * 0.92, color=COLORE[a], label=nomi[a] if i == 0 else None)
        ax.text(x, valori[(a, mis)] + 0.01, v(valori[(a, mis)]), ha="center", va="bottom", fontsize=12)
ax.set_xticks(range(2), [f"{mis}\ncon il voto vero NewsGuard" for mis in MISURE]); ax.grid(axis="x", visible=False)
ax.set_ylim(0, 0.9); ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=3, bbox_to_anchor=(1, 1.12))
ax.set_title(f"Stesse {n} testate", pad=28)
salva("mbfc_confronto.png")
print({k: round(x, 3) for k, x in valori.items()}, "| diagonale", np.trace(C), "| fasce", fa, "| classi", cl)

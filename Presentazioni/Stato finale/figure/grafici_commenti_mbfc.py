"""Grafici per i commenti di John del 6 novembre (3) su FinalState.odp:
- curve_annotatore.png: solo il riquadro «con NewsGuard» di curve_mbfc.png (che John aveva ritagliato perdendo la
  legenda), con la legenda e il titolo che dice che i regressori sono STIRATI; numeri da valutazione/curve_mbfc.csv
  (scritto da grafici_finali.py), nessun nuovo calcolo dei modelli;
- quote_tagli.png: come il regressore diventa un livello con le quote del training (Mistral): a sinistra le quote del
  training del giro 1 e quante testate di test vanno in ogni livello; a destra, su tutte le 226 testate, i livelli veri,
  quelli dai confini fissi applicati al voto previsto e quelli dalle quote.
Uso, da questa cartella:  python grafici_commenti_mbfc.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/Pipeline"))
import mbfc as MB      # noqa: E402
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834"}
COLORI_LIV = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
LIVELLI = ["l1\n0-39", "l2\n40-59", "l3\n60-74", "l4\n75-99", "l5\n100"]
CONFINI = [39.5, 59.5, 74.5, 99.5]
K = [1, 2, 3, 5, 10, 20, 50, "tutti"]
X = [1, 2, 3, 5, 10, 20, 50, 160]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
v = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")
art, testate = MO.load()

# ------------------------------------------------------------------------------------------------ 1. il modello come annotatore
mb = MB.load(testate)
esperti = spearmanr(testate.score.reindex(mb.index), mb).correlation
cmb = pd.read_csv(OUT / "curve_mbfc.csv", dtype={"k": str})
cmb = cmb[cmb.con == "NewsGuard"]
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.axhline(esperti, color="#111111", ls="--", lw=1.5)
ax.text(160, esperti - 0.006, f"MBFC con NewsGuard: {v(esperti, 2)}", fontsize=10, va="top", ha="right")
for m in MODELLI:
    d = cmb[cmb.modello == m].set_index("k").reindex([str(k) for k in K])
    ax.plot(X, d.Spearman, "-", lw=1.8, color=COLORE[m], label=f"{m} stirato")
    ax.scatter(X, d.Spearman, s=14, color=COLORE[m], zorder=3)
    vince = d["batte TF-IDF"].to_numpy(bool)
    ax.scatter(np.array(X)[vince], d.Spearman.to_numpy()[vince], s=90, color=COLORE[m], edgecolor="#111111", linewidth=1.6, zorder=4)
ax.scatter([], [], s=90, color="white", edgecolor="#111111", linewidth=1.6, label="batte il TF-IDF stirato\n(in modo significativo)")
ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
ax.set_xlabel("articoli letti per testata"); ax.set_ylabel("Spearman con NewsGuard")
ax.set_title(f"Regressori stirati: Spearman con NewsGuard sulle {len(mb)} testate in comune con MBFC")
ax.legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "curve_annotatore.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------------------------------------ 2. i tagli alle quote
p = MO.predictions("regression", "Mistral-24B").groupby("domain").prediction.mean().reindex(testate.index).to_numpy()
fold, livello, voto = testate.fold.to_numpy(), testate.label.to_numpy().astype(int), testate.score.to_numpy()
quote_liv = ME.levels_quotas(p, fold, livello)
stirato = ME.remapped_score(p, fold, voto)
uguali = np.mean(quote_liv == np.digitize(stirato, CONFINI))                     # quote = stirare + confini veri
GIRO = 1
te = fold == GIRO
quote = np.bincount(livello[~te], minlength=5) / (~te).sum()
n_test = np.bincount(quote_liv[te], minlength=5)
tagli_l5 = {f: np.quantile(p[fold == f], np.cumsum(np.bincount(livello[fold != f], minlength=5) / (fold != f).sum())[3])
            for f in np.unique(fold)}

fig, (a, b) = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
a.bar(range(5), quote * 100, color=COLORI_LIV, width=0.65)
for k in range(5):
    a.text(k, quote[k] * 100 + 0.8, f"{quote[k]:.0%}".replace(".", ","), ha="center", fontsize=13, fontweight="bold")
    a.text(k, -10, f"→ {n_test[k]}", ha="center", fontsize=13, color=COLORI_LIV[k], fontweight="bold")
a.text(-0.55, -5, f"nel test ({te.sum()} testate):", fontsize=11, color="#52514e")
a.set_xticks(range(5), LIVELLI); a.set_ylim(-14, quote.max() * 100 + 8); a.set_yticks([]); a.grid(False)
a.spines["left"].set_visible(False); a.axhline(0, color="#c3c2b7", lw=0.8)
a.set_title(f"Giro {GIRO}: livelli delle {(~te).sum()} testate di training\ne (sotto) quante delle {te.sum()} testate di test")

veri = np.bincount(livello, minlength=5)
fissi = np.bincount(np.digitize(p, CONFINI), minlength=5)
dalle_quote = np.bincount(quote_liv, minlength=5)
larg = 0.27
for j, (nome, conta, colore) in enumerate([("veri (NewsGuard)", veri, "#c3c2b7"), ("confini fissi sul voto previsto", fissi, "#eb6834"),
                                           ("quote del training", dalle_quote, "#1d4f91")]):
    x = np.arange(5) + (j - 1) * larg
    b.bar(x, conta, larg * 0.92, color=colore, label=nome)
    for xi, c in zip(x, conta):
        b.text(xi, c + 1, str(c), ha="center", fontsize=10)
b.set_xticks(range(5), LIVELLI); b.grid(axis="x", visible=False); b.set_ylabel("testate")
b.set_title(f"Mistral, tutte le {len(p)} testate: il voto previsto va da {v(p.min(), 0)} a {v(p.max(), 0)}")
b.legend(frameon=False, fontsize=10, loc="upper left")
plt.tight_layout(w_pad=4); plt.savefig(OUT / "quote_tagli.png", bbox_inches="tight"); plt.close()
print("esperti", round(esperti, 3), "| quote giro 1", (quote * 100).round(1), "test", n_test,
      "| taglio l5 per giro", {f: round(t, 1) for f, t in tagli_l5.items()}, "| veri", veri, "fissi", fissi, "quote", dalle_quote,
      "| quote = stirato + confini:", uguali)

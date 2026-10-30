"""Figure delle slide "dal numero alla classe": come i punteggi previsti dal regressore diventano livelli (confini fissi,
stiramento min-max, quote) e affidabile sì/no. Esempio: Mistral 24B regressore, tutti gli articoli, primo split del
tentativo 18. Legge Dataset/ e le previsioni locali del tentativo 18. Uso, da questa cartella:
    python genera_figure_stiramento.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

QUI = Path(__file__).resolve().parent
R = QUI.parents[2]
D = R / "Dataset" / "tentativi"
GRIGIO, CHIARO, BLU, ROSSO = "#52514e", "#b5b3ab", "#2a78d6", "#c0392b"
COLORI = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]          # l1 … l5: due poli e il grigio in mezzo
SOGLIE = [39.5, 59.5, 74.5, 99.5]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

t = pd.read_csv(D / "16_DataPreparation" / "testate.csv", index_col="domain")
split = pd.read_csv(D / "18_PunteggioEsatto" / "split.csv", index_col=0).reindex(t.index)
pa = pd.read_csv(R / "Code/WebScience2027/Tentativi/18_PunteggioEsatto/Results/lineari/previsioni_articoli.csv.gz")
pa = pa[(pa.modello == "Mistral 24B") & (pa.split == "split1") & (pa.condizione == "A") & (pa.tipo == "regressore")]
T = pd.DataFrame({"vero": t.label, "score": t.score, "fold": split.split1, "media": pa.groupby("domain").previsione.mean()})

# 1. lo schiacciamento: punteggio vero contro media prevista, e le due distribuzioni
fig, assi = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = assi[0]
giusto = (T.media >= 60) == (T.score >= 60)
ax.plot([0, 100], [0, 100], color=CHIARO, lw=1)
ax.axhline(60, color=GRIGIO, ls=":", lw=1); ax.axvline(60, color=GRIGIO, ls=":", lw=1)
ax.scatter(T.score[giusto], T.media[giusto], s=16, color=BLU, label="affidabile sì/no giusto")
ax.scatter(T.score[~giusto], T.media[~giusto], s=16, color=ROSSO, label="sbagliato")
for d, dx, dy in [("lifezette.com", 4, 6), ("americanprogress.org", -38, 10)]:
    ax.annotate(d, (T.score[d], T.media[d]), xytext=(T.score[d] + dx, T.media[d] + dy), fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": GRIGIO, "lw": 0.7})
ax.set_xlim(-3, 103); ax.set_ylim(-3, 115)
ax.set_xlabel("punteggio vero NewsGuard"); ax.set_ylabel("punteggio previsto (media degli articoli)")
ax.set_title(f"226 testate: previste tra {T.media.min():.0f} e {T.media.max():.0f}"); ax.legend(fontsize=8, loc="upper left")
ax = assi[1]
b = np.arange(0, 115, 5)
ax.hist(T.score, bins=b, color=CHIARO, label="punteggio vero")
ax.hist(T.media, bins=b, histtype="step", lw=2, color=BLU, label="punteggio previsto")
ax.set_xlabel("punteggio"); ax.set_ylabel("testate"); ax.legend(fontsize=8.5)
ax.set_title("Lo stesso, come distribuzione: il previsto è più stretto")
plt.tight_layout(); plt.savefig(QUI / "schiacciamento.png", bbox_inches="tight"); plt.close()

# 2. lo stiramento nel fold di lifezette: tre modi di portare le medie previste sulla scala dei livelli
k = int(split.split1["lifezette.com"])
te = T[T.fold == k].sort_values("media"); tr = t[split.split1 != k]
m = te.media.to_numpy(); lo, hi = m.min(), m.max()
quote = np.cumsum(np.bincount(tr.label, minlength=5))[:-1] / len(tr)
tagli = np.quantile(m, quote)
modi = [("1. confini fissi NewsGuard: nessuna trasformazione", np.array(SOGLIE)),
        (f"2. stiramento lineare (min–max): {lo:.1f} → 0, {hi:.1f} → 100", lo + np.array(SOGLIE) / 100 * (hi - lo)),
        ("3. quote del training: si stira ogni livello per conto suo", tagli)]
fig, assi = plt.subplots(3, 1, figsize=(12, 5.4))
for ax, (titolo, confini) in zip(assi, modi):
    ax.grid(False); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-2, 102); ax.set_ylim(-0.35, 1.45); ax.set_xticks([])
    sopra = np.r_[min(lo, 0), confini, max(hi, 100)]                      # i confini dei livelli sulla scala prevista
    sotto = np.r_[0, SOGLIE, 100]                                            # i confini dei livelli NewsGuard
    for i in range(5):
        a, b_ = np.clip(sopra[i], 0, 100), np.clip(sopra[i + 1], 0, 100)
        ax.add_patch(Polygon([[a, 1], [b_, 1], [sotto[i + 1], 0], [sotto[i], 0]], closed=True, color=COLORI[i], alpha=0.35, lw=0))
        ax.plot([a, b_], [1, 1], color=COLORI[i], lw=6, solid_capstyle="butt")
        ax.plot([sotto[i], sotto[i + 1]], [0, 0], color=COLORI[i], lw=6, solid_capstyle="butt")
        ax.text((sotto[i] + sotto[i + 1]) / 2, -0.22, f"l{i + 1}", ha="center", va="center", fontsize=9, color=GRIGIO)
        n = ((m > sopra[i]) & (m <= sopra[i + 1])).sum() if i else (m <= sopra[1]).sum()
        ax.text((a + b_) / 2, 1.13, str(n), ha="center", va="bottom", fontsize=8.5, color=COLORI[i] if i != 2 else GRIGIO)
    ax.scatter(m, np.full(len(m), 1.0), s=14, color="black", zorder=4)
    for v in range(0, 101, 10):
        ax.text(v, 0.9, str(v), ha="center", va="top", fontsize=7, color=GRIGIO)
    ax.text(-2, 1.33, titolo, fontsize=10, fontweight="bold", ha="left")
    ax.text(101, 1.0, "previsto", ha="left", va="center", fontsize=8.5, color=GRIGIO)
    ax.text(101, 0.0, "livelli", ha="left", va="center", fontsize=8.5, color=GRIGIO)
plt.tight_layout(h_pad=0.6); plt.savefig(QUI / "stiramento.png", bbox_inches="tight"); plt.close()
print("fold", k, "testate", len(m), "tagli", np.round(tagli, 1), "min-max", round(lo, 1), round(hi, 1))
print("conteggi quote", np.bincount(np.searchsorted(tagli, m), minlength=5), "veri", np.bincount(te.vero, minlength=5),
      "fisse", np.bincount(np.digitize(m, SOGLIE), minlength=5), "minmax", np.bincount(np.digitize((m - lo) / (hi - lo) * 100, SOGLIE), minlength=5))

# 3. le quote passo passo nel fold di lifezette: medie in ordine, livello vero, tagli
fig, ax = plt.subplots(figsize=(11, 3.2))
liv = np.searchsorted(tagli, m)
rng = np.random.default_rng(0)
ax.scatter(m, te.vero + 1 + rng.uniform(-0.12, 0.12, len(m)), s=40, c=[COLORI[i] for i in liv], edgecolor=GRIGIO, lw=0.6, zorder=3)
for x in tagli:
    ax.axvline(x, color=GRIGIO, ls="--", lw=1)
bordi = np.r_[lo - 2, tagli, hi + 2]
for i in range(5):
    ax.text((bordi[i] + bordi[i + 1]) / 2, 5.55, f"l{i + 1}: {(liv == i).sum()}", ha="center", fontsize=9.5, fontweight="bold",
            color=COLORI[i] if i != 2 else GRIGIO)
ax.annotate("lifezette.com", (te.media["lifezette.com"], 1), xytext=(te.media["lifezette.com"] + 1.5, 1.5), fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": GRIGIO, "lw": 0.7})
ax.set_yticks(range(1, 6), [f"l{i}" for i in range(1, 6)]); ax.set_ylim(0.5, 5.95)
ax.set_xlabel("punteggio previsto per la testata (media dei suoi articoli)"); ax.set_ylabel("livello vero")
ax.set_title(f"Fold {k}: {len(m)} testate di test in ordine, tagliate alle quote del training (colore = livello assegnato)")
plt.tight_layout(); plt.savefig(QUI / "quote_passo_passo.png", bbox_inches="tight"); plt.close()
print("figure scritte")

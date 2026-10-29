"""Figure della presentazione "Criteri NewsGuard e regressione". Legge NewsGuard 2026 e le testate del tentativo 16
(Dataset/ nella radice del repo) e le previsioni per testata del TF-IDF in regressione (Results/feature del tentativo 16).
Uso, da questa cartella:  python genera_figure.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
RADICE = QUI.parents[2]
T16 = RADICE / "Code/WebScience2027/Tentativi/16_DataPreparation"
BLU, GRIGIO, CHIARO, ROSSO, VERDE = "#2a78d6", "#52514e", "#b5b3ab", "#c0392b", "#1baf7a"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

# i 9 criteri: nome breve, punti, se si vede negli articoli
CRITERI = {"Does not publish false or egregiously misleading content at a significant level": ("niente notizie false", 22, "testo"),
           "Gathers and presents information responsibly": ("raccoglie e presenta le informazioni con cura", 18, "testo"),
           "Has effective practices for correcting errors": ("corregge gli errori", 12.5, "sito"),
           "Handles the difference between news and opinion responsibly": ("distingue notizia e opinione", 12.5, "testo"),
           "Avoids deceptive headlines": ("titoli non ingannevoli", 10, "testo"),
           "Website discloses ownership and financing": ("dichiara proprietà e finanziamento", 7.5, "sito"),
           "Clearly labels advertising": ("distingue la pubblicità", 7.5, "sito"),
           "Reveals who's in charge, including any possible conflicts of interest": ("dice chi comanda", 5, "sito"),
           "The site provides names of content creators, along with either contact or biographical information": ("nomi degli autori", 5, "sito")}

ng = pd.read_csv(RADICE / "Dataset/etichette/newsguard_2026.csv", low_memory=False)
ng = ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")
t = pd.read_csv(RADICE / "Dataset/tentativi/16_DataPreparation/testate.csv", index_col="domain")


def sinossi(d):
    X = d[list(CRITERI)].apply(lambda c: c.map({"Yes": 1.0, "No": 0.0})).fillna(1.0)     # criterio non applicato = punti presi
    s = d.Score.astype(float)
    punti = np.array([v[1] for v in CRITERI.values()])
    testo = (X * punti)[[k for k, v in CRITERI.items() if v[2] == "testo"]].sum(axis=1)
    return X, s, testo


X, s, testo = sinossi(ng.loc[t.index])
Xa, sa, _ = sinossi(ng[ng.Score.notna()])
righe = [{"criterio": v[0], "punti": v[1], "tipo": v[2], "soddisfatto": X[k].mean(), "r": np.corrcoef(X[k], s)[0, 1],
          "r tutto NewsGuard": np.corrcoef(Xa[k], sa)[0, 1]} for k, v in CRITERI.items()]
tab = pd.DataFrame(righe).sort_values("r", ascending=False)
tab.round(3).to_csv(QUI / "correlazioni.csv", index=False)
print(tab.round(2).to_string(index=False))
print(f"r punteggio dei 4 criteri sul testo / complessivo: {np.corrcoef(testo, s)[0, 1]:.3f} (226 testate); "
      f"tutti e 4 soddisfatti → affidabili: {(s[testo == 62.5] >= 60).mean():.0%} di {(testo == 62.5).sum()}")

# 1. correlazione di ogni criterio con il punteggio complessivo
fig, ax = plt.subplots(figsize=(8.5, 3.6))
y = np.arange(len(tab))[::-1]
ax.barh(y, tab.r, color=[BLU if x == "testo" else CHIARO for x in tab.tipo], height=0.6)
ax.scatter(tab["r tutto NewsGuard"], y, color=GRIGIO, marker="|", s=160, zorder=3, label="tutte le 11.564 testate valutate da NewsGuard")
for yy, (_, r) in zip(y, tab.iterrows()):
    ax.text(r.r - 0.015, yy, f"{r.r:.2f}".replace(".", ","), ha="right", va="center", color="white", fontsize=9, fontweight="bold")
ax.set_yticks(y, [f"{r.criterio}  ({r.punti:g} punti)".replace(".", ",") for _, r in tab.iterrows()], fontsize=9.5)
ax.set_xlim(0, 1); ax.set_xlabel("correlazione con il punteggio complessivo NewsGuard")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLU), plt.Rectangle((0, 0), 1, 1, color=CHIARO),
                   plt.Line2D([], [], color=GRIGIO, marker="|", ls="", markersize=12)],
          labels=["si vede negli articoli", "pratica del sito", "tutte le testate NewsGuard"], loc="lower right", fontsize=8.5)
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(QUI / "correlazioni.png", bbox_inches="tight"); plt.close()

# 2. i punti dei soli criteri sul testo contro il punteggio complessivo
fig, ax = plt.subplots(figsize=(5.6, 3.6))
rng = np.random.default_rng(0)
ax.scatter(testo + rng.uniform(-0.8, 0.8, len(testo)), s, s=14, color=BLU, alpha=0.6)
ax.axhline(60, color=GRIGIO, ls=":", lw=1); ax.text(1, 61, "affidabile (≥ 60)", fontsize=8, color=GRIGIO)
ax.set_xlabel("punti dei 4 criteri sul testo (al più 62,5)"); ax.set_ylabel("punteggio complessivo NewsGuard")
ax.set_title(f"226 testate · correlazione {np.corrcoef(testo, s)[0, 1]:.2f}".replace(".", ","))
plt.tight_layout(); plt.savefig(QUI / "testo_e_punteggio.png", bbox_inches="tight"); plt.close()

# 3. la regressione: i punteggi previsti stanno stretti intorno alla media
P = pd.read_csv(T16 / "Results/feature/previsioni_testate.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
giusto = (P["punteggio vero"] >= 60) == (P.regressione >= 60)
ax.scatter(P["punteggio vero"], P.regressione, s=14, color=np.where(giusto, BLU, ROSSO), alpha=0.7)
ax.plot([0, 100], [0, 100], color=CHIARO, lw=1)
ax.axhline(60, color=GRIGIO, ls=":", lw=1); ax.axvline(60, color=GRIGIO, ls=":", lw=1)
ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
ax.set_xlabel("punteggio vero NewsGuard"); ax.set_ylabel("punteggio previsto (media degli articoli)")
ax.set_title(f"TF-IDF in regressione · previsti tra {P.regressione.min():.0f} e {P.regressione.max():.0f}")
ax.legend(handles=[plt.Line2D([], [], color=c, marker="o", ls="") for c in [BLU, ROSSO]],
          labels=["affidabile sì/no giusto", "sbagliato"], fontsize=8.5, loc="upper left")
plt.tight_layout(); plt.savefig(QUI / "compressione.png", bbox_inches="tight"); plt.close()

# 4. dalle previsioni ai 5 livelli con le quote del training (un fold come esempio)
k = 1
te = P[P.fold == k].sort_values("regressione")
quote = np.cumsum(np.bincount(P["livello vero"][P.fold != k], minlength=5))[:-1] / (P.fold != k).sum()
tagli = np.quantile(te.regressione, quote)
COLORI = ["#b02a2a", "#ef8f8d", "#86b6ef", "#3987e5", "#1c5cab"]
fig, ax = plt.subplots(figsize=(8.5, 2.9))
for i, (_, r) in enumerate(te.iterrows()):
    ax.scatter(r.regressione, r["livello vero"] + rng.uniform(-0.15, 0.15), s=34, color=COLORI[int(r["livello vero"])],
               edgecolor="k" if r["livello vero"] == r["regressione → quote del training"] else "none", lw=1, zorder=3)
for x in tagli:
    ax.axvline(x, color=GRIGIO, ls="--", lw=1)
bordi = [te.regressione.min() - 1] + list(tagli) + [te.regressione.max() + 1]
for i in range(5):
    ax.text((bordi[i] + bordi[i + 1]) / 2, 4.75, f"l{i + 1}", ha="center", fontsize=10, fontweight="bold", color=COLORI[i])
ax.set_yticks(range(5), [f"l{i + 1}" for i in range(5)]); ax.set_ylim(-0.5, 5.1)
ax.set_xlabel("punteggio previsto per la testata"); ax.set_ylabel("livello vero")
ax.set_title(f"Fold {k}: {len(te)} testate di test in ordine di punteggio previsto, tagliate con le quote di livelli del training",
             fontsize=10)
plt.tight_layout(); plt.savefig(QUI / "quote.png", bbox_inches="tight"); plt.close()
print("quote del training (fold 1):", ", ".join(f"{q:.0%}" for q in np.diff(np.r_[0, quote, 1])), "| tagli:", np.round(tagli, 1))

# 5. dagli articoli alla testata: i punteggi previsti per ogni articolo di due testate di test (ModernBERT regressore)
mb = pd.concat([pd.read_csv(T16 / f"Results/modernbert/regressione/fold{k}.csv") for k in range(1, 6)])
n_art = mb.domain.value_counts()
media = mb.groupby("domain").s_regressione.mean()
candidati = t[(n_art.reindex(t.index) >= 40) & (n_art.reindex(t.index) <= 120)]
esempi = [candidati[candidati.score <= 30].score.idxmin(), candidati[candidati.score >= 95].index[0]]
fig, assi = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
for ax, d in zip(assi, esempi):
    v = mb[mb.domain == d].s_regressione.to_numpy()
    ax.scatter(rng.uniform(-0.3, 0.3, len(v)), v, s=12, color=CHIARO, label="un articolo")
    ax.axhline(media[d], color=BLU, lw=2.2, label=f"media dei {len(v)} articoli: {media[d]:.0f}")
    ax.axhline(t.score[d], color=ROSSO, lw=1.5, ls="--", label=f"punteggio vero NewsGuard: {t.score[d]:.0f}")
    ax.axhline(60, color=GRIGIO, ls=":", lw=1)
    ax.set_xticks([]); ax.set_ylim(-5, 105); ax.grid(axis="x", visible=False)
    ax.set_title(f"testata {'poco' if t.score[d] < 60 else 'molto'} affidabile (livello {t.livello[d]})")
    ax.legend(fontsize=8, loc="lower right" if t.score[d] >= 60 else "upper right")
assi[0].set_ylabel("punteggio previsto (0–100)")
plt.tight_layout(); plt.savefig(QUI / "da_articoli_a_testata.png", bbox_inches="tight"); plt.close()
print("esempi:", {d: (round(t.score[d]), int(n_art[d]), round(media[d], 1)) for d in esempi})

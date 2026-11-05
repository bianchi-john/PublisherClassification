"""Come sbaglia il regressore, testata per testata (Mistral; voto normale, cioè la media degli articoli), per le slide
«Come sbaglia» di FinalState.odp. Niente nomi di testata nei grafici: voto NewsGuard + dominio non vanno in git.
- valutazione/err_stesse.png:  errore con segno di ogni testata, Mistral contro TF-IDF (le stesse testate sono difficili);
- valutazione/err_estreme.png: le 10 testate più sovrastimate e le 10 più sottostimate: voto vero, previsto, criteri
                               NewsGuard falliti, sito di opinione sì/no;
- valutazione/err_non_conta.png: errore medio a parità di voto per tipo di criteri persi, per numero di articoli, per
                                 orientamento.
Legge i 9 criteri da Dataset/etichette/newsguard_2026.csv. Uso, da questa cartella:  python grafici_errori_testata.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
COLORI_FASCE = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
FASCE = ["0-39", "40-59", "60-74", "75-99", "100"]
NERO, BLU, ROSSO, GRIGIO = "#111111", "#1d4f91", "#c9211e", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def virgola(x, d=1):
    x = round(float(x), d)
    t = f"{abs(x):.0f}" if x.is_integer() else f"{abs(x):.{d}f}".replace(".", ",")
    return ("−" if x < 0 else "+" if x > 0 else "") + t


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


art, testate = MO.load()
ng = pd.read_csv(REPO / "Dataset/etichette/newsguard_2026.csv")
ng["dom"] = ng.Domain.str.lower().str.replace(r"^www\.", "", regex=True)
ng = ng.drop_duplicates("dom").set_index("dom").reindex(testate.index)
CRITERI = ng.columns[8:17].tolist()          # i 9 criteri, nell'ordine dei punti: 22, 18, 12.5, 12.5, 10, 7.5, 7.5, 5, 5
PESI = np.array([22, 18, 12.5, 12.5, 10, 7.5, 7.5, 5, 5])
TESTO, ALTRI = [0, 1, 3, 4], [2, 5, 6, 7, 8]  # i criteri che riguardano il testo (in rosso nella slide 2) e gli altri
assert CRITERI[0].startswith("Does not publish false"), CRITERI[0]
fallito = (ng[CRITERI] == "No").to_numpy()
persi_testo, persi_altri = (fallito[:, TESTO] * PESI[TESTO]).sum(1), (fallito[:, ALTRI] * PESI[ALTRI]).sum(1)
opinione = (ng["Opinion/Advocacy Journalism"] == "Yes").to_numpy()

E = {m: (lambda T: T.predicted - T["true score"])(ME.outlets_regressor(MO.predictions("regression", m), testate))
     for m in ["TF-IDF", "Mistral-24B"]}
vero = testate.score
prev = vero + E["Mistral-24B"]
e = E["Mistral-24B"]
fascia = np.digitize(vero, [39.5, 59.5, 74.5, 99.5])

# 1. le stesse testate sono difficili per tutti
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.axhline(0, color="#c3c2b7", lw=1); ax.axvline(0, color="#c3c2b7", lw=1)
ax.plot([-60, 70], [-60, 70], color="#b5b3ab", lw=1.2, ls="--")
for k in range(5):
    m = fascia == k
    ax.scatter(E["TF-IDF"][m], e[m], s=26, color=COLORI_FASCE[k], label=f"voto NewsGuard {FASCE[k]}")
r = np.corrcoef(E["TF-IDF"], e)[0, 1]
peggiori = lambda s: set(s.abs().nlargest(20).index)
comuni = len(peggiori(e) & peggiori(E["TF-IDF"]))
ax.set_xlabel("errore del TF-IDF (previsto − vero, punti)"); ax.set_ylabel("errore di Mistral (punti)")
ax.set_title(f"Ogni punto una testata: correlazione {virgola(r, 2)[1:]}; delle 20 peggiori di Mistral, {comuni} lo sono anche per il TF-IDF")
ax.text(-62, 66, "in alto a destra: sovrastimate da tutti e due\nin basso a sinistra: sottostimate da tutti e due",
        ha="left", va="top", fontsize=13, color=GRIGIO)
ax.legend(frameon=False, fontsize=11, loc="lower right")
salva("err_stesse.png")

# 2. le 10 più sovrastimate e le 10 più sottostimate
fig, assi = plt.subplots(1, 2, figsize=(17, 4.6), sharex=True)
for ax, (titolo, idx) in zip(assi, [("Le 10 più sovrastimate", e.nlargest(10).index), ("Le 10 più sottostimate", e.nsmallest(10).index)]):
    pos = testate.index.get_indexer(idx)
    for i, j in enumerate(pos):
        ax.plot([vero.iloc[j], prev.iloc[j]], [i, i], color="#b5b3ab", lw=3, zorder=1)
        ax.scatter(vero.iloc[j], i, s=80, color=NERO, zorder=3, label="voto NewsGuard" if i == 0 else None)
        ax.scatter(prev.iloc[j], i, s=80, color=BLU, zorder=3, label="voto previsto (Mistral)" if i == 0 else None)
        ax.text(103, i, f"{fallito[j].sum()}/9 criteri falliti · {'opinione' if opinione[j] else 'notizie'}", va="center", fontsize=14.5,
                color=ROSSO if fallito[j][0] else GRIGIO)
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(-3, 103); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("voto")
    ax.set_title(titolo); ax.grid(axis="y", visible=False)
fig.suptitle("nero = voto NewsGuard · blu = voto previsto (Mistral) · in rosso: fallisce «non pubblica contenuti falsi»",
             x=0.01, ha="left", fontsize=15, color=GRIGIO)
plt.tight_layout(w_pad=12); plt.savefig(OUT / "err_estreme.png", bbox_inches="tight"); plt.close()

# 3. che cosa non conta: errore medio assoluto
fig, assi = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
banda = ((vero >= 40) & (vero <= 80)).to_numpy()
quota_altri = np.divide(persi_altri, persi_altri + persi_testo, out=np.zeros(len(vero)), where=(persi_altri + persi_testo) > 0)
gruppi = [("Voto 40-80: dove ha perso i punti", [("soprattutto su criteri\ndi testo", banda & (quota_altri <= 0.5)),
                                                 ("soprattutto su criteri\nnon di testo", banda & (quota_altri > 0.5))]),
          ("Articoli per testata", [(f"{a}-{b}", (testate.articles >= a).to_numpy() & (testate.articles <= b).to_numpy())
                                    for a, b in [(10, 99), (100, 179), (180, 201)]]),
          ("Orientamento (NewsGuard)", [(n, (testate.orientation == o).to_numpy()) for n, o in [("sinistra", "Left"), ("destra", "Right"), ("nessuno", "none")]])]
for ax, (titolo, righe) in zip(assi, gruppi):
    for k, (nome, m) in enumerate(righe):
        v = np.abs(e.to_numpy()[m]).mean()
        ax.bar(k, v, color="#5b9be0", width=0.6)
        ax.text(k, v + 0.4, f"{virgola(v)[1:] if v else '0'}", ha="center", fontsize=13, fontweight="bold")
        ax.text(k, 0.8, f"{m.sum()} testate", ha="center", fontsize=10.5, color="white")
    ax.set_xticks(range(len(righe)), [n for n, _ in righe], fontsize=11.5); ax.set_title(titolo); ax.grid(axis="x", visible=False)
assi[0].set_ylabel("errore medio di Mistral (punti)"); assi[0].set_ylim(0, 22)
salva("err_non_conta.png")

alti = (vero >= 75).to_numpy()
print("r", round(r, 3), "comuni", comuni, "| voto ≥ 75: errore opinione", e[alti & opinione].mean().round(1), "n", (alti & opinione).sum(),
      "notizie", e[alti & ~opinione].mean().round(1), "n", (alti & ~opinione).sum())
print("criteri di testo / non di testo (40-80):", np.abs(e[banda & (quota_altri <= 0.5)]).mean().round(1), np.abs(e[banda & (quota_altri > 0.5)]).mean().round(1),
      "| spearman |e| ~ articoli", round(spearmanr(np.abs(e), testate.articles).correlation, 3))
print("orientamento, errore con segno:", pd.Series(e.to_numpy()).groupby(testate.orientation.to_numpy()).mean().round(1).to_dict())

"""Grafici dello stiramento dei voti e del confronto classificatore / regressore (tentativo 20), per FinalState.odp:
- stirare_dispersione.png: Mistral, voto previsto e voto stirato contro il voto vero;
- stirare_immagine.png: lo stesso gesto su una foto (immagine di esempio di matplotlib): contrasto schiacciato, poi stirato;
- stirare_confronto.png: previsto contro stirato, quattro misure, con la differenza sulle stesse testate (bootstrap);
- cls_vs_reg.png: classificatore contro regressore, sul voto (MAE) e sui livelli (F1 macro).
Legge Dataset/tentativi/20_FromScratch (intervals.csv, classifier_vs_regressor.csv, stirato_vs_previsto.csv).
Uso, da questa cartella:  python grafici_stirare.py"""
import sys
from pathlib import Path

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
D20 = REPO / "Dataset/tentativi/20_FromScratch"
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORI_FASCE = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
GRIGIO, PREVISTO, STIRATO, CLASSIF = "#52514e", "#8a8983", "#1d4f91", "#eb6834"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
virgola = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")
ic = pd.read_csv(D20 / "intervals.csv")
art, testate = MO.load()


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


def valore(approccio, vista, misura, modello):
    r = ic[(ic.approach == approccio) & (ic.view == vista) & (ic.metric == misura) & (ic.model == modello)].iloc[0]
    return r.value, r.low, r.high


# 1. Mistral prima e dopo
T = ME.outlets_regressor(MO.predictions("regression", "Mistral-24B"), testate)
stirato = ME.remapped_score(T.predicted, T.fold, T["true score"])
fascia = np.digitize(T["true score"], [39.5, 59.5, 74.5, 99.5])
fig, assi = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (p, t) in zip(assi, [(T.predicted, "Prima: voto previsto (schiacciato)"), (stirato, "Dopo: voto stirato")]):
    ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
    ax.scatter(T["true score"], p, s=16, c=[COLORI_FASCE[k] for k in fascia])
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_xlabel("voto NewsGuard"); ax.set_ylabel("voto di Mistral")
    ax.set_title(t)
salva("stirare_dispersione.png")

# 2. una foto: contrasto schiacciato verso il centro, poi stirato sulla distribuzione dell'originale
foto = plt.imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)).astype(float).mean(axis=2) / 255
schiacciata = 0.42 + 0.3 * foto                               # come i voti previsti: tutto verso il centro
q = (pd.Series(schiacciata.ravel()).rank().to_numpy() - 1) / (schiacciata.size - 1)
stirata = np.quantile(foto.ravel(), q).reshape(foto.shape)     # lo stesso codice di remapped_score
fig, assi = plt.subplots(2, 2, figsize=(9, 6.2), gridspec_kw={"height_ratios": [3, 1]})
for j, (img, t) in enumerate([(schiacciata, "Prima: grigi schiacciati al centro"), (stirata, "Dopo: grigi stirati")]):
    assi[0, j].imshow(img, cmap="gray", vmin=0, vmax=1); assi[0, j].axis("off"); assi[0, j].set_title(t)
    assi[1, j].hist(img.ravel(), bins=60, range=(0, 1), color="#52514e"); assi[1, j].set_yticks([])
    assi[1, j].set_xlabel("dal nero (0) al bianco (1)")
salva("stirare_immagine.png")

# 3. previsto contro stirato: le quattro misure, con la differenza sulle stesse testate
d = pd.read_csv(D20 / "stirato_vs_previsto.csv")
MISURE = [("MAE", "Errore medio (punti): più basso è meglio", 1), ("within 10 points", "Entro 10 punti (%)", 100),
          ("within 20 points", "Entro 20 punti (%)", 100), ("Spearman", "Spearman (ordine)", 1)]
fig, assi = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
for ax, (misura, titolo, f) in zip(assi, MISURE):
    for i, m in enumerate(MODELLI):
        for j, (vista, colore) in enumerate([("mean score", PREVISTO), ("remapped score", STIRATO)]):
            v, a, b = valore("regressor", vista, misura, m)
            y = i + (j - 0.5) * 0.3
            ax.plot([a * f, b * f], [y, y], color=colore, lw=1.6, alpha=0.5)
            ax.scatter(v * f, y, s=60, color=colore, zorder=3, label=("previsto" if j == 0 else "stirato") if i == 0 else None)
        r = d[(d.model == m) & (d.metric == misura)].iloc[0]
        vero = r.low > 0 or r.high < 0
        testo = f"{'+' if r['diff'] > 0 else '−'}{virgola(abs(r['diff']) * f, 1 if f == 100 or misura == 'MAE' else 3)}"
        ax.text(1.0, i, testo + (" *" if vero else ""), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=10, fontweight="bold" if vero else "normal", color=STIRATO if vero else GRIGIO)
    ax.set_title(titolo); ax.grid(axis="y", visible=False)
assi[0].set_yticks(range(4), MODELLI); assi[0].invert_yaxis()
assi[0].legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.suptitle("Previsto o stirato? A destra di ogni pannello: stirato − previsto sulle stesse testate (* = non è il caso)",
             x=0.01, ha="left", fontweight="bold")
salva("stirare_confronto.png")

# 4. classificatore contro regressore: sul voto e sui livelli
fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True)
for ax, righe, titolo, f, d_ in [
        (a, [("classifier", "expected score", "classificatore (voto atteso)", CLASSIF), ("regressor", "mean score", "regressore", PREVISTO),
             ("regressor", "remapped score", "regressore stirato", STIRATO)], "Sul voto: errore medio (punti), più basso è meglio", "MAE", 1),
        (b, [("classifier", "majority level", "classificatore (livello più votato)", CLASSIF),
             ("regressor", "levels: training quotas", "regressore (quote del training)", STIRATO)], "Sui livelli l1-l5: F1 macro, più alto è meglio", "F1 macro", 2)]:
    for i, m in enumerate(MODELLI):
        for j, (ap, vista, nome, colore) in enumerate(righe):
            v, lo, hi = valore(ap, vista, f, m)
            y = i + (j - (len(righe) - 1) / 2) * 0.25
            ax.plot([lo, hi], [y, y], color=colore, lw=1.6, alpha=0.5)
            ax.scatter(v, y, s=60, color=colore, zorder=3, label=nome if i == 0 else None)
            ax.text(hi, y, "  " + virgola(v, d_), va="center", fontsize=9, color=colore)
    ax.set_title(titolo); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(righe))
a.set_yticks(range(4), MODELLI); a.invert_yaxis()
salva("cls_vs_reg.png")
print(sorted(p.name for p in OUT.glob("stirare_*.png")), "cls_vs_reg.png")

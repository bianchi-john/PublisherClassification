"""La parte dei regressori di FinalState.odp in meno slide (revisione del 5 novembre, sera 3):
- reg_prestazioni.png: il voto normale (media degli articoli), quattro misure con intervallo al 95% e «senza testo»;
- reg_dove.png: Mistral voto vero / previsto e previsto − vero per fascia di voto, tutti i modelli;
- stirare_idea.png: la foto e Mistral, prima e dopo lo stiramento;
- contro_tfidf2.png: ogni modello − TF-IDF, voto normale e voto stirato;
- dna2.png (+ dna2.csv): quanti articoli servono, voto normale (linea piena) e stirato (tratteggiata).
Legge Dataset/tentativi/20_FromScratch. Uso, da questa cartella:  python grafici_regressori.py"""
import sys
from pathlib import Path

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
D20 = REPO / "Dataset/tentativi/20_FromScratch"
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834"}
COLORI_FASCE = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
FASCE = ["0-39", "40-59", "60-74", "75-99", "100"]
GRIGIO, PREVISTO, STIRATO = "#52514e", "#8a8983", "#1d4f91"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
virgola = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")
ic = pd.read_csv(D20 / "intervals.csv")
tutte = pd.read_csv(D20 / "all_metrics.csv")
contro = pd.read_csv(D20 / "models_vs_tfidf.csv")
art, testate = MO.load()
PR = {m: MO.predictions("regression", m) for m in MODELLI}
TR = {m: ME.outlets_regressor(PR[m], testate) for m in MODELLI}
ST = {m: ME.remapped_score(TR[m].predicted, TR[m].fold, TR[m]["true score"]) for m in MODELLI}
y_vero = testate.score.reindex(TR["TF-IDF"].index).to_numpy()
fascia = np.digitize(y_vero, [39.5, 59.5, 74.5, 99.5])
MISURE = [("MAE", "Errore medio (punti), più basso è meglio", 1), ("within 10 points", "Entro 10 punti (%)", 100),
          ("within 20 points", "Entro 20 punti (%)", 100), ("Spearman", "Spearman (ordine)", 1)]


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


# 1. le prestazioni del voto normale
fig, assi = plt.subplots(1, 4, figsize=(17, 6.2), sharey=True)
for ax, (misura, titolo, f) in zip(assi, MISURE):
    for i, m in enumerate(MODELLI):
        r = ic[(ic.approach == "regressor") & (ic.view == "mean score") & (ic.metric == misura) & (ic.model == m)].iloc[0]
        ax.plot([r.low * f, r.high * f], [i, i], color="#b5b3ab", lw=2)
        ax.scatter(r.value * f, i, s=80, color=COLORE[m], zorder=3)
        ax.text(r.high * f, i, "  " + virgola(r.value * f, 1 if f == 100 or misura == "MAE" else 2), va="center", fontsize=10)
    base = tutte[(tutte.approach == "baseline") & (tutte.space == "score")][misura].iloc[0] * f
    if misura == "Spearman":
        ax.set_title(titolo + " (senza testo: 0)"); ax.grid(axis="y", visible=False); continue
    ax.axvline(base, color=GRIGIO, ls=":", lw=1.2)
    ax.text(base, 3.55, f" senza testo: {virgola(base, 1 if f == 100 or misura == 'MAE' else 2)}", fontsize=9, color=GRIGIO)
    ax.set_title(titolo); ax.grid(axis="y", visible=False)
assi[0].set_yticks(range(4), MODELLI); assi[0].invert_yaxis()
salva("reg_prestazioni.png")

# 2. dove sbagliano: Mistral, e previsto − vero per fascia
fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})
T = TR["Mistral-24B"]
a.plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
a.scatter(T["true score"], T.predicted, s=14, c=[COLORI_FASCE[k] for k in fascia])
a.set_xlabel("voto NewsGuard"); a.set_ylabel("voto previsto"); a.set_title("Mistral: voto vero e voto previsto")
for j, m in enumerate(MODELLI):
    d = TR[m].predicted.to_numpy() - y_vero
    b.bar(np.arange(5) + (j - 1.5) * 0.2, [np.mean(d[fascia == k]) for k in range(5)], width=0.2, color=COLORE[m], label=m)
b.axhline(0, color=GRIGIO, lw=0.8)
b.set_xticks(range(5), [f"{f}\n({n} testate)" for f, n in zip(FASCE, np.bincount(fascia, minlength=5))])
b.set_title("Previsto − vero per fascia di voto: sopra 0 sovrastima, sotto sottostima"); b.grid(axis="x", visible=False)
b.legend(ncol=4, frameon=False, fontsize=9, loc="lower left")
salva("reg_dove.png")

# 3. l'idea dello stiramento: la foto e Mistral
foto = plt.imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)).astype(float).mean(axis=2) / 255
schiacciata = 0.42 + 0.3 * foto
q = (pd.Series(schiacciata.ravel()).rank().to_numpy() - 1) / (schiacciata.size - 1)
stirata = np.quantile(foto.ravel(), q).reshape(foto.shape)
fig, assi = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"width_ratios": [0.8, 0.8, 1, 1]})
for ax, img, t in [(assi[0], schiacciata, "Foto: grigi schiacciati"), (assi[1], stirata, "Foto stirata")]:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(t)
for ax, p, t in [(assi[2], T.predicted, "Mistral: voto previsto"), (assi[3], ST["Mistral-24B"], "Mistral: voto stirato")]:
    ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
    ax.scatter(T["true score"], p, s=12, c=[COLORI_FASCE[k] for k in fascia])
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_xlabel("voto NewsGuard"); ax.set_title(t)
salva("stirare_idea.png")

# 4. contro il TF-IDF: voto normale e voto stirato
righe = []
for m in MODELLI[1:]:
    for misura, _, _ in MISURE:
        if misura == "within 10 points":
            continue
        r = contro[(contro.approach == "regressor") & (contro.view == "mean score") & (contro.metric == misura) & (contro.model == m)].iloc[0]
        righe.append({"modello": m, "misura": misura, "voto": "normale", "diff": r.difference, "low": r.low, "high": r.high})
        d, (lo, hi) = ME.bootstrap_difference(("score", y_vero, ST[m]), ("score", y_vero, ST["TF-IDF"]), misura)
        righe.append({"modello": m, "misura": misura, "voto": "stirato", "diff": d, "low": lo, "high": hi})
ct = pd.DataFrame(righe); ct.to_csv(OUT / "contro_tfidf2.csv", index=False)
fig, assi = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
for ax, (misura, titolo, f) in zip(assi, [MISURE[0], MISURE[2], MISURE[3]]):
    for i, m in enumerate(MODELLI[1:]):
        for j, (voto, colore) in enumerate([("normale", PREVISTO), ("stirato", STIRATO)]):
            r = ct[(ct.modello == m) & (ct.misura == misura) & (ct.voto == voto)].iloc[0]
            yy = i + (j - 0.5) * 0.3
            ax.plot([r.low * f, r.high * f], [yy, yy], color=colore, lw=1.6, alpha=0.5)
            ax.scatter(r["diff"] * f, yy, s=60, color=colore, zorder=3, label=voto if i == 0 else None)
            ax.text(r.high * f, yy, "  " + virgola(r["diff"] * f, 1 if f == 100 or misura == "MAE" else 2), va="center", fontsize=9, color=colore)
    ax.axvline(0, color=GRIGIO, ls=":", lw=1.2)
    ax.set_title(titolo.replace(", più basso è meglio", ": sotto 0 = meglio")); ax.grid(axis="y", visible=False)
assi[0].set_yticks(range(3), MODELLI[1:]); assi[0].invert_yaxis()
assi[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.suptitle("Ogni modello − TF-IDF sulle stesse testate (0 = come il TF-IDF), con il voto normale e con quello stirato",
             x=0.01, ha="left", fontweight="bold")
salva("contro_tfidf2.png")

# 5. il DNA: quanti articoli servono, normale e stirato
rng = np.random.default_rng(42)
K = [1, 2, 3, 5, 10, 20, 50]
fold = TR["TF-IDF"].fold
curva = []
for m in MODELLI:
    gruppi = {d: g.prediction.to_numpy() for d, g in PR[m].groupby("domain")}
    for k in K:
        for r in range(20):
            prev = pd.Series({d: rng.choice(v, size=min(k, len(v)), replace=False).mean() for d, v in gruppi.items()}).reindex(fold.index)
            vero = testate.score.reindex(prev.index)
            st = ME.remapped_score(prev, fold, vero)
            for voto, p in [("normale", prev.to_numpy()), ("stirato", st)]:
                curva.append({"modello": m, "k": k, "voto": voto, "Spearman": spearmanr(vero, p).correlation,
                              "MAE": np.mean(np.abs(p - vero.to_numpy()))})
curva = pd.DataFrame(curva).groupby(["modello", "voto", "k"]).mean().reset_index()
curva.to_csv(OUT / "dna2.csv", index=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.4))
for m in MODELLI:
    for voto, stile in [("normale", "-"), ("stirato", "--")]:
        d = curva[(curva.modello == m) & (curva.voto == voto)]
        a.plot(d.k, d.Spearman, "o" + stile, ms=4, color=COLORE[m], label=m if voto == "normale" else None)
        b.plot(d.k, d.MAE, "o" + stile, ms=4, color=COLORE[m])
for ax, t in [(a, "Spearman con il voto vero"), (b, "Errore medio (punti)")]:
    ax.set_xscale("log"); ax.set_xticks(K, [str(k) for k in K]); ax.set_xlabel("articoli letti per testata"); ax.set_title(t)
a.plot([], [], "k-", label="normale"); a.plot([], [], "k--", label="stirato")
a.legend(frameon=False, fontsize=8, ncol=2)
salva("dna2.png")
print(ct.round(3).to_string(index=False))
print(curva[(curva.modello == "Mistral-24B") & curva.k.isin([1, 5, 50])].round(3).to_string(index=False))

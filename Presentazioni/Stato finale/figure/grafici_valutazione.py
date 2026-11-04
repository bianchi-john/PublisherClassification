"""Grafici della valutazione (tentativo 20) per la presentazione «Stato finale»: classificatori, regressori, confronti,
curva del DNA e accordo con Media Bias/Fact Check. Legge le previsioni per articolo in Dataset/tentativi/20_FromScratch.
Uso, da questa cartella:  python grafici_valutazione.py"""
import sys
from pathlib import Path

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

OUT = QUI / "valutazione"; OUT.mkdir(exist_ok=True)
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834", "nessun testo": "#c3c2b7"}
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
COLORI_LIV = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
GRIGIO = "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
virgola = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")

art, testate = MO.load()
ic = pd.read_csv(D20 / "intervals.csv")
tutte = pd.read_csv(D20 / "all_metrics.csv")
coppie = pd.read_csv(D20 / "classifier_vs_regressor.csv")
contro = pd.read_csv(D20 / "models_vs_tfidf.csv")
TC = {m: ME.outlets_classifier(MO.predictions("classification", m), testate) for m in MODELLI}
PR = {m: MO.predictions("regression", m) for m in MODELLI}
TR = {m: ME.outlets_regressor(PR[m], testate) for m in MODELLI}


def foresta(ax, righe, titolo, rif=None, rif_testo="senza testo"):
    """Punti con intervallo al 95%, uno per modello (righe: lista di (nome, valore, da, a))."""
    y = np.arange(len(righe))
    for i, (n, v, a, b) in enumerate(righe):
        ax.plot([a, b], [i, i], color="#b5b3ab", lw=2)
        ax.scatter(v, i, s=80, color=COLORE.get(n, GRIGIO), zorder=3)
        ax.text(b, i, "  " + virgola(v, 1 if abs(v) > 2 else 2), va="center", fontsize=10)
    if rif is not None:
        ax.axvline(rif, color=GRIGIO, ls=":", lw=1.2)
        ax.text(rif, len(righe) - 0.45, f" {rif_testo}: {virgola(rif, 1 if abs(rif) > 2 else 2)}", fontsize=9, color=GRIGIO)
    ax.set_yticks(y, [r[0] for r in righe]); ax.invert_yaxis(); ax.set_title(titolo); ax.grid(axis="y", visible=False)


def righe_ic(approccio, vista, misura):
    d = ic[(ic.approach == approccio) & (ic.view == vista) & (ic.metric == misura)].set_index("model")
    return [(m, d.value[m], d.low[m], d.high[m]) for m in MODELLI]


def base(spazio, misura):
    return tutte[(tutte.approach == "baseline") & (tutte.space == spazio)][misura].iloc[0]


# ------------------------------------------------------------------------------------------------ classificatori
fig, ax = plt.subplots(figsize=(10, 3.4))
foresta(ax, righe_ic("classifier", "majority level", "F1 macro"), "Classificatore: F1 macro sui 5 livelli (per testata)",
        base("levels", "F1 macro"))
plt.tight_layout(); plt.savefig(OUT / "cls_f1.png", bbox_inches="tight"); plt.close()

fig, assi = plt.subplots(2, 2, figsize=(13, 6))
for ax, (misura, titolo) in zip(assi.flat, [("tolerant (±1 level)", "Tolerant (±1 livello)"), ("quadratic kappa", "Kappa quadratica"),
                                            ("Spearman", "Spearman con il livello vero"), ("reliable: accuracy", "Affidabile sì/no: accuratezza")]):
    rif = None if misura == "Spearman" else base("levels", misura)
    foresta(ax, righe_ic("classifier", "majority level", misura), titolo, rif)
plt.tight_layout(); plt.savefig(OUT / "cls_altre.png", bbox_inches="tight"); plt.close()

fig, assi = plt.subplots(1, 4, figsize=(16, 4))
for ax, m in zip(assi, MODELLI):
    C = pd.crosstab(TC[m].true, TC[m].predicted).reindex(index=range(5), columns=range(5), fill_value=0).to_numpy()
    P = C / C.sum(1, keepdims=True)
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for a in range(5):
        for b in range(5):
            ax.text(b, a, C[a, b], ha="center", va="center", fontsize=10, color="white" if P[a, b] > 0.5 else "black")
    ax.set_xticks(range(5), LIVELLI); ax.set_yticks(range(5), LIVELLI); ax.grid(False)
    ax.set_xlabel("previsto"); ax.set_ylabel("vero" if m == MODELLI[0] else ""); ax.set_title(m)
plt.tight_layout(); plt.savefig(OUT / "cls_confusione.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(11, 3.8))
d = tutte[(tutte.approach == "classifier") & (tutte.view == "majority level")].set_index("model")
x = np.arange(5)
for j, m in enumerate(MODELLI):
    ax.bar(x + (j - 1.5) * 0.2, [d.loc[m, f"F1 {l}"] for l in LIVELLI], width=0.2, color=COLORE[m], label=m)
ax.set_xticks(x, LIVELLI); ax.set_ylabel("F1"); ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper left")
ax.set_title("Classificatore: F1 di ogni livello"); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "cls_per_livello.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------------------------------------ regressori
fig, assi = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
for ax, m in zip(assi, MODELLI):
    T = TR[m]
    ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
    ax.scatter(T["true score"], T.predicted, s=14, c=[COLORI_LIV[k] for k in T.true])
    rho = spearmanr(T["true score"], T.predicted).correlation
    ax.set_title(f"{m}  (Spearman {virgola(rho)})"); ax.set_xlabel("voto NewsGuard")
assi[0].set_ylabel("voto previsto (media degli articoli)")
plt.tight_layout(); plt.savefig(OUT / "reg_dispersione.png", bbox_inches="tight"); plt.close()

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 3.4))
foresta(a, righe_ic("regressor", "mean score", "MAE"), "Regressore: errore medio in punti (MAE)", base("score", "MAE"), "media del training")
foresta(b, righe_ic("regressor", "mean score", "Spearman"), "Regressore: Spearman con il voto vero")
plt.tight_layout(); plt.savefig(OUT / "reg_mae_spearman.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(11, 3.8))
for j, m in enumerate(MODELLI):
    T = TR[m]
    err = [np.mean(np.abs(T.predicted - T["true score"])[T.true == k]) for k in range(5)]
    ax.bar(np.arange(5) + (j - 1.5) * 0.2, err, width=0.2, color=COLORE[m], label=m)
ax.set_xticks(range(5), [f"{l}\n{r}" for l, r in zip(LIVELLI, ["0-39", "40-59", "60-74", "75-99", "100"])])
ax.set_ylabel("errore medio (punti)"); ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center")
ax.set_title("Regressore: errore per livello vero (gli estremi sono i più difficili)"); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "reg_per_livello.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(11, 3.8))
viste = [("classifier", "majority level", "classificatore (voto degli articoli)"), ("regressor", "levels: NewsGuard bounds", "regressore, confini NewsGuard"),
         ("regressor", "levels: training quotas", "regressore, quote del training"), ("regressor", "levels: article vote", "regressore, voto degli articoli")]
tonalita = ["#1d4f91", "#c3c2b7", "#eb6834", "#f2b48f"]
for j, (ap, v, etichetta) in enumerate(viste):
    d = ic[(ic.approach == ap) & (ic.view == v) & (ic.metric == "F1 macro")].set_index("model").reindex(MODELLI)
    xx = np.arange(4) + (j - 1.5) * 0.2
    ax.bar(xx, d.value, width=0.2, color=tonalita[j], label=etichetta)
    ax.errorbar(xx, d.value, yerr=[d.value - d.low, d.high - d.value], fmt="none", ecolor=GRIGIO, lw=1, capsize=2)
ax.axhline(base("levels", "F1 macro"), color=GRIGIO, ls=":", lw=1)
ax.set_xticks(range(4), MODELLI); ax.set_ylabel("F1 macro"); ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
ax.set_ylim(0, 0.62); ax.set_title("Dal voto ai livelli: quattro modi, F1 macro"); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "reg_livelli.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------------------------------------ confronti
fig, assi = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, (spazio, misura, vista, titolo) in zip(assi, [("levels", "F1 macro", "levels: training quotas", "Livelli, F1 (regressore con le quote)"),
                                                      ("levels", "Spearman", "levels: training quotas", "Livelli, Spearman"),
                                                      ("score", "MAE", "mean score", "Voto, MAE (regressore: media)")]):
    d = coppie[(coppie.space == spazio) & (coppie.metric == misura) & (coppie["regressor view"] == vista)].set_index("model").reindex(MODELLI)
    foresta(ax, [(m, d.difference[m], d.low[m], d.high[m]) for m in MODELLI], titolo, 0, "nessuna differenza")
fig.suptitle("Regressore − classificatore, stesse testate", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "confronto_cls_reg.png", bbox_inches="tight"); plt.close()

fig, assi = plt.subplots(1, 3, figsize=(16, 3.2))
for ax, (ap, v, misura, titolo) in zip(assi, [("classifier", "majority level", "F1 macro", "Classificatore, F1"),
                                              ("regressor", "mean score", "MAE", "Regressore, MAE (punti)"),
                                              ("regressor", "mean score", "Spearman", "Regressore, Spearman")]):
    d = contro[(contro.approach == ap) & (contro.view == v) & (contro.metric == misura)].set_index("model").reindex(MODELLI[1:])
    foresta(ax, [(m, d.difference[m], d.low[m], d.high[m]) for m in MODELLI[1:]], titolo, 0, "come il TF-IDF")
fig.suptitle("Ogni modello − TF-IDF, stesse testate", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "contro_tfidf.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------------------------------------ il DNA: quanti articoli servono
rng = np.random.default_rng(42)
K = [1, 2, 3, 5, 10, 20, 50]
curva = []
for m in ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]:
    pa = PR[m]
    gruppi = {d: g.prediction.to_numpy() for d, g in pa.groupby("domain")}
    for k in K:
        for r in range(20):
            prev = pd.Series({d: rng.choice(v, size=min(k, len(v)), replace=False).mean() for d, v in gruppi.items()})
            vero = testate.score.reindex(prev.index)
            curva.append({"modello": m, "k": k, "Spearman": spearmanr(vero, prev).correlation, "MAE": np.mean(np.abs(prev - vero))})
curva = pd.DataFrame(curva).groupby(["modello", "k"]).mean().reset_index()
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 3.8))
for m in MODELLI:
    d = curva[curva.modello == m]
    a.plot(d.k, d.Spearman, "o-", color=COLORE[m], label=m); b.plot(d.k, d.MAE, "o-", color=COLORE[m], label=m)
for ax, t in [(a, "Spearman con il voto vero"), (b, "Errore medio (punti)")]:
    ax.set_xscale("log"); ax.set_xticks(K, [str(k) for k in K]); ax.set_xlabel("articoli letti per testata"); ax.set_title(t)
a.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "dna.png", bbox_inches="tight"); plt.close()
curva.to_csv(OUT / "dna.csv", index=False)

# ------------------------------------------------------------------------------------------------ Media Bias/Fact Check
FATTUALITA = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
mb = pd.read_csv(REPO / "Dataset/tentativi/10_EtichetteMBFC/mbfc_pulito.csv")
mb["d"] = mb.dominio.str.lower().str.replace(r"^www\.", "", regex=True)
mb = mb[~mb.morto].drop_duplicates("d").set_index("d")
comuni = testate.index[testate.index.str.lower().isin(mb.index)]
fatt = pd.Series(mb.factual.reindex(comuni.str.lower()).map(FATTUALITA).to_numpy(), index=comuni)
ng = testate.score.reindex(comuni)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
nomi_mb = ["Very Low", "Low", "Mixed", "Mostly\nFactual", "High /\nVery High"]
C = pd.crosstab(testate.label.reindex(comuni), fatt).reindex(index=range(5), columns=range(5), fill_value=0)
ax.imshow(C.to_numpy(), cmap="Blues")
for a_ in range(5):
    for b_ in range(5):
        ax.text(b_, a_, C.iloc[a_, b_], ha="center", va="center", fontsize=11, color="white" if C.iloc[a_, b_] > C.to_numpy().max() / 2 else "black")
ax.set_xticks(range(5), nomi_mb, fontsize=9); ax.set_yticks(range(5), LIVELLI); ax.grid(False)
ax.set_xlabel("MBFC: Factual Reporting"); ax.set_ylabel("NewsGuard: livello")
ax.set_title(f"{len(comuni)} testate in comune: Spearman {virgola(spearmanr(ng, fatt).correlation)}")
plt.tight_layout(); plt.savefig(OUT / "mbfc_incrocio.png", bbox_inches="tight"); plt.close()

righe, B = [], 2000
rng = np.random.default_rng(42)
idx = np.arange(len(comuni))
coppie_mb = {"NewsGuard ↔ MBFC (esperti)": (ng.to_numpy(), fatt.to_numpy())}
for m in MODELLI:
    p = TR[m].predicted.reindex(comuni).to_numpy()
    coppie_mb[f"{m} ↔ NewsGuard"] = (ng.to_numpy(), p)
    coppie_mb[f"{m} ↔ MBFC"] = (fatt.to_numpy(), p)
for nome, (u, v) in coppie_mb.items():
    boot = [spearmanr(u[i], v[i]).correlation for i in (rng.integers(0, len(idx), len(idx)) for _ in range(B))]
    righe.append({"coppia": nome, "Spearman": spearmanr(u, v).correlation, "da": np.percentile(boot, 2.5), "a": np.percentile(boot, 97.5)})
acc = pd.DataFrame(righe); acc.to_csv(OUT / "mbfc_accordo.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 4.6))
y = np.arange(len(acc))
esperti = acc.Spearman.iloc[0]
ax.axvspan(acc.da.iloc[0], acc.a.iloc[0], color="#f0efe9", zorder=0)
ax.axvline(esperti, color=GRIGIO, ls=":", lw=1.2)
for i, r in acc.iterrows():
    nome = r.coppia.split(" ↔ ")[0]
    ax.plot([r.da, r.a], [i, i], color="#b5b3ab", lw=2)
    ax.scatter(r.Spearman, i, s=80, color=COLORE.get(nome, "#111111"), zorder=3, marker="D" if "esperti" in r.coppia else "o")
    ax.text(r.a, i, "  " + virgola(r.Spearman), va="center", fontsize=10)
ax.set_yticks(y, acc.coppia); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_title(f"Spearman sulle {len(comuni)} testate in comune (regressore: media); in grigio l'accordo tra esperti")
plt.tight_layout(); plt.savefig(OUT / "mbfc_accordo.png", bbox_inches="tight"); plt.close()
print(acc.round(3).to_string(index=False))
print(curva[curva.k.isin([1, 5, 50])].round(3).to_string(index=False))

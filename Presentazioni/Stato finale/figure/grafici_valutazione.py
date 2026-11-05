"""Grafici della valutazione (tentativo 20) per la presentazione «Stato finale»: SOLO i regressori, valutati con la
distanza dal voto NewsGuard vero; le classi compaiono solo nel confronto con Media Bias/Fact Check (mbfc.py del 20).
Legge le previsioni per articolo in Dataset/tentativi/20_FromScratch.
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
import mbfc as MB      # noqa: E402
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"; OUT.mkdir(exist_ok=True)
MODELLI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834",
          "senza testo": "#c3c2b7", "NewsGuard": "#111111"}
FASCE = ["0-39", "40-59", "60-74", "75-99", "100"]
COLORI_FASCE = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
COLORI_MBFC = ["#b02a2a", "#e0a43a", "#5b9be0", "#1d4f91"]
GRIGIO = "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
virgola = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
B, rng_seed = 2000, 42

art, testate = MO.load()
ic = pd.read_csv(D20 / "intervals.csv")
tutte = pd.read_csv(D20 / "all_metrics.csv")
contro = pd.read_csv(D20 / "models_vs_tfidf.csv")
PR = {m: MO.predictions("regression", m) for m in MODELLI}
TR = {m: ME.outlets_regressor(PR[m], testate) for m in MODELLI}
VERO = testate.score
# senza testo: a ogni testata la media dei voti delle testate di training del suo fold
SENZA = pd.Series({d: testate.score[testate.fold != f].mean() for d, f in testate.fold.items()})
STIRATO = {m: pd.Series(ME.remapped_score(TR[m].predicted, TR[m].fold, TR[m]["true score"]), index=TR[m].index) for m in MODELLI}
fascia = np.digitize(VERO, [39.5, 59.5, 74.5, 99.5])


def foresta(ax, righe, titolo, rif=None, rif_testo="senza testo"):
    """Punti con intervallo al 95%, uno per modello (righe: lista di (nome, valore, da, a))."""
    for i, (n, v, a, b) in enumerate(righe):
        ax.plot([a, b], [i, i], color="#b5b3ab", lw=2)
        ax.scatter(v, i, s=80, color=COLORE.get(n, GRIGIO), zorder=3)
        ax.text(b, i, "  " + virgola(v, 1 if abs(v) > 2 else 2), va="center", fontsize=10)
    if rif is not None:
        ax.axvline(rif, color=GRIGIO, ls=":", lw=1.2)
        ax.text(rif, len(righe) - 0.45, f" {rif_testo}: {virgola(rif, 1 if abs(rif) > 2 else 2)}", fontsize=9, color=GRIGIO)
    ax.set_yticks(range(len(righe)), [r[0] for r in righe]); ax.invert_yaxis(); ax.set_title(titolo); ax.grid(axis="y", visible=False)


def righe_ic(vista, misura):
    d = ic[(ic.approach == "regressor") & (ic.view == vista) & (ic.metric == misura)].set_index("model")
    return [(m, d.value[m], d.low[m], d.high[m]) for m in MODELLI]


def base(misura):
    return tutte[(tutte.approach == "baseline") & (tutte.space == "score")][misura].iloc[0]


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


# ------------------------------------------------------------------------------------------------ come si misura la distanza
T = TR["Mistral-24B"].sort_values("true score")
scelte = T.iloc[[int(q * (len(T) - 1)) for q in (0.04, 0.22, 0.42, 0.62, 0.8, 0.97)]]
fig, ax = plt.subplots(figsize=(11, 3.6))
for i, (_, r) in enumerate(scelte.iterrows()):
    ax.plot([r["true score"], r.predicted], [i, i], color="#b5b3ab", lw=3, zorder=1)
    ax.scatter(r["true score"], i, s=90, color="#111111", zorder=3, label="voto NewsGuard" if i == 0 else None)
    ax.scatter(r.predicted, i, s=90, color=COLORE["Mistral-24B"], zorder=3, label="voto previsto (Mistral)" if i == 0 else None)
    ax.text(max(r["true score"], r.predicted) + 1.5, i, f"distanza {abs(r.predicted - r['true score']):.0f} punti", va="center", fontsize=10)
ax.set_yticks(range(len(scelte)), [f"testata {c}" for c in "ABCDEF"]); ax.invert_yaxis(); ax.set_xlim(0, 115)
ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("voto (0-100)"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0, -0.42))
ax.set_title(f"Sei testate vere: l'errore medio (MAE) è la media di queste distanze, su tutte le {len(T)} testate")
salva("esempio_distanza.png")

# ------------------------------------------------------------------------------------------------ il bootstrap
fig, ax = plt.subplots(figsize=(11, 3.6))
for m, colore in [("TF-IDF", COLORE["TF-IDF"]), ("Mistral-24B", COLORE["Mistral-24B"])]:
    err = np.abs(TR[m].predicted - TR[m]["true score"]).to_numpy()
    rng = np.random.default_rng(rng_seed)
    boot = np.array([err[rng.integers(0, len(err), len(err))].mean() for _ in range(B)])
    a, b = np.percentile(boot, [2.5, 97.5])
    ax.hist(boot, bins=60, color=colore, alpha=0.55, label=m)
    for v in (a, b):
        ax.axvline(v, color=colore, ls="--", lw=1.2)
    ax.text(err.mean(), ax.get_ylim()[1] * 0.92, f"{m}\n{virgola(err.mean(), 1)} [{virgola(a, 1)}; {virgola(b, 1)}]", ha="center",
            fontsize=10, color=colore, fontweight="bold")
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.set_xlabel("errore medio (punti) ricalcolato su 2.000 campioni di testate"); ax.set_ylabel("quante volte")
ax.set_title("Bootstrap: le stesse testate ripescate a caso 2.000 volte; tra le linee tratteggiate il 95% dei valori")
salva("bootstrap.png")

# ------------------------------------------------------------------------------------------------ i regressori
fig, assi = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
for ax, m in zip(assi, MODELLI):
    T = TR[m]
    ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
    ax.scatter(T["true score"], T.predicted, s=14, c=[COLORI_FASCE[k] for k in fascia])
    ax.set_title(f"{m}  (Spearman {virgola(spearmanr(T['true score'], T.predicted).correlation)})"); ax.set_xlabel("voto NewsGuard")
assi[0].set_ylabel("voto previsto (media degli articoli)")
salva("reg_dispersione.png")

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 3.4))
foresta(a, righe_ic("mean score", "MAE"), "Errore medio in punti (MAE): più basso è meglio", base("MAE"), "senza testo")
foresta(b, righe_ic("mean score", "Spearman"), "Spearman con il voto vero: più alto è meglio (senza testo: 0)")
salva("reg_mae_spearman.png")

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(0, 61)
for m in MODELLI + ["senza testo"]:
    p = SENZA if m == "senza testo" else TR[m].predicted
    err = np.abs(p.reindex(VERO.index) - VERO).to_numpy()
    ax.plot(x, [100 * np.mean(err <= v) for v in x], color=COLORE[m], lw=2.2 if m == "Mistral-24B" else 1.6,
            ls=":" if m == "senza testo" else "-", label=m)
for v in (10, 20):
    ax.axvline(v, color=GRIGIO, lw=0.8)
ax.set_xlabel("distanza dal voto vero (punti)"); ax.set_ylabel("testate entro quella distanza (%)")
ax.set_ylim(0, 101); ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Quante testate sono vicine al voto vero")
salva("reg_entro.png")
entro = pd.DataFrame({m: {v: np.mean(np.abs((SENZA if m == "senza testo" else TR[m].predicted).reindex(VERO.index) - VERO) <= v)
                          for v in (10, 20)} for m in MODELLI + ["senza testo"]})

fig, (a, b) = plt.subplots(1, 2, figsize=(15, 3.8))
for j, m in enumerate(MODELLI):
    d = (TR[m].predicted.reindex(VERO.index) - VERO).to_numpy()
    xx = np.arange(5) + (j - 1.5) * 0.2
    a.bar(xx, [np.mean(np.abs(d[fascia == k])) for k in range(5)], width=0.2, color=COLORE[m], label=m)
    b.bar(xx, [np.mean(d[fascia == k]) for k in range(5)], width=0.2, color=COLORE[m], label=m)
n_fascia = np.bincount(fascia, minlength=5)
for ax, t in [(a, "Errore medio per fascia di voto vero (punti)"), (b, "Previsto − vero: sopra 0 sovrastima, sotto sottostima")]:
    ax.set_xticks(range(5), [f"{f}\n({n} testate)" for f, n in zip(FASCE, n_fascia)]); ax.set_title(t); ax.grid(axis="x", visible=False)
b.axhline(0, color=GRIGIO, lw=0.8); a.legend(ncol=2, frameon=False, fontsize=9, loc="upper center")
salva("reg_per_fascia.png")

fig, assi = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
T = TR["Mistral-24B"]
for ax, (p, t) in zip(assi[:2], [(T.predicted, "Mistral: voto previsto"), (STIRATO["Mistral-24B"], "Mistral: voto stirato")]):
    ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
    ax.scatter(T["true score"], p, s=14, c=[COLORI_FASCE[k] for k in fascia])
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_xlabel("voto NewsGuard")
    ax.set_title(f"{t}  (MAE {virgola(np.mean(np.abs(p - T['true score'])), 1)})")
ax = assi[2]
for j, (vista, etichetta, colore) in enumerate([("mean score", "previsto", "#b5b3ab"), ("remapped score", "stirato", "#1d4f91")]):
    d = ic[(ic.approach == "regressor") & (ic.view == vista) & (ic.metric == "MAE")].set_index("model").reindex(MODELLI)
    xx = np.arange(4) + (j - 0.5) * 0.36
    ax.bar(xx, d.value, width=0.36, color=colore, label=etichetta)
    ax.errorbar(xx, d.value, yerr=[d.value - d.low, d.high - d.value], fmt="none", ecolor=GRIGIO, lw=1, capsize=2)
    for xv, v in zip(xx, d.value):
        ax.text(xv, 1, virgola(v, 1), ha="center", fontsize=8, color="white")
ax.set_xticks(range(4), MODELLI, fontsize=9); ax.set_ylabel("errore medio (punti)"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Errore medio, previsto e stirato"); ax.grid(axis="x", visible=False)
salva("reg_stirati.png")

fig, assi = plt.subplots(1, 3, figsize=(16, 3.0))
for ax, (misura, titolo) in zip(assi, [("MAE", "Errore medio (punti): sotto 0 = meglio"), ("Spearman", "Spearman: sopra 0 = meglio"),
                                       ("within 20 points", "Testate entro 20 punti (punti %): sopra 0 = meglio")]):
    d = contro[(contro.approach == "regressor") & (contro.view == "mean score") & (contro.metric == misura)].set_index("model").reindex(MODELLI[1:])
    f = 100 if misura == "within 20 points" else 1
    foresta(ax, [(m, d.difference[m] * f, d.low[m] * f, d.high[m] * f) for m in MODELLI[1:]], titolo, 0, "come il TF-IDF")
fig.suptitle("Ogni modello − TF-IDF, sulle stesse testate", x=0.01, ha="left", fontweight="bold")
salva("contro_tfidf.png")

# ------------------------------------------------------------------------------------------------ il DNA: quanti articoli servono
rng = np.random.default_rng(42)
K = [1, 2, 3, 5, 10, 20, 50]
curva = []
for m in MODELLI:
    gruppi = {d: g.prediction.to_numpy() for d, g in PR[m].groupby("domain")}
    for k in K:
        for r in range(20):
            prev = pd.Series({d: rng.choice(v, size=min(k, len(v)), replace=False).mean() for d, v in gruppi.items()})
            vero = VERO.reindex(prev.index)
            curva.append({"modello": m, "k": k, "Spearman": spearmanr(vero, prev).correlation, "MAE": np.mean(np.abs(prev - vero))})
curva = pd.DataFrame(curva).groupby(["modello", "k"]).mean().reset_index()
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 3.8))
for m in MODELLI:
    d = curva[curva.modello == m]
    a.plot(d.k, d.Spearman, "o-", color=COLORE[m], label=m); b.plot(d.k, d.MAE, "o-", color=COLORE[m], label=m)
for ax, t in [(a, "Spearman con il voto vero"), (b, "Errore medio (punti)")]:
    ax.set_xscale("log"); ax.set_xticks(K, [str(k) for k in K]); ax.set_xlabel("articoli letti per testata"); ax.set_title(t)
a.legend(frameon=False, fontsize=9)
salva("dna.png")
curva.to_csv(OUT / "dna.csv", index=False)

# ------------------------------------------------------------------------------------------------ Media Bias/Fact Check
y = MB.load(testate)                                     # la classe MBFC (0-3) delle testate in comune
comuni, fold = y.index, testate.fold.reindex(y.index)
FONTI = {"NewsGuard": VERO.reindex(comuni)} | {m: TR[m].predicted.reindex(comuni) for m in MODELLI}
NOMI_MB = ["Low", "Mixed", "Mostly\nFactual", "High"]
CONVERSIONI = {"quotas": "quote", "medians": "mediane", "thresholds": "soglie"}

fig, ax = plt.subplots(figsize=(9, 4.2))
r = np.random.default_rng(0)
for k in range(4):
    v = VERO.reindex(comuni)[y == k]
    ax.scatter(k + r.uniform(-0.18, 0.18, len(v)), v, s=18, color=COLORI_MBFC[k], alpha=0.8)
    ax.plot([k - 0.3, k + 0.3], [v.median()] * 2, color="#111111", lw=2)
    ax.text(k + 0.32, v.median(), f" mediana {v.median():.0f}\n {len(v)} testate", va="center", fontsize=9)
ax.set_xticks(range(4), NOMI_MB); ax.set_xlim(-0.5, 4); ax.set_xlabel("MBFC: Factual Reporting"); ax.set_ylabel("voto NewsGuard")
ax.set_title(f"{len(comuni)} testate in comune: Spearman {virgola(spearmanr(VERO.reindex(comuni), y).correlation)}")
ax.grid(axis="x", visible=False)
salva("mbfc_esperti.png")

righe = []
rng = np.random.default_rng(42)
idx = [rng.integers(0, len(comuni), len(comuni)) for _ in range(B)]
coppie = {"NewsGuard ↔ MBFC (esperti)": (FONTI["NewsGuard"].to_numpy(), y.to_numpy())}
for m in MODELLI:
    coppie[f"{m} ↔ NewsGuard"] = (FONTI["NewsGuard"].to_numpy(), FONTI[m].to_numpy())
    coppie[f"{m} ↔ MBFC"] = (y.to_numpy(), FONTI[m].to_numpy())
for nome, (u, v) in coppie.items():
    boot = [spearmanr(u[i], v[i]).correlation for i in idx]
    righe.append({"coppia": nome, "Spearman": spearmanr(u, v).correlation, "da": np.percentile(boot, 2.5), "a": np.percentile(boot, 97.5)})
acc = pd.DataFrame(righe); acc.to_csv(OUT / "mbfc_accordo.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axvspan(acc.da.iloc[0], acc.a.iloc[0], color="#f0efe9", zorder=0)
ax.axvline(acc.Spearman.iloc[0], color=GRIGIO, ls=":", lw=1.2)
for i, r_ in acc.iterrows():
    nome = r_.coppia.split(" ↔ ")[0]
    ax.plot([r_.da, r_.a], [i, i], color="#b5b3ab", lw=2)
    ax.scatter(r_.Spearman, i, s=80, color=COLORE.get(nome, "#111111"), zorder=3, marker="D" if "esperti" in r_.coppia else "o")
    ax.text(r_.a, i, "  " + virgola(r_.Spearman), va="center", fontsize=10)
ax.set_yticks(range(len(acc)), acc.coppia); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_title(f"Spearman sulle {len(comuni)} testate in comune (il voto previsto, senza classi); in grigio gli esperti tra loro")
salva("mbfc_accordo.png")

# i tre modi di passare dal voto alla classe MBFC, mostrati sul voto NewsGuard (imparati su tutte le testate in comune)
fig, assi = plt.subplots(4, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1, 1]})
s, c = VERO.reindex(comuni).to_numpy(), y.to_numpy()
ax = assi[0]
for k in range(4):
    ax.scatter(s[c == k], np.random.default_rng(k).uniform(-0.35, 0.35, (c == k).sum()), s=16, color=COLORI_MBFC[k], label=MB.CLASSES[k])
ax.set_yticks([]); ax.set_title("Le testate sul voto NewsGuard, colorate con la classe MBFC"); ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.02))
ax.set_ylim(-0.5, 1.4); ax.grid(False)
griglia = np.linspace(0, 100, 1001)
for ax, (conv, nome) in zip(assi[1:], [("quotas", "Quote: stesse proporzioni delle classi MBFC"),
                                        ("medians", "Mediane: la classe con il voto tipico più vicino"),
                                        ("thresholds", "Soglie: i tagli che sbagliano meno classi")]):
    if conv == "quotas":
        classe = np.searchsorted(np.quantile(s, np.cumsum(np.bincount(c, minlength=4))[:-1] / len(c)), griglia)
    else:
        classe = MB.CONVERSIONS[conv](s, c, griglia)
    for k in range(4):
        z = griglia[classe == k]
        if len(z):
            ax.axvspan(z.min(), z.max(), color=COLORI_MBFC[k], alpha=0.75)
            ax.text((z.min() + z.max()) / 2, 0.5, MB.CLASSES[k], ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_yticks([]); ax.set_title(nome, fontsize=11); ax.grid(False); ax.set_ylim(0, 1)
assi[-1].set_xlim(0, 100); assi[-1].set_xlabel("voto (0-100)")
salva("mbfc_metodi.png")

ris = []
for fonte, punteggio in FONTI.items():
    for conv in CONVERSIONI:
        p = MB.by_fold(conv, punteggio, y, fold)
        val = MB.agreement(y, p)
        boot = {k: [] for k in val}
        for i in idx:
            for k, v in MB.agreement(y.to_numpy()[i], p[i]).items():
                boot[k].append(v)
        for k in val:
            ris.append({"fonte": fonte, "conversione": conv, "misura": k, "valore": val[k],
                        "da": np.percentile(boot[k], 2.5), "a": np.percentile(boot[k], 97.5)})
ris = pd.DataFrame(ris); ris.to_csv(OUT / "mbfc_classi.csv", index=False)
sempre = {"exact": np.mean(y == np.bincount(y).argmax()), "quadratic kappa": 0.0}
fig, assi = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True)
fonti = list(FONTI)
for ax, (misura, titolo) in zip(assi, [("exact", "Classe MBFC giusta (%)"), ("quadratic kappa", "Kappa quadratica (0 = caso, 1 = perfetto)")]):
    for j, (conv, nome) in enumerate(CONVERSIONI.items()):
        d = ris[(ris.conversione == conv) & (ris.misura == misura)].set_index("fonte").reindex(fonti)
        yy = np.arange(len(fonti)) + (j - 1) * 0.24
        f = 100 if misura == "exact" else 1
        ax.hlines(yy, d.da * f, d.a * f, color="#c3c2b7", lw=2)
        ax.scatter(d.valore * f, yy, s=55, color=["#1d4f91", "#eb6834", "#1baf7a"][j], zorder=3, label=nome)
    ax.axvline(sempre[misura] * (100 if misura == "exact" else 1), color=GRIGIO, ls=":", lw=1.2)
    ax.text(sempre[misura] * (100 if misura == "exact" else 1), len(FONTI) - 0.45, " sempre «Mixed»", fontsize=9, color=GRIGIO)
    ax.axhspan(-0.45, 0.45, color="#f0efe9", zorder=0)
    ax.set_title(titolo); ax.grid(axis="y", visible=False)
assi[0].set_yticks(range(len(fonti)), ["NewsGuard (esperto)"] + MODELLI); assi[0].invert_yaxis()
assi[0].legend(frameon=False, fontsize=10, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08), title="dal voto alla classe:", title_fontsize=10)
salva("mbfc_classi.png")

fig, assi = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, fonte in zip(assi, ["NewsGuard", "Mistral-24B"]):
    p = MB.by_fold("quotas", FONTI[fonte], y, fold)
    C = pd.crosstab(y.to_numpy(), p).reindex(index=range(4), columns=range(4), fill_value=0).to_numpy()
    ax.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for a_ in range(4):
        for b_ in range(4):
            ax.text(b_, a_, C[a_, b_], ha="center", va="center", fontsize=11,
                    color="white" if C[a_, b_] / C[a_].sum() > 0.5 else "black")
    ax.set_xticks(range(4), NOMI_MB, fontsize=9); ax.set_yticks(range(4), NOMI_MB, fontsize=9); ax.grid(False)
    ax.set_ylabel("classe MBFC vera"); ax.set_xlabel(f"classe dal voto {'NewsGuard' if fonte == 'NewsGuard' else 'previsto da Mistral'}")
    ax.set_title(f"{'NewsGuard (esperto)' if fonte == 'NewsGuard' else 'Mistral-24B'}: {np.trace(C)} su {C.sum()} giuste")
salva("mbfc_incrocio.png")

pd.set_option("display.width", 200)
print(entro.round(3).to_string())
print(acc.round(3).to_string(index=False))
print(ris.pivot_table(index=["fonte"], columns=["misura", "conversione"], values="valore").round(3).to_string())
print(curva[curva.k.isin([1, 5, 50])].round(3).to_string(index=False))

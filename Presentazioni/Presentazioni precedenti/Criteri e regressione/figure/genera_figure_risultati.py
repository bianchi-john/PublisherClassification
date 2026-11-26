"""Figure dei risultati per la presentazione "Criteri e regressione" (riunione con Fabio): misure classiche e di accordo
per articolo e per testata, contro NewsGuard (226 e 170 testate) e contro MBFC, e la curva del DNA. Usa le previsioni
per articolo del tentativo 16 e le funzioni del tentativo 17. Gira dove ci sono le previsioni (sul server):
    python genera_figure_risultati.py <cartella del tentativo 16> <cartella del tentativo 17> <cartella di uscita>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T16, T17, OUT = (Path(a).expanduser() for a in sys.argv[1:4])
sys.path[:0] = [str(T16), str(T17)]
import accordo as A                     # noqa: E402
import funzioni as F                    # noqa: E402

BLU, ARANCIO, GRIGIO, CHIARO = "#2a78d6", "#eb6834", "#52514e", "#b5b3ab"
plt.rcParams.update(F.STILE_GRAFICI); plt.rcParams.update({"figure.dpi": 150})
t = pd.read_csv(F.DATASET / "testate.csv", index_col="domain")
mb = A.leggi_mbfc(F.DATI / "tentativi/10_EtichetteMBFC/mbfc_pulito.csv", t)
R16 = T16 / "Results"
mbert = [pd.read_csv(R16 / "modernbert" / c / f"fold{k}.csv") for c in ["5 classi", "regressione"] for k in range(1, 6)]
MODELLI = {"TF-IDF": pd.read_csv(R16 / "feature" / "previsioni_articoli.csv"),
           "ModernBERT": pd.concat(mbert[:5]).merge(pd.concat(mbert[5:])[["article_id", "s_regressione"]], on="article_id"),
           "Qwen3 8B": pd.read_csv(R16 / "analisi_embedding" / "articoli_previsioni.csv"),
           "Mistral 24B": pd.read_csv(R16 / "analisi_embedding" / "mistral27_articoli_previsioni.csv")}
LIVELLI = {m: A.livelli_articoli(pa, t).set_index("domain") for m, pa in MODELLI.items()}
INSIEMI = {"226 NewsGuard": (t.index, t.label.to_numpy(), A.SOGLIA["NewsGuard"]),
           "170 NewsGuard": (mb.index, t.loc[mb.index].label.to_numpy(), A.SOGLIA["NewsGuard"]),
           "170 MBFC": (mb.index, mb.fattualita.to_numpy(), A.SOGLIA["MBFC"])}
rng = np.random.default_rng(0)
righe = []
for ins, (testate, et, soglia) in INSIEMI.items():
    idx = [rng.integers(0, len(testate), len(testate)) for _ in range(500)]
    for m, lv in LIVELLI.items():
        for imp in ["classificatore", "regressore"]:
            g = A.gruppi(lv[imp], testate)
            vl, va = A.giudizio(g)
            art = lambda ii: (np.concatenate([g[i] for i in ii]), np.concatenate([np.full(len(g[i]), et[i]) for i in ii]))
            for unita, f in [("articolo", lambda ii: A.quattro_misure(*art(ii), soglia)),
                             ("testata", lambda ii: A.quattro_misure(vl[ii], et[ii], soglia, va[ii]))]:
                v = f(np.arange(len(testate))); b = pd.DataFrame([f(ii) for ii in idx])
                for mis in v:
                    righe.append({"insieme": ins, "modello": m, "impostazione": imp, "unità": unita, "misura": mis, "valore": v[mis],
                                  "lo": np.percentile(b[mis], 2.5), "hi": np.percentile(b[mis], 97.5)})
    print("fatto", ins, flush=True)
R = pd.DataFrame(righe); R.to_csv(OUT / "risultati.csv", index=False)
COPPIE = [(m, imp) for m in MODELLI for imp in ["classificatore", "regressore"]]


def grafico(ins, unita, misure, file, titolo, esperti=False):
    """Una riga di pannelli (una misura ciascuno); in ogni pannello una barra per modello, classificatore e regressore."""
    fig, assi = plt.subplots(1, len(misure), figsize=(6.2 * len(misure), 3.9))
    testate, et, soglia = INSIEMI[ins]
    x = np.arange(len(MODELLI))
    for ax, mis in zip(np.atleast_1d(assi), misure):
        d = R[(R.insieme == ins) & (R["unità"] == unita) & (R.misura == mis)]
        for dx, imp, colore in [(-0.19, "classificatore", BLU), (0.19, "regressore", ARANCIO)]:
            for i, m in enumerate(MODELLI):
                q = d[(d.modello == m) & (d.impostazione == imp)].iloc[0]
                ax.bar(i + dx, q.valore, width=0.36, color=colore, label=imp if (mis == misure[0] and i == 0) else None)
                ax.errorbar(i + dx, q.valore, yerr=[[q.valore - q.lo], [q.hi - q.valore]], color=GRIGIO, lw=1, capsize=2)
                ax.text(i + dx, q.hi + 0.015, f"{q.valore:.2f}".replace(".", ","), ha="center", fontsize=8, color=GRIGIO, fontweight="bold")
        rif = {"accuratezza": (np.mean(et >= soglia), "sempre «affidabile»"), "F1": (0.2, "a caso")}.get(mis)
        if rif:
            ax.axhline(rif[0], color=GRIGIO, ls=":", lw=1)
            ax.text(len(MODELLI) - 0.5, rif[0] + 0.01, f"{rif[1]}: {rif[0]:.2f}".replace(".", ","), ha="right", fontsize=8, color=GRIGIO)
        if esperti and mis in ("kappa", "Spearman"):
            e = A.misure(t.loc[mb.index].label.to_numpy(), mb.fattualita.to_numpy(), (t.loc[mb.index].label >= 2).astype(int).to_numpy(),
                         (mb.fattualita >= 3).astype(int).to_numpy())[mis]
            ax.axhline(e, color=GRIGIO, ls="--", lw=1.3)
            ax.text(-0.45, e + 0.015, f"i due esperti tra loro: {e:.2f}".replace(".", ","), fontsize=8, color=GRIGIO)
        nomi = {"accuratezza": "accuratezza su affidabile sì/no", "F1": "F1 sui 5 livelli", "kappa": "kappa su affidabile sì/no",
                "Spearman": "Spearman sui 5 livelli"}
        ax.set_title(nomi[mis], fontsize=11); ax.set_xticks(x, list(MODELLI), fontsize=10)
        ax.set_ylim(0, 1.0); ax.grid(axis="x", visible=False)
    fig.legend(loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(titolo, fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    plt.tight_layout(); plt.savefig(OUT / file, bbox_inches="tight"); plt.close()


for ins, nome in [("226 NewsGuard", "tutte le 226 testate"), ("170 NewsGuard", "le 170 testate valutate anche da MBFC")]:
    tag = ins.split()[0]
    grafico(ins, "articolo", ["accuratezza", "F1"], f"classiche_{tag}_articolo.png", f"Un articolo da solo · {nome} · contro NewsGuard")
    grafico(ins, "testata", ["accuratezza", "F1"], f"classiche_{tag}_testata.png", f"La testata (voto dei suoi articoli) · {nome} · contro NewsGuard")
grafico("226 NewsGuard", "testata", ["kappa", "Spearman"], "accordo_226_testata.png",
        "Le misure di accordo · la testata (voto) · tutte le 226 testate · contro NewsGuard", esperti=True)
for rif in ["NewsGuard", "MBFC"]:
    grafico(f"170 {rif}", "testata", ["kappa", "Spearman"], f"accordo_170_{rif}.png",
            f"Il modello come terzo annotatore · le 170 testate · contro {rif}", esperti=True)

# la curva del DNA: Spearman con NewsGuard e con MBFC, i classificatori, al crescere degli articoli letti
c = pd.read_csv(T17 / "Results" / "curva_articoli.csv")
c = c[c.impostazione == "classificatore"]
COLORI = {"TF-IDF": GRIGIO, "ModernBERT": "#9b59b6", "Qwen3-Embedding 8B": "#1baf7a", "Mistral-Small 24B": "#b02a2a"}
fig, assi = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True)
e = A.misure(t.loc[mb.index].label.to_numpy(), mb.fattualita.to_numpy(), (t.loc[mb.index].label >= 2).astype(int).to_numpy(),
             (mb.fattualita >= 3).astype(int).to_numpy())["Spearman"]
for ax, rif in zip(assi, ["↔ NewsGuard", "↔ MBFC"]):
    ax.axhline(e, color=GRIGIO, ls="--", lw=1.3); ax.text(1, e + 0.01, f"i due esperti tra loro: {e:.2f}".replace(".", ","), fontsize=8, color=GRIGIO)
    for m, d in c[c.riferimento == rif].groupby("modello"):
        ax.plot(d.k, d.Spearman, marker="o", ms=4, color=COLORI[m], label=m if rif == "↔ NewsGuard" else None)
    ax.set_xscale("log"); ax.set_xticks([1, 3, 5, 9, 15, 25, 999], ["1", "3", "5", "9", "15", "25", "tutti"])
    ax.set_xlabel("articoli letti per testata (voto di maggioranza)"); ax.set_title(f"Spearman: modello {rif}", fontsize=11)
fig.legend(loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Il DNA negli articoli: quanti articoli servono per giudicare la testata come un esperto", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "dna.png", bbox_inches="tight"); plt.close()
print("figure scritte in", OUT)

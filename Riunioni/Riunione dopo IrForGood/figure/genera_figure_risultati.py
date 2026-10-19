"""Le figure delle slide sui risultati (riunione dopo il weekend del 17-18 ottobre 2026).

    cd "Riunioni/Riunione dopo IrForGood/figure"
    ~/.cache/irgood-venv/bin/python genera_figure_risultati.py

Legge i risultati salvati in Code/WebScience2027 (primo dataset: EsperimentiPreliminari/1-PrimoDataset;
dataset nuovo: Results/ e Dataset/ scaricati dal server) e scrive i PDF qui accanto.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
W = QUI.parents[2] / "Code" / "WebScience2027"
P1 = W / "EsperimentiPreliminari" / "1-PrimoDataset"
R1, R2 = W / "Results" / "1-data-cleaning", W / "Results" / "2-classificazione"

# colori: livelli dall'arancio (inaffidabile) al blu (affidabile) con il grigio al centro; modelli e dataset per identità
LIV = ["l1", "l2", "l3", "l4", "l5"]
COL_LIV = {"l1": "#c4501f", "l2": "#f2a07c", "l3": "#b8b7ae", "l4": "#86b6ef", "l5": "#1c5cab"}
BLU, ARANCIO, ACQUA, GRIGIO, INCHIOSTRO, MUTO = "#2a78d6", "#eb6834", "#1baf7a", "#9a998f", "#2b2b28", "#6b6a63"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#b9b8b0", "axes.labelcolor": INCHIOSTRO, "xtick.color": MUTO,
                     "ytick.color": MUTO, "text.color": INCHIOSTRO, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


def salva(fig, nome):
    fig.savefig(QUI / f"{nome}.pdf")
    plt.close(fig)
    print("scritta", nome)


def etichetta_barre(ax, barre, fmt="{:.2f}", dentro=False):
    for b in barre:
        v = b.get_height()
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, color=INCHIOSTRO)


# =============================================================================================== primo dataset
val = pd.read_csv(P1 / "Results/1-data-cleaning/esplorazioni/04_testate_gruppi_valutazione.csv")
gruppi1 = pd.read_csv(P1 / "Results/1-data-cleaning/01_gruppi_testate.csv").set_index("domain")
val["dim"] = val.domain.map(gruppi1.testate_nel_gruppo)
val["liv"] = val.y_true.map(dict(enumerate(LIV)))

# 1. sembrava funzionare: fold per testata contro fold per gruppo
fig, ax = plt.subplots(figsize=(5.6, 3.4))
acc = [val.giusta_01.mean(), val.giusta_gruppo.mean()]
b = ax.bar(["fold per testata\n(come nel paper)", "fold per gruppo\n(reti mai viste)"], acc, color=[ARANCIO, BLU], width=0.5)
etichetta_barre(ax, b)
ax.axhline(val.y_true.value_counts(normalize=True).max(), color=GRIGIO, ls="--", lw=1.2)
ax.text(0.5, val.y_true.value_counts(normalize=True).max() + 0.012, "classe di maggioranza", color=MUTO, fontsize=9, ha="center")
ax.set_ylim(0, 0.8); ax.set_ylabel("accuratezza per testata (5 classi)")
ax.set_title("Primo dataset, TF-IDF: dipende tutto da come si divide", fontsize=11)
salva(fig, "fig_primo_due_fold")

# 2. dentro o fuori da una rete
fig, ax = plt.subplots(figsize=(6.2, 3.5))
x = np.arange(2); w = 0.34
dentro = [val[val.in_gruppo].giusta_01.mean(), val[val.in_gruppo].giusta_gruppo.mean()]
fuori = [val[~val.in_gruppo].giusta_01.mean(), val[~val.in_gruppo].giusta_gruppo.mean()]
b1 = ax.bar(x - w / 2 - 0.01, dentro, w, color=ARANCIO, label=f"testate in una rete ({int(val.in_gruppo.sum())})")
b2 = ax.bar(x + w / 2 + 0.01, fuori, w, color=BLU, label=f"testate isolate ({int((~val.in_gruppo).sum())})")
etichetta_barre(ax, b1); etichetta_barre(ax, b2)
ax.set_xticks(x, ["fold per testata", "fold per gruppo"]); ax.set_ylim(0, 1.08); ax.set_ylabel("accuratezza per testata")
ax.legend(loc="upper right", fontsize=9.5)
ax.set_title("Primo dataset: le testate in una rete si indovinano perché c'è la sorella nel training", fontsize=10)
salva(fig, "fig_primo_dentro_fuori")

# 3. quanta parte di ogni livello è una rete
fig, ax = plt.subplots(figsize=(6.4, 3.4))
tot = val.groupby("liv").size().reindex(LIV)
rete0 = val[val.gruppo == 0].groupby("liv").size().reindex(LIV, fill_value=0)
altre = val[val.in_gruppo & (val.gruppo != 0)].groupby("liv").size().reindex(LIV, fill_value=0)
isol = tot - rete0 - altre
for i, (serie, col, nome) in enumerate([(rete0, ARANCIO, "la rete più grande (131 siti)"), (altre, "#f2a07c", "altre reti"), (isol, BLU, "testate isolate")]):
    base = sum([rete0, altre, isol][:i]) if i else 0
    ax.bar(LIV, serie, bottom=base, color=col, label=nome, width=0.6, edgecolor="white", linewidth=1.5)
for j, l in enumerate(LIV):
    ax.text(j, tot[l] + 3, f"{tot[l]}", ha="center", fontsize=9.5)
ax.set_ylabel("testate"); ax.legend(fontsize=9.5); ax.set_ylim(0, 175)
ax.set_title("Primo dataset: l'86% delle testate $l_1$ è una sola rete", fontsize=11)
salva(fig, "fig_primo_livelli_reti")

# =============================================================================================== dataset nuovo
fold = pd.read_csv(W / "Dataset/versioni/fold_finale.csv")
fold["liv"] = fold.label.map(dict(enumerate(LIV)))

# 4. l'imbuto: dalle testate tentate al dataset finale
fasi = [("testate NewsGuard tentate", 3128, None), ("raggiunte dallo scraper", 1528, 170213),
        ("articoli in inglese, >100 parole", 1046, 101887), ("dopo la pulizia degli articoli", 963, 76091),
        ("senza copie tra gruppi diversi", 963, 75630), ("con almeno 20 articoli", 702, 73308),
        ("al massimo 5 testate per gruppo", 304, 44029)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
y = np.arange(len(fasi))[::-1]
a1.barh(y, [f[1] for f in fasi], color=BLU, height=0.6)
a2.barh(y, [f[2] or 0 for f in fasi], color=ACQUA, height=0.6)
for yi, f in zip(y, fasi):
    a1.text(f[1] + 40, yi, f"{f[1]:,}".replace(",", "."), va="center", fontsize=9.5)
    if f[2]:
        a2.text(f[2] + 1500, yi, f"{f[2]:,}".replace(",", "."), va="center", fontsize=9.5)
a1.set_yticks(y, [f[0] for f in fasi]); a1.set_title("testate", fontsize=11); a2.set_title("articoli", fontsize=11)
a1.set_xlim(0, 3700); a2.set_xlim(0, 205000); a1.grid(axis="y", visible=False); a2.grid(axis="y", visible=False)
a1.set_xticks([]); a2.set_xticks([])
salva(fig, "fig_nuovo_imbuto")

# 5. composizione del dataset finale
comp = pd.read_csv(R1 / "03_composizione.csv").set_index("label").loc[LIV]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
b = a1.bar(LIV, comp.testate, color=[COL_LIV[l] for l in LIV], width=0.6)
etichetta_barre(a1, b, "{:.0f}"); a1.set_title("testate (304)", fontsize=11); a1.set_ylim(0, 112)
b = a2.bar(LIV, comp.articoli, color=[COL_LIV[l] for l in LIV], width=0.6)
for bb in b:
    a2.annotate(f"{int(bb.get_height()):,}".replace(",", "."), (bb.get_x() + bb.get_width() / 2, bb.get_height()), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=9.5)
a2.set_title("articoli (44.029)", fontsize=11); a2.set_ylim(0, 16000)
salva(fig, "fig_nuovo_composizione")

# 6. i segnali delle reti contro l'owner NewsGuard
ver = pd.read_csv(R1 / "02_verifica_owner.csv", index_col=0)
nomi = {"nessun segnale (ogni testata da sola)": "nessun segnale", "solo autori": "solo autori", "solo codice": "solo codici HTML",
        "solo copie": "solo copie", "solo dominio": "solo dominio", "solo titolare": "solo titolare legale", "tutti i segnali": "tutti i segnali"}
v = ver.loc[list(nomi)]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
for n, r in v.iterrows():
    col = BLU if n == "tutti i segnali" else (GRIGIO if n.startswith("nessun") else ACQUA)
    ax.scatter(r.recall, r.precisione, s=70, color=col, zorder=3, edgecolor="white", linewidth=1)
    dx, dy, ha = {"tutti i segnali": (-8, 0, "right"), "solo titolare": (-8, -6, "right"), "solo copie": (8, -4, "left"),
                  "solo autori": (8, 4, "left")}.get(n, (8, 0, "left"))
    ax.annotate(nomi[n], (r.recall, r.precisione), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=9.5)
ax.set_xlabel("recall: testate con lo stesso owner nella stessa rete"); ax.set_ylabel("precisione")
ax.set_xlim(0.5, 0.95); ax.set_ylim(0.88, 1.01)
ax.set_title("Le reti trovate senza guardare NewsGuard, verificate con l'owner (B-cubed)", fontsize=10)
salva(fig, "fig_reti_verifica")

# 7. le reti più grandi tra le testate raccolte
reti = pd.read_csv(R1 / "02_reti.csv")
ng = pd.read_csv(W / "Dataset/newsguard.csv", low_memory=False).assign(d=lambda x: x.Domain.str.lower()).drop_duplicates("d").set_index("d")
reti["owner"] = reti.domain.map(ng.Owner).fillna("?")
reti["liv"] = pd.cut(reti.domain.map(ng.Score), [-1, 39, 59, 74, 99, 100], labels=LIV)
top = reti.groupby("rete").agg(n=("domain", "size"), owner=("owner", lambda s: s.value_counts().index[0]),
                               liv=("liv", lambda s: ", ".join(sorted(set(s.dropna().astype(str)))))).sort_values("n", ascending=False).head(8)
fig, ax = plt.subplots(figsize=(8.2, 3.6))
y = np.arange(len(top))[::-1]
ax.barh(y, top.n, color=BLU, height=0.6)
for yi, (_, r) in zip(y, top.iterrows()):
    ax.text(r.n + 5, yi, f"{r.n}", va="center", fontsize=9.5)
etich = [f"{('owner sconosciuto' if o in ('Unknown', '?') else o.replace(' Group', '').replace(', Inc.', ''))[:34]} ({l})" for o, l in zip(top.owner, top.liv)]
ax.set_yticks(y, etich); ax.grid(axis="y", visible=False); ax.set_xlabel("testate nella rete"); ax.set_xlim(0, 480)
ax.set_title("Le reti più grandi tra le 1.046 testate raccolte (owner più frequente, livelli)", fontsize=10)
salva(fig, "fig_reti_grandi")

# 8. i fold sono bilanciati
ct = pd.crosstab(fold.fold_gruppo, fold.liv).reindex(columns=LIV)
fig, ax = plt.subplots(figsize=(6.4, 3.3))
base = np.zeros(len(ct))
for l in LIV:
    ax.bar([f"fold {i}" for i in ct.index], ct[l], bottom=base, color=COL_LIV[l], label=l, width=0.6, edgecolor="white", linewidth=1.5)
    for i, (b0, v_) in enumerate(zip(base, ct[l])):
        ax.text(i, b0 + v_ / 2, str(v_), ha="center", va="center", fontsize=8.5, color="white" if l in ("l1", "l5") else INCHIOSTRO)
    base += ct[l].to_numpy()
ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.set_ylabel("testate di test"); ax.set_ylim(0, 70)
salva(fig, "fig_fold_bilanciati")

# 9. la sporcizia, prima e dopo: tre misure sullo stesso grafico
pred = pd.read_csv(R2 / "feature_tuned_fold_gruppo_articoli.csv")
t = pred.groupby("domain").agg(y=("y_true", "first"), p=("y_pred", lambda s: s.value_counts().sort_index().idxmax()))
t["in_gruppo"] = t.index.map(fold.set_index("domain").testate_nel_gruppo) > 1
nuovo_dentro, nuovo_fuori = (t.y == t.p)[t.in_gruppo].mean(), (t.y == t.p)[~t.in_gruppo].mean()
l1_rete_primo = rete0["l1"] / tot["l1"]
l1_rete_nuovo = fold[fold.label == 0].groupby("gruppo").size().max() / (fold.label == 0).sum()
fig, assi = plt.subplots(1, 3, figsize=(11, 3.5))
pannelli = [("quota delle testate $l_1$\nnella rete più grande", ["primo dataset", "dataset nuovo"], [l1_rete_primo, l1_rete_nuovo],
             [f"{l1_rete_primo:.0%}", f"{l1_rete_nuovo:.0%}"]),
            ("dataset nuovo: coppie di articoli copiati\ncon una copia nel training e una nel test", ["fold per testata", "fold per gruppo"],
             [6157 / 7669, 0.0], ["6.157 su 7.669", "0 su 7.669"]),
            ("accuratezza delle testate in una rete\nmeno quella delle testate isolate", ["primo dataset\n(fold per testata)", "dataset nuovo\n(fold per gruppo)"],
             [val[val.in_gruppo].giusta_01.mean() - val[~val.in_gruppo].giusta_01.mean(), nuovo_dentro - nuovo_fuori], None)]
for ax, (tit, xs, vv, testi) in zip(assi, pannelli):
    b = ax.bar(xs, vv, color=[ARANCIO, BLU], width=0.55)
    ax.set_title(tit, fontsize=9.5); ax.set_yticks([]); ax.axhline(0, color="#b9b8b0", lw=1)
    for bb, v_, tx in zip(b, vv, testi or [f"{x:+.2f}" for x in vv]):
        ax.annotate(tx, (bb.get_x() + bb.get_width() / 2, max(v_, 0)), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=9)
assi[0].set_ylim(0, 1.05); assi[1].set_ylim(0, 1.05); assi[2].set_ylim(-0.1, 0.65)
salva(fig, "fig_sporcizia_prima_dopo")
print(f"(numeri: l1 nella rete più grande {l1_rete_primo:.2f} -> {l1_rete_nuovo:.2f}; dentro-fuori nuovo {nuovo_dentro:.2f}/{nuovo_fuori:.2f})")

# 10. i risultati con gli intervalli
iv = pd.read_csv(R2 / "03_confronto_intervalli.csv", index_col=0)
tm = pd.read_csv(R2 / "03_confronto_tutte_le_metriche.csv", index_col=[0, 1])
scelti = [("TF-IDF (C in validazione)", "feature_tuned (voto)", BLU), ("TF-IDF (C = 1)", "feature (media)", BLU),
          ("ModernBERT-base, 3 seed", "modernbert_base_1024_ep2_ensemble3 (media)", ARANCIO),
          ("ModernBERT-large, 3 seed", "modernbert_large_1024_ep2_ensemble3 (media)", ARANCIO),
          ("e5-large congelato", "e5_large_v2 (media)", ACQUA), ("bge-large congelato", "bge_large_v15 (media)", ACQUA)]
bm = json.load(open(R2 / "baseline_maggioranza_fold_gruppo_metriche.json"))["testata_voto"]
bc = json.load(open(R2 / "baseline_a_caso_fold_gruppo_metriche.json"))["testata_voto"]
fig, assi = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (col, lo, hi, base_m, base_c, tit) in zip(assi, [("F1 macro", "F1 2.5%", "F1 97.5%", bm["f1_macro"], bc["f1_macro"], "F1 macro a 5 classi (per testata)"),
                                                         ("binario", "bin 2.5%", "bin 97.5%", bm["bin_accuratezza"], bc["bin_accuratezza"], "accuratezza binaria (per testata)")]):
    for i, (nome, k, c) in enumerate(scelti[::-1]):
        r = iv.loc[k]
        ax.plot([r[lo], r[hi]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r[col], i, color=c, s=60, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(r[hi] + 0.008, i, f"{r[col]:.3f}", va="center", fontsize=9)
    for k_, (v_, n_) in enumerate(sorted([(base_m, "maggioranza"), (base_c, "a caso")])):
        ax.axvline(v_, color=GRIGIO, ls="--", lw=1.1)
        ax.text(v_ + (-0.004 if k_ == 0 else 0.004), -0.75, n_, color=MUTO, fontsize=8.5, ha="right" if k_ == 0 else "left")
    ax.set_ylim(-1.0, len(scelti) - 0.5)
    ax.set_title(tit, fontsize=10.5, pad=8); ax.grid(axis="y", visible=False)
assi[0].set_yticks(range(len(scelti)), [s[0] for s in scelti[::-1]]); assi[0].set_xlim(0.05, 0.5); assi[1].set_xlim(0.48, 0.88)
salva(fig, "fig_risultati_intervalli")

# 11. recall per livello: chi riconosce cosa
modelli = [("TF-IDF (C in validazione)", ("feature_tuned", "voto"), BLU), ("ModernBERT-base, 3 seed", ("modernbert_base_1024_ep2_ensemble3", "media"), ARANCIO),
           ("ModernBERT-large, 3 seed", ("modernbert_large_1024_ep2_ensemble3", "media"), "#f2a07c")]
fig, ax = plt.subplots(figsize=(7.4, 3.5))
x = np.arange(5); w = 0.26
for j, (n, k, c) in enumerate(modelli):
    b = ax.bar(x + (j - 1) * (w + 0.02), [tm.loc[k, f"{l}_recall"] for l in LIV], w, color=c, label=n)
    for bb in b:
        ax.annotate(f"{bb.get_height():.2f}".lstrip("0"), (bb.get_x() + bb.get_width() / 2, bb.get_height()), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, [f"${l[0]}_{l[1]}$" for l in LIV]); ax.set_ylabel("recall per testata"); ax.set_ylim(0, 0.9)
ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
salva(fig, "fig_recall_livelli")

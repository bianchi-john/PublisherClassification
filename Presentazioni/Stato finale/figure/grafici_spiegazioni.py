"""Grafici di spiegazione per FinalState.odp (revisione del 6 novembre):
- stirare_passi.png: lo stiramento passo per passo su 6 testate vere del giro 1 (Mistral), con i voti del training;
- stirare_distribuzioni.png: giro 1, voti veri del training, previsti e stirati del test, e la regola previsto → stirato;
- mbfc_abbinamento.png: come abbiniamo le testate a MBFC (226 → 170) e le classi;
- mbfc_fasce.png: classe MBFC contro fascia di voto NewsGuard (dove i due esperti concordano e dove no);
- kendall_coppie.png: le 28 coppie delle 8 testate dell'esempio, concordanti e discordanti;
- kendall_mistral.png: le coppie di Mistral su tutte le 226 testate.
Uso, da questa cartella:  python grafici_spiegazioni.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import mbfc as MB      # noqa: E402
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
BLU, GRIGIO, VERDE, ROSSO, CHIARO = "#1d4f91", "#8a8983", "#1baf7a", "#c62828", "#f0efe9"
COLORI_LIV = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]
COLORI_MBFC = ["#b02a2a", "#e0a43a", "#5b9be0", "#1d4f91"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
v1 = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")
art, testate = MO.load()


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


def tabella(ax, intestazioni, righe, colori=None, larghezze=None):
    """Una tabella nello stile delle slide di John: intestazione su fondo chiaro, righe sottili."""
    ax.axis("off")
    t = ax.table(cellText=righe, colLabels=intestazioni, loc="center", cellLoc="center", colWidths=larghezze)
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.7)
    for (r, c), cella in t.get_celld().items():
        cella.set_edgecolor("#e1e0d9"); cella.visible_edges = "B"
        if r == 0:
            cella.set_facecolor(CHIARO); cella.set_text_props(fontweight="bold"); cella.visible_edges = "TB"
        elif colori and c in colori:
            cella.set_text_props(color=colori[c], fontweight="bold")
    return t


# ------------------------------------------------------------------------------------------------ 1. stirare, passo per passo
T = ME.outlets_regressor(MO.predictions("regression", "Mistral-24B"), testate)
te, tr = T[T.fold == 1].copy(), T[T.fold != 1]
te["posizione"] = te.predicted.rank()
te["q"] = (te.posizione - 1) / (len(te) - 1)
te["stirato"] = np.quantile(tr["true score"], te.q)
te = te.sort_values("posizione")
scelte = te.iloc[[0, 8, 17, 26, 35, 44]]
fig, (a, b) = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
tabella(a, ["testata", "voto previsto", f"posizione\ntra le {len(te)} di test", "posizione\nin %",
            f"voto del training\n(181 testate) a quel %", "voto vero\n(NewsGuard)"],
        [[c, v1(r.predicted), f"{r.posizione:.0f}ª", f"{100 * r.q:.0f}%", v1(r.stirato), v1(r["true score"])]
         for c, (_, r) in zip("ABCDEF", scelte.iterrows())], colori={4: BLU}, larghezze=[0.1, 0.15, 0.19, 0.13, 0.25, 0.17])
a.set_title("Giro 1, Mistral: il voto stirato (in blu) è il voto del training alla stessa posizione")
s = np.sort(tr["true score"].to_numpy())
b.plot(100 * np.arange(len(s)) / (len(s) - 1), s, color=GRIGIO, lw=2, label="voti veri del training, in fila")
for c, (_, r) in zip("ABCDEF", scelte.iterrows()):
    b.scatter(100 * r.q, r.stirato, s=90, color=BLU, zorder=3)
    b.annotate(c, (100 * r.q, r.stirato), xytext=(-14, 6), textcoords="offset points", fontsize=12, fontweight="bold", color=BLU)
b.set_xlabel("posizione in fila (%)"); b.set_ylabel("voto"); b.set_ylim(-3, 105)
b.set_title("Si legge il voto del training a quella posizione"); b.legend(frameon=False, fontsize=9, loc="lower right")
salva("stirare_passi.png")

# ------------------------------------------------------------------------------------------------ 2. le distribuzioni del giro 1
fig, assi = plt.subplots(1, 4, figsize=(18, 4.4))
bins = np.arange(0, 102.6, 5)
for ax, (v, nome, colore) in zip(assi[:3], [(tr["true score"], "Training: 181 voti veri", GRIGIO),
                                            (te.predicted, "Test: 45 voti previsti", "#b5b3ab"),
                                            (te.stirato, "Test: 45 voti stirati", BLU)]):
    ax.hist(v, bins=bins, color=colore, edgecolor="white")
    ax.set_title(nome); ax.set_xlabel("voto"); ax.set_xlim(0, 102); ax.set_yticks([])
o = te.sort_values("predicted")
assi[3].step(o.predicted, o.stirato, where="post", color=BLU, lw=2)
assi[3].scatter(o.predicted, o.stirato, s=14, color=BLU)
assi[3].plot([0, 100], [0, 100], color="#c3c2b7", lw=1)
assi[3].set_xlim(0, 102); assi[3].set_ylim(0, 102); assi[3].set_xlabel("voto previsto"); assi[3].set_ylabel("voto stirato")
assi[3].set_title("La regola: sale sempre (stesso ordine)")
salva("stirare_distribuzioni.png")

# ------------------------------------------------------------------------------------------------ 3. MBFC: l'abbinamento
y = MB.load(testate)
mb = pd.read_csv(MB.MBFC); mb["d"] = mb.dominio.str.lower().str.replace(r"^www\.", "", regex=True)
originali = mb.drop_duplicates("d").set_index("d").factual.reindex(y.index.str.lower())
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
ordine = ["Very Low", "Low", "Mixed", "Mostly Factual", "High", "Very High"]
conta = originali.value_counts().reindex(ordine, fill_value=0)
colori6 = [COLORI_MBFC[MB.FACTUAL[c]] for c in ordine]
a.bar(range(6), conta, color=colori6, edgecolor="white")
for i, c in enumerate(conta):
    a.text(i, c, str(c), ha="center", va="bottom", fontsize=11)
a.set_xticks(range(6), [c.replace(" ", "\n") for c in ordine]); a.grid(axis="x", visible=False)
a.set_title(f"Le {len(y)} testate trovate in MBFC (stesso dominio): Very Low → Low, Very High → High")
dentro = testate.index.isin(y.index)
for i, (nome, sel) in enumerate([(f"in MBFC ({dentro.sum()})", dentro), (f"non in MBFC ({(~dentro).sum()})", ~dentro)]):
    q = np.bincount(testate.label[sel], minlength=5) / sel.sum()
    sx = 0
    for k in range(5):
        b.barh(i, q[k], left=sx, color=COLORI_LIV[k], height=0.6, edgecolor="white")
        if q[k] > 0.06:
            b.text(sx + q[k] / 2, i, f"{q[k]:.0%}", ha="center", va="center", fontsize=10, color="white" if k in (0, 4) else "black")
        sx += q[k]
b.set_yticks([0, 1], [f"in MBFC ({dentro.sum()})", f"non in MBFC ({(~dentro).sum()})"]); b.invert_yaxis(); b.set_xlim(0, 1); b.grid(False)
b.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
b.set_title("Livelli NewsGuard: le testate trovate e quelle no")
for k, (liv, f) in enumerate(zip(["l1", "l2", "l3", "l4", "l5"], ["0-39", "40-59", "60-74", "75-99", "100"])):
    b.bar(0, 0, color=COLORI_LIV[k], label=f"{liv} ({f})")
b.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
salva("mbfc_abbinamento.png")

# ------------------------------------------------------------------------------------------------ 4. classe MBFC × fascia NewsGuard
fascia = pd.Series(np.digitize(testate.score.reindex(y.index), [39.5, 59.5, 74.5, 99.5]), index=y.index)
C = pd.crosstab(y, fascia).reindex(index=range(4), columns=range(5), fill_value=0).to_numpy()
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())
for i in range(4):
    for j in range(5):
        lontano = (i == 0 and j >= 3) or (i == 3 and j <= 1)
        ax.text(j, i, C[i, j], ha="center", va="center", fontsize=13, fontweight="bold" if lontano else "normal",
                color=ROSSO if lontano and C[i, j] else ("white" if C[i, j] > C.max() / 2 else "black"))
        if lontano:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ROSSO, lw=1.5))
ax.set_xticks(range(5), ["0-39", "40-59", "60-74", "75-99", "100"]); ax.set_yticks(range(4), MB.CLASSES); ax.grid(False)
ax.set_xlabel("voto NewsGuard"); ax.set_ylabel("classe MBFC")
ax.set_title("Le 170 testate: in rosso i disaccordi forti (Low ≥ 75 o High < 60)")
salva("mbfc_fasce.png")
forti = int(C[0, 3:].sum() + C[3, :2].sum())

# ------------------------------------------------------------------------------------------------ 5. Kendall: le coppie dell'esempio
nomi = list("ABCDEFGH")
ng = np.array([92.5, 87.5, 75, 69.5, 57.5, 44.5, 30, 20]); mi = np.array([86.0, 83.0, 91.3, 51.6, 61.2, 79.0, 53.0, 50.1])
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.1]})
tabella(a, ["testata", "voto NewsGuard", "voto previsto"], [[n, v1(x), v1(p)] for n, x, p in zip(nomi, ng, mi)],
        colori={2: BLU}, larghezze=[0.25, 0.35, 0.35])
a.set_title("Le 8 testate dell'esempio dello Spearman")
conc = disc = 0
for i in range(8):
    for j in range(8):
        if j <= i:
            continue
        ok = np.sign(ng[i] - ng[j]) == np.sign(mi[i] - mi[j])
        conc += ok; disc += not ok
        b.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=VERDE if ok else ROSSO, alpha=0.85))
        b.text(j, i, "✓" if ok else "✗", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
b.set_xlim(-0.6, 7.6); b.set_ylim(7.6, -0.6); b.set_xticks(range(8), nomi); b.set_yticks(range(8), nomi)
b.xaxis.tick_top(); b.grid(False); b.set_aspect("equal")
for s in b.spines.values():
    s.set_visible(False)
b.set_title(f"Le 28 coppie: {conc} nello stesso ordine (✓), {disc} invertite (✗)", pad=28)
salva("kendall_coppie.png")

# ------------------------------------------------------------------------------------------------ 6. Kendall: le coppie di Mistral
v, p = T["true score"].to_numpy(), T.predicted.to_numpy()
s = (np.sign(v[:, None] - v[None, :]) * np.sign(p[:, None] - p[None, :]))[np.triu_indices(len(v), 1)]
n_c, n_d, n_p = int((s > 0).sum()), int((s < 0).sum()), int((s == 0).sum())
fig, ax = plt.subplots(figsize=(15, 2.6))
sx = 0
for n, nome, colore in [(n_c, "stesso ordine", VERDE), (n_d, "invertite", ROSSO), (n_p, "stesso voto NewsGuard", "#c3c2b7")]:
    ax.barh(0, n, left=sx, color=colore, height=0.6, edgecolor="white")
    piccolo = n < 0.1 * len(s)
    ax.text(sx + n / 2, -0.45 if piccolo else 0, f"{nome}\n{n:,}".replace(",", "."), ha="center", va="top" if piccolo else "center",
            fontsize=11, color="black" if piccolo else "white", fontweight="bold")
    sx += n
ax.set_xlim(0, sx); ax.axis("off")
ax.set_ylim(-1.1, 0.4)
ax.set_title("Mistral, tutte le 226 testate: " + f"{len(s):,}".replace(",", ".") + " coppie; "
             + f"tolte quelle con lo stesso voto, {100 * n_c / (n_c + n_d):.0f}% nell'ordine giusto")
salva("kendall_mistral.png")
print("disaccordi forti MBFC/NG:", forti, "| Kendall esempio:", conc, disc, "| Mistral coppie:", n_c, n_d, n_p)
print(scelte[["predicted", "posizione", "q", "stirato", "true score"]].round(2).to_string())

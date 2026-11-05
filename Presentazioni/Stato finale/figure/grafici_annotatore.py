"""Lo Spearman su testate VERE, passo per passo, e il confronto con MBFC (commenti di John del 5 novembre, sera):
- valutazione/sp_vero_voti.png:      8 testate vere del test, voto NewsGuard e voto previsto da Mistral;
- valutazione/sp_vero_posizioni.png: le stesse con le due posizioni (due file separate) e le linee tra le posizioni;
- valutazione/sp_vero_conto.png:     differenze di posizione, i loro quadrati e il conto dello Spearman;
- valutazione/mbfc_posizioni.png:    8 delle 170 testate in comune: voto NewsGuard, classe MBFC e le posizioni (con i
                                     pari merito della stessa classe);
- valutazione/mbfc_triangolo.png:    i tre confronti sulle stesse 170 testate (esperti tra loro, Mistral con ognuno).
Uso, da questa cartella:  python grafici_annotatore.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import mbfc as MB      # noqa: E402
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

OUT = QUI / "valutazione"
NERO, BLU, ROSSO, VERDE, GRIGIO, CHIARO = "#111111", "#1d4f91", "#c9211e", "#1baf7a", "#52514e", "#f1f0ea"
plt.rcParams.update({"figure.dpi": 150, "font.size": 15})


def virgola(x, d=1):
    x = round(float(x), d)
    t = f"{abs(x):.0f}" if x.is_integer() else f"{abs(x):.{d}f}".replace(".", ",")
    return ("−" if x < 0 else "") + t


def posizione(v):
    """1 = il voto più alto; i pari merito prendono la posizione media."""
    return rankdata(-np.asarray(v, float), method="average")


def tabella(ax, colonne, righe, colori=None, larghezze=None, grassetto=()):
    """Una tabella disegnata a mano: intestazione su fondo chiaro, una riga per testata."""
    n = len(colonne)
    larghezze = larghezze or [1] * n
    x = np.concatenate([[0], np.cumsum(larghezze)]) / sum(larghezze)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(len(righe) + 0.6, -0.9)
    ax.add_patch(plt.Rectangle((0, -0.9), 1, 1.1, color=CHIARO, lw=0))
    for j, c in enumerate(colonne):
        ax.text((x[j] + x[j + 1]) / 2, -0.35, c, ha="center", va="center", fontsize=14, fontweight="bold")
    for i, r in enumerate(righe):
        for j, v in enumerate(r):
            colore = (colori[i][j] if colori else None) or NERO
            ax.text((x[j] + x[j + 1]) / 2, i + 0.6, v, ha="center", va="center", fontsize=15, color=colore,
                    fontweight="bold" if j in grassetto else "normal")
        ax.plot([0, 1], [i + 1.1, i + 1.1], color="#e1e0d9", lw=0.8)


def salva(nome):
    plt.tight_layout(); plt.savefig(OUT / nome, bbox_inches="tight"); plt.close()


art, testate = MO.load()
T = ME.outlets_regressor(MO.predictions("regression", "Mistral-24B"), testate)
RHO = spearmanr(T["true score"], T.predicted).correlation

# ------------------------------------------------------------------------------------------------ 1. lo Spearman su 8 testate vere
for seme in range(5000):
    s = T.sample(8, random_state=seme)
    v, p = s["true score"].to_numpy(), s.predicted.to_numpy()
    if (np.min(np.diff(np.sort(v))) >= 5 and v.max() - v.min() >= 70 and np.min(np.diff(np.sort(p))) >= 1
            and abs(spearmanr(v, p).correlation - RHO) < 0.04 and v.max() < 100):
        break
s = s.sort_values("true score", ascending=False).copy()
s["nome"] = list("ABCDEFGH")
s["pos vero"], s["pos prev"] = posizione(s["true score"]), posizione(s.predicted)
s["d"] = s["pos vero"] - s["pos prev"]
n, somma = len(s), int((s.d ** 2).sum())
rho8 = 1 - 6 * somma / (n * (n ** 2 - 1))
assert abs(rho8 - spearmanr(s["true score"], s.predicted).correlation) < 1e-9

fig, ax = plt.subplots(figsize=(9, 5.2))
tabella(ax, ["testata", "voto NewsGuard", "voto previsto\n(Mistral)"],
        [[f"testata {r.nome}", virgola(r["true score"]), f"{r.predicted:.1f}".replace(".", ",")] for _, r in s.iterrows()],
        colori=[[None, None, BLU]] * n, larghezze=[1, 1.2, 1.2])
salva("sp_vero_voti.png")

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
tabella(a, ["testata", "voto\nNewsGuard", "posizione\nNewsGuard", "voto\nprevisto", "posizione\nprevisto"],
        [[r.nome, virgola(r["true score"]), f"{r['pos vero']:.0f}", f"{r.predicted:.1f}".replace(".", ","), f"{r['pos prev']:.0f}"] for _, r in s.iterrows()],
        colori=[[None, GRIGIO, NERO, GRIGIO, BLU]] * n, larghezze=[0.8, 1, 1, 1, 1], grassetto=(2, 4))
b.axis("off"); b.set_xlim(-1.2, 2.2); b.set_ylim(n + 0.6, -0.9)
b.text(0.1, -0.35, "fila per\nNewsGuard", ha="right", va="center", fontsize=14, fontweight="bold")
b.text(0.9, -0.35, "fila per\nvoto previsto", ha="left", va="center", fontsize=14, fontweight="bold")
for _, r in s.iterrows():
    c = BLU if r.d == 0 else ROSSO
    b.plot([0.15, 0.85], [r["pos vero"], r["pos prev"]], color=c, lw=2.2)
    b.text(0.1, r["pos vero"], f"{r['pos vero']:.0f}. {r.nome}", ha="right", va="center", fontsize=15)
    b.text(0.9, r["pos prev"], f"{r['pos prev']:.0f}. {r.nome}", ha="left", va="center", fontsize=15)
salva("sp_vero_posizioni.png")

fig, ax = plt.subplots(figsize=(14, 5.6))
righe = [[f"testata {r.nome}", f"{r['pos vero']:.0f}", f"{r['pos prev']:.0f}", virgola(r.d), f"{r.d ** 2:.0f}"] for _, r in s.iterrows()]
righe.append(["", "", "", "somma", f"{somma}"])
tabella(ax, ["testata", "posizione\nNewsGuard", "posizione\nprevisto", "differenza", "differenza²"], righe,
        colori=[[None, None, None, ROSSO if r[3] != "0" else None, ROSSO if r[3] != "0" else None] for r in righe[:-1]] + [[None, None, None, NERO, NERO]],
        larghezze=[1.2, 1, 1, 1, 1], grassetto=())
ax.text(1.03, (n + 0.6) / 2, f"Spearman = 1 − 6 × somma / (n × (n² − 1))\n\n= 1 − 6 × {somma} / (8 × 63)\n\n= 1 − {6 * somma} / 504\n\n= {virgola(rho8, 2)}",
        transform=ax.transData, ha="left", va="center", fontsize=16, color=BLU, fontweight="bold")
ax.set_xlim(0, 1.75)
salva("sp_vero_conto.png")

# ------------------------------------------------------------------------------------------------ 2. NewsGuard e MBFC: un voto e una classe
y = MB.load(testate)
C = pd.DataFrame({"ng": testate.score[y.index], "mb": y})
classi = np.array(MB.CLASSES)
for seme in range(5000):
    s2 = C.sample(8, random_state=seme)
    if (sorted(s2.mb.value_counts().reindex(range(4), fill_value=0).tolist()) == [2, 2, 2, 2] and s2.ng.nunique() == 8
            and abs(spearmanr(s2.ng, s2.mb).correlation - 0.71) < 0.06
            and (s2.groupby("mb").ng.min().diff().dropna() < 0).any()):     # almeno un'eccezione: classe più alta, voto più basso
        break
s2 = s2.sort_values("ng", ascending=False).copy()
s2["nome"] = list("ABCDEFGH")
s2["pos ng"], s2["pos mb"] = posizione(s2.ng), posizione(s2.mb)
rho_mb = spearmanr(s2.ng, s2.mb).correlation
fig, ax = plt.subplots(figsize=(14, 5.4))
tabella(ax, ["testata", "voto\nNewsGuard", "posizione\nNewsGuard", "classe MBFC", "posizione\nMBFC"],
        [[r.nome, virgola(r.ng), f"{r['pos ng']:.0f}", classi[r.mb], virgola(r["pos mb"])] for _, r in s2.iterrows()],
        colori=[[None, GRIGIO, NERO, GRIGIO, BLU]] * 8, larghezze=[0.7, 1, 1, 1.4, 1], grassetto=(2, 4))
ax.text(0.5, 8.55, f"Spearman di queste 8 testate: {virgola(rho_mb, 2)}  ·  su tutte le 170: {virgola(spearmanr(C.ng, C.mb).correlation, 2)}",
        ha="center", fontsize=15, color=GRIGIO)
ax.set_ylim(9.0, -0.9)
salva("mbfc_posizioni.png")

# ------------------------------------------------------------------------------------------------ 3. i tre confronti sulle 170 testate
cmb = pd.read_csv(OUT / "curve_mbfc.csv")
mi = cmb[(cmb.modello == "Mistral-24B") & (cmb.k == "tutti")].set_index("con").Spearman
esperti = spearmanr(C.ng, C.mb).correlation
fig, ax = plt.subplots(figsize=(12, 5.4)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5.6)
nodi = {"NG": (2.2, 4.5, "NewsGuard\nvoto 0-100", NERO), "MB": (9.8, 4.5, "MBFC\nclasse (4)", GRIGIO),
        "MI": (6.0, 0.9, "Mistral\nvoto previsto (test)", BLU)}
for (a, b_, val, testo) in [("NG", "MB", esperti, "i due esperti"), ("MI", "NG", mi["NewsGuard"], "modello e NewsGuard"),
                            ("MI", "MB", mi["MBFC"], "modello e MBFC")]:
    (x1, y1, *_), (x2, y2, *_) = nodi[a], nodi[b_]
    ax.plot([x1, x2], [y1, y2], color="#b5b3ab", lw=3, zorder=1)
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + (0.35 if a == "NG" else 0), f"{virgola(val, 2)}\n{testo}", ha="center", va="center",
            fontsize=15, fontweight="bold", color=BLU if a == "MI" else NERO,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
for x, y_, t, c in nodi.values():
    ax.text(x, y_, t, ha="center", va="center", fontsize=16, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=c, edgecolor="none"))
salva("mbfc_triangolo.png")
print("Spearman 8 testate:", round(rho8, 3), "somma d²", somma, "| MBFC 8:", round(rho_mb, 3), "| Mistral:", mi.round(3).to_dict(), "esperti", round(esperti, 3))
print(s[["true score", "predicted", "pos vero", "pos prev", "d"]].round(1).to_string())
print(s2[["ng", "mb", "pos ng", "pos mb"]].to_string())

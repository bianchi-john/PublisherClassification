"""Lo Spearman spiegato con esempi inventati (5 testate, A-E), per le slide di valutazione di FinalState.odp:
- valutazione/spearman_idea.png: due classifiche delle stesse testate, un solo scambio (Spearman 0,9);
- valutazione/spearman_casi.png: quattro casi: 1, 0,9, 0, −1;
- valutazione/spearman_ordine.png: due modelli: uno in ordine ma schiacciato (Spearman 1, errore alto), uno vicino ma con
  due scambi (Spearman 0,8, errore basso).
Uso, da questa cartella:  python grafici_spearman.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

QUI = Path(__file__).resolve().parent
OUT = QUI / "valutazione"
NERO, BLU, ROSSO, GRIGIO = "#111111", "#1d4f91", "#c9211e", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 14, "axes.titlesize": 14, "axes.titleweight": "bold"})
def virgola(x, d=1):
    """Numero all'italiana: 1, −1, 0,9, 57,5 (niente «,0» e meno tipografico)."""
    x = round(float(x), d)
    t = f"{abs(x):.0f}" if x.is_integer() else f"{abs(x):.{d}f}".replace(".", ",")
    return ("−" if x < 0 else "") + t

NOMI = list("ABCDE")
VERO = np.array([12.5, 37, 57.5, 75, 95])


def posizioni(v):
    """1 = voto più alto."""
    return (-np.asarray(v)).argsort().argsort() + 1


def classifiche(ax, previsto, titolo=None, voti=True, fs=14, x_testo=0.3):
    """Due colonne: classifica per voto vero (sinistra) e per voto previsto (destra), linee tra le stesse testate."""
    pv, pp = posizioni(VERO), posizioni(previsto)
    ax.axis("off"); ax.set_xlim(-1.6, 3.6); ax.set_ylim(5.7, 0.2)
    for i, n in enumerate(NOMI):
        c = BLU if pv[i] == pp[i] else ROSSO
        ax.plot([0.9, 2.1], [pv[i], pp[i]], color=c, lw=2.2)
        ax.text(0.8, pv[i], f"{pv[i]}. {n}" + (f" ({virgola(VERO[i])})" if voti else ""), ha="right", va="center", fontsize=fs)
        ax.text(2.2, pp[i], f"{pp[i]}. {n}" + (f" ({virgola(previsto[i])})" if voti else ""), ha="left", va="center", fontsize=fs)
    rho = spearmanr(VERO, previsto).correlation
    if titolo:
        ax.set_title(titolo.format(rho=virgola(rho, 1)), fontsize=fs + 1)
    return rho


# 1. l'idea: un solo scambio
previsto = np.array([30, 45, 70, 62, 85])
fig, ax = plt.subplots(figsize=(12, 3.9))
classifiche(ax, previsto)
ax.set_ylim(5.7, -0.4)
ax.text(0.8, -0.15, "classifica per voto vero", ha="right", fontsize=14, fontweight="bold")
ax.text(2.2, -0.15, "classifica per voto previsto", ha="left", fontsize=14, fontweight="bold")
ax.text(3.3, 3.5, "C e D si sono\nscambiate di posto\n→ Spearman 0,9", ha="left", va="center", fontsize=14, color=ROSSO)
plt.tight_layout(); plt.savefig(OUT / "spearman_idea.png", bbox_inches="tight"); plt.close()

# 2. quattro casi (le posizioni previste di A..E, 1 = la più alta)
casi = [("Stessa classifica", [5, 4, 3, 2, 1]), ("Uno scambio", [5, 4, 2, 3, 1]),
        ("Nessun legame", [4, 1, 3, 5, 2]), ("Rovesciata", [1, 2, 3, 4, 5])]
fig, assi = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, (nome, pos) in zip(assi, casi):
    previsto = 100 - np.array(pos) * 15.0          # voti inventati che danno quelle posizioni
    assert list(posizioni(previsto)) == pos
    classifiche(ax, previsto, nome + ": Spearman {rho}", voti=False, fs=13)
plt.tight_layout(); plt.savefig(OUT / "spearman_casi.png", bbox_inches="tight"); plt.close()

# 3. conta solo l'ordine: due modelli
modelli = [("Modello 1: schiacciato ma in ordine", np.array([55, 58, 60, 62, 65])),
           ("Modello 2: vicino ma con due scambi", np.array([30, 20, 70, 95, 80]))]
fig, assi = plt.subplots(1, 2, figsize=(15, 3.9), sharey=True)
for ax, (nome, p) in zip(assi, modelli):
    for i, n in enumerate(NOMI):
        ax.plot([VERO[i], p[i]], [i, i], color="#b5b3ab", lw=3, zorder=1)
        ax.scatter(VERO[i], i, s=90, color=NERO, zorder=3, label="voto vero" if i == 0 else None)
        ax.scatter(p[i], i, s=90, color=BLU, zorder=3, label="voto previsto" if i == 0 else None)
    rho, mae = spearmanr(VERO, p).correlation, np.abs(p - VERO).mean()
    ax.set_title(f"{nome}\nSpearman {virgola(rho, 1)} · errore medio {virgola(mae)} punti", loc="left")
    ax.set_xlim(0, 105); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("voto")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e1e0d9")
assi[0].set_yticks(range(5), [f"testata {n}" for n in NOMI]); assi[0].invert_yaxis()
fig.legend(*assi[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(); plt.savefig(OUT / "spearman_ordine.png", bbox_inches="tight"); plt.close()
print("ok")

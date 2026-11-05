"""Grafico delle testate sorelle (stesso gruppo di proprietà NewsGuard) passo per passo della pulizia del tentativo 20.
Legge Dataset/tentativi/20_FromScratch/sorelle.csv (calcolato sul server da sorelle.py, che rifà la pulizia con le
funzioni di cleaning.py e i gruppi di proprietà di NewsGuard delle 1.293 testate scaricate).
Uso, da questa cartella:  python grafico_sorelle.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
t = pd.read_csv(QUI.parents[2] / "Dataset/tentativi/20_FromScratch/sorelle.csv")
t = t[t.passo != "con testo estratto"]          # stesso numero di sorelle del passo prima: non aggiunge niente
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
fig, ax = plt.subplots(figsize=(14, 4.4))
x = np.arange(len(t))
sorelle = t["sorelle (testate - gruppi)"].to_numpy()
ax.bar(x, t.gruppi, color="#2a78d6", width=0.62, label="una testata per gruppo")
ax.bar(x, sorelle, bottom=t.gruppi, color="#eb6834", width=0.62, label="testate sorelle (altre dello stesso gruppo)")
for i, (g, s) in enumerate(zip(t.gruppi, sorelle)):
    ax.text(i, g + s + 12, f"{s} sorelle", ha="center", fontsize=10, color="#b84a1f", fontweight="bold")
    ax.text(i, g / 2, f"{g}\ngruppi", ha="center", va="center", fontsize=9, color="white")
nomi = {"scaricate (1.293)": "mandate allo\nscraper", "almeno un articolo": "almeno un\narticolo", "inglese": "inglese",
        "100-1.500 parole": "100-1.500\nparole", "dal 2025": "dal 2025", "senza doppioni": "senza\ndoppioni",
        "senza quasi doppioni": "senza quasi\ndoppioni", "almeno 10 articoli": "almeno 10\narticoli", "una per gruppo": "una per\ngruppo"}
ax.set_xticks(x, [nomi.get(p, p) for p in t.passo], fontsize=10); ax.grid(axis="x", visible=False)
ax.set_ylabel("testate"); ax.set_ylim(0, t.testate.max() * 1.1)
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.set_title("Le testate dopo ogni passo: in arancione quelle che hanno una sorella nello stesso gruppo di proprietà")
plt.tight_layout(); plt.savefig(QUI / "sorelle.png", bbox_inches="tight"); plt.close()
print(t.to_string(index=False))

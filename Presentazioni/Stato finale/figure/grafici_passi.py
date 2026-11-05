"""Grafici per le slide «a passi» (commenti di John del 5 novembre, sera): ModernBERT spiegato passo per passo e la
valutazione di un regressore passo per passo.
- modelli/bert_ciclo.png: il ciclo di addestramento (leggi, prevedi, confronta, correggi);
- modelli/bert_lr.png: quanto è grande ogni correzione (learning rate) lungo le 5 epoche: warm-up e discesa;
- modelli/bert_memoria.png: ogni articolo ha come obiettivo il voto della sua testata: ~180 voti diversi;
- modelli/bert_errore.png, modelli/bert_ordine.png: ModernBERT contro gli altri (errore per articolo; Spearman);
- valutazione/distanze.png: le 226 distanze di Mistral, con l'errore medio e le soglie 10 / 20;
- valutazione/classifiche.png: lo Spearman come confronto di due classifiche (8 testate vere);
- valutazione/senza_testo.png: i modelli contro la baseline «voto medio» (errore medio, % testate con errore ≤ 20).
Uso, da questa cartella:  python grafici_passi.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
D20 = REPO / "Dataset/tentativi/20_FromScratch"
sys.path.insert(0, str(REPO / "Code/WebScience2027/Tentativi/20_FromScratch"))
import metrics as ME   # noqa: E402
import models as MO    # noqa: E402

M_, V_ = QUI / "modelli", QUI / "valutazione"
NOMI = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
COLORE = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834", "baseline: voto medio": "#c3c2b7"}
BLU, ARANCIO, VERDE, ROSSO, GRIGIO = "#2a78d6", "#eb6834", "#1baf7a", "#c9211e", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left"})
virgola = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")


def salva(percorso):
    plt.tight_layout(); plt.savefig(percorso, bbox_inches="tight"); plt.close()


def scatola(ax, x, y, w, h, colore, testo, bianco=True, fs=14):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.15", color=colore, lw=0))
    ax.text(x, y, testo, ha="center", va="center", fontsize=fs, color="white" if bianco else "black")


def freccia(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, color=GRIGIO, lw=2, connectionstyle=f"arc3,rad={rad}"))


art, testate = MO.load()
ic = pd.read_csv(D20 / "intervals.csv")


def valore(modello, misura, vista="mean score", approccio="regressor"):
    r = ic[(ic.approach == approccio) & (ic.view == vista) & (ic.metric == misura) & (ic.model == modello)].iloc[0]
    return r.value, r.low, r.high


# 1. il ciclo di addestramento
fig, ax = plt.subplots(figsize=(13, 3.6)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6)
passi = [(1.3, "un articolo\ndi training", "#a8a69c"), (4.4, "ModernBERT\n(149 milioni di pesi)", BLU),
         (7.5, "voto previsto\nes. 61", BLU), (10.8, "confronto con il voto\ndella testata, es. 80\n→ errore 19", ARANCIO)]
for x, t, c in passi:
    scatola(ax, x, 2.4, 2.5 if x < 10 else 3.0, 1.45, c, t)
for (x1, *_), (x2, *_) in zip(passi, passi[1:]):
    freccia(ax, (x1 + 1.3, 2.4), (x2 - (1.55 if x2 > 10 else 1.3), 2.4))
freccia(ax, (10.8, 1.6), (4.4, 1.6), rad=-0.25)
ax.text(7.6, 0.25, "si correggono un poco TUTTI i pesi, per sbagliare meno la volta dopo", ha="center", fontsize=14, color=GRIGIO)
salva(M_ / "bert_ciclo.png")

# 2. il learning rate lungo l'addestramento (stessa regola del codice: warm-up 6%, poi discesa lineare)
n_train = len(art) * 4 / 5
passi_epoca = int(np.ceil(n_train / 32)); tot = 5 * passi_epoca; warm = int(0.06 * tot)
s = np.arange(tot)
lr = np.where(s < warm, s / max(warm, 1), (tot - s) / (tot - warm)) * 3e-5
fig, ax = plt.subplots(figsize=(13, 3.6))
ax.plot(s, lr * 1e5, color=BLU, lw=2.5)
for e in range(1, 5):
    ax.axvline(e * passi_epoca, color="#c3c2b7", lw=1, ls=":")
for e in range(5):
    ax.text((e + 0.5) * passi_epoca, 3.25, f"epoca {e + 1}", ha="center", fontsize=12, color=GRIGIO)
ax.annotate("warm-up: si parte\ncon passi piccoli", xy=(warm * 0.6, 1.9), xytext=(warm + 250, 1.0), fontsize=12,
            arrowprops=dict(arrowstyle="->", color=GRIGIO))
ax.set_ylim(0, 3.5); ax.set_xlim(0, tot); ax.set_xlabel(f"passo di correzione (uno ogni 32 articoli; ~{passi_epoca} per epoca)")
ax.set_ylabel("grandezza del passo\n(learning rate, ×10⁻⁵)", fontsize=12)
ax.set_title("Quanto si corregge a ogni passo: sale in fretta, poi scende fino a zero")
salva(M_ / "bert_lr.png")

# 3. perché impara a memoria: tutti gli articoli di una testata hanno lo stesso obiettivo
fig, ax = plt.subplots(figsize=(13, 3.4)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3.4)
for j, (nome, voto, c) in enumerate([("testata A", "92,5", "#1d4f91"), ("testata B", "62,5", "#a8a69c"), ("testata C", "20", "#b02a2a")]):
    x0 = 0.3 + j * 4.2
    scatola(ax, x0 + 1.8, 2.85, 3.6, 0.7, c, f"{nome}: voto {voto}")
    for k in range(4):
        scatola(ax, x0 + 0.45 + k * 0.9, 1.75, 0.8, 1.0, "#e1e0d9", f"art.\n{k + 1}", bianco=False, fs=11)
    ax.text(x0 + 1.8, 0.85, f"obiettivo di tutti: {voto}", ha="center", fontsize=13, color=c, fontweight="bold")
ax.text(6.5, 0.15, "~180 testate di training → solo ~180 voti diversi da indovinare", ha="center", fontsize=14, color=GRIGIO)
salva(M_ / "bert_memoria.png")

# 4. ModernBERT contro gli altri: errore per articolo e ordine delle testate
errore = {}
for m in NOMI:
    p = MO.predictions("regression", m).merge(art[["article_id", "score"]], on="article_id")
    errore[m] = np.sqrt(np.mean((p.prediction - p.score) ** 2))
bert_train = np.mean([np.sqrt(pd.read_csv(D20 / f"predictions/regression/ModernBERT/loss_fold{k}.csv").loss.iloc[-1]) * 100
                      for k in range(1, 6)])
fig, ax = plt.subplots(figsize=(13, 3.9))
etichette = ["ModernBERT\nsul training"] + [f"{m}\nsul test" for m in NOMI]
valori = [bert_train] + [errore[m] for m in NOMI]
barre = ax.bar(range(5), valori, color=[COLORE["ModernBERT"]] + [COLORE[m] for m in NOMI], width=0.6)
barre[0].set_hatch("//"); barre[0].set_alpha(0.45)
for k, v in enumerate(valori):
    ax.text(k, v + 0.6, virgola(v), ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(5), etichette, fontsize=13); ax.set_ylim(0, 30); ax.grid(axis="x", visible=False)
ax.set_ylabel("punti"); ax.set_title("Errore per articolo (scarto quadratico medio, punti)")
salva(M_ / "bert_errore.png")

fig, ax = plt.subplots(figsize=(13, 3.6))
for k, m in enumerate(NOMI):
    v, a, b = valore(m, "Spearman")
    ax.plot([a, b], [k, k], color=COLORE[m], lw=2.5, alpha=0.45)
    ax.scatter(v, k, s=120, color=COLORE[m], zorder=3)
    ax.text(b + 0.008, k, virgola(v, 2), va="center", fontsize=14, color=COLORE[m], fontweight="bold")
ax.set_yticks(range(4), NOMI); ax.invert_yaxis(); ax.set_xlim(0.45, 0.85); ax.grid(axis="y", visible=False)
ax.set_xlabel("Spearman tra voto vero e voto previsto delle 226 testate (con intervallo al 95%)")
ax.set_title("Ordine delle testate di test: ModernBERT è il peggiore")
salva(M_ / "bert_ordine.png")

# 5. le distanze di Mistral: errore medio, entro 10, entro 20
T = ME.outlets_regressor(MO.predictions("regression", "Mistral-24B"), testate)
dist = (T.predicted - T["true score"]).abs()
fig, ax = plt.subplots(figsize=(13, 3.9))
ax.hist(dist, bins=np.arange(0, dist.max() + 2.5, 2.5), color=COLORE["Mistral-24B"], alpha=0.85)
for soglia, testo, lato in [(10, f"errore ≤ 10:\n{(dist <= 10).mean():.0%} delle testate  ", "right"), (20, f"  errore ≤ 20:\n  {(dist <= 20).mean():.0%} delle testate", "left")]:
    ax.axvline(soglia, color=VERDE, lw=2.2)
    ax.text(soglia, ax.get_ylim()[1] * 0.98, testo, ha=lato, va="top", color=VERDE, fontsize=14, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
ax.axvline(dist.mean(), color=ARANCIO, lw=2.2, ls="--")
ax.text(dist.mean() + 0.6, ax.get_ylim()[1] * 0.5, f"errore medio\n{virgola(dist.mean())} punti", color=ARANCIO, fontsize=14, fontweight="bold")
ax.set_xlabel("distanza tra voto previsto e voto NewsGuard (punti)"); ax.set_ylabel("testate")
ax.set_title(f"Mistral: le distanze delle {len(dist)} testate")
ax.grid(axis="x", visible=False)
salva(V_ / "distanze.png")

# 6. lo Spearman: due classifiche delle stesse testate
# 6 testate con voti veri tutti diversi e lontani tra loro, il cui Spearman è vicino a quello di tutte le 226
RHO = valore("Mistral-24B", "Spearman")[0]
for seme in range(1000):
    scelte = T.sample(6, random_state=seme).sort_values("true score")
    vero = scelte["true score"].to_numpy()
    if np.min(np.diff(vero)) >= 8 and vero[-1] - vero[0] >= 70 and scelte.predicted.nunique() == 6 \
            and abs(spearmanr(vero, scelte.predicted).correlation - RHO) < 0.06:
        break
scelte = scelte.copy()
scelte["nome"] = [f"testata {c}" for c in "ABCDEF"]
scelte["pos vero"] = scelte["true score"].rank(ascending=False, method="first")
scelte["pos prev"] = scelte.predicted.rank(ascending=False, method="first")
rho = spearmanr(scelte["true score"], scelte.predicted).correlation
fig, ax = plt.subplots(figsize=(13, 3.8)); ax.axis("off"); ax.set_xlim(-0.2, 4.2); ax.set_ylim(7.0, -0.3)
ax.text(0.9, -0.2, "classifica per voto NewsGuard", ha="center", fontsize=14, fontweight="bold")
ax.text(3.1, -0.2, "classifica per voto previsto (Mistral)", ha="center", fontsize=14, fontweight="bold")
for _, r in scelte.iterrows():
    y1, y2 = r["pos vero"], r["pos prev"]
    c = BLU if abs(y1 - y2) <= 1 else ROSSO
    ax.plot([1.75, 2.25], [y1, y2], color=c, lw=2)
    ax.text(1.7, y1, f"{int(y1)}. {r.nome} ({virgola(r['true score'])})", ha="right", va="center", fontsize=13)
    ax.text(2.3, y2, f"{int(y2)}. {r.nome} ({virgola(r.predicted)})", ha="left", va="center", fontsize=13)
ax.text(2.0, 6.9, f"Spearman di queste 6 testate: {virgola(rho, 2)}  ·  su tutte le 226: {virgola(valore('Mistral-24B', 'Spearman')[0], 2)}",
        ha="center", fontsize=14, color=GRIGIO)
salva(V_ / "classifiche.png")

# 7. i modelli contro la baseline «voto medio»
righe = [("baseline: voto medio", "no text", "mean training score", "baseline")] + [(m, m, "mean score", "regressor") for m in NOMI]
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
for k, (nome, modello, vista, approccio) in enumerate(righe):
    for ax, misura, f, d in [(a, "MAE", 1, 1), (b, "within 20 points", 100, 0)]:
        v = valore(modello, misura, vista, approccio)[0] * f
        ax.barh(k, v, color=COLORE[nome], height=0.6)
        ax.text(v + (0.3 if f == 1 else 1), k, virgola(v, d) + ("%" if f == 100 else ""), va="center", fontsize=13, fontweight="bold")
a.set_yticks(range(5), [r[0] for r in righe]); a.invert_yaxis()
a.set_title("Errore medio (punti): più basso è meglio"); b.set_title("% testate con errore ≤ 20: più alto è meglio")
a.set_xlim(0, 26); b.set_xlim(0, 80)
for ax in (a, b):
    ax.grid(axis="y", visible=False)
salva(V_ / "senza_testo.png")
print("ok; passi per epoca", passi_epoca, "warm-up", warm, "| scelte:", scelte[["true score", "predicted"]].round(1).values.tolist(), "rho", round(rho, 2))

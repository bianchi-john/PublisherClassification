"""Grafici della presentazione «Stato finale»: composizione di NewsGuard, degli articoli scaricati, della pulizia e del
dataset finale. Legge Dataset/ (radice del repo) e il notebook 02 eseguito del tentativo 20.
Uso, da questa cartella:  python genera_grafici.py"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2]
D = REPO / "Dataset"
T20 = REPO / "Code/WebScience2027/Tentativi/20_FromScratch"
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
COLORI = ["#b02a2a", "#ee8f8f", "#a8a69c", "#5b9be0", "#1d4f91"]      # l1 … l5: due poli e il grigio in mezzo
BLU, GRIGIO = "#2a78d6", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
livello = lambda s: np.digitize(np.asarray(s, float), [39.5, 59.5, 74.5, 99.5])

# --------------------------------------------------------------------------- dati
ng = pd.read_csv(D / "etichette/newsguard_2026.csv", low_memory=False)
ng = ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")
stati = pd.DataFrame([json.load(open(f)) for c in ["NuovoScraping_ottobre", "NuovoScraping_giro3"]
                      for f in (D / "scraping" / c / "stato").glob("*.json")]).set_index("dominio")
estratti = pd.read_csv(D / "tentativi/16_DataPreparation/cache/estratti.csv.gz", usecols=["domain"])   # = passo 2.1 del 20
testate = pd.read_csv(D / "tentativi/20_FromScratch/outlets.csv", index_col="domain")
articoli = pd.read_csv(D / "tentativi/20_FromScratch/articles_anonymous.csv.gz", usecols=["domain", "date", "label"])

voti_ng = ng.Score[(ng.Language == "en") & ng.Score.notna()]
argomenti = ng.Topics.reindex(stati.index).fillna("").str.split(",").map(lambda x: [a.strip() for a in x if a.strip()])
politiche = stati.index[(argomenti.map(lambda x: x == ["Political news or commentary"]) & ng.Score.reindex(stati.index).notna()).to_numpy()]
stati = stati.loc[politiche]                     # lo scraping delle sole testate politiche (1.293)
gruppi = {"NewsGuard 2026\n(inglese, con voto)": voti_ng,
          "solo politiche,\nscaricate": ng.Score.reindex(stati.index).dropna(),
          "con almeno\nun articolo": ng.Score.reindex(estratti.domain.unique()).dropna(),
          "dataset finale": testate.score}

# --------------------------------------------------------------------------- 1. NewsGuard
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
bins = np.arange(0, 102.6, 2.5)
a.hist([voti_ng[livello(voti_ng) == k] for k in range(5)], bins, stacked=True, color=COLORI, label=LIVELLI)
a.axvline(59.5, color=GRIGIO, ls="--", lw=1); a.text(61, a.get_ylim()[1] * 0.72, "affidabile →", color=GRIGIO)
a.set_title(f"NewsGuard 2026: {len(voti_ng):,} testate in inglese con un voto".replace(",", "."))
a.set_xlabel("voto NewsGuard"); a.set_ylabel("testate"); a.legend(ncol=5, fontsize=9, frameon=False, loc="upper left")
for i, (nome, v) in enumerate(gruppi.items()):
    quote = np.bincount(livello(v), minlength=5) / len(v)
    sinistra = 0
    for k in range(5):
        b.barh(i, quote[k], left=sinistra, color=COLORI[k], height=0.62, edgecolor="white", linewidth=1)
        if quote[k] > 0.06:
            b.text(sinistra + quote[k] / 2, i, f"{quote[k]:.0%}", ha="center", va="center", fontsize=9,
                   color="white" if k in (0, 4) else "black")
        sinistra += quote[k]
    b.text(1.01, i, f"{len(v):,}".replace(",", "."), va="center", fontsize=9, color=GRIGIO)
b.set_yticks(range(len(gruppi)), list(gruppi)); b.invert_yaxis(); b.set_xlim(0, 1.1); b.grid(False)
b.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
b.set_title("Quota di testate per livello, dal catalogo al dataset (a destra: quante)")
plt.tight_layout(); plt.savefig(QUI / "newsguard.png", bbox_inches="tight"); plt.close()

# --------------------------------------------------------------------------- 2. gli articoli scaricati
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.2))
n = stati.articoli[stati.articoli > 0]
a.hist(n, bins=np.arange(0, 205, 5), color=BLU)
a.set_title(f"Articoli sopra le 100 parole per testata ({len(n):,} testate con almeno uno)".replace(",", "."))
a.set_xlabel("articoli"); a.set_ylabel("testate")
per_testata = estratti.domain.value_counts()
per_livello = pd.Series(livello(ng.Score.reindex(per_testata.index)), index=per_testata.index)
conta = per_testata.groupby(per_livello).sum().reindex(range(5), fill_value=0)
b.bar(LIVELLI, conta, color=COLORI)
for k, v in enumerate(conta):
    b.text(k, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=10)
b.set_title(f"Testate solo politiche: {len(estratti):,} articoli per livello".replace(",", "."))
b.set_ylabel("articoli")
plt.tight_layout(); plt.savefig(QUI / "articoli_scaricati.png", bbox_inches="tight"); plt.close()

# --------------------------------------------------------------------------- 3. la pulizia, passo per passo
NOMI = {"articles of political outlets": "articoli scaricati", "with extracted text": "con testo estratto",
        "English": "in inglese", "100-1,500 words": "100-1.500 parole", "published from 2025": "pubblicati dal 2025",
        "no exact duplicates": "senza doppioni", "no near duplicates (Jaccard >= 0.5)": "senza quasi doppioni",
        "outlets with >= 10 articles": "testate con almeno 10 articoli", "one outlet per ownership group": "una testata per gruppo di proprietà"}
passi = []
for c in nbformat.read(T20 / "02_cleaning.ipynb", 4).cells:
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            for m in re.finditer(r"^(.+?): ([\d,]+) articles, ([\d,]+) outlets$", o.text, flags=re.M):
                passi.append((NOMI.get(m.group(1), m.group(1)), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))))
passi = pd.DataFrame(passi, columns=["passo", "articoli", "testate"]).drop_duplicates("passo")
fig, assi = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, col in zip(assi, ["articoli", "testate"]):
    ax.barh(passi.passo[::-1], passi[col][::-1], color=BLU)
    for y, v in enumerate(passi[col][::-1]):
        ax.text(v, y, f" {v:,}".replace(",", "."), va="center", fontsize=9, color=GRIGIO)
    ax.set_title(col); ax.grid(axis="y", visible=False); ax.set_xlim(0, passi[col].max() * 1.18)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
plt.tight_layout(); plt.savefig(QUI / "pulizia.png", bbox_inches="tight"); plt.close()

# --------------------------------------------------------------------------- 4. il dataset finale
fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
x = np.arange(5)
t_l = testate.label.value_counts().reindex(range(5)); a_l = articoli.label.value_counts().reindex(range(5))
a.bar(LIVELLI, t_l, color=COLORI)
for k, v in enumerate(t_l):
    a.text(k, v, str(v), ha="center", va="bottom", fontsize=10)
a.set_title(f"{len(testate)} testate per livello"); a.set_ylabel("testate")
b.bar(LIVELLI, a_l, color=COLORI)
for k, v in enumerate(a_l):
    b.text(k, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
b.set_title(f"{len(articoli):,} articoli per livello".replace(",", ".")); b.set_ylabel("articoli")
mesi = pd.to_datetime(articoli.date, errors="coerce").dt.to_period("M").value_counts().sort_index()
c.bar([str(p) for p in mesi.index], mesi.values, color=BLU)
c.set_xticks(range(0, len(mesi), 3), [str(p) for p in mesi.index][::3], rotation=45, ha="right", fontsize=9)
c.set_title("Articoli per mese di pubblicazione"); c.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(QUI / "dataset_finale.png", bbox_inches="tight"); plt.close()
print("passi della pulizia:", len(passi)); print(passi.to_string(index=False))

# --------------------------------------------------------------------------- 5. l'anonimizzazione: quanto tocca ogni regola
REGOLE = {"signature": "firma nei titoli", "email": "email", "url": "link", "account": "account @", "author": "autori della testata",
          "name": "nome della testata"}
righe = None
for c in nbformat.read(T20 / "03_anonymization.ipynb", 4).cells:
    for o in c.get("outputs", []):
        if o.output_type == "execute_result" and "articles touched" in o.data.get("text/plain", ""):
            righe = o.data["text/plain"].splitlines()[1:]
tab = pd.DataFrame([r.split() for r in righe], columns=["regola", "testo", "sostituzioni", "titoli"]).set_index("regola")
tab = tab.astype(float).rename(index=REGOLE)
fig, ax = plt.subplots(figsize=(11, 3.4))
y = np.arange(len(tab))
ax.barh(y, tab.testo, height=0.6, color=BLU)
for i, a_ in enumerate(tab.testo):
    ax.text(a_, i, f" {a_:.1f}%".replace(".", ","), va="center", fontsize=10)
ax.set_yticks(y, tab.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("articoli con almeno una sostituzione nel testo (%)")
ax.set_title("Quanto tocca ogni regola")
plt.tight_layout(); plt.savefig(QUI / "anonimizzazione.png", bbox_inches="tight"); plt.close()

# --------------------------------------------------------------------------- 6. lo split per voto
split = pd.read_csv(D / "tentativi/20_FromScratch/split.csv", index_col="domain").fold.reindex(testate.index)
COLORI_FOLD = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#52514e"]
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
casuale = np.random.default_rng(1).random(len(testate))       # lo stesso ordine dello split (seme 1, parità a caso)
o = testate.score.iloc[np.lexsort((casuale, testate.score.to_numpy()))].head(30)
for i in range(0, 30, 5):
    a.axvspan(i - 0.45, i + 4.45, color="#f0efe9" if (i // 5) % 2 == 0 else "white", zorder=0)
for i, (d, v) in enumerate(o.items()):
    a.scatter(i, v, s=70, color=COLORI_FOLD[int(split[d]) - 1], zorder=3)
a.set_xticks([]); a.set_ylabel("voto NewsGuard"); a.grid(False)
a.set_title("Le 30 testate con il voto più basso, in ordine: in ogni blocco di 5 un colore per fold")
conta = pd.crosstab(split, testate.label).reindex(columns=range(5))
sinistra = np.zeros(5)
for k in range(5):
    b.barh(conta.index, conta[k], left=sinistra, color=COLORI[k], edgecolor="white", label=LIVELLI[k])
    for f, v, s_ in zip(conta.index, conta[k], sinistra):
        b.text(s_ + v / 2, f, str(v), ha="center", va="center", fontsize=9, color="white" if k in (0, 4) else "black")
    sinistra += conta[k].to_numpy()
b.set_yticks(conta.index, [f"fold {f}" for f in conta.index]); b.invert_yaxis(); b.grid(False)
b.set_title("Testate per livello in ogni fold"); b.legend(ncol=5, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.25))
plt.tight_layout(); plt.savefig(QUI / "split.png", bbox_inches="tight"); plt.close()

# --------------------------------------------------------------------------- 7. split per voto e split per livello (tentativo 16) a confronto
livello16 = pd.read_csv(D / "tentativi/16_DataPreparation/testate.csv", index_col="domain").fold.reindex(testate.index)
fig, assi = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for ax, (fold, titolo) in zip(assi, [(livello16, "Per livello (tentativo 16)"), (split, "Per voto (quello usato)")]):
    r = np.random.default_rng(0)
    for f in range(1, 6):
        t = testate[fold == f]
        ax.scatter(f + r.uniform(-0.25, 0.25, len(t)), t.score, s=12, color=[COLORI[k] for k in t.label], alpha=0.85)
        ax.plot([f - 0.32, f + 0.32], [t.score.mean()] * 2, color="#111111", lw=2)
        l1 = t.score[t.label == 0].mean()
        ax.plot([f - 0.32, f + 0.32], [l1] * 2, color=COLORI[0], lw=2.5)
        ax.text(f + 0.34, l1, f"{l1:.0f}", va="center", fontsize=9, color=COLORI[0], fontweight="bold")
    medie = testate.score[testate.label == 0].groupby(fold[testate.label == 0]).mean()
    ax.set_title(f"{titolo}: voto medio delle l1 da {medie.min():.0f} a {medie.max():.0f}")
    ax.set_xticks(range(1, 6), [f"fold {f}" for f in range(1, 6)]); ax.grid(axis="x", visible=False)
assi[0].set_ylabel("voto NewsGuard")
assi[1].plot([], [], color="#111111", lw=2, label="voto medio del fold"); assi[1].plot([], [], color=COLORI[0], lw=2.5, label="voto medio delle l1")
fig.legend(frameon=False, fontsize=10, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(); plt.savefig(QUI / "split_confronto.png", bbox_inches="tight"); plt.close()
print("anonimizzazione e split fatti")

# --------------------------------------------------------------------------- 8. le testate mandate allo scraper (solo il dataset di partenza, niente imbuto)
v = ng.Score.reindex(politiche).astype(float)
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
a.hist([v[livello(v) == k] for k in range(5)], np.arange(0, 102.6, 2.5), stacked=True, color=COLORI, label=LIVELLI)
a.axvline(59.5, color=GRIGIO, ls="--", lw=1); a.text(61, a.get_ylim()[1] * 0.9, "affidabile →", color=GRIGIO)
a.set_xlabel("voto NewsGuard 2026"); a.set_ylabel("testate"); a.legend(ncol=5, fontsize=9, frameon=False, loc="upper left")
a.set_title("Il voto delle testate")
conta = pd.Series(livello(v)).value_counts().reindex(range(5), fill_value=0)
b.bar(LIVELLI, conta, color=COLORI)
for k, c_ in enumerate(conta):
    b.text(k, c_, f"{c_:,}".replace(",", "."), ha="center", va="bottom", fontsize=10)
b.set_xticks(range(5), [f"{l}\n{r}" for l, r in zip(LIVELLI, ["0-39", "40-59", "60-74", "75-99", "100"])])
b.set_title("Testate per livello"); b.grid(axis="x", visible=False)
fig.suptitle(f"Le {len(v):,} testate mandate allo scraper".replace(",", ".") + ": solo politiche, con un voto NewsGuard 2026",
             x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.savefig(QUI / "selezione_scraping.png", bbox_inches="tight"); plt.close()
print("testate mandate allo scraper:", len(v), conta.to_dict())

"""Costruisce il notebook del tentativo 17 (accordo tra annotatori). Se il codice del notebook non cambia, il notebook
resta com'è, con i suoi output; se cambia, si riscrive senza output e va rieseguito."""
from pathlib import Path

import nbformat as nbf

CARTELLA = Path(__file__).resolve().parent


def notebook(celle, nome, kernel="johnvenv"):
    for c in celle:                         # la cella dei parametri (MAIUSCOLE = valore) va etichettata per papermill
        if c.cell_type == "code" and c.source.split("\n")[0].split("=")[0].strip().isupper() and "import" not in c.source:
            c.metadata["tags"] = ["parameters"]; break
    for c in celle:
        if c.cell_type == "code":
            compile(c.source, nome, "exec")
    vecchio = CARTELLA / nome
    if vecchio.exists() and [c.source for c in nbf.read(vecchio, as_version=4).cells
                             if "injected-parameters" not in c.metadata.get("tags", [])] == [c.source for c in celle]:
        print(nome, "invariato"); return
    nb = nbf.v4.new_notebook(cells=celle)
    nb.metadata["kernelspec"] = {"name": kernel, "display_name": "johnVenv", "language": "python"}
    nbf.write(nb, vecchio)
    print(nome, len(celle), "celle")


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

notebook([
    md("""# Tentativo 17 · Il modello come terzo annotatore

**La domanda.** Dal solo testo dei suoi articoli si può giudicare l'affidabilità di una testata, come fanno gli esperti?
I criteri di NewsGuard e di Media Bias/Fact Check (MBFC) guardano cose che un modello non vede (le fonti, le correzioni,
chi possiede il sito). La scommessa è che l'affidabilità lasci comunque un'impronta, una specie di **DNA**, in ogni
articolo.

**Il disegno.** Tre annotatori danno un giudizio sulle stesse testate:
- **NewsGuard**: il punteggio complessivo, in 5 livelli (l1–l5); affidabile = almeno 60 (l3–l5);
- **MBFC**: la fattualità ("factual reporting", da Very Low a Very High), in 5 livelli; affidabile = almeno "Mostly Factual";
- **il modello**: legge gli articoli della testata, dà un livello a ognuno, e la testata prende il livello votato dalla
  maggioranza; affidabile = la maggioranza degli articoli è almeno l3.

Si misura quanto vanno d'accordo, a coppie, con due misure standard: la **correlazione di Spearman** tra i livelli e la
**kappa di Cohen** su affidabile sì/no (l'accordo corretto per il caso: 0 = a caso, 1 = sempre d'accordo). Il punto di
riferimento è l'accordo tra i **due esperti**, NewsGuard e MBFC.

**Perché è pulito.**
- Non si addestra niente: le previsioni sono quelle del tentativo 16, e ogni testata è stata giudicata da un modello che
  non l'aveva mai vista (era nel fold di test).
- MBFC non è mai stato usato per addestrare: il confronto con MBFC è del tutto indipendente.
- Soglie, misure e modelli sono fissati qui, prima di guardare i risultati.
- I modelli sono quattro, uno per famiglia: TF-IDF, ModernBERT addestrato, Qwen3-Embedding 8B congelato, Mistral-Small 24B
  congelato (strato 27); ognuno come classificatore a 5 classi e come regressore sul punteggio."""),
    code('''T16_DIR = "../16_DataPreparation"          # il tentativo 16: dataset, fold e previsioni dei modelli
B = 2000                                    # ricampionamenti bootstrap per gli intervalli al 95%
R = 200                                     # ripetizioni per ogni numero di articoli nella curva del punto 3'''),
    code('''import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T16 = Path(T16_DIR).expanduser(); sys.path.insert(0, str(T16))
import funzioni as F                        # le funzioni del tentativo 16 (dove sono i dati)
import accordo as A

warnings.filterwarnings("ignore")
OUT = Path("Results"); OUT.mkdir(exist_ok=True)
BLU, ARANCIO, GRIGIO, CHIARO = "#2a78d6", "#eb6834", "#52514e", "#b5b3ab"
plt.rcParams.update(F.STILE_GRAFICI)
t = pd.read_csv(F.DATASET / "testate.csv", index_col="domain")
mb = A.leggi_mbfc(F.DATI / "tentativi" / "10_EtichetteMBFC" / "mbfc_pulito.csv", t)
TESTATE = mb.index
ng = t.loc[TESTATE]
print(f"testate del tentativo 16: {len(t)}; valutate anche da MBFC (stesso dominio): {len(TESTATE)}")
print("livelli NewsGuard di queste testate:", ng.livello.value_counts().sort_index().to_dict())'''),
    md("""## 1. I due esperti

Le 170 testate valutate da tutti e due. La tabella incrocia il livello NewsGuard con la fattualità MBFC: se i due
esperti fossero sempre d'accordo, i numeri starebbero sulla diagonale. Sotto, il loro accordo e il confronto con la
credibilità MBFC (che oltre alla fattualità tiene conto anche dell'orientamento politico e del traffico del sito: per
questo la misura principale è la fattualità)."""),
    code('''ordine = ["Very Low", "Low", "Mixed", "Mostly Factual", "High", "Very High"]
display(pd.crosstab(ng.livello, pd.Categorical(mb.factual, ordine, ordered=True), rownames=["NewsGuard"], colnames=["MBFC fattualità"]))
NG_LV, NG_AFF = ng.label.to_numpy(), (ng.label >= A.SOGLIA["NewsGuard"]).astype(int).to_numpy()
MB_LV, MB_AFF = mb.fattualita.to_numpy(), (mb.fattualita >= A.SOGLIA["MBFC"]).astype(int).to_numpy()
CR_LV, CR_AFF = mb.credibilita_livello.to_numpy(), (mb.credibilita_livello >= A.SOGLIA["MBFC credibilità"]).astype(int).to_numpy()
pd.DataFrame({"NewsGuard ↔ MBFC fattualità": A.misure(NG_LV, MB_LV, NG_AFF, MB_AFF),
              "NewsGuard ↔ MBFC credibilità": A.misure(NG_LV, CR_LV, NG_AFF, CR_AFF)}).T.round(3)'''),
    md("""## 2. L'accordo dei modelli, con tutti gli articoli

Per ogni modello e impostazione: l'accordo con NewsGuard e con MBFC, accanto a quello tra i due esperti. Le ultime due
colonne rispondono alla domanda del tentativo:
- **Δ con NewsGuard**: il modello è vicino a NewsGuard quanto lo è MBFC? (modello↔NewsGuard − MBFC↔NewsGuard)
- **Δ con MBFC**: il modello è vicino a MBFC quanto lo è NewsGuard? (modello↔MBFC − NewsGuard↔MBFC)

Zero vuol dire "come un esperto"; tra parentesi l'intervallo al 95% (bootstrap sulle testate, appaiato)."""),
    code('''R16 = T16 / "Results"
mbert = [pd.read_csv(R16 / "modernbert" / c / f"fold{k}.csv") for c in ["5 classi", "regressione"] for k in range(1, 6)]
MODELLI = {"TF-IDF": pd.read_csv(R16 / "feature" / "previsioni_articoli.csv"),
           "ModernBERT": pd.concat(mbert[:5]).merge(pd.concat(mbert[5:])[["article_id", "s_regressione"]], on="article_id"),
           "Qwen3-Embedding 8B": pd.read_csv(R16 / "analisi_embedding" / "articoli_previsioni.csv"),
           "Mistral-Small 24B": pd.read_csv(R16 / "analisi_embedding" / "mistral27_articoli_previsioni.csv")}
GRUPPI = {}                                          # (modello, impostazione) → i livelli degli articoli di ogni testata
for m, pa in MODELLI.items():
    lv = A.livelli_articoli(pa, t).set_index("domain")
    for imp in ["classificatore", "regressore"]:
        GRUPPI[(m, imp)] = A.gruppi(lv[imp], TESTATE)
GIUDIZI = {k: A.giudizio(g) for k, g in GRUPPI.items()}

coppie = {("esperti", "", "NewsGuard ↔ MBFC"): (NG_LV, MB_LV, NG_AFF, MB_AFF)}
for (m, imp), (lv, aff) in GIUDIZI.items():
    coppie[(m, imp, "↔ NewsGuard")] = (lv, NG_LV, aff, NG_AFF)
    coppie[(m, imp, "↔ MBFC")] = (lv, MB_LV, aff, MB_AFF)
valori, campioni = A.bootstrap(coppie, len(TESTATE), B)
umano = campioni[("esperti", "", "NewsGuard ↔ MBFC")]


def cella(v, c):
    lo, hi = np.percentile(c, [2.5, 97.5]); return f"{v:.2f} [{lo:.2f}; {hi:.2f}]"


righe = {}
for (m, imp) in GIUDIZI:
    r = {}
    for mis in ["Spearman", "kappa"]:
        for rif in ["↔ NewsGuard", "↔ MBFC"]:
            r[f"{mis} {rif}"] = cella(valori[(m, imp, rif)][mis], campioni[(m, imp, rif)][mis])
            d = campioni[(m, imp, rif)][mis] - umano[mis]
            r[f"Δ {mis} {'con NewsGuard' if 'News' in rif else 'con MBFC'}"] = cella(valori[(m, imp, rif)][mis] - valori[("esperti", "", "NewsGuard ↔ MBFC")][mis], d)
    righe[(m, imp)] = r
tabella = pd.DataFrame(righe).T
tabella.to_csv(OUT / "accordo.csv")
print("esperti, NewsGuard ↔ MBFC:", {k: cella(v, umano[k]) for k, v in valori[("esperti", "", "NewsGuard ↔ MBFC")].items()})
tabella'''),
    md("""## 3. Il DNA: quanti articoli servono?

La stessa misura, ma il modello giudica la testata leggendo solo **k articoli presi a caso** (1, 3, 5, 9, 15, 25 o tutti),
con il voto di maggioranza: con k = 1 è il giudizio di **un solo articolo**. Ogni punto è la media di 200 estrazioni. La
fascia grigia è l'accordo tra i due esperti (con il suo intervallo al 95%)."""),
    code('''K = [1, 3, 5, 9, 15, 25, None]
rng = np.random.default_rng(0)
curva = []
for (m, imp), g in GRUPPI.items():
    for k in K:
        ripetizioni = R if k is not None else 1
        for _ in range(ripetizioni):
            lv, aff = A.giudizio(g, k, rng)
            for rif, (a, a_aff) in {"↔ NewsGuard": (NG_LV, NG_AFF), "↔ MBFC": (MB_LV, MB_AFF)}.items():
                curva.append({"modello": m, "impostazione": imp, "k": k or 999, "riferimento": rif, **A.misure(lv, a, aff, a_aff)})
curva = pd.DataFrame(curva).groupby(["modello", "impostazione", "riferimento", "k"])[["Spearman", "kappa", "accordo"]].mean().reset_index()
curva.to_csv(OUT / "curva_articoli.csv", index=False)

COLORI = {"TF-IDF": GRIGIO, "ModernBERT": "#9b59b6", "Qwen3-Embedding 8B": "#1baf7a", "Mistral-Small 24B": "#b02a2a"}   # non blu/arancio: lì vogliono dire NewsGuard/MBFC
fig, assi = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
for r, mis in enumerate(["Spearman", "kappa"]):
    lo, hi = np.percentile(umano[mis], [2.5, 97.5]); v = valori[("esperti", "", "NewsGuard ↔ MBFC")][mis]
    for c, rif in enumerate(["↔ NewsGuard", "↔ MBFC"]):
        ax = assi[r, c]
        ax.axhspan(lo, hi, color=CHIARO, alpha=0.35); ax.axhline(v, color=GRIGIO, lw=1.2)
        ax.text(1, v, f"  i due esperti (NewsGuard ↔ MBFC): {v:.2f}".replace(".", ","), va="bottom", fontsize=8.5, color=GRIGIO)
        for (m, imp), d in curva[curva.riferimento == rif].groupby(["modello", "impostazione"]):
            ax.plot(d.k, d[mis], marker="o", ms=3.5, color=COLORI[m], ls="-" if imp == "classificatore" else "--",
                    label=f"{m}, {imp}" if (r, c) == (0, 0) else None)
        ax.set_xscale("log"); ax.set_xticks([1, 3, 5, 9, 15, 25, 999], ["1", "3", "5", "9", "15", "25", "tutti"])
        ax.set_title(f"{'Spearman tra i livelli' if mis == 'Spearman' else 'kappa su affidabile sì/no'}: modello {rif}")
        if r == 1: ax.set_xlabel("articoli letti per testata (voto di maggioranza)")
fig.legend(loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f"Il DNA negli articoli: l'accordo del modello cresce con gli articoli letti · {len(TESTATE)} testate",
             fontsize=12.5, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "curva_articoli.png", dpi=180, bbox_inches="tight"); plt.show()'''),
    md("""## 4. Dove gli esperti non sono d'accordo

Le testate si dividono in due gruppi: quelle su cui NewsGuard e MBFC danno lo **stesso** giudizio affidabile sì/no, e
quelle su cui **non** sono d'accordo. Per ogni modello, quante volte il suo giudizio coincide con NewsGuard nei due gruppi.
Se il modello sbaglia soprattutto dove anche gli esperti litigano, i suoi errori stanno nei casi davvero incerti."""),
    code('''concordi = NG_AFF == MB_AFF
print(f"testate su cui i due esperti concordano: {concordi.sum()} su {len(concordi)} ({concordi.mean():.0%})")
dis = pd.DataFrame({f"{m}, {imp}": {"esperti d'accordo": (aff == NG_AFF)[concordi].mean(), "esperti in disaccordo": (aff == NG_AFF)[~concordi].mean()}
                    for (m, imp), (lv, aff) in GIUDIZI.items()}).T
dis.to_csv(OUT / "disaccordo.csv")
ax = dis.plot.barh(color=[BLU, ARANCIO], figsize=(8, 4.2), width=0.75)
ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("quota di testate su cui il modello coincide con NewsGuard (affidabile sì/no)")
ax.set_title(f"Il modello sbaglia dove sbagliano gli esperti? ({concordi.sum()} testate concordi, {(~concordi).sum()} in disaccordo)")
plt.tight_layout(); plt.savefig(OUT / "disaccordo.png", dpi=180, bbox_inches="tight"); plt.show()
dis.round(2)'''),
    md("""## 5. Controllo: la credibilità MBFC al posto della fattualità

Le stesse misure del punto 2 con la credibilità MBFC (bassa, media, alta; affidabile = alta). Se il quadro regge, il
risultato non dipende dalla scelta della misura MBFC."""),
    code('''controllo = {"esperti: NewsGuard ↔ MBFC credibilità": A.misure(NG_LV, CR_LV, NG_AFF, CR_AFF)}
for (m, imp), (lv, aff) in GIUDIZI.items():
    controllo[f"{m}, {imp} ↔ MBFC credibilità"] = A.misure(lv, CR_LV, aff, CR_AFF)
pd.DataFrame(controllo).T.round(3)'''),
    md("""## 6. Il quadro in un colpo d'occhio

Per ogni modello (classificatore e regressore, con tutti gli articoli e il voto di maggioranza): in blu l'accordo con
NewsGuard, in arancio con MBFC, linee grigie: intervalli al 95%. La fascia grigia verticale è l'accordo tra i due esperti.
Un modello che arriva alla fascia giudica le testate come un esperto in più."""),
    code('''righe_g = list(GIUDIZI)
fig, assi = plt.subplots(1, 2, figsize=(15, 0.62 * len(righe_g) + 2), sharey=True)
y = np.arange(len(righe_g))[::-1]
for ax, mis, titolo in zip(assi, ["Spearman", "kappa"], ["Spearman tra i livelli (5 livelli)", "kappa di Cohen su affidabile sì/no"]):
    v = valori[("esperti", "", "NewsGuard ↔ MBFC")][mis]; lo, hi = np.percentile(umano[mis], [2.5, 97.5])
    ax.axvspan(lo, hi, color=CHIARO, alpha=0.35); ax.axvline(v, color=GRIGIO, lw=1.2)
    for dy, rif, colore in [(0.18, "↔ NewsGuard", BLU), (-0.18, "↔ MBFC", ARANCIO)]:
        for yy, k in zip(y, righe_g):
            x = valori[(*k, rif)][mis]; a, b = np.percentile(campioni[(*k, rif)][mis], [2.5, 97.5])
            ax.barh(yy + dy, x, height=0.34, color=colore)
            ax.errorbar(x, yy + dy, xerr=[[x - a], [b - x]], color=GRIGIO, lw=1, capsize=2.5)
            ax.text(b + 0.01, yy + dy, f"{x:.2f}".replace(".", ","), va="center", fontsize=8, color=GRIGIO, fontweight="bold")
    ax.set_xlim(0, 1); ax.grid(False); ax.grid(axis="x")
    ax.set_title(f"{titolo}\\n▮ fascia grigia: i due esperti, NewsGuard ↔ MBFC = {v:.2f}".replace(".", ","), fontsize=10)
assi[0].set_yticks(y, [f"{m}\\n{imp}" for m, imp in righe_g], fontsize=9.5)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in [BLU, ARANCIO, CHIARO]],
           ["modello ↔ NewsGuard (il modello è addestrato su NewsGuard, testate mai viste)", "modello ↔ MBFC (mai usato per addestrare)",
            "i due esperti tra loro"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=9)
fig.suptitle(f"Il modello come terzo annotatore · {len(TESTATE)} testate valutate da NewsGuard e da MBFC · voto di maggioranza degli articoli",
             fontsize=12.5, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "colpo_d_occhio.png", dpi=180, bbox_inches="tight"); plt.show()'''),
    md("""## 7. Altre viste dello stesso accordo

**Che cosa vuol dire "accordo".** Due annotatori giudicano le stesse 170 testate. La **correlazione di Spearman** dice se
le mettono nello stesso ordine, dalla meno alla più affidabile (1 = stesso ordine, 0 = nessun legame). La **kappa** dice
quante volte danno lo stesso giudizio affidabile sì/no, togliendo le volte in cui sarebbero d'accordo per caso (1 = sempre,
0 = come a caso). Il modello si tratta come un annotatore qualunque.

**Il consenso dei due esperti** (fissato prima di guardare i risultati): per i livelli, la media del livello NewsGuard e
del livello MBFC; per affidabile sì/no, le 135 testate su cui i due esperti danno lo stesso giudizio.

### 7.1 Una barra per ogni coppia di annotatori
In cima, in grigio, i due esperti tra loro. Sotto, per ogni modello: con NewsGuard (blu), con MBFC (arancio), con il
consenso dei due esperti (verde)."""),
    code("""CONS_LV = (NG_LV + MB_LV) / 2                          # consenso sui livelli: la media dei due esperti
D = NG_AFF == MB_AFF                                     # testate su cui i due esperti concordano su affidabile sì/no
coppie_c = {k: (lv, CONS_LV, aff, NG_AFF) for k, (lv, aff) in GIUDIZI.items()}
val_c, camp_c = A.bootstrap(coppie_c, len(TESTATE), B)                          # Spearman con il consenso (170 testate)
coppie_d = {k: (lv[D], NG_LV[D], aff[D], NG_AFF[D]) for k, (lv, aff) in GIUDIZI.items()}
val_d, camp_d = A.bootstrap(coppie_d, int(D.sum()), B)                          # kappa sulle testate concordi (135)

VERDE = "#1baf7a"
fig, assi = plt.subplots(1, 2, figsize=(15, 0.95 * len(GIUDIZI) + 2.6), sharey=True)
nomi = ["NewsGuard ↔ MBFC\\n(i due esperti tra loro)"] + [f"{m}\\n{imp}" for m, imp in GIUDIZI]
y = np.arange(len(nomi))[::-1]
for ax, mis, titolo in zip(assi, ["Spearman", "kappa"], ["Spearman tra i livelli: stesso ordine delle testate?",
                                                         "kappa su affidabile sì/no: stesso giudizio, oltre il caso?"]):
    v = valori[("esperti", "", "NewsGuard ↔ MBFC")][mis]; lo, hi = np.percentile(umano[mis], [2.5, 97.5])
    ax.barh(y[0], v, height=0.5, color=GRIGIO); ax.errorbar(v, y[0], xerr=[[v - lo], [hi - v]], color="k", lw=1, capsize=2.5)
    ax.text(hi + 0.01, y[0], f"{v:.2f}".replace(".", ","), va="center", fontsize=8.5, fontweight="bold")
    ax.axvline(v, color=GRIGIO, ls=":", lw=1)
    for yy, k in zip(y[1:], GIUDIZI):
        barre = [(0.27, valori[(*k, "↔ NewsGuard")][mis], campioni[(*k, "↔ NewsGuard")][mis], BLU),
                 (0.0, valori[(*k, "↔ MBFC")][mis], campioni[(*k, "↔ MBFC")][mis], ARANCIO),
                 (-0.27, (val_c if mis == "Spearman" else val_d)[k][mis], (camp_c if mis == "Spearman" else camp_d)[k][mis], VERDE)]
        for dy, x, c, colore in barre:
            a, b = np.percentile(c, [2.5, 97.5])
            ax.barh(yy + dy, x, height=0.25, color=colore); ax.errorbar(x, yy + dy, xerr=[[x - a], [b - x]], color=GRIGIO, lw=0.8, capsize=2)
            ax.text(b + 0.01, yy + dy, f"{x:.2f}".replace(".", ","), va="center", fontsize=7.5, color=GRIGIO)
    ax.set_xlim(0, 1.05); ax.grid(False); ax.grid(axis="x"); ax.set_title(titolo, fontsize=10.5)
assi[0].set_yticks(y, nomi, fontsize=9)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in [GRIGIO, BLU, ARANCIO, VERDE]],
           ["i due esperti tra loro", "modello ↔ NewsGuard", "modello ↔ MBFC (mai usato per addestrare)",
            "modello ↔ consenso dei due esperti (livelli: media; sì/no: le 135 testate concordi)"],
           ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=9)
fig.suptitle(f"Chi va d'accordo con chi · {len(TESTATE)} testate · linee: intervalli al 95%", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "barre_per_coppia.png", dpi=180, bbox_inches="tight"); plt.show()"""),
    md("""### 7.2 La matrice dell'accordo tra tutti gli annotatori
Ogni casella è l'accordo tra l'annotatore della riga e quello della colonna (con tutti gli articoli e il voto di
maggioranza). La casella NewsGuard–MBFC è l'accordo tra i due esperti: i modelli che hanno con NewsGuard (o con MBFC) un
valore simile sono "esperti in più"."""),
    code("""ANNOTATORI = {"NewsGuard": (NG_LV, NG_AFF), "MBFC": (MB_LV, MB_AFF)}
ANNOTATORI.update({f"{m}, {imp}": v for (m, imp), v in GIUDIZI.items()})
nomi_a = list(ANNOTATORI)
fig, assi = plt.subplots(1, 2, figsize=(17, 7.2))
for ax, mis, titolo in zip(assi, ["Spearman", "kappa"], ["Spearman tra i livelli", "kappa su affidabile sì/no"]):
    M = np.array([[1.0 if a == b else A.misure(ANNOTATORI[a][0], ANNOTATORI[b][0], ANNOTATORI[a][1], ANNOTATORI[b][1])[mis]
                   for b in nomi_a] for a in nomi_a])
    pd.DataFrame(M, index=nomi_a, columns=nomi_a).round(3).to_csv(OUT / f"matrice_{mis}.csv")
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(nomi_a)):
        for j in range(len(nomi_a)):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.6 else "black", fontweight="bold" if {nomi_a[i], nomi_a[j]} == {"NewsGuard", "MBFC"} else None)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 2, 2, fill=False, edgecolor=ARANCIO, lw=2))
    ax.set_xticks(range(len(nomi_a)), nomi_a, rotation=55, ha="right", fontsize=8); ax.set_yticks(range(len(nomi_a)), nomi_a, fontsize=8)
    ax.grid(False); ax.set_title(f"{titolo} · riquadro arancio: i due esperti", fontsize=10.5)
fig.suptitle("L'accordo tra tutti gli annotatori, a coppie", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "matrice_accordo.png", dpi=180, bbox_inches="tight"); plt.show()"""),
    md("""## 8. Tutte le 226 testate, solo NewsGuard: le misure classiche e quelle di accordo

Prima di confrontarli con gli esperti, i tre classificatori a 5 classi su **tutte le 226 testate** del tentativo 16,
contro NewsGuard, con quattro misure: le due classiche (accuratezza su affidabile sì/no, F1 sui 5 livelli) e le due di
accordo (kappa su affidabile sì/no, Spearman sui livelli). Ognuna in due modi: **articolo singolo** (ogni articolo contro
l'etichetta della sua testata) e **testata** (il voto di maggioranza dei suoi articoli). Intervalli al 95%: bootstrap
sulle testate. Linee tratteggiate: chi risponde sempre allo stesso modo (accuratezza: sempre «affidabile»; F1: a caso;
kappa e Spearman: 0). Linea grigia nelle misure di accordo: i due esperti tra loro, sulle 170 testate in comune."""),
    code("""SCELTI = ["TF-IDF", "Qwen3-Embedding 8B", "Mistral-Small 24B"]
TUTTE, ET = t.index, t.label.to_numpy()
rng = np.random.default_rng(2)
idx = [rng.integers(0, len(TUTTE), len(TUTTE)) for _ in range(1000)]
righe = []
for m in SCELTI:
    g = A.gruppi(A.livelli_articoli(MODELLI[m], t).set_index("domain")["classificatore"], TUTTE)
    voto_lv, voto_aff = A.giudizio(g)
    art = lambda ii: (np.concatenate([g[i] for i in ii]), np.concatenate([np.full(len(g[i]), ET[i]) for i in ii]))
    for unita, calcola_misure in [("articolo singolo", lambda ii: A.quattro_misure(*art(ii), A.SOGLIA["NewsGuard"])),
                                  ("testata (voto)", lambda ii: A.quattro_misure(voto_lv[ii], ET[ii], A.SOGLIA["NewsGuard"], voto_aff[ii]))]:
        v = calcola_misure(np.arange(len(TUTTE))); boot = pd.DataFrame([calcola_misure(ii) for ii in idx])
        for mis in v:
            righe.append({"modello": m, "unità": unita, "misura": mis, "valore": v[mis],
                          "lo": np.percentile(boot[mis], 2.5), "hi": np.percentile(boot[mis], 97.5)})
classiche = pd.DataFrame(righe); classiche.to_csv(OUT / "tutte_226_quattro_misure.csv", index=False)

MISURE = [("accuratezza", "accuratezza su affidabile sì/no", (t.label >= 2).mean(), "sempre «affidabile»"),
          ("F1", "F1 sui 5 livelli", 0.2, "a caso"), ("kappa", "kappa su affidabile sì/no", 0.0, None),
          ("Spearman", "Spearman sui 5 livelli", 0.0, None)]
fig, assi = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (mis, titolo, rif, nome_rif) in zip(assi, MISURE):
    d = classiche[classiche.misura == mis]
    for i, m in enumerate(SCELTI):
        for dx, unita, colore in [(-0.19, "articolo singolo", "#9cc4f0"), (0.19, "testata (voto)", BLU)]:
            q = d[(d.modello == m) & (d["unità"] == unita)].iloc[0]
            ax.bar(i + dx, q.valore, width=0.36, color=colore, label=unita if (mis, i) == ("accuratezza", 0) else None)
            ax.errorbar(i + dx, q.valore, yerr=[[q.valore - q.lo], [q.hi - q.valore]], color=GRIGIO, lw=1, capsize=2.5)
            ax.text(i + dx, q.hi + 0.015, f"{q.valore:.2f}".replace(".", ","), ha="center", fontsize=8, color=GRIGIO, fontweight="bold")
    if nome_rif:
        ax.axhline(rif, color=GRIGIO, ls=":", lw=1); ax.text(2.5, rif + 0.01, f"{nome_rif}: {rif:.2f}".replace(".", ","), ha="right", fontsize=7.5, color=GRIGIO)
    if mis in ("kappa", "Spearman"):
        e = valori[("esperti", "", "NewsGuard ↔ MBFC")][mis]
        ax.axhline(e, color=GRIGIO, ls="--", lw=1.2)
        ax.text(-0.45, e + 0.012, f"i due esperti tra loro (170 testate): {e:.2f}".replace(".", ","), fontsize=7.5, color=GRIGIO)
    ax.set_xticks(range(len(SCELTI)), [m.replace(" ", "\\n", 1) for m in SCELTI], fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.grid(axis="x", visible=False); ax.set_title(titolo, fontsize=10)
fig.legend(loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f"Classificatori a 5 classi · tutte le {len(TUTTE)} testate · contro NewsGuard · misure classiche e di accordo · linee: intervalli al 95%",
             fontsize=12, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "tutte_226_quattro_misure.png", dpi=180, bbox_inches="tight"); plt.show()
classiche.pivot_table(index="modello", columns=["misura", "unità"], values="valore").round(3)"""),
    md("""## 9. Articolo singolo o testata intera: accuratezza e F1

Le misure classiche (accuratezza su affidabile sì/no, F1 sui 5 livelli) per i tre classificatori a 5 classi, in due modi:
- **articolo singolo**: ogni articolo, da solo, contro l'etichetta della sua testata (le testate con più articoli pesano di più);
- **testata**: il livello votato dalla maggioranza dei suoi articoli, una volta per testata.

E su tre insiemi: **tutte le 226 testate** contro NewsGuard; **le 170 valutate anche da MBFC** contro NewsGuard; le stesse
170 **contro MBFC**. Gli intervalli al 95% vengono dal bootstrap sulle testate (anche per gli articoli, che dentro una
testata non sono indipendenti). Linee tratteggiate: accuratezza di chi dice sempre "affidabile" (testate); F1 a caso: 0,20."""),
    code("""SCELTI = ["TF-IDF", "Qwen3-Embedding 8B", "Mistral-Small 24B"]
TUTTE = t.index
INSIEMI = {"tutte le 226 testate\\ncontro NewsGuard": (TUTTE, t.label.to_numpy(), A.SOGLIA["NewsGuard"]),
           "le 170 con MBFC\\ncontro NewsGuard": (TESTATE, NG_LV, A.SOGLIA["NewsGuard"]),
           "le 170 con MBFC\\ncontro MBFC": (TESTATE, MB_LV, A.SOGLIA["MBFC"])}
rng = np.random.default_rng(1)
righe = []
for m in SCELTI:
    lv_art = A.livelli_articoli(MODELLI[m], t).set_index("domain")["classificatore"]
    for nome_ins, (testate_ins, etichetta, soglia) in INSIEMI.items():
        g = A.gruppi(lv_art, testate_ins)
        C_art = A.conteggi(g)
        voto_lv, voto_aff = A.giudizio(g)
        C_tes = np.eye(5, dtype=int)[voto_lv]
        idx = [rng.integers(0, len(g), len(g)) for _ in range(1000)]
        for unita, C, aff in [("articolo singolo", C_art, None), ("testata (voto)", C_tes, voto_aff)]:
            acc, f1 = A.accuratezza_e_f1(C, etichetta, soglia, aff)
            boot = np.array([A.accuratezza_e_f1(C[i], etichetta[i], soglia, None if aff is None else aff[i]) for i in idx])
            righe.append({"modello": m, "insieme": nome_ins, "unità": unita, "accuratezza": acc, "F1": f1,
                          "accuratezza lo": np.percentile(boot[:, 0], 2.5), "accuratezza hi": np.percentile(boot[:, 0], 97.5),
                          "F1 lo": np.percentile(boot[:, 1], 2.5), "F1 hi": np.percentile(boot[:, 1], 97.5)})
singoli = pd.DataFrame(righe); singoli.to_csv(OUT / "articolo_o_testata.csv", index=False)

fig, assi = plt.subplots(2, 3, figsize=(15, 7.5), sharey="row")
CHIARO_BLU = "#9cc4f0"
for c, (nome_ins, (testate_ins, etichetta, soglia)) in enumerate(INSIEMI.items()):
    for r, (mis, titolo) in enumerate([("accuratezza", "accuratezza su affidabile sì/no"), ("F1", "F1 sui 5 livelli")]):
        ax = assi[r, c]; d = singoli[singoli.insieme == nome_ins]
        for i, m in enumerate(SCELTI):
            for dx, unita, colore in [(-0.19, "articolo singolo", CHIARO_BLU), (0.19, "testata (voto)", BLU)]:
                q = d[(d.modello == m) & (d["unità"] == unita)].iloc[0]
                ax.bar(i + dx, q[mis], width=0.36, color=colore, label=unita if (r, c, i) == (0, 0, 0) else None)
                ax.errorbar(i + dx, q[mis], yerr=[[q[mis] - q[f"{mis} lo"]], [q[f"{mis} hi"] - q[mis]]], color=GRIGIO, lw=1, capsize=2.5)
                ax.text(i + dx, q[f"{mis} hi"] + 0.015, f"{q[mis]:.2f}".replace(".", ","), ha="center", fontsize=8, color=GRIGIO, fontweight="bold")
        rif = np.mean(etichetta >= soglia) if mis == "accuratezza" else 0.2
        ax.axhline(rif, color=GRIGIO, ls=":", lw=1)
        ax.text(2.45, rif + 0.01, ("sempre «affidabile»: " if mis == "accuratezza" else "a caso: ") + f"{rif:.2f}".replace(".", ","),
                ha="right", fontsize=7.5, color=GRIGIO)
        ax.set_xticks(range(len(SCELTI)), [m.replace(" ", "\\n", 1) for m in SCELTI], fontsize=8.5)
        ax.set_ylim(0, 1.05 if mis == "accuratezza" else 0.75); ax.grid(axis="x", visible=False)
        ax.set_title(f"{titolo}\\n{nome_ins}", fontsize=9.5)
fig.legend(loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Classificatori a 5 classi: un articolo da solo o tutta la testata (voto di maggioranza) · linee: intervalli al 95%",
             fontsize=12.5, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "articolo_o_testata.png", dpi=180, bbox_inches="tight"); plt.show()
singoli.pivot_table(index=["insieme", "modello"], columns="unità", values=["accuratezza", "F1"]).round(3)"""),
    md("""## 10. Che cosa viene fuori

**I due esperti** sono d'accordo fino a un certo punto: Spearman 0,70 [0,61; 0,77], kappa 0,58 [0,47; 0,70], stesso
giudizio affidabile sì/no su 135 testate su 170 (79%). È il tetto naturale: due valutazioni umane di qualità.

**I modelli grandi congelati sono annotatori al livello degli esperti rispetto a NewsGuard.**
- Qwen3-Embedding 8B e Mistral-Small 24B, leggendo solo gli articoli, vanno d'accordo con NewsGuard **quanto MBFC**:
  Spearman 0,64–0,71 contro 0,70, kappa 0,59–0,64 contro 0,58; le differenze dall'accordo tra esperti stanno tra −0,06
  e +0,06 e i loro intervalli contengono lo zero.
- **Con MBFC, mai usato per addestrare**, l'accordo è un po' più basso di quello di NewsGuard (Spearman 0,60–0,61 contro
  0,70; kappa 0,47–0,54 contro 0,58), ma resta lontano dal caso; per il regressore di Mistral la kappa (0,54) sta dentro
  l'intervallo degli esperti. È il confronto più severo: il modello ha imparato le etichette di NewsGuard.
- TF-IDF e ModernBERT restano chiaramente sotto (da −0,14 a −0,30 dagli esperti).

**Il DNA è negli articoli, e si somma.** Un solo articolo dà già un accordo con NewsGuard ben sopra il caso (Spearman
0,42–0,60, kappa 0,34–0,49 per i modelli grandi); il voto di più articoli lo fa crescere, e con circa 9–25 articoli i
modelli grandi arrivano al livello dei due esperti.

**Gli errori stanno dove anche gli esperti litigano.** Sulle 135 testate su cui NewsGuard e MBFC concordano, i modelli
grandi coincidono con NewsGuard nell'85–88% dei casi; sulle 35 in disaccordo, nel 66–77%.

**Controllo.** Con la credibilità MBFC al posto della fattualità il quadro non cambia (esperti: kappa 0,49; regressore di
Mistral: 0,49).

**Limiti.** Le testate sono 170 (quelle con MBFC, il 75% delle 226), quindi gli intervalli sono larghi circa ±0,1; il
modello è addestrato su NewsGuard (su testate diverse), per questo il confronto indipendente è quello con MBFC."""),
], "01_accordo_annotatori.ipynb")

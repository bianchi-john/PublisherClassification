"""Costruisce i notebook del tentativo 18 (il punteggio esatto). Se il codice di un notebook non cambia, il notebook resta
com'è, con i suoi output; se cambia, si riscrive senza output e va rieseguito."""
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

SETUP = '''import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T16 = Path(T16_DIR).expanduser(); sys.path.insert(0, str(T16))
import funzioni as F16                      # il tentativo 16: dataset e cartella dei dati
import punteggio as P

warnings.filterwarnings("ignore")
plt.rcParams.update(F16.STILE_GRAFICI)
BLU, ARANCIO, GRIGIO, CHIARO, VERDE, ROSSO = "#2a78d6", "#eb6834", "#52514e", "#b5b3ab", "#1baf7a", "#b02a2a"
DATI18 = F16.DATI / "tentativi" / "18_PunteggioEsatto"; DATI18.mkdir(parents=True, exist_ok=True)
OUT = Path("Results"); OUT.mkdir(exist_ok=True)
art = pd.read_csv(F16.DATASET / "articoli.csv.gz")
t = pd.read_csv(F16.DATASET / "testate.csv", index_col="domain")
print(f"{len(art):,} articoli di {len(t)} testate")'''

# ------------------------------------------------------------------------------------------------ 1. split
notebook([
    md("""# Tentativo 18 · 1. Gli split sul punteggio

**L'obiettivo del tentativo.** Prevedere il **punteggio esatto NewsGuard (0–100)** di una testata dai suoi articoli,
senza classi in nessun punto: in addestramento e in valutazione conta solo il numero.

**Il problema degli split del tentativo 16.** I fold erano stratificati sui 5 livelli: ogni fold ha la stessa quota di
testate per livello, ma il valore **dentro** il livello è lasciato al caso (le testate l1 del fold 2 hanno in media 31
punti, quelle del fold 5 ne hanno 12). In più l'addestramento è per articolo e le testate hanno da 10 a 201 articoli: le
testate che scrivono molto decidono che cosa vede il modello.

**Qui.** Le testate si ordinano per punteggio e, a blocchi di 5 consecutive, ognuna va in un fold diverso: ogni fold ha
la stessa distribuzione del punteggio, anche dentro i livelli. Tre split con semi diversi, per sapere quanto i risultati
dipendono dallo split."""),
    code('''T16_DIR = "../16_DataPreparation"
SEMI = "1,2,3"'''),
    code(SETUP),
    code('''SEMI = [int(s) for s in str(SEMI).split(",")]
split = pd.DataFrame({f"split{s}": P.crea_split(t.score, s) for s in SEMI})
split.to_csv(DATI18 / "split.csv")
print(f"salvati in {DATI18 / 'split.csv'}")

from scipy.stats import ks_2samp
def equilibrio(fold):
    righe = []
    for k in range(1, 6):
        te, tr = t[fold == k], t[fold != k]
        righe.append({"fold": k, "punteggio medio": te.score.mean(), "media delle l1": te.score[te.label == 0].mean(),
                      "media delle l4": te.score[te.label == 3].mean(), "media per articolo": np.repeat(te.score, te.articoli).mean(),
                      "KS testate (p)": ks_2samp(te.score, tr.score).pvalue})
    return pd.DataFrame(righe).set_index("fold")
display(pd.concat({"split del tentativo 16 (per livello)": equilibrio(t.fold), "nuovo split 1 (per punteggio)": equilibrio(split.split1)}).round(2))'''),
    md("""## Come sono distribuiti punteggi e articoli
A sinistra il punteggio delle 226 testate; al centro gli articoli per testata; a destra il punteggio delle testate di
ogni fold, nei vecchi split e nei nuovi."""),
    code('''fig, assi = plt.subplots(1, 3, figsize=(16, 3.8))
assi[0].hist(t.score, bins=np.arange(0, 105, 5), color=BLU); assi[0].set_title("punteggio NewsGuard delle 226 testate"); assi[0].set_xlabel("punteggio")
assi[1].hist(t.articoli, bins=20, color=ARANCIO); assi[1].set_title("articoli per testata (da 10 a 201)"); assi[1].set_xlabel("articoli")
for dx, (nome, f) in zip([-0.18, 0.18], [("vecchi (per livello)", t.fold), ("nuovi (per punteggio)", split.split1)]):
    b = assi[2].boxplot([t.score[f == k] for k in range(1, 6)], positions=np.arange(1, 6) + dx, widths=0.3, patch_artist=True)
    for p_ in b["boxes"]: p_.set_facecolor(CHIARO if "vecchi" in nome else BLU)
assi[2].set_xticks(range(1, 6), [f"fold {k}" for k in range(1, 6)]); assi[2].set_title("punteggio per fold: vecchi (grigio) e nuovi (blu)")
plt.tight_layout(); plt.savefig(OUT / "split.png", dpi=170, bbox_inches="tight"); plt.show()'''),
], "01_split.ipynb")

# ------------------------------------------------------------------------------------------------ 2. modelli lineari
notebook([
    md("""# Tentativo 18 · 2. TF-IDF e Mistral 24B: regressori e classificatori, cinque condizioni

Due lettori del testo:
- **TF-IDF**: parole e coppie di parole (100.000), vocabolario imparato sui soli testi di training;
- **Mistral-Small 24B congelato**: il vettore di 5.120 numeri di ogni articolo (strato 27, calcolato nel tentativo 16).

Sopra, due modelli lineari:
- **regressore**: ridge sul punteggio 0–100;
- **classificatore**: logistica sui 5 livelli, trasformata in punteggio (la media dei punteggi di ogni livello nel training,
  pesata con le probabilità): serve a vedere se prevedere il numero serve davvero.

Cinque **condizioni dei dati** (cambiano solo l'addestramento; il test sono sempre tutti gli articoli delle testate di test):

| | condizione |
|---|---|
| A | tutti gli articoli (chi scrive di più pesa di più) |
| B | tutti gli articoli, **ogni testata pesa uguale** (peso 1 / articoli della testata) |
| C | al più **10 articoli** per testata, presi a caso |
| D | al più **30 articoli** per testata |
| E | **un vettore per testata** (la media dei vettori dei suoi articoli) |

La penalità (alpha della ridge, C della logistica) si sceglie con una validazione interna sulle testate di training
(3 fold sul punteggio, errore medio assoluto per testata). Tre split, cinque fold: ogni testata è prevista da un modello
che non l'ha mai vista. Circa un'ora sul server (solo CPU)."""),
    code('''T16_DIR = "../16_DataPreparation"
N_JOBS = 32
MODELLI = "TF-IDF,Mistral 24B"
MISTRAL = "Results/embedding/Mistral-Small-24B-Base-2501_strato27"     # nel tentativo 16'''),
    code(SETUP),
    code('''from joblib import Parallel, delayed
split = pd.read_csv(DATI18 / "split.csv", index_col=0).reindex(t.index)
X = np.load(T16 / MISTRAL / "embedding.npy")
assert np.array_equal(np.load(T16 / MISTRAL / "embedding_article_id.npy"), art.article_id.to_numpy())
art["score"] = t.score.reindex(art.domain).to_numpy()
LAVORI = OUT / "lineari" / "lavori"; LAVORI.mkdir(parents=True, exist_ok=True)
lavori = [(m, s, k, c, tipo) for m in MODELLI.split(",") for s in split.columns for k in range(1, 6) for c in P.CONDIZIONI
          for tipo in ["regressore", "classificatore"]]
nome = lambda m, s, k, c, tipo: LAVORI / f"{m.replace(' ', '')}_{s}_f{k}_{c}_{tipo}.csv.gz"
mancano = [l for l in lavori if not nome(*l).exists()]
print(f"{len(lavori)} lavori, {len(mancano)} da fare")


def esegui(m, s, k, c, tipo):
    out, testate, info = P.un_lavoro(m, X if m != "TF-IDF" else None, art, split[s], k, c, tipo, int(s[-1]))
    out.assign(modello=m, split=s, condizione=c, tipo=tipo).to_csv(nome(m, s, k, c, tipo), index=False)
    if testate is not None:
        testate.assign(modello=m, split=s, condizione=c, tipo=tipo).to_csv(str(nome(m, s, k, c, tipo)).replace(".csv.gz", "_testate.csv.gz"), index=False)
    pd.DataFrame([info]).to_csv(str(nome(m, s, k, c, tipo)).replace(".csv.gz", "_info.csv"), index=False)
    return True

# prima i lavori di TF-IDF (vocabolario per fold, più lenti), poi Mistral
Parallel(n_jobs=N_JOBS, verbose=5)(delayed(esegui)(*l) for l in sorted(mancano, key=lambda l: l[0] != "TF-IDF"))
prev = pd.concat([pd.read_csv(nome(*l)) for l in lavori], ignore_index=True)
prev.to_csv(OUT / "lineari" / "previsioni_articoli.csv.gz", index=False)
tes = pd.concat([pd.read_csv(str(nome(*l)).replace(".csv.gz", "_testate.csv.gz")) for l in lavori if l[3] == "E"], ignore_index=True)
tes.to_csv(OUT / "lineari" / "previsioni_testate_E.csv.gz", index=False)
scelte = pd.concat([pd.read_csv(str(nome(*l)).replace(".csv.gz", "_info.csv")) for l in lavori], ignore_index=True)
scelte.to_csv(OUT / "lineari" / "penalita_scelte.csv", index=False)
print(f"{len(prev):,} previsioni per articolo")'''),
    md("## La penalità scelta\nQuante volte ogni valore della griglia è stato scelto, per modello, tipo e condizione."),
    code('''scelte.groupby(["modello", "tipo", "condizione"]).penalità.value_counts().unstack(fill_value=0)'''),
], "02_lineari.ipynb")

# ------------------------------------------------------------------------------------------------ 3. BERT
notebook([
    md("""# Tentativo 18 · 3. Un BERT addestrato come regressore

Il transformer si addestra tutto (tutti i pesi) a prevedere il punteggio/100, con l'errore quadratico pesato secondo la
condizione (A, B, C, D come nel notebook 2), 5 epoche, learning rate 3e-5, lotti di 32, nessuna validazione (si tiene
l'ultima epoca). Il test sono tutti gli articoli delle testate del fold. Si lancia una volta per modello e condizione,
sullo split 1:
- **ModernBERT-base** (150 M di parametri, articoli interi fino a 2.048 token);
- **DeBERTa-v3-base** (184 M di parametri, i primi 512 token: un'architettura diversa, molto forte nella regressione)."""),
    code('''T16_DIR = "../16_DataPreparation"
MODELLO = "answerdotai/ModernBERT-base"
NOME = "ModernBERT"
CONDIZIONE = "B"
SPLIT = 1
MAX_TOKEN = 2048
EPOCHE = 5
LR = 3e-5
GB_GPU = 14
PROVA = 0                                   # > 0: solo quel numero di articoli per testata e 1 epoca (prova veloce)'''),
    code(SETUP),
    code('''import time
split = pd.read_csv(DATI18 / "split.csv", index_col=0).reindex(t.index)
fold = split[f"split{SPLIT}"]
art["score"] = t.score.reindex(art.domain).to_numpy()
cartella = OUT / ("prova" if PROVA else "") / "bert" / f"{NOME}_{CONDIZIONE}"; cartella.mkdir(parents=True, exist_ok=True)
if PROVA:
    art = art.groupby("domain").head(PROVA).reset_index(drop=True); EPOCHE = 1
log = open(OUT / "bert_avanzamento.log", "a")
def registra(m):
    riga = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {NOME} {CONDIZIONE}: {m}"; print(riga, flush=True); log.write(riga + "\\n"); log.flush()
pezzi = [P.bert_fold(art, fold, k, CONDIZIONE, MODELLO, cartella, registra, MAX_TOKEN, epoche=EPOCHE, lr=LR, gb_gpu=GB_GPU) for k in range(1, 6)]
prev = pd.concat(pezzi, ignore_index=True).assign(modello=NOME, split=f"split{SPLIT}", condizione=CONDIZIONE, tipo="regressore")
prev.to_csv(cartella / "previsioni.csv.gz", index=False)
print(f"{len(prev):,} articoli previsti")'''),
    code('''curve = pd.concat([pd.read_csv(cartella / f"curva_fold{k}.csv").assign(fold=k) for k in range(1, 6)])
for k, c in curve.groupby("fold"):
    plt.plot(c.epoca, np.sqrt(c.loss) * 100, marker="o", label=f"fold {k}")
plt.title(f"{NOME}, condizione {CONDIZIONE}: errore sul training (RMSE in punti) dopo ogni epoca"); plt.xlabel("epoca"); plt.legend(fontsize=7)
plt.savefig(cartella / "curve.png", dpi=150, bbox_inches="tight"); plt.show()'''),
], "03_bert.ipynb")

# ------------------------------------------------------------------------------------------------ 4. confronto
notebook([
    md("""# Tentativo 18 · 4. Il confronto: quanto bene si prevede il punteggio esatto?

Tutte le configurazioni (modello, regressore o classificatore, condizione dei dati) si valutano sul punteggio vero,
senza classi:
- **MAE**: di quanti punti si sbaglia in media; **RMSE**: come il MAE ma pesa di più gli errori grandi;
- **R²**: quanta parte della variabilità dei punteggi si spiega (0 = come dire sempre la media, 1 = perfetto);
- **Spearman**: se l'ordine delle testate è giusto; **Pearson**: se il legame è lineare.

Ogni misura per **articolo singolo** (ogni articolo contro il punteggio della sua testata) e per **testata** (la media dei
suoi articoli; per la condizione E il vettore medio della testata). Riferimento: **la media del training** (chi prevede
sempre il punteggio medio delle testate di training). I modelli lineari hanno tre split: barre = media, linee = il più
basso e il più alto dei tre; i BERT hanno lo split 1."""),
    code('''T16_DIR = "../16_DataPreparation"
B = 1000                                    # ricampionamenti bootstrap (split 1)
R = 100                                     # ripetizioni per ogni numero di articoli nella curva del DNA'''),
    code(SETUP),
    code('''split = pd.read_csv(DATI18 / "split.csv", index_col=0).reindex(t.index)
vero = t.score
pa = [pd.read_csv(OUT / "lineari" / "previsioni_articoli.csv.gz")] if (OUT / "lineari" / "previsioni_articoli.csv.gz").exists() else []
pa += [pd.read_csv(f) for f in sorted((OUT / "bert").glob("*/previsioni.csv.gz"))]
pa = pd.concat(pa, ignore_index=True)
pa["vero"] = vero.reindex(pa.domain).to_numpy(); pa["previsione"] = pd.to_numeric(pa.previsione)
# il riferimento: la media dei punteggi delle testate di training, fold per fold
base = []
for s in split.columns:
    for k in range(1, 6):
        te = split.index[split[s] == k]; m = vero[split[s] != k].mean()
        a = art[art.domain.isin(te)]
        base.append(pd.DataFrame({"article_id": a.article_id, "domain": a.domain, "previsione": m, "modello": "media del training",
                                  "split": s, "condizione": "-", "tipo": "-", "vero": vero.reindex(a.domain).to_numpy()}))
pa = pd.concat([pa] + base, ignore_index=True)
CHIAVI = ["modello", "tipo", "condizione", "split"]
pt = pa.groupby(CHIAVI + ["domain"], as_index=False).previsione.mean()
fE = OUT / "lineari" / "previsioni_testate_E.csv.gz"
if fE.exists():                                  # condizione E: la previsione della testata viene dal suo vettore medio
    e = pd.read_csv(fE); pt = pt.merge(e, on=CHIAVI + ["domain"], how="left", suffixes=("", "_E"))
    pt["previsione"] = pt.previsione_E.fillna(pt.previsione); pt = pt.drop(columns="previsione_E")
pt["vero"] = vero.reindex(pt.domain).to_numpy()
mis = []
for unita, d in [("articolo", pa), ("testata", pt)]:
    for chiave, g in d.groupby(CHIAVI):
        mis.append({**dict(zip(CHIAVI, chiave)), "unità": unita, **P.misure(g.vero, g.previsione)})
mis = pd.DataFrame(mis); mis.to_csv(OUT / "misure_per_split.csv", index=False)
sintesi = mis.drop(columns="split").groupby(["modello", "tipo", "condizione", "unità"]).agg(["mean", "min", "max"])
print(f"{pa[CHIAVI].drop_duplicates().shape[0]} configurazioni × split")'''),
    md("""## 1. Il quadro completo
Una riga per configurazione. **MAE** (più basso è meglio) e **Spearman** (più alto è meglio), per testata e per articolo.
Linea tratteggiata: la media del training."""),
    code('''COLORE = {"TF-IDF": GRIGIO, "Mistral 24B": ROSSO, "ModernBERT": "#9b59b6", "DeBERTa": VERDE, "media del training": CHIARO}
righe_c = sintesi.xs("testata", level="unità").sort_index().index.tolist()
ordine = sorted(righe_c, key=lambda r: (list(COLORE).index(r[0]) if r[0] in COLORE else 9, r[1], r[2]))
fig, assi = plt.subplots(1, 4, figsize=(19, 0.34 * len(ordine) + 1.8), sharey=True)
y = np.arange(len(ordine))[::-1]
for ax, (unita, m_) in zip(assi, [("testata", "MAE"), ("testata", "Spearman"), ("articolo", "MAE"), ("articolo", "Spearman")]):
    for yy, r in zip(y, ordine):
        v = sintesi.loc[(*r, unita), m_]
        ax.barh(yy, v["mean"], color=COLORE.get(r[0], BLU), height=0.7, alpha=0.55 if r[1] == "classificatore" else 1)
        ax.errorbar(v["mean"], yy, xerr=[[v["mean"] - v["min"]], [v["max"] - v["mean"]]], color="k", lw=0.8, capsize=2)
        ax.text(v["max"] + (0.3 if m_ == "MAE" else 0.01), yy, f"{v['mean']:.2f}" if m_ != "MAE" else f"{v['mean']:.1f}", va="center", fontsize=7)
    b = sintesi.loc[("media del training", "-", "-", unita), m_]["mean"]
    if m_ == "MAE": ax.axvline(b, color=GRIGIO, ls="--", lw=1)
    ax.set_title(f"{m_} · {unita}" + (" (punti, più basso è meglio)" if m_ == "MAE" else ""), fontsize=10); ax.grid(axis="y", visible=False)
assi[0].set_yticks(y, [f"{m} · {tipo} · {c}" if m != "media del training" else m for m, tipo, c in ordine], fontsize=8)
fig.suptitle("Il punteggio esatto: tutte le configurazioni (chiaro = classificatore trasformato in punteggio) · linee: min–max sui 3 split",
             fontsize=12, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(OUT / "quadro.png", dpi=170, bbox_inches="tight"); plt.show()
sintesi.xs("testata", level="unità")[["MAE", "RMSE", "R²", "Spearman"]].xs("mean", axis=1, level=1).round(3)'''),
    md("""## 2. Regressore o classificatore?
Per TF-IDF e Mistral, nelle cinque condizioni: l'errore medio per testata del regressore (pieno) e del classificatore
trasformato in punteggio (vuoto). Se il regressore non è sotto il classificatore, prevedere il numero non aggiunge nulla."""),
    code('''fig, assi = plt.subplots(1, 2, figsize=(14, 4))
for ax, m_ in zip(assi, ["MAE", "Spearman"]):
    for i, mod in enumerate(["TF-IDF", "Mistral 24B"]):
        for tipo, stile in [("regressore", "o-"), ("classificatore", "s--")]:
            try:
                v = [sintesi.loc[(mod, tipo, c, "testata"), (m_, "mean")] for c in P.CONDIZIONI]
            except KeyError:
                continue
            ax.plot(list(P.CONDIZIONI), v, stile, color=COLORE[mod], label=f"{mod}, {tipo}", mfc="white" if tipo == "classificatore" else COLORE[mod])
    if m_ == "MAE": ax.axhline(sintesi.loc[("media del training", "-", "-", "testata"), ("MAE", "mean")], color=GRIGIO, ls="--", lw=1, label="media del training")
    ax.set_title(f"{m_} per testata nelle cinque condizioni (media dei 3 split)"); ax.set_xlabel("condizione"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "regressore_o_classificatore.png", dpi=170, bbox_inches="tight"); plt.show()
print({c: d for c, d in P.CONDIZIONI.items()})'''),
    md("""## 3. Le migliori configurazioni, con gli intervalli al 95%
Per ogni modello la configurazione con l'errore medio per testata più basso (split 1), con gli intervalli bootstrap sulle
testate, e la differenza dalla media del training."""),
    code('''s1 = mis[(mis.split == "split1") & (mis["unità"] == "testata")]
migliori = s1[s1.modello != "media del training"].sort_values("MAE").groupby("modello").head(1)
scelte = [tuple(r) for r in migliori[["modello", "tipo", "condizione"]].to_numpy()] + [("media del training", "-", "-")]
rng = np.random.default_rng(0)
dominii = t.index[split.split1.notna()]
campioni = [rng.choice(len(t), len(t)) for _ in range(B)]
righe = []
for cfg in scelte:
    g = pt[(pt.modello == cfg[0]) & (pt.tipo == cfg[1]) & (pt.condizione == cfg[2]) & (pt.split == "split1")].set_index("domain").reindex(t.index)
    v = P.misure(g.vero, g.previsione); bb = pd.DataFrame([P.misure(g.vero.iloc[i], g.previsione.iloc[i]) for i in campioni])
    righe.append({"configurazione": " · ".join(cfg), **{k_: f"{v[k_]:.2f} [{np.percentile(bb[k_], 2.5):.2f}; {np.percentile(bb[k_], 97.5):.2f}]" for k_ in v}})
pd.DataFrame(righe).set_index("configurazione")'''),
    md("""## 4. Dove si sbaglia: l'errore per fascia del punteggio vero
Per le migliori configurazioni: l'errore medio con il segno (previsto − vero: sopra lo zero = sovrastima) e l'errore
medio assoluto, nelle fasce del punteggio vero. Un regressore che si schiaccia verso la media sovrastima le testate
peggiori e sottostima le migliori."""),
    code('''FASCE = pd.cut(t.score, [-1, 39, 59, 74, 99, 100], labels=["0–39", "40–59", "60–74", "75–99", "100"])
fig, assi = plt.subplots(1, 3, figsize=(17, 4))
for cfg in scelte:
    g = pt[(pt.modello == cfg[0]) & (pt.tipo == cfg[1]) & (pt.condizione == cfg[2]) & (pt.split == "split1")].set_index("domain").reindex(t.index)
    e = g.previsione - g.vero
    assi[0].plot(FASCE.cat.categories, e.groupby(FASCE).mean(), "o-", color=COLORE.get(cfg[0], BLU), label=" · ".join(cfg))
    assi[1].plot(FASCE.cat.categories, e.abs().groupby(FASCE).mean(), "o-", color=COLORE.get(cfg[0], BLU))
    if cfg[0] != "media del training":
        assi[2].scatter(g.vero, g.previsione, s=10, alpha=0.6, color=COLORE.get(cfg[0], BLU))
assi[0].axhline(0, color=GRIGIO, lw=1); assi[0].set_title("errore con il segno (previsto − vero), per fascia"); assi[0].legend(fontsize=7)
assi[1].set_title("errore medio assoluto, per fascia")
assi[2].plot([0, 100], [0, 100], color=GRIGIO, lw=1); assi[2].set_title("previsto contro vero (testate, migliori configurazioni)")
assi[2].set_xlabel("punteggio vero"); assi[2].set_ylabel("punteggio previsto")
plt.tight_layout(); plt.savefig(OUT / "errore_per_fascia.png", dpi=170, bbox_inches="tight"); plt.show()
print("testate per fascia:", FASCE.value_counts().sort_index().to_dict())'''),
    md("""## 5. Il DNA in punti: quanti articoli servono?
Per le migliori configurazioni (split 1): la testata giudicata con la media dei punteggi di **k articoli presi a caso**
(1, 3, 5, 9, 15, 25 o tutti). Con k = 1 è il giudizio di un solo articolo. Ogni punto è la media di 100 estrazioni."""),
    code('''K = [1, 3, 5, 9, 15, 25, None]
curva = []
for cfg in scelte:
    if cfg[0] == "media del training" or cfg[2] == "E":
        continue
    g = pa[(pa.modello == cfg[0]) & (pa.tipo == cfg[1]) & (pa.condizione == cfg[2]) & (pa.split == "split1")]
    gruppi = [v.to_numpy() for _, v in g.groupby("domain").previsione]
    veri = g.groupby("domain").vero.first().to_numpy()
    for k in K:
        for _ in range(R if k else 1):
            p_ = np.array([v.mean() if not k or k >= len(v) else rng.choice(v, k, replace=False).mean() for v in gruppi])
            curva.append({"configurazione": " · ".join(cfg), "k": k or 999, **P.misure(veri, p_)})
curva = pd.DataFrame(curva).groupby(["configurazione", "k"]).mean().reset_index(); curva.to_csv(OUT / "curva_articoli.csv", index=False)
fig, assi = plt.subplots(1, 2, figsize=(13, 4))
for c, d in curva.groupby("configurazione"):
    for ax, m_ in zip(assi, ["MAE", "Spearman"]):
        ax.plot(d.k, d[m_], "o-", color=COLORE.get(c.split(" · ")[0], BLU), label=c)
for ax, m_ in zip(assi, ["MAE", "Spearman"]):
    ax.set_xscale("log"); ax.set_xticks([1, 3, 5, 9, 15, 25, 999], ["1", "3", "5", "9", "15", "25", "tutti"]); ax.set_xlabel("articoli letti per testata")
    ax.set_title(f"{m_} per testata" + (" (punti)" if m_ == "MAE" else ""))
assi[0].axhline(sintesi.loc[("media del training", "-", "-", "testata"), ("MAE", "mean")], color=GRIGIO, ls="--", lw=1)
assi[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "dna_punti.png", dpi=170, bbox_inches="tight"); plt.show()'''),
    md("""## 6. Quanto conta lo split
Per i modelli lineari, l'errore medio per testata in ognuno dei 3 split: se le linee sono piatte, i risultati non
dipendono dallo split."""),
    code('''d = mis[(mis["unità"] == "testata") & mis.modello.isin(["TF-IDF", "Mistral 24B"])]
piv = d.pivot_table(index=["modello", "tipo", "condizione"], columns="split", values="MAE")
display(piv.round(2))
print("differenza media tra lo split migliore e il peggiore:", round((piv.max(axis=1) - piv.min(axis=1)).mean(), 2), "punti")'''),
], "04_confronto.ipynb")

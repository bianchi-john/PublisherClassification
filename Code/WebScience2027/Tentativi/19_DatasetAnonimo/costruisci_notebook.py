"""Costruisce i notebook del tentativo 19 (il dataset anonimo). Se il codice di un notebook non cambia, il notebook resta
com'è, con i suoi output; se cambia, si riscrive senza output e va rieseguito."""
from pathlib import Path

import nbformat as nbf

CARTELLA = Path(__file__).resolve().parent


def notebook(celle, nome, kernel="irgood-venv"):
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
    nb.metadata["kernelspec"] = {"name": kernel, "display_name": kernel, "language": "python"}
    nbf.write(nb, vecchio)
    print(nome, len(celle), "celle")


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

notebook([
    md("""# Tentativo 19 · 1. L'anonimizzazione

**Perché.** I modelli grandi (Qwen, Mistral) hanno letto il web: se nel testo resta il nome della testata, possono
riconoscerla e usare quello che già sanno della sua affidabilità, invece di leggere l'articolo. Togliamo dal testo
tutto quello che dice **chi** l'ha pubblicato, e niente di quello che dice **che cosa** racconta.

**Il punto di partenza.** Il dataset del tentativo 16 così com'è (226 testate, 25.524 articoli, stessa pulizia con
trafilatura, stessi gruppi). Cambia solo il testo (e il titolo).

**Le regole**, semplici e una per tipo di identificativo, nell'ordine:

| # | che cosa | come si trova | diventa |
|---|---|---|---|
| 1 | la **firma** della testata | la coda che la testata ripete in almeno il 30% dei suoi titoli («… \\| The New York Sun») | `[TESTATA]` |
| 2 | email | espressione regolare | `[EMAIL]` |
| 3 | link | http…, www…, domini come `bellingcat.com` | `[URL]` |
| 4 | account | `@nome` | `[ACCOUNT]` |
| 5 | **autori** della testata | il campo autore: nomi che firmano solo in questa testata; nomi di una parola solo dopo «by» | `[AUTORE]` |
| 6 | **nome** della testata | il dominio, scritto attaccato o staccato, con o senza «the» («The American Conservative», «100PercentFedUp»); un «online» in coda non fa parte del nome (spiked-online → spiked) | `[TESTATA]` |

Se il nome è anche una **parola comune** (politics, time, reason…) si toglie solo con l'iniziale maiuscola, così la
parola minuscola resta; tranne quando la testata scrive così il proprio nome, cioè la parola minuscola sta in almeno l'80%
dei suoi articoli (spiked: 100%; il più alto degli altri è time, 56%). Persone, luoghi e fatti di cui parla l'articolo non si toccano. I controlli sono automatici
(sezione 4)."""),
    code('''import re

import pandas as pd

import anonimizza as A

pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
OUT = A.Path("Results"); OUT.mkdir(exist_ok=True)
art = pd.read_csv(A.ORIGINE / "articoli.csv.gz")
t = pd.read_csv(A.ORIGINE / "testate.csv", index_col="domain")
print(f"{len(art):,} articoli di {art.domain.nunique()} testate, dal tentativo 16")'''),

    md("""## 1. Il nome di ogni testata e i nomi che sono parole comuni
Il nome viene dal dominio (`tldextract`, senza rete): la parte prima del suffisso; per le testate su Substack il
sottodominio; senza un «online» in coda. Un nome è una parola comune se compare minuscolo, come parola intera, più di 20 volte negli articoli delle
**altre** testate (fuori da link e account)."""),
    code('''nomi = {d: A.nomi_testata(d) for d in sorted(art.domain.unique())}
comuni = A.parole_comuni(nomi, art.text, art.domain)
minuscoli = A.nomi_minuscoli(nomi, comuni, art.text, art.domain)
tab = pd.DataFrame({"nomi": {d: " / ".join(v) for d, v in nomi.items()}}).rename_axis("domain")
tab["parola comune"] = [any((n in comuni or (n.startswith("the") and n[3:] in comuni)) and n not in minuscoli for n in v)
                        for v in nomi.values()]
tab.to_csv(OUT / "nomi.csv")
print(f"{tab['parola comune'].sum()} testate su {len(tab)} hanno un nome che è una parola comune:", ", ".join(sorted(comuni)))
print("parole comuni che una testata usa come proprio nome minuscolo:", ", ".join(sorted(minuscoli)) or "nessuna")
tab[tab["parola comune"]]'''),

    md("""## 2. Le firme nei titoli e gli autori
La **firma** trova i nomi che dal dominio non si ricavano (cfr.org si firma «Council on Foreign Relations»). Gli **autori**
sono quelli del campo autore che firmano in una sola testata e hanno l'aspetto di un nome di persona; chi firma in più
testate (agenzie, «Staff») non identifica la testata."""),
    code('''firme = A.firme_titoli(art)
autori = A.autori_personali(art)
f = pd.DataFrame({"firma": firme}).rename_axis("domain")
f["nel testo (articoli)"] = [art[art.domain == d].text.astype(str).str.contains(re.escape(x)).sum() for d, x in firme.items()]
f.to_csv(OUT / "firme.csv")
print(f"firme trovate: {len(firme)} testate; autori da togliere: {sum(map(len, autori.values())):,} in {len(autori)} testate")
f'''),

    md("## 3. L'anonimizzazione"),
    code('''anonimo, conti = A.anonimizza(art, nomi, comuni, minuscoli, autori, firme)
REGOLE = ["firma", "email", "url", "account", "autore", "testata"]
riepilogo = pd.DataFrame({
    "articoli toccati (testo, %)": [(conti[k] > 0).mean() * 100 for k in REGOLE],
    "sostituzioni nel testo": [conti[k].sum() for k in REGOLE],
    "titoli toccati (%)": [(conti[f"titolo_{k}"] > 0).mean() * 100 for k in REGOLE]}, index=REGOLE).round(1)
toccati = (conti[REGOLE].sum(axis=1) + conti[[f"titolo_{k}" for k in REGOLE]].sum(axis=1)) > 0
print(f"articoli con almeno una sostituzione (testo o titolo): {toccati.mean():.1%}")
conti.groupby("domain")[REGOLE].sum().to_csv(OUT / "sostituzioni_per_testata.csv")
riepilogo'''),

    md("""Qualche esempio, scelto a caso, di che cosa si è tolto (il testo intorno al segnaposto)."""),
    code('''for segno in ["[TESTATA]", "[AUTORE]", "[URL]", "[ACCOUNT]"]:
    print("=====", segno)
    for c in A.contesti(anonimo.text, re.compile(re.escape(segno)), quanti=5, larghezza=60, seme=1):
        print("  ·", c)'''),

    md("""## 4. I controlli automatici
1. **Il nome e la firma della propria testata** non devono restare come parola intera; si contano anche quelli rimasti
   **dentro** altre parole (refusi come «Quilletteessay», sigle come «AAFF»), con qualche esempio.
2. Per i **nomi che sono parole comuni** resta la parola minuscola, per scelta: si contano a parte."""),
    code('''res = A.residui(anonimo, nomi, comuni, minuscoli, firme)
res.to_csv(OUT / "residui.csv", index=False)
dist = res[~res.parola_comune]
print(f"nomi distintivi e firme ({len(dist)} testate): articoli con un residuo come parola intera {dist.residuo_come_parola.sum()}, "
      f"anche dentro altre parole {dist.residuo_anche_dentro_parole.sum()}")
for d in dist[dist.residuo_anche_dentro_parole > 0].domain:
    g = anonimo[anonimo.domain == d]
    cerca = re.compile("|".join([re.escape(n) for n in nomi[d]] + ([re.escape(firme[d])] if d in firme else [])), re.I)
    print(" ", d, "→", A.contesti(g.text + " " + g.titolo.astype(str), cerca, quanti=2, larghezza=35))
print("\\nnomi che sono parole comuni: resta la forma minuscola, per scelta")
res[res.parola_comune]'''),

    md("## 5. Il dataset anonimo"),
    code('''A.DATASET.mkdir(parents=True, exist_ok=True)
colonne = ["article_id", "domain", "titolo", "data", "text", "score", "label", "livello"]   # niente url né autore
anonimo[colonne].to_csv(A.DATASET / "articoli.csv.gz", index=False)
t.drop(columns=["fold"]).to_csv(A.DATASET / "testate.csv")
print(f"scritto {A.DATASET / 'articoli.csv.gz'}: {len(anonimo):,} articoli, {anonimo.domain.nunique()} testate")
print("parole per articolo (mediana): prima", int(art.text.str.split().str.len().median()),
      "dopo", int(anonimo.text.str.split().str.len().median()))'''),
], "01_anonimizzazione.ipynb")

notebook([
    md("""# Tentativo 19 · 2. Lo split

**Uno split, generato una volta sola qui, per i due compiti.** Le testate si ordinano per **voto esatto** (a parità di
voto, a caso con il seme `SEME_SPLIT` = 1) e, a blocchi di 5 consecutive, ognuna va in un fold diverso
(`modelli.crea_split`). Il risultato si salva in `split.csv`, e tutti i notebook successivi leggono solo quel file.
Con lo stesso seme è identico al primo split del tentativo 18, così i risultati restano confrontabili.

**Perché uno solo, per voto.** I livelli sono fasce di voto consecutive, quindi lo split è bilanciato anche per livello
(al più una testata di differenza tra i fold), e in più è bilanciato **dentro** i livelli, cosa che lo split per livello
del tentativo 16 non fa. Con le stesse testate negli stessi fold, classificatore e regressore si confrontano a coppie.
Una testata non è mai divisa tra fold; i gruppi di proprietà sono già risolti nel tentativo 16."""),
    code('''import pandas as pd

import anonimizza as A
import modelli as Mo

t = pd.read_csv(A.DATASET / "testate.csv", index_col="domain")
fold = Mo.crea_split(t.score, Mo.SEME_SPLIT)
fold.rename("fold").rename_axis("domain").to_csv(A.DATASET / "split.csv")
print(f"scritto split.csv: {len(t)} testate, 5 fold, seme {Mo.SEME_SPLIT}")
diciotto = pd.read_csv(A.DATI / "tentativi" / "18_PunteggioEsatto" / "split.csv", index_col=0).split1.reindex(t.index)
print("identico al primo split del tentativo 18:", bool((fold == diciotto).all()))'''),
    md("**Testate per livello in ogni fold**: al più una di differenza."),
    code('''pd.crosstab(fold.rename("fold"), t.livello)'''),
    md("""**Il voto in ogni fold**: stessa media e dispersione, e lo stesso voto medio dentro i livelli. Per confronto lo split
per livello del tentativo 16, dove il voto medio delle l1 andava da 12 a 31 a seconda del fold."""),
    code('''livello16 = pd.read_csv(A.ORIGINE / "testate.csv", index_col="domain").fold.reindex(t.index)
righe = []
for nome, f in [("per voto (questo)", fold), ("per livello (tentativo 16)", livello16)]:
    for k in range(1, 6):
        g = t[f == k]
        righe.append({"split": nome, "fold": k, "testate": len(g), "voto medio": g.score.mean(), "dev. std": g.score.std(),
                      "voto medio l1": g.score[g.livello == "l1"].mean(), "voto medio l4": g.score[g.livello == "l4"].mean()})
pd.DataFrame(righe).set_index(["split", "fold"]).round(1)'''),
], "02_split.ipynb")

# ------------------------------------------------------------------------------------------------ 3. token ed embedding (GPU)
REGISTRO = '''def registra(msg, file=OUT / "avanzamento.log"):
    riga = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(riga, flush=True)
    with open(file, "a") as f:
        f.write(riga + "\\n")'''

notebook([
    md("""# Tentativo 19 · 3. Lunghezze ed embedding

**Ogni modello legge ogni articolo intero.** Si contano i token di ogni articolo con il tokenizer di ogni modello: il
limite (`MAX_TOKEN` = 4.096, più l'istruzione per Qwen) deve essere sopra l'articolo più lungo, cioè **nessun articolo
troncato**. ModernBERT arriva a 8.192 token, Qwen3-Embedding a 32.768, Mistral-Small a 32.768.

Poi i vettori dei due modelli congelati, sugli articoli anonimi, come nel tentativo 16: **Qwen3-Embedding-8B** (ultimo
token, con l'istruzione sull'affidabilità) e **Mistral-Small-24B Base** in 4 bit (media dei token, strati 27 e 40 di 40;
i modelli lineari usano il 27, il migliore nel tentativo 16)."""),
    code('''MODELLO_BERT = "answerdotai/ModernBERT-base"
MODELLO_QWEN = "Qwen/Qwen3-Embedding-8B"
MODELLO_MISTRAL = "mistralai/Mistral-Small-24B-Base-2501"
STRATI = "27,40"
BUDGET_QWEN = 32768                 # token per lotto (lunghezza massima × articoli)
BUDGET_MISTRAL = 8192                # come il tentativo 16 (4 × 2.048): tiene in memoria tutti gli strati
GB_GPU = 22'''),
    code('''import time
import warnings

import numpy as np
import pandas as pd

import modelli as Mo

warnings.filterwarnings("ignore")
OUT = Mo.Path("Results"); OUT.mkdir(exist_ok=True)
Mo.EMBEDDING.mkdir(parents=True, exist_ok=True)
art = pd.read_csv(Mo.DATASET / "articoli.csv.gz")
testi = art.text.fillna("").tolist()
''' + REGISTRO + '''
print(f"{len(art):,} articoli"); print(Mo.ambiente())'''),
    md("## 1. Quanti token ha ogni articolo"),
    code('''righe = []
for nome, modello, prefisso in [("ModernBERT", MODELLO_BERT, ""), ("Qwen3 8B", MODELLO_QWEN, f"Instruct: {Mo.ISTRUZIONE_QWEN}\\nQuery:"),
                                ("Mistral 24B", MODELLO_MISTRAL, "")]:
    n = Mo.conta_token(testi, modello, prefisso)
    limite = Mo.MAX_TOKEN + (64 if prefisso else 0)
    righe.append({"modello": nome, "mediana": int(np.median(n)), "99%": int(np.percentile(n, 99)), "massimo": int(n.max()),
                  "limite": limite, "articoli troncati": int((n > limite).sum())})
lung = pd.DataFrame(righe).set_index("modello"); lung.to_csv(OUT / "lunghezze_token.csv")
assert lung["articoli troncati"].sum() == 0, "qualche articolo supera il limite: alzare MAX_TOKEN"
lung'''),
    md("## 2. Qwen3-Embedding-8B"),
    code('''file_qwen = Mo.EMBEDDING / "Qwen3-Embedding-8B.npy"
if not file_qwen.exists():
    Mo.aspetta_gpu(GB_GPU, registra)
    registra("Qwen3-Embedding-8B: inizio")
    E = Mo.embedding_qwen(testi, MODELLO_QWEN, BUDGET_QWEN, registra)
    np.save(file_qwen, E); registra(f"Qwen3-Embedding-8B: fatto, {E.shape}")
E = np.load(file_qwen); print(E.shape, "valori finiti:", np.isfinite(E).all())'''),
    md("## 3. Mistral-Small-24B Base"),
    code('''strati = [int(s) for s in str(STRATI).split(",")]
file_mistral = {s: Mo.EMBEDDING / f"Mistral-Small-24B_strato{s}.npy" for s in strati}
if not all(f.exists() for f in file_mistral.values()):
    Mo.aspetta_gpu(GB_GPU, registra)
    registra("Mistral-Small-24B: inizio")
    E = Mo.embedding_mistral(testi, MODELLO_MISTRAL, strati, BUDGET_MISTRAL, registra, Mo.EMBEDDING / "mistral_parziale.npy")
    for s in strati:
        np.save(file_mistral[s], E[s])
    (Mo.EMBEDDING / "mistral_parziale.npy").unlink(missing_ok=True)
    registra("Mistral-Small-24B: fatto")
for s, f in file_mistral.items():
    E = np.load(f); print(f"strato {s}: {E.shape}, valori finiti: {np.isfinite(E).all()}")
pd.Series(art.article_id).to_csv(Mo.EMBEDDING / "ordine_articoli.csv", index=False)     # le righe dei vettori'''),
], "03_embedding.ipynb", kernel="johnvenv")

# ------------------------------------------------------------------------------------------------ 4. modelli lineari (CPU)
notebook([
    md("""# Tentativo 19 · 4. I modelli lineari

Per i due compiti, con lo stesso split (per voto): **TF-IDF**, **Qwen3 8B** e **Mistral 24B** (strato 27) congelati, con sopra una
**logistica multinomiale** a classi bilanciate (classificazione, 5 livelli) o una **ridge** (regressione, voto 0–100).

**La penalità** si sceglie in ogni fold con 3 fold interni sulle sole testate di training (divise per voto, come lo
split esterno), con la misura per testata: F1 macro sul voto degli articoli (classificazione), Spearman tra voto vero e media
prevista (regressione). Le griglie hanno un ordine di grandezza in più del tentativo 18, dove la scelta cadeva sul bordo;
qui sotto si vede se ci cade ancora.

**Si salvano**, per ogni articolo di ogni fold, i logit (e le probabilità) o il voto previsto, e il modello di ogni fold
(preparazione + modello lineare)."""),
    code('''COMPITI = "classificazione,regressione"
NOMI = "TF-IDF,Qwen3 8B,Mistral 24B"
N_JOBS = 16'''),
    code('''import time
import warnings

import numpy as np
import pandas as pd

import modelli as Mo

warnings.filterwarnings("ignore")
OUT = Mo.Path("Results"); OUT.mkdir(exist_ok=True)
FILE_EMB = {"Qwen3 8B": "Qwen3-Embedding-8B.npy", "Mistral 24B": "Mistral-Small-24B_strato27.npy"}
print(Mo.ambiente())
''' + REGISTRO),
    code('''for compito in COMPITI.split(","):
    art, t = Mo.leggi()
    for nome in NOMI.split(","):
        X = None
        if nome != "TF-IDF":
            ordine = pd.read_csv(Mo.EMBEDDING / "ordine_articoli.csv").article_id
            assert (ordine.to_numpy() == art.article_id.to_numpy()).all()
            X = np.load(Mo.EMBEDDING / FILE_EMB[nome])
        for k in range(1, 6):
            Mo.lineare_fold(compito, nome, art, k, X=X, n_jobs=N_JOBS, registra=registra)'''),
    md("## Le penalità scelte"),
    code('''righe = []
for compito in COMPITI.split(","):
    for nome in NOMI.split(","):
        for f in sorted((Mo.PREVISIONI / compito / nome).glob("penalita_fold*.csv")):
            p = pd.read_csv(f); g = Mo.GRIGLIE[(compito, "TF-IDF" if nome == "TF-IDF" else "embedding")]
            s = p[p.scelto].iloc[0]
            righe.append({"compito": compito, "modello": nome, "fold": s.fold, "penalità": s.valore,
                          "misura interna": round(s["misura interna"], 3), "sul bordo": s.valore in (g[0], g[-1])})
pen = pd.DataFrame(righe); pen.to_csv(OUT / "penalita_scelte.csv", index=False)
pen.pivot_table(index=["compito", "modello"], columns="fold", values="penalità")'''),
    code('''print("scelte sul bordo della griglia:", int(pen["sul bordo"].sum()), "su", len(pen))
pen[pen["sul bordo"]]'''),
], "04_lineari.ipynb", kernel="johnvenv")

# ------------------------------------------------------------------------------------------------ 5. ModernBERT (GPU)
notebook([
    md("""# Tentativo 19 · 5. ModernBERT addestrato

ModernBERT si addestra tutto, un fold alla volta, sul compito scelto, con lo split per voto, come nei tentativi 16 e 18:
5 epoche, lr 3·10⁻⁵, batch effettivo 32, nessuna validazione (si tiene l'ultima epoca).
- **Classificazione**: 5 uscite, entropia incrociata con le classi bilanciate e label smoothing 0,1.
- **Regressione**: un'uscita, errore quadratico sul voto/100 (l'uscita si rimoltiplica per 100).

Ogni articolo entra intero (`MAX_TOKEN` = 4.096, sopra l'articolo più lungo: notebook 3). Si salvano i logit o il voto
previsto per ogni articolo di ogni fold, la curva della loss e il modello di ogni fold (`Modelli/19_DatasetAnonimo/`)."""),
    code('''COMPITO = "classificazione"
MODELLO = "answerdotai/ModernBERT-base"
NOME = "ModernBERT"
MAX_TOKEN = 4096
EPOCHE = 5
LR = 3e-5
BATCH = 4
GB_GPU = 16'''),
    code('''import time
import warnings

import pandas as pd

import modelli as Mo

warnings.filterwarnings("ignore")
OUT = Mo.Path("Results"); OUT.mkdir(exist_ok=True)
''' + REGISTRO + '''
art, t = Mo.leggi()
print(f"{COMPITO}: {len(art):,} articoli, {len(t)} testate"); print(Mo.ambiente())'''),
    code('''for k in range(1, 6):
    Mo.bert_fold(COMPITO, art, k, MODELLO, NOME, registra, max_token=MAX_TOKEN, epoche=EPOCHE, lr=LR, batch=BATCH, gb_gpu=GB_GPU)'''),
    code('''curve = pd.concat([pd.read_csv(f).assign(fold=f.stem[-1]) for f in sorted((Mo.PREVISIONI / COMPITO / NOME).glob("curva_fold*.csv"))])
curve.pivot(index="epoca", columns="fold", values="loss").round(4)'''),
], "05_modernbert.ipynb", kernel="johnvenv")

# ------------------------------------------------------------------------------------------------ 6. valutazione
notebook([
    md("""# Tentativo 19 · 6. La valutazione

Tutte le misure si ricalcolano dalle uscite per articolo dei notebook 4 e 5 (`previsioni/`), per testata. Il notebook
salva le tabelle in `Results/`; il notebook 7 le mostra in una pagina.

**Le due filosofie sullo stesso terreno** (`metriche.py`):

| | spazio dei **livelli** | spazio del **voto** |
|---|---|---|
| classificatore | livello votato dagli articoli | voto atteso: probabilità × voto medio del livello nel training |
| regressore | confini NewsGuard · quote del training (16) · voto degli articoli (17) | media prevista · media riadattata ai quantili del training |
| nessun testo | classe più frequente nel training | media del training |

Nello spazio dei livelli: F1, accuratezza, **tolerant** (±1 livello), kappa, Spearman… Nello spazio del voto: MAE,
Spearman, R², **entro 10 e 20 punti** (la tolerant del voto)… Ogni trasformazione si calcola una volta; il bootstrap
ricampiona le testate (2.000 volte, seme fisso)."""),
    code('''NOMI = "TF-IDF,Qwen3 8B,Mistral 24B,ModernBERT"
B = 2000'''),
    code('''import warnings

import numpy as np
import pandas as pd

import metriche as Me
import modelli as Mo

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
OUT = Mo.Path("Results"); OUT.mkdir(exist_ok=True)
art, t = Mo.leggi()
print(Mo.ambiente())'''),

    md("## 1. Quali modelli hanno tutti e 5 i fold"),
    code('''PA = {}
for nome in NOMI.split(","):
    for filosofia, compito in [("classificatore", "classificazione"), ("regressore", "regressione")]:
        pa = Mo.previsioni(compito, nome)
        completo = pa is not None and pa.domain.nunique() == len(t)
        print(f"{nome:12s} {filosofia:15s} {'pronto' if completo else 'mancano dei fold'}")
        if completo:
            PA[(nome, filosofia)] = pa
NOMI = [n for n in NOMI.split(",") if (n, "classificatore") in PA or (n, "regressore") in PA]'''),

    md("## 2. Le viste di ogni modello e le loro misure (per testata)"),
    code('''VISTE = {}                                                  # (modello, filosofia, vista) -> (spazio, vero, previsto)
for (nome, filosofia), pa in PA.items():
    if filosofia == "classificatore":
        viste = Me.viste_classificatore(Me.testate_classificatore(pa, t), t)
    else:
        viste = Me.viste_regressore(Me.testate_regressore(pa, t), pa, art)
    VISTE.update({(nome, filosofia, v): x for v, x in viste.items()})
VISTE.update({("nessun testo", "riferimento", v): x for v, x in Me.viste_riferimento(t).items()})
tutte = pd.concat([Me.misure_viste({k[2]: x}).assign(modello=k[0], filosofia=k[1]) for k, x in VISTE.items()], ignore_index=True)
tutte = tutte[["modello", "filosofia", "vista", "spazio"] + [c for c in tutte.columns if c not in ("modello", "filosofia", "vista", "spazio")]]
tutte.to_csv(OUT / "tutte_le_misure.csv", index=False)
print(f"{len(tutte)} viste, {tutte.shape[1] - 4} misure: Results/tutte_le_misure.csv")'''),
    md("**Spazio dei livelli.**"),
    code('''colonne_l = ["F1 macro", "accuratezza", "accuratezza bilanciata", "tolerant (±1 livello)", "errore medio in livelli",
            "kappa quadratica", "Spearman", "affidabile: accuratezza"] + [f"F1 {l}" for l in Mo.LIVELLI]
tutte[tutte.spazio == "livelli"].set_index(["modello", "filosofia", "vista"])[colonne_l].round(3)'''),
    md("**Spazio del voto.**"),
    code('''colonne_v = ["MAE", "RMSE", "R²", "Spearman", "Pearson", "CCC di Lin", "entro 10 punti", "entro 20 punti",
            "distorsione (previsto − vero)", "affidabile: accuratezza", "affidabile: AUC"]
tutte[tutte.spazio == "voto"].set_index(["modello", "filosofia", "vista"])[colonne_v].round(3)'''),

    md("## 3. Le misure con le probabilità del classificatore, e le misure per articolo"),
    code('''prob = pd.DataFrame({nome: Me.misure_probabilita(Me.testate_classificatore(PA[(nome, "classificatore")], t))
                     for nome in NOMI if (nome, "classificatore") in PA}).T
prob.to_csv(OUT / "misure_probabilita.csv")
righe = []
for (nome, filosofia), pa in PA.items():                     # ogni articolo da solo, contro il valore della sua testata
    a = pa.merge(art[["article_id", "label", "score"]], on="article_id")
    m = Me.misure_livelli(a.label, a.previsione) if filosofia == "classificatore" else Me.misure_voto(a.score, a.previsione)
    righe.append({"modello": nome, "filosofia": filosofia, **m})
per_articolo = pd.DataFrame(righe); per_articolo.to_csv(OUT / "misure_per_articolo.csv", index=False)
display(prob.round(3))
per_articolo.round(3)'''),

    md("## 4. Intervalli al 95% delle misure principali"),
    code('''CHIAVI = {"livelli": ["F1 macro", "tolerant (±1 livello)", "kappa quadratica", "Spearman", "affidabile: accuratezza"],
          "voto": ["MAE", "Spearman", "entro 10 punti", "entro 20 punti", "affidabile: accuratezza"]}
righe = []
for (nome, filosofia, vista), (spazio, y, p) in VISTE.items():
    ic = Me.bootstrap(y, p, spazio, CHIAVI[spazio], B=B)
    m = Me.MISURA[spazio](y, p)
    righe += [{"modello": nome, "filosofia": filosofia, "vista": vista, "spazio": spazio, "misura": c,
               "valore": m[c], "da": ic[c][0], "a": ic[c][1]} for c in CHIAVI[spazio]]
intervalli = pd.DataFrame(righe); intervalli.to_csv(OUT / "intervalli.csv", index=False)
print(f"{len(intervalli)} intervalli: Results/intervalli.csv")'''),

    md("""## 5. Classificatore contro regressore, a coppie
Per ogni modello, le stesse testate, la stessa misura: **regressore − classificatore**. Nello spazio dei livelli il
classificatore usa il suo livello votato e il regressore le tre trasformazioni; nello spazio del voto il classificatore
usa il voto atteso e il regressore la media, così com'è o riadattata. Se l'intervallo non contiene lo 0, la differenza
non è caso. Per misure in cui meno è meglio (MAE, errore in livelli) una differenza negativa è a favore del regressore."""),
    code('''CONFRONTI = {"livelli": ("livello votato dagli articoli", ["livelli: confini NewsGuard", "livelli: quote del training",
                                                              "livelli: voto degli articoli"], CHIAVI["livelli"]),
             "voto": ("voto atteso dalle probabilità", ["media prevista", "media riadattata ai quantili del training"], CHIAVI["voto"])}
righe = []
for nome in NOMI:
    if (nome, "classificatore") not in PA or (nome, "regressore") not in PA:
        continue
    for spazio, (vista_c, viste_r, chiavi) in CONFRONTI.items():
        for vista_r in viste_r:
            for c in chiavi:
                d, (a, b) = Me.bootstrap_differenza(VISTE[(nome, "regressore", vista_r)], VISTE[(nome, "classificatore", vista_c)], c, B=B)
                righe.append({"modello": nome, "spazio": spazio, "regressore": vista_r, "classificatore": vista_c, "misura": c,
                              "differenza": d, "da": a, "a": b, "non è caso": not (a <= 0 <= b)})
coppie = pd.DataFrame(righe); coppie.to_csv(OUT / "classificatore_contro_regressore.csv", index=False)
coppie.round(3)'''),
], "06_valutazione.ipynb", kernel="johnvenv")

# ------------------------------------------------------------------------------------------------ 7. i risultati in una pagina
notebook([
    md("""# Tentativo 19 · 7. I risultati in una pagina

Dataset anonimo, 226 testate, uno split per voto, 5 fold; ogni misura è per testata, con l'intervallo al 95% (bootstrap
sulle testate). Le tabelle vengono dal notebook 6 (`Results/`), che spiega come sono fatte.

- **Spazio dei livelli**: il livello NewsGuard l1–l5. Il classificatore vota il livello; il regressore ci arriva con i
  confini NewsGuard, con le quote del training o con il voto degli articoli.
- **Spazio del voto**: il voto 0–100. Il regressore fa la media; il classificatore ci arriva con il voto atteso.
- **Nessun testo**: chi risponde sempre la classe più frequente o la media del training.

Colori: più scuro = meglio (per MAE ed errore, più chiaro = meglio)."""),
    code('''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "Results/"
tutte = pd.read_csv(OUT + "tutte_le_misure.csv")
ic = pd.read_csv(OUT + "intervalli.csv")
coppie = pd.read_csv(OUT + "classificatore_contro_regressore.csv")
pd.set_option("display.width", 220)


def con_intervallo(spazio, misure):
    """Una tabella modello × vista con «valore [da; a]» per le misure scelte."""
    d = ic[ic.spazio == spazio].copy()
    d["testo"] = [f"{v:.3f} [{a:.3f}; {b:.3f}]" if abs(v) < 2 else f"{v:.1f} [{a:.1f}; {b:.1f}]"
                  for v, a, b in zip(d.valore, d.da, d.a)]
    return d.pivot_table(index=["modello", "filosofia", "vista"], columns="misura", values="testo", aggfunc="first")[misure]


def colori(spazio, misure, al_contrario=()):
    """La stessa tabella con i soli valori, colorata colonna per colonna."""
    d = tutte[tutte.spazio == spazio].set_index(["modello", "filosofia", "vista"])[misure]
    try:
        st = d.style.format("{:.3f}")
    except AttributeError:                                  # senza jinja2 niente colori, solo i numeri
        return d.round(3)
    for c in misure:
        st = st.background_gradient(cmap="Blues_r" if c in al_contrario else "Blues", subset=[c])
    return st'''),

    md("## 1. Spazio dei livelli"),
    code('''LIV = ["F1 macro", "tolerant (±1 livello)", "kappa quadratica", "Spearman", "affidabile: accuratezza"]
colori("livelli", LIV + ["accuratezza", "errore medio in livelli"], al_contrario=["errore medio in livelli"])'''),
    code('''con_intervallo("livelli", LIV)'''),

    md("## 2. Spazio del voto"),
    code('''VOT = ["MAE", "Spearman", "entro 10 punti", "entro 20 punti", "affidabile: accuratezza"]
colori("voto", VOT + ["R²", "CCC di Lin"], al_contrario=["MAE"])'''),
    code('''con_intervallo("voto", VOT)'''),

    md("""## 3. Classificatore o regressore?
Regressore − classificatore, stesso modello, stesse testate. ✓ = l'intervallo non contiene lo 0. Per MAE ed errore,
negativo = meglio il regressore; per tutte le altre misure, positivo = meglio il regressore."""),
    code('''c = coppie.copy()
c["regressore − classificatore"] = [f"{d:+.3f} [{a:+.3f}; {b:+.3f}]{' ✓' if n else ''}" for d, a, b, n in
                                    zip(c.differenza, c.da, c.a, c["non è caso"])]
c.pivot_table(index=["spazio", "misura", "regressore"], columns="modello", values="regressore − classificatore", aggfunc="first")'''),

    md("## 4. In un grafico"),
    code('''RIGHE = [("livelli", ["F1 macro", "tolerant (±1 livello)"]), ("voto", ["MAE", "Spearman"])]
COLORE = {"classificatore": "#2a78d6", "regressore": "#eb6834", "riferimento": "#8a8983"}
n = [ic[ic.spazio == s].groupby(["modello", "vista"]).ngroups for s, _ in RIGHE]
fig, assi = plt.subplots(2, 2, figsize=(13, 0.32 * sum(n) + 2), gridspec_kw={"height_ratios": n})
for riga, (spazio, misure) in enumerate(RIGHE):
    for col, misura in enumerate(misure):
        ax = assi[riga, col]
        d = ic[(ic.spazio == spazio) & (ic.misura == misura)].reset_index(drop=True)
        y = np.arange(len(d))
        ax.errorbar(d.valore, y, xerr=[d.valore - d.da, d.a - d.valore], fmt="none", ecolor="#b5b3ab", lw=1.5, capsize=2)
        ax.scatter(d.valore, y, s=40, c=[COLORE[f] for f in d.filosofia], zorder=3)
        ax.set_yticks(y, [f"{m} · {v}" for m, v in zip(d.modello, d.vista)] if col == 0 else [""] * len(y), fontsize=8)
        ax.invert_yaxis(); ax.set_title(f"{spazio}: {misura}", fontsize=10, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
for f, c in COLORE.items():
    assi[0, 0].scatter([], [], c=c, label=f)
fig.legend(loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(); plt.savefig(OUT + "risultati.png", dpi=150, bbox_inches="tight"); plt.show()'''),

    md("## 5. In breve (generato dalle tabelle)"),
    code('''def migliore(spazio, misura, minore=False):
    d = tutte[(tutte.spazio == spazio) & (tutte.filosofia != "riferimento")]
    r = d.loc[d[misura].idxmin() if minore else d[misura].idxmax()]
    rif = tutte[(tutte.spazio == spazio) & (tutte.filosofia == "riferimento")][misura].iloc[0]
    return f"- **{spazio}, {misura}**: {r.modello} ({r.filosofia}, {r.vista}) = {r[misura]:.3f}; senza testo {rif:.3f}"


print("\\n".join([migliore("livelli", "F1 macro"), migliore("livelli", "tolerant (±1 livello)"),
                  migliore("livelli", "Spearman"), migliore("voto", "MAE", minore=True), migliore("voto", "Spearman"),
                  migliore("voto", "entro 10 punti")]))
vinte = coppie[coppie["non è caso"]]
print(f"\\nConfronti classificatore/regressore con una differenza che non è caso: {len(vinte)} su {len(coppie)}")'''),
], "07_risultati.ipynb", kernel="johnvenv")

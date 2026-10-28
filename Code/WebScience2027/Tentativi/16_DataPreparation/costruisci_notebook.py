"""Costruisce i notebook 3 (feature), 4 (embedding), 5 (ModernBERT a 2048 token) e 6 (confronto) del tentativo 16.

I notebook sono salvati già eseguiti, con gli output: se il codice di un notebook non cambia, resta com'è; se cambia,
si riscrive senza output e va rieseguito (esegui_tutto.sh)."""
from pathlib import Path

import nbformat as nbf

CARTELLA = "/home/john/Repos/PublisherClassification/Code/WebScience2027/Tentativi/16_DataPreparation/"

TABELLA = """| | che cosa impara | su quali articoli | come si decide la testata |
|---|---|---|---|
| **1. 5 classi** | il livello (l1…l5) | tutti | voto degli articoli |
| **2. binario** | affidabile sì/no (punteggio ≥ 60) | tutti | voto degli articoli |
| **3. binario sugli estremi** | affidabile sì/no | solo l1, l2, l4, l5 | voto degli articoli |
| **4. regressione** | il punteggio 0–100 | tutti | media dei punteggi previsti |
| **5. regressione bilanciata** | il punteggio 0–100, ogni livello pesa uguale | tutti | media dei punteggi previsti |
| **6. regressione sugli estremi** | il punteggio 0–100 | solo l1, l2, l4, l5 | media dei punteggi previsti |

Per le regressioni i 5 livelli si ricavano con le **quote del training** (le testate si ordinano per punteggio previsto
e si tagliano con le stesse quote di livelli del training) e "affidabile" = punteggio previsto ≥ 60."""

SETUP = '''import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import funzioni as F

warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
OUT = Path("Results/{nome}"); OUT.mkdir(parents=True, exist_ok=True)
art = pd.read_csv(F.DATASET / "articoli.csv.gz")
t = pd.read_csv(F.DATASET / "testate.csv", index_col="domain")
print(f"{{len(art):,}} articoli di {{len(t)}} testate, 5 fold")'''

RISULTATI = '''# {n}.1 dagli articoli alle testate, le misure (salvate in {out}), il grafico
T = F.testate_da_articoli(pa, t); T.to_csv(OUT / "previsioni_testate.csv")
tab5, tab2 = F.misure_testate(T); tab5.to_csv(OUT / "cinque_livelli.csv"); tab2.to_csv(OUT / "binario.csv")
F.grafico_riassunto(T, tab5, tab2, "{titolo}", OUT / "riassunto.png")'''


def notebook(celle, nome, kernel="johnvenv"):
    for c in celle:                         # la cella dei parametri (MAIUSCOLE = valore) va etichettata per papermill
        if c.cell_type == "code" and c.source.split("\n")[0].split("=")[0].strip().isupper() and "import" not in c.source:
            c.metadata["tags"] = ["parameters"]; break
    nb = nbf.v4.new_notebook(cells=celle)
    nb.metadata["kernelspec"] = {"name": kernel, "display_name": "johnVenv", "language": "python"}
    vecchio = Path(CARTELLA + nome)
    if vecchio.exists() and [c.source for c in nbf.read(vecchio, as_version=4).cells
                             if "injected-parameters" not in c.metadata.get("tags", [])] == [c.source for c in celle]:
        print(nome, "invariato"); return          # stesso codice: il notebook resta com'è, con i suoi output
    nbf.write(nb, CARTELLA + nome)
    for c in celle:
        if c.cell_type == "code":
            compile(c.source, nome, "exec")
    print(nome, len(celle), "celle")


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

# ------------------------------------------------------------------------------------------------ 3. feature
notebook([
    md(f"""# Tentativo 16 · 3. Il classificatore a feature

Il classificatore di sempre: **TF-IDF** di parole e coppie di parole (al più 100.000) su **ogni articolo**, un modello
lineare (regressione logistica con C = 10 e classi bilanciate, oppure ridge con alpha = 1), poi si mettono insieme gli
articoli di ogni testata. Stessi 5 fold del notebook 1; a ogni fold il TF-IDF e i modelli si imparano sui soli articoli di
training. Sei configurazioni:

{TABELLA}"""),
    code(SETUP.format(nome="feature")),
    md("## 1. Addestramento: sei modelli per fold\nCirca 4 minuti la prima volta; poi si ricarica."),
    code('''# 1.1 TF-IDF e i sei modelli lineari, fold per fold
def un_fold(k):
    tr, te = (art.fold != k).to_numpy(), (art.fold == k).to_numpy()
    v = F._tfidf(); Xtr, Xte = v.fit_transform(art.text[tr]), v.transform(art.text[te])
    out = art.loc[te, ["article_id", "domain"]].reset_index(drop=True)
    return out.assign(**F.modelli_lineari(Xtr, Xte, art.label.to_numpy()[tr], art.score.to_numpy()[tr]))

f = OUT / "previsioni_articoli.csv"
pa = pd.read_csv(f) if f.exists() else None
if pa is None or set(pa.article_id) != set(art.article_id):          # se il dataset è cambiato si ricalcola
    pa = pd.concat(Parallel(n_jobs=5)(delayed(un_fold)(k) for k in range(1, 6)), ignore_index=True); pa.to_csv(f, index=False)
print(f"{len(pa):,} articoli previsti")'''),
    md("## 2. I risultati\nUna riga per configurazione; a destra le quattro misure. Linee grigie: intervalli al 95% (bootstrap sulle testate)."),
    code(RISULTATI.format(n=2, out="Results/feature", titolo="Il classificatore a feature (TF-IDF) in sei configurazioni · 5 fold · linee grigie: intervalli al 95%")),
], "03_feature.ipynb")

# ------------------------------------------------------------------------------------------------ 4. embedding
notebook([
    md(f"""# Tentativo 16 · 4. Embedding + modelli lineari

Ogni articolo diventa un **vettore** prodotto da un modello nato per gli embedding (**Qwen3-Embedding**), usato così
com'è, **senza addestrarlo**: legge l'articolo intero (gli articoli arrivano al più a 1.500 parole, circa 2.000 token)
con un'istruzione che dice a che cosa serve il vettore. Sopra i vettori, gli **stessi modelli lineari** del notebook 3,
nelle stesse sei configurazioni e sugli stessi fold: così il confronto con il TF-IDF riguarda solo la rappresentazione.

{TABELLA}"""),
    code('''MODELLO = "Qwen/Qwen3-Embedding-0.6B"      # anche "Qwen/Qwen3-Embedding-8B"
MAX_TOKEN = 2304                            # 2.048 per l'articolo più l'istruzione
BATCH = 8
ISTRUZIONE = "Represent this news article to judge the reliability and journalistic standards of the outlet that published it"
GB_GPU = 20                                 # memoria GPU libera necessaria per partire
PROVA = 0                                   # > 0: solo quel numero di articoli per testata (prova veloce, in Results/prova)
ETICHETTA = ""                              # se c'è, i risultati vanno in Results/embedding/<modello>_<etichetta> (per un'altra istruzione)'''),
    code(SETUP.format(nome="embedding") + '''
import time
if PROVA:
    art = art.groupby("domain").head(PROVA).reset_index(drop=True); OUT = Path("Results/prova/embedding"); print(f"PROVA: {len(art):,} articoli")
NOME = MODELLO.split("/")[-1] + (f"_{ETICHETTA}" if ETICHETTA else "")
OUT = OUT / NOME; OUT.mkdir(parents=True, exist_ok=True)
registra = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)'''),
    md("## 1. Gli embedding\nUna volta sola, senza etichette; si salvano. Con lo 0,6B circa 15 minuti, con l'8B circa 2–3 ore (sulla V100)."),
    code('''# 1.1 un vettore per articolo
f, f_id = OUT / "embedding.npy", OUT / "embedding_article_id.npy"
if f.exists() and f_id.exists() and np.array_equal(np.load(f_id), art.article_id.to_numpy()):
    E = np.load(f)
else:                                                                 # mancano, o il dataset è cambiato: si ricalcolano
    F.aspetta_gpu(GB_GPU, registra)
    E = F.embedding_articoli(art.text.tolist(), MODELLO, MAX_TOKEN, BATCH, ISTRUZIONE, registra)
    np.save(f, E); np.save(f_id, art.article_id.to_numpy())
print(f"{E.shape[0]:,} articoli, vettori di {E.shape[1]} dimensioni")'''),
    md("## 2. I sei modelli lineari, fold per fold\nVettori standardizzati sul training di ogni fold; stessi modelli del notebook 3."),
    code('''# 2.1 addestramento e previsione
from sklearn.preprocessing import StandardScaler

def un_fold(k):
    tr, te = (art.fold != k).to_numpy(), (art.fold == k).to_numpy()
    s = StandardScaler().fit(E[tr])
    out = art.loc[te, ["article_id", "domain"]].reset_index(drop=True)
    return out.assign(**F.modelli_lineari(s.transform(E[tr]), s.transform(E[te]), art.label.to_numpy()[tr], art.score.to_numpy()[tr]))

pa = pd.concat(Parallel(n_jobs=5)(delayed(un_fold)(k) for k in range(1, 6)), ignore_index=True)
pa.to_csv(OUT / "previsioni_articoli.csv", index=False)
print(f"{len(pa):,} articoli previsti")'''),
    md("## 3. I risultati"),
    code(RISULTATI.format(n=3, out="Results/embedding/<modello>", titolo="Embedding {NOME} + modelli lineari, sei configurazioni · 5 fold · linee grigie: intervalli al 95%").replace('"Embedding {NOME}', 'f"Embedding {NOME}')),
], "04_embedding.ipynb")

# ------------------------------------------------------------------------------------------------ 5. ModernBERT
notebook([
    md(f"""# Tentativo 16 · 5. ModernBERT a 2.048 token

**ModernBERT-base** addestrato (fine-tuning) sui nostri articoli. Ora ogni articolo entra intero: al più 1.500 parole,
circa 2.000 token, dentro una finestra di **2.048 token**, quindi niente tagli e niente pezzi. Rispetto ai tentativi
precedenti (modello sottoallenato: circa 300 passi, 20 articoli per testata) qui ModernBERT vede **tutti gli articoli**
di training a ogni epoca (migliaia di passi), fino a 5 epoche, con learning rate 3e-5.

Prima una **diagnostica su un solo fold** (6 epoche, loss e F1 su validazione e test dopo ogni epoca) per vedere quante
epoche servono. Poi i 5 fold con un **numero di epoche fisso**, scelto dalla diagnostica, su tutte le testate di training
(una validazione di poche testate si è rivelata troppo rumorosa per scegliere l'epoca). Le configurazioni sono quelle del
notebook 3; per tempo si lanciano quelle scelte in `CONFIGURAZIONI`:

{TABELLA}

Classificatori: entropia incrociata con classi bilanciate e label smoothing 0,1. Regressori: errore quadratico sul
punteggio. Precisione fp16 (la V100 non supporta bf16)."""),
    code('''CONFIGURAZIONI = "regressione,5 classi"   # anche: binario, binario_estremi, regressione_bilanciata, regressione_estremi
DIAGNOSTICA = True                           # prima un fold solo (il 2), 6 epoche, con le curve
EPOCHE = 5                                   # fisso: la diagnostica mostra il plateau dalla 4ª–5ª epoca
VALIDAZIONE = 0.0                            # 0 = niente validazione: tutte le testate di training addestrano, si tiene l'ultima epoca
MAX_TOKEN = 2048
LR = 3e-5
BATCH = 4                                    # batch effettivo 32 (accumulo dei gradienti)
GB_GPU = 14                                  # memoria GPU libera necessaria per partire
PROVA = 0                                    # > 0: solo quel numero di articoli per testata (prova veloce, in Results/prova)'''),
    code(SETUP.format(nome="modernbert") + '''
import time
if PROVA:
    art = art.groupby("domain").head(PROVA).reset_index(drop=True); OUT = Path("Results/prova/modernbert"); OUT.mkdir(parents=True, exist_ok=True)
    print(f"PROVA: {len(art):,} articoli")
LOG = OUT / "avanzamento.log"
def registra(m):
    riga = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {m}"; print(riga, flush=True)
    with open(LOG, "a") as f:
        f.write(riga + "\\n")
SCELTE = [c.strip() for c in CONFIGURAZIONI.split(",") if c.strip()]
print("configurazioni:", SCELTE)'''),
    md("## 0. Diagnostica: quante epoche servono?\nFold 2, 5 classi, 6 epoche. Dopo ogni epoca: loss di training, F1 sulla validazione e sul fold di test (il test qui serve solo a guardare la curva)."),
    code('''# 0.1 un fold solo, 6 epoche, con le curve
if DIAGNOSTICA:
    cartella = OUT / "diagnostica"; cartella.mkdir(exist_ok=True)
    F.modernbert_fold(art, 2, "5 classi", cartella, registra, max_token=MAX_TOKEN, epoche=6, lr=LR, batch=BATCH, gb_gpu=GB_GPU, misura_test=True)
    c = pd.read_csv(cartella / "curva_fold2.csv")
    fig, assi = plt.subplots(1, 2, figsize=(11, 3.4))
    assi[0].plot(c.epoca, c.loss, marker="o"); assi[0].axhline(np.log(5), ls=":", color="grey"); assi[0].set_title("loss di training (··· a caso: ln 5)")
    assi[1].plot(c.epoca, c.validazione, marker="o", label="validazione"); assi[1].plot(c.epoca, c.test, marker="o", label="test (fold 2)")
    assi[1].set_title("F1 per testata"); assi[1].legend()
    for a in assi:
        a.set_xlabel("epoca")
    plt.tight_layout(); plt.savefig(OUT / "diagnostica.png", dpi=160, bbox_inches="tight"); plt.show()
    display(c.round(3))'''),
    md("## 1. Addestramento\nUn fold alla volta, una configurazione alla volta. Ogni fold finito si salva: se il notebook si interrompe, si riparte da lì."),
    code('''# 1.1 tutte le configurazioni scelte, fold per fold
parti = []
for config in SCELTE:
    cartella = OUT / config; cartella.mkdir(exist_ok=True)
    p = pd.concat([F.modernbert_fold(art, k, config, cartella, registra, max_token=MAX_TOKEN, epoche=EPOCHE, lr=LR, batch=BATCH, gb_gpu=GB_GPU,
                                     quota_validazione=VALIDAZIONE) for k in range(1, 6)], ignore_index=True)
    parti.append(p.set_index(["article_id", "domain"]))
pa = pd.concat(parti, axis=1).reset_index()
pa.to_csv(OUT / "previsioni_articoli.csv", index=False)
print(f"{len(pa):,} articoli previsti")'''),
    code('''# 1.2 la loss di training epoca per epoca, per ogni fold
fig, assi = plt.subplots(1, len(SCELTE), figsize=(6 * len(SCELTE), 3.4), squeeze=False)
for ax, config in zip(assi[0], SCELTE):
    for k in range(1, 6):
        c = pd.read_csv(OUT / config / f"curva_fold{k}.csv")
        ax.plot(c.epoca, c.loss, marker="o", label=f"fold {k}")
    ax.set_title(f"{config}: loss di training"); ax.set_xlabel("epoca"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "curve.png", dpi=160, bbox_inches="tight"); plt.show()'''),
    md("## 2. I risultati"),
    code(RISULTATI.format(n=2, out="Results/modernbert", titolo="ModernBERT a 2.048 token · 5 fold · linee grigie: intervalli al 95%")),
], "05_modernbert.ipynb")

# ------------------------------------------------------------------------------------------------ 6. confronto finale
notebook([
    md("""# Tentativo 16 · I risultati

**Che cosa abbiamo fatto**
- **Dataset**: testate politiche, testo estratto con trafilatura (modalità precisione), articoli tra 100 e 1.500 parole
  (entrano interi in 2.048 token), doppioni tolti tra tutte le testate, **una testata per gruppo** Owner/Parent Domain.
- **TF-IDF** (il riferimento): parole e coppie di parole + modello lineare.
- **Embedding congelati**: Qwen3-Embedding (0,6B e 8B), senza addestramento, + lo stesso modello lineare.
- **ModernBERT** addestrato: articoli interi a 2.048 token, tutti gli articoli, 5 epoche.

Ogni modello in due impostazioni: **classificazione a 5 classi** e **regressione sul punteggio** NewsGuard (0–100).
Stessi 5 fold per tutti."""),
    code('''import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funzioni as F

warnings.filterwarnings("ignore")
OUT = Path("Results/confronto"); OUT.mkdir(parents=True, exist_ok=True)
FONTI = {"TF-IDF": "Results/feature", "Embedding Qwen3 0.6B": "Results/embedding/Qwen3-Embedding-0.6B",
         "Embedding Qwen3 8B": "Results/embedding/Qwen3-Embedding-8B", "ModernBERT 2048": "Results/modernbert"}
T = {nome: pd.read_csv(Path(c) / "previsioni_testate.csv", index_col=0) for nome, c in FONTI.items()
     if (Path(c) / "previsioni_testate.csv").exists()}
t = pd.read_csv(F.DATASET / "testate.csv", index_col="domain")
print(f"{t.articoli.sum():,} articoli di {len(t)} testate · per livello: " + ", ".join(f"{l} {n}" for l, n in t.livello.value_counts().sort_index().items()))
print("modelli con i risultati:", ", ".join(T))'''),
    md("""## 1. Il confronto

Quattro misure: **F1** sui 5 livelli, **tolerant ±1** (giusto anche se sbaglia di un livello), e l'accuratezza nel dire
se una testata è **affidabile** (punteggio ≥ 60), su tutte le testate e solo sulle estreme (l1, l2, l4, l5). In blu la
classificazione a 5 classi, in arancio la regressione; linee grigie: intervalli al 95%."""),
    code('''valori, intervalli, differenze, vero = F.misure_modelli(T)
valori.to_csv(OUT / "confronto.csv")
F.grafico_confronto(valori, intervalli, vero, OUT / "confronto.png")'''),
    md("""### Lo stesso grafico, con i dettagli

Accanto ai nomi che cosa è ogni modello, nei titoli su quali testate si misura, sotto come si addestra e come si testa.
La **baseline** è il valore di chi risponde sempre allo stesso modo senza leggere i testi: per «affidabile sì/no» è dire
sempre «affidabile», cioè la quota di testate affidabili (0,65 su tutte, 0,60 sulle estreme). Un modello serve solo se
sta sopra."""),
    code('F.grafico_confronto(valori, intervalli, vero, OUT / "confronto_dettagli.png", dettagli=True)'),
    md("## 2. In numeri\nIn alto i valori; sotto la differenza da TF-IDF con la stessa impostazione (è chiara se l'intervallo non contiene lo zero)."),
    code('''display(valori.round(3))
differenze if len(differenze) else "per ora c'è solo il TF-IDF"'''),
    md("""## 3. ModernBERT è addestrato abbastanza? La loss epoca per epoca

Per ogni fold, la loss di training dopo ognuna delle 5 epoche e, accanto, la stessa loss sugli articoli del fold di test
(testate mai viste), misurata dopo l'ultima epoca.
- **5 classi**: entropia incrociata, con classi bilanciate e label smoothing 0,1 come nell'addestramento. Due riferimenti:
  *a caso* (1/5 a ogni livello, 1,63) e il *minimo possibile* (circa 0,42). Con la label smoothing la loss non può
  arrivare a zero: il minimo è quello di un modello che ha imparato a memoria ogni articolo di training.
- **Regressione**: errore medio in punti NewsGuard (radice dell'errore quadratico); riferimento: dire sempre il
  punteggio medio del training (circa 28 punti).

Al centro e a destra il test per fold, ModernBERT contro TF-IDF; in alto a destra la diagnostica fatta prima dei 5 fold
(fold 2, 6 epoche, il test misurato dopo ogni epoca)."""),
    code('''if (Path("Results/modernbert/regressione") / "fold1.csv").exists():
    L = F.loss_epoche("Results/modernbert", t)
    per_fold = F.misure_per_fold(T)
    f = Path("Results/modernbert/diagnostica/curva_fold2.csv")
    F.grafico_loss_epoche(L, per_fold, pd.read_csv(f) if f.exists() else None, OUT / "loss_epoche.png")
    L.to_csv(OUT / "loss_epoche.csv")
    display(L.round(2))
else:
    print("mancano le previsioni per articolo di ModernBERT (Results/modernbert/*/fold*.csv)")'''),
    md("""**Che cosa vuol dire.** ModernBERT non è sottoallenato: dopo 5 epoche la loss di training è 0,43, quasi il minimo
possibile (0,42), e l'errore della regressione sul training è di 4 punti. Il training è imparato quasi a memoria.

Sul test invece la loss è circa 2,1, più alta che rispondendo a caso (1,63): sulle testate nuove il modello sbaglia
con sicurezza. L'errore della regressione è di circa 25 punti, poco meno dei 28 di chi dice sempre la media. Ogni
articolo eredita il livello della sua testata, quindi il modello impara a riconoscere le testate di training più che
un segnale che vale per testate nuove. Più epoche non servirebbero: nella diagnostica il test si ferma a 0,30 dalla 4ª
epoca, e per fold ModernBERT e TF-IDF si alternano."""),
], "06_confronto.ipynb")

# ------------------------------------------------------------------------------------------------ 7. analisi degli embedding
notebook([
    md("""# Tentativo 16 · 7. Analisi degli embedding

L'embedding **Qwen3-Embedding-8B congelato** con un modello lineare è il primo modello sopra il TF-IDF (notebook 6). Qui
proviamo a migliorarlo senza toccare il modello di embedding:
1. **il parametro del modello lineare scelto dentro il training**: nel notebook 4 C = 10 e alpha = 1 erano fissi;
2. **altri modi di mettere insieme**: un vettore per testata (la media dei suoi articoli) invece del voto degli
   articoli, la regressione **ordinale** (i livelli sono in ordine), e **TF-IDF + embedding** (la media delle due previsioni);
3. **un'altra istruzione** per l'embedding: sullo stile invece che sull'affidabilità.

**Regola**: ogni scelta si fa dentro il training di ogni fold, con una cross-validation interna sulle testate di
training (3 fold, testate mai divise); il fold di test non entra mai nella scelta. Stessi 5 fold di tutto il tentativo."""),
    code('''EMBEDDING = "Results/embedding/Qwen3-Embedding-8B"              # gli embedding del notebook 4 (istruzione sull'affidabilità)
ALTRA = "Results/embedding/Qwen3-Embedding-8B_stile"            # punto 3: gli stessi con l'istruzione sullo stile
INTERNI = 3                                                     # fold della cross-validation interna
N_JOBS = 32'''),
    code('''import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funzioni as F

warnings.filterwarnings("ignore")
OUT = Path("Results/analisi_embedding"); OUT.mkdir(parents=True, exist_ok=True)
art = pd.read_csv(F.DATASET / "articoli.csv.gz")
t = pd.read_csv(F.DATASET / "testate.csv", index_col="domain")


def leggi(cartella):
    """Gli embedding salvati dal notebook 4, se ci sono e sono di questo dataset."""
    f, f_id = Path(cartella) / "embedding.npy", Path(cartella) / "embedding_article_id.npy"
    if f.exists() and f_id.exists() and np.array_equal(np.load(f_id), art.article_id.to_numpy()):
        return np.load(f)


def calcola(nome, funzione):
    """Le previsioni e la griglia di una variante: si calcolano una volta, poi si rileggono."""
    f, g = OUT / f"{nome}_previsioni.csv", OUT / f"{nome}_griglia.csv"
    if not (f.exists() and g.exists()):
        pa, griglia = funzione(); pa.to_csv(f, index=False); griglia.to_csv(g, index=False)
    return pd.read_csv(f), pd.read_csv(g)


E = leggi(EMBEDDING)
T = {"TF-IDF": pd.read_csv("Results/feature/previsioni_testate.csv", index_col=0),
     "8B come nel notebook 4 (C=10, alpha=1)": pd.read_csv(Path(EMBEDDING) / "previsioni_testate.csv", index_col=0)}
print(f"{len(art):,} articoli di {len(t)} testate · embedding {E.shape[1]} dimensioni")'''),
    md("""## 1. Il parametro scelto dentro il training

Per ogni fold, la cross-validation interna prova cinque valori: C della logistica (5 classi) da 0,001 a 10, alpha della
ridge (regressione) da 1 a 10.000. Criterio: F1 sui 5 livelli delle testate di validazione (voto degli articoli) per la
logistica, correlazione di Spearman tra punteggio vero e medio previsto per la ridge. Poi il modello si riaddestra su
tutto il training con il valore migliore. Circa 15 minuti la prima volta."""),
    code('''pa1, g1 = calcola("articoli", lambda: F.modelli_scelti(E, art, interni=INTERNI, n_jobs=N_JOBS))
F.grafico_griglia(g1, "1. Embedding 8B, un vettore per articolo", OUT / "griglia_articoli.png")
T["8B, parametro scelto"] = F.testate_da_previsioni(pa1, t)'''),
    md("""## 2. Altri modi di mettere insieme

- **Un vettore per testata**: la media dei vettori dei suoi articoli; il modello lineare impara sulle testate (circa
  180 per fold) invece che sugli articoli, quindi niente voto. Tre modelli: 5 classi, **ordinale** (quattro logistiche
  «livello ≥ l2», «≥ l3»…, Frank e Hall 2001) e regressione, ognuno con il parametro scelto come al punto 1.
- **TF-IDF + 8B**: per ogni articolo la media delle probabilità (5 classi) o dei punteggi (regressione) del TF-IDF del
  notebook 3 e dell'8B del punto 1, con pesi uguali fissati prima (nessuna scelta sul test)."""),
    code('''M, righe = F.media_per_testata(E, art)
pa2, g2 = calcola("testate", lambda: F.modelli_scelti(M, righe, tipi=("5 classi", "ordinale", "regressione"), interni=INTERNI, n_jobs=N_JOBS))
F.grafico_griglia(g2, "2. Embedding 8B, un vettore per testata", OUT / "griglia_testate.png")
T["8B, un vettore per testata"] = F.testate_da_previsioni(pa2, t)
T["8B, un vettore per testata, ordinale"] = F.testate_da_previsioni(pa2, t, ordinale=True)

tfidf = pd.read_csv("Results/feature/previsioni_articoli.csv").set_index("article_id")
insieme = pa1.set_index("article_id")
colonne = F.CLASSI + ["s_regressione"]
insieme[colonne] = (insieme[colonne] + tfidf.loc[insieme.index, colonne]) / 2
T["TF-IDF + 8B (media)"] = F.testate_da_previsioni(insieme.reset_index(), t)'''),
    md("""## 3. Un'altra istruzione

Qwen3-Embedding legge l'articolo preceduto da un'istruzione che dice a che cosa serve il vettore. Nel notebook 4:
*"Represent this news article to judge the reliability and journalistic standards of the outlet that published it"*.
Qui: *"Represent the writing style, tone, sourcing and rigor of this news article, regardless of its topic"*, per
spingere il vettore verso **come** è scritto l'articolo invece che **di che cosa** parla. Gli embedding si calcolano con il
notebook 4 (`ETICHETTA = "stile"`, circa 2 ore di GPU); sopra, gli stessi modelli dei punti 1 e 2."""),
    code('''E2 = leggi(ALTRA)
if E2 is None:
    print("gli embedding con l'istruzione sullo stile non ci sono ancora")
else:
    T["8B «stile» come nel notebook 4 (C=10, alpha=1)"] = pd.read_csv(Path(ALTRA) / "previsioni_testate.csv", index_col=0)
    pa3, g3 = calcola("stile_articoli", lambda: F.modelli_scelti(E2, art, interni=INTERNI, n_jobs=N_JOBS))
    T["8B «stile», parametro scelto"] = F.testate_da_previsioni(pa3, t)
    M2, _ = F.media_per_testata(E2, art)
    pa4, g4 = calcola("stile_testate", lambda: F.modelli_scelti(M2, righe, tipi=("5 classi", "ordinale", "regressione"), interni=INTERNI, n_jobs=N_JOBS))
    T["8B «stile», un vettore per testata"] = F.testate_da_previsioni(pa4, t)
    T["8B «stile», un vettore per testata, ordinale"] = F.testate_da_previsioni(pa4, t, ordinale=True)
    F.grafico_griglia(pd.concat([g3.assign(tipo=g3.tipo + ", articoli"), g4.assign(tipo=g4.tipo + ", testate")]),
                      "3. Embedding 8B con l'istruzione sullo stile", OUT / "griglia_stile.png")'''),
    md("""## 3 bis. Un LLM grande come lettore: Mistral-Small 24B

Non esistono modelli di embedding aperti più grandi di 8B, quindi il modello più grande è un **LLM generativo**,
Mistral-Small 24B, congelato e in 4 bit (notebook 8): ogni articolo è la media degli stati nascosti dei suoi token, allo
strato 27 di 40 (circa due terzi) e all'ultimo. Sopra, gli stessi modelli dei punti 1 e 2: il parametro scelto nel training
e un vettore per testata, come classificatore (5 classi) e come regressore."""),
    code('''MISTRAL, STRATI_MISTRAL = "Results/embedding/Mistral-Small-24B-Base-2501_strato{s}", [27, 40]
for s in STRATI_MISTRAL:
    Em = leggi(MISTRAL.format(s=s))
    if Em is None:
        print(f"i vettori di Mistral (strato {s}) non ci sono ancora: vanno calcolati con il notebook 8"); continue
    T[f"Mistral 24B, strato {s}, come nel notebook 8 (C=10, alpha=1)"] = pd.read_csv(Path(MISTRAL.format(s=s)) / "previsioni_testate.csv", index_col=0)
    pam, gm = calcola(f"mistral{s}_articoli", lambda: F.modelli_scelti(Em, art, interni=INTERNI, n_jobs=N_JOBS))
    T[f"Mistral 24B, strato {s}, parametro scelto"] = F.testate_da_previsioni(pam, t)
    Mm, _ = F.media_per_testata(Em, art)
    pbm, gbm = calcola(f"mistral{s}_testate", lambda: F.modelli_scelti(Mm, righe, interni=INTERNI, n_jobs=N_JOBS))
    T[f"Mistral 24B, strato {s}, un vettore per testata"] = F.testate_da_previsioni(pbm, t)
    F.grafico_griglia(pd.concat([gm.assign(tipo=gm.tipo + ", articoli"), gbm.assign(tipo=gbm.tipo + ", testate")]),
                      f"3 bis. Mistral-Small 24B, strato {s}", OUT / f"griglia_mistral{s}.png")'''),
    md("""## 4. Tutti i risultati insieme

Qui ci sono anche i modelli dei notebook 3–5, per avere tutto in un posto solo:

| nel grafico | che cos'è |
|---|---|
| TF-IDF | parole e coppie di parole, logistica/ridge (notebook 3): il riferimento |
| ModernBERT | ModernBERT-base addestrato, 5 epoche, articoli interi (notebook 5) |
| Qwen3 0,6B / 8B | embedding congelati con l'istruzione sull'affidabilità, C = 10 e alpha = 1 fissi (notebook 4) |
| parametro scelto | come sopra, ma C e alpha scelti dentro il training (punto 1) |
| un vettore per testata | la media dei vettori degli articoli, il modello impara sulle testate (punto 2) |
| ordinale | come sopra, con la regressione ordinale invece delle 5 classi (punto 2) |
| TF-IDF + 8B | la media delle previsioni dei due modelli (punto 2) |
| «stile» | gli stessi modelli 8B con l'istruzione sullo stile (punto 3) |
| Mistral 24B | l'LLM grande come lettore, strato 27 o 40, con C = 10 e alpha = 1 fissi (notebook 8), con il parametro scelto o con un vettore per testata (punto 3 bis) |

Le misure sono quelle del notebook 6: **F1** sui 5 livelli, **tolerant ±1** (giusto o sbagliato di un livello),
**affidabile sì/no** (punteggio ≥ 60) su tutte le testate e solo su l1–l2 contro l4–l5. In blu la classificazione a 5
classi, in arancio la regressione sul punteggio; linee grigie: intervalli al 95% (bootstrap sulle testate)."""),
    code("""TUTTI = {"TF-IDF": T["TF-IDF"],
         "ModernBERT 2048": pd.read_csv("Results/modernbert/previsioni_testate.csv", index_col=0),
         "Embedding Qwen3 0.6B": pd.read_csv("Results/embedding/Qwen3-Embedding-0.6B/previsioni_testate.csv", index_col=0)}
TUTTI.update({nome: v for nome, v in T.items() if nome != "TF-IDF"})
AFF, STI = "istruzione sull'affidabilità", "istruzione sullo stile"
DESCRIZIONI = {
    "TF-IDF": "TF-IDF\\nparole e coppie di parole\\n+ logistica / ridge (C=10, alpha=1)",
    "ModernBERT 2048": "ModernBERT-base, addestrato\\n5 epoche · lr 3e-5\\narticoli interi (2.048 token)",
    "Embedding Qwen3 0.6B": f"Qwen3-Embedding 0,6B congelato\\n{AFF}\\n+ logistica / ridge (C=10, alpha=1)",
    "8B come nel notebook 4 (C=10, alpha=1)": f"Qwen3-Embedding 8B congelato\\n{AFF}\\n+ logistica / ridge (C=10, alpha=1)",
    "8B, parametro scelto": f"Qwen3 8B, {AFF}\\nun vettore per articolo\\nC e alpha scelti nel training",
    "8B, un vettore per testata": f"Qwen3 8B, {AFF}\\nun vettore per testata (media)\\nC e alpha scelti nel training",
    "8B, un vettore per testata, ordinale": f"Qwen3 8B, {AFF}\\nun vettore per testata\\nregressione ordinale, C scelto",
    "TF-IDF + 8B (media)": "TF-IDF + Qwen3 8B (parametro scelto)\\nmedia delle due previsioni\\nper ogni articolo",
    "8B «stile» come nel notebook 4 (C=10, alpha=1)": f"Qwen3-Embedding 8B congelato\\n{STI}\\n+ logistica / ridge (C=10, alpha=1)",
    "8B «stile», parametro scelto": f"Qwen3 8B, {STI}\\nun vettore per articolo\\nC e alpha scelti nel training",
    "8B «stile», un vettore per testata": f"Qwen3 8B, {STI}\\nun vettore per testata (media)\\nC e alpha scelti nel training",
    "8B «stile», un vettore per testata, ordinale": f"Qwen3 8B, {STI}\\nun vettore per testata\\nregressione ordinale, C scelto"}
for s in [27, 40]:
    DESCRIZIONI[f"Mistral 24B, strato {s}, come nel notebook 8 (C=10, alpha=1)"] = f"Mistral-Small 24B (LLM) congelato, 4 bit\\nmedia dello strato {s} di 40\\n+ logistica / ridge (C=10, alpha=1)"
    DESCRIZIONI[f"Mistral 24B, strato {s}, parametro scelto"] = f"Mistral-Small 24B, strato {s} di 40\\nun vettore per articolo\\nC e alpha scelti nel training"
    DESCRIZIONI[f"Mistral 24B, strato {s}, un vettore per testata"] = f"Mistral-Small 24B, strato {s} di 40\\nun vettore per testata (media)\\nC e alpha scelti nel training"
NOTA = ("Addestramento: sui 4 fold di training, tutte le testate (l1…l5, anche le l3). Test: le testate del fold escluso, "
        "una previsione per testata (voto o media dei suoi articoli; con «un vettore per testata»\\nla previsione è già per "
        "testata). «Scelti nel training»: con una cross-validation interna alle testate di training, senza guardare il test. "
        "Baseline: chi risponde sempre allo stesso modo, senza leggere i testi.")
valori, intervalli, differenze, vero = F.misure_modelli(TUTTI)
valori.to_csv(OUT / "confronto.csv")
F.grafico_confronto(valori, intervalli, vero, OUT / "confronto.png")"""),
    md("### Lo stesso grafico, con i dettagli\nAccanto ai nomi che cosa è ogni modello, nei titoli su quali testate si misura, sotto come si addestra e come si testa."),
    code('F.grafico_confronto(valori, intervalli, vero, OUT / "confronto_dettagli.png", dettagli=True, descrizioni=DESCRIZIONI, nota=NOTA)'),
    md("### In numeri\nIn alto i valori; sotto la differenza da TF-IDF con la stessa impostazione (è chiara se l'intervallo non contiene lo zero)."),
    code("""display(valori.round(3))
differenze"""),
    md("""## 5. La testata decisa dal voto dei suoi articoli

Nei grafici sopra la testata prende il voto dei suoi articoli solo nei classificatori; nelle regressioni prende la media
dei punteggi. Qui **tutti i modelli che giudicano gli articoli** decidono la testata allo stesso modo, con il **voto di
maggioranza** (a parità vince il livello più basso):
- classificatore: ogni articolo vota il suo livello più probabile;
- regressione: in ogni fold i punteggi previsti degli articoli si ordinano e si tagliano con le quote di livelli degli
  articoli di training, poi ogni articolo vota il suo livello;
- affidabile sì/no: la maggioranza degli articoli ha un livello da l3 in su.

I modelli con «un vettore per testata» giudicano già la testata intera e restano come sopra."""),
    code('''def articoli(f):
    """Le previsioni per articolo salvate da un notebook (se ci sono)."""
    f = Path(f)
    return pd.read_csv(f) if f.exists() else None


mb = [pd.read_csv(f"Results/modernbert/{c}/fold{k}.csv") for c in ["5 classi", "regressione"] for k in range(1, 6)]
PA = {"TF-IDF": tfidf.reset_index(),
      "ModernBERT 2048": pd.concat(mb[:5]).merge(pd.concat(mb[5:])[["article_id", "s_regressione"]], on="article_id"),
      "Embedding Qwen3 0.6B": articoli("Results/embedding/Qwen3-Embedding-0.6B/previsioni_articoli.csv"),
      "8B come nel notebook 4 (C=10, alpha=1)": articoli(Path(EMBEDDING) / "previsioni_articoli.csv"),
      "8B, parametro scelto": pa1, "TF-IDF + 8B (media)": insieme.reset_index(),
      "8B «stile» come nel notebook 4 (C=10, alpha=1)": articoli(Path(ALTRA) / "previsioni_articoli.csv"),
      "8B «stile», parametro scelto": articoli(OUT / "stile_articoli_previsioni.csv")}
for s in STRATI_MISTRAL:
    PA[f"Mistral 24B, strato {s}, come nel notebook 8 (C=10, alpha=1)"] = articoli(Path(MISTRAL.format(s=s)) / "previsioni_articoli.csv")
    PA[f"Mistral 24B, strato {s}, parametro scelto"] = articoli(OUT / f"mistral{s}_articoli_previsioni.csv")
VOTO = {nome: F.testate_voto(PA[nome], t) if PA.get(nome) is not None else v for nome, v in TUTTI.items()}
NOTA_VOTO = ("La testata è decisa dal voto di maggioranza dei suoi articoli per tutti i modelli che giudicano gli articoli (regressione: ogni "
             "articolo diventa un livello con le quote del training);\\nchi ha «un vettore per testata» giudica già la testata intera. "
             "Addestramento sui 4 fold di training, test sul fold escluso. Baseline: chi risponde sempre allo stesso modo.")
valori_v, intervalli_v, differenze_v, _ = F.misure_modelli(VOTO)
valori_v.to_csv(OUT / "confronto_voto.csv")
F.grafico_confronto(valori_v, intervalli_v, vero, OUT / "confronto_voto_dettagli.png", dettagli=True, descrizioni=DESCRIZIONI, nota=NOTA_VOTO)'''),
    md("### In numeri, con il voto degli articoli"),
    code('''display(valori_v.round(3))
differenze_v'''),
    md("""## 6. Che cosa viene fuori

- **Scegliere il parametro dentro il training aiuta**: con C e alpha scelti, l'8B batte il TF-IDF sui 5 livelli in modo
  chiaro (F1 0,41 contro 0,33, intervallo della differenza da +0,02 a +0,15). La ridge sceglie quasi sempre alpha =
  1.000 (nel notebook 4 era 1): la regressione sale da 0,31 a 0,34. La C della logistica invece cambia molto tra i fold.
- **Un vettore per testata** è il migliore sui 5 livelli: F1 0,44 (+0,11 sul TF-IDF, da +0,03 a +0,19); l'ordinale non
  aggiunge nulla.
- **TF-IDF + 8B** è il più equilibrato: migliore del TF-IDF su tutte e quattro le misure (tolerant 0,88, affidabile
  sì/no 0,83), quindi le due rappresentazioni vedono cose in parte diverse.
- **L'istruzione sullo stile non aiuta**: stessi valori o un po' meno (un vettore per testata: 0,35 contro 0,44). Spingere
  il vettore lontano dall'argomento toglie segnale, come nelle analisi del tentativo 15 (argomento e geografia pesano).
- **Mistral-Small 24B** (un LLM tre volte più grande, notebook 8) arriva dove arriva l'8B: F1 sui 5 livelli 0,40–0,44
  (strato 27), tolerant fino a 0,88, affidabile sì/no fino a 0,82. Un lettore più grande non sposta i numeri: siamo vicini
  al tetto che le etichette permettono (vedi il tentativo 17, accordo con gli esperti).
- **Il voto degli articoli** (punto 5) cambia poco: per i classificatori è lo stesso; per le regressioni l'F1 sui 5 livelli
  scende un po' (la media dei punteggi tiene più informazione del voto), mentre affidabile sì/no resta uguale o sale di
  poco (TF-IDF + 8B: 0,83).
- Cautela: le varianti sono state confrontate sul test; per il paper va fissata prima una variante e le altre vanno
  presentate come analisi."""),
], "07_analisi_embedding.ipynb")

# ------------------------------------------------------------------------------------------------ 8. un LLM grande come lettore
notebook([
    md(f"""# Tentativo 16 · 8. Un LLM grande come lettore

Non esistono modelli **di embedding** aperti più grandi di 8B (Qwen3-Embedding-8B è il più grande della sua famiglia).
Per vedere se un modello più grande legge meglio le testate, usiamo un **LLM generativo da 24 miliardi di parametri**,
**Mistral-Small-24B** (base, licenza Apache), congelato: ogni articolo diventa la **media degli stati nascosti** dei suoi
token, in due punti del modello (lo strato 27 di 40, a circa due terzi, e l'ultimo). Nessuna istruzione: il modello legge
solo il testo.

Il modello è caricato **in 4 bit** (NF4) con i calcoli in fp16: in fp16 pieno servirebbero 48 GB e la V100 ne ha 32 e non
supporta bf16 (per questo non Gemma 27B, che in fp16 dà valori fuori scala). Sopra i vettori, gli **stessi modelli lineari**
dei notebook 3 e 4, nelle stesse sei configurazioni e sugli stessi fold; le varianti con il parametro scelto nel training e
con un vettore per testata sono nel notebook 7.

{TABELLA}"""),
    code('''MODELLO = "mistralai/Mistral-Small-24B-Base-2501"
MAX_TOKEN = 2048                            # ogni articolo entra intero (al più 1.500 parole)
BATCH = 4
STRATI = "27,40"                            # indici degli stati nascosti: 27 di 40 (circa due terzi) e l'ultimo
GB_GPU = 22                                 # memoria GPU libera necessaria per partire (il modello ne usa circa 19)
PROVA = 0                                   # > 0: solo quel numero di articoli per testata (prova veloce, in Results/prova)'''),
    code(SETUP.format(nome="embedding") + '''
import time
if PROVA:
    art = art.groupby("domain").head(PROVA).reset_index(drop=True); OUT = Path("Results/prova/embedding"); OUT.mkdir(parents=True, exist_ok=True)
    print(f"PROVA: {len(art):,} articoli")
NOME = MODELLO.split("/")[-1]
STRATI = [int(s) for s in str(STRATI).split(",")]
CARTELLE = {s: OUT / f"{NOME}_strato{s}" for s in STRATI}
registra = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)'''),
    md("""## 1. I vettori
Una volta sola, senza etichette; si salvano (uno per strato). Circa 4,5 ore sulla V100; se si interrompe, riparte dal
lavoro salvato."""),
    code('''# 1.1 un vettore per articolo e per strato
pronti = lambda c: (c / "embedding.npy").exists() and np.array_equal(np.load(c / "embedding_article_id.npy"), art.article_id.to_numpy())
if all(pronti(c) for c in CARTELLE.values()):
    E = {s: np.load(c / "embedding.npy") for s, c in CARTELLE.items()}
else:
    F.aspetta_gpu(GB_GPU, registra)
    E = F.embedding_llm(art.text.tolist(), MODELLO, MAX_TOKEN, BATCH, STRATI, registra, file_parziale=OUT / f"{NOME}_parziale.npy")
    for s, c in CARTELLE.items():
        c.mkdir(parents=True, exist_ok=True); np.save(c / "embedding.npy", E[s]); np.save(c / "embedding_article_id.npy", art.article_id.to_numpy())
    (OUT / f"{NOME}_parziale.npy").unlink(missing_ok=True)
for s in STRATI:
    print(f"strato {s}: {E[s].shape[0]:,} articoli, vettori di {E[s].shape[1]} dimensioni, tutti finiti: {np.isfinite(E[s]).all()}")'''),
    md("## 2. I sei modelli lineari, fold per fold, per ogni strato\nVettori standardizzati sul training di ogni fold; stessi modelli dei notebook 3 e 4."),
    code('''# 2.1 addestramento, previsione e misure
from sklearn.preprocessing import StandardScaler

def un_fold(X, k):
    tr, te = (art.fold != k).to_numpy(), (art.fold == k).to_numpy()
    s = StandardScaler().fit(X[tr])
    out = art.loc[te, ["article_id", "domain"]].reset_index(drop=True)
    return out.assign(**F.modelli_lineari(s.transform(X[tr]), s.transform(X[te]), art.label.to_numpy()[tr], art.score.to_numpy()[tr]))

for s, c in CARTELLE.items():
    pa = pd.concat(Parallel(n_jobs=5)(delayed(un_fold)(E[s], k) for k in range(1, 6)), ignore_index=True)
    pa.to_csv(c / "previsioni_articoli.csv", index=False)
    T = F.testate_da_articoli(pa, t); T.to_csv(c / "previsioni_testate.csv")
    tab5, tab2 = F.misure_testate(T); tab5.to_csv(c / "cinque_livelli.csv"); tab2.to_csv(c / "binario.csv")
    F.grafico_riassunto(T, tab5, tab2, f"{NOME}, strato {s} + modelli lineari, sei configurazioni · 5 fold · linee grigie: intervalli al 95%",
                        c / "riassunto.png")'''),
], "08_llm_grande.ipynb")

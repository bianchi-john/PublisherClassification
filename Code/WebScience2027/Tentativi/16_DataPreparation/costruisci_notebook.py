"""Costruisce i notebook 3 (feature), 4 (embedding) e 5 (ModernBERT a 2048 token) del tentativo 16."""
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
art = pd.read_csv("Results/dataset/articoli.csv.gz")
t = pd.read_csv("Results/dataset/testate.csv", index_col="domain")
print(f"{{len(art):,}} articoli di {{len(t)}} testate, 5 fold")'''

RISULTATI = '''# {n}.1 dagli articoli alle testate, le misure (salvate in {out}), il grafico
T = F.testate_da_articoli(pa, t); T.to_csv(OUT / "previsioni_testate.csv")
tab5, tab2 = F.misure_testate(T); tab5.to_csv(OUT / "cinque_livelli.csv"); tab2.to_csv(OUT / "binario.csv")
F.grafico_riassunto(T, tab5, tab2, "{titolo}", OUT / "riassunto.png")'''


def notebook(celle, nome, kernel="johnvenv"):
    nb = nbf.v4.new_notebook(cells=celle)
    nb.metadata["kernelspec"] = {"name": kernel, "display_name": "johnVenv", "language": "python"}
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
PROVA = 0                                   # > 0: solo quel numero di articoli per testata (prova veloce, in Results/prova)'''),
    code(SETUP.format(nome="embedding") + '''
import time
if PROVA:
    art = art.groupby("domain").head(PROVA).reset_index(drop=True); OUT = Path("Results/prova/embedding"); print(f"PROVA: {len(art):,} articoli")
NOME = MODELLO.split("/")[-1]
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
epoche servono. Poi i 5 fold: una parte delle testate di training (15%) fa da **validazione**, dopo ogni epoca si misura
lì (F1 per i classificatori, correlazione per i regressori) e si tiene **l'epoca migliore**. Le configurazioni sono
quelle del notebook 3; per tempo si lanciano quelle scelte in `CONFIGURAZIONI` (circa 5 ore ciascuna sulla V100):

{TABELLA}

Classificatori: entropia incrociata con classi bilanciate e label smoothing 0,1. Regressori: errore quadratico sul
punteggio. Precisione fp16 (la V100 non supporta bf16)."""),
    code('''CONFIGURAZIONI = "5 classi,regressione"   # anche: binario, binario_estremi, regressione_bilanciata, regressione_estremi
DIAGNOSTICA = True                           # prima un fold solo (il 2), 6 epoche, con le curve
EPOCHE = 5
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
    p = pd.concat([F.modernbert_fold(art, k, config, cartella, registra, max_token=MAX_TOKEN, epoche=EPOCHE, lr=LR, batch=BATCH, gb_gpu=GB_GPU)
                   for k in range(1, 6)], ignore_index=True)
    parti.append(p.set_index(["article_id", "domain"]))
pa = pd.concat(parti, axis=1).reset_index()
pa.to_csv(OUT / "previsioni_articoli.csv", index=False)
print(f"{len(pa):,} articoli previsti")'''),
    code('''# 1.2 la validazione epoca per epoca (per vedere quante epoche servono)
fig, assi = plt.subplots(1, len(SCELTE), figsize=(6 * len(SCELTE), 3.4), squeeze=False)
for ax, config in zip(assi[0], SCELTE):
    for k in range(1, 6):
        c = pd.read_csv(OUT / config / f"curva_fold{k}.csv")
        ax.plot(c.epoca, c.validazione, marker="o", label=f"fold {k}")
    ax.set_title(f"{config}: validazione dopo ogni epoca"); ax.set_xlabel("epoca")
    ax.set_ylabel("F1 per testata" if not config.startswith("regressione") else "correlazione per testata"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "curve.png", dpi=160, bbox_inches="tight"); plt.show()'''),
    md("## 2. I risultati"),
    code(RISULTATI.format(n=2, out="Results/modernbert", titolo="ModernBERT a 2.048 token · 5 fold · linee grigie: intervalli al 95%")),
], "05_modernbert.ipynb")

# ------------------------------------------------------------------------------------------------ 6. confronto finale
notebook([
    md("""# Tentativo 16 · 6. Il confronto finale

TF-IDF (notebook 3), embedding (notebook 4) e ModernBERT (notebook 5) **sulle stesse testate e sugli stessi 5 fold**,
ognuno in due impostazioni: **classificazione a 5 classi** e **regressione sul punteggio**. Quattro misure: F1 sui 5
livelli, tolerant ±1, e l'accuratezza nel dire se una testata è affidabile (punteggio ≥ 60), su tutte le testate e solo
sulle estreme (l1, l2, l4, l5). Si mostrano i modelli che hanno già finito."""),
    code('''import warnings
from pathlib import Path

import pandas as pd

import funzioni as F

warnings.filterwarnings("ignore")
OUT = Path("Results/confronto"); OUT.mkdir(parents=True, exist_ok=True)'''),
    md("## 1. I risultati di ogni modello"),
    code('''# 1.1 le previsioni per testata salvate dai notebook 3, 4 e 5
FONTI = {"TF-IDF": "Results/feature", "Embedding Qwen3 0.6B": "Results/embedding/Qwen3-Embedding-0.6B",
         "Embedding Qwen3 8B": "Results/embedding/Qwen3-Embedding-8B", "ModernBERT 2048": "Results/modernbert"}
T = {nome: pd.read_csv(Path(c) / "previsioni_testate.csv", index_col=0) for nome, c in FONTI.items()
     if (Path(c) / "previsioni_testate.csv").exists()}
print("modelli con i risultati:", ", ".join(T))'''),
    code('''# 1.2 le quattro misure (intervalli al 95% con un bootstrap sulle testate)
valori, intervalli, differenze, vero = F.misure_modelli(T)
valori.to_csv(OUT / "confronto.csv")
valori.round(3)'''),
    md("## 2. Il grafico\nUna riga per modello: in blu la classificazione a 5 classi, in arancio la regressione."),
    code('''F.grafico_confronto(valori, intervalli, vero, OUT / "confronto.png")'''),
    md("## 3. Le differenze dal TF-IDF\nStessa impostazione, stessi ricampionamenti: la differenza è chiara se l'intervallo non contiene lo zero."),
    code('''differenze if len(differenze) else "per ora c'è solo il TF-IDF"'''),
], "06_confronto.ipynb")

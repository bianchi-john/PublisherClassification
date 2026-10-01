# Cosa ho trovato riorganizzando i risultati

Indagine sui notebook `4 - Plotter`, `5 - Political Error Analysis`,
`6 - Domain inpact analisys`, `7 - Get prediction scores` (ora `4 - …`) e sull'intera cartella
`Results/`. Ogni affermazione qui sotto è stata **verificata ricalcolando i numeri
dai CSV**, non dedotta leggendo il codice.

Verificato il 30 settembre 2026.

---

## 1. Difetti che hanno toccato il paper

### 1.1 RoBERTa e Llama sono scambiati nei numeri article-level — CONFERMATO

`4 - Plotter.ipynb`, prime righe:

```python
bertDf    = pd.read_csv('Results/Classification/bert.csv')
robertaDf = pd.read_csv('Results/Classification/llama.csv')      # ← legge Llama
llamaDf   = pd.read_csv('Results/Classification/roberta.csv')    # ← legge RoBERTa
```

poi `robertaDf['ModelName']='RobertaLarge'` stampa l'etichetta sbagliata sui dati.

**Quale file contiene cosa** — risalito ai writer nel notebook 2: il trainer Llama
(`model_id="meta-llama/Meta-Llama-3-8B"`) scrive `llama.csv`, quello RoBERTa
(`model_name="roberta-large"`) scrive `roberta.csv`. Conferma indipendente dallo
schema: `llama_enriched.csv` è l'unico con le colonne `Train_Losses_Per_Epoch`,
che solo il writer Llama produce. **I file sono giusti; è il Plotter a incrociarli.**

**Effetto sui numeri pubblicati:**

| | F1-macro reale | Tol@1.0 reale | Il paper attribuisce a |
|---|---|---|---|
| **Llama** | **0.5484** (il più alto) | **0.8377** (il più alto) | RoBERTa |
| **RoBERTa** | 0.5444 | 0.8303 | Llama |

Frasi da correggere nel paper:
- *"RoBERTa shows the highest mean F1-macro of 0.55"* → è **Llama**.
- *"RoBERTa achieves a top score of 0.84"* (Tol@1.0) → è **Llama**.

**Propagazione.** Lo scambio è stato congelato dentro
`Results/predictions/finetuned/confusion_matrices_all_models.csv`, prodotto dal Plotter: lì la
chiave `'llama3'` contiene i dati di `roberta.csv` e `'roberta'` quelli di
`llama.csv` (verificato su tutti e 5 i fold). Il notebook 5 legge quel file nelle
celle 32–36, quindi anche 5 sue figure hanno le serie scambiate.

**Quali figure del paper sono colpite: solo una.** Incrociando l'analisi dei
notebook 5 e 6:

| Figura del paper | Origine | Stato |
|---|---|---|
| `results_boxplot_baselines_updated.pdf` | nb4 | ❌ **scambiata** |
| `error_rate_analysis.pdf` | nb5 cella 37, legge i CSV grezzi | ✅ corretta |
| `mean_error_direction.pdf` | nb5 cella 42, legge i CSV grezzi | ✅ corretta |
| `orientationDistributionInverted.pdf` | nb5 cella 27 | ✅ corretta |
| `politicalErrorRates.pdf` | nb5 cella 41 | ✅ corretta |
| `politicalErrorsDirection.pdf` | nb5 cella 44 | ✅ corretta |
| `domain_metrics_comparison.pdf` | nb6 | ✅ corretta |

I numeri **publisher-level del paper sono corretti**: li ho riprodotti a 4 decimali
(BERT 0.6651, RoBERTa 0.6902, Llama 0.6651, Mistral 0.6834). Anche la tabella del
chi-quadro è corretta.

### 1.2 "Mean Error Direction" ha due definizioni diverse

Il paper definisce nella metodologia:

> MED = (1/|E|) Σ_{i∈E} (ŷ_i − y_i),  con E = insieme degli errori

Ma i valori riportati per livello di reliability si ottengono **dividendo per N**
(tutti gli articoli della classe, inclusi i corretti che contribuiscono 0):

| l5 | BERT | RoBERTa | Llama | Mistral |
|---|---|---|---|---|
| diviso per \|E\| (formula dichiarata) | −1.60 | −1.72 | −1.67 | −1.74 |
| **diviso per N** | **−1.29** | **−1.37** | **−1.20** | **−1.36** |
| **valori nel paper** | **−1.29** | **−1.37** | **−1.20** | **−1.36** |

Nel frattempo il MED **politico** divide davvero per |E| (verificato: i `count` in
`political_error_direction_stats.csv` sono i conteggi degli errori). Quindi due
quantità diverse portano lo stesso nome e la stessa etichetta sugli assi.

**Da decidere:** o si corregge la formula nel testo, o si ricalcolano i numeri.
Nel codice nuovo la scelta è un parametro esplicito, `over="errors"` / `over="all"`.

### 1.3 Il tie-break dell'aggregazione non è documentato

`scipy.stats.mode` in caso di parità restituisce **l'etichetta più bassa**. Su una
scala ordinale di affidabilità significa che ogni parità viene risolta verso *meno
affidabile*, sempre nella stessa direzione.

Quanti publisher decide: BERT 13, RoBERTa 20, Llama 15, Mistral 11 su 439
(2.5–4.6%). Vale circa 1 punto di accuracy. Mistral non era mai stato misurato.

Non è un bug — i numeri del paper si riproducono solo con questa regola — ma è una
scelta analitica da dichiarare.

---

## 2. Problemi di riproducibilità (non toccano i numeri, ma bloccano tutto)

1. **Tutti i notebook sono rotti.** Ogni percorso punta a `Results/Classification/…`,
   che non esiste più: i file sono in `Classification/…`. Peggio, il notebook 6
   gestisce `FileNotFoundError` stampando un messaggio e proseguendo, quindi
   **fallisce in silenzio lasciando sul disco i PDF vecchi che sembrano aggiornati**.
2. **`Results/Plots` era disallineato dal paper.** 4 delle 7 figure del paper
   differivano dalla versione in `Results/Plots`, e 36 dei 43 file erano orfani.
3. **La ricostruzione dei fold era indovinata, non letta.** Il notebook 5
   ri-derivava lo split con `StratifiedKFold(shuffle=True, random_state=42)`
   sperando che coincidesse con quello del training, con un troncamento silenzioso
   in caso di disallineamento. Gli `Article_IDs` erano già salvati nei file: il
   codice nuovo li usa.
4. **Un percorso con lo slash mancante**: `'Results/PlotserrorsDirection_reliability.pdf'`.
5. **Bug di indentazione** nella curva-k del notebook 6: i domini con *esattamente*
   k articoli venivano calcolati e poi scartati (430 domini invece di 439 a k=1).
6. **RNG non seminati o ri-seminati dentro i loop**, quindi i campionamenti a k
   diversi sono correlati per costruzione.
7. **Convenzioni di colore contraddittorie**: nel notebook 5 Llama è arancione in
   alcune figure e verde in altre, dentro lo stesso paper.
8. **Nomi di metrica collidenti**: "F1" è macro in una sezione e weighted in
   un'altra; "Tolerant Accuracy" è punteggio a credito parziale in un posto e
   booleano |err|≤1 in un altro.

---

## 3. Un risultato analitico nuovo, utile per rispondere a R1.2

Il paper confronta 0.60 (articolo) con 0.69 (publisher), ma quei due numeri
cambiano **due cose insieme**: l'unità di valutazione *e* la pesatura — l'accuracy
per articolo è dominata dai publisher più prolifici. Il revisore 1 aveva ragione.

La curva di aggregazione separa i due effetti: stessa unità (il publisher), stessa
popolazione fissa di 270 publisher, cambia solo *k*, il numero di articoli votati.

| | k=1 | k=5 | k=10 | k=30 | guadagno 1→30 |
|---|---|---|---|---|---|
| BERT | 0.667 | 0.692 | 0.699 | 0.705 | +0.038 |
| RoBERTa | 0.683 | 0.711 | 0.714 | 0.715 | +0.032 |
| Llama | 0.673 | 0.693 | 0.706 | 0.703 | +0.030 |
| Mistral | 0.684 | 0.688 | 0.690 | 0.693 | +0.009 |

**Lettura onesta:** passare dall'unità-articolo all'unità-publisher vale già da
solo ~0.60 → ~0.67 (è solo un cambio di pesatura). L'aggregazione vera e propria
aggiunge **+3 punti circa**, e satura intorno a k=5–10. È molto meno spettacolare
di "0.60 → 0.69", ma è difendibile e risponde esattamente all'obiezione.

---

## 3bis. Il baseline zero-shot nel boxplot article-level

> **Nota (30 set 2026).** Questa analisi è conservata come risultato, ma il
> confronto con GPT **non fa più parte** di `paper_results.ipynb`: il notebook
> carica solo i quattro modelli fine-tuned. I dati grezzi restano in
> `Results/predictions/zeroshot/`; le figure e la tabella derivate sono in
> `Results/_archive/gpt_comparison/`. Per riattivare il confronto basta
> `data.load_all_predictions(include_gpt=True)` — il pacchetto `irgood` continua
> a supportarlo, sono le celle del notebook a non usarlo.

I quattro modelli fine-tuned condividono **esattamente lo stesso** `article -> fold`
(verificato), quindi le predizioni di GPT si possono valutare sulle stesse cinque
partizioni publisher-disjoint e finire nel boxplot come quinta serie:
`Results/figures/article_metrics_boxplot_with_gpt.pdf`.

Controllo di coerenza: le metriche per fold ricalcolate dalle predizioni coincidono
con quelle salvate dai training a meno di 5.5e-17, quindi le due versioni della
figura sono costruite sugli stessi numeri.

**Cosa mostra.** GPT non è solo peggiore: è **al livello o sotto ogni baseline**.
Accuracy 0.245 contro una majority-class di 0.258 (sotto); metriche macro esattamente
sulla riga del caso (1/K = 0.20); Tolerant α=0.5 pari alla baseline. Un LLM
generalista, a cui è stata data la rubrica NewsGuard completa, non batte un
classificatore che ignora il testo.

**Caveat da dichiarare in didascalia.** Il box significa cose diverse: per i
fine-tuned la dispersione mescola training e test diversi, per GPT c'è solo la
variabilità della partizione di test. È scritto sotto l'asse della figura.

**Dato controintuitivo:** la dispersione di GPT fra i fold è la **più ampia** delle
cinque (range di accuracy 0.082 contro 0.038-0.069 dei fine-tuned). Lo scorer
zero-shot è anche il meno stabile al variare delle testate, non il più stabile.

---

## 4. Il bias politico regge alla correzione statistica

Ricalcolato a tre livelli di rigore crescente:

| | χ² | Cramér's V |
|---|---|---|
| **Articolo, non corretto** (come pubblicato) | 1208–1684 | 0.25–0.29 |
| **Publisher, clusterizzato** (1 osservazione per publisher) | 19–45 | 0.21–0.32 |

Il χ² crolla di due ordini di grandezza — era gonfiato dalla pseudo-replicazione,
esattamente come diceva il revisore — ma **resta significativo** (p < 0.001) e
l'effect size è sostanzialmente invariato.

Regressione logistica con SE clusterizzati su publisher, controllando per il
livello di reliability (odds ratio di errore rispetto a "Non labelled"):

| | Far Left | Far Right | Slightly Right |
|---|---|---|---|
| senza controllo | 4.0–5.4 | 3.3–4.4 | ~1.0 (n.s.) |
| **con controllo** | **4.0–6.3** | **5.6–6.3** | **2.5–2.7** (sig.) |

**L'effetto non si dissolve controllando per il confounder: si rafforza.** È un
risultato solido da mettere nel paper e nella Theory of Change.

---

## 4bis. Le figure del paper non hanno tutte lo stesso stile

Ricostruendo il rendering originale ho scoperto che **il set pubblicato non è
internamente coerente**. Sei figure su sette sono composte da LaTeX (backend PGF,
`text.usetex=True`, font Computer Modern). La settima,
`domain_metrics_comparison.pdf`, viene dal notebook 6, che imposta
`"text.usetex": False` e non passa mai al backend PGF: il suo testo è in
**DejaVu Serif**, non in Computer Modern.

Verificato leggendo i font incorporati nei PDF del paper:

| figura | font incorporati |
|---|---|
| le altre sei | `CMR10/CMR12/CMMI12/CMTI12` (Computer Modern) |
| `domain_metrics_comparison` | `DejaVuSerif` |

La pipeline nuova **riproduce anche questo**, per fedeltà: `fig_publisher_metrics`
ha `match_paper=True` di default e rende quella figura fuori dal backend PGF.
Passando `match_paper=False` viene composta come le altre sei, se e quando si
decide di uniformare il set.

Un secondo dettaglio nascosto: i rcParams delle figure venivano impostati in
celle lontane da quelle di plotting, quindi ogni figura ereditava dimensioni di
font diverse a seconda dell'ordine di esecuzione. Nel codice nuovo ogni figura
dichiara i propri, trascritti da quelli effettivamente in vigore al momento.

**Verifica di equivalenza.** Confrontando i PDF rigenerati con quelli del paper su
tre criteri (dimensioni fisiche della pagina, testo estratto, font incorporati):
**6 su 7 identiche su tutti e tre**. La settima differisce solo di 0,72 pt nel
bounding box (ritaglio `bbox_inches='tight'`), con testo e font identici.

## 4quater. Lo scambio RoBERTa/Llama colpisce ANCHE il supplementary

Rigenerando le matrici di confusione article-level ho trovato una **quinta
occorrenza** dello scambio, in figure che non avevo esaminato prima: nel
supplementary pubblicato, `cm_roberta.pdf` contiene i dati di **Llama** e
`cm_llama.pdf` quelli di **RoBERTa**.

Prova diretta dai CSV sorgente (fold 1, prima riga della matrice):

| | riga 1 fold 1 |
|---|---|
| `roberta_enriched.csv` | `[770, 33, 7, 86, 21]` |
| `llama_enriched.csv` | `[776, 22, 11, 90, 18]` |
| `cm_roberta.pdf` del paper | inizia con **776 22 11 90** -> e' Llama |
| `cm_llama.pdf` del paper | inizia con **770 33 7 86** -> e' RoBERTa |

**Queste due figure sono le uniche in cui NON riproduco la versione pubblicata**,
perche' riprodurla significherebbe riprodurre un errore di dati. Le versioni
rigenerate sono corrette: ogni file contiene il proprio modello. Di conseguenza
differiscono dal PDF vecchio di un carattere nel testo estratto, ed e' voluto.

---

## 4ter. Il supplementary: `text.usetex=False` era inerte

Ricostruendo anche le 14 figure del supplementary è emerso un equivoco che
spiega diverse incongruenze del vecchio codice.

`5 - Political Error Analysis.ipynb` imposta `matplotlib.use("pgf")` in cima e
poi, dentro la funzione delle matrici per political leaning, scrive
`"text.usetex": False`. **Quella riga non ha alcun effetto**: il backend PGF
compone il testo con LaTeX *comunque*, `text.usetex` è irrilevante sotto PGF. Lo
conferma il PDF pubblicato, che incorpora font Computer Modern
(`CMBX10/CMBX12/CMR10/CMR12/CMTI12`) — cioè LaTeX — nonostante quella riga.

Nel codice nuovo le matrici politiche restano quindi sotto PGF, mentre le
matrici publisher (notebook 6, che non passa mai a PGF) usano davvero mathtext,
con `mathtext.fontset='cm'` per riprodurre i `Cmmi10/Cmr10` del PDF pubblicato.

Un secondo scarto trovato solo confrontando i PDF: le `cm_*.pdf` pubblicate sono
**15×12 pollici**, mentre il codice sopravvissuto dice `figsize=(15, 10)`. Come
per `error_rate_analysis`, la versione pubblicata viene da codice più recente di
quello rimasto nei notebook.

---

## 5. Cosa ho fatto

**Riorganizzazione di `Results/`** — nulla è stato cancellato, tutto il vecchio è
in `Results/_archive/`:

```
Results/
├── figures/     11 figure rigenerate (output canonico)
├── tables/      12 tabelle CSV
├── derived/     predizioni per articolo e per publisher, formato tidy
└── _archive/    plots/ tables_old/ cm/ errors/ fullanalysis/  ← intatti
```

I risultati grezzi dei modelli non sono stati modificati, solo spostati da
`Classification/` a `Results/predictions/finetuned/` nella riorganizzazione del repo.

**Nuova pipeline** in `2-analysis/`:

```
2-analysis/
├── paper_results.ipynb   il notebook unico: eseguilo dall'alto in basso
├── build_notebook.py     lo genera (il notebook è un artefatto, il sorgente è questo)
├── FINDINGS.md           questo documento
└── irgood/
    ├── config.py     percorsi, registro modelli, etichette, colori
    ├── data.py       caricamento → una tabella tidy unica
    ├── metrics.py    ogni metrica definita una volta sola
    ├── aggregate.py  aggregazione publisher + curva di aggregazione
    ├── bias.py       analisi politica, test clusterizzati
    └── plots.py      una funzione per figura
```

La difesa contro il bug 1.1: il nome del modello è legato al suo file **una volta
sola**, in `config.FINETUNED_MODELS`, e viaggia insieme ai dati. Inoltre
`data.validate()` verifica che tutti i modelli vedano la stessa ground truth per
lo stesso articolo — il controllo che avrebbe intercettato subito lo scambio.

**Verifica.** La pipeline riproduce a 4 decimali i valori publisher-level del
notebook 6 (0.6651 / 0.6902 / 0.6651 / 0.6834), il conteggio delle parità
(13/20/15), il χ² article-level (1684.13 / 1418.31 / 1207.83 / 1343.98), i MED con
denominatore N, il baseline GPT (art. 0.245 / pub. 0.239) e le numerosità per
leaning (7490/6422/2407/2142/1015).

---

## 6. Da decidere (non l'ho fatto io)

1. **Correggere il paper sullo scambio RoBERTa/Llama** — due frasi nei risultati.
2. **Scegliere una definizione di MED** e usarla ovunque.
3. **Dichiarare il tie-break** oppure cambiarlo (su scala ordinale la mediana dei
   modi tied sarebbe più naturale di "il più basso").
4. **`Non labelled` è trattato come una posizione politica**: è il segnaposto per i
   publisher che NewsGuard non ha classificato, ma sta in mezzo alla scala
   sinistra–destra ed entra nel chi-quadro come quinta categoria. È il secondo
   gruppo per dimensione (6.422 articoli) e ha l'error rate più basso, quindi
   guida sia la statistica sia il gradiente visivo. Va deciso esplicitamente.
5. **Se rigenerare le figure del paper** con lo stile nuovo: il notebook ha
   `PUBLISH = False`, mettilo a `True` per sovrascrivere `Paper/NewIRForGood/img/`.

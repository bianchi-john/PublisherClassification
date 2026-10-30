# Guida al repo (per chi arriva, anche un'altra istanza di Claude)

**Il problema.** Prevedere l'affidabilità di una testata giornalistica online (il punteggio NewsGuard 0–100, o i 5 livelli
l1–l5, o affidabile sì/no = punteggio ≥ 60) **solo dal testo dei suoi articoli**, su testate mai viste in addestramento.
La domanda del lavoro attuale (Web Science 2027): un transformer fa meglio di un classificatore TF-IDF?

Codice, notebook, commenti e README dei tentativi 01–19 sono **in italiano**. **Dal tentativo 20 tutto è in inglese**
(codice, commenti, notebook, segnaposto dell'anonimizzazione), per scelta di John.

## Dove guardare

```
Code/WebScience2027/                il lavoro attuale
  README.md                         il problema, come valutiamo, la storia dei tentativi: leggerlo per primo
  Tentativi/NN_nome/                un esperimento per cartella, in ordine di tempo (01 … 20), ognuno col suo README
  crea_archivio_dataset.py          crea Dataset.zip (i dati del protocollo attuale, < 1 GB)
Code/IrForGood/, Code/RejectedTCSS/ versioni precedenti del paper (con i loro dati in Code/<versione>/Dataset/)
Dataset/                            i dati di WebScience2027: fuori da git, vedi Dataset/LEGGIMI.md
Modelli/                            i modelli addestrati: fuori da git
Paper/, Presentazioni/              paper e slide
TODO.md, UTILI.md                   appunti di John (fuori da git)
```

**Il punto d'arrivo è `Tentativi/20_FromScratch`**: tutta la pipeline rifatta da zero, in ordine e in inglese
(scraping → pulizia → anonimizzazione → split → classificazione e regressione → valutazione → risultati), un notebook per
passo (01–09, generati da `build_notebooks.py`), tutto riproducibile; i risultati in `09_results.ipynb`. Riprende le
regole del 16 (dataset), l'anonimizzazione del 19 e lo split per voto del 18.

Il **tentativo 16** (`Tentativi/16_DataPreparation`) è il dataset su cui si basa: regole semplici, 226 testate politiche,
25.524 articoli, NewsGuard 2026.
- **Notebook 01–07:** sono salvati già eseguiti, con gli output.
- **`06_confronto.ipynb`:** la pagina dei risultati.
- **`07_analisi_embedding.ipynb`:** le varianti con gli embedding.
- **Risultato attuale:**
  - Qwen3-Embedding-8B congelato + un modello lineare batte il TF-IDF: F1 sui 5 livelli fino a 0,44 contro 0,33–0,35;
  - ModernBERT addestrato lo eguaglia soltanto: impara le testate di training e non generalizza.

I tentativi 17–19 sono l'accordo tra annotatori, il punteggio esatto e la prima versione anonima; 14 e 15 il protocollo
precedente e le analisi del dataset; 01–13 la storia (il README di WebScience2027 riassume che cosa ha detto ognuno).

## I dati

Tutto sta in `Dataset/` nella radice del repo, fuori da git (struttura in `Dataset/LEGGIMI.md`):
- `scraping/`: gli scraping grezzi;
- `etichette/`: NewsGuard 2022 e 2026;
- `risorse/`: i lessici NRC;
- `tentativi/<tentativo>/`: quello che produce ogni tentativo, cioè dati intermedi e dataset finali.

Per lavorare sul protocollo attuale basta **`Dataset.zip`** (305 MB), da estrarre nella radice del repo (`unzip
Dataset.zip`). Bastano le etichette, lo stato dello scraping, la cache dei testi estratti e i dataset finali di 14 e 16.
Con questo si rifà il 16 per intero e si rifanno 14 (classificazione) e 15. La cartella completa (22 GB) è sul server.

Come il codice trova i dati:
- **20:** `common.py` trova `Dataset/` allo stesso modo e scrive in `Dataset/tentativi/20_FromScratch/`.
- **14–16:** `funzioni.cartella_dati()` sale dalla cartella di lavoro fino a una `Dataset/` con `LEGGIMI.md`. Il notebook usa `F.DATI` (la cartella) e `F.DATASET` (la parte del tentativo).
- **01–13:** usano percorsi relativi alla radice del repo (`../../../../../Dataset/...`, dalla sottocartella da cui gira il notebook).

**Dati riservati.** Le etichette NewsGuard, MBFC e i lessici NRC non si ridistribuiscono e non vanno in git, nemmeno
dentro i risultati. Per questo `previsioni_testate.csv` (che contiene il punteggio vero) è in `.gitignore`.

## Come si lavora

- **Un tentativo è autonomo.** Ha il suo README (domanda, disegno fissato prima dei risultati, risultati), i suoi notebook, `funzioni.py` e `Results/`.
- **Nel 16 i notebook 03–07 sono generati da `costruisci_notebook.py`.** Il builder non tocca un notebook se il suo codice non cambia, quindi gli output restano. Se il codice cambia, il notebook va rieseguito.
- **Il calcolo pesante gira sul server artes4** (GPU V100 32 GB, condivisa con altri):
  - venv `~/John/johnVenv`, kernel Jupyter `johnvenv`;
  - i tentativi stanno in `~/John/TentativoNN`, i dati in `~/John/Dataset`;
  - si lancia in `screen` con papermill (`esegui_tutto.sh`); dal PC si segue con `stato.sh` e si scarica con `scarica.sh`;
  - prima di usare la GPU si controlla la memoria libera, un lavoro alla volta;
  - non si toccano gli screen degli altri.
- **Sul PC** il kernel è `irgood-venv`.
- **La valutazione è per testata**:
  - il voto o la media degli articoli dà la previsione della testata;
  - 5 fold di testate stratificati per livello, con una testata per gruppo di proprietà (Owner/Parent Domain di NewsGuard);
  - intervalli al 95% con il bootstrap sulle testate;
  - le baseline vanno sempre mostrate: per affidabile sì/no, «sempre affidabile» vale 0,65.
- **Prudenza con i dati.** Non si cancella niente in modo definitivo: si sposta in una cartella `_da_cancellare` e decide John. Non si fanno commit o push senza che lo chieda.

# Guida al repo (per chi arriva, anche un'altra istanza di Claude)

**Il problema.** Prevedere l'affidabilità di una testata giornalistica online (il punteggio NewsGuard 0–100) **solo dal
testo dei suoi articoli**, su testate mai viste in addestramento, con il testo anonimizzato. La domanda: i grandi modelli
linguistici usati come encoder congelati fanno meglio di un modello TF-IDF? Il prossimo invio sarà a una rivista (non
ancora scelta, probabilmente non Web Science): la bozza è in `Paper/Journal/`.

Il codice del paper (`Code/Pipeline`) è **in inglese**; i tentativi che ci hanno portato (`Code/Attempts`, 01–19) sono in
italiano.

## Dove guardare

```
Code/Pipeline/                      la pipeline del paper: un notebook per passo (01–12), generati da build_notebooks.py
  README.md                         passi, regole, risultati: leggerlo per primo
Code/Attempts/                      i 19 tentativi di ottobre-novembre 2026 (in italiano), uno per cartella in Tentativi/
  README.md                         che cosa ha detto ogni tentativo
Code/IrForGood/, Code/RejectedTCSS/ versioni precedenti del paper (con i loro dati in Code/<versione>/Dataset/)
Paper/Journal/                      bozza LaTeX del paper per rivista (template generico; risultati da riempire)
Dataset/                            tutti i dati: fuori da git, vedi Dataset/LEGGIMI.md
Modelli/                            i modelli addestrati: fuori da git
Presentazioni/                      slide (la più recente: Stato finale/FinalState.odp)
TODO.md, UTILI.md                   appunti di John (fuori da git)
```

**La pipeline** (`Code/Pipeline`, nata come tentativo 20 «from scratch»):
1. scraping (ottobre 2026, HTML grezzo);
2. pulizia: testo dall'HTML con trafilatura in modalità precisione; solo testate politiche (unico argomento NewsGuard
   «Political news or commentary»); inglese; 100–1.500 parole; dal 2025; doppioni tolti tra tutte le testate; almeno 10
   articoli; una testata per gruppo di proprietà → 226 testate, 25.524 articoli;
3. anonimizzazione: link ed e-mail, il nome della testata dal suo dominio, e con GLiNER (NER zero-shot) tutte le testate
   e tutti i giornalisti → segnaposto;
4. split: 5 fold di testate stratificati sul voto esatto;
5. embedding congelati (Qwen3-Embedding-8B, Mistral-Small-24B strato 27);
7. regressione del voto: TF-IDF, Qwen, Mistral (+ ridge) e ModernBERT addestrato; voto della testata = media degli
   articoli;
8–9. valutazione per testata (Spearman, Kendall, MAE, entro 20 punti; bootstrap); 12: confronto con un'anonimizzazione
   più semplice (solo regole, prima esecuzione, conservata in `Dataset/tentativi/20_FromScratch/rules_only/`).
I passi 6 (classificazione), 10 (altri regressori sugli embedding) e 11 (meta-articoli) sono analisi laterali, non nel
paper. Lo stiramento dei voti e il confronto con MBFC non vanno nel paper.

I dati della pipeline restano in `Dataset/tentativi/20_FromScratch/` (nome storico, uguale sul server).

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

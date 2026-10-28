# Tentativo 16: data preparation

`01_data_preparation.ipynb`: dai dati grezzi al dataset, con regole semplici fissate prima dei risultati.
Le funzioni stanno in `funzioni.py`, quelle del tentativo 14 più la riestrazione dall'HTML (sezione 9). I dati sono in
`Dataset/` nella radice del repo (fuori da git): lo scraping in `Dataset/scraping/`, NewsGuard 2026 in `Dataset/etichette/`,
il dataset finale (`articoli.csv.gz`, `testate.csv`, la cache di trafilatura) in `Dataset/tentativi/16_DataPreparation/`.
`funzioni.py` trova la cartella da solo (`F.DATI`, `F.DATASET`); sul server sta in `~/John/Dataset`.

**Le regole**
1. Solo testate **politiche**: per NewsGuard 2026 hanno un solo argomento, "Political news or commentary", e un
   punteggio. Solo lo **scraping di ottobre 2026**.
2. Testo **riestratto dalla pagina HTML salvata con trafilatura in modalità precisione**: `favor_precision=True`, senza
   commenti, tabelle, immagini, link e formattazione. Nessun'altra regola di pulizia scritta da noi.
3. Solo inglese, **tra 100 e 1.500 parole** (ogni articolo entra intero in 2.048 token di ModernBERT), pubblicati dal
   2025.
4. **Doppioni e quasi doppioni** (Jaccard dei 5-grammi ≥ 0,5) tolti tra **tutte** le testate: si tiene la copia più
   vecchia.
5. Almeno 10 articoli per testata.
6. **Una sola testata per gruppo** di testate parenti secondo NewsGuard 2026 (stesso Owner o stesso Parent Domain): si
   tiene quella con più articoli.
7. Su ogni articolo il **punteggio NewsGuard 2026** (0–100) e il livello; 5 fold di testate stratificati per livello.

**Il dataset**: **226 testate, 25.524 articoli** (l1 35, l2 43, l3 30, l4 79, l5 39).

**Da sapere**
- I doppioni esatti tra testate fanno sparire intere reti di siti che ripubblicano gli stessi articoli: da 532 a 363
  testate al passo 4.1.
- La modalità precisione toglie in media solo il 3,8% delle parole. Il 7,2% degli articoli perde più di metà del
  testo, senza controlli a mano.
- La sporcizia residua cala di poco: articoli con inviti all'iscrizione o newsletter dal 14,5% al 12,6%, rimandi
  (*Also read*…) dal 4,8% al 3,7%, crediti foto dall'1,1% allo 0,7%. È misurata nella sezione 8 e non tolta.
- Controllo: 58 coppie di testate rimaste si erano scambiate almeno un articolo quasi uguale. Le copie sono state tolte,
  le testate restano. Spesso sono articoli di agenzia ripresi da testate indipendenti.

Sul server, in `~/John/Tentativo16`, circa 6 minuti:
`python -m papermill -k johnvenv 01_data_preparation.ipynb 01_data_preparation.ipynb`.

## I notebook

Sono salvati **già eseguiti**, con tutti gli output. Il 4 ha gli output degli embedding 8B; quelli dello 0,6B sono in
`Results/04_embedding_0.6B_eseguito.ipynb`.

1. `01_data_preparation.ipynb`: il dataset.
2. `02_data_analysis.ipynb`: lo stile sul piano (ragionamento ed emotività, l1–l2 contro l4–l5).
3. `03_feature.ipynb`: il classificatore a feature (TF-IDF) in sei configurazioni (5 classi, binario, binario sugli
   estremi, regressione, regressione bilanciata, regressione sugli estremi), con il grafico riassuntivo.
4. `04_embedding.ipynb`: embedding di Qwen3-Embedding (0,6B e 8B, senza addestramento) più gli stessi modelli lineari.
5. `05_modernbert.ipynb`: ModernBERT a 2.048 token, classificatore a 5 classi e regressore. Prima una diagnostica
   su un fold (6 epoche, curve di training, validazione e test); poi i 5 fold con **5 epoche fisse**, learning rate
   3e-5, batch 32, su tutte le testate di training (la validazione, circa 27 testate, era troppo rumorosa per scegliere
   l'epoca).
6. `06_confronto.ipynb`: **la pagina dei risultati**. TF-IDF, embedding e ModernBERT in un solo grafico (5 classi e
   regressione; F1, tolerant, binario), lo stesso grafico con i dettagli (che cosa è ogni modello, su quali testate si
   misura, le baseline), le differenze dal TF-IDF con i loro intervalli, e la loss di ModernBERT epoca per epoca
   accanto al test.

7. `07_analisi_embedding.ipynb`: come migliorare l'embedding 8B congelato senza toccarlo: il parametro del modello
   lineare scelto con una cross-validation interna al training, un vettore per testata (anche con la regressione
   ordinale), TF-IDF + embedding, e un'altra istruzione (sullo stile). Si lancia con `esegui_07.sh`, che prima calcola
   gli embedding con la nuova istruzione (circa 2 ore di GPU).

Le parti comuni sono in `funzioni.py`, sezioni 10–15. `costruisci_notebook.py` rigenera i notebook 3–6.
`esegui_tutto.sh` lancia tutto in fila sul server: 1–3 in circa 15 minuti, embedding 0,6B circa 15 minuti, 8B circa
2 ore, ModernBERT circa 13 ore (2 configurazioni × 5 fold × 5 epoche, circa 15 minuti a epoca su una V100).

## Che cosa viene fuori

| | F1 5 livelli | tolerant ±1 | affidabile sì/no, tutte | affidabile sì/no, l1–l2 contro l4–l5 |
|---|---|---|---|---|
| baseline | 0,20 (a caso) | 0,65 (sempre l4) | 0,65 (sempre «affidabile») | 0,60 |
| TF-IDF, regressione | 0,35 | 0,82 | **0,81** | 0,81 |
| Embedding Qwen3 8B congelato, 5 classi | **0,40** | 0,83 | 0,79 | 0,83 |
| Embedding Qwen3 8B congelato, regressione | 0,31 | **0,87** | 0,80 | **0,84** |
| ModernBERT addestrato, regressione | 0,33 | 0,82 | 0,79 | 0,80 |

- L'**embedding 8B congelato** è il primo modello sopra il TF-IDF: F1 +0,07 sul TF-IDF a 5 classi (intervallo al 95%
  da −0,00 a +0,14), tolerant e binario sugli estremi migliori. Con 226 testate la differenza è al limite.
- **ModernBERT** è pari al TF-IDF (regressione) o sotto (5 classi). Non è sottoallenato: dopo 5 epoche la loss di
  training è quasi il minimo possibile (0,43 contro 0,42 con la label smoothing) e l'errore sul training è di 4 punti;
  sul test la loss è 2,1 (a caso 1,63) e l'errore 25 punti. Impara le testate di training, non un segnale che vale per
  testate nuove.
- La regressione sul punteggio va meglio della classificazione a 5 classi per TF-IDF e ModernBERT.

## Lanciare e controllare

- **Sul server** (va avanti anche se ci si disconnette): `cd ~/John/Tentativo16 && screen -dmS t16 ./esegui_tutto.sh`.
  Prima di embedding e ModernBERT c'è una prova veloce (2 articoli per testata, 1 epoca): se qualcosa non va, la
  catena si ferma subito.
- **Dal PC**, nella cartella del tentativo:
  - `./stato.sh` dice a che punto è: passi fatti, notebook in corso, ultime epoche di ModernBERT, errori, GPU e disco;
  - `./scarica.sh` porta qui i notebook eseguiti (al posto di quelli del PC), i grafici e le tabelle.

  Chiedono la password del server una volta.

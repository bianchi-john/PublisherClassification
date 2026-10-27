# Tentativo 16: data preparation

`01_data_preparation.ipynb`: dai dati grezzi al dataset, con regole semplici fissate prima dei risultati.
Le funzioni stanno in `funzioni.py`, quelle del tentativo 14 più la riestrazione dall'HTML (sezione 9). I dati sono in
`Dataset/`, fuori da git: sul PC sono link al tentativo 12, sul server sono hardlink.

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

**Il dataset**: **233 testate, 29.524 articoli** (l1 38, l2 42, l3 31, l4 83, l5 39).

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
`python -m papermill -k johnvenv 01_data_preparation.ipynb Results/01_data_preparation_eseguito.ipynb`.

## I notebook

1. `01_data_preparation.ipynb`: il dataset.
2. `02_data_analysis.ipynb`: lo stile sul piano (ragionamento ed emotività, l1–l2 contro l4–l5).
3. `03_feature.ipynb`: il classificatore a feature (TF-IDF) in sei configurazioni (5 classi, binario, binario sugli
   estremi, regressione, regressione bilanciata, regressione sugli estremi), con il grafico riassuntivo.
4. `04_embedding.ipynb`: embedding di Qwen3-Embedding (0,6B e 8B, senza addestramento) più gli stessi modelli lineari.
5. `05_modernbert.ipynb`: ModernBERT a 2.048 token, classificatore a 5 classi e regressore. Prima una diagnostica
   su un fold (6 epoche, curve di training, validazione e test); poi i 5 fold, al più 5 epoche, learning rate 3e-5,
   scelta dell'epoca sulla validazione.
6. `06_confronto.ipynb`: TF-IDF, embedding e ModernBERT in un solo grafico (5 classi e regressione; F1, tolerant,
   binario), con le differenze dal TF-IDF e i loro intervalli.

Le parti comuni sono in `funzioni.py`, sezioni 10–12. `costruisci_notebook.py` rigenera i notebook 3–6.
`esegui_tutto.sh` lancia tutto in fila sul server: 1–3 in circa 15 minuti, embedding 0,6B circa 15 minuti, 8B circa
2–3 ore, ModernBERT circa 9 ore.

## Lanciare e controllare

- **Sul server** (va avanti anche se ci si disconnette): `cd ~/John/Tentativo16 && screen -dmS t16 ./esegui_tutto.sh`.
  Prima di embedding e ModernBERT c'è una prova veloce (2 articoli per testata, 1 epoca): se qualcosa non va, la
  catena si ferma subito.
- **Dal PC**, nella cartella del tentativo:
  - `./stato.sh` dice a che punto è: passi fatti, notebook in corso, ultime epoche di ModernBERT, errori, GPU e disco;
  - `./scarica.sh` porta qui i notebook eseguiti, i grafici e le tabelle.

  Chiedono la password del server una volta.

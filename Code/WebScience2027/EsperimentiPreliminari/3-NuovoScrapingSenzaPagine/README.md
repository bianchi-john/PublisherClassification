# 3 — Nuovo scraping senza pagine, pulizia moderata (193 testate)

Il lavoro del 2026-10-16/17: solo il testo degli articoli del nuovo scraping, pulizia "rigorosa ma non
eccessiva", reti di testate parenti da cinque segnali ciechi (copie e autori sistematici, codici di account
nell'HTML, titolare legale, dominio) uniti all'owner NewsGuard, al massimo 5 testate per gruppo. Il processo è
replicabile bit per bit (verificato rieseguendo tutto da zero e confrontando gli md5 dei file).

| | |
|---|---|
| `1-data-cleaning/` | `01_pulizia_articoli`, `02_reti`, `03_dataset_finale` (29.129 articoli, 193 testate, 143 gruppi) |
| `2-classificazione/` | `01_classificatore_feature`, `02_transformer` (ricetta: 20 articoli per testata per epoca, pesi di classe, warmup, early stopping sulla F1 per testata), `03_confronto`, `coda_weekend.py` |
| `Results/` | notebook eseguiti, metriche e predizioni (le predizioni complete stanno sul server) |
| `Dataset/` | sul server: `versioni/` e `derived/` di questa versione |

## Risultati del weekend (testata, fold per gruppo)

| | F1 macro | binario |
|---|---|---|
| TF-IDF (voto) | 0,274 | 0,653 |
| ModernBERT-base 1024, media di 3 seed | 0,255 | 0,634 |
| ModernBERT-base 2048, media di 3 seed | 0,242 | 0,625 |
| ModernBERT-large 1024, media di 3 seed | 0,256 | 0,615 |
| maggioranza | 0,097 | 0,606 |

Nessuna differenza significativa con il feature (gli intervalli bootstrap per gruppo contengono lo zero); la
varianza tra seed è grande quanto le differenze tra modelli. DeBERTa-v3 è fallito tre volte allo stesso punto
(dopo l'early stopping del primo fold). Messo da parte per ripartire con più dati: il dataset vecchio di
IrForGood unito a questo, con la stessa pulizia.

# Web Science 2027

Linea di lavoro verso Web Science 2027. Riparte da un **nuovo scraping**, con più testate e con le
pagine del sito (about, proprietà, correzioni…), e confronta due input: **solo articoli** contro
**articoli + pagine del sito**.

```
0-scraping/              scraper + costruzione del dataset grezzo (articoli e pagine del sito)
1-data-cleaning/         testate parenti (gruppi) e costruzione del dataset definitivo (272 testate)
2-classificazione/       classificazione: A = solo articoli, C = articoli + pagine
Dataset/                 dati (fuori da git)
Results/                 risultati (fuori da git)
EsperimentiPreliminari/  il lavoro sul primo dataset, messo da parte
```

## La pipeline

| | |
|---|---|
| `0-scraping/scraper.py` + `costruisci_dataset.ipynb` | `Dataset/NuovoScraping/dataset_articoli.csv` e `dataset_pagine_sito.csv` |
| `1-data-cleaning/01_testate_parenti.ipynb` | gruppi di testate parenti (MinHash + verifica con l'owner NewsGuard) |
| `1-data-cleaning/02_dataset_finale.ipynb` | `Dataset/versioni/dataset_finale.csv` (272 testate, ≥20 articoli e una pagina about), `pagine_finale.csv`, `fold_finale.csv` |
| `2-classificazione/01_classificatore_feature.py` | TF-IDF + regressione logistica, input A e C |
| `2-classificazione/02_transformer.py` | ModernBERT (modello finale) o BERT, input A e C |

Valutazione **per gruppo** (onesta: nessuna testata sorella nel training) e per testata; lettura per
articolo e per testata (voto di maggioranza). Si salvano le probabilità per classe e la loss.

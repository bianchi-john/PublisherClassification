# Web Science 2027

Linea di lavoro verso Web Science 2027: predire l'affidabilità NewsGuard di una testata **dal solo testo dei
suoi articoli**, valutando su testate di **reti mai viste** in addestramento.

```
0-scraping/              scraper + costruzione del dataset grezzo di articoli
1-data-cleaning/         pulizia degli articoli, reti di testate parenti, dataset finale e fold
2-classificazione/       classificazione (per ora TF-IDF + regressione logistica)
Dataset/                 dati (fuori da git)
Results/                 risultati e notebook eseguiti
EsperimentiPreliminari/  i tentativi precedenti (dataset di IrForGood; nuovo scraping con le pagine del sito)
stato.sh                 cosa gira sul server e da quanto
```

## La pipeline

| | |
|---|---|
| `0-scraping/scraper.py` + `costruisci_dataset.ipynb` | `Dataset/NuovoScraping/dataset_articoli.csv` (+ l'HTML grezzo, sul server) |
| `1-data-cleaning/01_pulizia_articoli.ipynb` | articoli puliti: senza copie tra testate, recenti, senza testo ripetuto né cornice della pagina |
| `1-data-cleaning/02_reti.ipynb` | reti di testate parenti da testo, HTML (codici di account, titolare legale), dominio e autori |
| `1-data-cleaning/03_dataset_finale.ipynb` | `Dataset/versioni/dataset_finale.csv`, `fold_finale.csv` |
| `2-classificazione/01_classificatore_feature.ipynb` | TF-IDF + regressione logistica, con le baseline |
| `2-classificazione/02_transformer.ipynb`, `04_embedding.ipynb` | ModernBERT (fine-tuning) ed embedding congelati |
| `2-classificazione/03_confronto.ipynb` | il confronto, con gli intervalli bootstrap per gruppo |

Valutazione **per gruppo** (principale: le testate di una rete o dello stesso owner stanno tutte nello stesso fold) e
per testata (confronto); lettura per articolo e per testata.

## Rieseguire tutto

Sul server (serve la GPU e, per `02_reti`, l'HTML grezzo in `Dataset/NuovoScraping/html/`), con la cartella
`Code/WebScience2027` copiata in `~/John/WebScience2027` e il venv `~/John/johnVenv` (kernel Jupyter `johnvenv`):

```bash
cd ~/John/WebScience2027
screen -dmS tutto bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'
./stato.sh                       # a che punto è
```

`esegui_tutto.py` fa scraping → `costruisci_dataset` → pulizia (01–03) → classificazione (`coda_weekend.py`),
con tempi massimi, tentativi e ripresa da dove si era fermato. Dal PC: `./controlla_server.sh`.

## Risultati (testata, fold per gruppo, 304 testate)

| | F1 macro | binario |
|---|---|---|
| TF-IDF + LR (C in validazione) | 0,380 | 0,740 |
| ModernBERT-base, ensemble di 3 seed | 0,366 | 0,717 |
| ModernBERT-large, ensemble di 3 seed | 0,348 | 0,743 |
| embedding congelati (e5 / bge) | 0,331 / 0,325 | 0,727 / 0,730 |
| classe di maggioranza | 0,096 | 0,553 |

Nessuna differenza significativa tra feature e transformer (bootstrap per gruppo). Dettagli in
`Results/2-classificazione/03_confronto_*.csv` e nella presentazione `Riunioni/Stato dei lavori/`.

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

Valutazione **per rete** (principale: le testate di una rete stanno tutte nello stesso fold) e per testata
(confronto); lettura per articolo e per testata (voto di maggioranza).

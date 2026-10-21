# Tentativo 05 — Versione 1: la pipeline completa (304 testate)

> **Tentativo autonomo**: `0-scraping` → `1-data-cleaning` → `2-classificazione`, dati in `Dataset/` (fuori da git),
> risultati in `Results/`. Qui la catena è stata davvero eseguita tutta, scraping compreso (`esegui_tutto.py`, 18
> ottobre). I dati grezzi dello scraping (`Dataset/NuovoScraping/`) sono quelli copiati poi nei tentativi 06–08. Il
> percorso di tutti i tentativi è nel README di `Code/WebScience2027`.

**La domanda.** Dopo i tentativi 01–04: con uno scraping solo ma allargato, una pulizia rigorosa ma non eccessiva, le
reti di testate parenti trovate senza NewsGuard e la valutazione per gruppo, un transformer batte il TF-IDF?

## La catena

| | |
|---|---|
| `0-scraping/scraper.py` + `costruisci_dataset.ipynb` | due giri, 13 e 18 ottobre (argomento NewsGuard *esatto* "Political news or commentary", poi *che contiene*): 3.128 testate tentate, 1.528 raggiunte → `Dataset/NuovoScraping/dataset_articoli.csv` (+ l'HTML grezzo) |
| `1-data-cleaning/01_pulizia_articoli.ipynb` | articoli puliti: senza copie tra testate, recenti, senza testo ripetuto né cornice della pagina |
| `1-data-cleaning/02_reti.ipynb` | reti di testate parenti da testo, HTML (codici di account, titolare legale), dominio e autori |
| `1-data-cleaning/03_dataset_finale.ipynb` | gruppi = reti ∪ owner; al più 5 testate per gruppo, almeno 20 articoli → `Dataset/versioni/dataset_finale.csv`, `fold_finale.csv` |
| `2-classificazione/01_classificatore_feature.ipynb` | TF-IDF + regressione logistica, con le baseline |
| `2-classificazione/02_transformer.ipynb`, `04_embedding.ipynb` | ModernBERT (fine-tuning) ed embedding congelati |
| `2-classificazione/03_confronto.ipynb` | il confronto, con gli intervalli bootstrap per gruppo |

Valutazione **per testata**, fold **per gruppo** (le testate di una rete o dello stesso owner tutte nello stesso fold),
304 testate, 44.029 articoli.

## Risultati

| | F1 macro | binario |
|---|---|---|
| TF-IDF + LR (C in validazione) | 0,380 | 0,740 |
| ModernBERT-base, ensemble di 3 seed | 0,366 | 0,717 |
| ModernBERT-large, ensemble di 3 seed | 0,348 | 0,743 |
| embedding congelati (e5 / bge) | 0,331 / 0,325 | 0,727 / 0,730 |
| classe di maggioranza | 0,096 | 0,553 |

Nessuna differenza significativa tra feature e transformer (bootstrap per gruppo). Dettagli in
`Results/2-classificazione/03_confronto_*.csv`.

## Rieseguire

Sul server, con questa cartella in `~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate` (con
`Dataset/newsguard.csv`):

```bash
cd ~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate
screen -dmS tutto bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'
./stato.sh                       # a che punto è; dal PC: ./controlla_server.sh
```

`esegui_tutto.py` fa scraping → `costruisci_dataset` → pulizia (01–03) → classificazione (`coda_weekend.py`), con
tempi massimi, tentativi e ripresa da dove si era fermato.

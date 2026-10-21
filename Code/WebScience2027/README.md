# Web Science 2027

Linea di lavoro verso Web Science 2027: predire l'affidabilità NewsGuard di una testata **dal solo testo dei
suoi articoli**, valutando su testate di **reti mai viste** in addestramento. La domanda di fondo: un transformer fa
meglio di un classificatore a feature (TF-IDF + regressione logistica)?

## Il percorso in breve

| Fase | Cartella | La domanda | Il risultato |
|---|---|---|---|
| Tentativi preliminari (13–18 ottobre) | `EsperimentiPreliminari/1–4` | primo dataset, pagine del sito, due scraping uniti… | le **reti** di siti gonfiano i numeri (0,68 → 0,39 con i fold per gruppo); ogni tentativo ha lasciato una regola |
| **Versione 1** (18 ottobre) | questa cartella: `0-scraping`, `1-data-cleaning`, `2-classificazione` | con scraping nuovo, pulizia, reti e fold per gruppo, chi vince? | **pari**: TF-IDF F1 0,38, ModernBERT 0,35–0,37 |
| **Versione 2** (19–20 ottobre) | `Versione2/` | più pulizia e più testate nel training aiutano i transformer? | **no**; l'analisi del TF-IDF (`05_analisi_feature`) mostra che legge il **genere** della testata (orientamento, tipo di sito, paese) |
| **Versione 3** (20–21 ottobre) | `Versione3/` | a parità di genere, il testo dice l'affidabilità? | sì, ma poco (0,6 contro 0,5) ed è **stile**; il TF-IDF resta davanti |

Ogni cartella ha il suo README con il disegno (fissato prima dei risultati), i risultati e come rilanciarla. La
storia completa, con i grafici, è nella presentazione `Riunioni/Stato dei lavori/stato_dei_lavori.pdf`.

```
0-scraping/              scraper + costruzione del dataset grezzo di articoli (comune a tutte le versioni)
1-data-cleaning/         Versione 1: pulizia, reti di testate parenti, dataset finale e fold
2-classificazione/       Versione 1: TF-IDF, ModernBERT, embedding congelati, confronto
Results/                 Versione 1: risultati e notebook eseguiti
Versione2/               più pulizia, training allargato, analisi del TF-IDF
Versione3/               a parità di genere: affidabile sì/no dentro ogni orientamento politico
EsperimentiPreliminari/  i tentativi prima della Versione 1
Dataset/                 dati (fuori da git): lo scraping (NuovoScraping/, con l'HTML grezzo), newsguard.csv,
                         derived/ e versioni/ della Versione 1
esegui_tutto.py, stato.sh, controlla_server.sh   catena e controlli della Versione 1 (Versione2/ e Versione3/ hanno i loro)
```

## La Versione 1, in dettaglio

| | |
|---|---|
| `0-scraping/scraper.py` + `costruisci_dataset.ipynb` | `Dataset/NuovoScraping/dataset_articoli.csv` (+ l'HTML grezzo) |
| `1-data-cleaning/01_pulizia_articoli.ipynb` | articoli puliti: senza copie tra testate, recenti, senza testo ripetuto né cornice della pagina |
| `1-data-cleaning/02_reti.ipynb` | reti di testate parenti da testo, HTML (codici di account, titolare legale), dominio e autori |
| `1-data-cleaning/03_dataset_finale.ipynb` | `Dataset/versioni/dataset_finale.csv`, `fold_finale.csv` |
| `2-classificazione/01_classificatore_feature.ipynb` | TF-IDF + regressione logistica, con le baseline |
| `2-classificazione/02_transformer.ipynb`, `04_embedding.ipynb` | ModernBERT (fine-tuning) ed embedding congelati |
| `2-classificazione/03_confronto.ipynb` | il confronto, con gli intervalli bootstrap per gruppo |

Valutazione **per testata**, fold **per gruppo** (le testate di una rete o dello stesso owner stanno tutte nello
stesso fold; al più 5 testate per gruppo), 304 testate.

| Versione 1 | F1 macro | binario |
|---|---|---|
| TF-IDF + LR (C in validazione) | 0,380 | 0,740 |
| ModernBERT-base, ensemble di 3 seed | 0,366 | 0,717 |
| ModernBERT-large, ensemble di 3 seed | 0,348 | 0,743 |
| embedding congelati (e5 / bge) | 0,331 / 0,325 | 0,727 / 0,730 |
| classe di maggioranza | 0,096 | 0,553 |

Nessuna differenza significativa tra feature e transformer (bootstrap per gruppo). Dettagli in
`Results/2-classificazione/03_confronto_*.csv`.

## Rieseguire

Sul server, con questa cartella copiata in `~/John/WebScience2027` e il venv `~/John/johnVenv` (kernel Jupyter
`johnvenv`). La Versione 1 intera (scraping compreso):

```bash
cd ~/John/WebScience2027
screen -dmS tutto bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'
./stato.sh                       # a che punto è; dal PC: ./controlla_server.sh
```

`esegui_tutto.py` fa scraping → `costruisci_dataset` → pulizia (01–03) → classificazione (`coda_weekend.py`), con
tempi massimi, tentativi e ripresa da dove si era fermato. Le Versioni 2 e 3 si lanciano dalle loro cartelle (vedi i
loro README); la 3 usa il dataset della 2.

Il server è condiviso: di solito ci sono circa 7 GB di GPU liberi, quindi i transformer girano con batch piccoli e
accumulo del gradiente, e ModernBERT-large non entra. Su artes4 NVML è rotto (driver/library mismatch): la memoria
finita arriva come `RuntimeError ... NVML ... CUDACachingAllocator`, e i notebook della Versione 2 e 3 la riconoscono.

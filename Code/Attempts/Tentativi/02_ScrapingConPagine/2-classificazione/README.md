# 2 — Classificazione

Sul **dataset definitivo** (`Dataset/versioni/dataset_finale.csv`, 272 testate con ≥20 articoli e una
pagina about utile). Due input, due letture della stessa predizione.

- **A** — solo testo degli articoli.
- **C** — testo dell'articolo **+** le pagine del sito della testata (about, ownership, correzioni…) in coda.
- Lettura **per articolo** e **per testata** (voto di maggioranza).
- Fold **per gruppo** (principale, onesto) e **per testata** (confronto), stratificati per livello.

| Notebook | |
|---|---|
| `01_classificatore_feature.ipynb` | TF-IDF + regressione logistica; salva le probabilità per classe |
| `02_transformer.ipynb` | BERT (confronto) o ModernBERT (finale); salva probabilità e loss per epoca |
| `03_cosa_impara_il_feature.ipynb` | perché il feature batte il transformer, e quali leve provare nel rilancio |
| `comune.py` | caricamento dei dati, percorsi e letture (5 classi, F1, binario, tolerant) comuni ai notebook |

Tutto viene salvato (probabilità p_l1..p_l5 per articolo, loss per epoca) per poter rigiocare
aggregazione, soglie e fusione in un secondo momento, senza riaddestrare. Se i risultati ci sono già, i
notebook li ricaricano invece di riaddestrare. Ognuno ha una cella di parametri (`PROVA = True` gira su 40
testate per controllare il codice).

Sul server si lanciano con papermill dentro `screen` (comando nella prima cella di ogni notebook);
il notebook eseguito finisce in `../Results/2-classificazione/esecuzioni/`.
Log: `../Results/2-classificazione/avanzamento_<modello>.log`. Cosa gira e da quanto: `../stato.sh`.

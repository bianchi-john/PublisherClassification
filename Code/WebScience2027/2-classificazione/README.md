# 2 — Classificazione

Sul **dataset definitivo** (`Dataset/versioni/dataset_finale.csv`, 272 testate con ≥20 articoli e una
pagina about utile). Due input, due letture della stessa predizione.

- **A** — solo testo degli articoli.
- **C** — testo dell'articolo **+** le pagine del sito della testata (about, ownership, correzioni…) in coda.
- Lettura **per articolo** e **per testata** (voto di maggioranza).
- Fold **per gruppo** (principale, onesto) e **per testata** (confronto), stratificati per livello.

| Script | |
|---|---|
| `01_classificatore_feature.py` | TF-IDF + regressione logistica; salva le probabilità per classe |
| `02_transformer.py` | BERT (confronto) o ModernBERT (finale); salva probabilità e loss per epoca |

Tutto viene salvato (probabilità p_l1..p_l5 per articolo, loss per epoca) per poter rigiocare
aggregazione, soglie e fusione in un secondo momento, senza riaddestrare.
Log: `../Results/2-classificazione/avanzamento_<modello>.log`.

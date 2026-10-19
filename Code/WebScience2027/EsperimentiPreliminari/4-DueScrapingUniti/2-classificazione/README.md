# 2 — Classificazione

Sul dataset finale (`../Dataset/versioni/dataset_finale.csv`), solo il testo degli articoli. Fold **per rete**
(principale) e **per testata** (confronto), stratificati per livello. Lettura per articolo e per testata (voto
di maggioranza).

| Notebook | |
|---|---|
| `01_classificatore_feature.ipynb` | TF-IDF + regressione logistica, con le baseline maggioranza e a caso |
| `comune.py` | caricamento dei dati, percorsi e letture (5 classi, F1, binario, tolerant) |

Sul server si lanciano con papermill dentro `screen` (comando nella prima cella); l'eseguito finisce in
`../Results/2-classificazione/esecuzioni/`. Cosa gira e da quanto: `../stato.sh`.

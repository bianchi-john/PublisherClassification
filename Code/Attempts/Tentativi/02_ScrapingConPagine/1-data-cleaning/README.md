# 1 — Dal dataset grezzo al dataset definitivo

Cieco alle etichette NewsGuard (si guardano solo alla fine, per descrivere).

| Notebook | |
|---|---|
| `01_testate_parenti.ipynb` | trova le testate parenti: articoli quasi uguali tra siti (MinHash) → gruppi; verifica col proprietario |
| `02_dataset_finale.ipynb` | costruisce l'**unico** dataset: testate con ≥20 articoli → pulizia degli articoli → pagine about informative e pulite → 272 testate; fold per gruppo e per testata |

Output in `../Dataset/versioni/`: `dataset_finale.csv` (articoli), `pagine_finale.csv` (pagine per testata),
`fold_finale.csv`. File intermedi grandi in `../Dataset/derived/` (fuori da git).

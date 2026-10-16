# 1 — Data cleaning

Dal dataset dello scraping al dataset finale. Cieco alle etichette NewsGuard e al campo Owner: le etichette
tornano solo in `03`, l'Owner solo per verificare le reti in `02`. I notebook girano sul server (serve l'HTML
grezzo); il comando è nella prima cella di ognuno, gli eseguiti stanno in `../Results/1-data-cleaning/esecuzioni/`.

| Notebook | |
|---|---|
| `01_pulizia_articoli.ipynb` | via le copie (esatte e quasi esatte, MinHash) anche tra testate, gli articoli prima del 2025, le frasi condivise con altre testate o ripetute nella stessa testata, titolo e timbro all'inizio del testo, i doppioni, i testi sotto le 100 parole |
| `02_reti.ipynb` | testate parenti da cinque segnali: copie e autori in comune (se sistematici, ≥10% degli articoli), codici di account nell'HTML (AdSense, Analytics, Tag Manager, Facebook, Chartbeat…), titolare legale (stessa privacy policy su un altro dominio), dominio e redirect; verifica con l'Owner (B-cubed) |
| `03_dataset_finale.ipynb` | testate con almeno 20 articoli, etichette, fold per rete e per testata |
| `comune.py` | percorsi e funzioni comuni |

Output: `../Dataset/derived/01_articoli_puliti.csv`, `01_copie_tra_testate.csv`; `../Results/1-data-cleaning/02_reti.csv`,
`02_collegamenti.csv`, `02_verifica_owner.csv`; `../Dataset/versioni/dataset_finale.csv`, `fold_finale.csv`.

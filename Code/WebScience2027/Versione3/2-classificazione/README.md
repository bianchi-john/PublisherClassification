# 2 — Classificazione a parità di genere (Versione 3)

> Il disegno e i risultati sono in `../README.md`. Il dataset è quello della Versione 2 (`../../Versione2/Dataset/versioni`).

Compito binario (affidabile = punteggio NewsGuard ≥ 60), pesi che rendono affidabili e inaffidabili ugualmente
pesanti **dentro ogni orientamento** NewsGuard, valutazione per testata dentro ogni orientamento. Metrica principale:
la media sugli orientamenti dell'accuratezza bilanciata (i soli metadati fanno 0,50).

| File | |
|---|---|
| `comune.py` | dati (dalla Versione 2, con orientamento e y binaria), fattori di genere, metriche |
| `01_feature_genere.ipynb` | TF-IDF + regressione logistica pesata; `C_GRIGLIA` per il C in validazione interna, `TESTO` per le varianti (`solo_funzione`, `senza_nomi`) |
| `02_transformer_genere.ipynb` | ModernBERT-base binario, loss pesata per genere, tetto a rotazione; a ogni fold il log scrive anche il TF-IDF sullo stesso fold |
| `04_embedding_genere.ipynb` | e5 / bge congelati (embedding già calcolati nella Versione 2), media per testata, LR pesata |
| `03_confronto_genere.ipynb` | tutti i modelli finiti, ensemble dei seed, combinazione 0,5/0,5, bootstrap per gruppo |
| `coda.py` | la coda, in due corsie: `python coda.py cpu` e `python coda.py gpu` |

Risultati in `../Results/2-classificazione/`: una cartella per modello (`articoli.csv` con la probabilità di
"affidabile" per articolo, `metriche.json`, `FATTO`), `transformer/<nome>/` per ModernBERT (anche i fold e le
epoche), i log `avanzamento_*.log`, `03_genere_metriche.csv` e `03_genere_intervalli.csv`.

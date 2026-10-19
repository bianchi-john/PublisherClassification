# 2 — Classificazione

> **Versione 2** (copia di `../../2-classificazione` con le modifiche descritte in `../README.md`: passo 4c della pulizia, testate di solo training con il tetto a rotazione, combinazione fissata a priori). La coda è `coda.py`.

Sul dataset finale (`../Dataset/versioni/dataset_finale.csv`), solo il testo degli articoli. Valutazione **per
testata**, fold **per gruppo** (principale) e per testata (confronto), stratificati per livello. Metriche
principali, fissate prima dei risultati: F1 macro a 5 classi e binario (punteggio ≥ 60).

| File | |
|---|---|
| `01_classificatore_feature.ipynb` | TF-IDF + regressione logistica (C fisso o scelto in validazione interna), baseline maggioranza e a caso |
| `02_transformer.ipynb` | fine-tuning (ModernBERT…): k articoli per testata a ogni epoca, pesi di classe, early stopping o epoche fisse |
| `04_embedding.ipynb` | embedding congelati (e5, bge) + regressione logistica, una riga per testata |
| `03_confronto.ipynb` | tutte le metriche, ensemble dei seed, intervalli e differenze con il feature (bootstrap per gruppo) |
| `coda.py` | la coda degli esperimenti: li lancia uno dopo l'altro con papermill, riprova, salta, registra |
| `comune.py` | caricamento dei dati, percorsi e metriche (5 classi, F1, binario con TP/TN/FP/FN, tolerant) |

Per ogni modello si salvano logit e probabilità di ogni articolo e `metriche.json`: si può ricalcolare qualsiasi
metrica senza riaddestrare. Risultati in `../Results/2-classificazione/` (le predizioni dei transformer in
`transformer/<nome>/`).

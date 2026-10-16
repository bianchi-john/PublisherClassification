# 1 — Primo dataset (IrForGood)

Il lavoro del 2026-10-13 sul **dataset di IrForGood** (439 testate): ricerca delle testate parenti, pulizia,
classificatore a feature e BERT con i fold per testata e per gruppo.

Messo da parte perché una sola rete di siti $l_1$ (130 testate, tutte con punteggio 32) copriva l'81% degli
articoli $l_1$: le testate indipendenti erano troppo poche. Da qui la decisione di un nuovo scraping.

| | |
|---|---|
| `0-scraping/` | i notebook di scraping di IrForGood (copie), per ricordare come nasceva il dataset |
| `1-data-cleaning/` | testate parenti e pulizia; `esplorazioni/` con le prime prove |
| `2-classificazione/` | classificatore a feature e BERT sulle versioni min0/min10/min20 |
| `Dataset/` | il dataset e le sue versioni pulite (fuori da git) |
| `Results/` | metriche e predizioni |

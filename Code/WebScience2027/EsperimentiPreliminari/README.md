# Esperimenti preliminari

Il lavoro del 2026-10-13 fatto sul **dataset vecchio** (quello di IrForGood, 439 testate): ricerca delle
testate parenti, pulizia, classificatore a feature e BERT con i fold per testata e per gruppo.

Messo da parte quando abbiamo deciso di ripartire da un **nuovo scraping** (cartella `../0-scraping`):
questi risultati restano come riferimento, non fanno parte della linea principale.

| | |
|---|---|
| `0-scraping/` | i notebook di scraping di IrForGood (copie), per ricordare come nasceva il dataset vecchio |
| `1-data-cleaning/` | testate parenti e pulizia; `esplorazioni/` con le prime prove |
| `2-classificazione/` | classificatore a feature e BERT sulle versioni min0/min10/min20 |
| `Dataset/` | il dataset vecchio e le sue versioni pulite (fuori da git) |
| `Results/` | metriche e predizioni |

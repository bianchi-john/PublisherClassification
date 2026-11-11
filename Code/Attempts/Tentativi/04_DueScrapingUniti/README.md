# Tentativo 04 — I due scraping uniti (269 testate)

> **Tentativo autonomo**: `0-scraping` → `1-data-cleaning` → `2-classificazione`, dati in `Dataset/` (fuori da git), risultati in `Results/`. `0-scraping/` è lo scraper del **primo giro** (13 ottobre). Il vecchio scraping di IrForGood è in `Dataset/VecchioScraping/`; `1-data-cleaning/00_unione_dataset` li unisce. Dati grezzi non riscaricati. Il percorso di tutti i tentativi è nel README di `Code/WebScience2027`.

Il lavoro del 2026-10-18 mattina: il vecchio scraping di IrForGood (marzo 2025, grezzo) unito al nuovo
(ottobre 2026), con gli stessi filtri e la stessa pulizia moderata (`1-data-cleaning/00_unione_dataset` + 01–03).
Dataset: 40.001 articoli, 269 testate, 199 gruppi. Feature: F1 macro testata 0,282, binario 0,677
(maggioranza 0,104 / 0,643); ModernBERT-base 1024 seed 42: 0,244 / 0,632.

Messo da parte perché **l'origine è legata all'etichetta**: l'89% degli articoli $l_1$ viene dal nuovo scraping
(contro ~70% degli altri livelli), e i due scraping hanno epoche (2024-25 contro 2025-26) e strumenti di
estrazione diversi: un modello potrebbe imparare l'epoca o lo strumento invece dell'affidabilità. Le testate
che stanno solo nel vecchio non si possono recuperare: 170 su 207 oggi bloccano lo scraper (HTTP 403).

Si riparte da un solo scraping, allargato alle testate NewsGuard il cui argomento *contiene* "Political news
or commentary" (prima si prendevano solo quelle con quell'argomento esatto).

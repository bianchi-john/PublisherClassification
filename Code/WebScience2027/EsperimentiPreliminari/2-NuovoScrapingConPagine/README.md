# 2 — Nuovo scraping con le pagine del sito (272 testate)

Il lavoro del 2026-10-14/16 sul nuovo scraping (`../../0-scraping`), con due input: solo articoli (A) e
articoli + pagine del sito (about, ownership, corrections…; C).

| | |
|---|---|
| `1-data-cleaning/` | testate parenti con MinHash sugli articoli; dataset definitivo da 272 testate (≥20 articoli e una pagina about) |
| `2-classificazione/` | TF-IDF + regressione logistica, ModernBERT (anche con early stopping), analisi di cosa impara il feature |
| `Dataset/` | `versioni/` (dataset, pagine, fold), `derived/`, `NuovoScraping_pagine/` (le pagine del sito grezze; fuori da git) |
| `Results/` | gruppi, predizioni, metriche, analisi |

## Cosa abbiamo trovato

- Con i fold per gruppo: feature C 0,596 per testata (5 classi), ModernBERT C 0,504; il feature vince per la
  lunghezza delle pagine e perché ModernBERT impara a memoria le testate (le pagine sono identiche in tutti gli
  articoli di una testata).
- **Ma i gruppi MinHash non catturano le reti di proprietà.** FiveZero Media (18 $l_1$, stesso template e stesso
  codice AdSense) è in 18 gruppi diversi, Boone Newspapers (53 $l_3$) in 19, sparse su tutti i fold. Il feature
  indovina tutte le $l_1$ FiveZero e nessuna delle altre 24 $l_1$.
- Mettendo nello stesso gruppo anche le testate con lo stesso owner NewsGuard: feature C 0,596 → 0,364,
  feature A 0,555 → 0,290 (sotto la classe di maggioranza, 0,298).
- Le pagine del sito sono spesso sbagliate (liste di titoli, orari, articoli) e non controllabili.

Da qui la decisione di rifare il data cleaning da zero, senza pagine, con un rilevamento delle reti che usi
anche l'HTML (codici di analytics e pubblicità condivisi).

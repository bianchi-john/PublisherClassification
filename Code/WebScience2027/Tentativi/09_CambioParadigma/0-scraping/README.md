# 0 — Scraping (Web Science 2027)

Punto di partenza della linea di lavoro. Raccoglie gli articoli di ogni testata NewsGuard in lingua inglese, con
un punteggio, il cui argomento (Topics) **contiene** "Political news or commentary" (3.128 domini), in ordine
casuale (seed 42). **Nessuna scelta dipende dalle etichette di affidabilità NewsGuard.**

| File | |
|---|---|
| `scraper.py` | lo scraper (vedi la sua docstring per i dettagli e l'output) |
| `avvia.sh` | tiene in vita lo scraper: se il processo principale si ferma prima della fine, lo rilancia |
| `costruisci_dataset.ipynb` | dai file raccolti al dataset di articoli: filtri (>100 parole, niente doppioni, solo inglese), etichette NewsGuard, riepilogo |

## Come è stato lanciato

Sul server artes4, in due giri, con lo stesso scraper e gli stessi parametri:

1. **13 ottobre 2026**: le 1.921 testate con argomento *esattamente* "Political news or commentary" (1h14);
2. **18 ottobre 2026**: le testate il cui argomento *contiene* "Political news or commentary" (+1.207), lanciato da
   `../esegui_tutto.py` (57 minuti). Le testate già tentate si saltano.

In tutto: 3.128 testate tentate, 1.528 raggiunte, 170.213 testi estratti. Molte testate rifiutano le richieste
automatiche (HTTP 403): una rete di 1.076 siti $l_1$ quasi per intero e altre 176 testate.

## Output

`../Dataset/NuovoScraping/`: `articoli/`, `html/` (l'HTML grezzo di ogni articolo, serve a `1-data-cleaning/02_reti`),
`stato/`, `config.json` (parametri e versioni del primo giro), `elenco_testate.csv`, i log. Il notebook
`costruisci_dataset.ipynb` ci legge e scrive `../Dataset/NuovoScraping/dataset_articoli.csv` (101.887 articoli in
inglese, 1.046 testate). Il dataset del solo primo giro è in `dataset_articoli_13ottobre.csv`.

# 0 — Scraping (Web Science 2027)

> Lo scraper del **primo giro** (13 ottobre 2026), com'era quando è stato usato da questo tentativo (dalla storia di git, commit `4548932`): argomento esatto "Political news or commentary", articoli e pagine del sito.

Nuovo scraping, punto di partenza di questa linea di lavoro. Raccoglie gli articoli di ogni
testata NewsGuard con argomento "Political news or commentary", lingua inglese e un punteggio (1.921 domini),
in ordine casuale (seed 42). **Nessuna scelta dipende dalle etichette di affidabilità NewsGuard.**

| File | |
|---|---|
| `scraper.py` | lo scraper (vedi la sua docstring per i dettagli e l'output) |
| `avvia.sh` | tiene in vita lo scraper: se il processo principale si ferma prima della fine, lo rilancia |
| `costruisci_dataset.ipynb` | dai file raccolti al dataset di articoli: filtri (>100 parole, niente doppioni, solo inglese), etichette NewsGuard, riepilogo |

## Come è stato lanciato

Sul server artes4, 2026-10-13, dentro `screen`:

```bash
cd ~/John/WebScience2027/0-scraping
screen -dmS scraping_ws27 bash avvia.sh ../Dataset/NuovoScraping 64
```

Finito in 1h14, senza riavvii: 1.921 testate tentate, 600 raggiunte, ~80.000 articoli grezzi.
Molte testate (soprattutto una grande rete di cloni $l_1$ con lo stesso proprietario) rifiutano le richieste
automatiche (HTTP 403) e vengono saltate e annotate.

## Output

`../Dataset/NuovoScraping/` (articoli, html, stato, config.json, ...). Il notebook
`costruisci_dataset.ipynb` ci legge e scrive `../Dataset/NuovoScraping/dataset_articoli.csv`
(54.532 articoli in inglese, 354 testate). L'HTML grezzo di ogni articolo (`html/`, solo sul server) serve
al data cleaning per riconoscere le reti di siti.

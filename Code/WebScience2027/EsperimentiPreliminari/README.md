# Esperimenti preliminari

I due tentativi fatti prima della linea principale. Restano come riferimento (e come storia delle scelte),
non fanno parte della pipeline.

| | Dataset | Perché è stato messo da parte |
|---|---|---|
| `1-PrimoDataset/` | il dataset di IrForGood (439 testate) | una sola rete di siti $l_1$ copre l'81% degli articoli $l_1$; troppo poche testate indipendenti. Si riparte da un nuovo scraping |
| `2-NuovoScrapingConPagine/` | il nuovo scraping, articoli + pagine del sito (272 testate) | i gruppi trovati dal MinHash sugli articoli non catturano le reti di proprietà (FiveZero Media, Boone Newspapers…), che restano sparse tra i fold; le pagine del sito sono troppo sporche per essere controllate. Si rifà il data cleaning da zero, senza pagine |

Ogni cartella ha il suo README con il dettaglio. I percorsi nel codice puntano ai dati dentro la cartella
stessa (`Dataset/`, `Results/`) e, per i dati grezzi del nuovo scraping, a `../../Dataset/NuovoScraping/`.

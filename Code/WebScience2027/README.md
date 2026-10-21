# Web Science 2027

Predire l'affidabilità NewsGuard di una testata **dal solo testo dei suoi articoli**, valutando su testate di **reti
mai viste** in addestramento. La domanda di fondo: un transformer fa meglio di un classificatore a feature (TF-IDF +
regressione logistica)?

Il lavoro è diviso in **tentativi**, in ordine di tempo, in `Tentativi/`. Ogni tentativo è **autonomo**: contiene
tutta la catena, dallo scraping ai risultati, come se si ripartisse da zero, e legge e scrive solo dentro la sua
cartella. Nella realtà i tentativi successivi hanno riusato lo scraping e la pulizia dei precedenti; il README di ogni
tentativo dice che cosa è stato davvero rieseguito e che cosa è stato copiato.

## I tentativi

| | Tentativo | La domanda | Il risultato |
|---|---|---|---|
| 01 | `01_PrimoDataset` (13 ottobre) | perché il TF-IDF va come i transformer del paper IrForGood? | le **reti** di siti gonfiano i numeri: 0,68 → 0,39 con i fold per gruppo; una rete di 131 siti fa l'86% di $l_1$ |
| 02 | `02_ScrapingConPagine` (14–16 ottobre) | un nuovo scraping, con articoli e pagine del sito | i gruppi trovati dalle copie non separano le reti di proprietà; pagine troppo sporche |
| 03 | `03_SoloArticoli193Testate` (16–17 ottobre) | solo articoli, pulizia moderata, reti da 5 segnali | tutti poco sopra la maggioranza: troppe poche testate (193) |
| 04 | `04_DueScrapingUniti` (18 ottobre) | il vecchio scraping unito al nuovo | l'origine è legata all'etichetta (89% delle $l_1$ dal nuovo): scartato |
| 05 | `05_Versione1_Pipeline304Testate` (18 ottobre) | scraping allargato (due giri), pulizia, reti, fold per gruppo: chi vince? | **pari**: TF-IDF F1 0,38, ModernBERT 0,35–0,37 |
| 06 | `06_Versione2_PiuPuliziaPiuTraining` (19–20 ottobre) | più pulizia e più testate nel training aiutano i transformer? | **no**; l'analisi del TF-IDF mostra che legge il **genere** della testata (orientamento, tipo di sito, paese) |
| 07 | `07_Versione3_PariGenere` (20–21 ottobre) | a parità di orientamento politico, il testo dice l'affidabilità? | sì, ma poco (0,6 contro 0,5) ed è **stile**; il TF-IDF resta davanti |
| 08 | `08_Versione4_TitoloCriteri` (21 ottobre, **in corso**) | il titolo letto insieme all'articolo e i criteri NewsGuard come etichette in più aiutano i transformer? | in corso |

La storia completa, con i grafici, è nella presentazione `Riunioni/Stato dei lavori/stato_dei_lavori.pdf`.

## Com'è fatto un tentativo

```
0-scraping/          scraper e costruzione del dataset grezzo di articoli  ->  Dataset/NuovoScraping/
1-data-cleaning/     pulizia, reti di testate parenti, dataset finale e fold  ->  Dataset/derived/, Dataset/versioni/
2-classificazione/   TF-IDF, transformer, embedding, confronto  ->  Results/2-classificazione/
Dataset/             i dati (fuori da git: dati NewsGuard non redistribuibili, e troppo grandi)
Results/             notebook eseguiti, log, predizioni, metriche (in git, tranne embedding .npy e modelli)
README.md            la domanda, il disegno fissato prima dei risultati, i risultati, come rilanciarlo
stato.sh, controlla_server.sh, esegui_tutto.py / coda.py   per farlo girare sul server senza nessuno davanti
```

I notebook si eseguono dalla loro cartella (`papermill --cwd`), con percorsi relativi: `../Dataset`, `../Results`.
`Dataset/newsguard.csv` (l'export NewsGuard) va messo a mano; il resto lo produce la catena. Ogni scelta di pulizia
è una regola nel codice, deterministica (random state fissati): nessuna modifica a mano né fatta da un LLM.

## Sul server

Server condiviso `artes4` (Tesla V100S 32 GB, ma di solito solo circa 7 GB liberi), venv `~/John/johnVenv` con il
kernel Jupyter `johnvenv`. Un tentativo va copiato in `~/John/WebScience2027/Tentativi/<tentativo>` e lanciato
dalla sua cartella in uno `screen` (vedi il README del tentativo). I transformer girano con batch piccoli e accumulo
del gradiente; su artes4 NVML è rotto, e la memoria GPU finita arriva come `RuntimeError ... NVML ...
CUDACachingAllocator`: i notebook dal tentativo 06 in poi la riconoscono e dimezzano il batch.

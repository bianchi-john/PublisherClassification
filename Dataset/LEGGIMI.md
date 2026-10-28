# Dataset: la cartella unica dei dati di WebScience2027

Tutti i dati dei tentativi in `Code/WebScience2027/Tentativi/` stanno qui, fuori da git. Il codice li trova da solo, a
patto che questa cartella stia nella **radice del repo** (accanto a `Code/`), con questo file dentro: i tentativi 14–16
cercano `Dataset/LEGGIMI.md` salendo dalla cartella di lavoro, gli altri usano percorsi relativi alla radice del repo.

| cartella | che cosa c'è | chi la usa |
|---|---|---|
| `scraping/NuovoScraping` | lo scraping di ottobre 2026 (primo giro): articoli, stato, HTML (9 GB) | 02–11 |
| `scraping/VecchioScraping` | lo scraping di IrForGood (marzo 2025) | 04–05 |
| `scraping/NuovoScraping_ottobre`, `scraping/NuovoScraping_giro3` | i giri di ottobre 2026 con le testate politiche (8 e 5 GB) | 12, 14, 16 |
| `etichette/newsguard_2022.csv` | l'esportazione NewsGuard di agosto 2022 | 01–12 |
| `etichette/newsguard_2026.csv`, `etichette/NewsGuard2026/` | NewsGuard 2026 (il feed del 23 ottobre 2026; `newsguard_2026.csv` è il suo `metadata-2026102308.csv`) | 11–16 |
| `risorse/lessici` | lessici NRC (emozioni, VAD) | 13, 15, 16 |
| `tentativi/<tentativo>/` | quello che produce ogni tentativo: `derived/`, `versioni/`, il dataset finale (`articoli.csv.gz`, `testate.csv`), dati propri (MBFC nel 10, segnali nel 09…) | il tentativo stesso |

I modelli addestrati non sono qui ma in `Modelli/` (radice del repo), anche quella fuori da git.

**Dati riservati.** NewsGuard (etichette e punteggi), MBFC e i lessici NRC non si ridistribuiscono: questa cartella si
passa solo a chi lavora al progetto.

## L'archivio leggero `Dataset.zip` (meno di 1 GB)

Si crea con `python Code/WebScience2027/crea_archivio_dataset.py` e si estrae **nella radice del repo**
(`unzip Dataset.zip` accanto a `Code/`): nasce questa cartella, con solo quello che serve al protocollo attuale.

| tentativo | che cosa si può rifare con l'archivio |
|---|---|
| 16_DataPreparation | **tutto**, dal notebook 1: il testo riestratto con trafilatura è nella cache (`tentativi/16_DataPreparation/cache/estratti.csv.gz`), quindi l'HTML non serve; dello scraping basta lo stato di ogni testata (`scraping/*/stato`) |
| 14_DatasetRipensatoChunk | la classificazione (notebook 2), dal dataset finale; il notebook 1 vuole l'HTML |
| 15_AnalisiDataset | tutto (legge il dataset del 14) |
| 01–13 | no: servono lo scraping completo e i dati intermedi, che sono sul server |

Gli embedding (notebook 4 del 16) e ModernBERT si ricalcolano da soli con la GPU.

**Per rifare gli esperimenti con la cartella completa**
- Tentativo 16 da zero senza la cache: servono anche l'HTML di `scraping/NuovoScraping_ottobre` e `scraping/NuovoScraping_giro3`.
- `tentativi/12_PiuTestatePolitiche/NuovoScraping` è l'unione dei due giri di ottobre fatta con collegamenti fisici
  (`0-scraping/unisci.sh` del tentativo 12): in un archivio zip si lascia fuori, si rifà in un minuto.

**Sul server** (artes4) la stessa struttura sta in `~/John/Dataset`, accanto alle cartelle `~/John/TentativoNN`.

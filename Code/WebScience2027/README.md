# Web Science 2027 — affidabilità delle testate dal solo testo

Questa cartella contiene tutto il lavoro sperimentale verso Web Science 2027. È organizzata in **tentativi**
(`Tentativi/01…08`), in ordine di tempo: ognuno è una catena completa e autonoma, dallo scraping ai risultati.
Questo README spiega il problema, come lo valutiamo, che cosa ha detto ogni tentativo e come si rilancia tutto.

---

## 1. Il problema

**Obiettivo.** Data una testata giornalistica online (un dominio, per esempio `breitbart.com`), predire quanto è
affidabile secondo **NewsGuard** usando **solo il testo dei suoi articoli**: niente URL, niente metadati, niente
reputazione esterna.

**Le etichette.** NewsGuard dà a ogni testata un punteggio da 0 a 100. Lo dividiamo in 5 livelli:

| livello | punteggio | | 
|---|---|---|
| $l_1$ | 0–39 | inaffidabile |
| $l_2$ | 40–59 | |
| $l_3$ | 60–74 | |
| $l_4$ | 75–99 | |
| $l_5$ | 100 | massima affidabilità |

Il compito **binario** è "affidabile" = punteggio ≥ 60 (livelli $l_3$–$l_5$). NewsGuard dà anche 9 criteri sì/no per
testata (contenuti falsi, informazione responsabile, rettifiche, titoli ingannevoli…), l'orientamento politico e il
proprietario (*Owner*).

**La domanda di ricerca.** Un transformer (ModernBERT, addestrato sugli articoli) fa meglio di un classificatore a
feature (TF-IDF + regressione logistica)? Il paper precedente (IrForGood) diceva che i transformer arrivavano a 0,69 di
accuratezza per testata; il punto di partenza di questo lavoro è stato scoprire che un TF-IDF faceva lo stesso, e
capire perché.

## 2. Come valutiamo (e perché così)

- **Per testata.** Il modello giudica ogni articolo; la testata prende la **media delle probabilità** dei suoi
  articoli (o il voto di maggioranza). L'etichetta vera esiste solo per la testata.
- **Le reti di siti.** Molte testate non sono indipendenti: reti di decine o centinaia di siti con articoli quasi
  uguali o lo stesso proprietario (Metric Media, Natural News, catene di giornali locali…). Se una testata sta nel test
  e le sue "sorelle" nel training, il modello riconosce la rete invece di giudicare l'affidabilità. Nel tentativo 01
  questo da solo portava l'accuratezza da 0,39 a 0,68.
- **Fold per gruppo.** Per questo le testate vengono raggruppate: un **gruppo** = una rete trovata dal testo e
  dall'HTML (copie di articoli, stessi codici AdSense/Analytics, stessa privacy policy, stesso dominio, stessi autori)
  **oppure** lo stesso proprietario NewsGuard. Un gruppo sta tutto nel training o tutto nel test (5 fold, stratificati
  per livello). Il rilevatore di reti, che non guarda NewsGuard, coincide con l'Owner con precisione 0,91 e recall 0,88.
- **Al più 5 testate per gruppo nel test**, almeno 20 articoli per testata: una rete enorme non può riempire un
  livello. Dal tentativo 06 il training usa anche le altre testate dei gruppi, con un peso (un gruppo vale al più 5).
- **Metriche.** Principali: **F1 macro a 5 livelli** e **accuratezza binaria**, per testata. Anche accuratezza a 5
  livelli, *tolerant* (un errore di un livello vale 0,5 o 1) e il dettaglio per livello. Intervalli al 95% con un
  **bootstrap per gruppo** (2000 ricampionamenti dei gruppi, perché le testate di un gruppo sbagliano insieme).
- **Regole del gioco.** Ogni scelta (pulizia, metriche, configurazioni) è scritta **prima** di vedere i risultati e
  vale uguale per tutti i modelli. La pulizia è fatta solo di regole nel codice, deterministiche (random state
  fissati): niente modifiche a mano né fatte da un LLM. Si riporta tutto, non solo il migliore.

## 3. I tentativi, uno per uno

| | Tentativo | La domanda | Che cosa è venuto fuori |
|---|---|---|---|
| 01 | `01_PrimoDataset` (13 ott.) | Perché il TF-IDF va come i transformer del paper IrForGood (439 testate)? | Le **reti** gonfiano i numeri: TF-IDF 0,68 con i fold per testata, **0,39** con i fold per gruppo (BERT 0,40). Una rete di 131 siti fa l'86% di $l_1$: troppo poche testate indipendenti → serve un dataset nuovo |
| 02 | `02_ScrapingConPagine` (14–16 ott.) | Nuovo scraping, articoli + pagine del sito (about, proprietà…), 272 testate | Il TF-IDF con le pagine sembra forte (0,60), ma i gruppi trovati solo dalle copie non vedono le reti di proprietà (FiveZero Media, giornali Boone); con l'Owner nei gruppi scende a 0,36. Le pagine del sito sono troppo sporche → le reti vanno cercate anche nell'HTML, niente pagine |
| 03 | `03_SoloArticoli193Testate` (16–17 ott.) | Solo articoli, pulizia moderata, reti da 5 segnali + Owner | Tutti poco sopra la maggioranza (TF-IDF F1 0,274, ModernBERT ~0,25): con 193 testate non si distingue nulla → servono più testate |
| 04 | `04_DueScrapingUniti` (18 ott.) | Unire il vecchio scraping di IrForGood al nuovo (269 testate) | **Scartato**: l'origine è legata all'etichetta (l'89% degli articoli $l_1$ viene dal nuovo scraping, epoche e strumenti diversi) → un solo scraping, ma allargato |
| 05 | `05_Versione1_Pipeline304Testate` (18 ott.) | **Versione 1**: scraping in due giri (3.128 testate tentate), pulizia, reti, fold per gruppo, 304 testate | **Pari**: TF-IDF F1 0,380 / binario 0,740; ModernBERT-base 0,366 / 0,717; large 0,348; nessuna differenza significativa |
| 06 | `06_Versione2_PiuPuliziaPiuTraining` (19–20 ott.) | **Versione 2**: una regola di pulizia in più (resti di link e tweet), 863 testate nel training, combinazione TF-IDF + ModernBERT | **Nessun guadagno** (TF-IDF 0,370–0,389, ModernBERT 0,341). L'analisi del TF-IDF (`05_analisi_feature`) spiega perché: il segnale è sparso e segue il **genere** della testata (orientamento politico, tipo di sito, paese); bastano le sole parole funzione (0,364), i soli metadati NewsGuard fanno 0,339 |
| 07 | `07_Versione3_PariGenere` (20–21 ott.) | **Versione 3**: affidabile sì/no **dentro** ogni orientamento politico, con pesi che tolgono l'aiuto dell'orientamento | Un segnale oltre il genere c'è ma è modesto (0,55–0,62 di accuratezza bilanciata contro 0,50) ed è **stile**; il TF-IDF resta davanti (0,593–0,618 contro 0,551 di ModernBERT). Unico indizio pro transformer: l'orientamento "Slightly Right" |
| 08 | `08_Versione4_TitoloCriteri` (21 ott.) | **Versione 4**: ModernBERT legge *titolo + articolo* (può confrontarli) e impara anche i 5 criteri NewsGuard visibili nel testo | Il titolo aiuta ModernBERT (ensemble 0,341 → 0,356; con i criteri 0,360) ma aiuta **anche** il TF-IDF (0,370 → 0,390): il distacco resta, non significativo |
| 09 | `09_CambioParadigma` (22–23 ott.) | **Oltre il testo**: per ogni testata tre famiglie di segnali dall'HTML (testo TF-IDF, domini citati e loro affidabilità, segnali della pagina) più ModernBERT; un modello per famiglia, la combinazione (stacking fuori fold) e un modello "alla NewsGuard" che stima i 9 criteri; gruppi con anche il Parent Domain (301 testate) | **Stesso tetto**: riferimento TF-IDF 0,340 / binario 0,684; link 0,247 e pagina 0,241 (significativamente sotto); migliore combinazione testo+pagina 0,354 (+0,01, non significativo); ModernBERT 0,325; alla NewsGuard 0,319 / 0,718. La sola divisione in fold sposta il riferimento di quasi 2 punti |
| 10 | `10_EtichetteMBFC` (23 ott.) | **Le etichette sono vecchie?** NewsGuard è un'esportazione di agosto 2022, gli articoli sono del 2025–26: confronto con le valutazioni MBFC di oggi (API ufficiale), abbinate alle nostre testate | MBFC copre il 50% del dataset finale e il 70% delle valutate (mancano soprattutto i domini di Natural News). **Accordo alto**: affidabile per NewsGuard ~ credibilità MBFC alta/media nel 88–94% delle testate, Spearman −0,84 tra punteggio NewsGuard e factual score MBFC: sul binario le etichette 2022 reggono |

**In sintesi, finora.** Con una valutazione che tiene conto delle reti, il solo testo dà un segnale moderato (F1
≈ 0,38–0,39, binario ≈ 0,74–0,78) e i transformer non battono un TF-IDF fatto bene, in quattro versioni
dell'esperimento (05–08), nemmeno dando loro il titolo e i criteri NewsGuard. Il TF-IDF va bene perché i livelli
NewsGuard vanno di pari passo con il genere della testata, che si legge ovunque nel testo. La presentazione con tutti i
grafici è `Riunioni/Stato dei lavori/stato_dei_lavori.pdf`.

## 4. Com'è fatto un tentativo

Ogni tentativo è **autonomo**: contiene la catena completa e legge e scrive solo dentro la sua cartella, come se si
ripartisse da zero.

```
Tentativi/NN_nome/
  README.md            la domanda, il disegno fissato prima dei risultati, i risultati, come rilanciarlo
  0-scraping/          scraper + costruisci_dataset.ipynb            ->  Dataset/NuovoScraping/
  1-data-cleaning/     01 pulizia, 02 reti di testate, 03 dataset finale e fold
                                                                        ->  Dataset/derived/, Dataset/versioni/
  2-classificazione/   01 TF-IDF, 02 transformer, 03 confronto, (04 embedding…), coda degli esperimenti
                                                                        ->  Results/2-classificazione/
  Dataset/             i dati: fuori da git (le etichette NewsGuard non sono redistribuibili, e lo scraping pesa ~9 GB)
  Results/             notebook eseguiti, log, predizioni per articolo, metriche (in git, tranne embedding e modelli)
  stato.sh, controlla_server.sh, esegui_tutto.py / coda.py   per far girare tutto sul server senza nessuno davanti
```

Nella realtà i tentativi hanno riusato il lavoro dei precedenti: lo scraping è stato fatto davvero solo due volte (il
primo giro, usato da 02–04, e quello completo del 05, copiato in 06–08), e 07–08 usano il dataset pulito del 06. Ogni
README dice in apertura che cosa è stato rieseguito e che cosa è copiato. Rieseguendo una catena si riottiene lo
stesso dataset, perché tutti i passi sono deterministici.

**Per aggiungere un esperimento** si crea `Tentativi/09_nome/` copiando dentro la catena del tentativo da cui si
parte, si scrive nel README la domanda e il disegno prima di lanciare, e non si usano mai percorsi verso altri
tentativi.

## 5. Rilanciare

**I dati.** `Dataset/newsguard.csv` (l'export NewsGuard, dati proprietari) va messo a mano in ogni tentativo; tutto
il resto lo produce la catena (`0-scraping` scarica gli articoli e l'HTML grezzo, che serve a `02_reti`).

**Il server.** Si lavora sul server condiviso `artes4` (GPU Tesla V100S da 32 GB, di solito solo ~7 GB liberi
perché la usano anche altri), con il venv `~/John/johnVenv` e il kernel Jupyter `johnvenv`. Un tentativo va copiato
in `~/John/WebScience2027/Tentativi/<tentativo>` e lanciato dalla sua cartella in uno `screen`, come scritto nel suo
README; per esempio, per l'intera catena del tentativo 05:

```bash
cd ~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate
screen -dmS tutto bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'
./stato.sh                     # sul server: a che punto è
./controlla_server.sh          # dal PC: la stessa cosa via ssh
```

Le code (`esegui_tutto.py`, `coda.py`) eseguono i notebook con papermill in processi separati, con tempi massimi,
fino a 3 tentativi per esperimento e ripresa da dove si erano fermate; i log stanno in `Results/`.

**Note pratiche sulla GPU.** I transformer girano con batch piccoli (4) e accumulo del gradiente fino a 32, così
entrano in ~5 GB; ModernBERT-large non entra nei ~7 GB liberi. Su artes4 NVML è rotto (driver/library mismatch):
`nvidia-smi` non funziona e la memoria GPU finita arriva come `RuntimeError ... NVML ... CUDACachingAllocator`
invece di `OutOfMemoryError`; i notebook dal tentativo 06 in poi la riconoscono e ripartono con il batch dimezzato.

## 6. Glossario

- **Testata**: un sito di notizie (un dominio), con un punteggio NewsGuard.
- **Rete**: testate parenti trovate senza NewsGuard (articoli copiati, stessi codici di account nell'HTML, stessa
  privacy policy, stesso dominio, stessi autori).
- **Gruppo**: rete trovata **oppure** stesso proprietario NewsGuard; l'unità dei fold.
- **Testate valutate / di solo training**: quelle nel test (almeno 20 articoli, al più 5 per gruppo) e le altre, usate
  solo per addestrare (dal tentativo 06).
- **Fold per gruppo / per testata**: un gruppo tutto da una parte / ogni testata a caso (il secondo solo per confronto).
- **Ensemble**: media delle probabilità dello stesso modello addestrato con seed diversi.
- **Genere** di una testata: orientamento politico, tipo di sito, paese; è ciò che il TF-IDF legge di più.

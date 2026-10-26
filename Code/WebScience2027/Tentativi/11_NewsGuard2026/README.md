# Tentativo 11 — Le etichette NewsGuard aggiornate (2026)

> **Tentativo autonomo**: dati in `Dataset/` (fuori da git), risultati in `Results/`. Il percorso di tutti i tentativi è
> nel README di `Code/WebScience2027`.

**La domanda.** Fino al tentativo 10 le etichette erano un'esportazione NewsGuard di **agosto 2022**: per i nostri
articoli (2025–2026) avevano in mediana 4,2 anni, e i 5 livelli si erano spostati (tentativo 10: confronto con MBFC).
Dal 23 ottobre 2026 abbiamo accesso al **feed S3 di NewsGuard**, con un'istantanea completa ogni ora dal 2019. Con
etichette aggiornate, il quadro cambia?

## I dati (in `Dataset/`, fuori da git)

| file | che cos'è |
|---|---|
| `NewsGuard2026/metadata-2026102308.csv` | tutte le valutazioni NewsGuard del 23 ottobre 2026, ore 08 di New York (s3://newsguard-feeds/full-metadata/2026/10/); provenienza e checksum in `PROVENIENZA.json` |
| `NewsGuard2026/all-sources-metadata.csv`, `label-2026102308.json` | tutte le fonti (anche non siti) e le schede complete: per ora non usati |
| `newsguard_2022.csv` | l'esportazione del 2022 usata nei tentativi 01–10 (per il confronto) |
| `NuovoScraping/`, `derived/` | gli articoli grezzi dello scraping (13 e 18 ottobre), quelli puliti (`01_articoli_puliti.csv`), le copie e le reti: collegamenti fisici ai file del tentativo 09 |

**Accesso al feed** (per scaricare un'istantanea nuova): le credenziali **non** stanno nel repository; ognuno le mette
nel proprio `~/.aws/credentials` (profilo `newsguard`, regione `us-east-1`) e usa `boto3.Session(profile_name="newsguard")`.
I dati NewsGuard sono riservati: non si ridistribuiscono.

## 1. Le etichette 2026 e i nostri articoli (`1-etichette/01_newsguard_2026.ipynb`)

- **Il punteggio è esattamente la somma pesata dei 9 criteri, con gli stessi pesi** (22, 18, 12,5, 12,5, 10, 7,5, 7,5,
  5, 5; un criterio non applicato vale i suoi punti) per il 100% delle 11.564 testate con punteggio, sia nel 2026 sia
  nel 2022. Due criteri hanno cambiato nome ("Does not publish false or egregiously misleading content at a
  significant level", "Has effective practices for correcting errors"), non peso.
- **Rating**: `T` = affidabile (punteggio 60–100), `N` = non affidabile (0–59,5); gli altri codici (`G`, `FL`, `L`,
  `AC`, `P`, `S`) non hanno punteggio.
- **Copertura** (`01_copertura.csv`): delle 1.046 testate con articoli scaricati, **1.015** hanno un punteggio 2026.
  Sugli articoli puliti: 835 testate con almeno 5 articoli (71.663 articoli; l1 409, l2 95, l3 48, l4 232, l5 51), 677
  con almeno 20.
- **Cambiamenti dal 2022** (835 testate, `01_cambiamenti.csv`, `01_transizioni_2022_2026.csv`): punteggio cambiato per
  257, **livello cambiato per 161** (128 salite, 33 scese), affidabile/inaffidabile cambiato per 36. Il passaggio più
  frequente è l3 → l4 (59 delle 86 testate l3 del 2022).
- **Età delle etichette**: tra l'ultimo aggiornamento NewsGuard e la data mediana degli articoli, **3 mesi** in mediana
  (erano 4,2 anni); l'88% delle testate è stato rivalutato nel 2025–2026.

## 2. Il dataset (`2-dataset/01_dataset.ipynb`)

Gli **articoli puliti** dei tentativi 06–09 con **due etichette** per testata, il livello NewsGuard **2026**
(l'obiettivo) e quello **2022** (il controllo), e **le stesse testate e gli stessi fold** per le due. Regole del
tentativo 09: testate con un punteggio in tutti e due i feed; gruppi = reti ∪ owner ∪ Parent Domain, presi dal 2022 **e**
dal 2026; via le copie tra gruppi diversi; valutate = almeno 20 articoli e al più 5 per gruppo; di solo training = almeno
5 articoli, peso min(1, 5/n); 5 fold per gruppo stratificati sul livello 2026.

| | testate | gruppi | l1 | l2 | l3 | l4 | l5 | affidabili |
|---|---|---|---|---|---|---|---|---|
| valutate, etichette 2026 | 296 | 215 | 51 | 67 | 39 | 94 | 45 | 60% |
| valutate, etichette 2022 | 296 | 215 | 66 | 63 | 35 | 96 | 36 | 56% |
| tutte (training), 2026 | 835 | 254 | 409 | 95 | 48 | 232 | 51 | 40% |

Tra 2022 e 2026 cambia il livello di 79 valutate su 296 e il binario di 29. Articoli: 71.218 (42.490 delle valutate).

## 3. Gli esperimenti del weekend (`2-dataset/02_titoli_e_politici.ipynb`, `3-classificazione/`)

Fissati venerdì 23 ottobre prima dei risultati: le cose più semplici, **un solo seme** per ModernBERT (42), tre
varianti, ognuna con le etichette **2026** e, per controllo, **2022** (stesse testate, stessi fold).

| variante | che cosa cambia | testate valutate |
|---|---|---|
| **base** | solo il testo dell'articolo | 296 |
| **titolo** | anche il titolo, **pulito dalla cornice del sito** (un pezzo iniziale o finale, separato da un trattino, una barra verticale o simili, che torna in almeno il 20% dei titoli della testata: 21% dei titoli, 127 testate); il classificatore a feature ha un TF-IDF a parte sul titolo, ModernBERT legge la coppia *titolo [SEP] articolo* (come nella Versione 4) | 296 |
| **politici** | training e test solo sulle testate che per NewsGuard 2026 hanno **un solo argomento, "Political news or commentary"** (via quelle con più generi, per esempio politica + salute + complotti: la rete di Natural News); stessi fold, peso ricalcolato | 143 (187 in training) |

Modelli: il **classificatore a feature** (TF-IDF + LR, C in validazione interna, voto per testata) e **ModernBERT-base**
(ricetta della Versione 2, media delle probabilità per testata). Il confronto (`03_confronto`) valuta ogni modello sulle
etichette del suo anno; i modelli base 2022 anche sulle 2026; i modelli base anche sulle 143 testate politiche, per
confrontarli con la variante politici sulle stesse testate; e le differenze titolo − base, politici − base, 2026 − 2022,
ModernBERT − feature, con bootstrap per gruppo.

**La coda** (`3-classificazione/coda.py`), due corsie in due screen che ripartono da sole se si bloccano:

- `t11_cpu`: feature base 2026 e 2022 (già fatti), titolo 2026, politici 2026, titolo 2022, politici 2022;
- `t11_gpu`: ModernBERT base 2026 e 2022, titolo 2026, politici 2026, titolo 2022, politici 2022 (un modello alla volta;
  aspetta che la GPU condivisa abbia memoria libera; riparte dai fold già finiti).

Ogni esperimento si riprova fino a 3 volte (con un tempo massimo), poi si passa al successivo; dopo ogni esperimento
finito il confronto si aggiorna. Tutto è stato provato prima del lancio (versioni brevi di ogni notebook, in locale e
sul server).

```
2-dataset/02_titoli_e_politici.ipynb       titoli puliti (Dataset/versioni/titoli.csv) e testate solo politiche (solo_politici.csv)
3-classificazione/comune.py                dati, pesi e metriche; carica(ETICHETTA, titolo=, solo_politici=)
3-classificazione/01_feature_tfidf.ipynb   il classificatore a feature (ETICHETTA, TITOLO, SOLO_POLITICI)
3-classificazione/02_modernbert.ipynb      ModernBERT-base (ETICHETTA, TITOLO, SOLO_POLITICI, SEED)
3-classificazione/03_confronto.ipynb       intervalli, differenze, matrici di confusione (usa i modelli finiti)
3-classificazione/coda.py                  la coda del weekend, corsie cpu e gpu
stato.sh, controlla_server.sh              a che punto siamo (sul server; dal PC)
esegui_tutto.sh                            il primo lancio del 23 ottobre (sostituito dalla coda: aveva tre semi e si era
                                           bloccato aspettando la GPU occupata da se stesso)
```

Sul server: `~/John/Tentativo11`, screen `t11_cpu` e `t11_gpu`, lanciati venerdì 23 ottobre alle 16:48 e 16:59. Dal PC:
`./controlla_server.sh`. Previsto: il classificatore a feature entro venerdì sera, ModernBERT circa 1h20 per modello
(i 6 modelli entro sabato, anche con la GPU in parte occupata da altri).

## Risultati (coda del weekend, finita venerdì 23 ottobre alle 23:04; confronto rifatto domenica 25)

Tutti i 12 modelli sono finiti al primo tentativo. Il confronto si era fermato per il tempo massimo (30 minuti, troppo
pochi per il bootstrap con tanti modelli): domenica è stato reso più veloce (le 4 metriche calcolate direttamente,
con un controllo che coincidano con quelle complete) e rifatto. Per testata, F1 macro a 5 livelli e binario, intervalli
al 95% con bootstrap per gruppo (`Results/3-classificazione/confronto/`):

| variante | modello | etichette | testate | F1 [IC 95%] | binario |
|---|---|---|---|---|---|
| base | feature | 2026 | 296 | 0,351 [0,283; 0,413] | 0,676 |
| base | ModernBERT | 2026 | 296 | 0,343 [0,278; 0,402] | 0,689 |
| base | feature | 2022 | 296 | 0,341 [0,285; 0,397] | 0,716 |
| base | ModernBERT | 2022 | 296 | 0,331 [0,271; 0,384] | 0,662 |
| titolo | feature | 2026 | 296 | 0,359 [0,292; 0,420] | 0,689 |
| titolo | ModernBERT | 2026 | 296 | **0,371** [0,300; 0,434] | 0,699 |
| titolo | feature | 2022 | 296 | 0,339 [0,280; 0,391] | 0,696 |
| titolo | ModernBERT | 2022 | 296 | 0,309 [0,252; 0,364] | 0,649 |
| base | feature / ModernBERT | 2026 | 143 politiche | 0,305 / 0,306 | 0,615 / 0,643 |
| politici | feature / ModernBERT | 2026 | 143 politiche | 0,276 / 0,266 | 0,615 / 0,559 |

- **Etichette aggiornate**: nessun guadagno significativo (feature +0,010 [−0,066; +0,079], ModernBERT +0,012
  [−0,058; +0,087], ognuno sul suo anno). I modelli addestrati sul 2022 vanno anzi un po' meglio sulle etichette 2026
  (feature 0,376, ModernBERT 0,386), non in modo significativo: nel 2026 la classe l3 del training si è quasi svuotata
  (48 testate contro 87).
- **Titolo**: con le etichette 2026 aiuta un po' tutti e due (feature +0,009, ModernBERT +0,027 [−0,025; +0,079]) e
  ModernBERT passa davanti al feature (+0,012); con le etichette 2022 succede il contrario (ModernBERT −0,022). Effetti
  piccoli e incoerenti: rumore.
- **Solo politici**: addestrare solo sulle testate politiche non aiuta (feature −0,029, ModernBERT −0,040 rispetto al
  modello addestrato su tutte, sulle stesse 143 testate).
- **ModernBERT contro feature**: mai significativamente diversi in F1; l'unica differenza significativa di tutto il
  tentativo è il binario con le etichette 2022, a favore del feature (−0,054 [−0,100; −0,010]).
- In sintesi: anche con etichette di 3 mesi invece che di 4 anni, con il titolo o solo sulle testate politiche,
  tutto resta tra 0,33 e 0,37 di F1 e tra 0,66 e 0,72 di binario.

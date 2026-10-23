# Tentativo 10 — Le etichette MBFC sulle nostre testate

> **Tentativo autonomo**: `1-etichette`, dati in `Dataset/` (fuori da git), risultati in `Results/`. Il percorso di
> tutti i tentativi è nel README di `Code/WebScience2027`.

**La domanda.** Le etichette NewsGuard che usiamo sono un'esportazione di **agosto 2022** (ultimo aggiornamento delle
schede tra il 2020 e il 2022), mentre gli articoli sono del 2025–2026: in mediana l'etichetta ha 4,4 anni più
dell'articolo. Sono ancora buone? Una seconda fonte indipendente e aggiornata: **Media Bias/Fact Check (MBFC)**, che
per ogni sito dà factual reporting, credibilità, bias e altro.

## I dati (in `Dataset/`, fuori da git)

| file | che cos'è |
|---|---|
| `mbfc_api_2026-10-23.json` | l'esportazione completa dall'**API ufficiale MBFC** (piano gratuito su RapidAPI, scaricata da John il 23 ottobre 2026): 11.176 schede con nome, link alla scheda, categoria di bias, orientamento politico, factual reporting e factual score, credibilità, paese, tipo di media |
| `mbfc_estensione_2026-10-22.json` | il file pubblico dell'estensione ufficiale MBFC per browser (aggiornato ogni giorno; 9.777 siti): solo per il controllo incrociato |
| `newsguard.csv`, `versioni/`, `NuovoScraping/elenco_testate.csv`, `NuovoScraping/dataset_articoli.csv` | collegamenti fisici agli stessi file del tentativo 09 |
| `mbfc_pulito.csv`, `testate_mbfc.csv` | le uscite di `01` e `02` |

I dati MBFC, come quelli NewsGuard, **non si ridistribuiscono** (termini d'uso di MBFC e di RapidAPI); chi li usa cita
MediaBiasFactCheck.com.

## I passi

```
1-etichette/comune.py                 percorsi, livelli MBFC in ordine, dominio registrato, livelli NewsGuard
1-etichette/01_pulisci_mbfc.ipynb     sito -> dominio, siti morti, refusi, factual score: Dataset/mbfc_pulito.csv
1-etichette/02_abbina_testate.ipynb   un dominio una riga, abbinamento alle testate, confronto con NewsGuard e con l'estensione
```

**Pulizia** (`Results/1-etichette/01_passi.csv`): 1.091 siti morti (`DEAD`, `Offline`) messi da parte; 935 schede di
una sezione (`frontiersin.org/journals/...`) non usate; 9.006 domini; 98 domini con più schede (per esempio più TV
sullo stesso sito), 23 in disaccordo su qualche campo (vale la maggioranza, a parità il campo resta vuoto);
credibilità `Medum` → `Medium`; `N/A` della satira e valori fuori elenco → vuoti; un factual score letto da Excel come
data (5,3); 6 factual score fuori dalla fascia del loro factual reporting, segnalati. L'API **non** dà la data di
aggiornamento delle schede.

**Abbinamento**: dominio esatto, oppure dominio registrato della testata se MBFC ha una scheda proprio per quello
(25 casi su 1.634).

## Risultati (23 ottobre)

**Copertura** (`02_copertura.csv`):

| insieme | testate | in MBFC |
|---|---|---|
| tentate nello scraping | 3.127 | 1.634 (52%) |
| con articoli scaricati | 1.046 | 473 (45%) |
| dataset finale | 863 | 432 (50%) |
| valutate (tentativo 09) | 301 | 211 (70%) |

Le mancanti del dataset finale sono soprattutto la rete di Natural News: 274 dei suoi 347 domini non hanno una scheda
MBFC (MBFC valuta naturalnews.com, non i domini satellite).

**Accordo con NewsGuard 2022** (`02_accordo_newsguard.csv`):

| insieme | NG affidabile ~ MBFC credibilità alta/media | NG affidabile ~ MBFC factual ≥ Mostly Factual | Spearman punteggio NG ~ factual score MBFC |
|---|---|---|---|
| dataset finale (431) | 94% | 90% | −0,84 |
| valutate (211) | 88% | 83% | −0,82 |
| valutate, una per gruppo (167) | 88% | 84% | −0,83 |
| tutte le tentate con punteggio NG (1.632) | 96% | 94% | −0,87 |

Le due fonti, a quattro anni di distanza e con metodi diversi, sono **molto d'accordo**: sul binario le etichette
NewsGuard 2022 reggono. I disaccordi (`02_disaccordi_newsguard.csv`) sono soprattutto testate di destra statunitensi che
NewsGuard nel 2022 dava affidabili e che MBFC oggi considera *Questionable* (Townhall, The Blaze, Just The News, Not the
Bee, ...); MBFC è anche più severo sul factual (38 testate affidabili per NewsGuard ma *Mixed* o peggio per MBFC).

**Distribuzioni** (`02_incroci_per_livello_newsguard.csv`, dataset finale): l1 → factual *Very Low*/*Low*, credibilità
bassa, categoria *Conspiracy-Pseudoscience* o *Questionable*, orientamento *Extreme Right*; l2 → *Mixed*, bassa
credibilità, *Questionable*; l3 → *High*/*Mixed*, soprattutto *Right-Center*; l4–l5 → *High*, credibilità alta,
*Least Biased*/*Left-Center*.

**Controllo incrociato** (`02_controllo_estensione.csv`): API ed estensione coincidono nel 99% dei casi su factual e
credibilità; 146 testate abbinate si trovano solo nell'API.

## Rieseguire

Da `1-etichette/`: `01_pulisci_mbfc.ipynb`, poi `02_abbina_testate.ipynb` (in locale, pochi secondi; serve
`Dataset/` come sopra). Le esecuzioni salvate sono in `Results/1-etichette/esecuzioni/`.

## Possibili esperimenti successivi

- I modelli del tentativo 09 con le **etichette MBFC** come obiettivo (factual a 6 livelli, credibilità a 3), sulle
  testate abbinate: il tetto cambia con un'etichetta più recente?
- Un'etichetta **combinata** (testate su cui NewsGuard e MBFC sono d'accordo) per misurare quanto del rumore viene
  dalle etichette.

# Tentativo 12 — Più testate politiche (terzo giro di scraping)

> **Tentativo autonomo**: `0-scraping` → `1-data-cleaning` → `2-dataset` → `3-classificazione`, dati in `Dataset/`
> (fuori da git), risultati in `Results/`. Il percorso di tutti i tentativi è nel README di `Code/WebScience2027`.

**La domanda.** Nel tentativo 11 l'esperimento "solo politici" (testate con il solo argomento politico per NewsGuard
2026) aveva solo 187 testate (143 valutate), e addestrare solo su quelle andava peggio che addestrare su tutte.
Con **più testate politiche** il classificatore a feature e ModernBERT vanno meglio?

## Che cosa si fa (fissato domenica 25 ottobre, prima dei risultati)

1. **Terzo giro di scraping** (`0-scraping/`, lo scraper dei giri di ottobre, stessi parametri: al più 200 articoli per
   testata, 64 processi): le testate della **lista NewsGuard 2026** con la stessa regola dei giri di ottobre (argomento
   che contiene "Political news or commentary", lingua inglese, con un punteggio) **mai tentate prima**: 1.985 testate,
   di cui 825 con il solo argomento politico (`Dataset/newsguard_giro3.csv`).
2. **Unione** con i due giri di ottobre (`0-scraping/unisci.sh`, collegamenti fisici) e **costruzione del dataset** di
   articoli (`0-scraping/costruisci_dataset`: più di 100 parole, niente doppioni, solo inglese; etichette NewsGuard 2026).
3. **Pulizia e reti** (`1-data-cleaning/01`, `02`), gli stessi notebook dei tentativi 06–09, su tutte le testate
   (ottobre + terzo giro): pulizia della Versione 2, copie tra testate, reti dall'HTML.
4. **Dataset solo politici** (`2-dataset/01_dataset_politici`): testate con un punteggio 2026 e il solo argomento
   politico; gruppi = reti ∪ owner ∪ Parent Domain (2026 e, dove c'è, 2022); via le copie tra gruppi diversi; valutate
   = almeno 20 articoli e al più 5 per gruppo; di solo training = almeno 5 articoli, peso min(1, 5/n); 5 fold per
   gruppo stratificati sul livello 2026.
5. **Modelli** (`3-classificazione/`, gli stessi del tentativo 11): il classificatore a feature (TF-IDF + LR, C in
   validazione interna, voto per testata) e ModernBERT-base (ricetta della Versione 2, seme 42, media delle
   probabilità), etichette NewsGuard 2026.
6. **Confronto** (`03_confronto`): feature contro ModernBERT; e l'effetto delle testate in più, sulle testate valutate
   sia qui sia nel tentativo 11, contro i modelli "politici" del tentativo 11 (copiati in `Dataset/riferimento_t11/`).

## Sul server

Tutto in `~/John/Tentativo12`:

- screen `t12_scraping`: `0-scraping/avvia.sh ../Dataset/NuovoScraping_giro3 64` (lanciato alle 15:39; lo scraping
  vecchio sta in `Dataset/NuovoScraping_ottobre`, collegamenti fisici alla copia del server, nessuno spazio in più);
- screen `t12`: `esegui_tutto.sh`, che aspetta la fine dello scraping e poi fa i passi 1–6 da solo (ogni passo si
  riprova fino a 3 volte; se rilanciato riprende dal passo a cui era arrivato).

Dal PC: `./controlla_server.sh`.

```
0-scraping/scraper.py, avvia.sh            lo scraper di ottobre (avvia.sh: lista del terzo giro, cartella NuovoScraping_giro3)
0-scraping/unisci.sh                       unisce ottobre e terzo giro in Dataset/NuovoScraping
0-scraping/costruisci_dataset.ipynb        il dataset di articoli (etichette NewsGuard 2026)
1-data-cleaning/01_pulizia_articoli.ipynb  la pulizia della Versione 2 e le copie tra testate
1-data-cleaning/02_reti.ipynb              le reti di siti (copie, codici, privacy policy, autori, domini)
2-dataset/01_dataset_politici.ipynb        il dataset solo politici con le etichette 2026
3-classificazione/01_feature_tfidf.ipynb   il classificatore a feature
3-classificazione/02_modernbert.ipynb      ModernBERT-base
3-classificazione/03_confronto.ipynb       intervalli e differenze (anche contro il tentativo 11)
esegui_tutto.sh, stato.sh, controlla_server.sh
```

## Risultati (catena finita domenica 25 ottobre alle 18:05)

Tutti i passi sono riusciti al primo tentativo. **Terzo giro**: 1.985 testate tentate, 466 completate (200 articoli),
708 con meno articoli, 744 irraggiungibili (connessione, 403, 503); 115.124 articoli.

**Dataset solo politici** (`Results/2-dataset/01_composizione.csv`):

| | testate | gruppi | l1 | l2 | l3 | l4 | l5 | affidabili |
|---|---|---|---|---|---|---|---|---|
| valutate | 254 | 191 | 46 | 55 | 31 | 82 | 40 | 60% |
| … di cui di ottobre | 140 | 109 | 18 | 37 | 20 | 44 | 21 | 61% |
| … di cui del terzo giro | 114 | 98 | 28 | 18 | 11 | 38 | 19 | 60% |
| tutte (training) | 495 | 230 | 215 | 98 | 37 | 101 | 44 | 37% |
| … di cui del terzo giro | 308 | 116 | 194 | 35 | 13 | 46 | 20 | 26% |

Le testate solo politiche passano da 187 a **495** (valutate da 144 a **254**). Il terzo giro porta soprattutto testate
l1 (194 su 308) in pochi grandi gruppi: nel test entrano al più 5 per gruppo.

**Modelli** (per testata, etichette NewsGuard 2026, intervalli al 95% con bootstrap per gruppo):

| modello | testate | F1 [IC 95%] | ±1 livello | binario |
|---|---|---|---|---|
| feature | 254 valutate | **0,383** [0,315; 0,444] | 0,768 | 0,697 |
| ModernBERT | 254 valutate | 0,312 [0,239; 0,379] | 0,689 | 0,638 |
| maggioranza / a caso | 254 valutate | 0,098 / 0,184 | 0,602 / 0,547 | 0,602 / 0,528 |
| feature, tentativo 12 / tentativo 11 | 133 in comune | 0,320 / 0,279 | 0,729 / 0,752 | 0,632 / 0,639 |
| ModernBERT, tentativo 12 / tentativo 11 | 133 in comune | 0,280 / 0,271 | 0,684 / 0,714 | 0,639 / 0,579 |

- **ModernBERT − feature**: ΔF1 −0,071 [−0,148; +0,008], Δ±1 −0,079, Δbinario −0,059: il feature resta davanti, e qui
  di più che nei tentativi precedenti (al limite della significatività).
- **Più testate**, sulle 133 testate valutate anche nel tentativo 11: feature +0,040 [−0,035; +0,117], ModernBERT +0,009
  [−0,109; +0,118] in F1; sul ±1 livello e sul binario nessun guadagno. Le testate in più aiutano un po' il feature,
  quasi niente ModernBERT, e in nessun caso in modo significativo.

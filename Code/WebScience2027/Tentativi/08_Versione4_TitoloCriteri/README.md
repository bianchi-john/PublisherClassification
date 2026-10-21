# Tentativo 08 — Versione 4: il titolo e i criteri NewsGuard

> **Tentativo autonomo**: `0-scraping` → `1-data-cleaning` → `2-classificazione`, dati in `Dataset/` (fuori da git), risultati in `Results/`. Tutta la catena è qui: `0-scraping/` (lo scraper finale) e `1-data-cleaning/` sono **copie** di quelli del tentativo 06. Nella realtà non sono stati rieseguiti: il dataset in `Dataset/versioni/` è quello del tentativo 06 (rieseguendo `1-data-cleaning` si ottiene lo stesso). **In corso** (21 ottobre): la corsa della notte gira sul server nella vecchia posizione `~/John/WebScience2027/Versione4`; `stato.sh` e `controlla_server.sh` la trovano lì finché non sarà spostata. Il percorso di tutti i tentativi è nel README di `Code/WebScience2027`.

**La domanda.** Dalle Versioni 1–3: sul solo testo dell'articolo i transformer non battono il TF-IDF, perché il segnale
è sparso e lessicale (il genere della testata, lo stile). Che cosa potrebbe vedere un transformer e non un TF-IDF?
Due cose, entrambe dentro il "solo testo" del paper:

1. **Il titolo insieme all'articolo**: il modello legge la coppia *titolo [SEP] articolo* e può confrontarli. Un titolo
   ingannevole o gonfiato rispetto al contenuto è uno dei criteri NewsGuard ("Avoids deceptive headlines"); un TF-IDF
   può contare le parole del titolo, non misurarne la coerenza con il testo.
2. **I criteri NewsGuard come etichette in più**: oltre al livello, il modello impara i 5 criteri visibili negli
   articoli (non pubblica ripetutamente contenuti falsi, raccoglie e presenta le informazioni in modo responsabile,
   corregge gli errori, distingue notizie e opinioni, evita titoli ingannevoli). Più supervisione per ogni testata.
   Gli altri 4 criteri riguardano il sito (proprietà, pubblicità, responsabili, autori) e restano fuori.

**Il disegno** (fissato il 21 ottobre, prima dei risultati):

- lo **stesso dataset** della Versione 2 (`Dataset/versioni`, da `1-data-cleaning`): i due giri del nuovo scraping (13 e 18
  ottobre), la stessa pulizia, le stesse **304 testate di test** (almeno 20 articoli, al più 5 per gruppo), gli stessi
  fold per gruppo, le testate di solo training con il loro peso. Così il confronto con ModernBERT sul solo articolo
  della Versione 2 è diretto, sulle stesse testate e con lo stesso training (`Results/riferimenti_v2/`);
- compito a **5 livelli** (più il binario), metriche della Versione 2;
- il TF-IDF riceve anche lui il titolo, come blocco di parole a parte (`01_feature_titolo`).

**Le prove preliminari** (21 ottobre, mattina): scelgono la ricetta (titolo sì/no, peso dei criteri 0,5 o 1,0) su
una **divisione di sviluppo** (`comune.divisione_sviluppo`): le testate di training del fold 1, cioè i fold 2–5, con
1/7 dei gruppi in validazione. Il fold 1 non si guarda mai. La validazione è piccola: le prove servono a controllare
che tutto funzioni e a fare scelte grossolane, non a dire chi vince.

## File

```
2-classificazione/comune.py                 dati (dalla Versione 2), i 5 criteri, la divisione di sviluppo, metriche
2-classificazione/01_feature_titolo.ipynb   TF-IDF + LR con (o senza) un TF-IDF a parte sui titoli; SVILUPPO per le prove
2-classificazione/02_transformer_titolo.ipynb  ModernBERT: TITOLO (coppia titolo/articolo), LAMBDA_CRITERI (criteri come
                                            etichette in più), SVILUPPO (prove preliminari)
2-classificazione/03_confronto.ipynb        tutti i modelli + i riferimenti della Versione 2, ensemble dei seed, bootstrap
2-classificazione/coda.py                   la coda, in corsie (sviluppo_cpu, sviluppo_gpu; poi quelle della notte)
Results/riferimenti_v2/                     le predizioni della Versione 2 sul solo articolo (TF-IDF, ModernBERT s42/s1/s2)
Results/2-classificazione/sviluppo/         le prove preliminari
```

## Le prove preliminari (21 ottobre, mattina)

Sulla divisione di sviluppo, che ha solo **26 testate valutate** (molti gruppi di validazione sono testate di solo
training): F1 per testata.

| | seed 42 | seed 1 |
|---|---|---|
| TF-IDF, solo articolo | 0,524 | |
| TF-IDF, titolo come blocco a parte | 0,463 | |
| ModernBERT, solo articolo | 0,203 | 0,218 |
| ModernBERT, titolo + articolo | 0,162 | 0,186 |
| ModernBERT, titolo + articolo + criteri (peso 0,5) | 0,142 | 0,242 |
| ModernBERT, titolo + articolo + criteri (peso 1,0) | 0,167 | 0,242 |

Con 26 testate una testata vale 4 punti di accuratezza: le prove non discriminano tra le varianti. Hanno confermato
che tutto funziona (titolo, criteri, memoria 5,1 GB, 15 minuti a prova). Per la notte si tiene la ricetta decisa a
priori: titolo + articolo, con e senza criteri (peso 0,5; il peso 1,0 non dà segni di essere meglio).

## La notte del 21 ottobre (5 fold, 304 testate)

- corsia `notte_cpu`: TF-IDF con il titolo, C = 1 e C in validazione interna (`feature_titolo_tuned`, il riferimento);
- corsia `notte_gpu`: ModernBERT titolo + articolo (`bert_titolo`) e titolo + articolo + criteri (`bert_titolo_criteri05`),
  seed 42, 1, 2;
- il confronto (`03_confronto`) mette accanto TF-IDF e ModernBERT sul solo articolo della Versione 2.

Riferimento fold per fold, ModernBERT solo articolo (Versione 2), F1 per testata: s42 0,436 0,291 0,363 0,347 0,326;
s1 0,308 0,268 0,270 0,412 0,335; s2 0,446 0,318 0,325 0,315 0,358.

## Sul server

```bash
cd ~/John/WebScience2027/Tentativi/08_Versione4_TitoloCriteri/2-classificazione
screen -dmS v4_notte_cpu bash -c 'until ~/John/johnVenv/bin/python coda.py notte_cpu; do sleep 60; done'
screen -dmS v4_notte_gpu bash -c 'until ~/John/johnVenv/bin/python coda.py notte_gpu; do sleep 60; done'
../stato.sh                                   # dal PC: ./controlla_server.sh [risultati]
```

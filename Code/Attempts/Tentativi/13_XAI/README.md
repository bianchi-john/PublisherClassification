# Tentativo 13 — Spiegare i modelli (XAI)

> **Tentativo autonomo**: `1-dataset` → `2-modelli` → `3-xai` (fase 2, in preparazione), dati in `Dataset/` e modelli in
> `Modelli/` (fuori da git), risultati in `Results/`. Il percorso di tutti i tentativi è nel README di `Code/WebScience2027`.

**La domanda.** Il classificatore a feature (TF-IDF) non si batte: ModernBERT va pari o peggio in tutte le
configurazioni. Ma *per quali ragioni* decidono i due modelli? Se il transformer, anche con un'accuratezza pari o
leggermente più bassa, si appoggia a segnali più sensati (stile, fonti, cautela) e meno a scorciatoie (nomi, argomenti,
orientamento, cornice del sito), è un risultato interessante per il paper.

## Fase 1: modelli salvati e performance (`1-dataset`, `2-modelli`; lanciata lunedì 26 ottobre)

- **Dataset** (`1-dataset/01_dataset`): le testate solo politiche del tentativo 12 (NewsGuard 2026, argomento solo
  "Political news or commentary", due giri di ottobre + terzo giro) **senza quelle con meno di 10 articoli** (172, di
  cui 161 l1, quasi tutte di una rete di circa 160 siti con 7 articoli ciascuno): **323 testate, 254 valutate** (le
  stesse del tentativo 12), 42.219 articoli. Stessi fold per gruppo; peso di solo training ricalcolato. Orientamento e
  paese NewsGuard servono solo per l'analisi per genere, mai come ingresso. Lo squilibrio di genere (poche testate Left
  inaffidabili, poche Right affidabili) non si corregge: è com'è fatto l'ecosistema delle testate.
- **Modelli** (`2-modelli/01`, `02`): il classificatore a feature (TF-IDF + LR, C in validazione interna) e ModernBERT-base
  (ricetta della Versione 2, seme 42), etichetta NewsGuard 2026, 5 fold per gruppo: ogni testata valutata è prevista
  da un modello che non ha visto né lei né il suo gruppo. Si salvano:
  - le **probabilità delle 5 classi** di ogni articolo di test (`Results/2-modelli/...articoli.csv`);
  - i **modelli di ogni fold** e un **modello finale addestrato su tutte le testate** (per usarli in un tool):
    `Modelli/feature/fold<k>.joblib`, `finale.joblib` (vettorizzatore TF-IDF + regressione logistica:
    `p = modello.predict_proba(tfidf.transform([testo]))`) e `Modelli/modernbert/fold<k>/`, `finale/` (formato Hugging
    Face: `AutoModelForSequenceClassification.from_pretrained(cartella, reference_compile=False)` più il tokenizer;
    circa 575 MB ciascuno). Classi: 0 = l1 ... 4 = l5.
- **Performance** (`2-modelli/03_confronto`): metriche con intervalli (bootstrap per gruppo), **per fold**, **per
  orientamento**, le probabilità medie per classe di ogni testata (`testate_probabilita.csv`) e il riferimento "solo
  orientamento + paese" (senza testo).

Sul server: `~/John/Tentativo13`, screen `t13` (`esegui_fase1.sh`). Dal PC: `./controlla_server.sh`; alla fine
`./scarica_modelli.sh` porta qui modelli (in `Modelli/`, ignorata da git) e risultati.

## Fase 2: le spiegazioni (`3-xai`; parte da sola alla fine della fase 1, screen `t13_xai`, `esegui_fase2.sh`)

```
3-xai/comune_xai.py                  lessici, categorie, campione, Integrated Gradients per i due modelli
3-xai/01_campione_e_categorie.ipynb  il campione (300 articoli), i termini di parte per fold, le categorie dei token (spaCy)
3-xai/02_attribuzioni.ipynb          le attribuzioni dei due modelli (GPU), con i controlli di completezza
3-xai/03_analisi.ipynb               categorie (anche per orientamento e per fold), parole, frasi, accordo, robustezza
3-xai/04_maschere.ipynb              le maschere (GPU), con il controllo a caso
```

Risorse esterne (scaricate il 26 ottobre con il permesso di John, in `Dataset/lessici/`, fuori da git): Moral
Foundations Dictionary 2.0 (OSF), NRC Emotion Lexicon e NRC VAD Lexicon (saifmohammad.com, uso di ricerca); sul server
`captum` 0.9 e `spacy` 3.8 con `en_core_web_sm` 3.8 (installati senza cambiare le altre librerie).

Prova in piccolo (fold 1, 10 articoli, sulla CPU) superata: le attribuzioni del TF-IDF sono esatte (completezza 0);
per ModernBERT IG non torna esattamente (errore mediano circa 0,6 sul punteggio; non migliora con più passi: è un limite
noto di IG sui transformer), ma le quote per categoria coincidono con quelle di un secondo metodo, gradiente × input
(correlazione 0,98): le conclusioni sulle categorie non dipendono dal metodo.

Fissata prima dei risultati (discussa con John il 26 ottobre):

- **un solo numero per parola**: quanto spinge verso "affidabile" (l3–l5) o "non affidabile" (l1–l2), cioè il log del
  rapporto tra le probabilità dei due gruppi di livelli; Integrated Gradients per ModernBERT (64 passi, fp32) e per il
  TF-IDF (50 passi), stesso metodo e stessa quantità; per ModernBERT anche gradiente × input come controllo;
- un **campione di circa 300 articoli del test**, bilanciato per livello, spiegati da **tutti e due** i modelli, ognuno
  con il modello del fold in cui era nel test;
- tre livelli: **parole** (solo esempi), **categorie** di parole (il risultato principale: che quota di importanza va
  a nomi, termini di parte, morale e religione, emozioni, sensazionalismo, cautela, fonti e attribuzione, numeri,
  rivolgersi al lettore, cornice del sito, argomento, parole funzione; da liste pubblicate dove esistono), **frasi**
  (le frasi che spostano di più la decisione e i loro tratti: citazioni, numeri, domande ed esclamazioni, prima e
  seconda persona, posizione, lunghezza); tutto anche **per orientamento**;
- **maschere**: si tolgono dal test le parole di una categoria (nomi e termini di parte, argomento, cornice) e si
  misura quanto peggiora ciascun modello;
- **stabilità**: le categorie fold per fold.

## Risultati (26 ottobre; fase 1 finita alle 12:08, fase 2 alle 13:15)

**Performance** (254 testate valutate, `Results/2-modelli/confronto/`):

| modello | F1 a 5 livelli [IC 95%] | ±1 livello | binario |
|---|---|---|---|
| feature | **0,390** [0,320; 0,453] | 0,756 | 0,677 |
| ModernBERT | 0,310 [0,236; 0,373] | 0,673 | 0,638 |
| solo orientamento + paese (senza testo) | 0,254 [0,206; 0,299] | 0,693 | **0,724** |
| maggioranza | 0,098 | 0,602 | 0,602 |

ModernBERT − feature: −0,080 [−0,159; −0,003], **significativo**; il feature vince in 4 fold su 5. ModernBERT non è
significativamente meglio dei soli metadati (+0,056 [−0,025; +0,127]); il feature sì (+0,136 [+0,061; +0,205]). Sul
binario i soli metadati battono tutti e due i modelli di testo.

**Categorie** (300 articoli, quota di importanza **senza la punteggiatura**, che il TF-IDF per costruzione non vede;
`Results/3-xai/analisi/categorie_senza_punteggiatura.csv`):

| categoria | feature | ModernBERT | Δ [IC 95%] |
|---|---|---|---|
| parole funzione | 0,286 | **0,406** | +0,120 [+0,114; +0,127] |
| argomento | 0,135 | 0,117 | −0,017 |
| nomi propri | 0,126 | 0,111 | −0,015 |
| emozioni | 0,081 | 0,062 | −0,019 |
| termini di parte | 0,071 | 0,050 | −0,021 |
| numeri e date | 0,055 | 0,053 | −0,003 |
| morale e religione | 0,030 | 0,024 | −0,005 |
| sensazionalismo | 0,029 | 0,027 | −0,002 |
| cautela ed enfasi | 0,024 | 0,018 | −0,005 |
| fonti e attribuzione | 0,023 | 0,016 | −0,007 |
| rivolgersi al lettore | 0,011 | 0,014 | +0,003 |
| cornice del sito | 0,002 | 0,001 | −0,001 |

ModernBERT mette in più il 21% dell'importanza sulla punteggiatura. Le differenze sono stabili nei 5 fold e uguali nei
tre orientamenti. Con un secondo metodo (gradiente × input) le quote di ModernBERT sono quasi le stesse.

- **Profili simili, con una differenza netta**: ModernBERT sposta importanza sulle **parole funzione** (e sulla
  punteggiatura), cioè sulla struttura del testo, e ne toglie un po' a nomi, termini di parte, emozioni e argomento;
  ma ne toglie anche a fonti e attribuzione e a cautela ed enfasi.
- **Parole** (`parole.csv`): le più forti del feature sono di parte, di argomento e di geografia (verso affidabile:
  *reform, bipartisan, economy, britain, uk, ohio, midterm, democracy, labour*; verso non affidabile: *democrat, biden,
  rally, socialist, china, arizona, georgia, massive*). Quelle di ModernBERT sono più di stile e di cautela (verso
  affidabile: *ceasefire, perhaps, appropriate, particularly, potentially, increasingly, spokesperson, insist*; verso
  non affidabile: linguaggio di violenza e cronaca, *deadly, fight, injure, defeat, witness, scene*).
- **Frasi** (`frasi_tratti.csv`): tutti e due i modelli si appoggiano soprattutto alle frasi con citazioni (25%/23%
  contro il 19% di tutte le frasi), attribuzioni (38%/35% contro 24%) e numeri (52%/51% contro 38%); le esclamazioni
  spingono verso "non affidabile". Su questo i due modelli quasi coincidono.
- **Accordo**: sui singoli token quasi nullo (Spearman mediano 0,01): guardano parole diverse, con profili per categoria
  simili.
- **Maschere** (`Results/3-xai/maschere/`, al più 50 articoli per testata): il feature regge a tutto tranne che alla
  **cornice del sito** (−0,020 [−0,038; −0,004]: un po' di scorciatoia sul sito); ModernBERT perde 6–7 punti togliendo
  nomi e termini di parte, ma **altrettanto togliendo lo stesso numero di parole a caso**: non dipende dai nomi in
  particolare, è fragile a qualunque taglio del testo. Nessuna differenza tra i due modelli è significativa tranne la
  cornice.

**In sintesi**: ModernBERT non decide per ragioni "più giuste" in modo misurabile. Le sue parole più forti sembrano più
di stile e di cautela, quelle del feature più di parte e di geografia, ma in quantità i due modelli usano le stesse
categorie, ModernBERT un po' meno nomi e termini di parte e un po' meno fonti e cautela, e molto di più parole
funzione e punteggiatura. Il feature resta più accurato e più robusto ai tagli del testo; il suo unico punto debole
misurato è una piccola dipendenza dalla cornice del sito.

# Modifiche al testo del paper

Lista di lavoro per portare `main.tex` ai risultati del 17/11/2026. Va seguita nell'ordine del paper; i numeri nuovi
vengono dalle tabelle del passo 15 in `Dataset/tentativi/20_FromScratch/paper_*.csv`, mai ricopiati a mano da altre
fonti. Tra parentesi quadre il punto di `PIANO_REVISIONE.md` a cui risponde ogni modifica.

**Regole decise con John:** il MAE è la misura principale e la ρ è secondaria; la notorietà non entra; per le sorelle
il messaggio è che conta la somiglianza dei testi, la parentela è solo un indizio.

Stato: ⬜ da fare · 🟡 in corso · ✅ fatto · ❓ decisione di John

---

## 0. Decisioni prima di scrivere ❓

- ❓ **Metriche aggiuntive** (RMSE, R², AUC per affidabile sì/no a 60) in una tabella in appendice, calcolate dalle
  previsioni salvate (minuti, nessun addestramento).
- ❓ **Dove va la scala.** Proposta: una sottosezione «How much the protocol matters» al posto di «Robustness to the
  anonymization». Le righe testo originale / solo regole / GLiNER della scala sostituiscono la Tabella `tab:anon`.
  Attenzione: `tab:anon` viene dal passo 12, con le sei regole della prima esecuzione e ModernBERT; L3 della scala usa
  invece le regole del passo 3 attuale. Non vanno mescolate nella stessa tabella: o la scala sola, o `tab:anon` in
  appendice.
- ❓ **Figura per la scala** (MAE per gradino, e per le sorelle con e senza sorella), oppure solo la tabella.
- ❓ **Una frase sulle date nei limiti**: gli articoli sono dal 2025 e Mistral-Small-24B-Base-2501 è di gennaio 2025,
  quindi il modello non ha visto questi articoli in pre-addestramento (può conoscere le testate).

## 1. Abstract ⬜ [6]

- ⬜ Il MAE per primo: Mistral 15,8 punti contro 17,8 di TF-IDF (significativo anche con Holm) e 22,1 della costante
  migliore (la mediana, non più la media 22,6). La ρ (0,735 contro 0,674) senza dire «significativamente».
- ⬜ «With five articles it already ranks better than TF-IDF reading all» → in MAE: con 5 articoli 16,1 contro 17,8
  (Holm p 0,006); sulla ρ non è significativo.
- ⬜ «Masking every outlet and journalist leaves its results unchanged» → «no significant difference», più una frase
  sulla scala: con una testata dello stesso proprietario in addestramento l'errore scende di 7–12 punti sulle testate
  interessate, quasi come vedere la testata stessa.
- ⬜ Le baseline nuove in mezza frase: chiedere al modello (zero-shot) fa peggio della costante (MAE 24,4); ModernBERT
  congelato vale quanto addestrato.
- ⬜ Orientamento: il divario a parità di voto (circa 9 punti) è quello che produce la compressione verso il centro
  con medie diverse per gruppo; Mistral è calibrato per gruppo (+0,8). Non «assign left-leaning outlets 7–10 points
  more» come pregiudizio.

## 2. Introduzione ⬜ [2, 6]

- ⬜ Contributi: il punto 3 dice «significantly better than TF-IDF on every measure and with any number of articles»:
  vero solo per il MAE (e per Mistral in tutte le curve con Holm per modello); riscrivere.
- ⬜ Aggiungere il contributo della scala: quanto contano split per testata, sorelle e anonimizzazione, misurato.
- ⬜ Cosa del voto vede il testo: i criteri NewsGuard richiedono verifiche esterne; metafora del DNA (un capello rivela
  tratti, non identità). La curva per numero di articoli diventa «quanti capelli servono».

## 3. Related work ⬜ [8]

- ⬜ Lavori più vicini: Rashkin 2017, Baly 2018, Horne 2019, Bozarth & Budak 2020, Fairbanks 2018.
- ⬜ Tabella di confronto: solo testo, editori separati, gruppi separati, anonimizzato, punteggio continuo.
- ⬜ Bibliografia: niente «verify» stampati; unire il doppione Przybyła & Soto.

## 4. Problema e metriche (§3) ⬜ [6, 7, 10]

- ⬜ Dire che il MAE è la misura principale (distanza in punti dal voto, ciò che serve a chi usa il sistema) e la ρ
  la secondaria (ordinamento).
- ⬜ Baseline: aggiungere la mediana (la costante migliore per il MAE, 22,1); correggere «the no-text baseline … is the
  best constant prediction»; lasciare la media e «solo orientamento».
- ⬜ «Within 20 points corresponds to about one NewsGuard level»: i livelli sono larghi 40, 20, 15, 25 punti;
  riformulare. [10]
- ⬜ Statistica: 10.000 ricampionamenti per la tabella principale e 2.000 per le curve (non 1.000); Holm dentro ogni
  famiglia (tabella principale: 28 confronti; curve: per modello; affermazioni del testo).
- ⬜ Fonte dell'orientamento: il campo «Orientation» di NewsGuard. [7]

## 5. Dataset (§4) ⬜ [7, 10]

- ⬜ «Owner» = proprietario registrato da NewsGuard.
- ⬜ Critiche di parzialità rivolte a NewsGuard: una frase con riferimento.
- ⬜ Chi resta fuori (`paper_selection.csv`): le 619 testate mai raccolte hanno voto medio 41 (contro 68,5 del
  dataset); le 170 tolte dalla deduplicazione sono per l'89% nel livello più basso e per l'88% di destra (la rete di
  164 siti). Una frase e un rimando all'appendice.
- ⬜ Incoerenze da verificare sui dati: 704 contro 797 testate con sorelle; 81 contro 61 sorelle dopo la
  deduplicazione; «up to 200» contro «from 10 to 201» articoli. [10]
- ⬜ §4.3 (gruppi): rimandare alla scala, che ora misura l'effetto; il 74% delle testate ha lo stesso voto identico
  delle sorelle (`paper_owner_scores.csv`).

## 6. Setup sperimentale (§5.1) ⬜ [4, 10]

- ⬜ Modelli nuovi: Qwen3-Embedding-0.6B congelato + ridge (separa la dimensione, ed è il modello leggero del tool),
  ModernBERT congelato + ridge, Mistral-Small-24B-Instruct zero-shot (prompt con i nove criteri, decodifica greedy,
  primo intero; le 24 risposte senza numero prendono la media di addestramento).
- ⬜ Mistral: lo strato si sceglie sui fold interni tra 27 e 40, ed esce il 27 in tutti i fold. Togliere «fixed before
  the experiments».
- ⬜ Griglia della penalità: «inside the grid in every fold» non vale più (Qwen3-0,6B e ModernBERT congelato scelgono
  spesso 10); allargando fino a 0,01 la ρ cambia meno di 0,001.
- ⬜ Iperparametri di ModernBERT da ricontrollare con il codice. [10]
- ⬜ Tempi di calcolo aggiornati.

## 7. Risultati per articolo e per testata (§5.2–5.3) ⬜ [4, 6, 10]

- ⬜ Tabella per articolo: aggiungere i modelli nuovi (`paper_article_vs_outlet.csv`).
- ⬜ Tabella principale (`paper_main_table.csv`): colonna MAE per prima; righe nuove: mediana, Qwen3-0,6B, ModernBERT
  congelato, zero-shot; significatività con Holm; grassetto di Qwen su «within 20» (69,9 > 68,1). [10]
- ⬜ Riscrivere i punti elenco:
  - entrambi gli LLM congelati abbassano il MAE in modo significativo (Mistral −2,0, Qwen −1,5); la ρ più alta non è
    significativa dopo Holm;
  - la dimensione conta: Qwen 8B contro 0,6B, −2,5 di MAE e +0,085 di ρ, significativi;
  - ModernBERT congelato e addestrato si equivalgono (0,619 contro 0,610): l'addestramento non aggiunge nulla;
  - zero-shot: ρ 0,35 ma MAE 24,4, peggio della costante; chiedere il voto non funziona, leggere con una testa lineare
    sì (+0,39 di ρ, −8,6 di MAE, `paper_claims.csv`);
  - compressione: Mistral prevede tra 37 e 96.

## 8. Quanti articoli servono (§5.4) ⬜ [6]

- ⬜ Holm per modello (`paper_curves.csv`): Mistral significativo per ogni k su entrambe le misure; Qwen sul MAE per ogni
  k, sulla ρ non per 20, 50 e tutti.
- ⬜ «Con cinque articoli» solo in MAE (16,1 contro 17,8).

## 9. La scala delle fughe (nuova, al posto di §5.5) ⬜ [1]

- ⬜ Disegno: test sempre sulle stesse 226 testate e 25.524 articoli; gradini split per articolo, sorelle sempre in
  addestramento, testo originale, solo regole, GLiNER; ridge riaddestrata in ogni fold.
- ⬜ Tabella (`paper_ladder.csv`): MAE e ρ per gradino. Split per articolo: MAE 8,7–10,7 (−5,6…−7,4); testo originale e
  regole entro 0,4 punti da GLiNER.
- ⬜ Sorelle (31 testate con una sorella): MAE da 18–20 a 8–12 (−11,6 TF-IDF, −6,9 Qwen, −8,7 Mistral), quasi come lo
  split per articolo; le 195 senza sorella non migliorano (+0,4…+0,9); sulle 226 l'effetto si diluisce (n.s.).
- ⬜ Somiglianza (`paper_sisters_similarity.csv`, `paper_sisters_gain.csv`): da 0,19 a 0,60; per il 70% degli articoli
  il più simile in addestramento è di una sorella; il 50% ha una quasi-copia; il guadagno segue la somiglianza
  (ρ 0,60 per TF-IDF; 0,25–0,32 per gli LLM, non significativo). Limite: somiglianza misurata sulle parole.
- ⬜ Voto dentro il proprietario (`paper_owner_scores.csv`): 74% identico, scarto medio 3,8 contro 29,6.
- ⬜ Messaggio: ogni testo simile in addestramento con lo stesso voto, della stessa testata o di una sorella, abbassa
  l'errore in modo artificiale; le regole del dataset servono a questo.
- ⬜ Nota: con le sorelle Qwen sceglie la penalità più bassa in 3 fold su 5, quindi il suo calo è semmai sottostimato.

## 10. Errori per livello (§5.6) ⬜ [6]

- ⬜ Numeri da `paper_errors.csv` (errore con segno per livello, pendenze); «nessuna differenza significativa» invece
  di «invariato».

## 11. Orientamento politico (§5.7) ⬜ [3]

- ⬜ Riscrivere con `paper_orientation.csv`: divario a parità di voto con retta, spline ed effetti fissi per livello
  (Mistral 9,6 / 8,8 / 8,7); divario prodotto dalla sola compressione (12,6); calibrazione per gruppo (Mistral +0,8,
  Qwen 4,8, TF-IDF 7,1, ModernBERT 12,8, zero-shot 24,8).
- ⬜ Conclusione: non è un pregiudizio del modello ma la conseguenza delle medie diverse per gruppo e della
  compressione (Kleinberg et al. 2017; Chouldechova 2017); a parità di previsione Mistral dà lo stesso voto vero ai due
  gruppi.
- ⬜ Allineare discussione, etica e lavori futuri («rate right-leaning outlets about ten points lower», «reduce the
  orientation gap»). [10: «circa dieci punti» contro 6,7–9,6]

## 12. Tool (§6) ⬜ [9]

- ⬜ Un paragrafo di disponibilità; il resto (fedeltà, leakage guard, modelli personalizzati) nel supplemento.
- ⬜ Correggere «not even indirectly».

## 13. Discussione e conclusione (§7) ⬜ [6, 7]

- ⬜ Numeri nuovi, MAE per primo; DNA.
- ⬜ **La spiegazione cambia.** Il testo dice che il modello addestrato memorizza mentre una testa lineare su
  rappresentazioni congelate no, e che la dimensione non spiega il risultato. Ora ModernBERT congelato vale quanto
  addestrato, e Qwen 8B batte 0,6B: conta la dimensione del pre-addestramento, non il congelamento. Riscrivere
  come ipotesi.
- ⬜ Meccanismi come ipotesi; compressione «in parte dal disegno»; «nuove o piccole» e uso pratico come prospettiva.
- ⬜ Limiti: la scala (che cosa succede senza le regole), somiglianza lessicale, critiche a NewsGuard, eventuale frase
  sulle date.

## 14. Figure ⬜

- ⬜ `paper_figures.ipynb` legge i `paper_*.csv`; poi `figures/analyses.py` in `_da_cancellare`.
- ⬜ Aggiornare `results`, `curves`, `errors`, `scatter` con i modelli nuovi dove servono (stile `tab`, tratteggi e
  marcatori).
- ⬜ `groups` e `network`: numeri coerenti con il testo (704/797, 81/61).
- ⬜ Eventuale figura della scala (punto 0).

## 15. Chiusura ⬜

- ⬜ Ogni numero del testo ricontrollato contro i `paper_*.csv`.
- ⬜ Compilare il PDF: niente riferimenti rotti, tabelle che escono dai margini o `\todo` rimasti.
- ⬜ Aggiornare `PIANO_REVISIONE.md` (stato e registro).

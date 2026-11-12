# Piano per rendere il paper "a prova di revisore" (rivista Q1)

Documento vivo: lo aggiorniamo a ogni esperimento. Ogni voce ha un **ID**, una **priorità**, uno **stato** e un
**criterio di "fatto"**. Quando si chiude una voce si aggiorna la tabella, si scrive il risultato nella voce e si
aggiunge una riga al **registro** in fondo.

- Stati: ⬜ da fare · 🟡 in corso · ✅ fatto · ⛔ bloccato · ✖ scartato (con il motivo)
- Priorità: **P1** critica (senza, rischio di rigetto) · **P2** importante (rende il paper solido) · **P3** rifinitura
- Costo: tempo di calcolo stimato (CPU = server o PC; GPU = V100 del server, condivisa) più il lavoro di scrittura
- Bozza: `Paper/Journal/main.tex` · grafici: `Paper/Journal/figures/paper_figures.ipynb` · pipeline: `Code/Pipeline`

Ultimo aggiornamento: 12/11/2026.

---

## 0. Blocchi esterni e decisioni aperte

| ID | Cosa | Stato | Effetto |
|---|---|---|---|
| X1 | **GPU del server inutilizzabile**: `nvidia-smi` → "Failed to initialize NVML: Driver/library version mismatch" (12/11) | ⛔ | blocca tutte le voci GPU (A1–A3 per ModernBERT e LLM, B1, C1, C2, C4, J2b). Serve un riavvio o un intervento degli amministratori di artes4 |
| X2 | **Licenza NewsGuard**: possiamo pubblicare previsioni per dominio reale, ottenute da modelli addestrati sui loro voti? | ✅ | Risolto (John, 12/11): **non** si rilascia nessuna etichetta (né voti, né livelli, né orientamento, né altre); si possono **presentare analisi** delle etichette e **rilasciare i modelli** addestrati su di esse. Il rilascio della sezione 6 è coerente |
| D1 | Media Bias/Fact Check come **secondo valutatore** (validità esterna)? Era stato tolto dal paper | ⬜ | I1, D5 |
| D2 | Metriche di **classificazione ricavate dai punteggi** (AUC affidabile sì/no, F1 sui 5 livelli) per confrontarsi con la letteratura? La classificazione era stata esclusa | ⬜ | L1 |
| D3 | **Calibrazione** delle previsioni compresse (era lo "stiramento", escluso)? | ⬜ | K1 |
| D4 | **Annotazione manuale** di un campione per misurare l'anonimizzazione (John non può; coautori?) | ⬜ | J3 |
| D5 | **Rivista bersaglio** (formato, lunghezza, taglio) | ⬜ | S1–S4 |
| D6 | **Licenze** proposte nel paper: codice e modelli Apache-2.0, previsioni e split CC BY 4.0 | ⬜ | R1 |
| D7 | Le **analisi laterali** (passi 10 e 11: altri regressori, meta-articoli) in appendice come robustezza? | ⬜ | B4, E2 |

### Raccomandazioni sulle decisioni
- **D1: sì**, in una sottosezione di validità esterna. Il tentativo 17 mostrava che i grandi modelli concordano con
  NewsGuard quanto MBFC (ρ ≈ 0,70 su 170 testate): risponde alla critica "un solo valutatore" senza nuovi modelli.
- **D2: sì**, come metriche secondarie ricavate dagli stessi punteggi, senza classificatori nuovi.
- **D3: sì, ma come analisi post-hoc** con la calibrazione stimata solo sui fold di training. ρ non cambia; il MAE agli
  estremi sì. Senza, la domanda "perché non calibrate?" resta senza risposta.
- **D4:** se nessuno può annotare, si usa J1 (riferimento "silver") e si dichiara il limite.
- **D5:** candidati Q1: *Information Processing & Management*, *EPJ Data Science*, *Online Social Networks and Media*,
  *ACM Transactions on the Web*, *IEEE TCSS* (ripresentazione con risposta punto per punto).
- **D7: sì** per il passo 10 (il vantaggio non dipende dal regressore, tetto ≈ 0,75): risponde a "perché proprio la
  ridge?" e a R2.1 (i modelli per testata non aiutano).

---

## 1. Stato generale

| ID | Voce | Pri. | Tipo | Costo stimato | Dipende da | Stato |
|---|---|---|---|---|---|---|
| A1 | Ablazione: split per **articolo** (fuga di testata) | P1 | CPU + GPU | TF-IDF e congelati ~1 h CPU; ModernBERT ~8 h GPU | X1 per ModernBERT | ⬜ |
| A2 | Ablazione: split per testata **con reti e copie** (senza gruppi né deduplicazione) | P1 | CPU + GPU | anonimizzare ed estrarre ~15.000 articoli in più: GLiNER ~1 h, Qwen ~1 h, Mistral ~2,5 h GPU; poi CPU | X1 | ⬜ |
| A3 | Ablazione: **testo non anonimizzato** | P1 | CPU + GPU | TF-IDF ~10 min CPU; Qwen ~2 h, Mistral ~5 h, ModernBERT ~8 h GPU | X1 | ⬜ |
| B1 | Baseline **LLM zero-shot** (punteggio 0–100 chiesto a un modello instruct) | P1 | GPU | ~6 h GPU | X1 | ⬜ |
| B2 | Baseline **classificatore di credibilità per articolo** già pubblicato | P1 | GPU/CPU | ~1–2 h, più la ricerca di un modello adatto | X1 | ⬜ |
| B3 | Baseline **feature stilistiche** interpretabili | P1 | CPU | spaCy sui 25.524 articoli ~2 h CPU | — | ⬜ |
| B4 | Modello **per testata** (vettore medio, impara direttamente sulle testate) | P2 | CPU | già fatto nel passo 10: estrarre e riportare | D7 | ⬜ |
| C1 | **ModernBERT congelato + ridge** contro ModernBERT addestrato | P1 | GPU + CPU | ~30 min GPU | X1 | ⬜ |
| C2 | **Qwen3-Embedding 0,6B / 4B / 8B** congelati + ridge (dimensione a parità di metodo) | P1 | GPU + CPU | ~3 h GPU | X1 | ⬜ |
| C3 | **Strato di Mistral scelto dentro i fold** (27 contro 40; curva per strato) | P1 | CPU (+ GPU) | 27 contro 40: ~10 min CPU (embedding già salvati); curva completa ~5 h GPU | — (curva: X1) | ⬜ |
| C4 | **ModernBERT ottimizzato** (learning rate ed epoche sui fold interni) e 3 seed | P1 | GPU | ~25–40 h GPU | X1 | ⬜ |
| C5 | Riscrivere le **affermazioni sulla dimensione dei modelli** in base a C1–C4 | P1 | scrittura | 1 h | C1–C4 | ⬜ |
| D1 | Bias politico: **modello per testata** con spline del voto, effetti fissi per livello, paese, n. articoli; test a permutazione | P1 | CPU | ~1 h | — | ⬜ |
| D2 | Bias politico **per paese** (USA contro gli altri) | P1 | CPU | ~30 min | — | ⬜ |
| D3 | **Mitigazione**: ridge con pesi bilanciati per orientamento | P2 | CPU | ~30 min | — | ⬜ |
| D4 | Lettura del bias: **criteri NewsGuard** per orientamento a parità di voto | P2 | CPU | ~1 h | — | ⬜ |
| D5 | Orientamento da **un'altra fonte** (MBFC bias) | P2 | CPU | ~1 h | D1 decisione | ⬜ |
| E1 | **Aggregazione misurata alla stessa unità** (k = 1 contro tutti, per testata) al posto di "articolo contro testata" | P1 | CPU + scrittura | ~30 min | — | ⬜ |
| E2 | **Funzioni di aggregazione** (media, mediana, media troncata, voto) | P2 | CPU | ~20 min | — | ⬜ |
| F1 | **Bias di selezione**: chi si perde da 1.293 a 674 a 226 (voto, orientamento, paese) | P1 | CPU | ~1 h | — | ⬜ |
| G1 | **Più split** (5 seed): TF-IDF e congelati | P2 | CPU | ~2 h | — | ⬜ |
| G2 | Più split: ModernBERT | P2 | GPU | ~8 h GPU per seed | X1 | ⬜ |
| M1 | **Confronti multipli** (Holm) su tabella e curve; test della frase "5 articoli di Mistral battono TF-IDF con tutti" | P1 | CPU | ~1 h | — | ⬜ |
| H1 | Che cosa leggono i modelli: **correlazione con indici di stile** | P2 | CPU | ~1 h dopo B3 | B3 | ⬜ |
| H2 | **Parole più pesanti** di TF-IDF (per segno) e categorie | P2 | CPU | ~30 min | — | ⬜ |
| H3 | **Controllo per argomento** (temi con un topic model; risultati dentro i temi) | P2 | CPU | ~2 h | — | ⬜ |
| H4 | **Analisi qualitativa degli errori** (tipi di testata sbagliati, senza nomi) | P2 | CPU + scrittura | ~2 h | — | ⬜ |
| I1 | Validità esterna con **MBFC** | P2 | CPU | ~1 h (codice del 17 e `mbfc.py`) | D1 | ⬜ |
| I2 | **Tempo**: risultati per data degli articoli (più vecchi contro più recenti) | P2 | CPU | ~1 h | — | ⬜ |
| I3 | **Testate non politiche** come test fuori dominio | P3 | CPU + GPU | pulizia, anonimizzazione, embedding: 1–2 giorni | X1 | ⬜ |
| J1 | Qualità dell'anonimizzazione con un **secondo riconoscitore** (riferimento "silver") | P2 | CPU | ~2 h | — | ⬜ |
| J2 | **Identità residua**: quanto il testo anonimo riconosce ancora la testata (probe) | P2 | CPU (+ GPU) | ~1 h su embedding esistenti; confronto con testo grezzo dopo A3 | (A3) | ⬜ |
| J3 | **Campione annotato a mano** (~100 articoli) | P3 | persone | ~4 h di annotazione | D4 | ⬜ |
| J4 | Verificare o rifare il dato "GLiNER da solo lascia il nome nel ~25% degli articoli" | P2 | GPU | ~1 h GPU (GLiNER sul testo originale) | X1 | ⬜ |
| K1 | **Calibrazione** monotona stimata sui fold di training | P2 | CPU | ~30 min | D3 | ⬜ |
| L1 | **AUC e F1** ricavati dai punteggi (confronto con letteratura e con il paper precedente) | P2 | CPU | ~30 min | D2 | ⬜ |
| R1 | **Rilascio**: repository `publisher-evaluation` su GitHub, LICENSE nel repository dello studio, DOI Zenodo per modelli e previsioni | P1 | lavoro | ~2 h | D6 | ⬜ |
| S1 | **Bibliografia**: verificare le voci `verify`, aggiungere i lavori 2024–2026 | P1 | scrittura | ~1 giorno | — | ⬜ |
| S2 | **Affermazioni da ridimensionare** e precisazioni (elenco nella voce) | P1 | scrittura | ~2 h | C1–C4 per una parte | ⬜ |
| S3 | **Appendice / materiale supplementare** (per fold, iperparametri, analisi di robustezza) | P2 | scrittura | ~1 giorno | — | ⬜ |
| S4 | **Formato della rivista**, lunghezza, sezione del tool accorciata | P2 | scrittura | ~1 giorno | D5 | ⬜ |
| S5 | **Lettera di risposta** alle critiche TCSS (anche se cambia rivista: serve come controllo) | P2 | scrittura | ~3 h | tutto | ⬜ |

---

## 2. Le critiche del rigetto TCSS e dove le risolviamo

Fonte: `Presentazioni/Riunione dopo IrForGood/materiali/reviews.txt` (TCSS-2026-06-1033, 24/09/2026; *reject*:
"limited novelty, insufficient rigor in technical analysis, inadequate experimental work").

| Critica | Oggi | Voci |
|---|---|---|
| R1.1 Poca novità di metodo (modelli pronti + voto) | aperta: frozen LLM + ridge non è un metodo nuovo | **A1–A3** (dimostrare che il protocollo conta), S2 (presentarlo come studio di misura e di protocollo) |
| R1.2 Il guadagno dell'aggregazione cambia unità e pesi | **ancora presente** (Tabella Stage 1, Fig. results c) | **E1**, E2 |
| R1.3 Bias politico confuso con il livello | parziale (controllo lineare sul voto) | **D1**, D2, D4 |
| R1.4 Pseudo-replicazione (articoli come indipendenti) | risolta (tutto per testata) | — |
| R1.5 Indizi dell'editore e artefatti non esclusi | quasi risolta (trafilatura, deduplicazione, anonimizzazione, due anonimizzazioni) | **A3**, J1, J2 |
| R1.6 Bias di selezione dello scraping | aperta | **F1** |
| R1.7 Dimensione del modello non controllata | **ancora presente** nella Discussione | **C1–C5** |
| R1.8 Mancano intervalli e test | risolta (bootstrap, confronti appaiati) | M1, G1 |
| R2.1 Perché a livello di testata se è solo aggregazione? | parziale | **B4**, E1 |
| R2.2 Nessun confronto con modelli per articolo o con un LLM generale | aperta, richiesta esplicita | **B1**, **B2**, B3 |
| R2.3 "Large-scale" non giustificato: poche testate | peggiorata (da 439 a 226) | S2 (qualità e indipendenza invece di quantità), F1, G1, I3 |

---

## 3. Le voci in dettaglio

### A. Dimostrare che il protocollo conta (la novità del paper)

**Problema.** Oggi il paper afferma che split per testata, gruppi di proprietà, deduplicazione e anonimizzazione sono
necessari, ma non misura quanto cambierebbero i risultati senza. Il revisore 1 ha dato significatività 1/5 e
contributo 1/5: senza questa prova il paper resta "modelli pronti su un dataset nuovo".

**Prove che abbiamo già.**
- Tentativo 01, vecchio dataset (439 testate): TF-IDF 0,68 con fold per testata, **0,39** con fold per gruppo.
- `rules_only/control_level_split.csv`: due split diversi del dataset attuale danno Mistral ρ 0,735 contro 0,708.
- Passo 12: due anonimizzazioni (solo regole contro regole + GLiNER).

**Cosa fare.** Stessi modelli, stessi iperparametri, una condizione alla volta:
- **A1, split per articolo:** articoli dello stesso outlet in training e in test (5 fold casuali sugli articoli), sul
  dataset finale anonimo. Misure per testata e per articolo.
- **A2, split per testata senza gruppi né deduplicazione tra testate:** dataset al passo "published from 2025" (531
  testate, 40.232 articoli) con le stesse regole di lunghezza e lingua. Valutazione sulle testate del dataset finale
  e, a parte, sulle sorelle. Servono anonimizzazione ed embedding dei ~15.000 articoli in più.
- **A3, testo non anonimizzato:** dataset finale, testo pulito ma non anonimo.

**Risultato nel paper.** Nuova sottosezione "How much the protocol matters" con una tabella: condizione × modello, ρ e
MAE per testata; la differenza dal nostro protocollo con intervalli appaiati.

**Fatto quando** le tre condizioni hanno i quattro modelli (ModernBERT almeno per A1 e A3) e la tabella è nel paper.

**Esiti possibili.** Se A1–A2 gonfiano molto (atteso, dal tentativo 01), è la novità principale. Se gonfiano poco, lo
scriviamo comunque: il protocollo diventa una garanzia, non una scoperta, e la novità si sposta sulla misura per
testata e sul bias.

### B. Le baseline chieste dal revisore 2

- **B1, LLM zero-shot.** Un modello instruct (Qwen3-8B-Instruct o Mistral-Small-Instruct), prompt fisso con la
  definizione dei criteri NewsGuard, chiede il voto 0–100 dell'articolo anonimo. Punteggio come valore atteso sulle
  cifre, così niente campionamento e niente parsing fragile. Media per testata. Nessun addestramento: è il confronto
  con "un LLM generale come Qwen" chiesto alla lettera.
- **B2, classificatore di credibilità per articolo.** Un modello pubblicato per fake news o credibilità degli articoli,
  da scegliere tra quelli addestrati su dati che non contengono le nostre testate. La sua probabilità di "affidabile"
  va aggregata per testata. *Rischio:* un modello adatto e pulito potrebbe non esistere; in quel caso lo scriviamo.
- **B3, feature stilistiche.** Leggibilità, lunghezza delle frasi, emotività, connettivi, cautela, citazioni (le 35
  misure spaCy del tentativo 15), ridge sulla media per testata o per articolo. Risponde anche alla domanda "è solo
  stile?".
- **B4, modello per testata** (dal passo 10): vettore medio della testata e ridge sulle testate. Mostra che imparare
  per articolo e poi aggregare è meglio (risponde a R2.1).

**Risultato nel paper:** righe nuove nella tabella principale e nel grafico dei risultati.
**Fatto quando** B1, B3 e B4 sono nella tabella principale e B2 è nella tabella oppure scartata con il motivo.

### C. Confronti controllati sui modelli

- **C1.** ModernBERT congelato (media degli stati dell'ultimo strato) + ridge contro ModernBERT addestrato: stesso
  modello, solo "congelato contro addestrato".
- **C2.** Qwen3-Embedding 0,6B, 4B e 8B, tutti congelati + ridge: solo la dimensione cambia.
- **C3.** Lo strato 27 di Mistral oggi viene da esperimenti su una versione precedente del dataset (*data snooping*).
  Si sceglie lo strato **dentro i fold** tra 27 e 40, che sono già salvati (`embeddings/mistral_small_24b_layer40.npy`),
  e si riporta la curva per strato (estrazione di più strati in un'unica passata).
- **C4.** ModernBERT oggi ha iperparametri fissi e un seed: un revisore lo dirà sleale. Learning rate {1, 3, 5}·10⁻⁵ ed
  epoche {2, 3, 5} scelti sui fold interni, oppure early stopping su validazione interna; 3 seed con intervallo.
- **C5.** Riscrivere la Discussione: oggi "using a large model without adapting it is safer than adapting a small one"
  non è controllato. Si scrive solo quello che C1–C4 mostrano.

**Fatto quando** C1–C3 sono in una tabella "controlled comparisons" e la Discussione dice solo ciò che i dati
sostengono.

### D. Il bias politico, reso solido

**Oggi:** a parità di voto (residui di una regressione lineare dell'errore sul voto), le testate di sinistra ricevono
6,7–9,6 punti più di quelle di destra (Kruskal–Wallis p < 10⁻⁹ per Mistral). È il risultato più delicato del paper.

- **D1.** Regressione per testata: errore ~ spline del voto + effetti fissi per livello + orientamento + paese + log(n.
  articoli). Intervalli bootstrap sulle testate e test a permutazione dell'orientamento (si permuta l'orientamento dentro
  ogni livello).
- **D2.** Il campo "orientamento" di NewsGuard riguarda testate di paesi diversi (USA, Regno Unito, Canada,
  Australia…): si stima il divario solo sulle testate statunitensi e sulle altre.
- **D3.** Mitigazione: ridge con pesi che bilanciano gli orientamenti nel training. Il divario si riduce? Quanto costa
  in ρ?
- **D4.** Per capire la causa: a parità di voto, quali dei 9 criteri NewsGuard mancano più spesso a destra e a
  sinistra? Se le testate di destra con lo stesso voto mancano criteri visibili nel testo, il "bias" è in parte
  segnale.
- **D5.** Orientamento da MBFC (decisione D1) al posto di quello NewsGuard.

**Fatto quando** D1–D2 sono nel paper (tabella o appendice) e la discussione del bias si basa su D4 e, se possibile,
su D3.

### E. L'aggregazione, misurata bene

- **E1.** Il revisore 1 aveva notato che confrontare articoli e testate cambia unità e pesi. **Lo facciamo ancora**: la
  Tabella "Stage 1" e il pannello (c) della figura dei risultati confrontano 25.524 articoli con 226 testate. Al loro
  posto si confronta, **sempre per testata e con lo stesso peso**, k = 1 articolo con k = tutti. I numeri ci sono già
  (`paper_curves.csv`): Mistral ρ 0,62 → 0,735, TF-IDF 0,39 → 0,674. Si aggiunge l'intervallo appaiato della
  differenza. La tabella per articolo resta solo come descrizione.
- **E2.** Media contro mediana, media troncata e voto di maggioranza (dopo la discretizzazione), per mostrare che il
  risultato non dipende dalla funzione di aggregazione.

**Fatto quando** E1 sostituisce Tabella e pannello attuali ed E2 è in appendice.

### F. Il bias di selezione

Delle 1.293 testate politiche mandate allo scraper, 674 hanno articoli e 226 restano. Per tutte abbiamo il voto
NewsGuard, l'orientamento e il paese. Si confrontano le distribuzioni per ogni passo (test di Kolmogorov–Smirnov sul
voto, chi-quadro per orientamento e livello) e si discute apertamente la rete di 164 siti l1 eliminata: il dataset
sottorappresenta le reti di siti clonati, quindi il risultato vale per testate indipendenti.

**Fatto quando** c'è una tabella o figura in appendice e un paragrafo in "Limitations".

### G. Più split

- **G1.** 5 seed di split stratificato per voto per TF-IDF, Qwen e Mistral (le ridge girano in secondi sugli embedding
  esistenti): media e deviazione standard di ρ e MAE.
- **G2.** Lo stesso per ModernBERT (GPU).

**Fatto quando** la tabella principale riporta anche la variabilità tra split.

### M. Statistica

- **M1.** Correzione di Holm per i confronti multipli (tabella principale: 3 modelli × 4 misure; curve: 8 k × 2 misure ×
  3 modelli). La frase "con 5 articoli Mistral ordina meglio di TF-IDF con tutti" oggi **non è testata**: serve un
  bootstrap appaiato (stesse testate) tra Mistral a k = 5 e TF-IDF a k = tutti.

**Fatto quando** ogni frase "significantly" del paper corrisponde a un test corretto.

### H. Che cosa leggono i modelli

- **H1.** Correlazione delle previsioni (e dei residui) con gli indici di stile di B3. La Discussione oggi ipotizza
  "how claims are sourced and hedged, how emotional the language is": va dimostrato o tolto.
- **H2.** Le parole e i bigrammi più pesanti di TF-IDF, per segno, raggruppati per categoria (parte politica,
  geografia, stile, fonti).
- **H3.** Temi con un topic model sugli articoli; ρ dentro i temi più frequenti: il segnale sopravvive a parità di
  argomento?
- **H4.** Tipi di testata con l'errore più grande (satira, aggregatori, siti d'opinione, regionali), senza nomi.

**Fatto quando** la Discussione cita H1–H4 invece di ipotesi.

### I. Validità esterna

- **I1.** MBFC (decisione D1): accordo modello–MBFC contro NewsGuard–MBFC sulle testate in comune (circa 170 nel
  tentativo 17, da ricalcolare sul dataset attuale).
- **I2.** Articoli più vecchi contro più recenti (prima e dopo la mediana della data): la previsione dipende dal tempo?
  Si discute anche lo scarto tra la data dei voti (istantanea NewsGuard del 23/10/2026) e quella degli articoli.
- **I3.** Testate non politiche (terzo giro di scraping, tentativo 12) come test fuori dominio, con i modelli
  addestrati sulle 226. Aumenta N e risponde a R2.3, ma è il lavoro più grande.

### J. Qualità dell'anonimizzazione

- **J1.** Secondo riconoscitore (spaCy `en_core_web_trf` o Flair) come riferimento "silver" su un campione: stima di
  precisione e richiamo di GLiNER per persone e organizzazioni giornalistiche.
- **J2.** Probe di identità: un classificatore che riconosce la testata (tra 226) dagli embedding di articoli della
  stessa testata (split per articolo): accuratezza sul testo anonimo; dopo A3, confronto con il testo grezzo.
  Quantifica "lo stile identifica ancora la testata" invece di dichiararlo.
- **J3.** Campione annotato a mano (decisione D4).
- **J4.** Il paper dice che GLiNER da solo lascia il nome della testata in circa un articolo su quattro: è un dato di un
  esperimento precedente non riprodotto nella pipeline finale. Va rifatto (GLiNER sul testo originale) oppure tolto.

### K, L. Calibrazione e metriche confrontabili

- **K1.** Calibrazione isotonica stimata sui fold di training (previsioni fuori fold dei fold interni): MAE ed entro 20
  punti prima e dopo (decisione D3).
- **L1.** AUC su affidabile sì/no (soglia 60 di NewsGuard), F1 macro sui 5 livelli con le soglie NewsGuard e con le
  quote del training: confronto con la letteratura e con il nostro paper precedente (decisione D2).

### R. Rilascio

- Creare `github.com/bianchi-john/publisher-evaluation` e fare il commit del tool (modelli in pezzi da 50 MB).
- Aggiungere il file LICENSE al repository dello studio (il paper dichiara Apache-2.0) e renderlo pubblico.
- DOI Zenodo per modelli, previsioni e split: le riviste preferiscono archivi permanenti.
- X2 risolto: nessuna etichetta NewsGuard viene rilasciata (né voti, né livelli, né orientamento); analisi delle
  etichette e modelli addestrati su di esse si possono pubblicare. Le previsioni rilasciate sono uscite dei modelli,
  non etichette.

### S. Scrittura

- **S1 Bibliografia.** Verificare tutte le voci con `note = {verify}` (autori, pagine, DOI); aggiungere i lavori
  2024–2026 su LLM e credibilità delle fonti, sulla profilazione dei media e sulla fuga di informazioni nei benchmark.
  Il revisore 1 aveva dato 2/5 alle referenze.
- **S2 Affermazioni da ridimensionare o precisare:**
  - "Two observations explain why the frozen models do better…": diventa un'ipotesi, o si dimostra con H1 e C1–C2;
  - "using a large model without adapting it is safer than adapting a small one": dopo C1–C4;
  - "the recognizer alone leaves the outlet's own name in about one article in four": J4;
  - "in preliminary experiments on an earlier version of the dataset, layer 27 was better": C3;
  - "We use the ratings available in 2026": scrivere la data esatta dell'istantanea (23/10/2026);
  - etica: precisare che lo scraper rispetta robots.txt e fa al massimo una richiesta al secondo per sito (verificato
    in `scraper.py`);
  - mai "large-scale": parlare di testate indipendenti e di un protocollo rigoroso.
- **S3 Appendice:** risultati per fold, iperparametri scelti, curve per strato, E2, F1, G1, D1–D2.
- **S4 Formato:** template della rivista scelta (D5); la sezione del tool si accorcia e il dettaglio va nel materiale
  supplementare.
- **S5 Risposta alle critiche:** un documento che riprende R1.1–R2.3 punto per punto con la voce che le risolve.

---

## 4. Ordine di lavoro

1. **Subito, senza GPU (1–2 giorni):** E1, M1, F1, C3 (27 contro 40), D1, D2, G1, B3, B4, L1/K1 (se decisi), S2, S1
   iniziato, controllo della licenza NewsGuard (X2).
2. **Ancora CPU (1–2 giorni):** H1–H4, I2, J1, D3, D4, E2, I1 (se deciso).
3. **Quando la GPU torna (circa una settimana di coda condivisa):** A1–A3, B1, B2, C1, C2, C4, G2, J2 sul testo grezzo,
   J4.
4. **Scrittura finale:** sezione "How much the protocol matters", tabella delle baseline, confronti controllati,
   appendice, formato della rivista, risposta alle critiche, rilascio (R1).

---

## 5. Registro

| Data | Voce | Cosa è cambiato |
|---|---|---|
| 12/11/2026 | — | Piano creato dalla rilettura del paper e delle review TCSS; GPU del server ferma (X1) |
| 12/11/2026 | X2 | Licenza NewsGuard chiarita da John: niente etichette rilasciate; analisi e modelli sì |

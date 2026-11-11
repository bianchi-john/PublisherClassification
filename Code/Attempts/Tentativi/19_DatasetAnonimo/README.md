# Tentativo 19: il dataset anonimo

**La domanda.** Il modello prevede l'affidabilità leggendo l'articolo, o riconosce la testata dal suo nome e usa quello
che già sa di lei? I modelli grandi (Qwen, Mistral) hanno letto il web: un «Sign up to receive Bellingcat's
investigations» rimasto nel testo basta a riconoscere la testata. Togliamo dal testo tutto quello che dice **chi** l'ha
pubblicato, con regole semplici, e niente di quello che dice **che cosa** racconta.

**Il disegno, fissato prima dei risultati.**
- **Stesso dataset del tentativo 16** (226 testate, 25.524 articoli, stessa pulizia con trafilatura, stessi doppioni
  tolti, una testata per gruppo di proprietà). Cambiano solo il testo e il titolo, così il confronto con il 16 misura
  l'effetto dell'anonimizzazione e nient'altro.
- **Sei regole**, nell'ordine (`anonimizza.py`):

  | # | che cosa | come si trova | diventa |
  |---|---|---|---|
  | 1 | firma della testata | la coda ripetuta in almeno il 30% dei suoi titoli («… \| The New York Sun») | `[TESTATA]` |
  | 2 | email | espressione regolare | `[EMAIL]` |
  | 3 | link | http…, www…, domini (`bellingcat.com`) | `[URL]` |
  | 4 | account | `@nome` | `[ACCOUNT]` |
  | 5 | autori della testata | campo autore: nomi che firmano solo in questa testata; nomi di una parola solo dopo «by» | `[AUTORE]` |
  | 6 | nome della testata | il dominio, attaccato o staccato, con o senza «the»; senza un «online» in coda | `[TESTATA]` |

  Un nome che è anche una parola comune (politics, time, reason: compare minuscolo più di 20 volte nelle altre testate)
  si toglie solo con l'iniziale maiuscola. Si toglie qualche parola vera («Daylight Saving Time» in time.com), mai si
  lascia il nome: l'errore sta dalla parte prudente. Eccezione per regola, non a mano: se la testata scrive il proprio
  nome minuscolo in almeno l'80% dei suoi articoli si toglie anche minuscolo (solo spiked: 100%; il più alto degli
  altri è time, 56%). La firma si cerca senza badare alle maiuscole e fino alla fine della parola («Frontpage Mag» →
  anche «FrontPage Magazine»); una firma di una sola parola comune si scarta («Signed»).
- **Controlli automatici, senza letture a mano**: il nome distintivo non deve restare (cercato anche dentro altre
  parole), nessuna firma deve restare; per i nomi comuni si conta quanto resta in minuscolo.
- **Non si tocca**: persone, luoghi, fatti, citazioni di altre testate (sono contenuto).
- **Uno split per i due compiti** (`02_split.ipynb`, `split.csv`): stratificato sul voto esatto, il primo split del 18.
  È bilanciato anche per livello (al più una testata di differenza tra i fold) e dentro i livelli (le l1 hanno in media
  20–22 punti in ogni fold, con lo split per livello del 16 da 12 a 31). Con le stesse testate negli stessi fold,
  classificatore e regressore si confrontano a coppie (notebook 6, sezione 4). Anche i fold interni per la penalità
  sono per voto.
- **Due compiti, gli stessi modelli.** Classificazione a 5 livelli; regressione sul voto esatto 0–100 senza classi (la
  soluzione principale, come nel 18). Modelli, tutti con un contesto che prende ogni articolo intero (`MAX_TOKEN` =
  4.096; il notebook 3 controlla che nessun articolo sia troncato):
  - TF-IDF + logistica (classi bilanciate) o ridge;
  - Qwen3-Embedding-8B congelato (istruzione del 16, ultimo token) + logistica o ridge;
  - Mistral-Small-24B Base congelato, 4 bit (media dei token, strato 27) + logistica o ridge;
  - ModernBERT-base addestrato tutto (5 epoche, lr 3·10⁻⁵, nessuna validazione; classi bilanciate con label smoothing
    0,1, oppure errore quadratico sul voto/100).
  DeBERTa no: legge al più 512 token.
- **La penalità** dei modelli lineari si sceglie con 3 fold interni sulle sole testate di training (divisi come lo split
  del compito), con la misura per testata: F1 macro (classificazione) o Spearman (regressione). Griglie con un ordine
  di grandezza in più del 18, dove cadeva sul bordo.
- **Si salva tutto**: per ogni articolo di ogni fold i logit e le probabilità (classificazione) o il voto previsto
  (regressione), in `Dataset/tentativi/19_DatasetAnonimo/previsioni/`; i modelli di ogni fold in
  `Modelli/19_DatasetAnonimo/` (sul server `~/John/Modelli/`).
- **Riproducibile**: un seme per tutto (`SEME` = 42; lo split ha `SEME_SPLIT` = 1 ed è generato una volta sola nel
  notebook 2), `random_state` nei modelli lineari, algoritmi deterministici sulla GPU (`modelli.riproducibile`), le
  versioni delle librerie stampate in ogni notebook. Due esecuzioni dei modelli lineari danno uscite identiche.
- **Le misure** (`metriche.py`, notebook 6), ricalcolate dalle uscite per articolo, per testata, con intervalli
  bootstrap sulle testate. **Le due filosofie sullo stesso terreno**:

  | | spazio dei **livelli** | spazio del **voto** |
  |---|---|---|
  | classificatore | livello votato dagli articoli | voto atteso: probabilità × voto medio del livello nel training |
  | regressore | confini NewsGuard · quote del training (16) · voto degli articoli (17) | media prevista · media riadattata ai quantili del training |
  | nessun testo | classe più frequente | media del training |

  Livelli: F1 (macro, pesata, per livello), accuratezza (anche bilanciata), **tolerant** ±1, errore in livelli, kappa
  (anche quadratica), Spearman, Kendall, affidabile sì/no. Voto: MAE, RMSE, errore mediano, R², Spearman, Pearson,
  Kendall, CCC di Lin, distorsione, **entro 10 e 20 punti** (la tolerant del voto), affidabile ≥ 60 con AUC. Più le
  misure con le probabilità del classificatore (AUC, log loss, Brier) e quelle per articolo. Il confronto
  classificatore/regressore è a coppie, sulle stesse testate.
- **Il notebook 7** mostra tutto in una pagina: le due tabelle colorate con gli intervalli, il confronto a coppie, un
  grafico e le frasi di sintesi generate dalle tabelle.

**Come si esegue.** In locale 01–02 (fatti). Sul server: `./carica.sh` dal PC, poi in `~/John/Tentativo19`
`screen -dmS t19_gpu ./esegui_gpu.sh` e `screen -dmS t19_cpu ./esegui_cpu.sh`; si segue con `./stato.sh` e si scarica
con `./scarica.sh` (`./scarica.sh tutto` porta anche le uscite per articolo).

**I file.** I notebook 01–07 (generati da `costruisci_notebook.py`), `anonimizza.py` (le regole e i controlli),
`modelli.py` (dati, embedding, modelli lineari, ModernBERT), `metriche.py` (le misure), `Results/` (senza etichette
NewsGuard). Il dataset anonimo sta in `Dataset/tentativi/19_DatasetAnonimo/` (`articoli.csv.gz` senza url e autore,
`testate.csv`, i due split).

**I risultati** (notebook, sezioni 3 e 4).
- **Il 54,6% degli articoli** ha almeno una sostituzione nel testo o nel titolo. Per regola, articoli toccati nel testo
  (titoli tra parentesi): firma 6,5% (21,6%), email 1,3%, link 14,5% (0,8%), account 9,9%, autori 11,3% (1,3%), nome
  della testata 17,4% (1,0%).
- 18 testate hanno un nome che è una parola comune; 54 hanno una firma nei titoli; 4.597 autori da togliere in 198
  testate.
- **Controlli**: per le 208 testate con nome distintivo, nessun articolo con il nome o la firma come parola intera. Tre
  articoli lo hanno dentro un'altra parola: due falsi allarmi (cf.org: «DCF», «CFO») e un refuso («Quilletteessay»).
  Per le 18 testate con nome comune resta solo la forma minuscola (es. «time» in 87 articoli di time.com), per scelta.
- La lunghezza non cambia (mediana 573 parole prima e dopo).

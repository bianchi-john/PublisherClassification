# Versione 3 — a parità di genere

**La domanda.** L'analisi del TF-IDF (`../Versione2/2-classificazione/05_analisi_feature`) mostra che i livelli
NewsGuard vanno di pari passo con il "genere" delle testate: orientamento politico, tipo di sito, paese. Il TF-IDF
riconosce l'orientamento dal testo, e i soli metadati NewsGuard (senza testo) fanno quasi quanto lui. Qui ci si chiede:
**a parità di genere**, il testo distingue ancora le testate affidabili da quelle no? E lì i transformer fanno meglio
del TF-IDF?

**Il disegno** (fissato il 20 ottobre 2026, prima dei risultati):

- stesso dataset della Versione 2 (`../Versione2/Dataset/versioni`: stessa pulizia, stesse 304 testate di test, stessi
  fold per gruppo, le testate di solo training con il loro peso);
- compito **binario**: affidabile = punteggio NewsGuard ≥ 60 ($l_3$–$l_5$);
- **pesi a parità di genere**: dentro ogni orientamento NewsGuard (Far Right, Slightly Right, Slightly Left, Far Left,
  nessuno) affidabili e inaffidabili pesano uguale nel training, così l'orientamento non aiuta a indovinare;
- metrica principale: **media sugli orientamenti dell'accuratezza bilanciata** per testata (testata = media delle
  probabilità dei suoi articoli, soglia 0,5). I soli metadati fanno 0,50 per costruzione. Poi l'AUC media sugli
  orientamenti e il dettaglio per orientamento; intervalli con un bootstrap per gruppo.

Uno sguardo a posteriori sui modelli della Versione 2 (addestrati senza questi pesi): dentro gli orientamenti il
TF-IDF fa 0,66, ModernBERT-base (ensemble) 0,54.

## File

```
2-classificazione/comune.py                 dati (dalla Versione 2), pesi di genere, metriche
2-classificazione/01_feature_genere.ipynb   TF-IDF + regressione logistica (C = 1, o C in validazione interna)
2-classificazione/02_transformer_genere.ipynb  ModernBERT-base, binario, loss pesata per genere; a ogni fold il log
                                            scrive anche il TF-IDF sullo stesso fold
2-classificazione/03_confronto_genere.ipynb tutti i modelli, ensemble dei seed, bootstrap per gruppo
2-classificazione/coda.py                   due corsie in parallelo: cpu (TF-IDF) e gpu (ModernBERT, seed 42, 1, 2)
stato.sh / controlla_server.sh              a che punto è (sul server / dal PC)
```

## Sul server

```bash
cd ~/John/WebScience2027/Versione3/2-classificazione
screen -dmS v3_cpu bash -c 'until ~/John/johnVenv/bin/python coda.py cpu; do sleep 60; done'
screen -dmS v3_gpu bash -c 'until ~/John/johnVenv/bin/python coda.py gpu; do sleep 60; done'
../stato.sh
```

## Stato al 20 ottobre, sera

Risultati per testata (304 testate, metrica principale = media sugli orientamenti dell'accuratezza bilanciata):

| | media sugli orientamenti | AUC media | Slightly Right | Far Right | su tutto |
|---|---|---|---|---|---|
| TF-IDF, C = 1 | 0,618 | 0,671 | 0,56 | 0,65 | 0,703 |
| TF-IDF, C in validazione (riferimento) | 0,593 | 0,735 | | | 0,707 |
| ModernBERT-base, seed 42 | 0,579 | 0,632 | 0,71 | 0,52 | 0,709 |

Tutti sopra il caso (0,50): un segnale di affidabilità oltre il genere c'è. Sulla metrica principale il TF-IDF resta
un po' avanti; ModernBERT va meglio in Slightly Right (l'unico orientamento bilanciato), peggio in Far Right. Un
seed solo: si aspetta l'ensemble.

Lanciati per la notte (decisi dopo il seed 42):

- corsia gpu: ModernBERT-base seed 1 e 2 (verso le 18:45), poi seed 3 e 4 (screen `v3_gpu_seed34`, verso le 22) →
  ensemble di 5 seed;
- corsia cpu: TF-IDF **solo parole funzione** e **senza nomi propri** a parità di genere (stile o contenuto?),
  embedding congelati **e5** e **bge** a parità di genere (`04_embedding_genere`);
- in `03_confronto_genere`: la **combinazione** 0,5/0,5 del TF-IDF di riferimento e dell'ensemble di ModernBERT.

Domattina: `./controlla_server.sh` (stato, fold per fold) e `./controlla_server.sh risultati` (intervalli bootstrap).

## Risultati finali (21 ottobre; tutti gli esperimenti finiti, nessuno fallito)

| | media sugli orientamenti [IC 95%] | AUC media | Slightly Right | Far Right |
|---|---|---|---|---|
| TF-IDF, C in validazione (riferimento) | 0,593 [0,53; 0,67] | **0,735** | 0,51 | 0,60 |
| TF-IDF, C = 1 | **0,618** [0,55; 0,74] | 0,671 | 0,56 | **0,65** |
| TF-IDF, solo parole funzione | **0,618** [0,55; 0,74] | 0,610 | 0,60 | 0,64 |
| TF-IDF, senza nomi propri | 0,589 [0,53; 0,67] | 0,656 | 0,52 | 0,65 |
| ModernBERT-base, i 5 seed | 0,53–0,60 | 0,62–0,65 | **0,57–0,71** | 0,52–0,58 |
| ModernBERT-base, ensemble di 5 | 0,551 [0,49; 0,63] | 0,639 | 0,63 | 0,57 |
| e5 / bge congelati | 0,569 / 0,502 | 0,611 / 0,593 | 0,60 / 0,58 | 0,59 / 0,43 |
| combinazione TF-IDF + ModernBERT | 0,597 [0,54; 0,68] | 0,693 | 0,60 | 0,56 |

- Un segnale di affidabilità oltre il genere c'è, ma modesto (0,55–0,62 contro 0,50).
- Nessun modello batte il TF-IDF in modo significativo (ensemble di ModernBERT −0,04 [−0,10; +0,02]); bge fa peggio.
- Il TF-IDF sulle sole parole funzione fa quanto quello sul testo intero: dentro l'orientamento il segnale è stile.
- In Slightly Right (l'unico orientamento bilanciato) tutti e 5 i seed di ModernBERT fanno meglio del TF-IDF di
  riferimento; in Far Right il contrario. È una lettura a posteriori su 78 testate: un indizio, non un risultato.
- L'ensemble è sotto i singoli seed sulla soglia 0,5 ma non sull'AUC: la media delle probabilità sposta la calibrazione.

Dettagli in `Results/2-classificazione/03_genere_*.csv` e nella presentazione `Riunioni/Stato dei lavori/`.

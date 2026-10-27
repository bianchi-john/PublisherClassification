# 15: analisi del dataset (non un tentativo)

Due notebook sul dataset del tentativo 14.

## 01_analisi_dataset.ipynb

`01_analisi_dataset.ipynb` guarda **dentro il dataset del tentativo 14**: 323 testate politiche, 42.219 articoli,
livelli NewsGuard 2026. La domanda è se nei testi ci sono differenze tra i livelli di affidabilità che un
classificatore, semplice o complesso, può sfruttare.

Il notebook legge dataset, fold e previsioni dei modelli da `../14_DatasetRipensatoChunk` (parametro `T14_DIR`) e non
addestra modelli grandi. Le sezioni:
1. che cosa misura NewsGuard (i 9 criteri per livello);
2. lo stile (10 misure semplici);
3. le parole tipiche di ogni livello (log-odds, Monroe 2008);
4. una mappa delle testate (t-SNE);
5. una scala di classificatori sugli stessi fold;
6. l'AUC per ogni coppia di livelli;
7. affidabili contro inaffidabili dentro lo stesso orientamento;
8. i numeri principali;
9. le conclusioni.

## Esecuzione

Sul server, in `~/John/Tentativo15`, in circa 6 minuti:

```
python -m papermill -k johnvenv -p T14_DIR ~/John/Tentativo14 -p N_JOBS 5 01_analisi_dataset.ipynb Results/analisi_dataset_eseguita.ipynb
```

Sul PC basta lanciarlo con il kernel `irgood-venv`, in circa 20 minuti (un fold alla volta).

## Che cosa viene fuori

- **Dal 35% al 60% dei punti NewsGuard persi** riguarda pratiche del sito (correzioni, proprietà, chi comanda, autori),
  invisibili negli articoli.
- La scala di classificatori:

  | Classificatore | F1 (5 livelli) |
  |---|---|
  | Maggioranza | 0,09 |
  | Solo orientamento | 0,23 |
  | Stile | 0,25 |
  | Parole | 0,38–0,40 |
  | ModernBERT | 0,27–0,28 |

- Le parole tipiche di ogni livello sono soprattutto argomenti e geografia.
- Coppie di livelli: l1 contro l5 dà AUC 0,85 (il solo orientamento fa già 0,80), mentre **l2 contro l3 dà 0,52**:
  proprio sul confine affidabile/inaffidabile il testo non distingue quasi nulla.
- Dentro lo stesso orientamento resta un segnale debole: sinistra 0,73, destra 0,63.

## 02_mappe_e_stile.ipynb

Riprende la domanda di fondo del paper: le testate molto affidabili **scrivono** in modo diverso? I livelli si
guardano a finestre di due livelli vicini (l1–l2, l2–l3, l3–l4, l4–l5), come nella tolerant, e con il punteggio
continuo. Ogni testata ha tre rappresentazioni:
- **parole** (TF-IDF);
- **stile profondo**: 35 misure con spaCy, calcolate in `stile.py`. Coprono leggibilità, sintassi, connettivi, cautela ed enfasi (Hyland), attribuzioni, entità, tono ed emozioni (lessici NRC in `Dataset/lessici`, fuori da git);
- **testo senza argomento** (text distortion, Stamatatos 2017): restano parole funzionali e punteggiatura, il resto diventa la categoria grammaticale.

Il notebook contiene:
- le mappe con PCA, t-SNE, Isomap e MDS;
- la coerenza dei vicini contro il caso;
- i classificatori sugli stessi fold, con AUC sugli estremi (in tutto il dataset e dentro ogni orientamento) e su l2 contro l3;
- le costruzioni tipiche senza argomento.

Sul server in circa 15 minuti (11 se l'analisi spaCy è già in `Results/cache`):
`python -m papermill -k johnvenv -p T14_DIR ~/John/Tentativo14 02_mappe_e_stile.ipynb Results/02_mappe_e_stile_eseguito.ipynb`.

**Che cosa viene fuori**
- Le l4–l5, rispetto alle l1–l2, usano:
  - più connettivi avversativi (delta di Cliff +0,44);
  - meno parole emotive (−0,40);
  - più cautela (+0,37);
  - più avverbi e lessico più vario;
  - meno maiuscole e punti esclamativi.
- **Senza argomento** i vicini si somigliano più che con le parole (correlazione dei punteggi 0,42 contro 0,35; a caso
  −0,03), e gli estremi si separano con AUC 0,75 (parole 0,79).
- Effetti grandi dentro la sinistra (AUC dello stile 0,72–0,75, come le parole), piccoli dentro la destra (0,57–0,60).
- l2 contro l3 resta a caso con ogni rappresentazione.

## 03_separazione_visiva.ipynb

Le testate sul piano cartesiano, un punto per testata, con **4 indici di stile** scelti per significato prima dei
risultati: ragionamento, emotività, complessità, fonti (medie di misure del notebook 2, in deviazioni standard). Il
notebook mostra:
- l'AUC di ogni indice per cinque confronti (l1/l5, l1–l2/l4–l5, l1/l4, l2/l4 e l2/l3 come controllo);
- coppie di indici con un'ellisse per gruppo;
- le finestre di livelli;
- il percorso dei centri da l1 a l5;
- il piano dentro sinistra e destra;
- le due misure concrete: connettivi avversativi e parole emotive.

Legge le misure di stile salvate dal notebook 2 (`Results/cache/stile_articoli.csv.gz`, fuori da git) e sul server
gira in 15 secondi.

**Che cosa viene fuori**
- L'emotività separa l1 da l5 all'81% (più bassa nelle affidabili), il ragionamento al 70%; le nuvole si sovrappongono.
- l2 contro l3: al massimo 0,59.
- **Due stili di inaffidabilità**:
  - a destra le testate poco affidabili sono più emotive (AUC 0,80; il ragionamento 0,58);
  - a sinistra ragionano meno (AUC 0,80), e l'emotività va al contrario.

Una seconda parte (classificatori addestrati solo sugli estremi, poi tutte le testate sul piano) è pronta ma messa da
parte, da rivedere dopo questa.

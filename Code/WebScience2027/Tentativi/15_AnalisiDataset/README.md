# 15: analisi del dataset (non un tentativo)

`analisi_dataset.ipynb` guarda **dentro il dataset del tentativo 14**: 323 testate politiche, 42.219 articoli,
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
python -m papermill -k johnvenv -p T14_DIR ~/John/Tentativo14 -p N_JOBS 5 analisi_dataset.ipynb Results/analisi_dataset_eseguita.ipynb
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

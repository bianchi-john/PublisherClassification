# 17: il modello come terzo annotatore

**La domanda.** Dal solo testo dei suoi articoli si può giudicare l'affidabilità di una testata, come fanno gli esperti?
I criteri di NewsGuard e di Media Bias/Fact Check (MBFC) guardano cose che un modello non vede (fonti, correzioni,
proprietà del sito). La scommessa è che l'affidabilità lasci comunque un'impronta, un "DNA", in ogni articolo.

**Il disegno** (fissato prima dei risultati).
- Tre annotatori sulle stesse 170 testate del tentativo 16 valutate anche da MBFC (dominio identico):
  - NewsGuard, con il punteggio complessivo in 5 livelli;
  - MBFC, con la fattualità ("factual reporting") in 5 livelli;
  - il modello.
- Affidabile:
  - NewsGuard ≥ 60;
  - MBFC almeno "Mostly Factual";
  - per il modello, la maggioranza dei suoi articoli è almeno l3.
- Il modello dà un livello a ogni articolo e la testata prende il livello votato dalla maggioranza degli articoli. Per il regressore, ogni articolo diventa un livello con le quote del training.
- Due misure di accordo a coppie:
  - Spearman tra i livelli;
  - kappa di Cohen su affidabile sì/no.

  Gli intervalli al 95% vengono dal bootstrap sulle testate; il riferimento è l'accordo tra i due esperti.
- I modelli sono quattro, uno per famiglia, ognuno come classificatore e come regressore:
  - TF-IDF;
  - ModernBERT addestrato;
  - Qwen3-Embedding 8B congelato (parametro scelto nel training);
  - Mistral-Small 24B congelato (strato 27, parametro scelto).
- Nessun addestramento nuovo. Le previsioni sono quelle del tentativo 16, fatte su testate mai viste (fold di test). MBFC non è mai stato usato per addestrare.
- Controlli:
  - la curva con 1, 3, …, 25 articoli o tutti: basta un articolo?;
  - le testate su cui gli esperti sono d'accordo contro quelle su cui non lo sono;
  - la credibilità MBFC al posto della fattualità.

**File**
- `01_accordo_annotatori.ipynb`: il notebook, già eseguito.
- `accordo.py`: le funzioni.
- `costruisci_notebook.py`: il builder.
- `Results/`: tabelle e grafici; quello riassuntivo è `colpo_d_occhio.png`, la curva `curva_articoli.png`.

**Dati.** Il notebook legge:
- il dataset del 16 e MBFC (`Dataset/tentativi/10_EtichetteMBFC/mbfc_pulito.csv`, anche in `Dataset.zip`);
- le previsioni per articolo del 16 (`Results/feature`, `Results/modernbert`, `Results/analisi_embedding`). Stanno sul server; per rifarle in locale si riesegue il 16.

Sul server, in `~/John/Tentativo17`, in pochi minuti e solo con la CPU:
`python -m papermill -k johnvenv -p T16_DIR ~/John/Tentativo16 01_accordo_annotatori.ipynb 01_accordo_annotatori.ipynb`.

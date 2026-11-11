#!/bin/bash
# Analisi degli embedding (28/10): prima gli embedding 8B con un'altra istruzione (circa 2 ore di GPU), poi il notebook 7
# (solo CPU). Da lanciare in uno screen:   screen -dmS t16 ./esegui_07.sh ["istruzione" etichetta]
# Avanzamento: Results/esegui_tutto.log (dal PC: ./stato.sh).
export TZ=Europe/Rome
cd ~/John/Tentativo16
source ~/John/johnVenv/bin/activate
LOG=Results/esegui_tutto.log
nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@" >> Results/esegui_tutto.out 2>&1; }
ISTRUZIONE="${1:-Represent the writing style, tone, sourcing and rigor of this news article, regardless of its topic}"
ETICHETTA="${2:-stile}"

nota "================ analisi degli embedding, istruzione «$ETICHETTA»: $ISTRUZIONE"
nota ">>> 4. embedding 8B con l'istruzione «$ETICHETTA» (circa 2 ore)"
pm -p MODELLO Qwen/Qwen3-Embedding-8B -p BATCH 4 -p ETICHETTA "$ETICHETTA" -p ISTRUZIONE "$ISTRUZIONE" \
   04_embedding.ipynb "Results/04_embedding_8B_${ETICHETTA}_eseguito.ipynb" || nota "ATTENZIONE: errore negli embedding «$ETICHETTA», continuo"
nota ">>> 7. analisi degli embedding (circa 30 min, solo CPU)"
pm 07_analisi_embedding.ipynb 07_analisi_embedding.ipynb || nota "ATTENZIONE: errore nel notebook 7"
nota "<<< FINITO"

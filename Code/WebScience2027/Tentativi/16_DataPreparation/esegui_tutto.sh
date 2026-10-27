#!/bin/bash
# Tentativo 16 sul server (~/John/Tentativo16), da lanciare in uno screen:   screen -dmS t16 ./esegui_tutto.sh
# Va avanti anche se ci si disconnette. Avanzamento: Results/esegui_tutto.log (dal PC: ./stato.sh).
export TZ=Europe/Rome                       # orari italiani nei log
cd ~/John/Tentativo16
source ~/John/johnVenv/bin/activate
mkdir -p Results
LOG=Results/esegui_tutto.log
nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@" >> Results/esegui_tutto.out 2>&1; }

nota "================ inizio"
# i risultati calcolati sul dataset senza il limite di 1.500 parole vanno da parte (la riestrazione con trafilatura no: non cambia)
VECCHI=Results/vecchi_senza_limite_1500; mkdir -p $VECCHI
for f in Results/feature Results/analisi Results/cache/stile_articoli.csv.gz; do [ -e $f ] && mv $f $VECCHI/ && nota "spostato $f in $VECCHI"; done

nota ">>> 1. data preparation (circa 6 min)";   pm 01_data_preparation.ipynb Results/01_data_preparation_eseguito.ipynb || { nota "FERMATO: errore nel notebook 1"; exit 1; }
nota ">>> 2. data analysis (circa 10 min)";     pm 02_data_analysis.ipynb Results/02_data_analysis_eseguito.ipynb || nota "ATTENZIONE: errore nel notebook 2, continuo"
nota ">>> 3. feature (circa 5 min)";            pm 03_feature.ipynb Results/03_feature_eseguito.ipynb || { nota "FERMATO: errore nel notebook 3"; exit 1; }

# i modelli Qwen3-Embedding si scaricano da Hugging Face (0.6B circa 1,2 GB, 8B circa 16 GB): si parte solo con il via
if [ ! -f Results/via_download ]; then
    nota "IN ATTESA: serve il via per scaricare Qwen3-Embedding (file Results/via_download)"
    while [ ! -f Results/via_download ]; do sleep 60; done
    nota "via ricevuto"
fi

# prova veloce delle parti con la GPU (2 articoli per testata, 1 epoca): se qualcosa non va, ci si ferma subito
nota ">>> prova: embedding 0.6B";             pm -p PROVA 2 04_embedding.ipynb Results/prova/04_prova.ipynb || { nota "FERMATO: errore nella prova degli embedding"; exit 1; }
nota ">>> prova: ModernBERT";                 pm -p PROVA 2 -p EPOCHE 1 -p DIAGNOSTICA False 05_modernbert.ipynb Results/prova/05_prova.ipynb || { nota "FERMATO: errore nella prova di ModernBERT"; exit 1; }
nota "prova superata"

nota ">>> 4. embedding 0.6B (circa 15 min)";    pm -p MODELLO Qwen/Qwen3-Embedding-0.6B 04_embedding.ipynb Results/04_embedding_0.6B_eseguito.ipynb || nota "ATTENZIONE: errore negli embedding 0.6B, continuo"
nota ">>> 4. embedding 8B (circa 2-3 ore)";     pm -p MODELLO Qwen/Qwen3-Embedding-8B -p BATCH 4 04_embedding.ipynb Results/04_embedding_8B_eseguito.ipynb || nota "ATTENZIONE: errore negli embedding 8B, continuo"
nota ">>> 6. confronto (TF-IDF ed embedding)"; pm 06_confronto.ipynb Results/06_confronto_eseguito.ipynb || nota "ATTENZIONE: errore nel confronto, continuo"
for i in 1 2 3; do                          # ModernBERT: se si interrompe riparte dai fold già salvati
    nota ">>> 5. ModernBERT (circa 11 ore; tentativo $i/3)"
    if pm 05_modernbert.ipynb Results/05_modernbert_eseguito.ipynb; then
        nota ">>> 6. confronto finale"; pm 06_confronto.ipynb Results/06_confronto_eseguito.ipynb
        nota "<<< FINITO"; exit 0
    fi
    nota "ATTENZIONE: ModernBERT interrotto, riprovo tra 5 minuti"; sleep 300
done
nota "FERMATO: ModernBERT fallito tre volte"

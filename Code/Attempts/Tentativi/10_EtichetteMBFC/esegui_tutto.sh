#!/bin/bash
# Tentativo 10: i modelli con le etichette MBFC, sul server. Due corsie insieme:
#   CPU: il classificatore a feature (TF-IDF, C in validazione interna) sul dataset grezzo, poi sul pulito;
#   GPU: ModernBERT-base (un seme) sul grezzo, poi sul pulito (un fold alla volta, aspetta la GPU condivisa);
# alla fine il confronto. Ogni passo già fatto (FATTO presente) si salta.
#   screen -dmS t10 ./esegui_tutto.sh            stato: ./stato.sh            log: Results/esegui_tutto.log
cd "$(dirname "$0")"
PY=~/John/johnVenv/bin/python; K=johnvenv; R=Results/3-classificazione; E=$R/esecuzioni
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p $E
log() { echo "$(date '+%F %T')  $*" | tee -a Results/esegui_tutto.log; }
passo() {   # nome, file FATTO, notebook, parametri papermill...
    local nome=$1 fatto=$2 nb=$3; shift 3
    if [ -f "$fatto" ]; then log "salto $nome (già fatto)"; return 0; fi
    log ">>> $nome"
    if (cd 3-classificazione && nice -n 10 $PY -m papermill -k $K $nb ../$E/$nome.ipynb "$@" > ../$E/$nome.out 2>&1); then
        log "<<< $nome fatto"
    else
        log "!!! $nome non riuscito: vedi $E/$nome.out"
    fi
}
cpu() {
    for v in grezzo pulito; do
        passo feature_$v $R/feature_tuned_${v}_FATTO 01_feature_tfidf.ipynb -p VERSIONE $v -p N_JOBS 5
    done
}
gpu() {
    for v in grezzo pulito; do
        passo modernbert_$v $R/transformer/modernbert_${v}_s42/FATTO 02_modernbert.ipynb -p VERSIONE $v
    done
}
log "=== tentativo 10: feature e ModernBERT, etichette MBFC ==="
cpu & gpu; wait
passo confronto /nessun_file 03_confronto.ipynb      # il confronto si rifà sempre
log "=== finito ==="

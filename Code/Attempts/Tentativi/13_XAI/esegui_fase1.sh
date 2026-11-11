#!/bin/bash
# Tentativo 13, fase 1 (sul server): dataset, poi il classificatore a feature (CPU) e ModernBERT (GPU) insieme, con
# performance per fold, probabilità per classe e modelli salvati in Modelli/, poi il confronto. Ogni passo si riprova
# fino a 3 volte; un passo già fatto (segno in Results/fatti/) si salta.
#   screen -dmS t13 ./esegui_fase1.sh            stato: ./stato.sh            log: Results/esegui.log
cd "$(dirname "$0")"
PY=~/John/johnVenv/bin/python; K=johnvenv; F=Results/fatti; E=Results/esecuzioni
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p $F $E
log() { echo "$(date '+%F %T')  $*" | tee -a Results/esegui.log; }
notebook() {   # nome, cartella, notebook, parametri papermill...
    local nome=$1 dir=$2 nb=$3; shift 3
    (cd $dir && nice -n 10 $PY -m papermill -k $K $nb ../$E/$nome.ipynb "$@" >> ../$E/$nome.out 2>&1)
}
passo() {      # nome, comando...: fino a 3 tentativi
    local nome=$1; shift
    if [ -f $F/$nome ]; then log "salto $nome (già fatto)"; return 0; fi
    for t in 1 2 3; do
        log ">>> $nome (tentativo $t/3)"
        if "$@"; then date '+%F %T' > $F/$nome; log "<<< $nome fatto"; return 0; fi
        log "!!! $nome non riuscito: vedi $E/$nome.out"; sleep 120
    done
    return 1
}
log "=== tentativo 13, fase 1: dataset, feature e ModernBERT con i modelli salvati, performance ==="
passo dataset notebook 01_dataset 1-dataset 01_dataset.ipynb || { log "=== FERMATO: dataset ==="; exit 1; }
passo feature notebook 01_feature_tfidf 2-modelli 01_feature_tfidf.ipynb -p ETICHETTA 2026 -p N_JOBS 5 &
passo modernbert notebook 02_modernbert 2-modelli 02_modernbert.ipynb -p ETICHETTA 2026 -p SEED 42 &
wait
rm -f $F/confronto; passo confronto notebook 03_confronto 2-modelli 03_confronto.ipynb
log "=== fase 1 finita ==="

#!/bin/bash
# Tentativo 11: i modelli con le etichette NewsGuard 2026 e, per controllo, 2022 (stesse testate, stessi fold), sul server.
#   CPU: il classificatore a feature (TF-IDF, C in validazione interna) con le etichette 2026, poi 2022;
#   GPU: ModernBERT-base, un modello alla volta, alternando gli anni: 2026 s42, 2022 s42, 2026 s1, 2022 s1, 2026 s2, 2022 s2
#        (circa 1h20 ciascuno; aspetta che la GPU condivisa abbia memoria libera);
#   il confronto si rifà dopo ogni modello finito (usa quello che c'è). Ogni passo già fatto (FATTO) si salta.
#   screen -dmS t11 ./esegui_tutto.sh            stato: ./stato.sh            log: Results/esegui_tutto.log
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
confronto() {
    (cd 3-classificazione && flock ../Results/.confronto.lock nice -n 10 $PY -m papermill -k $K 03_confronto.ipynb ../$E/03_confronto.ipynb > ../$E/03_confronto.out 2>&1) \
        && log "    confronto aggiornato" || log "!!! confronto non riuscito: vedi $E/03_confronto.out"
}
cpu() {
    for a in 2026 2022; do
        passo feature_$a $R/feature_tuned_${a}_FATTO 01_feature_tfidf.ipynb -p ETICHETTA $a -p N_JOBS 5
    done
    confronto
}
gpu() {
    for s in 42 1 2; do
        for a in 2026 2022; do
            passo modernbert_${a}_s$s $R/transformer/modernbert_${a}_s$s/FATTO 02_modernbert.ipynb -p ETICHETTA $a -p SEED $s
            [ -f $R/feature_tuned_2022_FATTO ] && confronto
        done
    done
}
log "=== tentativo 11: feature e ModernBERT, etichette NewsGuard 2026 e 2022 ==="
cpu & gpu; wait
confronto
log "=== finito ==="

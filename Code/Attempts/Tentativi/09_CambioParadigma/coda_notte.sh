#!/bin/bash
# Tentativo 09, la coda della notte del 22 ottobre: ModernBERT come famiglia del testo (04), seme 42, poi 1, poi 2,
# uno alla volta sulla GPU condivisa; dopo ogni seme, in parallelo sulla CPU, la combinazione con i semi finiti fin lì
# (05) e il confronto aggiornato (03). La combinazione aspetta che la catena principale (esegui_tutto.sh) sia finita.
#   sul server:  screen -dmS t09_notte ./coda_notte.sh        stato: ./stato.sh        log: Results/coda_notte.log
cd "$(dirname "$0")"
PY=~/John/johnVenv/bin/python; K=johnvenv; R=Results/3-classificazione; E=$R/esecuzioni
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p $E
log() { echo "$(date '+%F %T')  $*" | tee -a Results/coda_notte.log; }

combina() {   # 05 + 03 con i semi finiti fin qui
    local s=$1
    while [ ! -f $E/03_confronto.ipynb ] && screen -ls | grep -q "\.t09\s"; do sleep 300; done
    if [ ! -f $E/03_confronto.ipynb ]; then log "!!! la catena principale non è finita: niente combinazione"; return; fi
    log ">>> combinazione con ModernBERT (fino al seme $s)"
    if (cd 3-classificazione && nice -n 10 $PY -m papermill -k $K 05_combinazione_bert.ipynb ../$E/05_combinazione_bert_fino_s$s.ipynb > ../$E/05_combinazione_bert_fino_s$s.out 2>&1 \
        && nice -n 10 $PY -m papermill -k $K 03_confronto.ipynb ../$E/03_confronto_con_bert_fino_s$s.ipynb > ../$E/03_confronto_con_bert_fino_s$s.out 2>&1); then
        log "<<< combinazione e confronto aggiornati (fino al seme $s)"
    else
        log "!!! combinazione o confronto non riusciti (fino al seme $s): vedi $E/0[35]_*_fino_s$s.out"
    fi
}

log "=== coda della notte: ModernBERT semi 42, 1, 2 ==="
pid=""
for s in 42 1 2; do
    F=$R/modernbert/modernbert_s$s/FATTO
    for tentativo in 1 2 3; do
        [ -f $F ] && break
        log ">>> ModernBERT seme $s (tentativo $tentativo/3)"
        (cd 3-classificazione && nice -n 10 $PY -m papermill -k $K 04_modernbert.ipynb ../$E/04_modernbert_s$s.ipynb -p SEME $s >> ../$E/04_modernbert_s$s.out 2>&1)
        if [ -f $F ]; then log "<<< ModernBERT seme $s fatto"; else log "!!! ModernBERT seme $s non riuscito: vedi $E/04_modernbert_s$s.out"; sleep 120; fi
    done
    if [ -f $F ]; then
        [ -n "$pid" ] && wait $pid
        combina $s & pid=$!
    fi
done
[ -n "$pid" ] && wait $pid
log "=== coda della notte finita ==="

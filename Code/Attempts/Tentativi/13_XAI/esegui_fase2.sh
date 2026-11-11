#!/bin/bash
# Tentativo 13, fase 2 (XAI, sul server): aspetta che la fase 1 abbia finito (modelli salvati e confronto), poi
#   01 campione e categorie -> 02 attribuzioni (GPU) -> 03 analisi; e 04 maschere (GPU) dopo 02.
# Ogni passo si riprova fino a 3 volte; un passo già fatto (segno in Results/fatti/) si salta.
#   screen -dmS t13_xai ./esegui_fase2.sh        stato: ./stato.sh        log: Results/esegui.log
cd "$(dirname "$0")"
PY=~/John/johnVenv/bin/python; K=johnvenv; F=Results/fatti; E=Results/esecuzioni
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p $F $E
log() { echo "$(date '+%F %T')  $*" | tee -a Results/esegui.log; }
notebook() { local nome=$1 dir=$2 nb=$3; shift 3; (cd $dir && nice -n 10 $PY -m papermill -k $K $nb ../$E/$nome.ipynb "$@" >> ../$E/$nome.out 2>&1); }
passo() {
    local nome=$1; shift
    if [ -f $F/$nome ]; then log "salto $nome (già fatto)"; return 0; fi
    for t in 1 2 3; do
        log ">>> $nome (tentativo $t/3)"
        if "$@"; then date '+%F %T' > $F/$nome; log "<<< $nome fatto"; return 0; fi
        log "!!! $nome non riuscito: vedi $E/$nome.out"; sleep 120
    done
    return 1
}
log "=== tentativo 13, fase 2 (XAI): aspetto la fine della fase 1 ==="
until [ -f $F/feature ] && [ -f $F/modernbert ] && [ -f $F/confronto ]; do
    if ! screen -ls | grep -q "\.t13\s" && ! { [ -f $F/feature ] && [ -f $F/modernbert ]; }; then log "=== FERMATO: la fase 1 non è finita bene ==="; exit 1; fi
    sleep 300
done
passo xai_01_campione notebook xai_01_campione 3-xai 01_campione_e_categorie.ipynb || { log "=== FERMATO: 01 ==="; exit 1; }
passo xai_02_attribuzioni notebook xai_02_attribuzioni 3-xai 02_attribuzioni.ipynb || { log "=== FERMATO: 02 ==="; exit 1; }
passo xai_03_analisi notebook xai_03_analisi 3-xai 03_analisi.ipynb
passo xai_04_maschere notebook xai_04_maschere 3-xai 04_maschere.ipynb
log "=== fase 2 finita ==="

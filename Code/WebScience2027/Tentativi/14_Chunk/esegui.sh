#!/bin/bash
# Tentativo 14 sul server: esegue chunk.ipynb dall'inizio alla fine e salva il notebook con gli output delle celle in
# Results/chunk_eseguito.ipynb. Se si interrompe, riprova fino a 3 volte: i passi già fatti si caricano da Results/cache.
#   screen -dmS t14 ./esegui.sh          avanzamento: ./stato.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p Results
for t in 1 2 3; do
    echo "$(date '+%F %T')  >>> chunk.ipynb (tentativo $t/3)" >> Results/esegui.log
    if nice -n 10 ~/John/johnVenv/bin/python -m papermill -k johnvenv chunk.ipynb Results/chunk_eseguito.ipynb >> Results/esegui.out 2>&1; then
        echo "$(date '+%F %T')  <<< finito" >> Results/esegui.log; exit 0
    fi
    echo "$(date '+%F %T')  !!! interrotto: vedi Results/esegui.out; riprovo tra 2 minuti" >> Results/esegui.log; sleep 120
done
echo "$(date '+%F %T')  === FERMATO dopo 3 tentativi ===" >> Results/esegui.log

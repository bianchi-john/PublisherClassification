#!/bin/bash
# Tentativo 18, coda GPU: i BERT addestrati come regressori (split 1), in ordine di priorità; alla fine, quando anche la
# coda CPU ha finito, il confronto con tutto. Da lanciare in uno screen:  screen -dmS t18_gpu ./esegui_gpu.sh
export TZ=Europe/Rome
cd ~/John/Tentativo18 && source ~/John/johnVenv/bin/activate
LOG=Results/esegui.log; nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv -p T16_DIR ~/John/Tentativo16 "$@" >> Results/esegui_gpu.out 2>&1; }
for cfg in "answerdotai/ModernBERT-base ModernBERT 2048 B 5" "answerdotai/ModernBERT-base ModernBERT 2048 C 5" \
           "microsoft/deberta-v3-base DeBERTa 512 B 5" "answerdotai/ModernBERT-base ModernBERT 2048 A 5"; do
    set -- $cfg
    for i in 1 2 3; do
        nota "GPU: 3. $2, condizione $4 (tentativo $i/3)"
        pm -p MODELLO $1 -p NOME $2 -p MAX_TOKEN $3 -p CONDIZIONE $4 -p EPOCHE $5 03_bert.ipynb Results/03_bert_$2_$4.ipynb && { nota "GPU: $2 $4 finito"; break; }
        nota "ATTENZIONE: $2 $4 interrotto, riprovo tra 5 minuti dai fold salvati"; sleep 300
    done
done
until [ -f Results/cpu_fatto ]; do sleep 60; done
nota "GPU: 4. confronto finale con tutto"; pm 04_confronto.ipynb 04_confronto.ipynb || nota "ATTENZIONE: errore nel confronto finale"
nota "<<< FINITO"

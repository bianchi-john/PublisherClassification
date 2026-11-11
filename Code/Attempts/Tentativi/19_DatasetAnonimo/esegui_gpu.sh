#!/bin/bash
# Tentativo 19, coda GPU: i vettori di Qwen e Mistral, poi ModernBERT per la classificazione e per la regressione; alla
# fine, quando anche la coda CPU ha finito, la valutazione. Ogni passo riparte dai fold già salvati se si interrompe.
# Da lanciare in uno screen:  screen -dmS t19_gpu ./esegui_gpu.sh
export TZ=Europe/Rome
# server condiviso (128 core, 20 utenti): il lavoro è sulla GPU, alla CPU bastano 8 thread
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8          # calcoli deterministici sulla GPU (riproducibilità)
cd ~/John/Tentativo19 && source ~/John/johnVenv/bin/activate
mkdir -p Results; LOG=Results/esegui.log; nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@" >> Results/esegui_gpu.out 2>&1; }
for i in 1 2 3; do
    nota "GPU: 3. lunghezze ed embedding (tentativo $i/3)"
    pm 03_embedding.ipynb 03_embedding.ipynb && { nota "GPU: embedding finiti"; break; }
    nota "ATTENZIONE: embedding interrotti, riprovo tra 5 minuti"; sleep 300
done
# modello, nome, compito (aggiungere qui "answerdotai/ModernBERT-large ModernBERT-large ..." se si decide di farlo)
for cfg in "answerdotai/ModernBERT-base ModernBERT classificazione" "answerdotai/ModernBERT-base ModernBERT regressione"; do
    set -- $cfg
    for i in 1 2 3; do
        nota "GPU: 5. $2, $3 (tentativo $i/3)"
        pm -p MODELLO $1 -p NOME $2 -p COMPITO $3 05_modernbert.ipynb Results/05_$2_$3.ipynb && { nota "GPU: $2 $3 finito"; break; }
        nota "ATTENZIONE: $2 $3 interrotto, riprovo tra 5 minuti dai fold salvati"; sleep 300
    done
done
until [ -f Results/cpu_fatto ]; do sleep 60; done
nota "GPU: 6. valutazione"; pm 06_valutazione.ipynb 06_valutazione.ipynb || nota "ATTENZIONE: errore nella valutazione"
nota "GPU: 7. i risultati in una pagina"; pm 07_risultati.ipynb 07_risultati.ipynb || nota "ATTENZIONE: errore nei risultati"
nota "<<< FINITO"

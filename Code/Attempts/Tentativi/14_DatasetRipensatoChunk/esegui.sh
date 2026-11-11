#!/bin/bash
# Tentativo 14 sul server: prima una prova breve dei due notebook, poi (se va) la versione completa.
cd ~/John/Tentativo14
source ~/John/johnVenv/bin/activate
mkdir -p Results/prova
LOG=Results/esegui.log
nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@"; }

nota ">>> prova: notebook 1 su 100 testate"
pm -p TESTATE 100 01_preparazione_dataset.ipynb Results/prova/01_prova.ipynb >> Results/esegui.out 2>&1 || { nota "FERMATO: prova del notebook 1 fallita"; exit 1; }
nota ">>> prova: notebook 2 sul dataset di prova"
pm -p DATASET_DIR prova -p CACHE_DIR Results/prova/cache -p OUT_DIR Results/prova/classificazione \
   02_classificazione.ipynb Results/prova/02_prova.ipynb >> Results/esegui.out 2>&1 || { nota "FERMATO: prova del notebook 2 fallita"; exit 1; }

nota ">>> notebook 1 completo"
pm 01_preparazione_dataset.ipynb Results/01_preparazione_dataset_eseguito.ipynb >> Results/esegui.out 2>&1 || { nota "FERMATO: notebook 1 fallito"; exit 1; }
for i in 1 2 3; do                     # ModernBERT: se si interrompe si riparte dai fold già salvati in Results/cache
    nota ">>> notebook 2 completo (tentativo $i/3)"
    if pm 02_classificazione.ipynb Results/02_classificazione_eseguito.ipynb >> Results/esegui.out 2>&1; then
        nota "<<< finito"; exit 0
    fi
    sleep 300
done
nota "FERMATO: notebook 2 fallito tre volte"

#!/bin/bash
# Tentativo 19, coda CPU: i modelli lineari. Il TF-IDF parte subito; Qwen e Mistral quando la coda GPU ha scritto i loro
# vettori. Da lanciare in uno screen:  screen -dmS t19_cpu ./esegui_cpu.sh        Avanzamento dal PC: ./stato.sh
export TZ=Europe/Rome
# server condiviso (128 core, 20 utenti): 16 lavori paralleli × 2 thread ciascuno
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd ~/John/Tentativo19 && source ~/John/johnVenv/bin/activate
mkdir -p Results; LOG=Results/esegui.log; nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@" >> Results/esegui_cpu.out 2>&1; }
rm -f Results/cpu_fatto
for i in 1 2; do
    nota "CPU: 4. TF-IDF, classificazione e regressione (tentativo $i/2)"
    pm -p NOMI "TF-IDF" 04_lineari.ipynb Results/04_lineari_TF-IDF.ipynb && { nota "CPU: TF-IDF finito"; break; }
    nota "ATTENZIONE: TF-IDF interrotto, riparto dai fold salvati"
done
until [ -f ~/John/Dataset/tentativi/19_DatasetAnonimo/embedding/ordine_articoli.csv ]; do sleep 300; done
for i in 1 2; do
    nota "CPU: 4. Qwen3 8B e Mistral 24B, classificazione e regressione (tentativo $i/2)"
    pm 04_lineari.ipynb 04_lineari.ipynb && { nota "CPU: modelli lineari finiti"; break; }
    nota "ATTENZIONE: modelli lineari interrotti, riparto dai fold salvati"
done
touch Results/cpu_fatto; nota "CPU: fine"

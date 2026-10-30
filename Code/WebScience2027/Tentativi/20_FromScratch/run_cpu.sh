#!/bin/bash
# Server, CPU queue: steps 1-4 (scraping summary, cleaning, anonymization, split), then the linear models: TF-IDF at
# once, Qwen and Mistral as soon as the GPU queue has written their vectors.
# Start:  screen -dmS t20_cpu ./run_cpu.sh        Progress from the PC: ./status.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false   # shared server
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results; note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/run.log; }
pm() { python -m papermill -k johnvenv "$@" >> Results/run_cpu.out 2>&1; }
OUT=~/John/Dataset/tentativi/20_FromScratch
rm -f Results/data_ready Results/cpu_done
for nb in 01_scraping 02_cleaning 03_anonymization 04_split; do
    note "CPU: $nb"; pm $nb.ipynb $nb.ipynb || { note "ERROR in $nb: queue stopped"; exit 1; }
done
touch Results/data_ready
for nb in 06_classification 07_regression; do
    for i in 1 2; do note "CPU: $nb, TF-IDF (attempt $i)"; pm -p MODELS "TF-IDF" $nb.ipynb Results/${nb}_tfidf.ipynb && break; done
done
until [ -f $OUT/embeddings/article_order.csv ]; do sleep 300; done
for nb in 06_classification 07_regression; do
    for i in 1 2; do note "CPU: $nb, Qwen3-8B and Mistral-24B (attempt $i)"; pm -p MODELS "Qwen3-8B,Mistral-24B" $nb.ipynb Results/${nb}_frozen.ipynb && break; done
done
touch Results/cpu_done; note "CPU: done"

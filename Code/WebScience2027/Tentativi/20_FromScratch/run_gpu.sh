#!/bin/bash
# Server, GPU queue: waits for the dataset (CPU queue), then the frozen vectors (step 5), ModernBERT for classification
# and regression, and at the end (when the CPU queue is done) evaluation and results. Every step resumes from the saved
# folds if interrupted.  Start:  screen -dmS t20_gpu ./run_gpu.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8   # shared server
export CUBLAS_WORKSPACE_CONFIG=:4096:8                                                  # deterministic GPU
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results; note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/run.log; }
pm() { python -m papermill -k johnvenv "$@" >> Results/run_gpu.out 2>&1; }
until [ -f Results/data_ready ]; do sleep 60; done
for i in 1 2 3; do note "GPU: 05_embeddings (attempt $i)"; pm 05_embeddings.ipynb 05_embeddings.ipynb && break; sleep 300; done
for nb in 06_classification 07_regression; do
    for i in 1 2 3; do note "GPU: $nb, ModernBERT (attempt $i)"; pm -p MODELS "ModernBERT" $nb.ipynb Results/${nb}_modernbert.ipynb && break; sleep 300; done
done
until [ -f Results/cpu_done ]; do sleep 60; done
ALL="TF-IDF,Qwen3-8B,Mistral-24B,ModernBERT"
for nb in 06_classification 07_regression; do note "GPU: $nb summary"; pm -p MODELS "$ALL" $nb.ipynb $nb.ipynb; done
note "GPU: 08_evaluation"; pm 08_evaluation.ipynb 08_evaluation.ipynb || note "ERROR in 08_evaluation"
note "GPU: 09_results"; pm 09_results.ipynb 09_results.ipynb || note "ERROR in 09_results"
note "<<< FINISHED"

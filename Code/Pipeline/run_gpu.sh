#!/bin/bash
# Server, GPU queue of the whole pipeline. In order: anonymization part 2 (03b, language model) as soon as part 1 exists;
# frozen vectors (05); ModernBERT regression; when the CPU queue is done, the regression results (07 summary, 08, 09)
# and the effect of the anonymization (12); then the anonymization checks (03c), ModernBERT classification and again
# 08-09; at the end steps 10 and 11 (their own queues). Every step resumes from what is saved.
# Start:  screen -dmS t20_gpu ./run_gpu.sh        Progress from the PC: ./status.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8   # shared server
export CUBLAS_WORKSPACE_CONFIG=:4096:8                                                  # deterministic GPU
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results; note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/run.log; }
pm() { python -m papermill -k johnvenv "$@" >> Results/run_gpu.out 2>&1; }
try() {   # label, papermill arguments: up to 3 tries
    local label=$1; shift
    for i in 1 2 3; do note "GPU: $label (attempt $i)"; pm "$@" && return 0; sleep 300; done
    note "ERROR in $label"; return 1
}
OUT=~/John/Dataset/tentativi/20_FromScratch
rm -f Results/anonymized_ready Results/main_done Results/gpu_done
START=$(date +%s)                     # wait for a data_ready written AFTER this start (not one left by an earlier run)
until [ -f Results/data_ready ] && [ $(stat -c %Y Results/data_ready) -ge $START ]; do sleep 60; done
try "03b_anonymization_llm" 03b_anonymization_llm.ipynb 03b_anonymization_llm.ipynb || { note "no final text: queue stopped"; exit 1; }
[ $OUT/articles_anonymous.csv.gz -nt $OUT/articles_rules.csv.gz ] || { note "final text older than part 1: queue stopped"; exit 1; }
touch Results/anonymized_ready
try "05_embeddings" 05_embeddings.ipynb 05_embeddings.ipynb
try "07_regression, ModernBERT" -p MODELS "ModernBERT" 07_regression.ipynb Results/07_regression_modernbert.ipynb
until [ -f Results/cpu_done ]; do sleep 60; done
ALL="TF-IDF,Qwen3-8B,Mistral-24B,ModernBERT"
note "GPU: 07_regression summary"; pm -p MODELS "$ALL" 07_regression.ipynb 07_regression.ipynb
note "GPU: 08_evaluation"; pm 08_evaluation.ipynb 08_evaluation.ipynb || note "ERROR in 08_evaluation"
note "GPU: 09_results"; pm 09_results.ipynb 09_results.ipynb || note "ERROR in 09_results"
note "GPU: 12_anonymization_effect"; pm 12_anonymization_effect.ipynb 12_anonymization_effect.ipynb || note "ERROR in 12_anonymization_effect"
touch Results/main_done; note "GPU: main results ready (regression)"
try "03c_anonymization_checks" 03c_anonymization_checks.ipynb 03c_anonymization_checks.ipynb
try "06_classification, ModernBERT" -p MODELS "ModernBERT" 06_classification.ipynb Results/06_classification_modernbert.ipynb
note "GPU: 06_classification summary"; pm -p MODELS "$ALL" 06_classification.ipynb 06_classification.ipynb
note "GPU: 08_evaluation (with every classifier)"; pm 08_evaluation.ipynb 08_evaluation.ipynb || note "ERROR in 08_evaluation"
note "GPU: 09_results"; pm 09_results.ipynb 09_results.ipynb || note "ERROR in 09_results"
note "GPU: step 10 (its own queue)"; ./run_boost.sh > /dev/null 2>&1
note "GPU: step 11 (its own queue)"; ./run_meta.sh > /dev/null 2>&1
touch Results/gpu_done; note "<<< FINISHED"

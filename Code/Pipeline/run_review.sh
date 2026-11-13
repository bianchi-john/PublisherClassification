#!/bin/bash
# Server: the additions of November 2026 (steps 5, 7, 13, 14, 15), one after the other, on the shared GPU.
#   5   frozen vectors of Qwen3-Embedding-0.6B and of ModernBERT-base (the others are already saved)
#   7   ridge on the new vectors; Mistral-24B again with the layer (27 or 40) chosen on the inner folds
#   13  zero-shot Mistral-Small-24B-Instruct
#   14  the leakage ladder (stages, TF-IDF, then new vectors and the frozen models)
#   8, 9, 15  evaluation, results and every number of the paper
# Every step resumes from what is saved. Before step 7, the Mistral-24B regression of the previous run (layer 27 fixed)
# is moved to _da_cancellare/ (not deleted). Start:  screen -dmS t20_review ./run_review.sh   Progress: ./status_review.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results; note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/review.log; }
pm() { python -m papermill -k johnvenv "$@" >> Results/run_review.out 2>&1; }
try() {   # label, papermill arguments: up to 3 tries
    local label=$1; shift
    for i in 1 2 3; do note "$label (attempt $i)"; pm "$@" && return 0; sleep 300; done
    note "ERROR in $label"; return 1
}
OUT=~/John/Dataset/tentativi/20_FromScratch
OLD=$OUT/_da_cancellare/mistral_layer27_fixed
rm -f Results/review_done
note ">>> review queue started"
try "05_embeddings (Qwen3-0.6B, ModernBERT frozen)" -p GPU_GB 19 05_embeddings.ipynb 05_embeddings.ipynb
if [ ! -f $OUT/predictions/regression/Mistral-24B/penalty_fold1.csv ] || ! grep -q layer $OUT/predictions/regression/Mistral-24B/penalty_fold1.csv; then
    mkdir -p $OLD
    [ -d $OUT/predictions/regression/Mistral-24B ] && mv $OUT/predictions/regression/Mistral-24B $OLD/predictions
    [ -d ~/John/Modelli/20_FromScratch/regression/Mistral-24B ] && mv ~/John/Modelli/20_FromScratch/regression/Mistral-24B $OLD/models
    note "previous Mistral-24B regression moved to $OLD"
fi
try "07_regression (new frozen models, Mistral layer inside)" -p MODELS "TF-IDF,Qwen3-0.6B,Qwen3-8B,Mistral-24B,ModernBERT-frozen" -p N_JOBS 8 07_regression.ipynb 07_regression.ipynb
try "13_zero_shot" 13_zero_shot.ipynb 13_zero_shot.ipynb
try "14_leakage_ladder" -p N_JOBS 8 14_leakage_ladder.ipynb 14_leakage_ladder.ipynb
note "08_evaluation"; pm 08_evaluation.ipynb 08_evaluation.ipynb || note "ERROR in 08_evaluation"
note "09_results"; pm 09_results.ipynb 09_results.ipynb || note "ERROR in 09_results"
note "15_paper_results"; pm 15_paper_results.ipynb 15_paper_results.ipynb || note "ERROR in 15_paper_results"
touch Results/review_done; note "<<< review queue finished"

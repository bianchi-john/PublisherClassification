#!/bin/bash
# Server, step 11 (meta-articles): build them, frozen vectors (Qwen, Mistral), TF-IDF / Qwen / Mistral regressors,
# ModernBERT fine-tuned, then 11_meta.ipynb. Every piece is a separate process with a time limit: an error or a hang
# moves on, and everything is tried a second time at the end (saved pieces are skipped).
# Start:  screen -dmS t20_meta ./run_meta.sh        From the PC: ./status_meta.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false   # shared server
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results ~/John/Dataset/tentativi/20_FromScratch/meta
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/meta.log; }
step() {   # time limit, label, arguments
    local limit=$1; shift; note "round $round: $*"
    timeout "$limit" python meta_run.py "$@" >> Results/meta_run.out 2>&1; local code=$?
    [ $code -eq 124 ] && note "TIME LIMIT ($limit): $*, next one"
    [ $code -ne 0 ] && [ $code -ne 124 ] && note "FAILED ($code): $*, next one (see errors.txt)"
}
rm -f Results/meta_done
for round in 1 2; do
    step 1h build
    step 2h embed Qwen3-8B
    step 5h embed Mistral-24B
    step 2h fit TF-IDF
    for m in "Qwen3-8B · ridge" "Qwen3-8B · mlp" "Mistral-24B · ridge" "Mistral-24B · mlp"; do step 2h fit "$m"; done
    step 8h bert
done
note "results notebook"
python -m papermill -k johnvenv 11_meta.ipynb 11_meta.ipynb >> Results/meta_run.out 2>&1 || note "ERROR in 11_meta"
touch Results/meta_done; note "<<< FINISHED"

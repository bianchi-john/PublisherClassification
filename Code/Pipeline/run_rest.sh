#!/bin/bash
# Server: the rest of the November 2026 queue (run_review.sh stopped in step 14, waiting for the shared GPU).
#   13  zero-shot predictions again from the saved answers (no GPU: the model is not asked again)
#   14  the leakage ladder: frozen vectors of the original text (GPU), then the ridge models of L0-L2
#   8, 9, 15  evaluation, results and every number of the paper
# Every step resumes from what is saved (the frozen vectors also from a partial file saved along the way). A step that
# stops because the GPU memory ran out (taken by other users) starts again after 10 minutes, up to 100 times; any other
# error: 3 tries, then the queue stops and says why in Results/review.log.
# Start:  screen -dmS t20_rest ./run_rest.sh      Progress from the PC: ./status_review.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results; note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/review.log; }
MEMORY="OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|DeadKernelError"

step() {   # label, papermill arguments
    local label=$1 memory=0 errors=0; shift
    while true; do
        note "$label"
        if python -m papermill -k johnvenv "$@" > Results/last_step.out 2>&1; then
            cat Results/last_step.out >> Results/run_rest.out; note "$label: done"; return 0
        fi
        cat Results/last_step.out >> Results/run_rest.out
        if grep -qE "$MEMORY" Results/last_step.out && [ $memory -lt 100 ]; then
            memory=$((memory + 1)); note "$label: GPU memory ran out ($memory), again in 10 minutes"; sleep 600
        else
            errors=$((errors + 1))
            note "$label: ERROR $errors of 3: $(grep -E 'Error|Exception' Results/last_step.out | tail -n 1 | cut -c1-150)"
            [ $errors -ge 3 ] && { note "STOPPED at $label (details in Results/run_rest.out)"; exit 1; }
            sleep 300
        fi
    done
}

rm -f Results/review_done
note ">>> rest of the queue (run_rest.sh)"
step "13_zero_shot (predictions from the saved answers)" 13_zero_shot.ipynb 13_zero_shot.ipynb
step "14_leakage_ladder" -p N_JOBS 8 14_leakage_ladder.ipynb 14_leakage_ladder.ipynb
step "08_evaluation" 08_evaluation.ipynb 08_evaluation.ipynb
step "09_results" 09_results.ipynb 09_results.ipynb
step "15_paper_results" 15_paper_results.ipynb 15_paper_results.ipynb
touch Results/review_done; note "<<< queue finished"

#!/bin/bash
# Server: run some notebooks in order, CPU only (no wait for the GPU), stop at the first error.
# Usage:  screen -dmS t20_<name> ./run_notebooks.sh 16_other_splits 15_paper_results
# Progress from the PC: ./status_review.sh (the steps are logged in Results/review.log)
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/review.log; }
rm -f Results/review_done
note ">>> run_notebooks.sh $*"
for nb in "$@"; do
    note "$nb"
    if python -m papermill -k johnvenv -p N_JOBS 8 $nb.ipynb $nb.ipynb > Results/last_step.out 2>&1; then
        note "$nb: done"
    else
        note "STOPPED at $nb: $(grep -E 'Error|Exception' Results/last_step.out | tail -n 1 | cut -c1-150)"; exit 1
    fi
done
touch Results/review_done; note "<<< done: $*"

#!/bin/bash
# Server, 17/11/2026: step 14 again with the sisters rung (CPU only: the frozen vectors are saved), then step 15.
# Start: screen -dmS t20_sisters ./run_sisters.sh      Progress from the PC: ./status_review.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/review.log; }
rm -f Results/review_done
note ">>> sisters rung (run_sisters.sh)"
for nb in 14_leakage_ladder 15_paper_results; do
    note "$nb"
    extra=""; [ $nb = 14_leakage_ladder ] && extra="-p N_JOBS 8"
    if python -m papermill -k johnvenv $extra $nb.ipynb $nb.ipynb > Results/last_step.out 2>&1; then
        note "$nb: done"
    else
        note "STOPPED at $nb: $(grep -E 'Error|Exception' Results/last_step.out | tail -n 1 | cut -c1-150)"; exit 1
    fi
done
touch Results/review_done; note "<<< sisters done"

#!/bin/bash
# Server: the name-only probe of step 17 (GPU), started as soon as the shared GPU has room and tried again until it works.
# Every minute: if the GPU has GPU_GB free, run notebook 17 (the CPU checks are cached, so only the probe runs); if it
# fails (memory taken by other users, the driver not answering), try again later. Stops when paper_robustness.csv has
# the name-only rows, or after 30 failed runs. The vectors already computed are cached, so a new run goes on from there.
# Start: screen -dmS t20_probe ./run_probe.sh      Progress from the PC: ./status_review.sh
export TZ=Europe/Rome OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
GPU_GB=19
RESULT=~/John/Dataset/tentativi/20_FromScratch/paper_robustness.csv
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/review.log; }
free_gb() { python -c "import torch; print(int(torch.cuda.mem_get_info()[0] / 2**30))" 2>/dev/null || echo 0; }

rm -f Results/review_done
note ">>> run_probe.sh: name-only probe of step 17, GPU checked every minute"
failed=0
until grep -q "name only" "$RESULT" 2>/dev/null; do
    if [ "$(free_gb)" -ge $GPU_GB ]; then
        note "17_robustness: GPU free, run $((failed + 1))"
        if python -m papermill -k johnvenv 17_robustness.ipynb 17_robustness.ipynb > Results/last_step.out 2>&1; then
            note "17_robustness: done"
        else
            failed=$((failed + 1))
            note "17_robustness: failed ($(grep -E 'Error|Exception' Results/last_step.out | tail -n 1 | cut -c1-120))"
            [ $failed -ge 30 ] && { note "STOPPED: 30 failed runs (details in Results/last_step.out)"; exit 1; }
        fi
    fi
    sleep 60
done
touch Results/review_done
note "<<< done: name-only probe"

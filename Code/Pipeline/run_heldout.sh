#!/bin/bash
# Server: step 19 (held-out outlets, pre-registered in PREREGISTRATION_19.md). Builds the two held-out sets (CPU),
# anonymizes them, then the GPU parts wait for GPU_GB free on the shared GPU. A failed run (memory taken by other users,
# the driver not answering) is tried again a minute later, and every part is cached, so it goes on from there. It stops
# at once if the check on the 226 outlets fails (a bug, not a busy GPU), and after 30 failed runs.
# Start: screen -dmS t20_heldout ./run_heldout.sh      Progress: tail Results/heldout.log
export TZ=Europe/Rome OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/heldout.log; }

rm -f Results/heldout_done
note ">>> run_heldout.sh: step 19, commit $(cat PREREGISTRATION_COMMIT 2>/dev/null || echo unknown)"
failed=0
until python -m papermill -k johnvenv 19_heldout.ipynb 19_heldout.ipynb > Results/last_step.out 2>&1; do
    if grep -q "does not reproduce step 7" Results/last_step.out; then
        note "STOPPED: the check on the 226 outlets failed (details in Results/last_step.out)"; exit 1
    fi
    failed=$((failed + 1))
    note "19_heldout: failed, run $failed ($(grep -E 'Error|Exception' Results/last_step.out | tail -n 1 | cut -c1-120))"
    [ $failed -ge 30 ] && { note "STOPPED: 30 failed runs (details in Results/last_step.out)"; exit 1; }
    sleep 60
done
touch Results/heldout_done
note "<<< done: step 19"

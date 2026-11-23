#!/bin/bash
# Server: step 18 (boilerplate). TF-IDF retrained on the CPU first, then the GPU parts, which wait for GPU_GB free on the
# shared GPU. If a run fails (memory taken by other users, the driver not answering), it is tried again a minute later:
# every part is cached, so a new run goes on from there. Stops after 30 failed runs.
# Start: screen -dmS t20_boilerplate ./run_boilerplate.sh      Progress: tail Results/boilerplate.log
export TZ=Europe/Rome OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/boilerplate.log; }

rm -f Results/boilerplate_done
note ">>> run_boilerplate.sh: step 18"
failed=0
until python -m papermill -k johnvenv 18_boilerplate.ipynb 18_boilerplate.ipynb > Results/last_step.out 2>&1; do
    failed=$((failed + 1))
    note "18_boilerplate: failed, run $failed ($(grep -E 'Error|Exception' Results/last_step.out | tail -n 1 | cut -c1-120))"
    [ $failed -ge 30 ] && { note "STOPPED: 30 failed runs (details in Results/last_step.out)"; exit 1; }
    sleep 60
done
touch Results/boilerplate_done
note "<<< done: step 18"

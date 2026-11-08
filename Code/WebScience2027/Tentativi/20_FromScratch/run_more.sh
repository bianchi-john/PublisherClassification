#!/bin/bash
# Server, step 12 (more outlets in training): the whole step is the notebook 12_more_outlets.ipynb. It is run up to 3
# times: every piece is saved, so a re-run after an error or a crash resumes where it stopped. The GPU part waits for
# the step 11 queue (Results/meta_done).   Start:  screen -dmS t20_more ./run_more.sh    From the PC: ./status_more.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false   # shared server
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results ~/John/Dataset/tentativi/20_FromScratch/more_outlets
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/more.log; }
rm -f Results/more_done
# run 2 retries the pieces that failed in run 1 (logged in errors.txt); run 3 only if a run crashed
for i in 1 2 3; do
    note "12_more_outlets, run $i"
    if timeout 30h python -m papermill -k johnvenv 12_more_outlets.ipynb 12_more_outlets.ipynb >> Results/more_run.out 2>&1; then
        [ $i -ge 2 ] && break
    else
        note "run $i stopped with an error (see Results/more_run.out), trying again in 10 minutes"; sleep 600
    fi
done
touch Results/more_done; note "<<< FINISHED"

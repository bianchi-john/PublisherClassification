#!/bin/bash
# Server, step 10 (frozen embeddings, more strategies): one process per representation and method, with a time limit;
# a crash, an error or a hang (time limit) moves on to the next one, and every run is tried a second time at the end
# of the queue. Folds already saved are skipped, so the queue can simply be started again.
# Then the results notebook 10_boost.ipynb.   Start:  screen -dmS t20_boost ./run_boost.sh    From the PC: ./status_boost.sh
export TZ=Europe/Rome
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false   # shared server
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ~/John/Tentativo20 && source ~/John/johnVenv/bin/activate
mkdir -p Results ~/John/Dataset/tentativi/20_FromScratch/boost
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a Results/boost.log; }
rm -f Results/boost_done
EMBS="Qwen3-8B Mistral-24B Mistral-24B-L40 Qwen+Mistral Mistral-24B-L27+L40"
METHODS="ridge ridge_outlet ridge_lds ridge_median ridge_quantile outlet_mean outlet_bags outlet_mixup outlet_krr outlet_knn krr mlp mil mil_rank"
for round in 1 2; do
    for e in $EMBS; do for m in $METHODS; do
        note "round $round: $e / $m"
        timeout 3h python boost_run.py "$e" "$m" >> Results/boost_run.out 2>&1
        code=$?
        [ $code -eq 124 ] && note "TIME LIMIT: $e / $m (3 h), next one"
        [ $code -ne 0 ] && [ $code -ne 124 ] && note "FAILED ($code): $e / $m, next one (see errors.txt)"
    done; done
done
note "results notebook"
python -m papermill -k johnvenv 10_boost.ipynb 10_boost.ipynb >> Results/boost_run.out 2>&1 || note "ERROR in 10_boost"
touch Results/boost_done; note "<<< FINISHED"

#!/bin/bash
# From the PC: where is step 10 on the server.  Usage: ./status_boost.sh   (./status_boost.sh download: also brings back
# the results notebook, the tables and the figures of step 10, to Dataset/, outside git)
cd "$(dirname "$0")"
R="ssh -o BatchMode=yes -i $HOME/.ssh/id_ed25519_lumina_vm"; S=fabio.pinelli@10.181.100.6
$R $S 'bash -s' <<'REMOTE'
export TZ=Europe/Rome; cd ~/John/Tentativo20 || exit 1
B=~/John/Dataset/tentativi/20_FromScratch/boost
echo "now: $(date '+%d/%m %H:%M')"; screen -ls | grep -o "t20_boost" | sed 's/^/running: /' || echo "queue not running"
[ -f Results/boost_done ] && echo "QUEUE FINISHED"
echo; echo "---- queue"; tail -n 6 Results/boost.log 2>/dev/null
echo; echo "---- folds saved (of 5)"
for e in Qwen3-8B Mistral-24B Mistral-24B-L40 Qwen+Mistral Mistral-24B-L27+L40; do
    line="$e:"; for m in ridge ridge_outlet ridge_lds ridge_median ridge_quantile outlet_mean outlet_bags outlet_mixup outlet_krr outlet_knn krr mlp mil mil_rank; do
        line="$line $m=$(ls $B/predictions/$e/$m/fold*.csv.gz 2>/dev/null | wc -l)"; done; echo "$line"; done
echo; echo "---- last fits"; grep -v "start\|already" $B/log.txt 2>/dev/null | tail -n 4 | cut -c1-220
echo; echo "---- errors"; grep -c "^====" $B/errors.txt 2>/dev/null || echo 0; grep "ERROR\|TIME LIMIT\|FAILED" Results/boost.log 2>/dev/null | tail -n 3
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB used of {t/2**30:.0f}')" 2>/dev/null
REMOTE
if [ "$1" = "download" ]; then
    rsync -av -e "$R" $S:John/Tentativo20/10_boost.ipynb ./ | tail -n 1
    D=../../../../Dataset/tentativi/20_FromScratch/boost; mkdir -p $D
    rsync -av -e "$R" --include "*/" --include "*.csv" --include "*.png" --include "*.txt" --exclude "*" \
          $S:John/Dataset/tentativi/20_FromScratch/boost/ $D/ | tail -n 1
fi

#!/bin/bash
# From the PC: where is step 11 on the server.  Usage: ./status_meta.sh   (./status_meta.sh download: also brings back
# the results notebook and every output of step 11, to Dataset/, outside git)
cd "$(dirname "$0")"
R="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/id_ed25519_lumina_vm"; S=fabio.pinelli@10.181.100.6
$R $S 'bash -s' <<'REMOTE'
export TZ=Europe/Rome; cd ~/John/Tentativo20 || exit 1
B=~/John/Dataset/tentativi/20_FromScratch/meta
echo "now: $(date '+%d/%m %H:%M')"; screen -ls | grep -o "t20_meta" | sed 's/^/running: /' || echo "queue not running"
[ -f Results/meta_done ] && echo "QUEUE FINISHED"
echo; echo "---- queue"; tail -n 5 Results/meta.log 2>/dev/null
echo; echo "---- folds saved (of 5)"
for m in "TF-IDF" "Qwen3-8B · ridge" "Qwen3-8B · mlp" "Mistral-24B · ridge" "Mistral-24B · mlp" "ModernBERT"; do
    echo "$m: $(ls "$B/predictions/$m"/fold*.csv.gz 2>/dev/null | wc -l)"; done
echo; echo "---- last lines"; tail -n 5 $B/log.txt 2>/dev/null | cut -c1-200
echo; echo "---- errors"; grep -c "^====" $B/errors.txt 2>/dev/null || echo 0
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB used of {t/2**30:.0f}')" 2>/dev/null
REMOTE
if [ "$1" = "download" ]; then
    rsync -av -e "$R" $S:John/Tentativo20/11_meta.ipynb ./ | tail -n 1; rsync -a -e "$R" $S:John/Tentativo20/Results/meta.log Results/
    D=../../../../Dataset/tentativi/20_FromScratch/meta; mkdir -p $D
    rsync -av -e "$R" --exclude "embeddings/" $S:John/Dataset/tentativi/20_FromScratch/meta/ $D/ | tail -n 1
fi
